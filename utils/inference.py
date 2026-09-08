from __future__ import annotations

from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import torch

from models import NUM_CLASSES, build_model, infer_model_type


def _pad_to_patch(image: np.ndarray, patch_size):
    _, d, h, w = image.shape
    pd, ph, pw = patch_size
    pad_d = max(pd - d, 0)
    pad_h = max(ph - h, 0)
    pad_w = max(pw - w, 0)
    if pad_d == 0 and pad_h == 0 and pad_w == 0:
        return image, (0, 0, 0)

    padded = np.pad(
        image,
        ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
        mode="constant",
        constant_values=0,
    )
    return padded, (pad_d, pad_h, pad_w)


def _start_positions(length: int, patch: int, overlap: float) -> list[int]:
    if length <= patch:
        return [0]
    stride = max(1, int(round(patch * (1.0 - overlap))))
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


@torch.no_grad()
def sliding_window_predict_logits(
    model,
    image: np.ndarray,
    patch_size=(64, 64, 64),
    overlap: float = 0.5,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
):
    if device is None:
        device = next(model.parameters()).device

    if image.ndim == 3:
        image = image[None, ...]
    if image.ndim != 4:
        raise ValueError(f"Expected image shape (C, D, H, W) or (D, H, W), got {image.shape}")
    if image.shape[0] != 1:
        raise ValueError(f"Expected a single-channel CT volume, got {image.shape[0]} channels")

    image = np.ascontiguousarray(image.astype(np.float32))
    padded_image, pad = _pad_to_patch(image, patch_size)
    _, depth, height, width = padded_image.shape
    pd, ph, pw = patch_size

    z_starts = _start_positions(depth, pd, overlap)
    y_starts = _start_positions(height, ph, overlap)
    x_starts = _start_positions(width, pw, overlap)

    logits_sum = torch.zeros((NUM_CLASSES, depth, height, width), dtype=torch.float32)
    counts = torch.zeros((1, depth, height, width), dtype=torch.float32)

    pending_patches = []
    pending_coords = []

    def flush_pending():
        if not pending_patches:
            return
        batch = torch.from_numpy(np.stack(pending_patches, axis=0)).float().to(device)
        outputs = model(batch).detach().cpu()
        for coord, out in zip(pending_coords, outputs):
            z, y, x = coord
            logits_sum[:, z : z + pd, y : y + ph, x : x + pw] += out
            counts[:, z : z + pd, y : y + ph, x : x + pw] += 1.0
        pending_patches.clear()
        pending_coords.clear()

    for z in z_starts:
        for y in y_starts:
            for x in x_starts:
                patch = padded_image[:, z : z + pd, y : y + ph, x : x + pw]
                pending_patches.append(patch)
                pending_coords.append((z, y, x))
                if len(pending_patches) >= batch_size:
                    flush_pending()

    flush_pending()

    counts = counts.clamp_min(1.0)
    averaged_logits = logits_sum / counts

    pad_d, pad_h, pad_w = pad
    if pad_d > 0:
        averaged_logits = averaged_logits[:, : averaged_logits.shape[1] - pad_d]
    if pad_h > 0:
        averaged_logits = averaged_logits[:, :, : averaged_logits.shape[2] - pad_h, :]
    if pad_w > 0:
        averaged_logits = averaged_logits[:, :, :, : averaged_logits.shape[3] - pad_w]

    return averaged_logits


@torch.no_grad()
def sliding_window_predict(
    model,
    image: np.ndarray,
    patch_size=(64, 64, 64),
    overlap: float = 0.5,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
):
    logits = sliding_window_predict_logits(
        model,
        image,
        patch_size=patch_size,
        overlap=overlap,
        batch_size=batch_size,
        device=device,
    )
    return logits.argmax(dim=0).cpu().numpy().astype(np.uint8)


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: Optional[str] = None,
    base_channels: Optional[int] = None,
):
    checkpoint_path = Path(checkpoint_path)
    if device is None:
        device_obj = torch.device("cpu")
    else:
        device_obj = torch.device(device)

    checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
    if base_channels is None:
        base_channels = int(checkpoint.get("args", {}).get("base_channels", 8))
    model_type = infer_model_type(checkpoint)

    model = build_model(
        model_type,
        in_channels=1,
        num_classes=NUM_CLASSES,
        base_channels=base_channels,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device_obj)
    model.eval()
    checkpoint["model_type"] = model_type
    return model, checkpoint, device_obj, base_channels


def save_prediction_nifti(prediction: np.ndarray, affine: np.ndarray, output_path: Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(prediction.astype(np.uint8), affine=affine)
    nib.save(image, str(output_path))
