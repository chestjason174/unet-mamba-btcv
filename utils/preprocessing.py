from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


TARGET_SPACING = (1.5, 1.5, 2.0)
HU_WINDOW = (-175.0, 250.0)


@dataclass(frozen=True)
class PreprocessedBTCVCase:
    image_path: Path
    label_path: Optional[Path]
    image: np.ndarray
    label: Optional[np.ndarray]
    affine: np.ndarray
    header: nib.Nifti1Header
    original_shape: tuple[int, int, int]
    original_spacing: tuple[float, float, float]
    resampled_shape: tuple[int, int, int]
    target_spacing: tuple[float, float, float]
    effective_spacing: tuple[float, float, float]


@dataclass(frozen=True)
class FixedPatchSpec:
    image_path: Path
    label_path: Optional[Path]
    start: tuple[int, int, int]
    patch_size: tuple[int, int, int]
    num_foreground_classes: int
    foreground_voxels: int


def _strip_nii_gz_suffix(path: Path) -> str:
    name = Path(path).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return Path(path).stem


def _spacing_from_affine(affine: np.ndarray) -> tuple[float, float, float]:
    spacing = []
    for axis in range(3):
        spacing.append(float(np.linalg.norm(affine[:3, axis])))
    return tuple(spacing)


def _resampled_shape(
    original_shape: tuple[int, int, int],
    original_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
) -> tuple[int, int, int]:
    shape = []
    for size, source_spacing, desired_spacing in zip(original_shape, original_spacing, target_spacing):
        physical = float(size) * float(source_spacing)
        shape.append(max(1, int(round(physical / float(desired_spacing)))))
    return tuple(shape)


def _resampled_affine(affine: np.ndarray, target_spacing: tuple[float, float, float]) -> np.ndarray:
    new_affine = np.array(affine, dtype=np.float32, copy=True)
    for axis, desired_spacing in enumerate(target_spacing):
        direction = new_affine[:3, axis]
        norm = float(np.linalg.norm(direction))
        if norm > 0.0:
            new_affine[:3, axis] = direction / norm * float(desired_spacing)
    return new_affine


def _resample_volume(volume: np.ndarray, target_shape: tuple[int, int, int], mode: str) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape}")

    tensor = torch.from_numpy(np.ascontiguousarray(volume.astype(np.float32)))[None, None]
    kwargs = {"mode": mode, "size": tuple(int(v) for v in target_shape)}
    if mode == "trilinear":
        kwargs["align_corners"] = False
    out = F.interpolate(tensor, **kwargs)
    return out[0, 0].cpu().numpy()


def resample_image_and_label(
    image: np.ndarray,
    label: Optional[np.ndarray],
    original_affine: np.ndarray,
    target_spacing: tuple[float, float, float] = TARGET_SPACING,
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray, tuple[int, int, int]]:
    original_shape = tuple(int(v) for v in image.shape)
    original_spacing = _spacing_from_affine(original_affine)
    target_shape = _resampled_shape(original_shape, original_spacing, target_spacing)

    resampled_image = _resample_volume(image, target_shape, mode="trilinear")
    resampled_label = None
    if label is not None:
        resampled_label = _resample_volume(label.astype(np.float32), target_shape, mode="nearest").astype(np.int64)

    resampled_affine = _resampled_affine(original_affine, target_spacing)
    return resampled_image, resampled_label, resampled_affine, target_shape


def clip_and_scale_hu(image: np.ndarray, hu_window: tuple[float, float] = HU_WINDOW) -> np.ndarray:
    lo, hi = hu_window
    image = np.clip(image.astype(np.float32), lo, hi)
    return (image - lo) / (hi - lo)


def preprocess_btcv_case(
    image: np.ndarray,
    label: Optional[np.ndarray],
    original_affine: np.ndarray,
    target_spacing: tuple[float, float, float] = TARGET_SPACING,
    hu_window: tuple[float, float] = HU_WINDOW,
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray, tuple[int, int, int], tuple[float, float, float], tuple[float, float, float]]:
    original_shape = tuple(int(v) for v in image.shape)
    original_spacing = _spacing_from_affine(original_affine)
    resampled_image, resampled_label, resampled_affine, target_shape = resample_image_and_label(
        image,
        label,
        original_affine,
        target_spacing=target_spacing,
    )
    processed_image = clip_and_scale_hu(resampled_image, hu_window=hu_window)
    effective_spacing = tuple(float(o * s / r) for o, s, r in zip(original_shape, original_spacing, target_shape))
    return processed_image, resampled_label, resampled_affine, target_shape, original_spacing, effective_spacing


def _cache_name(image_path: Path, target_spacing: tuple[float, float, float], hu_window: tuple[float, float]) -> str:
    spacing_tag = "x".join(f"{v:g}" for v in target_spacing)
    window_tag = f"{hu_window[0]:g}_{hu_window[1]:g}"
    return f"{_strip_nii_gz_suffix(image_path)}__sp{spacing_tag}__win{window_tag}.npz"


class BTCVPreprocessingCache:
    def __init__(
        self,
        cache_dir: Path,
        target_spacing: tuple[float, float, float] = TARGET_SPACING,
        hu_window: tuple[float, float] = HU_WINDOW,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.target_spacing = tuple(float(v) for v in target_spacing)
        self.hu_window = tuple(float(v) for v in hu_window)
        self._memory: dict[str, PreprocessedBTCVCase] = {}

    def cache_path(self, image_path: Path) -> Path:
        return self.cache_dir / _cache_name(image_path, self.target_spacing, self.hu_window)

    def load_from_paths(self, image_path: Path, label_path: Optional[Path] = None) -> PreprocessedBTCVCase:
        key = str(self.cache_path(image_path))
        if key in self._memory:
            return self._memory[key]

        cache_path = self.cache_path(image_path)
        if cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as payload:
                case = PreprocessedBTCVCase(
                    image_path=Path(payload["image_path"].item()),
                    label_path=Path(payload["label_path"].item()) if payload["label_path"].item() else None,
                    image=payload["image"],
                    label=payload["label"] if payload["label_present"].item() else None,
                    affine=payload["affine"],
                    header=nib.Nifti1Header(),
                    original_shape=tuple(int(v) for v in payload["original_shape"].tolist()),
                    original_spacing=tuple(float(v) for v in payload["original_spacing"].tolist()),
                    resampled_shape=tuple(int(v) for v in payload["resampled_shape"].tolist()),
                    target_spacing=tuple(float(v) for v in payload["target_spacing"].tolist()),
                    effective_spacing=tuple(float(v) for v in payload["effective_spacing"].tolist()),
                )
            self._memory[key] = case
            return case

        image_obj = nib.load(str(image_path))
        image = image_obj.get_fdata(dtype=np.float32)
        label = None
        if label_path is not None:
            label_obj = nib.load(str(label_path))
            label = label_obj.get_fdata(dtype=np.float32).astype(np.int64)
            if label.shape != image.shape:
                raise ValueError(f"Shape mismatch: image {image.shape} vs label {label.shape} for {image_path.name}")

        processed_image, processed_label, resampled_affine, target_shape, original_spacing, effective_spacing = preprocess_btcv_case(
            image,
            label,
            image_obj.affine,
            target_spacing=self.target_spacing,
            hu_window=self.hu_window,
        )

        case = PreprocessedBTCVCase(
            image_path=Path(image_path),
            label_path=Path(label_path) if label_path is not None else None,
            image=np.ascontiguousarray(processed_image.astype(np.float32)),
            label=None if processed_label is None else np.ascontiguousarray(processed_label.astype(np.int64)),
            affine=resampled_affine,
            header=image_obj.header,
            original_shape=tuple(int(v) for v in image.shape),
            original_spacing=original_spacing,
            resampled_shape=tuple(int(v) for v in target_shape),
            target_spacing=self.target_spacing,
            effective_spacing=effective_spacing,
        )

        np.savez_compressed(
            cache_path,
            image_path=str(case.image_path),
            label_path="" if case.label_path is None else str(case.label_path),
            image=case.image,
            label=np.array([]) if case.label is None else case.label,
            label_present=np.array(case.label is not None),
            affine=case.affine.astype(np.float32),
            original_shape=np.asarray(case.original_shape, dtype=np.int64),
            original_spacing=np.asarray(case.original_spacing, dtype=np.float32),
            resampled_shape=np.asarray(case.resampled_shape, dtype=np.int64),
            target_spacing=np.asarray(case.target_spacing, dtype=np.float32),
            effective_spacing=np.asarray(case.effective_spacing, dtype=np.float32),
        )
        self._memory[key] = case
        return case

    def load_by_index(self, pairs: list[tuple[Path, Path]], index: int) -> PreprocessedBTCVCase:
        image_path, label_path = pairs[index]
        return self.load_from_paths(image_path, label_path)


def _foreground_class_count(label_patch: np.ndarray) -> int:
    classes = np.unique(label_patch)
    return int(np.sum(classes > 0))


def _foreground_voxel_count(label_patch: np.ndarray) -> int:
    return int(np.sum(label_patch > 0))


def _clip_start(start: int, size: int, patch: int) -> int:
    return int(np.clip(start, 0, max(0, size - patch)))


def choose_deterministic_multiorgan_start(label: np.ndarray, patch_size: tuple[int, int, int]) -> tuple[int, int, int]:
    if label.ndim != 3:
        raise ValueError(f"Expected 3D label volume, got {label.shape}")

    pd, ph, pw = patch_size
    d, h, w = label.shape
    foreground = np.argwhere(label > 0)
    if len(foreground) == 0:
        return ((d - pd) // 2, (h - ph) // 2, (w - pw) // 2)

    center = np.mean(foreground, axis=0)
    base = np.array([center[0] - pd / 2.0, center[1] - ph / 2.0, center[2] - pw / 2.0], dtype=np.float32)

    offsets = (-pd // 4, 0, pd // 4)
    best_start = None
    best_score = (-1, -1, -1)

    for dz in offsets:
        for dy in offsets:
            for dx in offsets:
                start = (
                    _clip_start(int(round(base[0] + dz)), d, pd),
                    _clip_start(int(round(base[1] + dy)), h, ph),
                    _clip_start(int(round(base[2] + dx)), w, pw),
                )
                patch = label[start[0] : start[0] + pd, start[1] : start[1] + ph, start[2] : start[2] + pw]
                score = (_foreground_class_count(patch), _foreground_voxel_count(patch), -abs(int(round(base[0] + dz)) - start[0]) - abs(int(round(base[1] + dy)) - start[1]) - abs(int(round(base[2] + dx)) - start[2]))
                if score > best_score:
                    best_score = score
                    best_start = start

    assert best_start is not None
    return best_start


def build_fixed_patch_specs(
    pairs: Iterable[tuple[Path, Path]],
    cache: BTCVPreprocessingCache,
    patch_size: tuple[int, int, int] = (64, 64, 64),
    count: int = 8,
) -> list[FixedPatchSpec]:
    specs: list[FixedPatchSpec] = []
    for image_path, label_path in list(pairs)[:count]:
        case = cache.load_from_paths(image_path, label_path)
        if case.label is None:
            continue
        start = choose_deterministic_multiorgan_start(case.label, patch_size)
        patch = case.label[
            start[0] : start[0] + patch_size[0],
            start[1] : start[1] + patch_size[1],
            start[2] : start[2] + patch_size[2],
        ]
        specs.append(
            FixedPatchSpec(
                image_path=case.image_path,
                label_path=case.label_path,
                start=start,
                patch_size=patch_size,
                num_foreground_classes=_foreground_class_count(patch),
                foreground_voxels=_foreground_voxel_count(patch),
            )
        )
    if len(specs) < count:
        raise RuntimeError(f"Expected at least {count} fixed patches, got {len(specs)}")
    return specs
