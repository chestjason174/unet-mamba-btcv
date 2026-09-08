from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from models.unet3d import NUM_CLASSES
from .preprocessing import (
    BTCVPreprocessingCache,
    FixedPatchSpec,
    PreprocessedBTCVCase,
    TARGET_SPACING,
    HU_WINDOW,
    build_fixed_patch_specs,
    choose_deterministic_multiorgan_start,
)


@dataclass(frozen=True)
class BTCVCase:
    image_path: Path
    label_path: Optional[Path]
    image: np.ndarray
    label: Optional[np.ndarray]
    affine: np.ndarray
    header: nib.Nifti1Header


def pair_btcv_files(root: Path):
    root = Path(root)
    img_dir = root / "RawData" / "Training" / "img"
    label_dir = root / "RawData" / "Training" / "label"

    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    pairs = []
    for img_path in sorted(img_dir.glob("img*.nii.gz")):
        label_path = label_dir / img_path.name.replace("img", "label", 1)
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label for {img_path.name}: {label_path}")
        pairs.append((img_path, label_path))

    if not pairs:
        raise RuntimeError(f"No BTCV image files found in {img_dir}")
    return pairs


def strip_nii_gz_suffix(path: Path) -> str:
    name = Path(path).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return Path(path).stem


def split_btcv_pairs(pairs, seed: int = 42, val_fraction: float = 0.2):
    pairs = list(pairs)
    if len(pairs) < 2 or val_fraction <= 0:
        return pairs, []

    indices = list(range(len(pairs)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_count = max(1, int(round(len(pairs) * val_fraction)))
    val_count = min(val_count, len(pairs) - 1)
    val_indices = set(indices[:val_count])

    train_pairs = [pair for idx, pair in enumerate(pairs) if idx not in val_indices]
    val_pairs = [pair for idx, pair in enumerate(pairs) if idx in val_indices]
    return train_pairs, val_pairs


def save_split_json(path: Path, train_pairs, val_pairs, seed: int, val_fraction: float, root: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "root": str(Path(root)),
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "train": [str(Path(p[0])) for p in train_pairs],
        "val": [str(Path(p[0])) for p in val_pairs],
        "train_with_labels": [[str(Path(p[0])), str(Path(p[1]))] for p in train_pairs],
        "val_with_labels": [[str(Path(p[0])), str(Path(p[1]))] for p in val_pairs],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_split_json(path: Path):
    path = Path(path)
    payload = json.loads(path.read_text())
    train_pairs = [(Path(img), Path(lbl)) for img, lbl in payload.get("train_with_labels", [])]
    val_pairs = [(Path(img), Path(lbl)) for img, lbl in payload.get("val_with_labels", [])]
    return payload, train_pairs, val_pairs


def zscore_normalize(image):
    image = image.astype(np.float32)
    return (image - image.mean()) / (image.std() + 1e-5)


def get_random_start(max_start):
    return np.random.randint(0, max_start + 1)


def get_foreground_start(label, patch_size):
    pd, ph, pw = patch_size
    d, h, w = label.shape
    max_start_d = d - pd
    max_start_h = h - ph
    max_start_w = w - pw

    foreground_positions = np.argwhere(label > 0)
    if len(foreground_positions) == 0:
        return (
            get_random_start(max_start_d),
            get_random_start(max_start_h),
            get_random_start(max_start_w),
        )

    center_d, center_h, center_w = foreground_positions[np.random.randint(len(foreground_positions))]
    start_d = np.clip(center_d - np.random.randint(0, pd), 0, max_start_d)
    start_h = np.clip(center_h - np.random.randint(0, ph), 0, max_start_h)
    start_w = np.clip(center_w - np.random.randint(0, pw), 0, max_start_w)
    return int(start_d), int(start_h), int(start_w)


def get_balanced_foreground_start(label, patch_size):
    foreground_classes = np.unique(label[label > 0])
    if len(foreground_classes) == 0:
        return get_foreground_start(label, patch_size), None

    target_class = int(foreground_classes[np.random.randint(len(foreground_classes))])
    class_positions = np.argwhere(label == target_class)
    if len(class_positions) == 0:
        return get_foreground_start(label, patch_size), target_class

    pd, ph, pw = patch_size
    d, h, w = label.shape
    max_start_d = d - pd
    max_start_h = h - ph
    max_start_w = w - pw

    center_d, center_h, center_w = class_positions[np.random.randint(len(class_positions))]
    start_d = np.clip(center_d - np.random.randint(0, pd), 0, max_start_d)
    start_h = np.clip(center_h - np.random.randint(0, ph), 0, max_start_h)
    start_w = np.clip(center_w - np.random.randint(0, pw), 0, max_start_w)
    return (int(start_d), int(start_h), int(start_w)), target_class


def crop_or_pad_3d(
    image,
    label,
    patch_size,
    random_crop,
    foreground_crop_prob,
    background_crop_prob=None,
):
    _, d, h, w = image.shape
    pd, ph, pw = patch_size

    pad_d = max(pd - d, 0)
    pad_h = max(ph - h, 0)
    pad_w = max(pw - w, 0)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        image = np.pad(
            image,
            ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=0,
        )
        label = np.pad(
            label,
            ((0, pad_d), (0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=0,
        )
        _, d, h, w = image.shape

    if random_crop:
        if background_crop_prob is None:
            background_crop_prob = max(0.0, 1.0 - float(foreground_crop_prob))

        use_background = np.random.rand() < background_crop_prob
        if use_background:
            start_d = get_random_start(d - pd)
            start_h = get_random_start(h - ph)
            start_w = get_random_start(w - pw)
        else:
            start, _ = get_balanced_foreground_start(label, patch_size)
            start_d, start_h, start_w = start
    else:
        start_d = (d - pd) // 2
        start_h = (h - ph) // 2
        start_w = (w - pw) // 2

    end_d = start_d + pd
    end_h = start_h + ph
    end_w = start_w + pw

    image_patch = image[:, start_d:end_d, start_h:end_h, start_w:end_w]
    label_patch = label[start_d:end_d, start_h:end_h, start_w:end_w]
    return image_patch, label_patch


def load_case_from_pair(image_path: Path, label_path: Optional[Path] = None) -> BTCVCase:
    image_obj = nib.load(str(image_path))
    image = image_obj.get_fdata(dtype=np.float32)

    label = None
    if label_path is not None:
        label_obj = nib.load(str(label_path))
        label = label_obj.get_fdata(dtype=np.float32).astype(np.int64)
        if label.shape != image.shape:
            raise ValueError(
                f"Shape mismatch: image {image.shape} vs label {label.shape} for {image_path.name}"
            )

    return BTCVCase(
        image_path=Path(image_path),
        label_path=Path(label_path) if label_path is not None else None,
        image=image,
        label=label,
        affine=image_obj.affine,
        header=image_obj.header,
    )


def load_case_by_index(root: Path, index: int) -> BTCVCase:
    pairs = pair_btcv_files(root)
    if not 0 <= index < len(pairs):
        raise IndexError(f"index must be between 0 and {len(pairs) - 1}")
    image_path, label_path = pairs[index]
    return load_case_from_pair(image_path, label_path)


class BTCVDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.pairs = pair_btcv_files(self.root)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, label_path = self.pairs[idx]
        case = load_case_from_pair(image_path, label_path)
        image = zscore_normalize(case.image)[None, ...]
        label = case.label.astype(np.int64)
        image = torch.from_numpy(np.ascontiguousarray(image)).float()
        label = torch.from_numpy(np.ascontiguousarray(label)).long()
        return image, label


class BTCVPatchDataset(Dataset):
    def __init__(
        self,
        root,
        patch_size=(64, 64, 64),
        random_crop=True,
        foreground_crop_prob=0.8,
        background_crop_prob=None,
        patches_per_case=1,
        overfit_one_sample=False,
        pairs=None,
        sanity_print=True,
    ):
        self.root = Path(root)
        self.patch_size = tuple(patch_size)
        self.random_crop = random_crop
        self.foreground_crop_prob = foreground_crop_prob
        self.background_crop_prob = background_crop_prob
        self.patches_per_case = int(patches_per_case)
        self.pairs = list(pairs) if pairs is not None else pair_btcv_files(self.root)

        if overfit_one_sample:
            self.pairs = self.pairs[:1]

        if sanity_print:
            self.print_sanity_check()

    def print_sanity_check(self):
        img_path, label_path = self.pairs[0]
        img_obj = nib.load(str(img_path))
        label_obj = nib.load(str(label_path))
        label_data = label_obj.get_fdata(dtype=np.float32)
        print("[sanity] pairs:", len(self.pairs))
        print("[sanity] first image:", img_path.name, "shape:", img_obj.shape)
        print("[sanity] first label:", label_path.name, "shape:", label_obj.shape)
        print("[sanity] label min/max:", int(label_data.min()), int(label_data.max()))

    def __len__(self):
        return len(self.pairs) * self.patches_per_case

    def __getitem__(self, idx):
        case_idx = idx % len(self.pairs)
        img_path, label_path = self.pairs[case_idx]
        image = nib.load(str(img_path)).get_fdata(dtype=np.float32)
        label = nib.load(str(label_path)).get_fdata(dtype=np.float32).astype(np.int64)

        if image.shape != label.shape:
            raise ValueError(f"Shape mismatch: {img_path.name} {image.shape}, {label_path.name} {label.shape}")
        if label.min() < 0 or label.max() >= NUM_CLASSES:
            raise ValueError(f"Label values must be in [0, {NUM_CLASSES - 1}], got {label.min()}..{label.max()}")

        image = zscore_normalize(image)
        image = image[None, ...]
        image, label = crop_or_pad_3d(
            image,
            label,
            self.patch_size,
            self.random_crop,
            self.foreground_crop_prob,
            self.background_crop_prob,
        )

        image = torch.from_numpy(np.ascontiguousarray(image)).float()
        label = torch.from_numpy(np.ascontiguousarray(label)).long()
        return image, label


def load_preprocessed_case_from_pair(
    image_path: Path,
    label_path: Optional[Path] = None,
    cache: Optional[BTCVPreprocessingCache] = None,
) -> PreprocessedBTCVCase:
    if cache is None:
        cache = BTCVPreprocessingCache(Path(image_path).parent.parent.parent.parent / ".cache" / "preprocessed_btcv")
    return cache.load_from_paths(image_path, label_path)


def load_preprocessed_case_by_index(
    root: Path,
    index: int,
    cache: Optional[BTCVPreprocessingCache] = None,
) -> PreprocessedBTCVCase:
    pairs = pair_btcv_files(root)
    if not 0 <= index < len(pairs):
        raise IndexError(f"index must be between 0 and {len(pairs) - 1}")
    image_path, label_path = pairs[index]
    return load_preprocessed_case_from_pair(image_path, label_path, cache=cache)


class PreprocessedBTCVPatchDataset(Dataset):
    def __init__(
        self,
        root,
        patch_size=(64, 64, 64),
        random_crop=True,
        foreground_crop_prob=0.8,
        background_crop_prob=None,
        patches_per_case=1,
        overfit_one_sample=False,
        pairs=None,
        cache_dir=None,
        target_spacing=TARGET_SPACING,
        hu_window=HU_WINDOW,
        sanity_print=True,
    ):
        self.root = Path(root)
        self.patch_size = tuple(patch_size)
        self.random_crop = random_crop
        self.foreground_crop_prob = foreground_crop_prob
        self.background_crop_prob = background_crop_prob
        self.patches_per_case = int(patches_per_case)
        self.pairs = list(pairs) if pairs is not None else pair_btcv_files(self.root)
        self.cache = BTCVPreprocessingCache(
            cache_dir=Path(cache_dir) if cache_dir is not None else self.root / ".cache" / "preprocessed_btcv",
            target_spacing=target_spacing,
            hu_window=hu_window,
        )

        if overfit_one_sample:
            self.pairs = self.pairs[:1]

        if sanity_print:
            self.print_sanity_check()

    def print_sanity_check(self):
        case = self.cache.load_from_paths(*self.pairs[0])
        print("[sanity] pairs:", len(self.pairs))
        print("[sanity] first image:", case.image_path.name, "orig_shape:", case.original_shape, "orig_spacing:", tuple(round(v, 6) for v in case.original_spacing))
        print("[sanity] resampled shape:", case.resampled_shape, "target_spacing:", case.target_spacing, "effective_spacing:", tuple(round(v, 6) for v in case.effective_spacing))
        print("[sanity] preprocessed image min/max:", float(case.image.min()), float(case.image.max()))
        print("[sanity] label unique values:", sorted(np.unique(case.label).astype(int).tolist()) if case.label is not None else [])

    def __len__(self):
        return len(self.pairs) * self.patches_per_case

    def __getitem__(self, idx):
        case_idx = idx % len(self.pairs)
        img_path, label_path = self.pairs[case_idx]
        case = self.cache.load_from_paths(img_path, label_path)
        if case.label is None:
            raise ValueError("PreprocessedBTCVPatchDataset requires labels.")

        image = case.image
        label = case.label

        if image.shape != label.shape:
            raise ValueError(f"Shape mismatch: {img_path.name} {image.shape}, {label_path.name} {label.shape}")
        if label.min() < 0 or label.max() >= NUM_CLASSES:
            raise ValueError(f"Label values must be in [0, {NUM_CLASSES - 1}], got {label.min()}..{label.max()}")

        image = image[None, ...]
        image, label = crop_or_pad_3d(
            image,
            label,
            self.patch_size,
            self.random_crop,
            self.foreground_crop_prob,
            self.background_crop_prob,
        )

        image = torch.from_numpy(np.ascontiguousarray(image)).float()
        label = torch.from_numpy(np.ascontiguousarray(label)).long()
        return image, label


class FixedPatchDataset(Dataset):
    def __init__(
        self,
        root,
        patch_specs,
        cache_dir=None,
        target_spacing=TARGET_SPACING,
        hu_window=HU_WINDOW,
        sanity_print=True,
    ):
        self.root = Path(root)
        self.patch_specs = list(patch_specs)
        self.cache = BTCVPreprocessingCache(
            cache_dir=Path(cache_dir) if cache_dir is not None else self.root / ".cache" / "preprocessed_btcv",
            target_spacing=target_spacing,
            hu_window=hu_window,
        )
        if sanity_print and self.patch_specs:
            spec = self.patch_specs[0]
            case = self.cache.load_from_paths(spec.image_path, spec.label_path)
            patch = case.label[
                spec.start[0] : spec.start[0] + spec.patch_size[0],
                spec.start[1] : spec.start[1] + spec.patch_size[1],
                spec.start[2] : spec.start[2] + spec.patch_size[2],
            ]
            print("[fixed patch sanity] first case:", spec.image_path.name)
            print("[fixed patch sanity] start:", spec.start, "patch_size:", spec.patch_size)
            print("[fixed patch sanity] foreground classes:", spec.num_foreground_classes, "foreground voxels:", spec.foreground_voxels)
            print("[fixed patch sanity] actual patch classes:", sorted(np.unique(patch).astype(int).tolist()))

    def __len__(self):
        return len(self.patch_specs)

    def __getitem__(self, idx):
        spec = self.patch_specs[idx]
        case = self.cache.load_from_paths(spec.image_path, spec.label_path)
        if case.label is None:
            raise ValueError("FixedPatchDataset requires labels.")

        image = case.image[None, ...]
        label = case.label
        start_d, start_h, start_w = spec.start
        pd, ph, pw = spec.patch_size
        image_patch = image[:, start_d : start_d + pd, start_h : start_h + ph, start_w : start_w + pw]
        label_patch = label[start_d : start_d + pd, start_h : start_h + ph, start_w : start_w + pw]

        image_patch = torch.from_numpy(np.ascontiguousarray(image_patch)).float()
        label_patch = torch.from_numpy(np.ascontiguousarray(label_patch)).long()
        return image_patch, label_patch
