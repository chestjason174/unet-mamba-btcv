from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import NUM_CLASSES, build_model, count_parameters, infer_model_type
from utils.data import (
    FixedPatchDataset,
    PreprocessedBTCVPatchDataset,
    load_preprocessed_case_from_pair,
    pair_btcv_files,
    save_split_json,
    split_btcv_pairs,
    strip_nii_gz_suffix,
)
from utils.metrics import foreground_dice_loss, hard_foreground_dice, mean_foreground_dice, soft_foreground_dice
from utils.preprocessing import BTCVPreprocessingCache, build_fixed_patch_specs
from utils.visualization import plot_per_class_validation_dice, plot_training_curves


BTCV_FOREGROUND_CLASSES = [
    (1, "spleen"),
    (2, "right_kidney"),
    (3, "left_kidney"),
    (4, "gallbladder"),
    (5, "esophagus"),
    (6, "liver"),
    (7, "stomach"),
    (8, "aorta"),
    (9, "inferior_vena_cava"),
    (10, "portal_vein"),
    (11, "pancreas"),
    (12, "right_adrenal"),
    (13, "left_adrenal"),
]

VAL_DICE_COLUMNS = [
    f"val_dice_c{class_idx:02d}_{class_name}"
    for class_idx, class_name in BTCV_FOREGROUND_CLASSES
]

HISTORY_NUMERIC_COLUMNS = {
    "train_loss",
    "val_loss",
    "mean_val_dice",
    "lr",
    "epoch_train_time_sec",
    "ce_loss",
    "dice_loss",
    "soft_fg_dice",
    "hard_fg_dice",
    *VAL_DICE_COLUMNS,
}


def parse_args():
    parser = argparse.ArgumentParser(description="BTCV 3D U-Net training entry point.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "btcv",
        help="BTCV dataset root containing RawData/Training/img and label.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for checkpoints, logs, and plots. Defaults depend on the selected mode.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-iters", type=int, default=None, help="Stop after this many optimizer steps. 0 disables.")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--loss",
        choices=("ce", "ce_dice"),
        default=None,
        help="Training loss to use for the segmentation objective.",
    )
    parser.add_argument("--patch-size", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--model-type", choices=("unet3d", "unet3d_mamba"), default="unet3d")
    parser.add_argument("--foreground-crop-prob", type=float, default=0.8)
    parser.add_argument("--background-crop-prob", type=float, default=None)
    parser.add_argument("--patches-per-case", type=int, default=None)
    parser.add_argument("--target-spacing", type=float, nargs=3, default=(1.5, 1.5, 2.0))
    parser.add_argument("--hu-window", type=float, nargs=2, default=(-175.0, 250.0))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--val-split", "--val-fraction", dest="val_fraction", type=float, default=0.2)
    parser.add_argument("--val-iters", type=int, default=4)
    parser.add_argument("--val-every", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-path", type=Path, default=None, help="Optional existing split.json to reuse exactly.")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-overlap", type=float, default=0.5)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--quick-val-iters", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--overfit-one-sample", action="store_true")
    parser.add_argument("--fixed-patch-overfit", action="store_true")
    parser.add_argument("--fixed-patch-count", type=int, default=8)
    parser.add_argument("--fixed-patch-steps", type=int, default=2000)
    parser.add_argument("--debug-shapes", action="store_true")
    return parser.parse_args()


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_mode_defaults(args):
    if args.overfit_one_sample:
        if args.output_dir is None:
            args.output_dir = Path(__file__).resolve().parent / "runs" / "unet3d_btcv"
        if args.epochs is None:
            args.epochs = 1
        if args.max_iters is None:
            args.max_iters = 20
        if args.lr is None:
            args.lr = 1e-2
        if args.loss is None:
            args.loss = "ce"
        if args.patches_per_case is None:
            args.patches_per_case = 1
        if args.val_every is None:
            args.val_every = 1
        if args.background_crop_prob is None:
            args.background_crop_prob = None
    elif args.fixed_patch_overfit:
        if args.output_dir is None:
            args.output_dir = Path(__file__).resolve().parent / "runs" / "unet3d_btcv_fixed_patch"
        if args.epochs is None:
            args.epochs = 1000
        if args.max_iters is None:
            args.max_iters = args.fixed_patch_steps
        if args.lr is None:
            args.lr = 1e-3
        if args.loss is None:
            args.loss = "ce_dice"
        if args.patches_per_case is None:
            args.patches_per_case = 1
        if args.val_every is None:
            args.val_every = 10**9
        if args.background_crop_prob is None:
            args.background_crop_prob = 0.0
    else:
        if args.output_dir is None:
            args.output_dir = Path(__file__).resolve().parent / "runs" / "unet3d_btcv_real"
        if args.epochs is None:
            args.epochs = 50
        if args.max_iters is None:
            args.max_iters = 0
        if args.lr is None:
            args.lr = 1e-3
        if args.loss is None:
            args.loss = "ce_dice"
        if args.patches_per_case is None:
            args.patches_per_case = 8
        if args.val_every is None:
            args.val_every = 5
        if args.background_crop_prob is None:
            args.background_crop_prob = 0.1


def _as_tuple(values, expected_len):
    if len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {values}")
    return tuple(float(v) for v in values)


def _parse_history_float(value):
    if value in (None, "", "nan", "NaN", "NAN"):
        return float("nan")
    return float(value)


def load_metrics_history(path: Path):
    path = Path(path)
    if not path.is_file():
        return []

    history = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = {}
            for key, value in raw_row.items():
                if key == "epoch":
                    row[key] = int(float(value))
                elif key in HISTORY_NUMERIC_COLUMNS:
                    row[key] = _parse_history_float(value)
                else:
                    row[key] = value

            row.setdefault("train_loss", float("nan"))
            row.setdefault("val_loss", float("nan"))
            row.setdefault("mean_val_dice", float("nan"))
            row.setdefault("lr", float("nan"))
            row.setdefault("epoch_train_time_sec", float("nan"))
            row.setdefault("ce_loss", float("nan"))
            row.setdefault("dice_loss", float("nan"))
            row.setdefault("soft_fg_dice", float("nan"))
            row.setdefault("hard_fg_dice", float("nan"))
            for column in VAL_DICE_COLUMNS:
                row.setdefault(column, float("nan"))
            history.append(row)

    history.sort(key=lambda row: row["epoch"])
    return history


def upsert_history_row(history, new_row):
    by_epoch = {row["epoch"]: row for row in history}
    by_epoch[int(new_row["epoch"])] = new_row
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def write_metrics_history(path: Path, history):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "mean_val_dice",
        "lr",
        "epoch_train_time_sec",
        "ce_loss",
        "dice_loss",
        "soft_fg_dice",
        "hard_fg_dice",
        *VAL_DICE_COLUMNS,
    ]
    with tmp_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            output_row = {}
            for field in fieldnames:
                value = row.get(field, float("nan"))
                if field == "epoch":
                    output_row[field] = int(value)
                elif isinstance(value, (int, float)) and np.isnan(value):
                    output_row[field] = "nan"
                else:
                    output_row[field] = value
            writer.writerow(output_row)
    tmp_path.replace(path)


def write_best_validation_summary(output_dir: Path, history):
    output_dir = Path(output_dir)
    finite_rows = [row for row in history if row.get("mean_val_dice") is not None and not np.isnan(row["mean_val_dice"])]
    if not finite_rows:
        return None

    best_row = max(finite_rows, key=lambda row: row["mean_val_dice"])
    path = output_dir / "best_validation_summary.csv"
    fieldnames = ["epoch", "mean_val_dice", *VAL_DICE_COLUMNS]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        summary_row = {"epoch": int(best_row["epoch"]), "mean_val_dice": float(best_row["mean_val_dice"])}
        for column in VAL_DICE_COLUMNS:
            summary_row[column] = best_row.get(column, float("nan"))
        writer.writerow(summary_row)
    return path


def print_checkpoint_status(output_dir: Path):
    output_dir = Path(output_dir)
    for name in ("best.pt", "last.pt"):
        path = output_dir / name
        if not path.is_file():
            print(f"[checkpoint status] {name}: missing")
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        epoch = int(checkpoint.get("epoch", -1)) + 1
        best_val_dice = checkpoint.get("best_val_dice", float("nan"))
        model_type = infer_model_type(checkpoint)
        print(f"[checkpoint status] {name}: epoch={epoch} best_val_dice={best_val_dice:.6f} model_type={model_type}")


def print_preprocessing_sanity(train_pairs, cache, limit=3):
    for image_path, label_path in train_pairs[:limit]:
        case = cache.load_from_paths(image_path, label_path)
        label_unique = sorted(np.unique(case.label).astype(int).tolist()) if case.label is not None else []
        print("[preprocess sanity]", case.image_path.name)
        print("  original shape:", case.original_shape)
        print("  original spacing:", tuple(round(v, 6) for v in case.original_spacing))
        print("  resampled shape:", case.resampled_shape)
        print("  target spacing:", case.target_spacing)
        print("  effective spacing:", tuple(round(v, 6) for v in case.effective_spacing))
        print("  image min/max after HU window:", float(case.image.min()), float(case.image.max()))
        print("  label unique values:", label_unique)
        print("  image/label aligned:", case.label is not None and case.image.shape == case.label.shape)


def save_checkpoint(path, model, optimizer, epoch, step, best_val_dice, args, split_payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    saved_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    torch.save(
        {
            "model_type": saved_args.get("model_type", "unet3d"),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "best_val_dice": best_val_dice,
            "args": saved_args,
            "split": split_payload,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


@torch.no_grad()
def validate_patch_level(model, loader, criterion, loss_name, device, max_iters):
    model.eval()
    losses = []
    dices = []

    for idx, (images, labels) in enumerate(loader):
        if max_iters > 0 and idx >= max_iters:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        ce_loss = criterion(logits, labels)
        if loss_name == "ce":
            loss = ce_loss
        else:
            loss = 0.5 * ce_loss + 0.5 * foreground_dice_loss(logits, labels)
        losses.append(loss.item())
        dices.append(mean_foreground_dice(logits, labels))

    model.train()
    finite_dices = [dice for dice in dices if not np.isnan(dice)]
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    mean_dice = float(np.mean(finite_dices)) if finite_dices else float("nan")
    return mean_loss, mean_dice


@torch.no_grad()
def validate_volume(model, val_pairs, criterion, loss_name, device, patch_size, overlap, batch_size, cache):
    from utils.inference import sliding_window_predict_logits
    from utils.metrics import dice_per_class

    model.eval()
    case_summaries = []
    per_class_values = {class_idx: [] for class_idx in range(1, NUM_CLASSES)}
    val_losses = []
    case_mean_dices = []

    for image_path, label_path in val_pairs:
        case = cache.load_from_paths(image_path, label_path)
        logits = sliding_window_predict_logits(
            model,
            case.image,
            patch_size=patch_size,
            overlap=overlap,
            batch_size=batch_size,
            device=device,
        )

        logits_b = logits.unsqueeze(0).to(device)
        labels_b = torch.from_numpy(case.label).long().unsqueeze(0).to(device)
        ce_loss = criterion(logits_b, labels_b)
        if loss_name == "ce":
            case_loss = ce_loss
        else:
            case_loss = 0.5 * ce_loss + 0.5 * foreground_dice_loss(logits_b, labels_b)
        val_losses.append(float(case_loss.item()))

        prediction = logits.argmax(dim=0).cpu().numpy().astype(np.uint8)
        per_class_dice, mean_fg_dice = dice_per_class(prediction, case.label)
        case_mean_dices.append(mean_fg_dice)

        case_id = strip_nii_gz_suffix(case.image_path)
        print(f"[val case] {case_id}")
        for class_idx in range(1, NUM_CLASSES):
            value = per_class_dice.get(class_idx, float("nan"))
            if np.isnan(value):
                print(f"  class {class_idx:02d} Dice = nan")
            else:
                print(f"  class {class_idx:02d} Dice = {value:.6f}")
            if not np.isnan(value):
                per_class_values[class_idx].append(value)
        print(f"  mean Dice = {mean_fg_dice:.6f}")

        case_summaries.append(
            {
                "case_id": case_id,
                "image_path": str(case.image_path),
                "label_path": str(case.label_path),
                "mean_foreground_dice": float(mean_fg_dice),
                "val_loss": float(case_loss.item()),
                "per_class_dice": {
                    str(class_idx): (None if np.isnan(value) else float(value))
                    for class_idx, value in per_class_dice.items()
                },
            }
        )

    # Absence handling:
    # - per-case Dice is NaN when a class is absent from both GT and prediction
    # - dataset-level class Dice averages only finite per-case values
    # This avoids rewarding missing organs with a perfect Dice of 1.0.
    dataset_per_class = {
        class_idx: (float(np.mean(values)) if values else float("nan"))
        for class_idx, values in per_class_values.items()
    }
    dataset_mean_foreground_dice = (
        float(np.mean([v for v in case_mean_dices if not np.isnan(v)])) if case_mean_dices else float("nan")
    )
    dataset_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

    print("[validation summary]")
    for class_idx in range(1, NUM_CLASSES):
        value = dataset_per_class[class_idx]
        if np.isnan(value):
            print(f"  class {class_idx:02d} mean Dice = nan")
        else:
            print(f"  class {class_idx:02d} mean Dice = {value:.6f}")
    print(f"  mean foreground Dice = {dataset_mean_foreground_dice:.6f}")

    model.train()
    return {
        "cases": case_summaries,
        "dataset_per_class": dataset_per_class,
        "mean_val_dice": dataset_mean_foreground_dice,
        "val_loss": dataset_val_loss,
    }


def main():
    args = parse_args()
    resolve_mode_defaults(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)
    if args.fixed_patch_overfit:
        mode_name = "fixed-patch-overfit"
    elif args.overfit_one_sample:
        mode_name = "overfit-one-sample"
    else:
        mode_name = "real-baseline"
    print("[mode]", mode_name)
    print(f"[loss] selected={args.loss}")
    print(f"[lr] requested={args.lr}")
    print(f"[model type] {args.model_type}")
    print(f"[target spacing] {tuple(float(v) for v in args.target_spacing)}")
    print(f"[hu window] {tuple(float(v) for v in args.hu_window)}")
    print_checkpoint_status(args.output_dir if args.output_dir is not None else Path(__file__).resolve().parent / "runs" / "unet3d_btcv_real")

    all_pairs = pair_btcv_files(args.root)
    split_payload = None
    train_pairs = all_pairs
    val_pairs = []

    if args.overfit_one_sample:
        train_pairs = all_pairs[:1]
    else:
        split_path = args.output_dir / "split.json"
        if args.split_path is not None:
            from utils.data import load_split_json

            split_payload, train_pairs, val_pairs = load_split_json(args.split_path)
            save_split_json(split_path, train_pairs, val_pairs, split_payload.get("seed", args.split_seed), split_payload.get("val_fraction", args.val_fraction), args.root)
        elif split_path.is_file():
            from utils.data import load_split_json

            split_payload, train_pairs, val_pairs = load_split_json(split_path)
        else:
            train_pairs, val_pairs = split_btcv_pairs(all_pairs, seed=args.split_seed, val_fraction=args.val_fraction)
            split_path = save_split_json(split_path, train_pairs, val_pairs, args.split_seed, args.val_fraction, args.root)
            split_payload = {
                "root": str(Path(args.root)),
                "seed": int(args.split_seed),
                "val_fraction": float(args.val_fraction),
                "train": [str(Path(p[0])) for p in train_pairs],
                "val": [str(Path(p[0])) for p in val_pairs],
                "train_with_labels": [[str(Path(p[0])), str(Path(p[1]))] for p in train_pairs],
                "val_with_labels": [[str(Path(p[0])), str(Path(p[1]))] for p in val_pairs],
            }
        if args.split_path is not None:
            from utils.data import load_split_json

            baseline_payload, baseline_train_pairs, baseline_val_pairs = load_split_json(args.split_path)
            train_identical = baseline_train_pairs == train_pairs
            val_identical = baseline_val_pairs == val_pairs
            print("[split verification]")
            print(f"train identical: {train_identical}")
            print(f"validation identical: {val_identical}")
        print("[split path]", args.output_dir / "split.json")
        print("[train cases]")
        for img_path, _ in train_pairs:
            print(f"  {Path(img_path).name}")
        print("[validation cases]")
        for img_path, _ in val_pairs:
            print(f"  {Path(img_path).name}")
        print(f"[patches per case] {args.patches_per_case}")
        print(f"[expected train samples per epoch] {len(train_pairs) * args.patches_per_case}")

    cache_dir = args.cache_dir if args.cache_dir is not None else args.output_dir / "preprocess_cache"
    preprocess_cache = BTCVPreprocessingCache(
        cache_dir=cache_dir,
        target_spacing=_as_tuple(args.target_spacing, 3),
        hu_window=_as_tuple(args.hu_window, 2),
    )
    print_preprocessing_sanity(train_pairs, preprocess_cache, limit=3)

    metrics_csv = args.output_dir / "metrics.csv"
    history = load_metrics_history(metrics_csv)
    if history:
        epochs = [row["epoch"] for row in history]
        print(f"[metrics resume] loaded_rows={len(history)} epoch_range={min(epochs)}-{max(epochs)}")
    else:
        print("[metrics resume] loaded_rows=0")

    quick_val_loader = None
    if args.fixed_patch_overfit:
        patch_specs = build_fixed_patch_specs(
            train_pairs,
            preprocess_cache,
            patch_size=tuple(args.patch_size),
            count=args.fixed_patch_count,
        )
        print("[fixed patch specs]")
        for idx, spec in enumerate(patch_specs):
            print(
                f"  {idx}: {spec.image_path.name} start={spec.start} "
                f"fg_classes={spec.num_foreground_classes} fg_voxels={spec.foreground_voxels}"
            )
        train_dataset = FixedPatchDataset(
            args.root,
            patch_specs=patch_specs,
            cache_dir=cache_dir,
            target_spacing=_as_tuple(args.target_spacing, 3),
            hu_window=_as_tuple(args.hu_window, 2),
            sanity_print=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        val_pairs = []
    else:
        train_dataset = PreprocessedBTCVPatchDataset(
            args.root,
            patch_size=tuple(args.patch_size),
            random_crop=not args.overfit_one_sample,
            foreground_crop_prob=args.foreground_crop_prob,
            background_crop_prob=args.background_crop_prob,
            patches_per_case=args.patches_per_case,
            overfit_one_sample=args.overfit_one_sample,
            pairs=train_pairs,
            cache_dir=cache_dir,
            target_spacing=_as_tuple(args.target_spacing, 3),
            hu_window=_as_tuple(args.hu_window, 2),
            sanity_print=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=not args.overfit_one_sample,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        if not args.overfit_one_sample and args.quick_val_iters > 0 and val_pairs:
            quick_val_dataset = PreprocessedBTCVPatchDataset(
                args.root,
                patch_size=tuple(args.patch_size),
                random_crop=False,
                foreground_crop_prob=0.0,
                background_crop_prob=1.0,
                patches_per_case=1,
                overfit_one_sample=False,
                pairs=val_pairs,
                cache_dir=cache_dir,
                target_spacing=_as_tuple(args.target_spacing, 3),
                hu_window=_as_tuple(args.hu_window, 2),
                sanity_print=False,
            )
            quick_val_loader = DataLoader(
                quick_val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )

    model = build_model(
        args.model_type,
        in_channels=1,
        num_classes=NUM_CLASSES,
        base_channels=args.base_channels,
        debug_shapes=args.debug_shapes,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"[lr] optimizer_before_resume={[group['lr'] for group in optimizer.param_groups]}")
    print(f"[len(train_loader)] {len(train_loader)}")
    total_params, trainable_params = count_parameters(model)
    print(f"[params] total={total_params} trainable={trainable_params}")

    start_epoch = 0
    step = 0
    best_val_dice = float("-inf")
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume, model, optimizer, device)
        checkpoint_model_type = infer_model_type(checkpoint)
        if checkpoint_model_type != args.model_type:
            print(f"[resume warning] checkpoint model_type={checkpoint_model_type} args.model_type={args.model_type}")
        start_epoch = int(checkpoint["epoch"]) + 1
        step = int(checkpoint["step"])
        best_val_dice = float(checkpoint.get("best_val_dice", best_val_dice))
        print(f"[resume] {args.resume} epoch={start_epoch} step={step} best_val_dice={best_val_dice:.6f}")
        print(f"[lr] optimizer_after_resume={[group['lr'] for group in optimizer.param_groups]}")

    train_start = time.time()
    fixed_patch_mode = args.fixed_patch_overfit

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        model.train()
        train_losses = []
        ce_losses = []
        dice_losses = []
        soft_dices = []
        hard_dices = []
        stop_training = False
        should_validate = (
            (not args.overfit_one_sample)
            and (not fixed_patch_mode)
            and bool(val_pairs)
            and ((epoch + 1) % args.val_every == 0)
        )

        print(f"[epoch] {epoch + 1}/{args.epochs}")
        for images, labels in train_loader:
            labels_cpu = labels.detach().cpu()
            if step == 0 or (step + 1) % args.log_every == 0:
                print("[batch] image:", tuple(images.shape), "label:", tuple(labels.shape))
                print("[batch] foreground voxels:", int((labels_cpu > 0).sum().item()))
                print("[batch] labels present:", sorted(torch.unique(labels_cpu).tolist()))
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            ce_loss = criterion(logits, labels)
            dice_loss = foreground_dice_loss(logits, labels)
            if args.loss == "ce":
                loss = ce_loss
            else:
                loss = 0.5 * ce_loss + 0.5 * dice_loss
            loss.backward()
            optimizer.step()

            step += 1
            train_losses.append(float(loss.item()))
            ce_losses.append(float(ce_loss.item()))
            dice_losses.append(float(dice_loss.item()))
            soft_dice = soft_foreground_dice(logits, labels)
            hard_dice = hard_foreground_dice(logits, labels)
            soft_dices.append(float(soft_dice) if soft_dice == soft_dice else float("nan"))
            hard_dices.append(float(hard_dice) if hard_dice == hard_dice else float("nan"))
            if step == 1 or step % args.log_every == 0:
                if fixed_patch_mode:
                    print(
                        f"[iter {step}] epoch={epoch + 1} ce={ce_loss.item():.6f} dice={dice_loss.item():.6f} "
                        f"total={loss.item():.6f} soft_fg_dice={soft_dice:.6f} hard_fg_dice={hard_dice:.6f} "
                        f"lr={[group['lr'] for group in optimizer.param_groups]}"
                    )
                else:
                    print(
                        f"[iter {step}] epoch={epoch + 1} loss={loss.item():.6f} "
                        f"lr={[group['lr'] for group in optimizer.param_groups]}"
                    )
                if device.type == "cuda":
                    allocated = torch.cuda.memory_allocated(device) / 1024**3
                    reserved = torch.cuda.memory_reserved(device) / 1024**3
                    print(f"[cuda] allocated={allocated:.2f}GB reserved={reserved:.2f}GB")

            if args.max_iters > 0 and step >= args.max_iters:
                stop_training = True
                break

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        mean_ce_loss = float(np.mean(ce_losses)) if ce_losses else float("nan")
        mean_dice_loss = float(np.mean(dice_losses)) if dice_losses else float("nan")
        mean_soft_dice = float(np.nanmean(soft_dices)) if soft_dices else float("nan")
        mean_hard_dice = float(np.nanmean(hard_dices)) if hard_dices else float("nan")
        val_loss = float("nan")
        mean_val_dice = float("nan")

        if should_validate:
            val_start = time.time()
            if quick_val_loader is not None:
                quick_loss, quick_dice = validate_patch_level(
                    model,
                    quick_val_loader,
                    criterion,
                    args.loss,
                    device,
                    args.quick_val_iters,
                )
                print(f"[quick val] loss={quick_loss:.6f} mean_fg_dice={quick_dice:.6f}")

            val_summary = validate_volume(
                model,
                val_pairs,
                criterion,
                args.loss,
                device,
                patch_size=tuple(args.patch_size),
                overlap=args.val_overlap,
                batch_size=args.val_batch_size,
                cache=preprocess_cache,
            )
            val_loss = val_summary["val_loss"]
            mean_val_dice = val_summary["mean_val_dice"]
            print(f"[validation runtime] {time.time() - val_start:.2f}s")

            if not np.isnan(mean_val_dice) and mean_val_dice > best_val_dice:
                best_val_dice = mean_val_dice
                save_checkpoint(
                    args.output_dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    step,
                    best_val_dice,
                    args,
                    split_payload,
                )
                print(
                    f"[best checkpoint] epoch={epoch + 1} mean_val_dice={mean_val_dice:.6f} "
                    f"saved to {args.output_dir / 'best.pt'}"
                )
        else:
            print(f"[validation] skipped at epoch {epoch + 1}; full-volume validation runs every {args.val_every} epochs")

        lr = float(optimizer.param_groups[0]["lr"])
        epoch_time = time.time() - epoch_start
        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mean_val_dice": mean_val_dice,
            "lr": lr,
            "epoch_train_time_sec": epoch_time,
            "ce_loss": mean_ce_loss,
            "dice_loss": mean_dice_loss,
            "soft_fg_dice": mean_soft_dice,
            "hard_fg_dice": mean_hard_dice,
        }
        for column in VAL_DICE_COLUMNS:
            epoch_row[column] = float("nan")
        if should_validate:
            for class_idx, class_name in BTCV_FOREGROUND_CLASSES:
                column = f"val_dice_c{class_idx:02d}_{class_name}"
                epoch_row[column] = float(val_summary["dataset_per_class"].get(class_idx, float("nan")))

        history = upsert_history_row(history, epoch_row)
        write_metrics_history(metrics_csv, history)
        plot_training_curves(history, args.output_dir)
        plot_per_class_validation_dice(history, args.output_dir)
        print(f"[metrics] {metrics_csv}")
        print(f"[plots] {args.output_dir / 'loss_curve.png'}")
        print(f"[plots] {args.output_dir / 'dice_curve.png'}")
        print(f"[plots] {args.output_dir / 'per_class_val_dice.png'}")

        save_checkpoint(
            args.output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            step,
            best_val_dice,
            args,
            split_payload,
        )
        print("[checkpoint] saved last.pt")
        print(f"[epoch runtime] {epoch_time:.2f}s")

        write_best_validation_summary(args.output_dir, history)

        if stop_training:
            break

    elapsed_min = (time.time() - train_start) / 60.0
    print(f"[done] epochs={len(history)} steps={step} elapsed_min={elapsed_min:.2f} best_val_dice={best_val_dice:.6f}")


if __name__ == "__main__":
    main()
