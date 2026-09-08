from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import nibabel as nib
import torch

from models import NUM_CLASSES
from utils.data import PreprocessedBTCVPatchDataset, load_preprocessed_case_from_pair, load_split_json, strip_nii_gz_suffix
from utils.inference import load_model_from_checkpoint, sliding_window_predict_logits
from utils.metrics import dice_per_class
from utils.visualization import choose_slice_index, save_segmentation_comparison_figure, slice_volume


def parse_args():
    parser = argparse.ArgumentParser(description="BTCV 3D U-Net evaluation entry point.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a checkpoint. Defaults depend on the selected mode.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "btcv",
        help="BTCV dataset root containing RawData/Training/img and label.",
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=None,
        help="Path to split.json for real-baseline evaluation.",
    )
    parser.add_argument("--device", default=None, help="cuda, cpu, or leave empty for auto.")
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--axis", type=int, choices=(0, 1, 2), default=None)
    parser.add_argument("--slice-index", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overfit-one-sample", action="store_true")
    parser.add_argument("--image-path", type=Path, default=None)
    parser.add_argument("--label-path", type=Path, default=None)
    parser.add_argument("--prediction-path", type=Path, default=None)
    parser.add_argument("--target-spacing", type=float, nargs=3, default=(1.5, 1.5, 2.0))
    parser.add_argument("--hu-window", type=float, nargs=2, default=(-175.0, 250.0))
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser.parse_args()


def resolve_mode_defaults(args):
    if args.overfit_one_sample:
        if args.checkpoint is None:
            args.checkpoint = Path(__file__).resolve().parent / "runs" / "unet3d_btcv" / "last.pt"
        if args.axis is None:
            args.axis = 0
    else:
        if args.checkpoint is None:
            args.checkpoint = Path(__file__).resolve().parent / "runs" / "unet3d_btcv_real" / "best.pt"
        if args.split_path is None:
            args.split_path = Path(__file__).resolve().parent / "runs" / "unet3d_btcv_real" / "split.json"
        if args.axis is None:
            args.axis = 2


def load_real_case(args, checkpoint):
    cache_dir = args.cache_dir if args.cache_dir is not None else Path(__file__).resolve().parent / "runs" / "unet3d_btcv_real" / "preprocess_cache"
    if args.split_path.is_file():
        _, _, val_pairs = load_split_json(args.split_path)
    else:
        split = checkpoint.get("split")
        if split is None:
            raise FileNotFoundError(
                f"Split file not found at {args.split_path} and checkpoint does not contain split metadata."
            )
        val_pairs = [(Path(img), Path(lbl)) for img, lbl in split.get("val_with_labels", [])]

    if not val_pairs:
        raise RuntimeError("No validation cases found in the saved split.")
    if not 0 <= args.index < len(val_pairs):
        raise IndexError(f"index must be between 0 and {len(val_pairs) - 1}")

    image_path, label_path = val_pairs[args.index]
    from utils.preprocessing import BTCVPreprocessingCache

    cache = BTCVPreprocessingCache(
        cache_dir=cache_dir,
        target_spacing=tuple(float(v) for v in args.target_spacing),
        hu_window=tuple(float(v) for v in args.hu_window),
    )
    return load_preprocessed_case_from_pair(image_path, label_path, cache=cache)


def main():
    args = parse_args()
    resolve_mode_defaults(args)

    model, checkpoint, device, base_channels = load_model_from_checkpoint(
        args.checkpoint,
        device=args.device,
        base_channels=args.base_channels,
    )
    model_type = checkpoint.get("model_type", "unet3d")
    print("[model type]", model_type)

    if args.overfit_one_sample:
        dataset = PreprocessedBTCVPatchDataset(
            args.root,
            patch_size=tuple(args.patch_size),
            random_crop=False,
            foreground_crop_prob=0.0,
            background_crop_prob=1.0,
            overfit_one_sample=True,
            sanity_print=False,
            cache_dir=args.cache_dir if args.cache_dir is not None else Path(__file__).resolve().parent / "runs" / "unet3d_btcv" / "preprocess_cache",
            target_spacing=tuple(float(v) for v in args.target_spacing),
            hu_window=tuple(float(v) for v in args.hu_window),
        )
        image, label = dataset[0]
        image_path, label_path = dataset.pairs[0]
        if args.prediction_path is not None:
            prediction = nib.load(str(args.prediction_path)).get_fdata(dtype=np.float32).astype(np.uint8)
            prediction = prediction.astype(np.uint8)
            logits_shape = None
        else:
            image_batch = image.unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(image_batch)
                prediction = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            logits_shape = tuple(logits.shape)
        gt = label.numpy()
        case_id = strip_nii_gz_suffix(image_path)
        ct_volume = image.squeeze(0).cpu().numpy()
    else:
        case = load_real_case(args, checkpoint)
        image_path, label_path = case.image_path, case.label_path
        case_id = strip_nii_gz_suffix(image_path)
        ct_volume = case.image
        gt = case.label
        if args.prediction_path is not None:
            prediction = nib.load(str(args.prediction_path)).get_fdata(dtype=np.float32).astype(np.uint8)
            logits_shape = None
        else:
            logits = sliding_window_predict_logits(
                model,
                case.image,
                patch_size=tuple(args.patch_size),
                overlap=args.overlap,
                batch_size=args.batch_size,
                device=device,
            )
            prediction = logits.argmax(dim=0).cpu().numpy().astype(np.uint8)
            logits_shape = tuple(logits.shape)

    if gt is None:
        raise ValueError("This evaluation requires a ground-truth label volume.")
    if prediction.shape != ct_volume.shape:
        raise ValueError(f"Prediction shape {prediction.shape} does not match image shape {ct_volume.shape}")

    per_class_dice, mean_fg_dice = dice_per_class(prediction, gt)

    print("[checkpoint]", args.checkpoint)
    print("[device]", device)
    print("[case]", case_id)
    print("[input shape]", tuple((1, 1) + tuple(ct_volume.shape)))
    if logits_shape is not None:
        print("[logits shape]", logits_shape)
    print("[gt unique labels]", sorted(np.unique(gt).astype(int).tolist()))
    print("[pred unique labels]", sorted(np.unique(prediction).astype(int).tolist()))
    for class_idx in range(1, NUM_CLASSES):
        value = per_class_dice.get(class_idx, float("nan"))
        if np.isnan(value):
            print(f"class {class_idx:02d} Dice = nan")
        else:
            print(f"class {class_idx:02d} Dice = {value:.6f}")
    print(f"mean foreground Dice = {mean_fg_dice:.6f}")

    slice_index = choose_slice_index(gt, args.axis, args.slice_index)
    ct_slice = slice_volume(ct_volume, args.axis, slice_index)
    gt_slice = slice_volume(gt, args.axis, slice_index)
    pred_slice = slice_volume(prediction, args.axis, slice_index)

    output_path = args.output
    if output_path is None:
        if args.overfit_one_sample:
            output_path = (
                Path(__file__).resolve().parent
                / "runs"
                / "visualizations"
                / f"{case_id}_overfit_eval_axis{args.axis}_slice{slice_index}.png"
            )
        else:
            output_path = (
                Path(__file__).resolve().parent
                / "runs"
                / "unet3d_btcv_real"
                / "visualizations"
                / f"{case_id}_val{args.index}_axis{args.axis}_slice{slice_index}.png"
            )

    title = f"{case_id} | axis {args.axis} | slice {slice_index}"
    saved_path = save_segmentation_comparison_figure(ct_slice, gt_slice, pred_slice, output_path, title)

    summary = {
        "checkpoint": str(args.checkpoint),
        "case_id": case_id,
        "image_path": str(image_path),
        "label_path": str(label_path) if label_path is not None else None,
        "device": str(device),
        "base_channels": base_channels,
        "mean_foreground_dice": float(mean_fg_dice),
        "dice_per_class": {
            str(class_idx): (None if np.isnan(value) else float(value))
            for class_idx, value in per_class_dice.items()
        },
        "prediction_unique_labels": [int(x) for x in np.unique(prediction)],
        "gt_unique_labels": [int(x) for x in np.unique(gt)],
        "slice_index": int(slice_index),
        "axis": int(args.axis),
        "visualization_path": str(saved_path),
    }
    summary_path = saved_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print("[saved visualization]", saved_path)
    print("[saved summary]", summary_path)


if __name__ == "__main__":
    main()
