from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from models.unet3d import NUM_CLASSES


BTCV_CLASS_NAMES = [
    "background",
    "spleen",
    "right kidney",
    "left kidney",
    "gallbladder",
    "esophagus",
    "liver",
    "stomach",
    "aorta",
    "inferior vena cava",
    "portal vein and splenic vein",
    "pancreas",
    "right adrenal gland",
    "left adrenal gland",
]

BTCV_FOREGROUND_CLASS_COLUMNS = {
    1: "val_dice_c01_spleen",
    2: "val_dice_c02_right_kidney",
    3: "val_dice_c03_left_kidney",
    4: "val_dice_c04_gallbladder",
    5: "val_dice_c05_esophagus",
    6: "val_dice_c06_liver",
    7: "val_dice_c07_stomach",
    8: "val_dice_c08_aorta",
    9: "val_dice_c09_inferior_vena_cava",
    10: "val_dice_c10_portal_vein",
    11: "val_dice_c11_pancreas",
    12: "val_dice_c12_right_adrenal",
    13: "val_dice_c13_left_adrenal",
}

BTCV_CLASS_COLORS = [
    "#000000",
    "#1f77b4",  # spleen
    "#ff7f0e",  # right kidney
    "#2ca02c",  # left kidney
    "#d62728",  # gallbladder
    "#9467bd",  # esophagus
    "#8c564b",  # liver
    "#17becf",  # stomach
    "#7f7f7f",  # aorta
    "#bcbd22",  # inferior vena cava
    "#393b79",  # portal vein and splenic vein
    "#fdbb84",  # pancreas
    "#7CFC00",  # right adrenal gland
    "#FF00FF",  # left adrenal gland
]


def get_btcv_label_cmap():
    cmap = ListedColormap(BTCV_CLASS_COLORS[:NUM_CLASSES], name="btcv_labels")
    norm = BoundaryNorm(np.arange(-0.5, NUM_CLASSES + 0.5, 1.0), cmap.N)
    return cmap, norm


def get_btcv_label_legend_handles():
    cmap, _ = get_btcv_label_cmap()
    return [
        Patch(facecolor=cmap(idx), edgecolor="black", label=f"{idx}: {BTCV_CLASS_NAMES[idx]}")
        for idx in range(NUM_CLASSES)
    ]


def normalize_for_display(image_slice: np.ndarray) -> np.ndarray:
    image_slice = image_slice.astype(np.float32)
    lo, hi = np.percentile(image_slice, [1, 99])
    if hi <= lo:
        return image_slice
    image_slice = np.clip(image_slice, lo, hi)
    return (image_slice - lo) / (hi - lo + 1e-6)


def choose_slice_index(volume: np.ndarray, axis: int, slice_index: int | None):
    if slice_index is not None:
        return slice_index
    foreground_counts = np.sum(volume > 0, axis=tuple(i for i in range(volume.ndim) if i != axis))
    if np.any(foreground_counts > 0):
        return int(np.argmax(foreground_counts))
    return volume.shape[axis] // 2


def slice_volume(volume: np.ndarray, axis: int, slice_index: int):
    return np.take(volume, indices=slice_index, axis=axis)


def overlay_mask(ax, ct_slice, mask_slice, title, cmap, norm, alpha=0.45):
    ax.imshow(normalize_for_display(ct_slice), cmap="gray")
    masked = np.ma.masked_where(mask_slice == 0, mask_slice)
    ax.imshow(masked, cmap=cmap, norm=norm, alpha=alpha, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")


def save_segmentation_comparison_figure(
    ct_slice: np.ndarray,
    gt_slice: np.ndarray,
    pred_slice: np.ndarray,
    output_path: Path,
    title: str,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    label_cmap, label_norm = get_btcv_label_cmap()
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))

    axes[0].imshow(normalize_for_display(ct_slice), cmap="gray")
    axes[0].set_title("CT")
    axes[0].axis("off")

    axes[1].imshow(gt_slice, cmap=label_cmap, norm=label_norm, interpolation="nearest")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_slice, cmap=label_cmap, norm=label_norm, interpolation="nearest")
    axes[2].set_title("U-Net Prediction")
    axes[2].axis("off")

    overlay_mask(axes[3], ct_slice, gt_slice, "CT + GT Overlay", cmap=label_cmap, norm=label_norm)
    overlay_mask(axes[4], ct_slice, pred_slice, "CT + Prediction Overlay", cmap=label_cmap, norm=label_norm)

    fig.suptitle(title)
    fig.legend(
        handles=get_btcv_label_legend_handles(),
        loc="lower center",
        ncol=2,
        frameon=True,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.subplots_adjust(bottom=0.34)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_validation_dice_subset(history, output_path: Path, class_indices, title: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return None

    fig, ax = plt.subplots(figsize=(12, 7))
    plotted = False
    for class_idx in class_indices:
        if not 1 <= class_idx < NUM_CLASSES:
            continue
        label_name = BTCV_CLASS_NAMES[class_idx]
        column_name = BTCV_FOREGROUND_CLASS_COLUMNS[class_idx]
        epochs = []
        values = []
        for row in history:
            value = row.get(column_name, float("nan"))
            if value is None or np.isnan(value):
                continue
            epochs.append(int(row["epoch"]))
            values.append(float(value))
        if not epochs:
            continue
        ax.plot(epochs, values, marker="o", linewidth=1.6, label=f"{class_idx}: {label_name}", color=BTCV_CLASS_COLORS[class_idx])
        plotted = True

    ax.set_xlabel("epoch")
    ax.set_ylabel("Dice")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_per_class_validation_dice(history, output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not history:
        return None, None, None

    all_classes = list(range(1, NUM_CLASSES))
    large_organs = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
    small_organs = [5, 12, 13]

    all_path = _plot_validation_dice_subset(
        history,
        output_dir / "per_class_val_dice.png",
        all_classes,
        "BTCV Validation Dice by Class",
    )
    large_path = _plot_validation_dice_subset(
        history,
        output_dir / "large_organs_val_dice.png",
        large_organs,
        "BTCV Validation Dice - Large Organs",
    )
    small_path = _plot_validation_dice_subset(
        history,
        output_dir / "small_organs_val_dice.png",
        small_organs,
        "BTCV Validation Dice - Small Organs",
    )
    return all_path, large_path, small_path


def plot_training_curves(history, output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not history:
        return

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history if row["val_loss"] is not None and not np.isnan(row["val_loss"])]
    val_loss_epochs = [row["epoch"] for row in history if row["val_loss"] is not None and not np.isnan(row["val_loss"])]
    val_dice = [
        row["mean_val_dice"]
        for row in history
        if row["mean_val_dice"] is not None and not np.isnan(row["mean_val_dice"])
    ]
    val_dice_epochs = [
        row["epoch"]
        for row in history
        if row["mean_val_dice"] is not None and not np.isnan(row["mean_val_dice"])
    ]

    loss_fig = output_dir / "loss_curve.png"
    dice_fig = output_dir / "dice_curve.png"

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", label="train loss")
    if val_loss:
        plt.plot(val_loss_epochs, val_loss, marker="o", label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("BTCV 3D U-Net Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_fig, dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    if val_dice:
        plt.plot(val_dice_epochs, val_dice, marker="o", label="mean val foreground Dice")
    plt.xlabel("epoch")
    plt.ylabel("Dice")
    plt.ylim(0.0, 1.0)
    plt.title("BTCV 3D U-Net Dice")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(dice_fig, dpi=160, bbox_inches="tight")
    plt.close()

    return loss_fig, dice_fig
