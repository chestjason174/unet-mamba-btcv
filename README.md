# BTCV 3D Multi-Organ Segmentation with U-Net and Mamba

This repository contains a 3D abdominal CT multi-organ segmentation project on the **BTCV** dataset.

The main goal is to compare a conventional lightweight **3D U-Net** baseline with a **3D U-Net + single Mamba bottleneck** under a controlled training setup.

## Project Overview

Two main models are implemented:

- **3D U-Net**
- **3D U-Net + Mamba**
  - same encoder
  - same CNN bottleneck
  - one Mamba block inserted at the deepest feature level
  - same decoder

For the `base_channels=8` experiment:

| Model | Parameters | Best Mean Validation Dice |
|---|---:|---:|
| 3D U-Net | 85,134 | 0.5098 |
| 3D U-Net + Mamba | 95,118 | **0.5980** |

The Mamba model adds only **9,984 parameters (+11.7%)** while improving the best mean validation Dice by approximately **0.0882**.

> These results are from the same BTCV split and controlled training pipeline. They are intended as an architecture comparison rather than a state-of-the-art benchmark.

## Dataset

This project uses the **BTCV (Beyond the Cranial Vault)** abdominal multi-organ CT dataset.

The dataset itself is **not included in this repository**.

Place the dataset under:

```text
data/btcv/
```

The current experiment uses:

- 30 BTCV cases
- 24 training cases
- 6 validation cases
- deterministic case-level split with seed 42
- 13 foreground organ classes + background

## Preprocessing

The training and inference pipeline uses the same cached preprocessing:

- resample CT and labels to **1.5 × 1.5 × 2.0 mm**
- trilinear interpolation for CT images
- nearest-neighbor interpolation for segmentation labels
- HU clipping to **[-175, 250]**
- linear intensity scaling to **[0, 1]**
- cached preprocessed volumes for repeated training/inference

Training uses `64 × 64 × 64` 3D patches with foreground-aware sampling.

## Repository Structure

```text
Segmentation/
├── models/
│   ├── __init__.py
│   ├── factory.py
│   ├── unet3d.py
│   └── unet3d_mamba.py
├── utils/
│   ├── __init__.py
│   ├── data.py
│   ├── inference.py
│   ├── metrics.py
│   ├── preprocessing.py
│   └── visualization.py
├── docs/
│   ├── EXPERIMENTS.md
│   ├── MAMBA_BOTTLENECK_V1.md
│   ├── UNET_BASELINE_EXPERIMENT.md
│   └── WORKSPACE_INVENTORY.md
├── archive/
├── train.py
├── evaluate.py
├── README.md
└── .gitignore
```

`archive/` contains older scripts kept for reference. The current training and evaluation entry points are `train.py` and `evaluate.py`.

## Environment

The project was developed with:

```text
Python 3.10
PyTorch 2.11.0 + CUDA 12.8
NVIDIA GPU
mamba-ssm
nibabel
numpy
matplotlib
```

A CUDA-capable GPU is strongly recommended for 3D training and full-volume validation.

## Training

### 3D U-Net

```bash
python train.py \
  --root data/btcv \
  --output-dir runs/unet_base8 \
  --model-type unet3d \
  --base-channels 8 \
  --loss ce_dice \
  --lr 1e-3 \
  --patch-size 64 64 64 \
  --patches-per-case 8 \
  --batch-size 1 \
  --val-every 5 \
  --epochs 200
```

### 3D U-Net + Mamba

```bash
python train.py \
  --root data/btcv \
  --output-dir runs/mamba_base8 \
  --model-type unet3d_mamba \
  --base-channels 8 \
  --loss ce_dice \
  --lr 1e-3 \
  --patch-size 64 64 64 \
  --patches-per-case 8 \
  --batch-size 1 \
  --val-every 5 \
  --epochs 200
```

For a fair architecture comparison, use the **same split, preprocessing, sampling strategy, loss, patch size, base channels, and training budget** for both models.

## Mamba Bottleneck

For a `64^3` input patch with `base_channels=8`, the deepest CNN feature tensor is approximately:

```text
[B, 32, 16, 16, 16]
        ↓ flatten spatial dimensions
[B, 4096, 32]
        ↓ LayerNorm
      Mamba
        ↓ residual connection
[B, 4096, 32]
        ↓ reshape
[B, 32, 16, 16, 16]
```

The current model uses a **single deterministic flattened 3D scan**. It does not use bidirectional scanning, multi-axis Mamba branches, or additional attention blocks.

## Loss

Training uses a combination of:

- multi-class Cross Entropy
- foreground Dice loss

Background is excluded from the foreground Dice calculation.

## Validation

Validation is performed on held-out **full CT volumes** using sliding-window inference.

The main reported metric is:

```text
Mean foreground validation Dice
```

Per-organ Dice scores are also recorded.

## Current Result

For the controlled `base_channels=8` comparison:

| Model | Best Epoch | Mean Dice |
|---|---:|---:|
| 3D U-Net | 165 | 0.5098 |
| 3D U-Net + Mamba | 195 | **0.5980** |

The Mamba model improved 11 of the 13 foreground organ classes at the respective best checkpoints.

Small and thin structures such as the adrenal glands, pancreas, and portal vein remain the main segmentation challenges.

## Evaluation

Example:

```bash
python evaluate.py \
  --checkpoint runs/mamba_base8/best.pt \
  --split-path runs/mamba_base8/split.json \
  --index 0
```

The evaluation pipeline reconstructs the model from checkpoint metadata and performs full-volume sliding-window inference.

## Notes for Collaborators

The following files are intentionally not tracked by Git:

```text
data/
runs/
*.pt
*.pth
*.ckpt
```

After cloning the repository, collaborators need to prepare the BTCV dataset locally and either train a model or obtain a checkpoint separately.

Experiment details and development notes are available under `docs/`.

## Research Status

This is an experimental research project rather than a production segmentation system.

Current work focuses on lightweight 3D segmentation, Mamba-based long-range context modeling, model-capacity experiments, and difficult small-organ segmentation.
