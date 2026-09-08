# BTCV 3D Abdominal Organ Segmentation

This repository contains a small 3D U-Net baseline for BTCV abdominal multi-organ segmentation.

## Current Layout

```text
Segmentation/
├── models/
│   └── unet3d.py
├── utils/
│   ├── data.py
│   ├── inference.py
│   ├── metrics.py
│   └── visualization.py
├── train.py
├── evaluate.py
├── runs/
├── data/
├── archive/
├── README.md
└── .gitignore
```

The old debugging scripts are preserved under `archive/`.

## Data

The BTCV dataset is expected locally at:

```text
data/btcv/RawData/Training/img
data/btcv/RawData/Training/label
```

## Workflows

### Overfit-one-sample debugging

```bash
conda run -n mamba python train.py --overfit-one-sample
```

Evaluate the overfit checkpoint:

```bash
conda run -n mamba python evaluate.py --overfit-one-sample
```

### Real multi-patient baseline

```bash
conda run -n mamba python train.py \
  --root data/btcv \
  --output-dir runs/unet3d_btcv_real \
  --loss ce_dice \
  --lr 1e-3 \
  --epochs 300 \
  --patch-size 64 64 64 \
  --base-channels 8
```

Evaluate one unseen validation patient from the saved split:

```bash
conda run -n mamba python evaluate.py \
  --checkpoint runs/unet3d_btcv_real/best.pt \
  --split-path runs/unet3d_btcv_real/split.json \
  --index 0
```

## Outputs

The real baseline training run writes:

- `split.json`
- `metrics.csv`
- `loss_curve.png`
- `dice_curve.png`
- `last.pt`
- `best.pt`
- validation visualizations under `runs/unet3d_btcv_real/visualizations/`

## Notes

- The U-Net architecture is defined in [models/unet3d.py](models/unet3d.py).
- Dataset loading and patch sampling are in [utils/data.py](utils/data.py).
- Sliding-window inference is in [utils/inference.py](utils/inference.py).
- Dice and Dice loss are in [utils/metrics.py](utils/metrics.py).
- Plotting and segmentation visualization are in [utils/visualization.py](utils/visualization.py).
