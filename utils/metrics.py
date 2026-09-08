from __future__ import annotations

import numpy as np
import torch

from models.unet3d import NUM_CLASSES


def mean_foreground_dice(logits, labels, num_classes=NUM_CLASSES, eps=1e-6):
    preds = logits.argmax(dim=1)
    dices = []
    for class_idx in range(1, num_classes):
        pred_mask = preds == class_idx
        label_mask = labels == class_idx
        denominator = pred_mask.sum() + label_mask.sum()
        if denominator == 0:
            continue
        intersection = (pred_mask & label_mask).sum()
        dices.append((2.0 * intersection.float() + eps) / (denominator.float() + eps))

    if not dices:
        return float("nan")
    return torch.stack(dices).mean().item()


def foreground_dice_loss(logits, labels, num_classes=NUM_CLASSES, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    total_loss = 0.0
    counted_classes = 0

    for class_idx in range(1, num_classes):
        target = (labels == class_idx).float()
        if target.sum() == 0:
            continue

        pred = probs[:, class_idx]
        intersection = (pred * target).sum(dim=(1, 2, 3))
        denominator = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + eps) / (denominator + eps)
        total_loss = total_loss + (1.0 - dice).mean()
        counted_classes += 1

    if counted_classes == 0:
        return logits.new_tensor(0.0)

    return total_loss / counted_classes


def soft_foreground_dice(logits, labels, num_classes=NUM_CLASSES, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    total_dice = 0.0
    counted_classes = 0

    for class_idx in range(1, num_classes):
        target = (labels == class_idx).float()
        if target.sum() == 0:
            continue

        pred = probs[:, class_idx]
        intersection = (pred * target).sum(dim=(1, 2, 3))
        denominator = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + eps) / (denominator + eps)
        total_dice = total_dice + dice.mean()
        counted_classes += 1

    if counted_classes == 0:
        return float("nan")
    return (total_dice / counted_classes).item()


def dice_per_class(prediction: np.ndarray, target: np.ndarray, num_classes: int = NUM_CLASSES):
    dices = {}
    for class_idx in range(1, num_classes):
        pred_mask = prediction == class_idx
        target_mask = target == class_idx
        denominator = pred_mask.sum() + target_mask.sum()
        if denominator == 0:
            dices[class_idx] = float("nan")
            continue
        intersection = np.logical_and(pred_mask, target_mask).sum()
        dices[class_idx] = float((2.0 * intersection) / float(denominator))
    finite = [value for value in dices.values() if not np.isnan(value)]
    mean_foreground_dice = float(np.mean(finite)) if finite else float("nan")
    return dices, mean_foreground_dice


def hard_foreground_dice(logits, labels, num_classes=NUM_CLASSES, eps=1e-6):
    preds = logits.argmax(dim=1)
    dices = []
    for class_idx in range(1, num_classes):
        pred_mask = preds == class_idx
        label_mask = labels == class_idx
        denominator = pred_mask.sum() + label_mask.sum()
        if denominator == 0:
            continue
        intersection = (pred_mask & label_mask).sum()
        dices.append((2.0 * intersection.float() + eps) / (denominator.float() + eps))

    if not dices:
        return float("nan")
    return torch.stack(dices).mean().item()
