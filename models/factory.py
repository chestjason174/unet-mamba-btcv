from __future__ import annotations

from typing import Any

from .unet3d import NUM_CLASSES, SmallUNet3D
from .unet3d_mamba import MambaBottleneckUNet3D


DEFAULT_MODEL_TYPE = "unet3d"


def infer_model_type(checkpoint: dict[str, Any]) -> str:
    model_type = checkpoint.get("model_type")
    if model_type:
        return str(model_type)

    args = checkpoint.get("args", {})
    if isinstance(args, dict):
        model_type = args.get("model_type")
        if model_type:
            return str(model_type)
    return DEFAULT_MODEL_TYPE


def build_model(
    model_type: str = DEFAULT_MODEL_TYPE,
    *,
    in_channels: int = 1,
    num_classes: int = NUM_CLASSES,
    base_channels: int = 8,
    debug_shapes: bool = False,
):
    model_type = str(model_type or DEFAULT_MODEL_TYPE)
    if model_type == "unet3d":
        return SmallUNet3D(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            debug_shapes=debug_shapes,
        )
    if model_type == "unet3d_mamba":
        return MambaBottleneckUNet3D(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            debug_shapes=debug_shapes,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def count_parameters(model) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return int(total), int(trainable)
