from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from mamba_ssm import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

try:
    from causal_conv1d import causal_conv1d_fn
except Exception:  # pragma: no cover - optional dependency path
    causal_conv1d_fn = None

from .unet3d import DoubleConv, NUM_CLASSES


class MambaBottleneckBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        debug_shapes: bool = False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.mamba = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)
        self.debug_shapes = debug_shapes

    def _print_shape(self, name: str, x: torch.Tensor) -> None:
        if self.debug_shapes:
            print(f"[model] {name}: {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected 5D tensor [B, C, D, H, W], got {tuple(x.shape)}")

        batch, channels, depth, height, width = x.shape
        residual = x.flatten(2).transpose(1, 2)
        seq = self.norm(residual)
        self._print_shape("mamba_seq_in", seq)

        if seq.is_cuda and self.mamba.use_fast_path:
            seq = self.mamba(seq)
        else:
            seq_len = seq.shape[1]
            mamba = self.mamba
            xz = rearrange(
                mamba.in_proj.weight @ rearrange(seq, "b l d -> d (b l)"),
                "d (b l) -> b d l",
                l=seq_len,
            )
            if mamba.in_proj.bias is not None:
                xz = xz + rearrange(mamba.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

            A = -torch.exp(mamba.A_log.float())
            x, z = xz.chunk(2, dim=1)
            if causal_conv1d_fn is None or not seq.is_cuda:
                x = mamba.act(mamba.conv1d(x)[..., :seq_len])
            else:
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(mamba.conv1d.weight, "d 1 w -> d w"),
                    bias=mamba.conv1d.bias,
                    activation=mamba.activation,
                )

            x_dbl = mamba.x_proj(rearrange(x, "b d l -> (b l) d"))
            dt, B, C = torch.split(x_dbl, [mamba.dt_rank, mamba.d_state, mamba.d_state], dim=-1)
            dt = mamba.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seq_len)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seq_len).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seq_len).contiguous()
            seq = selective_scan_ref(
                x,
                dt,
                A,
                B,
                C,
                mamba.D.float(),
                z=z,
                delta_bias=mamba.dt_proj.bias.float(),
                delta_softplus=True,
            )
            seq = rearrange(seq, "b d l -> b l d")
            seq = mamba.out_proj(seq)

        self._print_shape("mamba_seq_out", seq)
        seq = seq + residual
        out = seq.transpose(1, 2).reshape(batch, channels, depth, height, width)
        self._print_shape("mamba_bottleneck_out", out)
        return out


class MambaBottleneckUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = NUM_CLASSES,
        base_channels: int = 8,
        debug_shapes: bool = False,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
    ):
        super().__init__()
        c = base_channels
        self.debug_shapes = debug_shapes
        self.mamba_d_state = int(mamba_d_state)
        self.mamba_d_conv = int(mamba_d_conv)
        self.mamba_expand = int(mamba_expand)

        self.enc1 = DoubleConv(in_channels, c)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv(c, c * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.cnn_bottleneck = DoubleConv(c * 2, c * 4)
        self.mamba_bottleneck = MambaBottleneckBlock(
            c * 4,
            d_state=self.mamba_d_state,
            d_conv=self.mamba_d_conv,
            expand=self.mamba_expand,
            debug_shapes=debug_shapes,
        )

        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c * 4, c * 2)
        self.up1 = nn.ConvTranspose3d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c * 2, c)

        self.out = nn.Conv3d(c, num_classes, kernel_size=1)

    def _print_shape(self, name: str, x: torch.Tensor) -> None:
        if self.debug_shapes:
            print(f"[model] {name}: {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._print_shape("input", x)
        e1 = self.enc1(x)
        self._print_shape("enc1", e1)
        e2 = self.enc2(self.pool1(e1))
        self._print_shape("enc2", e2)
        b = self.cnn_bottleneck(self.pool2(e2))
        self._print_shape("cnn_bottleneck", b)

        b = self.mamba_bottleneck(b)

        d2 = self.up2(b)
        self._print_shape("up2", d2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        self._print_shape("dec2", d2)

        d1 = self.up1(d2)
        self._print_shape("up1", d1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        self._print_shape("dec1", d1)

        logits = self.out(d1)
        self._print_shape("logits", logits)
        return logits
