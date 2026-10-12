# SPDX-License-Identifier: Apache-2.0
"""Checkpoint encodings shared with the AppMana DeepSeek V4 checkpoints.

Channel UINT8 stores signed codes plus 128 and BF16 per-output scales.
Vision group INT8 stores signed codes with FP32 scales over 32 input channels.
Neither encoding implies integer activation arithmetic at runtime.
"""

import torch
import torch.nn.functional as F


def _matrix(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2 or not weight.is_floating_point() or not weight.numel():
        raise ValueError("Expected a nonempty floating-point weight matrix")
    value = weight.float()
    if not torch.isfinite(value).all():
        raise ValueError("Weight contains non-finite values")
    return value


def quantize_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode [N, K] as biased UINT8 and [N] BF16 scales, without packing."""
    value = _matrix(weight)
    scale = value.abs().amax(dim=1).clamp_min(torch.finfo(torch.float32).tiny)
    scale = scale / 127.0
    codes = (value / scale[:, None]).round().clamp(-128, 127)
    # Match DSV4's quantize-then-cast convention exactly.
    stored_scale = scale.to(torch.bfloat16)
    if not torch.isfinite(stored_scale).all() or (stored_scale == 0).any():
        raise ValueError("Channel scales are not representable as nonzero BF16")
    return (codes.to(torch.int16) + 128).to(torch.uint8), stored_scale


def quantize_group32(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode [N, K] as signed INT8 and [N, ceil(K/32)] FP32 scales."""
    value = _matrix(weight)
    n, k = value.shape
    grouped = F.pad(value, (0, (-k) % 32)).reshape(n, -1, 32)
    scale = grouped.abs().amax(dim=2) / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = (grouped / scale[..., None]).round().clamp(-127, 127).to(torch.int8)
    return codes.reshape(n, -1)[:, :k].contiguous(), scale


def dequantize_channel(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (codes.float() - 128) * scale.float()[:, None]


def dequantize_group32(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return codes.float() * scale.repeat_interleave(32, dim=1)[:, : codes.shape[1]]
