# SPDX-License-Identifier: Apache-2.0
"""Guard the published checkpoint encoding, especially zero rows and K tails."""

import pytest
import torch

from consumer_int8.checkpoint import encoding_for
from consumer_int8.quantization import (
    dequantize_channel,
    dequantize_group32,
    quantize_channel,
    quantize_group32,
)


def test_channel_codes_match_dsv4_bias_and_scale_convention():
    weight = torch.tensor([[0.0, -127.0, 63.5, 127.0], [0.0, 0.0, 0.0, 0.0]])
    codes, scale = quantize_channel(weight)
    assert codes.dtype == torch.uint8
    assert scale.dtype == torch.bfloat16
    assert codes.tolist() == [[128, 1, 192, 255], [128, 128, 128, 128]]
    assert scale[0] == 1
    assert torch.isfinite(dequantize_channel(codes, scale)).all()
    torch.testing.assert_close(dequantize_channel(codes, scale)[1], weight[1])


@pytest.mark.parametrize("k", [32, 72, 128, 4304])
def test_group32_roundtrip_keeps_unaligned_vision_tail(k):
    weight = torch.randn(7, k, generator=torch.Generator().manual_seed(4))
    weight[0] = 0
    codes, scale = quantize_group32(weight)
    restored = dequantize_group32(codes, scale)
    assert codes.shape == weight.shape
    assert scale.shape == (7, (k + 31) // 32)
    bound = scale.repeat_interleave(32, dim=1)[:, :k] / 2 + 1e-6
    assert ((restored - weight).abs() <= bound).all()


@pytest.mark.parametrize("quantize", [quantize_channel, quantize_group32])
def test_nonfinite_weights_are_rejected(quantize):
    with pytest.raises(ValueError, match="non-finite"):
        quantize(torch.tensor([[float("nan")]]))


@pytest.mark.parametrize(
    "name,expected",
    [
        (
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            "channel_uint8",
        ),
        ("model.language_model.layers.0.linear_attn.in_proj_a.weight", "passthrough"),
        ("model.language_model.layers.0.linear_attn.conv1d.weight", "passthrough"),
        ("model.language_model.embed_tokens.weight", "passthrough"),
        ("lm_head.weight", "passthrough"),
        ("mtp.fc.weight", "channel_uint8"),
        ("mtp.layers.0.mlp.gate_proj.weight", "channel_uint8"),
        ("model.visual.blocks.0.mlp.linear_fc2.weight", "group32_int8"),
        ("model.visual.patch_embed.proj.weight", "passthrough"),
    ],
)
def test_encoding_preserves_gates_embeddings_and_convolutions(name, expected):
    assert encoding_for(name) == expected
