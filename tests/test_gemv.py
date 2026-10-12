# SPDX-License-Identifier: Apache-2.0
"""Canonical W8 storage should permit small batches without a second weight copy."""

import pytest
import torch

from consumer_int8 import ops


@pytest.mark.parametrize("m", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("n,k", [(37, 528), (512, 5120)])
def test_w8a16_small_batch_matches_bf16_dequantized_linear(m, n, k):
    torch.manual_seed(134)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    scale = (torch.rand(n, device="cuda") * 0.002).bfloat16()
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    ops.channel_gemv(x, weight, scale, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.channel_gemv(x, weight, scale, out)
    x.mul_(0.75)
    graph.replay()
    # cuBLAS may truncate BF16 partial reductions for these small-M shapes.
    # Use a CPU FP64 dot product of the intended BF16 multiplicands as oracle.
    dequantized = (weight.cpu().float() * scale.cpu()[:, None].float()).bfloat16()
    reference = torch.nn.functional.linear(
        x.cpu().double(), dequantized.double()
    ).bfloat16()
    actual = out.cpu()
    torch.testing.assert_close(actual, reference, atol=0.004, rtol=0.008)
    error = (actual.float() - reference.float()).norm() / reference.float().norm()
    assert error.item() < 0.003
