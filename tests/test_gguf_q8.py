# SPDX-License-Identifier: Apache-2.0
"""Raw GGUF layout, ragged/grouped strides, and allocation-free graph replay."""

import os

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def library():
    path = os.environ.get("GGUF_NATIVE_LIBRARY")
    if not path:
        pytest.skip("Set GGUF_NATIVE_LIBRARY to the explicitly built AOT library")
    torch.cuda.set_device(int(os.environ.get("GGUF_TEST_DEVICE", "1")))
    torch.ops.load_library(path)


@pytest.mark.parametrize("m", [1, 17, 32, 127, 128, 513])
@pytest.mark.parametrize("group_major", [False, True])
def test_raw_q8_grouped_oracle_and_graph(m, group_major):
    torch.manual_seed(37 + m)
    groups, n, k = 3, 37, 96
    scales = (torch.rand(groups, n, k // 32, 1) * 0.1).half()
    codes = torch.randint(-128, 128, (groups, n, k // 32, 32), dtype=torch.int8)
    raw = torch.cat((scales.view(torch.uint8), codes.view(torch.uint8)), dim=-1).cuda()
    dense = (scales.float() * codes.float()).reshape(groups, n, k).cuda().bfloat16()
    shape = (groups, m, k) if group_major else (m, groups, k)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    if group_major:
        x = x.transpose(0, 1)
    out = torch.empty(m, groups, n, device="cuda", dtype=torch.bfloat16)
    op = torch.ops.consumer_int8.gguf_q8_mm
    op(x, raw, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(x, raw, out)
    for _ in range(3):
        x.mul_(0.75)
        out.fill_(float("nan"))
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        reference = torch.bmm(x.transpose(0, 1).float(), dense.float().transpose(1, 2))
        reference = reference.transpose(0, 1).bfloat16().float()
        relative = (out.float() - reference).norm() / reference.norm()
        assert relative < 0.001
        assert torch.isfinite(out).all() and out.abs().max() > 0
