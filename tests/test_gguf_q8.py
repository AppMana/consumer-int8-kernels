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
@pytest.mark.parametrize("packed", [False, True])
def test_raw_q8_grouped_oracle_and_graph(m, group_major, packed):
    torch.manual_seed(37 + m)
    groups, n, k = 3, 37, 128 if packed else 96
    scales = (torch.rand(groups, n, k // 32, 1) * 0.1).half()
    codes = torch.randint(-128, 128, (groups, n, k // 32, 32), dtype=torch.int8)
    raw = torch.cat((scales.view(torch.uint8), codes.view(torch.uint8)), dim=-1).cuda()
    dense = (scales.float() * codes.float()).reshape(groups, n, k).cuda().bfloat16()
    shape = (groups, m, k) if group_major else (m, groups, k)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    if group_major:
        x = x.transpose(0, 1)
    out = torch.empty(m, groups, n, device="cuda", dtype=torch.bfloat16)
    packed_codes = codes.reshape(groups, n, k).cuda()
    packed_scales = scales.squeeze(-1).cuda()

    def run():
        if packed:
            torch.ops.consumer_int8.gguf_q8_packed_mm(
                x, packed_codes, packed_scales, out
            )
        else:
            torch.ops.consumer_int8.gguf_q8_mm(x, raw, out)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
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


@pytest.mark.parametrize("k", [32, 96, 2304, 5120])
def test_q8_expand_oracle_graph(k):
    torch.manual_seed(k)
    codes = torch.randint(-128, 128, (3, 37, k), device="cuda", dtype=torch.int8)
    scales = torch.randn(3, 37, k // 32, device="cuda", dtype=torch.float16) * 0.1
    out = torch.empty_like(codes, dtype=torch.bfloat16)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops.consumer_int8.gguf_q8_expand(codes, scales, out)
    for _ in range(3):
        scales.mul_(0.75)
        out.fill_(float("nan"))
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        ref = codes.view(3, 37, k // 32, 32).float() * scales.float()[..., None]
        assert torch.equal(out, ref.reshape_as(out).bfloat16())
