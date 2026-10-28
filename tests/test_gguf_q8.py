# SPDX-License-Identifier: Apache-2.0
"""Raw GGUF layout, ragged/grouped strides, and allocation-free graph replay."""

import os

import pytest
import torch
from test_gguf_iq2_xs import api as api


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


@pytest.mark.parametrize("k", [256, 768, 1280])
@pytest.mark.parametrize(
    "m,tile_rows,group_major", [(1, 32, False), (37, 64, True), (129, 64, False)]
)
def test_q6_raw_oracle_and_graph(api, k, m, tile_rows, group_major):
    """Catch Q6 split-nibble/sign/scale ordering and scratch reuse errors."""
    import gguf
    import numpy as np

    rng = np.random.default_rng(k + m)
    groups, n = 2, 37
    raw_np = rng.integers(0, 256, (groups, n, k // 256, 210), dtype=np.uint8)
    bases = (rng.standard_normal((groups, n, k // 256, 1)) * 0.01).astype("<f2")
    raw_np[..., 208:] = bases.view(np.uint8)
    raw = torch.from_numpy(raw_np).cuda()
    decoded = gguf.dequantize(raw_np.reshape(-1, 210), gguf.GGMLQuantizationType.Q6_K)
    dense = torch.from_numpy(decoded.reshape(groups, n, k)).cuda().bfloat16()
    expanded = torch.empty_like(dense)
    x = torch.randn(
        (groups, m, k) if group_major else (m, groups, k),
        dtype=torch.bfloat16,
        device="cuda",
    )
    if group_major:
        x = x.transpose(0, 1)
    out = torch.empty(m, groups, n, dtype=torch.bfloat16, device="cuda")
    expanded_out = torch.empty_like(out)
    api.q6_expand_mm(x, raw, expanded, expanded_out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        api.q6_expand_mm(x, raw, expanded, expanded_out)
        torch.ops.consumer_int8.gguf_q6_mm(x, raw, out, tile_rows)
    for _ in range(3):
        x.mul_(0.75)
        expanded.fill_(float("nan"))
        out.fill_(float("nan"))
        expanded_out.fill_(float("nan"))
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        assert torch.equal(expanded, dense)
        ref = torch.bmm(x.transpose(0, 1).float(), dense.float().transpose(1, 2))
        ref = ref.transpose(0, 1).bfloat16().float()
        assert torch.isfinite(out).all()
        assert (out.float() - ref).norm() / ref.norm() < 0.001
        assert torch.isfinite(expanded_out).all()
        assert (expanded_out.float() - ref).norm() / ref.norm() < 0.001


@pytest.mark.parametrize("n,k", [(8, 256), (40, 768), (128, 1280), (136, 512)])
def test_q6_b12x_packed_expansion_oracle_graph(api, n, k):
    """Consume the real public b12x packer's storage without a second weight copy."""
    import gguf
    import numpy as np
    pack_weight = pytest.importorskip("b12x.gemm.blockscaled").pack_weight

    rng = np.random.default_rng(n + k)
    raw_np = rng.integers(0, 256, (n, k // 256, 210), dtype=np.uint8)
    raw_np[..., 208:] = (
        (rng.standard_normal((n, k // 256, 1)) * 0.01).astype("<f2").view(np.uint8)
    )
    raw = torch.from_numpy(raw_np).cuda()
    weight = pack_weight(raw, recipe="q6_k")
    decoded = gguf.dequantize(raw_np.reshape(-1, 210), gguf.GGMLQuantizationType.Q6_K)
    reference = torch.from_numpy(decoded.reshape(n, k)).cuda().bfloat16()
    out = torch.empty_like(reference)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        api.q6_packed_expand(weight.values, weight.metadata, out)
    for _ in range(3):
        out.fill_(float("nan"))
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        assert torch.equal(out, reference)
    # Rebind the same storage to changed, independently decoded checkpoint bytes.
    raw_np[..., :192] ^= 0xFF
    replacement = pack_weight(torch.from_numpy(raw_np).cuda(), recipe="q6_k")
    weight.values.copy_(replacement.values)
    weight.metadata.copy_(replacement.metadata)
    decoded = gguf.dequantize(raw_np.reshape(-1, 210), gguf.GGMLQuantizationType.Q6_K)
    reference = torch.from_numpy(decoded.reshape(n, k)).cuda().bfloat16()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, reference)
