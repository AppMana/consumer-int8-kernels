# SPDX-License-Identifier: Apache-2.0
"""Independent gguf NumPy oracle exercises all grid/sign codes and scale nibbles."""

import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def api():
    library = os.environ.get("GGUF_NATIVE_LIBRARY")
    if not library:
        pytest.skip("Set GGUF_NATIVE_LIBRARY to an explicitly built library")
    torch.cuda.set_device(int(os.environ.get("GGUF_TEST_DEVICE", "1")))
    torch.ops.load_library(library)
    # Exercise the product packing helper without loading unrelated extensions.
    package = types.ModuleType("_gguf_test_package")
    package._gguf = types.ModuleType("_gguf")
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(
        package.__name__ + ".gguf",
        Path(__file__).parents[1] / "src/consumer_int8/gguf.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("m,k", [(1, 256), (17, 768), (33, 2304), (128, 5120)])
@pytest.mark.parametrize("group_major", [False, True])
def test_iq2_xs_oracle_graph(api, m, k, group_major):
    gguf = pytest.importorskip("gguf")
    rng = np.random.default_rng(173)
    g, n = 3, 37
    raw = rng.integers(0, 256, (g, n, k // 256, 74), dtype=np.uint8)
    bases = rng.uniform(-0.01, 0.01, (g, n, k // 256, 1)).astype(np.float16)
    raw[..., :2] = bases.view(np.uint8)
    dense = gguf.dequantize(raw.reshape(-1, 74), gguf.GGMLQuantizationType.IQ2_XS)
    dense = torch.from_numpy(dense.reshape(g, n, k)).cuda().bfloat16()
    codes, bases, scales = api.pack_iq2_xs(torch.from_numpy(raw).cuda())
    shape = (g, m, k) if group_major else (m, g, k)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    if group_major:
        x = x.transpose(0, 1)
    out = torch.empty(m, g, n, device="cuda", dtype=torch.bfloat16)
    api.iq2_xs_mm(x, codes, bases, scales, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        api.iq2_xs_mm(x, codes, bases, scales, out)
    for _ in range(3):
        x.mul_(0.75)
        out.fill_(float("nan"))
        before = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == before
        ref = torch.bmm(x.transpose(0, 1).float(), dense.transpose(1, 2).float())
        ref = ref.transpose(0, 1).bfloat16().float()
        assert torch.isfinite(out).all()
        assert (out.float() - ref).norm() / ref.norm() < 0.001


@pytest.mark.parametrize("k", [768, 2304, 5120])
@pytest.mark.parametrize("tile_rows", [16, 32])
@pytest.mark.parametrize("quant", ["IQ2_XS", "MXFP4"])
def test_routed_tasks_change_during_graph(api, k, tile_rows, quant):
    gguf = pytest.importorskip("gguf")
    rng = np.random.default_rng(971)
    g, n = 3, 37
    if quant == "IQ2_XS":
        raw = rng.integers(0, 256, (g, n, k // 256, 74), dtype=np.uint8)
        raw[..., :2] = (
            rng.uniform(-0.01, 0.01, (g, n, k // 256, 1))
            .astype(np.float16)
            .view(np.uint8)
        )
        packed = api.pack_iq2_xs(torch.from_numpy(raw).cuda())
        op = api.iq2_xs_routed_mm
    else:
        raw = rng.integers(0, 256, (g, n, k // 32, 17), dtype=np.uint8)
        raw[..., 0] = rng.integers(120, 130, (g, n, k // 32), dtype=np.uint8)
        packed = api.pack_mxfp4(torch.from_numpy(raw).cuda())
        op = api.mxfp4_routed_mm
    dense = gguf.dequantize(
        raw.reshape(-1, raw.shape[-1]), getattr(gguf.GGMLQuantizationType, quant)
    )
    dense = torch.from_numpy(dense.reshape(g, n, k)).cuda().bfloat16()
    x = torch.randn(29, k, device="cuda", dtype=torch.bfloat16)
    tokens = torch.full((3 * tile_rows + 1,), -1, device="cuda", dtype=torch.int32)
    tokens[:3] = torch.tensor([4, 4, 23], device="cuda", dtype=torch.int32)
    tokens[tile_rows : tile_rows + 19] = torch.arange(
        19, device="cuda", dtype=torch.int32
    )
    tasks = torch.tensor(
        [[1, 0], [2, tile_rows], [2, 2 * tile_rows], [-1, 0]],
        device="cuda",
        dtype=torch.int32,
    )
    out = torch.empty(3 * tile_rows + 1, n, device="cuda", dtype=torch.bfloat16)

    def run():
        op(x, *packed, tokens, tasks, out, tile_rows)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for trial in range(3):
        x.mul_(0.75)
        tasks[0, 0] = trial
        tokens[1] = 10 + trial
        out.fill_(float("nan"))
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        reference = torch.zeros(3 * tile_rows, n, device="cuda")
        for expert, begin in tasks.cpu().tolist():
            if expert < 0:
                continue
            ids = tokens[begin : begin + tile_rows].long()
            valid = ids >= 0
            values = x[ids.clamp_min(0)].float() @ dense[expert].float().T
            reference[begin : begin + tile_rows] = torch.where(
                valid[:, None], values, 0
            )
        reference = reference.bfloat16().float()
        assert torch.isfinite(out[: 3 * tile_rows]).all()
        assert (
            out[: 3 * tile_rows].float() - reference
        ).norm() / reference.norm() < 0.001
        assert torch.isnan(out[3 * tile_rows]).all()  # Not covered by any active task.


@pytest.mark.parametrize("group_major", [False, True])
def test_q8_expand_grouped_blas_graph(api, group_major):
    g, m, n, k = 3, 127, 37, 128
    codes = torch.randint(-128, 128, (g, n, k), device="cuda", dtype=torch.int8)
    scales = torch.randn(g, n, k // 32, device="cuda", dtype=torch.float16) * 0.1
    workspace = torch.empty(g, n, k, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(
        (g, m, k) if group_major else (m, g, k), device="cuda", dtype=torch.bfloat16
    )
    if group_major:
        x = x.transpose(0, 1)
    out = torch.empty(m, g, n, device="cuda", dtype=torch.bfloat16)
    api.q8_expand_mm(x, codes, scales, workspace, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        api.q8_expand_mm(x, codes, scales, workspace, out)
    for _ in range(3):
        x.mul_(0.75)
        scales.mul_(0.5)
        workspace.fill_(float("nan"))
        out.fill_(float("nan"))
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        dense = codes.view(g, n, k // 32, 32).float() * scales.float()[..., None]
        dense = dense.reshape(g, n, k).bfloat16().float()
        ref = torch.bmm(x.transpose(0, 1).float(), dense.transpose(1, 2))
        ref = ref.transpose(0, 1).bfloat16().float()
        assert torch.isfinite(out).all()
        assert (out.float() - ref).norm() / ref.norm() < 0.001


@pytest.mark.parametrize("tile_rows", [16, 32])
def test_mixed_iq_mx_chain_graph(api, tile_rows):
    gguf = pytest.importorskip("gguf")
    rng = np.random.default_rng(374)
    experts, h, intermediate, count, topk, capacity = 3, 256, 128, 19, 2, 32
    first = rng.integers(
        0, 256, (experts, 2 * intermediate, h // 256, 74), dtype=np.uint8
    )
    first[..., :2] = np.full((*first.shape[:-1], 1), 0.01, dtype=np.float16).view(
        np.uint8
    )
    second = rng.integers(0, 256, (experts, h, intermediate // 32, 17), dtype=np.uint8)
    second[..., 0] = rng.integers(120, 125, second.shape[:-1], dtype=np.uint8)
    w13 = (
        torch.from_numpy(
            gguf.dequantize(
                first.reshape(-1, 74), gguf.GGMLQuantizationType.IQ2_XS
            ).reshape(experts, 2 * intermediate, h)
        )
        .cuda()
        .bfloat16()
    )
    w2 = (
        torch.from_numpy(
            gguf.dequantize(
                second.reshape(-1, 17), gguf.GGMLQuantizationType.MXFP4
            ).reshape(experts, h, intermediate)
        )
        .cuda()
        .bfloat16()
    )
    packed13 = api.pack_iq2_xs(torch.from_numpy(first).cuda())
    packed2 = api.pack_mxfp4(torch.from_numpy(second).cuda())
    routes = experts * capacity
    x = torch.randn(count, h, device="cuda", dtype=torch.bfloat16) * 20
    tokens = torch.full((routes,), -1, device="cuda", dtype=torch.int32)
    identity = torch.arange(routes, device="cuda", dtype=torch.int32)
    weights = torch.zeros(routes, device="cuda", dtype=torch.float32)
    inverse = torch.full((count, topk), -1, device="cuda", dtype=torch.int32)
    tasks = torch.tensor(
        [
            (e, e * capacity + r)
            for e in range(experts)
            for r in range(0, capacity, tile_rows)
        ],
        device="cuda",
        dtype=torch.int32,
    )
    projected = torch.empty(
        routes, 2 * intermediate, device="cuda", dtype=torch.bfloat16
    )
    hidden = torch.empty(routes, intermediate, device="cuda", dtype=torch.bfloat16)
    contributions = torch.empty(routes, h, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(count, h, device="cuda", dtype=torch.bfloat16)

    def run():
        api.mixed_iq2_xs_mxfp4(
            x,
            packed13,
            packed2,
            tokens,
            identity,
            tasks,
            weights,
            inverse,
            projected,
            hidden,
            contributions,
            output,
            tile_rows=tile_rows,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for trial in range(3):
        # Expert3 is remote: those contributions must be absent on this rank.
        ids = (np.arange(count * topk).reshape(count, topk) + trial) % 4
        tw = rng.uniform(0.05, 0.95, (count, topk)).astype(np.float32)
        host_tokens = np.full(routes, -1, dtype=np.int32)
        host_weights = np.zeros(routes, dtype=np.float32)
        host_inverse = np.full((count, topk), -1, dtype=np.int32)
        for expert in range(experts):
            positions = np.argwhere(ids == expert)
            for j, (token, slot) in enumerate(positions):
                row = expert * capacity + j
                host_tokens[row] = token
                host_weights[row] = tw[token, slot]
                host_inverse[token, slot] = row
        tokens.copy_(torch.from_numpy(host_tokens))
        weights.copy_(torch.from_numpy(host_weights))
        inverse.copy_(torch.from_numpy(host_inverse))
        x.mul_(0.8)
        for tensor in (projected, hidden, contributions, output):
            tensor.fill_(float("nan"))
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        expected = torch.zeros(count, h, device="cuda", dtype=torch.float32)
        for expert in range(experts):
            rows, slots = np.where(ids == expert)
            if not len(rows):
                continue
            rows_gpu = torch.tensor(rows, device="cuda")
            p = (x[rows_gpu].float() @ w13[expert].float().T).bfloat16().float()
            gate, up = p.chunk(2, -1)
            gate = gate.clamp(max=10)
            up = up.clamp(-10, 10)
            a = (torch.nn.functional.silu(gate) * up).bfloat16().float()
            c = a @ w2[expert].float().T
            c *= torch.tensor(tw[rows, slots], device="cuda")[:, None]
            expected.index_add_(0, rows_gpu, c.bfloat16().float())
        expected = expected.bfloat16().float()
        assert torch.isfinite(output).all()
        assert (output.float() - expected).norm() / expected.norm() < 0.005
        # Independently check the weight-before-round epilogue at its boundary.
        for expert in range(experts):
            sl = slice(expert * capacity, (expert + 1) * capacity)
            ref = (
                (hidden[sl].float() @ w2[expert].float().T) * weights[sl, None]
            ).bfloat16()
            assert (
                contributions[sl].float() - ref.float()
            ).norm() / ref.float().norm() < 0.001
