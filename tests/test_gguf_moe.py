# SPDX-License-Identifier: Apache-2.0
"""Streamed expert ownership and arena replacement through the serving API."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_gguf_iq2_xs import api as api


@pytest.fixture(scope="module")
def adapter(api):
    package = sys.modules[api.__package__]
    package.gguf = api
    sys.modules[api.__name__] = api
    spec = importlib.util.spec_from_file_location(
        api.__package__ + ".gguf_moe", Path(api.__file__).with_name("gguf_moe.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "first,second",
    [
        ("IQ2_XS", "MXFP4"),
        ("IQ2_XXS", "IQ2_XXS"),
        ("IQ2_XXS", "IQ3_XXS"),
        ("IQ2_S", "MXFP4"),
        ("IQ2_S", "IQ3_S"),
        ("IQ3_XXS", "IQ3_S"),
        ("IQ3_XXS", "IQ3_XXS"),
    ],
)
def test_streamed_owner_and_reused_arena_graph(adapter, first, second):
    gguf = pytest.importorskip("gguf")
    rng = np.random.default_rng(758)
    h, intermediate, experts = 256, 256, 3
    reads = []
    dense = {}

    def source(name, kind, n, k):
        block = {
            "IQ2_XS": 74,
            "IQ2_XXS": 66,
            "IQ3_XXS": 98,
            "IQ2_S": 82,
            "IQ3_S": 110,
            "MXFP4": 17,
        }[kind]
        raw = rng.integers(
            0,
            256,
            (experts, n, k // (32 if kind == "MXFP4" else 256), block),
            dtype=np.uint8,
        )
        if kind == "MXFP4":
            raw[..., 0] = rng.integers(120, 125, raw.shape[:-1], dtype=np.uint8)
        else:
            raw[..., :2] = np.full((*raw.shape[:-1], 1), 0.01, dtype=np.float16).view(
                np.uint8
            )
        decoded = gguf.dequantize(
            raw.reshape(-1, block), getattr(gguf.GGMLQuantizationType, kind)
        )
        dense[name] = (
            torch.from_numpy(decoded.reshape(experts, n, k)).cuda().bfloat16().float()
        )
        raw_tensor = torch.from_numpy(raw)

        def read(expert):
            reads.append((name, expert))
            return raw_tensor[expert].clone()

        return SimpleNamespace(
            experts=experts, rows=n, columns=k, quant_type=kind, read_expert=read
        )

    sources = (
        source("gate", first, intermediate, h),
        source("up", first, intermediate, h),
        source("down", second, h, intermediate),
    )
    prepared = adapter.prepare_experts(
        *sources, (2, 0), device=torch.cuda.current_device()
    )
    assert prepared.expert_map.cpu().tolist() == [1, -1, 0]
    assert reads == [(name, e) for e in (2, 0) for name in ("gate", "up", "down")]
    plan = adapter.ExecutionPlan(prepared, max_tokens=17, topk=2)
    scratch = plan.allocate_scratch()
    x = torch.randn(17, h, device="cuda", dtype=torch.bfloat16) * 10
    ids = torch.randint(0, 3, (17, 2), device="cuda", dtype=torch.int32)
    weights = torch.rand(17, 2, device="cuda") * 0.9
    out = torch.empty_like(x)
    plan.run(x, ids, weights, out, scratch)
    first_out = out.clone()
    # A manager may move its arena during warmup: the next call must use new views.
    fresh = plan.allocate_scratch()
    plan.run(x, ids, weights, out, fresh)
    assert torch.equal(out, first_out)
    del scratch
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.run(x, ids, weights, out, fresh)
    for _ in range(3):
        ids.add_(1).remainder_(3)
        weights.mul_(0.8)
        x.mul_(0.75)
        for tensor in fresh:
            tensor.fill_(float("nan") if tensor.is_floating_point() else 7123)
        out.fill_(float("nan"))
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        expected = torch.zeros_like(x, dtype=torch.float32)
        for expert in (2, 0):
            row, slot = (ids == expert).nonzero(as_tuple=True)
            gate = (
                (x[row].float() @ dense["gate"][expert].T)
                .bfloat16()
                .float()
                .clamp(max=10)
            )
            up = (
                (x[row].float() @ dense["up"][expert].T)
                .bfloat16()
                .float()
                .clamp(-10, 10)
            )
            hidden = (torch.nn.functional.silu(gate) * up).bfloat16().float()
            contribution = (
                ((hidden @ dense["down"][expert].T) * weights[row, slot, None])
                .bfloat16()
                .float()
            )
            expected.index_add_(0, row, contribution)
        expected = expected.bfloat16().float()
        assert torch.isfinite(out).all()
        assert (out.float() - expected).norm() / expected.norm() < 0.005
    for count in (3, 0, 11):
        plan.run(x[:count], ids[:count], weights[:count], out[:count], fresh)
        assert torch.isfinite(out[:count]).all()
    with pytest.raises(ValueError, match="exceeds"):
        plan.run(
            torch.empty(18, h, device="cuda", dtype=torch.bfloat16),
            ids,
            weights,
            out,
            fresh,
        )
