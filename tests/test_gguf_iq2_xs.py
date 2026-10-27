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
