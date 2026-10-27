# SPDX-License-Identifier: Apache-2.0
"""Actual IQ2_XS/MXFP4 expert bytes, replicated; complete local MoE chain race.

Includes FC1, clipped SwiGLU, weighted FC2 and route reduction. CPU route
preparation is excluded. This is not full-model serving or varied expert data.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import gguf
import numpy as np
import torch


def main(args):
    torch.cuda.set_device(args.device)
    torch.manual_seed(376)
    torch.ops.load_library(str(args.library.resolve()))
    ops = torch.ops.consumer_int8
    manifest = json.loads(args.manifest.read_text())
    hashes = {}

    def sample(name):
        item = next(s for s in manifest["samples"] if s["name"] == name)
        data = (args.manifest.parent / Path(item["path"]).name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
        assert item["experts"] == 1
        hashes[name] = item["sha256"]
        k, n = item["dims"][:2]
        iq = item["type"] == "IQ2_XS"
        raw = (
            np.frombuffer(data, dtype=np.uint8)
            .copy()
            .reshape(n, k // (256 if iq else 32), 74 if iq else 17)
        )
        dense = gguf.dequantize(
            raw.reshape(-1, raw.shape[-1]),
            getattr(gguf.GGMLQuantizationType, item["type"]),
        )
        return torch.from_numpy(raw).cuda(), torch.from_numpy(
            dense.reshape(n, k)
        ).cuda().bfloat16()

    gate, gate_dense = sample("blk.0.ffn_gate_exps.weight")
    up, up_dense = sample("blk.0.ffn_up_exps.weight")
    down, down_dense = sample("blk.0.ffn_down_exps.weight")
    raw13 = torch.cat((gate, up), 0)
    codes13 = raw13[..., 2:66].contiguous().view(torch.int16).flatten(1)
    bases13 = raw13[..., :2].contiguous().view(torch.float16).squeeze(-1)
    scales13 = raw13[..., 66:].contiguous().flatten(1)
    if bases13.shape[-1] % 2:
        bases13 = torch.nn.functional.pad(bases13, (0, 1))
    packed13 = tuple(
        v.unsqueeze(0).repeat(args.experts, 1, 1) for v in (codes13, bases13, scales13)
    )
    packed2 = tuple(
        v.unsqueeze(0).repeat(args.experts, 1, 1)
        for v in (down[..., 1:].contiguous().flatten(1), down[..., 0].contiguous())
    )
    count, h, intermediate, topk = (
        args.tokens,
        gate_dense.shape[1],
        gate_dense.shape[0],
        6,
    )
    if args.routing == "balanced":
        ids = np.arange(count * topk).reshape(count, topk) % (args.experts * 2)
    else:
        route_rng = np.random.default_rng(928)
        ids = np.stack(
            [
                route_rng.choice(args.experts * 2, topk, replace=False)
                for _ in range(count)
            ]
        )
    counts = [int((ids == e).sum()) for e in range(args.experts)]
    capacity = (max(counts) + 31) // 32 * 32
    routes = args.experts * capacity
    rng = np.random.default_rng(194)
    tw = rng.uniform(0.01, 0.3, (count, topk)).astype(np.float32)
    host_tokens = np.full(routes, -1, dtype=np.int32)
    host_weights = np.zeros(routes, dtype=np.float32)
    host_inverse = np.full((count, topk), -1, dtype=np.int32)
    for e in range(args.experts):
        for j, (token, slot) in enumerate(np.argwhere(ids == e)):
            row = e * capacity + j
            host_tokens[row] = token
            host_weights[row] = tw[token, slot]
            host_inverse[token, slot] = row
    tokens = torch.from_numpy(host_tokens).cuda()
    weights = torch.from_numpy(host_weights).cuda()
    inverse = torch.from_numpy(host_inverse).cuda()
    identity = torch.arange(routes, device="cuda", dtype=torch.int32)
    x = torch.randn(count, h, device="cuda", dtype=torch.bfloat16) * 0.1
    projected = torch.zeros(
        routes, 2 * intermediate, device="cuda", dtype=torch.bfloat16
    )
    hidden = torch.zeros(routes, intermediate, device="cuda", dtype=torch.bfloat16)
    contributions = torch.zeros(routes, h, device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(x)
    p = (x.float() @ torch.cat((gate_dense, up_dense), 0).float().T).bfloat16().float()
    gate_ref, up_ref = p.chunk(2, -1)
    a = (
        (torch.nn.functional.silu(gate_ref.clamp(max=10)) * up_ref.clamp(-10, 10))
        .bfloat16()
        .float()
    )
    c = a @ down_dense.float().T
    expected = torch.zeros_like(x, dtype=torch.float32)
    for slot in range(topk):
        local = torch.tensor(ids[:, slot] < args.experts, device="cuda")
        contribution = (
            (c * torch.tensor(tw[:, slot], device="cuda")[:, None]).bfloat16().float()
        )
        expected += torch.where(local[:, None], contribution, 0)
    expected = expected.bfloat16().float()
    graphs = {}
    task_buffers = {}
    errors = {}
    outputs = []
    for tile in (16, 32):
        tasks = torch.tensor(
            [
                (e, e * capacity + r)
                for e in range(args.experts)
                for r in range(0, counts[e], tile)
            ],
            device="cuda",
            dtype=torch.int32,
        )

        task_buffers[tile] = tasks  # Graph input storage must outlive every replay.

        def run(tile=tile, tasks=tasks):
            ops.gguf_iq2_xs_routed_mm(x, *packed13, tokens, tasks, projected, tile)
            ops.gguf_swiglu(projected, hidden, 10.0)
            ops.gguf_mxfp4_routed_weighted_mm(
                hidden, *packed2, identity, tasks, weights, contributions, tile
            )
            ops.gguf_route_reduce(contributions, inverse, output)

        for _ in range(3):
            run()
        relative = ((output.float() - expected).norm() / expected.norm()).item()
        assert torch.isfinite(output).all() and relative < 0.005
        errors[tile] = relative
        outputs.append(output.clone())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        graphs[tile] = graph
    assert torch.equal(*outputs)
    for tile, graph in graphs.items():
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, outputs[0]), (tile, "stale graph input storage")
    times = {tile: [] for tile in graphs}
    flush = torch.empty(64 << 20, device="cuda", dtype=torch.uint8)
    for trial in range(10):
        for tile in (16, 32) if trial % 2 else (32, 16):
            flush.zero_()
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record()
            graphs[tile].replay()
            end.record()
            end.synchronize()
            times[tile].append(begin.elapsed_time(end))
    for graph in graphs.values():
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, outputs[0])
    record = dict(
        checkpoint=manifest["repo"],
        revision=manifest["revision"],
        samples=hashes,
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
        routing=args.routing,
        experts=args.experts,
        tokens=count,
        topk=topk,
        rows_per_expert=counts,
        relative_l2=errors,
        milliseconds=times,
        medians_ms={tile: statistics.median(v) for tile, v in times.items()},
        device=str(torch.cuda.get_device_properties(args.device)),
        torch=torch.__version__,
        scope="Replicated experts; full local chain; prepared routing; no collectives",
    )
    with args.output.open("x") as report:
        report.write(json.dumps(record) + "\n")
    print(
        json.dumps(
            {
                key: record[key]
                for key in ("experts", "tokens", "relative_l2", "medians_ms")
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experts", type=int, default=192)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--routing", choices=["random", "balanced"], default="random")
    parser.add_argument("--device", type=int, default=1)
    main(parser.parse_args())
