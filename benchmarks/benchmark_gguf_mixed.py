# SPDX-License-Identifier: Apache-2.0
"""Actual IQ2_XS/MXFP4 expert bytes, replicated; complete local MoE chain race.

Includes FC1, clipped SwiGLU, weighted FC2 and route reduction. --device-plan
includes GPU routing; otherwise CPU preparation is excluded. This is not
full-model serving or varied expert data.
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
    formats = {}

    def sample(name):
        item = next(s for s in manifest["samples"] if s["name"] == name)
        data = (args.manifest.parent / Path(item["path"]).name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
        assert item["experts"] == 1
        hashes[name] = item["sha256"]
        k, n = item["dims"][:2]
        formats[name] = item["type"]
        iq = item["type"] != "MXFP4"
        block_bytes = {"IQ2_XS": 74, "IQ2_XXS": 66, "IQ3_XXS": 98, "MXFP4": 17}[
            item["type"]
        ]
        raw = (
            np.frombuffer(data, dtype=np.uint8)
            .copy()
            .reshape(n, k // (256 if iq else 32), block_bytes)
        )
        dense = gguf.dequantize(
            raw.reshape(-1, raw.shape[-1]),
            getattr(gguf.GGMLQuantizationType, item["type"]),
        )
        return torch.from_numpy(raw).cuda(), torch.from_numpy(
            dense.reshape(n, k)
        ).cuda().bfloat16()

    gate_name = f"blk.{args.block}.ffn_gate_exps.weight"
    up_name = f"blk.{args.block}.ffn_up_exps.weight"
    down_name = f"blk.{args.block}.ffn_down_exps.weight"
    gate, gate_dense = sample(gate_name)
    up, up_dense = sample(up_name)
    down, down_dense = sample(down_name)
    first_format, second_format = formats[gate_name], formats[down_name]
    assert formats[up_name] == first_format

    def pack(raw, kind):
        if kind == "MXFP4":
            parts = (raw[..., 1:].contiguous().flatten(1), raw[..., 0].contiguous())
        else:
            bases = raw[..., :2].contiguous().view(torch.float16).squeeze(-1)
            if bases.shape[-1] % 2:
                bases = torch.nn.functional.pad(bases, (0, 1))
            if kind == "IQ2_XS":
                codes = raw[..., 2:66].contiguous().view(torch.int16).flatten(1)
                metadata = raw[..., 66:].contiguous().flatten(1)
            elif kind == "IQ2_XXS":
                chunks = raw[..., 2:].reshape(*raw.shape[:2], 8, 8)
                codes = chunks[..., :4].contiguous().flatten(1)
                metadata = chunks[..., 4:].contiguous().view(torch.int32).flatten(1)
            else:
                codes = raw[..., 2:66].contiguous().flatten(1)
                metadata = raw[..., 66:].contiguous().view(torch.int32).flatten(1)
            parts = codes, bases, metadata
        return tuple(v.unsqueeze(0).repeat(args.experts, 1, 1) for v in parts)

    packed13 = pack(torch.cat((gate, up), 0), first_format)
    packed2 = pack(down, second_format)
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
    if args.device_plan:
        routes = max(routes, count * topk + args.experts * 31)
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
    top_ids = torch.from_numpy(ids.astype(np.int32)).cuda()
    top_weights = torch.from_numpy(tw).cuda()
    expert_map = torch.cat(
        (
            torch.arange(args.experts, device="cuda", dtype=torch.int32),
            torch.full((args.experts,), -1, device="cuda", dtype=torch.int32),
        )
    )
    info = torch.empty(args.experts, 2, device="cuda", dtype=torch.int32)
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
    plan_graphs = {}
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

        if args.device_plan:
            required = count * topk + args.experts * (tile - 1)
            tasks = torch.empty(
                (required + tile - 1) // tile, 2, device="cuda", dtype=torch.int32
            )
        task_buffers[tile] = tasks  # Graph input storage must outlive every replay.

        def run(tile=tile, tasks=tasks):
            if args.device_plan:
                ops.gguf_plan_routes(
                    top_ids,
                    top_weights,
                    expert_map,
                    info,
                    tokens,
                    weights,
                    inverse,
                    tasks,
                    tile,
                )
            if first_format == "IQ2_XS":
                ops.gguf_iq2_xs_routed_mm(x, *packed13, tokens, tasks, projected, tile)
            else:
                ops.gguf_iq_xxs_routed_mm(
                    x,
                    *packed13,
                    tokens,
                    tasks,
                    weights,
                    projected,
                    int(first_format[2]),
                    tile,
                    False,
                )
            ops.gguf_swiglu(projected, hidden, 10.0)
            if second_format == "MXFP4":
                ops.gguf_mxfp4_routed_weighted_mm(
                    hidden, *packed2, identity, tasks, weights, contributions, tile
                )
            else:
                ops.gguf_iq_xxs_routed_mm(
                    hidden,
                    *packed2,
                    identity,
                    tasks,
                    weights,
                    contributions,
                    int(second_format[2]),
                    tile,
                    True,
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
        if args.device_plan:
            plan_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(plan_graph):
                ops.gguf_plan_routes(
                    top_ids,
                    top_weights,
                    expert_map,
                    info,
                    tokens,
                    weights,
                    inverse,
                    tasks,
                    tile,
                )
            plan_graphs[tile] = plan_graph
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
    plan_times = {tile: [] for tile in plan_graphs}
    for trial in range(10):
        for tile in list(plan_graphs)[:: 1 if trial % 2 else -1]:
            graph = plan_graphs[tile]
            flush.zero_()
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record()
            graph.replay()
            end.record()
            end.synchronize()
            plan_times[tile].append(begin.elapsed_time(end))
    record = dict(
        formats=formats,
        block=args.block,
        checkpoint=manifest["repo"],
        revision=manifest["revision"],
        samples=hashes,
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
        device_plan=args.device_plan,
        plan_medians_ms={tile: statistics.median(v) for tile, v in plan_times.items()},
        plan_milliseconds=plan_times,
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
        scope="Replicated experts; full local chain; no collectives",
    )
    with args.output.open("x") as report:
        report.write(json.dumps(record) + "\n")
    print(
        json.dumps(
            {
                key: record[key]
                for key in (
                    "experts",
                    "tokens",
                    "relative_l2",
                    "medians_ms",
                    "plan_medians_ms",
                )
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--experts", type=int, default=192)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--device-plan", action="store_true")
    parser.add_argument("--routing", choices=["random", "balanced"], default="random")
    parser.add_argument("--device", type=int, default=1)
    main(parser.parse_args())
