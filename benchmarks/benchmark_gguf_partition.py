# SPDX-License-Identifier: Apache-2.0
"""Race native CUDA and prepared b12x on actual, distinct local GGUF experts.

Streams the verified checkpoint; no replicated expert payloads. Synthetic routing,
no collectives, and sampled independent oracles make this a component benchmark.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import statistics
import sys
from functools import partial
from pathlib import Path

import gguf
import numpy as np
import torch
from benchmark_gguf_mixed import prepare_b12x


def hashed_source(source):
    # Preserve the loader's descriptor type required by GGUFExpertWeights.
    class TrackedSource(type(source)):
        def read_expert(self, expert):
            raw = super().read_expert(expert)
            digest = hashlib.sha256(memoryview(raw.numpy())).hexdigest()
            assert self.hashes.setdefault(expert, digest) == digest
            return raw

    result = TrackedSource(**vars(source))
    object.__setattr__(result, "hashes", {})
    return result


def make_routes(count, first, pattern):
    row = np.arange(count)[:, None]
    slot = np.arange(6)[None, :]
    if pattern == "balanced":
        return ((row * 6 + slot + first) % 384).astype(np.int32)
    if pattern == "legacy-balanced":
        return ((row * 6 + slot * 64 + first) % 384).astype(np.int32)
    if pattern != "random":
        raise ValueError("Unknown routing pattern")
    rng = np.random.default_rng(928)
    ids = np.stack([rng.choice(384, 6, replace=False) for _ in range(count)])
    if first not in ids[0]:
        ids[0, 0] = first
    return ids.astype(np.int32)


def reference(sources, x, ids, weights, owned):
    result = torch.zeros_like(x, dtype=torch.float32)
    for expert in ids.unique().tolist():
        if expert not in owned:
            continue
        rows, slots = torch.where(ids == expert)
        values = []
        for source in sources:
            raw = source.read_expert(expert).numpy().reshape(source.rows, -1)
            decoded = gguf.dequantize(
                raw, getattr(gguf.GGMLQuantizationType, source.quant_type)
            )
            w = torch.from_numpy(decoded).to(x.device, torch.bfloat16)
            if len(values) < 2:
                values.append((x[rows].float() @ w.float().T).bfloat16())
            else:
                gate, up = values
                mid = (
                    torch.nn.functional.silu(gate.float().clamp(max=10))
                    * up.float().clamp(-10, 10)
                ).bfloat16()
                contribution = (
                    (mid.float() @ w.float().T) * weights[rows, slots, None]
                ).bfloat16()
                result.index_add_(0, rows, contribution.float())
            del w
    return result.bfloat16().float()


def check(actual, expected):
    assert torch.isfinite(actual).all() and expected.norm() > 0
    rel = ((actual.float() - expected).norm() / expected.norm()).item()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.flatten(), dim=0
    ).item()
    assert rel < 0.02 and cosine > 0.9998, (rel, cosine)
    return dict(relative_l2=rel, cosine=cosine)


def main(args):
    import b12x
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.loader.gguf_experts import index_experts

    import consumer_int8

    name = "consumer_int8._gguf"
    spec = importlib.util.spec_from_file_location(name, args.library.resolve())
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    consumer_int8._gguf = module
    from consumer_int8.gguf_moe import ExecutionPlan, prepare_experts

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = args.checkpoint / "checkpoint-verified.json"
    records = json.loads(manifest.read_text())
    assert len(records) == 7
    paths = []
    for r in records:
        assert r["revision"] == "60ad3651641b5c97679c807e064ab1d7a3ca1455"
        path = args.checkpoint / r["file"]
        assert path.stat().st_size == r["bytes"]
        assert (
            json.loads((args.checkpoint / (path.name + ".verified.json")).read_text())
            == r
        )
        paths.append(path)
    indexed = index_experts(paths)
    first = args.rank * 192
    owned = tuple(range(first, first + args.experts))
    b12x_root = Path(b12x.__file__).parent
    native_root = Path(consumer_int8.__file__).parent
    identity = dict(
        b12x_source_sha256=hashlib.sha256(
            b"".join(
                str(p.relative_to(b12x_root)).encode() + p.read_bytes()
                for p in sorted(b12x_root.rglob("*.py"))
            )
        ).hexdigest(),
        native_adapter_sha256=hashlib.sha256(
            b"".join(
                (native_root / name).read_bytes() for name in ("gguf.py", "gguf_moe.py")
            )
        ).hexdigest(),
        shared_benchmark_sha256=hashlib.sha256(
            Path(__file__).with_name("benchmark_gguf_mixed.py").read_bytes()
        ).hexdigest(),
        command=sys.argv,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch=torch.__version__,
        device=str(torch.cuda.get_device_properties(device)),
        rank=args.rank,
        local_experts=args.experts,
        scope="Distinct resident checkpoint experts; synthetic routing; no collectives",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as report:
        for layer in args.layers:
            sources = tuple(
                hashed_source(indexed[f"blk.{layer}.ffn_{part}_exps.weight"])
                for part in ("gate", "up", "down")
            )
            gate, up, down = sources
            assert gate.experts == 384 and gate.columns == 5120 and gate.rows == 2304
            print(
                json.dumps(dict(stage="load", layer=layer, experts=args.experts)),
                flush=True,
            )
            native = prepare_experts(*sources, owned, device=device)
            for count in args.tokens:
                torch.manual_seed(182 + count)
                x = torch.randn(
                    count, gate.columns, device=device, dtype=torch.bfloat16
                )
                ids = torch.full((count, 6), first, device=device, dtype=torch.int32)
                weights = torch.softmax(torch.randn(count, 6, device=device), -1)
                calls = {}
                owners = []
                for tile in (16, 32):
                    plan = ExecutionPlan(native, count, 6, tile_rows=tile)
                    scratch = plan.allocate_scratch()
                    out = torch.empty_like(x)
                    calls[f"native{tile}"] = (
                        partial(plan.run, x, ids, weights, out, scratch),
                        out,
                    )
                    owners.append((plan, scratch))
                b12x = prepare_b12x(
                    *sources,
                    gate.quant_type,
                    down.quant_type,
                    x,
                    ids,
                    weights,
                    native.expert_map,
                    args.experts,
                    expert_ids=owned,
                )
                calls["b12x"] = (b12x.run, b12x.output)
                owners.append(b12x)
                for pattern in args.patterns:
                    host_ids = make_routes(count, first, pattern)
                    ids.copy_(torch.from_numpy(host_ids))
                    local = (
                        host_ids[
                            (host_ids >= first) & (host_ids < first + args.experts)
                        ]
                        - first
                    )
                    route_counts = np.bincount(local, minlength=args.experts)
                    graphs = {}
                    with kernel_resolution_guard("paired GGUF partition benchmark"):
                        for label, (run, _out) in calls.items():
                            for _ in range(3):
                                run()
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                run()
                            graphs[label] = graph
                        x.mul_(0.75)
                        weights.mul_(0.8)
                        samples = torch.tensor(
                            sorted({0, count // 2, count - 1}), device=device
                        )
                        expected = reference(
                            sources, x[samples], ids[samples], weights[samples], owned
                        )
                        references, correctness = {}, {}
                        for label, graph in graphs.items():
                            out = calls[label][1]
                            out.fill_(float("nan"))
                            allocated = torch.cuda.memory_stats()[
                                "allocation.all.allocated"
                            ]
                            graph.replay()
                            torch.cuda.synchronize()
                            assert (
                                torch.cuda.memory_stats()["allocation.all.allocated"]
                                == allocated
                            )
                            assert torch.isfinite(out).all()
                            correctness[label] = check(out[samples], expected)
                            references[label] = out.clone()
                        flush = torch.empty(64 << 20, device=device, dtype=torch.uint8)
                        timings = {label: [] for label in graphs}
                        for trial in range(args.trials):
                            for label in list(graphs)[:: 1 if trial % 2 else -1]:
                                flush.zero_()
                                begin, end = (
                                    torch.cuda.Event(enable_timing=True),
                                    torch.cuda.Event(enable_timing=True),
                                )
                                begin.record()
                                graphs[label].replay()
                                end.record()
                                end.synchronize()
                                timings[label].append(begin.elapsed_time(end))
                        for label, graph in graphs.items():
                            calls[label][1].fill_(float("nan"))
                            graph.replay()
                            torch.cuda.synchronize()
                            assert torch.equal(calls[label][1], references[label])
                    record = dict(
                        identity,
                        layer=layer,
                        tokens=count,
                        pattern=pattern,
                        formats=[gate.quant_type, down.quant_type],
                        local_active_experts=int(np.count_nonzero(route_counts)),
                        local_routes=int(route_counts.sum()),
                        routes_per_expert=route_counts.tolist(),
                        routes_sha256=hashlib.sha256(memoryview(host_ids)).hexdigest(),
                        correctness=correctness,
                        milliseconds=timings,
                        medians_ms={
                            k: statistics.median(v) for k, v in timings.items()
                        },
                        expert_hashes={s.name: s.hashes for s in sources},
                        allocated_bytes=torch.cuda.memory_allocated(),
                    )
                    report.write(json.dumps(record) + "\n")
                    report.flush()
                    print(
                        json.dumps(
                            {
                                k: record[k]
                                for k in (
                                    "layer",
                                    "tokens",
                                    "pattern",
                                    "medians_ms",
                                    "correctness",
                                )
                            }
                        ),
                        flush=True,
                    )
                    for graph in graphs.values():
                        graph.reset()
                    del graphs, references, flush
                del calls, owners, b12x, plan, scratch, out, run, graph
                gc.collect()
                torch.cuda.empty_cache()
            del native
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[0, 2, 5, 15, 20, 25, 39]
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=[512])
    parser.add_argument(
        "--patterns",
        choices=["balanced", "legacy-balanced", "random"],
        nargs="+",
        default=["balanced", "random"],
    )
    parser.add_argument("--rank", type=int, choices=[0, 1], default=0)
    parser.add_argument("--experts", type=int, default=192)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.experts <= 192 or min(*args.tokens, args.trials) < 1:
        parser.error("Invalid expert count, token count or trials")
    main(args)
