# SPDX-License-Identifier: Apache-2.0
"""Validate and time routed IQ2_XXS INT8 projections, excluding route planning.

--checkpoint uses distinct verified local experts. --manifest is a small
correctness fixture grouping distinct tensor slices as synthetic experts.
This measures one projection, not the complete MoE or full-model serving.
"""

import argparse
import ctypes
import hashlib
import json
import statistics
from pathlib import Path

import gguf
import numpy as np
import torch


def load_weights(args):
    identity = []
    if args.checkpoint:
        from b12x.loader.gguf_experts import index_experts

        records = json.loads((args.checkpoint / "checkpoint-verified.json").read_text())
        assert len(records) == 7
        paths = []
        for r in records:
            assert r["revision"] == "60ad3651641b5c97679c807e064ab1d7a3ca1455"
            path = args.checkpoint / r["file"]
            assert path.stat().st_size == r["bytes"]
            assert (
                json.loads(
                    (args.checkpoint / (path.name + ".verified.json")).read_text()
                )
                == r
            )
            paths.append(path)
        source = index_experts(paths)[args.tensor]
        assert source.quant_type == "IQ2_XXS" and source.experts == 384
        raw = []
        for expert in range(args.first_expert, args.first_expert + args.experts):
            value = source.read_expert(expert).numpy()
            identity.append(
                dict(expert=expert, sha256=hashlib.sha256(value.tobytes()).hexdigest())
            )
            raw.append(value)
        return np.stack(raw), identity
    manifest = json.loads(args.manifest.read_text())
    assert manifest["revision"] == "60ad3651641b5c97679c807e064ab1d7a3ca1455"
    items = [
        s
        for s in manifest["samples"]
        if s["type"] == "IQ2_XXS" and s["dims"][:2] == [5120, 2304]
    ]
    raw = []
    for item in items:
        data = (args.manifest.parent / Path(item["path"]).name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
        raw.append(np.frombuffer(data, np.uint8).reshape(2304, 20, 66).copy())
        identity.append(dict(tensor=item["name"], sha256=item["sha256"]))
    assert len(raw) > 1
    return np.stack(raw), identity


def main(args):
    torch.cuda.set_device(args.device)
    torch.manual_seed(189)
    torch.backends.cuda.matmul.allow_tf32 = False
    raw_np, identity = load_weights(args)
    if args.limit_n:
        raw_np = raw_np[:, : args.limit_n].copy()
    experts, n, blocks, _ = raw_np.shape
    k = blocks * 256
    raw = torch.from_numpy(raw_np).cuda()
    if args.native_library:
        torch.ops.load_library(str(args.native_library.resolve()))
        bases = raw[..., :2].contiguous().view(torch.float16).squeeze(-1)
        if blocks % 2:
            bases = torch.nn.functional.pad(bases, (0, 1))
        chunks = raw[..., 2:].reshape(*raw.shape[:3], 8, 8)
        codes = chunks[..., :4].contiguous().flatten(2)
        metadata = chunks[..., 4:].contiguous().view(torch.int32).flatten(2)
    lib = ctypes.CDLL(str(args.library.resolve()))
    call = lib.iq2xxs_routed_int8
    call.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 7 + [ctypes.c_void_p]
    call.restype = ctypes.c_int
    flush = torch.empty(64 << 20, device="cuda", dtype=torch.uint8)
    with args.output.open("x") as report:
        for count in args.tokens:
            x = torch.randn(count, k, device="cuda", dtype=torch.bfloat16) * 0.1
            x[:, :32] = 0
            a = torch.empty_like(x, dtype=torch.int8)
            sa = torch.empty(count, k // 32, device="cuda")
            # Balanced global top6 routes: local partition ownership is explicit.
            universe = 384 if args.checkpoint else experts
            ids = (np.arange(count)[:, None] * 6 + np.arange(6)[None]) % universe
            first = args.first_expert if args.checkpoint else 0
            slots = [np.where(ids == first + e)[0] for e in range(experts)]
            graphs, owners, errors, capacities = {}, [], {}, {}
            variants = [("int8", t) for t in (16, 32, 64)]
            if args.native_library:
                variants += [("bf16", t) for t in (16, 32)]
            for backend, tile in variants:
                name = f"{backend}_{tile}"
                tokens, tasks, offsets = [], [], []
                for e, rows in enumerate(slots):
                    start = len(tokens)
                    padded = (len(rows) + tile - 1) // tile * tile
                    offsets.append((start, len(rows), padded))
                    tokens.extend(rows.tolist() + [-1] * (padded - len(rows)))
                    tasks.extend((e, start + i) for i in range(0, padded, tile))
                token = torch.tensor(tokens, device="cuda", dtype=torch.int32)
                task = torch.tensor(tasks[::-1], device="cuda", dtype=torch.int32)
                out = torch.empty(len(tokens), n, device="cuda", dtype=torch.bfloat16)
                capacities[name] = dict(
                    rows=len(tokens), tasks=len(tasks), live=sum(map(len, slots))
                )

                weights = torch.ones(len(tokens), device="cuda")

                def run(
                    token=token,
                    task=task,
                    out=out,
                    tile=tile,
                    x=x,
                    a=a,
                    sa=sa,
                    count=count,
                    backend=backend,
                    weights=weights,
                ):
                    if backend == "bf16":
                        torch.ops.consumer_int8.gguf_iq_xxs_routed_mm(
                            x,
                            codes,
                            bases,
                            metadata,
                            token,
                            task,
                            weights,
                            out,
                            2,
                            tile,
                            False,
                        )
                        return
                    assert (
                        call(
                            *[v.data_ptr() for v in (x, a, sa, raw, token, task, out)],
                            count,
                            out.shape[0],
                            n,
                            k,
                            experts,
                            task.shape[0],
                            tile,
                            torch.cuda.current_stream().cuda_stream,
                        )
                        == 0
                    )

                for _ in range(3):
                    run()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                owners.append((token, task, out, graph, weights))
                graphs[name] = graph
                tested = (
                    range(experts) if experts <= 4 else (0, experts // 2, experts - 1)
                )
                references = {
                    e: torch.from_numpy(
                        gguf.dequantize(
                            raw_np[e].reshape(n, -1), gguf.GGMLQuantizationType.IQ2_XXS
                        )
                    ).cuda()
                    for e in tested
                }
                errors[name] = []
                for _ in range(2):
                    x.neg_()
                    token.copy_(torch.where(token >= 0, count - 1 - token, token))
                    out.fill_(float("nan"))
                    graph.replay()
                    assert torch.isfinite(out).all()
                    assert (out[token < 0] == 0).all()
                    if backend == "int8":
                        xb = x.float().reshape(count, k // 32, 32)
                        expected = (xb.abs().amax(-1) / 127).clamp_min(1e-20)
                        torch.testing.assert_close(sa, expected, rtol=1e-6, atol=0)
                        assert (
                            a.reshape_as(xb).float() - xb / expected[..., None]
                        ).abs().max() <= 0.50002
                    for e in tested:
                        start, live, _ = offsets[e]
                        if not live:
                            continue
                        selected = token[start : start + live].long()
                        expected_y = x[selected].float() @ references[e].T
                        actual = out[start : start + live].float()
                        relative = (
                            (actual - expected_y).norm() / expected_y.norm()
                        ).item()
                        assert relative < 0.015, (e, tile, relative)
                        errors[name].append(relative)
                del references
            times = {tile: [] for tile in graphs}
            for trial in range(20):
                for tile in list(graphs)[:: 1 if trial % 2 else -1]:
                    flush.zero_()
                    start, end = [
                        torch.cuda.Event(enable_timing=True) for _ in range(2)
                    ]
                    start.record()
                    graphs[tile].replay()
                    end.record()
                    end.synchronize()
                    times[tile].append(start.elapsed_time(end))
            record = dict(
                tokens=count,
                experts=experts,
                n=n,
                k=k,
                identity=identity,
                source="verified distinct experts"
                if args.checkpoint
                else "synthetic grouping of distinct tensor slices",
                medians_ms={t: statistics.median(v) for t, v in times.items()},
                milliseconds=times,
                capacities=capacities,
                relative_l2=errors,
                library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                native_library_sha256=(
                    hashlib.sha256(args.native_library.read_bytes()).hexdigest()
                    if args.native_library
                    else None
                ),
                benchmark_sha256=hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                device=str(torch.cuda.get_device_properties(args.device)),
                scope="quantization plus routed projection; excludes CPU routing",
            )
            report.write(json.dumps(record) + "\n")
            report.flush()
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--manifest", type=Path)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--native-library", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tensor", default="blk.2.ffn_gate_exps.weight")
    p.add_argument("--experts", type=int, default=192)
    p.add_argument("--first-expert", type=int, default=0)
    p.add_argument("--limit-n", type=int)
    p.add_argument("--tokens", type=int, nargs="+", default=[128, 512])
    p.add_argument("--device", type=int, default=1)
    main(p.parse_args())
