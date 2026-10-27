# SPDX-License-Identifier: Apache-2.0
"""Measure compressed IQ2_XXS -> shared INT8 MMA, with original block scales.

Build cuda/iq2xxs_block_int8.cu ahead of time; no JIT for the native candidate.
One actual expert, contiguous rows: not routed MoE or full-model performance.
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


def main(args):
    torch.cuda.set_device(args.device)
    torch.manual_seed(137)
    torch.backends.cuda.matmul.allow_tf32 = False
    lib = ctypes.CDLL(str(args.library.resolve()))
    entries = list(
        dict.fromkeys(
            [args.entry] + (["iq2xxs_block_int8"] if args.compare_scalar else [])
        )
    )
    manifest = json.loads(args.manifest.read_text())
    item = next(s for s in manifest["samples"] if s["name"] == args.tensor)
    assert item["type"] == "IQ2_XXS"
    data = (args.manifest.parent / Path(item["path"]).name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == item["sha256"]
    k, full_n, _ = item["dims"]
    n = min(args.limit_n or full_n, full_n)
    raw_np = np.frombuffer(data, np.uint8).reshape(full_n, k // 256, 66)[:n].copy()
    dense_np = gguf.dequantize(raw_np.reshape(n, -1), gguf.GGMLQuantizationType.IQ2_XXS)
    raw = torch.from_numpy(raw_np).cuda()
    dense = torch.from_numpy(dense_np).cuda().float()
    flush = torch.empty(64 << 20, device="cuda", dtype=torch.uint8)
    with args.output.open("x") as report:
        for m in args.rows:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
            x[:, :32] = 0
            a = torch.empty_like(x, dtype=torch.int8)
            sa = torch.empty(m, k // 32, device="cuda")
            graphs, outputs = {}, {}
            for entry, tile in [(e, t) for e in entries for t in (16, 32, 64)]:
                call = getattr(lib, entry)
                call.argtypes = (
                    [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
                )
                call.restype = ctypes.c_int
                out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

                def run(tile=tile, out=out, x=x, a=a, sa=sa, m=m, call=call):
                    assert (
                        call(
                            *[t.data_ptr() for t in (x, a, sa, raw, out)],
                            m,
                            n,
                            k,
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
                name = f"{entry.removeprefix('iq2xxs_block_')}_{tile}"
                graphs[name], outputs[name] = graph, out
            if args.compare_b12x:
                from benchmark_gguf_q8 import prepare_b12x

                out = torch.empty_like(out)
                prepared = prepare_b12x(raw[None], x[:, None], out[:, None], "IQ2_XXS")
                for _ in range(3):
                    prepared.run()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    prepared.run()
                graphs["b12x"], outputs["b12x"] = graph, out
            errors = {}
            for _ in range(2):
                x.neg_()
                reference = x.float() @ dense.T
                for name, graph in graphs.items():
                    outputs[name].fill_(float("nan"))
                    graph.replay()
                    out = outputs[name]
                    assert torch.isfinite(out).all()
                    relative = (
                        (out.float() - reference).norm() / reference.norm()
                    ).item()
                    assert relative < 0.015, (name, relative)
                    errors[name] = relative
                    if name.startswith("int8"):
                        xb = x.float().reshape(m, k // 32, 32)
                        expected_s = (xb.abs().amax(-1) / 127).clamp_min(1e-20)
                        torch.testing.assert_close(sa, expected_s, rtol=1e-6, atol=0)
                        codes = a.reshape_as(xb).float()
                        assert (
                            codes - xb / expected_s[..., None]
                        ).abs().max() <= 0.50002
                        oracle = torch.zeros_like(reference)
                        for block in range(k // 32):
                            oracle += (
                                codes[:, block]
                                @ dense[:, block * 32 : (block + 1) * 32].T
                            ) * sa[:, block, None]
                        torch.testing.assert_close(
                            out, oracle.bfloat16(), rtol=0.008, atol=1e-5
                        )
            times = {name: [] for name in graphs}
            for trial in range(20):
                for name in list(graphs)[:: 1 if trial % 2 else -1]:
                    flush.zero_()
                    start, end = [
                        torch.cuda.Event(enable_timing=True) for _ in range(2)
                    ]
                    start.record()
                    graphs[name].replay()
                    end.record()
                    end.synchronize()
                    times[name].append(start.elapsed_time(end))
            record = dict(
                entry=args.entry,
                m=m,
                n=n,
                k=k,
                tensor=args.tensor,
                checkpoint=manifest["repo"],
                revision=manifest["revision"],
                sample_sha256=item["sha256"],
                library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                benchmark_sha256=hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                device=str(torch.cuda.get_device_properties(args.device)),
                medians_ms={name: statistics.median(t) for name, t in times.items()},
                milliseconds=times,
                relative_l2=errors,
                scope="one expert; activation quantization plus IQ2_XXS INT8 MMA",
                weight_requantization=False,
            )
            report.write(json.dumps(record) + "\n")
            report.flush()
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", default="iq2xxs_block_int8")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tensor", default="blk.2.ffn_gate_exps.weight")
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 17, 128, 512])
    parser.add_argument("--limit-n", type=int)
    parser.add_argument("--compare-scalar", action="store_true")
    parser.add_argument("--compare-b12x", action="store_true")
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
