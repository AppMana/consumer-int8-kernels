# SPDX-License-Identifier: Apache-2.0
"""Validate/measure the experimental block32 INT8 MMA; no JIT or weight conversion.

Build benchmarks/cuda/q8_block_int8.cu with nvcc -O3 -shared -Xcompiler -fPIC
and explicit target architectures. The raw C ABI is benchmark-only.
"""

import argparse
import ctypes
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
import torch


def main(args):
    torch.cuda.set_device(args.device)
    torch.manual_seed(137)
    if args.native_library:
        torch.ops.load_library(str(args.native_library.resolve()))
    if args.compare_b12x and not args.manifest:
        raise ValueError("b12x comparison requires a raw Q8 manifest")
    b12x_source_hash = None
    if args.compare_b12x:
        import b12x

        root = Path(b12x.__file__).parent
        b12x_source_hash = hashlib.sha256(
            b"".join(
                str(path.relative_to(root)).encode() + path.read_bytes()
                for path in sorted(root.rglob("*.py"))
            )
        ).hexdigest()
    lib = ctypes.CDLL(str(args.library.resolve()))
    call = getattr(lib, args.entry)
    call.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    call.restype = ctypes.c_int
    manifest = json.loads(args.manifest.read_text()) if args.manifest else None
    flush = torch.empty(64 << 20, device="cuda", dtype=torch.uint8)
    with args.output.open("x") as report:
        for m, n, k in (
            [(m, 1024, 4096) for m in args.rows]
            if manifest
            else [(1, 1, 32), (17, 33, 96), (129, 65, 256)]
        ):
            sample_hash = None
            if manifest:
                item = manifest["samples"][0]
                assert item["type"] == "Q8_0" and item["dims"] == [4096, 8192]
                data = (args.manifest.parent / Path(item["path"]).name).read_bytes()
                sample_hash = hashlib.sha256(data).hexdigest()
                assert sample_hash == item["sha256"]
                raw = np.frombuffer(data, np.uint8).reshape(8192, 128, 34)[4096:5120]
                w = torch.from_numpy(raw[..., 2:].copy().view(np.int8)).cuda()
                sw = torch.from_numpy(
                    np.ascontiguousarray(raw[..., :2]).view("<f2").squeeze(-1)
                ).cuda()
            else:
                w = torch.randint(
                    -128, 128, (n, k // 32, 32), device="cuda", dtype=torch.int8
                )
                sw = torch.randn(n, k // 32, device="cuda", dtype=torch.float16) * 0.01
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
            # Exercise a zero block and strongly nonuniform block magnitudes.
            if not manifest:
                x[:, :32] = 0
                x[:, 32:64] *= 100
            a = torch.empty_like(x, dtype=torch.int8)
            sa = torch.empty(m, k // 32, device="cuda", dtype=torch.float32)
            y = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

            def run(x=x, a=a, sa=sa, w=w, sw=sw, y=y, m=m, n=n, k=k):
                error = call(
                    *[t.data_ptr() for t in (x, a, sa, w, sw, y)],
                    m,
                    n,
                    k,
                    torch.cuda.current_stream().cuda_stream,
                )
                assert error == 0, error

            for _ in range(3):
                run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            # Independent quantization and blockwise dot-product oracle.
            errors = []
            for _ in range(2):
                x.neg_()
                y.fill_(float("nan"))
                graph.replay()
                xb = x.float().reshape(m, k // 32, 32)
                expected_s = (xb.abs().amax(-1) / 127).clamp_min(1e-20)
                expected_a = (xb / expected_s[..., None]).round().to(torch.int8)
                torch.testing.assert_close(sa, expected_s, rtol=1e-6, atol=0)
                # CUDA division and PyTorch reciprocal multiplication can
                # round opposite ways at half-integer ties. Check the actual
                # quantizer error bound, then use its codes in the MMA oracle.
                actual_a = a.reshape_as(expected_a)
                assert (
                    (actual_a.float() - xb / expected_s[..., None]).abs() <= 0.50002
                ).all()
                oracle = torch.zeros(m, n, device="cuda")
                for block in range(k // 32):
                    dot = actual_a[:, block].float() @ w[:, block].float().T
                    oracle += dot * (
                        expected_s[:, block, None] * sw[None, :, block].float()
                    )
                torch.testing.assert_close(y, oracle.bfloat16(), rtol=0.008, atol=1e-5)
                reference = (
                    x.float() @ (w.float() * sw[..., None].float()).reshape(n, k).T
                )
                relative = (
                    (y.float() - reference).norm() / reference.norm().clamp_min(1e-20)
                ).item()
                assert torch.isfinite(y).all() and relative < 0.015
                errors.append(relative)
            graphs, outputs = {args.entry: graph}, {args.entry: y}
            candidates = []
            weight_error = None
            if args.native_library:
                dense = (w.float() * sw[..., None].float()).reshape(n, k).bfloat16()
                channel_s = (
                    (dense.float().abs().amax(1) / 127).clamp_min(1e-20).bfloat16()
                )
                channel_w = (
                    (dense.float() / channel_s.float()[:, None])
                    .round()
                    .clamp(-127, 127)
                    .to(torch.int8)
                )
                weight_error = (
                    (
                        channel_w.float() * channel_s.float()[:, None] - dense.float()
                    ).norm()
                    / dense.float().norm().clamp_min(1e-20)
                ).item()
                qa = torch.empty_like(x, dtype=torch.int8)
                row_s = torch.empty(m, device="cuda", dtype=torch.float32)
                for schedule in (0, 3, 6):
                    out = torch.empty_like(y)

                    def channel(
                        x=x,
                        qa=qa,
                        row_s=row_s,
                        channel_w=channel_w,
                        channel_s=channel_s,
                        out=out,
                        schedule=schedule,
                    ):
                        torch.ops.consumer_int8.quantize_rows(x, qa, row_s)
                        torch.ops.consumer_int8.fused_channel_mm(
                            qa, channel_w, row_s, channel_s, out, schedule
                        )

                    candidates.append((f"channel_{schedule}", channel, out))
            if args.compare_b12x:
                from benchmark_gguf_q8 import prepare_b12x

                raw_gpu = torch.from_numpy(raw.copy()).cuda().unsqueeze(0)
                out = torch.empty_like(y)
                prepared = prepare_b12x(raw_gpu, x[:, None], out[:, None], "Q8_0")
                candidates.append(("b12x", prepared.run, out))
            candidate_errors = {}
            for name, function, out in candidates:
                for _ in range(3):
                    function()
                other = torch.cuda.CUDAGraph()
                with torch.cuda.graph(other):
                    function()
                graphs[name], outputs[name] = other, out
            # Fresh changed input across all captured implementations.
            x.neg_()
            reference = -reference
            for name, captured in graphs.items():
                outputs[name].fill_(float("nan"))
                captured.replay()
                result = outputs[name].float()
                error = (
                    (result - reference).norm() / reference.norm().clamp_min(1e-20)
                ).item()
                assert torch.isfinite(result).all() and error < (
                    0.03 if name.startswith("channel_") else 0.015
                )
                candidate_errors[name] = error
            measurements = {name: [] for name in graphs}
            for trial in range(10):
                for name in list(graphs)[:: 1 if trial % 2 else -1]:
                    flush.zero_()
                    start, end = [
                        torch.cuda.Event(enable_timing=True) for _ in range(2)
                    ]
                    start.record()
                    graphs[name].replay()
                    end.record()
                    end.synchronize()
                    measurements[name].append(start.elapsed_time(end))
            times = measurements[args.entry]
            record = dict(
                m=m,
                n=n,
                k=k,
                milliseconds=times,
                median_ms=statistics.median(times),
                candidate_milliseconds=measurements,
                candidate_medians_ms={
                    name: statistics.median(values)
                    for name, values in measurements.items()
                },
                candidate_relative_l2=candidate_errors,
                channel_weight_requantization_l2=weight_error,
                native_library_sha256=hashlib.sha256(
                    args.native_library.read_bytes()
                ).hexdigest()
                if args.native_library
                else None,
                relative_l2=errors,
                sample_sha256=sample_hash,
                revision=manifest["revision"] if manifest else None,
                library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                benchmark_sha256=hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                b12x_source_sha256=b12x_source_hash,
                device=str(torch.cuda.get_device_properties(args.device)),
                weight_requantization=False,
                entry=args.entry,
                scope="activation quantization plus block32 INT8 MMA",
            )
            report.write(json.dumps(record) + "\n")
            report.flush()
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument(
        "--entry",
        choices=["q8_block_int8", "q8_block_int8_tiled", "q8_block_int8_tiled64"],
        default="q8_block_int8",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--native-library", type=Path)
    parser.add_argument("--compare-b12x", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--rows", type=int, nargs="+", default=[128, 512, 1024, 2048])
    main(parser.parse_args())
