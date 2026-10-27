# SPDX-License-Identifier: Apache-2.0
"""Race native Q8/IQ2_XS against resident dequantized cuBLAS (a memory-costly ceiling).

Loads an explicitly built library; does not compile during import or measurement.
The raw sample manifest binds checkpoint identity and bytes. JSONL output belongs
outside the source tree; record findings and commands in commit messages.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
import torch


def main(args):
    torch.cuda.set_device(args.device)
    torch.manual_seed(137)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.ops.load_library(str(args.library.resolve()))
    manifest = json.loads(args.manifest.read_text())
    flush = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
    with args.output.open("x") as report:
        for item in manifest["samples"]:
            if args.quant == "IQ2_XS" and item.get("type") != "IQ2_XS":
                continue
            if args.tensor and item["name"] != args.tensor:
                continue
            path = args.manifest.parent / Path(item["path"]).name
            data = path.read_bytes()
            assert hashlib.sha256(data).hexdigest() == item["sha256"]
            k, n = item["dims"][:2]
            nibbles = None
            if args.quant == "IQ2_XS":
                import gguf

                assert item.get("experts", 1) == 1
                raw_np = (
                    np.frombuffer(data, dtype=np.uint8).copy().reshape(n, k // 256, 74)
                )
                raw = torch.from_numpy(raw_np).unsqueeze(0).cuda()
                codes = raw[..., 2:66].contiguous().view(torch.int16).flatten(2)
                scales_gpu = raw[..., :2].contiguous().view(torch.float16).squeeze(-1)
                if scales_gpu.shape[-1] % 2:
                    scales_gpu = torch.nn.functional.pad(scales_gpu, (0, 1))
                nibbles = raw[..., 66:].contiguous().flatten(2)
                decoded = gguf.dequantize(
                    raw_np.reshape(-1, 74), gguf.GGMLQuantizationType.IQ2_XS
                )
                dense = torch.from_numpy(decoded.reshape(n, k)).cuda().bfloat16()
            else:
                raw_np = (
                    np.frombuffer(data, dtype=np.uint8).copy().reshape(n, k // 32, 34)
                )
                scales = (
                    np.ascontiguousarray(raw_np[..., :2]).view("<f2").astype(np.float32)
                )
                values = raw_np[..., 2:].view(np.int8).astype(np.float32)
                raw = torch.from_numpy(raw_np).unsqueeze(0).cuda()
                codes = raw[..., 2:].contiguous().view(torch.int8).reshape(1, n, k)
                scales_gpu = raw[..., :2].contiguous().view(torch.float16).squeeze(-1)
                dense = (
                    torch.from_numpy((scales * values).reshape(n, k)).cuda().bfloat16()
                )
            expanded = torch.empty(1, n, k, device="cuda", dtype=torch.bfloat16)
            for m in args.rows:
                x = torch.randn(m, 1, k, device="cuda", dtype=torch.bfloat16) * 0.1
                out = torch.empty(m, 1, n, device="cuda", dtype=torch.bfloat16)
                base = torch.empty_like(out)

                def native(x=x, raw=raw, out=out):
                    torch.ops.consumer_int8.gguf_q8_mm(x, raw, out)

                def packed(
                    x=x, codes=codes, scales_gpu=scales_gpu, out=out, nibbles=nibbles
                ):
                    if args.quant == "IQ2_XS":
                        torch.ops.consumer_int8.gguf_iq2_xs_mm(
                            x, codes, scales_gpu, nibbles, out
                        )
                    else:
                        torch.ops.consumer_int8.gguf_q8_packed_mm(
                            x, codes, scales_gpu, out
                        )

                def cublas(x=x, dense=dense, base=base):
                    torch.mm(x[:, 0], dense.T, out=base[:, 0])

                def expand_cublas(
                    x=x,
                    codes=codes,
                    scales_gpu=scales_gpu,
                    expanded=expanded,
                    base=base,
                ):
                    torch.ops.consumer_int8.gguf_q8_expand(codes, scales_gpu, expanded)
                    torch.mm(x[:, 0], expanded[0].T, out=base[:, 0])

                if args.quant == "Q8_0":
                    native()
                    raw_result = out.clone()
                packed()
                if args.quant == "Q8_0":
                    assert torch.equal(out, raw_result)
                reference = (x[:, 0].float() @ dense.float().T).bfloat16()
                relative = (
                    (out[:, 0].float() - reference.float()).norm()
                    / reference.float().norm()
                ).item()
                assert (
                    relative < 0.001
                    and torch.isfinite(out).all()
                    and out.abs().max() > 0
                )
                if args.profile_only:
                    torch.cuda.synchronize()
                    record = {"tensor": item["name"], "m": m, "relative_l2": relative}
                    report.write(json.dumps(record) + "\n")
                    report.flush()
                    continue
                graphs = {}
                candidates = [("packed_async", packed), ("resident_cublas", cublas)]
                if args.quant == "Q8_0":
                    expand_cublas()
                    assert torch.equal(expanded[0], dense)
                    cublas_result = base.clone()
                    cublas()
                    assert torch.equal(base, cublas_result)
                    candidates.insert(0, ("native", native))
                    candidates.append(("expand_cublas", expand_cublas))
                for name, fn in candidates:
                    for _ in range(3):
                        fn()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        fn()
                    graphs[name] = graph
                measurements = {name: [] for name in graphs}
                for trial in range(10):
                    for name in (
                        list(graphs) if trial % 2 == 0 else list(reversed(graphs))
                    ):
                        flush.zero_()
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        graphs[name].replay()
                        end.record()
                        end.synchronize()
                        measurements[name].append(start.elapsed_time(end))
                record = dict(
                    workspace_bytes=expanded.numel() * expanded.element_size(),
                    quant=args.quant,
                    checkpoint=manifest["repo"],
                    revision=manifest["revision"],
                    tensor=item["name"],
                    sample_sha256=item["sha256"],
                    library_sha256=hashlib.sha256(
                        args.library.read_bytes()
                    ).hexdigest(),
                    device=str(torch.cuda.get_device_properties(args.device)),
                    torch=torch.__version__,
                    m=m,
                    n=n,
                    k=k,
                    relative_l2=relative,
                    milliseconds=measurements,
                    medians_ms={
                        key: statistics.median(v) for key, v in measurements.items()
                    },
                    baseline_scope=(
                        "Resident BF16 cuBLAS excludes dequantization "
                        "and expanded-weight memory cost"
                    ),
                )
                report.write(json.dumps(record) + "\n")
                report.flush()
                print(
                    json.dumps(
                        {
                            key: record[key]
                            for key in ("tensor", "m", "relative_l2", "medians_ms")
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quant", choices=["Q8_0", "IQ2_XS"], default="Q8_0")
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--tensor", help="Profile or benchmark one manifest tensor")
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="One eager native call per shape, checked against FP32",
    )
    parser.add_argument("--rows", type=int, nargs="+", default=[128, 512, 1024, 2048])
    main(parser.parse_args())
