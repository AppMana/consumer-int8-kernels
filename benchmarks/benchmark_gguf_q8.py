# SPDX-License-Identifier: Apache-2.0
"""Race native Q8 against resident dequantized cuBLAS (a memory-costly ceiling).

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
            if args.tensor and item["name"] != args.tensor:
                continue
            path = args.manifest.parent / Path(item["path"]).name
            data = path.read_bytes()
            assert hashlib.sha256(data).hexdigest() == item["sha256"]
            k, n = item["dims"]
            raw_np = np.frombuffer(data, dtype=np.uint8).copy().reshape(n, k // 32, 34)
            scales = (
                np.ascontiguousarray(raw_np[..., :2]).view("<f2").astype(np.float32)
            )
            values = raw_np[..., 2:].view(np.int8).astype(np.float32)
            raw = torch.from_numpy(raw_np).unsqueeze(0).cuda()
            dense = torch.from_numpy((scales * values).reshape(n, k)).cuda().bfloat16()
            for m in args.rows:
                x = torch.randn(m, 1, k, device="cuda", dtype=torch.bfloat16) * 0.1
                out = torch.empty(m, 1, n, device="cuda", dtype=torch.bfloat16)
                base = torch.empty_like(out)

                def native(x=x, raw=raw, out=out):
                    torch.ops.consumer_int8.gguf_q8_mm(x, raw, out)

                def cublas(x=x, dense=dense, base=base):
                    torch.mm(x[:, 0], dense.T, out=base[:, 0])

                native()
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
                for name, fn in [("native", native), ("resident_cublas", cublas)]:
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
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--tensor", help="Profile or benchmark one manifest tensor")
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="One eager native call per shape, checked against FP32",
    )
    parser.add_argument("--rows", type=int, nargs="+", default=[128, 512, 1024, 2048])
    main(parser.parse_args())
