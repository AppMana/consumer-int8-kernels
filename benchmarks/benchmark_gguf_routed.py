# SPDX-License-Identifier: Apache-2.0
"""IQ2_XS gather/grouped race and MXFP4 routed16/routed32 race.

Replicates one actual checkpoint expert into distinct allocations. This isolates
routing/tile costs; it is not a full-model or mixed-quant MoE qualification.
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
    torch.manual_seed(571)
    torch.ops.load_library(str(args.library.resolve()))
    manifest = json.loads(args.manifest.read_text())
    item = next(s for s in manifest["samples"] if s["name"] == args.tensor)
    assert item["type"] in {"IQ2_XS", "MXFP4"} and item["experts"] == 1
    mx = item["type"] == "MXFP4"
    data = (args.manifest.parent / Path(item["path"]).name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == item["sha256"]
    k, n = item["dims"][:2]
    raw_np = (
        np.frombuffer(data, dtype=np.uint8)
        .copy()
        .reshape(n, k // (32 if mx else 256), 17 if mx else 74)
    )
    decoded = gguf.dequantize(
        raw_np.reshape(-1, raw_np.shape[-1]),
        getattr(gguf.GGMLQuantizationType, item["type"]),
    )
    dense = torch.from_numpy(decoded.reshape(n, k)).cuda().bfloat16()
    raw = torch.from_numpy(raw_np).cuda()
    if mx:
        codes_one = raw[..., 1:].contiguous().flatten(1)
        bases_one = None
        scales_one = raw[..., 0].contiguous()
    else:
        codes_one = raw[..., 2:66].contiguous().view(torch.int16).flatten(1)
        bases_one = raw[..., :2].contiguous().view(torch.float16).squeeze(-1)
        if bases_one.shape[-1] % 2:
            bases_one = torch.nn.functional.pad(bases_one, (0, 1))
        scales_one = raw[..., 66:].contiguous().flatten(1)
    flush = torch.empty(64 << 20, device="cuda", dtype=torch.uint8)
    x = torch.randn(512, k, device="cuda", dtype=torch.bfloat16) * 0.1
    with args.output.open("x") as report:
        for experts in args.experts:
            codes = codes_one.unsqueeze(0).repeat(experts, 1, 1)
            bases = None if mx else bases_one.unsqueeze(0).repeat(experts, 1, 1)
            scales = scales_one.unsqueeze(0).repeat(experts, 1, 1)
            for live in args.live_rows:
                width = 32 if mx else 16
                capacity = (live + width - 1) // width * width
                tokens = torch.randint(
                    0, 512, (experts, capacity), device="cuda", dtype=torch.int32
                )
                tokens[:, live:] = -1
                tokens = tokens.flatten()
                safe_ids = tokens.clamp_min(0).long()
                padding = (tokens < 0)[:, None]
                tasks = torch.tensor(
                    [
                        (e, e * capacity + offset)
                        for e in range(experts)
                        for offset in range(0, live, 16)
                    ],
                    device="cuda",
                    dtype=torch.int32,
                )
                tasks32 = torch.tensor(
                    [
                        (e, e * capacity + offset)
                        for e in range(experts)
                        for offset in range(0, capacity, 32)
                    ],
                    device="cuda",
                    dtype=torch.int32,
                )
                out = torch.zeros(
                    experts * capacity, n, device="cuda", dtype=torch.bfloat16
                )
                grouped = torch.empty(
                    capacity, experts, n, device="cuda", dtype=torch.bfloat16
                )
                grouped_route = torch.empty(
                    experts * capacity, n, device="cuda", dtype=torch.bfloat16
                )
                if mx:
                    grouped = grouped_route.view(experts, capacity, n).transpose(0, 1)
                gathered = torch.empty(
                    experts * capacity, k, device="cuda", dtype=torch.bfloat16
                )
                grouped_x = gathered.view(experts, capacity, k).transpose(0, 1)

                def routed(
                    codes=codes,
                    bases=bases,
                    scales=scales,
                    tokens=tokens,
                    tasks=tasks,
                    out=out,
                ):
                    if mx:
                        torch.ops.consumer_int8.gguf_mxfp4_routed_mm(
                            x, codes, scales, tokens, tasks, out
                        )
                    else:
                        torch.ops.consumer_int8.gguf_iq2_xs_routed_mm(
                            x, codes, bases, scales, tokens, tasks, out
                        )

                def routed32(
                    codes=codes,
                    bases=bases,
                    scales=scales,
                    tokens=tokens,
                    tasks32=tasks32,
                    out=out,
                ):
                    torch.ops.consumer_int8.gguf_iq2_xs_routed_mm(
                        x, codes, bases, scales, tokens, tasks32, out, 32
                    )

                def materialized(
                    safe_ids=safe_ids,
                    gathered=gathered,
                    padding=padding,
                    grouped_x=grouped_x,
                    codes=codes,
                    bases=bases,
                    scales=scales,
                    grouped=grouped,
                    tokens=tokens,
                    tasks32=tasks32,
                    grouped_route=grouped_route,
                ):
                    if mx:
                        torch.ops.consumer_int8.gguf_mxfp4_routed_mm(
                            x, codes, scales, tokens, tasks32, grouped_route, 32
                        )
                        return
                    torch.index_select(x, 0, safe_ids, out=gathered)
                    gathered.masked_fill_(padding, 0)
                    torch.ops.consumer_int8.gguf_iq2_xs_mm(
                        grouped_x, codes, bases, scales, grouped
                    )

                routed()
                materialized()
                assert torch.equal(
                    out.view(experts, capacity, n).transpose(0, 1), grouped
                )
                ref = (x.float() @ dense.float().T).bfloat16()[safe_ids]
                ref.masked_fill_(padding, 0)
                relative = (out.float() - ref.float()).norm() / ref.float().norm()
                assert torch.isfinite(out).all() and relative < 0.001
                candidates = [
                    ("routed16", routed),
                    ("routed32" if mx else "gather_grouped32", materialized),
                ]
                if not mx and capacity % 32 == 0:
                    routed32()
                    assert torch.equal(
                        out.view(experts, capacity, n).transpose(0, 1), grouped
                    )
                    candidates.append(("routed32", routed32))
                graphs = {}
                for name, fn in candidates:
                    for _ in range(3):
                        fn()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        fn()
                    graphs[name] = graph
                times = {name: [] for name in graphs}
                for trial in range(10):
                    for name in list(graphs)[:: 1 if trial % 2 else -1]:
                        flush.zero_()
                        begin = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        begin.record()
                        graphs[name].replay()
                        end.record()
                        end.synchronize()
                        times[name].append(begin.elapsed_time(end))
                record = dict(
                    quant=item["type"],
                    checkpoint=manifest["repo"],
                    revision=manifest["revision"],
                    sample_sha256=item["sha256"],
                    library_sha256=hashlib.sha256(
                        args.library.read_bytes()
                    ).hexdigest(),
                    tensor=item["name"],
                    experts=experts,
                    live_rows=live,
                    capacity=capacity,
                    n=n,
                    k=k,
                    relative_l2=relative.item(),
                    milliseconds=times,
                    medians_ms={
                        name: statistics.median(v) for name, v in times.items()
                    },
                    device=str(torch.cuda.get_device_properties(args.device)),
                    torch=torch.__version__,
                    scope="Replicated expert; projection only; prepared routes",
                )
                report.write(json.dumps(record) + "\n")
                report.flush()
                print(
                    json.dumps(
                        {
                            key: record[key]
                            for key in [
                                "experts",
                                "live_rows",
                                "medians_ms",
                                "relative_l2",
                            ]
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor", default="blk.0.ffn_gate_exps.weight")
    parser.add_argument("--experts", type=int, nargs="+", default=[32, 192])
    parser.add_argument("--live-rows", type=int, nargs="+", default=[4, 8, 16, 24, 32])
    parser.add_argument("--device", type=int, default=1)
    main(parser.parse_args())
