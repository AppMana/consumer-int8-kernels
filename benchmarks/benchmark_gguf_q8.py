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


def prepare_b12x(raw, x, out, recipe, packed_weights=None):
    from b12x.gemm import blockscaled
    from b12x.preparation import PreparationSession, PreparedCall, require_prepared

    calls, outputs = [], []
    with PreparationSession(
        device=x.device, autotune=False, compile_workers=1
    ) as session:
        for group in range(x.shape[1]):
            weight = (
                packed_weights[group]
                if packed_weights is not None
                else blockscaled.pack_weight(raw[group], recipe=recipe.lower())
            )
            gathered = torch.empty_like(
                x[:, group], memory_format=torch.contiguous_format
            )
            output = (
                torch.empty_like(out[:, group], memory_format=torch.contiguous_format)
                if x.shape[1] > 1
                else out[:, 0]
            )
            gathered.copy_(x[:, group])
            plan = blockscaled.plan(
                blockscaled.query_from_call(gathered, weight, out=output)
            )

            def factory(
                state, weight=weight, gathered=gathered, output=output, group=group
            ):
                def run():
                    if x.shape[1] > 1:
                        gathered.copy_(x[:, group])
                    state.run(
                        gathered if x.shape[1] > 1 else x[:, 0],
                        weight.values,
                        weight.metadata,
                        None,
                        out=output,
                    )

                return PreparedCall(
                    run=run, output=output, owners=(weight, state, gathered)
                )

            session.prepare(
                (plan.request(name="paired GGUF dense", prepare_call=factory),)
            )
            calls.append(factory(require_prepared(plan, "gemm.blockscaled_precision")))
            outputs.append(output)
        session.freeze()

    def run():
        for call in calls:
            call.run()
        if x.shape[1] > 1:
            torch.stack(outputs, dim=1, out=out)

    return PreparedCall(run=run, output=out, owners=tuple(calls))


def main(args):
    if args.compare_q6_packed and args.quant != "Q6_K":
        raise ValueError("Packed Q6 comparison requires Q6_K")
    if (
        args.k_start or args.k_count or args.pad_k_to or args.pad_n_to
    ) and args.quant != "Q6_K":
        raise ValueError("Padded/K partitioning currently requires Q6_K")
    torch.cuda.set_device(args.device)
    torch.manual_seed(137)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.ops.load_library(str(args.library.resolve()))
    b12x_source_hash = None
    if args.compare_b12x or args.compare_q6_packed:
        import b12x

        source_root = Path(b12x.__file__).parent
        b12x_source_hash = hashlib.sha256(
            b"".join(
                str(path.relative_to(source_root)).encode() + path.read_bytes()
                for path in sorted(source_root.rglob("*.py"))
            )
        ).hexdigest()
    manifest = json.loads(args.manifest.read_text())
    flush = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
    with args.output.open("x") as report:
        for item in manifest["samples"]:
            if item.get("type", "Q8_0") != args.quant:
                continue
            if args.tensor and item["name"] != args.tensor:
                continue
            path = args.manifest.parent / Path(item["path"]).name
            data = path.read_bytes()
            assert hashlib.sha256(data).hexdigest() == item["sha256"]
            k, source_n = item["dims"][:2]
            padded_n = args.pad_n_to or source_n
            selected_n = args.row_count or padded_n - args.row_start
            if (
                args.row_start < 0
                or padded_n < source_n
                or selected_n <= 0
                or args.row_start + selected_n > padded_n
                or selected_n % args.groups
            ):
                raise ValueError("Invalid source row partition")
            n = selected_n // args.groups
            if args.quant == "IQ2_XS" and args.groups != 1:
                raise ValueError("Grouped benchmark requires Q8 or Q6")
            nibbles = None
            if args.quant == "IQ2_XS":
                import gguf

                assert item.get("experts", 1) == 1
                raw_np = (
                    np.frombuffer(data, dtype=np.uint8)
                    .copy()
                    .reshape(source_n, k // 256, 74)[
                        args.row_start : args.row_start + selected_n
                    ]
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
                dense = torch.from_numpy(decoded.reshape(1, n, k)).cuda().bfloat16()
            elif args.quant == "Q6_K":
                import gguf

                raw_np = (
                    np.frombuffer(data, dtype=np.uint8)
                    .copy()
                    .reshape(source_n, k // 256, 210)
                )
                raw_np = np.pad(raw_np, ((0, padded_n - source_n), (0, 0), (0, 0)))
                raw_np = raw_np[args.row_start : args.row_start + selected_n].reshape(
                    args.groups, n, k // 256, 210
                )
                padded_k = args.pad_k_to or k
                selected_k = args.k_count or padded_k - args.k_start
                if (
                    padded_k < k
                    or padded_k % 256
                    or args.k_start < 0
                    or args.k_start % 256
                    or selected_k <= 0
                    or selected_k % 256
                    or args.k_start + selected_k > padded_k
                ):
                    raise ValueError("Q6 K padding/partition must use whole blocks")
                raw_np = np.pad(
                    raw_np, ((0, 0), (0, 0), (0, (padded_k - k) // 256), (0, 0))
                )
                raw_np = np.ascontiguousarray(
                    raw_np[
                        :, :, args.k_start // 256 : (args.k_start + selected_k) // 256
                    ]
                )
                k = selected_k
                raw = torch.from_numpy(raw_np).cuda()
                decoded = gguf.dequantize(
                    raw_np.reshape(-1, 210), gguf.GGMLQuantizationType.Q6_K
                )
                dense = (
                    torch.from_numpy(decoded.reshape(args.groups, n, k))
                    .cuda()
                    .bfloat16()
                )
                codes, scales_gpu = None, None
            else:
                raw_np = (
                    np.frombuffer(data, dtype=np.uint8)
                    .copy()
                    .reshape(source_n, k // 32, 34)[
                        args.row_start : args.row_start + selected_n
                    ]
                    .reshape(args.groups, n, k // 32, 34)
                )
                scales = (
                    np.ascontiguousarray(raw_np[..., :2]).view("<f2").astype(np.float32)
                )
                values = raw_np[..., 2:].view(np.int8).astype(np.float32)
                raw = torch.from_numpy(raw_np).cuda()
                codes = (
                    raw[..., 2:]
                    .contiguous()
                    .view(torch.int8)
                    .reshape(args.groups, n, k)
                )
                scales_gpu = raw[..., :2].contiguous().view(torch.float16).squeeze(-1)
                dense = (
                    torch.from_numpy((scales * values).reshape(args.groups, n, k))
                    .cuda()
                    .bfloat16()
                )
            expanded = torch.empty(
                args.groups, n, k, device="cuda", dtype=torch.bfloat16
            )
            packed_weights = None
            if args.compare_q6_packed:
                from b12x.gemm.blockscaled import pack_weight

                packed_weights = tuple(
                    pack_weight(raw[group], recipe="q6_k")
                    for group in range(args.groups)
                )
            for m in args.rows:
                x = (
                    torch.randn(m, args.groups, k, device="cuda", dtype=torch.bfloat16)
                    * 0.1
                )
                out = torch.empty(
                    m, args.groups, n, device="cuda", dtype=torch.bfloat16
                )
                base = torch.empty_like(out)

                def native(x=x, raw=raw, out=out):
                    if args.quant == "Q6_K":
                        torch.ops.consumer_int8.gguf_q6_mm(x, raw, out, 32)
                    else:
                        torch.ops.consumer_int8.gguf_q8_mm(x, raw, out)

                def packed(
                    x=x,
                    codes=codes,
                    scales_gpu=scales_gpu,
                    out=out,
                    nibbles=nibbles,
                    raw=raw,
                ):
                    if args.quant == "IQ2_XS":
                        torch.ops.consumer_int8.gguf_iq2_xs_mm(
                            x, codes, scales_gpu, nibbles, out
                        )
                    elif args.quant == "Q6_K":
                        torch.ops.consumer_int8.gguf_q6_mm(x, raw, out, 64)
                    else:
                        torch.ops.consumer_int8.gguf_q8_packed_mm(
                            x, codes, scales_gpu, out
                        )

                def cublas(x=x, dense=dense, base=base):
                    torch.bmm(
                        x.transpose(0, 1),
                        dense.transpose(1, 2),
                        out=base.transpose(0, 1),
                    )

                def expand_cublas(
                    x=x,
                    codes=codes,
                    scales_gpu=scales_gpu,
                    expanded=expanded,
                    base=base,
                    raw=raw,
                ):
                    if args.quant == "Q6_K":
                        torch.ops.consumer_int8.gguf_q6_expand(raw, expanded)
                    else:
                        torch.ops.consumer_int8.gguf_q8_expand(
                            codes, scales_gpu, expanded
                        )
                    torch.bmm(
                        x.transpose(0, 1),
                        expanded.transpose(1, 2),
                        out=base.transpose(0, 1),
                    )

                if args.quant in ("Q8_0", "Q6_K"):
                    native()
                    raw_result = out.clone()
                packed()
                if args.quant in ("Q8_0", "Q6_K"):
                    assert torch.equal(out, raw_result)
                reference = torch.einsum(
                    "mgk,gnk->mgn", x.float(), dense.float()
                ).bfloat16()
                relative = (
                    (out.float() - reference.float()).norm() / reference.float().norm()
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
                if args.quant in ("Q8_0", "Q6_K"):
                    expand_cublas()
                    assert torch.equal(expanded, dense)
                    cublas_result = base.clone()
                    cublas()
                    assert torch.equal(base, cublas_result)
                    candidates.insert(0, ("native", native))
                    candidates.append(("expand_cublas", expand_cublas))
                if args.quant == "Q6_K":
                    candidates = [
                        (
                            "inline32"
                            if name == "native"
                            else "inline64"
                            if name == "packed_async"
                            else name,
                            fn,
                        )
                        for name, fn in candidates
                    ]
                b12x_call = None
                if packed_weights is not None:

                    def packed_expand_cublas(
                        x=x, weights=packed_weights, expanded=expanded, base=base
                    ):
                        for group, weight in enumerate(weights):
                            torch.ops.consumer_int8.gguf_q6_packed_expand(
                                weight.values, weight.metadata, expanded[group]
                            )
                        torch.bmm(
                            x.transpose(0, 1),
                            expanded.transpose(1, 2),
                            out=base.transpose(0, 1),
                        )

                    packed_expand_cublas()
                    assert torch.equal(expanded, dense)
                    candidates.append(("packed_expand_cublas", packed_expand_cublas))
                if args.compare_b12x:
                    b12x_out = torch.empty_like(out)
                    b12x_call = prepare_b12x(
                        raw, x, b12x_out, args.quant, packed_weights
                    )
                    candidates.append(("b12x", b12x_call.run))
                candidate_outputs = {
                    "inline32": out,
                    "inline64": out,
                    "native": out,
                    "packed_async": out,
                    "resident_cublas": base,
                    "expand_cublas": base,
                    "packed_expand_cublas": base,
                }
                if b12x_call is not None:
                    candidate_outputs["b12x"] = b12x_call.output
                errors = {}
                for name, fn in candidates:
                    for _ in range(3):
                        fn()
                    result = candidate_outputs[name].float()
                    errors[name] = (
                        (result - reference.float()).norm() / reference.float().norm()
                    ).item()
                    assert torch.isfinite(result).all() and errors[name] < 0.005
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
                for name, graph in graphs.items():
                    candidate_outputs[name].fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    result = candidate_outputs[name].float()
                    assert torch.isfinite(result).all()
                    assert (
                        result - reference.float()
                    ).norm() / reference.float().norm() < 0.005
                record = dict(
                    benchmark_sha256=hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                    candidate_relative_l2=errors,
                    b12x_source_sha256=b12x_source_hash,
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
                    groups=args.groups,
                    row_start=args.row_start,
                    k_start=args.k_start,
                    pad_k_to=args.pad_k_to,
                    pad_n_to=args.pad_n_to,
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
    parser.add_argument("--quant", choices=["Q8_0", "IQ2_XS", "Q6_K"], default="Q8_0")
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--row-count", type=int)
    parser.add_argument("--k-start", type=int, default=0)
    parser.add_argument("--k-count", type=int)
    parser.add_argument("--pad-k-to", type=int)
    parser.add_argument("--pad-n-to", type=int)
    parser.add_argument("--compare-b12x", action="store_true")
    parser.add_argument("--compare-q6-packed", action="store_true")
    parser.add_argument("--tensor", help="Profile or benchmark one manifest tensor")
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="One eager native call per shape, checked against FP32",
    )
    parser.add_argument("--rows", type=int, nargs="+", default=[128, 512, 1024, 2048])
    main(parser.parse_args())
