# SPDX-License-Identifier: Apache-2.0
"""Compare packed attention and native cache conversion plus existing FA2."""

import argparse
import hashlib
import importlib
import itertools
import json
import os
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch
from microbench import time_us
from run_matrix import GpuMonitor
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode
from vllm.vllm_flash_attn import flash_attn_varlen_func

from consumer_int8 import ops


def triton_override(call, config):
    module = importlib.import_module("vllm.v1.attention.ops.triton_unified_attention")
    original = module.kernel_unified_attention

    class Launch:
        def __getitem__(self, grid):
            def run(**kwargs):
                m, n, warps, stages = config
                block_q = m // kwargs["num_queries_per_kv"]
                kwargs.update(
                    BLOCK_M=m,
                    BLOCK_Q=block_q,
                    TILE_SIZE=n,
                    num_warps=warps,
                    num_stages=stages,
                )
                grid = (
                    kwargs["query_ptr"].shape[0] // block_q + kwargs["num_seqs"],
                    kwargs["num_query_heads"] // kwargs["num_queries_per_kv"],
                )
                return original[grid](**kwargs)

            return run

    with patch.object(module, "kernel_unified_attention", Launch()):
        call()


def case(
    length,
    heads,
    concurrency,
    tokens=1024,
    schedules=(),
    triton_configs=(),
    skip_packed=False,
    progress=None,
    profile=None,
    reverse_order=False,
):
    assert tokens % concurrency == 0 and length % 32 == 0
    qlen = tokens // concurrency
    q = torch.randn(tokens, heads * 12, 256, device="cuda", dtype=torch.bfloat16)
    q = q[:, : heads * 6]
    cache = torch.empty(
        concurrency * length // 32, heads, 32, 520, device="cuda", dtype=torch.int8
    )
    k = torch.randn(
        concurrency * length, heads, 256, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    ops.kv_cache_update(k, v, cache, torch.arange(k.shape[0], device="cuda"))
    del k, v
    table = torch.arange(cache.shape[0], device="cuda", dtype=torch.int32).reshape(
        concurrency, -1
    )
    offsets = torch.arange(concurrency + 1, device="cuda", dtype=torch.int32) * qlen
    lengths = torch.full((concurrency,), length, device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)
    views = cache.transpose(1, 2)
    baseline = partial(
        unified_attention,
        q=q,
        k=views[..., :256],
        v=views[..., 260:516],
        out=out,
        cu_seqlens_q=offsets,
        max_seqlen_q=qlen,
        seqused_k=lengths,
        max_seqlen_k=length,
        softmax_scale=1 / 16,
        causal=True,
        window_size=(-1, -1),
        block_table=table,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
        k_scale_cache=views[..., 256:260].view(torch.float32).squeeze(-1),
        v_scale_cache=views[..., 516:520].view(torch.float32).squeeze(-1),
    )
    baseline()
    expected = out.clone()
    keys = torch.empty(
        concurrency, length, heads, 256, device="cuda", dtype=torch.bfloat16
    )
    values = torch.empty_like(keys)
    scratch_table = torch.arange(concurrency, device="cuda", dtype=torch.int32).reshape(
        -1, 1
    )
    unpack = partial(ops.unpack_kv_cache, cache, table, lengths, keys, values)
    fa2 = partial(
        flash_attn_varlen_func,
        q=q,
        k=keys,
        v=values,
        out=out,
        cu_seqlens_q=offsets,
        max_seqlen_q=qlen,
        seqused_k=lengths,
        max_seqlen_k=length,
        block_table=scratch_table,
        softmax_scale=1 / 16,
        causal=True,
        fa_version=2,
    )

    def converted_fa2():
        unpack()
        fa2()

    key_offsets = (
        torch.arange(concurrency + 1, device="cuda", dtype=torch.int32) * length
    )
    lse = torch.empty(q.shape[1], tokens, device="cuda", dtype=torch.float32)

    def dense_fa2():
        unpack()
        torch.ops._vllm_fa2_C.varlen_fwd(
            q,
            keys.flatten(0, 1),
            values.flatten(0, 1),
            out,
            offsets,
            key_offsets,
            lengths,
            None,
            None,
            None,
            qlen,
            length,
            0.0,
            1 / 16,
            False,
            True,
            -1,
            -1,
            0.0,
            False,
            0,
            None,
        )

    def native_fa2(schedule):
        unpack()
        ops.fa2_prefill(
            q, keys, values, offsets, lengths, out, lse, qlen, 1 / 16, schedule
        )

    converted_fa2()
    fa2_expected = out.clone()
    calls = {
        "triton_packed": baseline,
        "cuda_packed": partial(
            ops.paged_attention, q, cache, table, offsets, lengths, out, qlen, 1 / 16
        ),
        "cuda_conversion_and_fa2": converted_fa2,
    }
    if skip_packed:
        calls.pop("cuda_packed")
    calls["cuda_conversion_and_fa2_dense"] = dense_fa2
    for schedule in schedules:
        calls[f"cuda_conversion_and_fa2_schedule_{schedule}"] = partial(
            native_fa2, schedule
        )
    for config in triton_configs:
        calls["triton_" + "_".join(map(str, config))] = partial(
            triton_override, baseline, config
        )
    row = dict(
        kv_length=length,
        query_tokens=tokens,
        kv_heads=heads,
        concurrency=concurrency,
        bf16_workspace_bytes=keys.numel() * 4,
        timings={},
    )
    if profile is not None:
        call = calls[profile]
        call()
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        call()
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        return {"profiled": profile}
    if reverse_order:
        calls = dict(reversed(list(calls.items())))
    for name, call in calls.items():
        try:
            call()
            torch.cuda.synchronize()
        except Exception as exc:
            if "out of resource" not in str(exc).lower():
                raise
            row["timings"][name] = {"rejected": str(exc)}
            continue
        assert torch.isfinite(out).all()
        relative_l2 = (
            (out.float() - expected.float()).norm() / expected.float().norm()
        ).item()
        assert relative_l2 < 0.01, (name, relative_l2)
        row["timings"][name] = dict(
            **time_us(call, launches=3, repeats=5),
            relative_l2=(
                (out.float() - expected.float()).norm() / expected.float().norm()
            ).item(),
            max_error=(out.float() - expected.float()).abs().max().item(),
            fa2_relative_l2=(
                (out.float() - fa2_expected.float()).norm()
                / fa2_expected.float().norm()
            ).item(),
            fa2_bitwise_equal=torch.equal(out, fa2_expected),
        )
        print(json.dumps({"implementation": name, **row["timings"][name]}), flush=True)
        if progress is not None:
            progress(row)
    row["timings"]["conversion_only"] = time_us(unpack, launches=3, repeats=5)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[8192, 32768])
    parser.add_argument("--heads", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--query-tokens", type=int, nargs="+", default=[1024])
    parser.add_argument(
        "--native-schedules", type=int, choices=range(9), nargs="*", default=[]
    )
    parser.add_argument("--triton-sweep", action="store_true")
    parser.add_argument("--skip-packed", action="store_true")
    parser.add_argument(
        "--triton-config", nargs=4, type=int, action="append", default=[]
    )
    parser.add_argument("--profile")
    parser.add_argument("--reverse-order", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(20261012)
    properties = torch.cuda.get_device_properties(0)
    result = dict(
        gpu=properties.name,
        sm_count=properties.multi_processor_count,
        capability=torch.cuda.get_device_capability(),
        torch=torch.__version__,
        binary_sha256=hashlib.sha256(Path(ops._C.__file__).read_bytes()).hexdigest(),
        attention_binary_sha256=hashlib.sha256(
            Path(ops._fa2.__file__).read_bytes()
        ).hexdigest(),
        rows=[],
        status="running",
    )
    configs = (
        list(itertools.product([16, 32, 64, 128], [32, 64], [4, 8], [2]))
        if args.triton_sweep
        else []
    )

    configs.extend(args.triton_config)
    monitor = GpuMonitor(os.getpgrp(), args.output.with_suffix(".gpu.jsonl"))
    monitor.start()

    def progress(row):
        result["rows"][-1] = row
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    try:
        for tokens, heads, length, concurrency in itertools.product(
            args.query_tokens, args.heads, args.lengths, args.concurrency
        ):
            result["rows"].append({})
            row = case(
                length,
                heads,
                concurrency,
                tokens,
                args.native_schedules,
                configs,
                args.skip_packed,
                progress,
                args.profile,
                args.reverse_order,
            )
            result["rows"][-1] = row
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        raise
    finally:
        monitor.done.set()
        monitor.join()
    result["interference"] = monitor.problems
    result["status"] = "complete" if not monitor.problems else "interference"
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if monitor.problems:
        raise RuntimeError("Foreign GPU workload invalidates timing; see telemetry")


if __name__ == "__main__":
    main()
