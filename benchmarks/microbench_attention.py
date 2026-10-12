# SPDX-License-Identifier: Apache-2.0
"""Compare packed attention and native cache conversion plus existing FA2."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from microbench import time_us
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode
from vllm.vllm_flash_attn import flash_attn_varlen_func

from consumer_int8 import ops


def case(length, heads, concurrency, tokens=1024):
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

    calls = {
        "triton_packed": baseline,
        "cuda_packed": partial(
            ops.paged_attention, q, cache, table, offsets, lengths, out, qlen, 1 / 16
        ),
        "cuda_conversion_and_fa2": converted_fa2,
    }
    row = dict(
        kv_length=length,
        query_tokens=tokens,
        kv_heads=heads,
        concurrency=concurrency,
        bf16_workspace_bytes=keys.numel() * 4,
        timings={},
    )
    for name, call in calls.items():
        call()
        torch.cuda.synchronize()
        assert torch.isfinite(out).all()
        row["timings"][name] = dict(
            **time_us(call, launches=3, repeats=5),
            relative_l2=(
                (out.float() - expected.float()).norm() / expected.float().norm()
            ).item(),
            max_error=(out.float() - expected.float()).abs().max().item(),
        )
    row["timings"]["conversion_only"] = time_us(unpack, launches=3, repeats=5)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[8192, 32768])
    parser.add_argument("--heads", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    args = parser.parse_args()
    torch.manual_seed(20261012)
    result = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, rows=[])
    for heads in args.heads:
        for length in args.lengths:
            for concurrency in args.concurrency:
                row = case(length, heads, concurrency)
                result["rows"].append(row)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
