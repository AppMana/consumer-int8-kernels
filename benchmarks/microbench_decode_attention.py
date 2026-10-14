# SPDX-License-Identifier: Apache-2.0
"""Time short-query D256 GQA6 attention with the serving INT8 cache layout."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from microbench import time_us
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode

from consumer_int8 import ops


def case(length, concurrency, queries, heads):
    torch.manual_seed(731)
    pages = (length + 31) // 32
    cache = torch.empty(
        (concurrency * pages, heads, 32, 520), device="cuda", dtype=torch.int8
    )
    k = torch.randn(
        concurrency * pages * 32, heads, 256, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    ops.kv_cache_update(k, v, cache, torch.arange(k.shape[0], device="cuda"))
    del k, v
    q = torch.randn(
        concurrency * queries, heads * 12, 256, device="cuda", dtype=torch.bfloat16
    )[:, : heads * 6]
    out = torch.empty_like(q)
    table = torch.arange(concurrency * pages, device="cuda", dtype=torch.int32).reshape(
        concurrency, pages
    )
    offsets = torch.arange(concurrency + 1, device="cuda", dtype=torch.int32) * queries
    lengths = torch.full((concurrency,), length, device="cuda", dtype=torch.int32)
    view = cache.transpose(1, 2)
    baseline = partial(
        unified_attention,
        q=q,
        k=view[..., :256],
        v=view[..., 260:516],
        out=out,
        seq_threshold_3D=8,
        num_par_softmax_segments=16,
        softmax_segm_output=torch.empty(
            (q.shape[0], q.shape[1], 16, 256), device="cuda"
        ),
        softmax_segm_max=torch.empty((q.shape[0], q.shape[1], 16), device="cuda"),
        softmax_segm_expsum=torch.empty((q.shape[0], q.shape[1], 16), device="cuda"),
        cu_seqlens_q=offsets,
        max_seqlen_q=queries,
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
        k_scale_cache=view[..., 256:260].view(torch.float32).squeeze(-1),
        v_scale_cache=view[..., 516:520].view(torch.float32).squeeze(-1),
    )
    baseline()
    expected = out.clone()
    result = dict(
        length=length,
        concurrency=concurrency,
        queries=queries,
        kv_heads=heads,
        triton_before=time_us(baseline, launches=20),
        native={},
    )
    for parts in (4, 8, 16, 32, 64, 128):
        scratch = torch.empty((q.shape[0], q.shape[1], parts, 258), device="cuda")
        call = partial(
            ops.paged_decode,
            q,
            cache,
            table,
            offsets,
            lengths,
            out,
            queries,
            1 / 16,
            scratch,
        )
        call()
        torch.testing.assert_close(out, expected, atol=0.002, rtol=0.008)
        result["native"][parts] = time_us(call, launches=20)
    result["triton_after"] = time_us(baseline, launches=20)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lengths", type=int, nargs="+", default=[8000, 16000, 32000])
    p.add_argument("--concurrencies", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--queries", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--heads", type=int, nargs="+", default=[2, 4])
    args = p.parse_args()
    rows = []
    for heads in args.heads:
        for length in args.lengths:
            for c in args.concurrencies:
                for q in args.queries:
                    row = case(length, c, q, heads)
                    rows.append(row)
                    args.output.write_text(json.dumps(rows, indent=2) + "\n")
                    print(json.dumps(row), flush=True)
