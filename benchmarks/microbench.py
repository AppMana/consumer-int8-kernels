# SPDX-License-Identifier: Apache-2.0
"""Compare AOT candidates with the installed vLLM kernels on Qwen geometries."""

import argparse
import json
import statistics
from functools import partial

import torch

from consumer_int8 import ops
from consumer_int8.quantization import quantize_channel


def time_us(call, repeats=7, launches=100):
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(launches):
            call()
    timings = []
    for _ in range(repeats):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000 / launches)
    return {
        "median_us": statistics.median(timings),
        "min_us": min(timings),
        "max_us": max(timings),
    }


def _quantized_mm(implementation, x, a, sa, weights, scales, *outputs):
    ops.quantize_rows(x, a, sa)
    implementation(a, weights, sa, scales, *outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kind", choices=["gdn", "linear", "all"], default="all")
    args = parser.parse_args()
    torch.manual_seed(20261011)
    results = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "rows": [],
    }

    def record(row):
        results["rows"].append(row)
        with open(args.output, "w") as stream:
            json.dump(results, stream, indent=2)
        print(json.dumps(row), flush=True)

    if args.kind in ("gdn", "all"):
        from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
            fused_recurrent_gated_delta_rule_fwd,
            fused_recurrent_gated_delta_rule_packed_decode,
        )

        for hq, hv in [(16, 48), (8, 24)]:
            for batch in (1, 2, 4):
                q = torch.randn(batch, hq, 128, device="cuda", dtype=torch.bfloat16)
                k = torch.randn_like(q)
                v = torch.randn(batch, hv, 128, device="cuda", dtype=torch.bfloat16)
                g = -torch.rand(batch, hv, device="cuda")
                beta = torch.rand_like(g)
                state = torch.zeros(batch + 1, hv, 128, 128, device="cuda")
                slots = torch.arange(1, batch + 1, device="cuda", dtype=torch.int64)
                offsets = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
                out = torch.empty_like(v)

                native = partial(ops.gdn_decode, q, k, v, g, beta, state, slots, out)
                triton = partial(
                    fused_recurrent_gated_delta_rule_fwd,
                    q[None],
                    k[None],
                    v[None],
                    g[None],
                    beta[None],
                    128**-0.5,
                    state,
                    inplace_final_state=True,
                    cu_seqlens=offsets,
                    ssm_state_indices=slots,
                    use_qk_l2norm_in_kernel=True,
                )

                native_result = time_us(native)
                state.zero_()
                triton_result = time_us(triton)
                qkv = torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=1)
                ba = torch.randn(batch, 2 * hv, device="cuda", dtype=torch.bfloat16)
                raw_b, raw_a = ba[:, :hv], ba[:, hv:]
                log_a = torch.randn(hv, device="cuda")
                dt_bias = torch.randn(hv, device="cuda", dtype=torch.bfloat16)
                packed_native = partial(
                    ops.gdn_packed_decode,
                    qkv,
                    raw_a,
                    raw_b,
                    log_a,
                    dt_bias,
                    state,
                    slots,
                    out,
                )
                packed_triton = partial(
                    fused_recurrent_gated_delta_rule_packed_decode,
                    mixed_qkv=qkv,
                    a=raw_a,
                    b=raw_b,
                    A_log=log_a,
                    dt_bias=dt_bias,
                    scale=128**-0.5,
                    initial_state=state,
                    out=out.unsqueeze(1),
                    ssm_state_indices=slots,
                    use_qk_l2norm_in_kernel=True,
                )
                state.zero_()
                packed_native_result = time_us(packed_native)
                state.zero_()
                packed_triton_result = time_us(packed_triton)
                record(
                    {
                        "op": "gdn_decode",
                        "batch": batch,
                        "key_heads": hq,
                        "value_heads": hv,
                        "native": native_result,
                        "triton": triton_result,
                        "native_packed": packed_native_result,
                        "triton_packed": packed_triton_result,
                    }
                )

    if args.kind in ("linear", "all"):
        from vllm import _custom_ops as vops

        for n, k in [
            (16384, 5120),  # GDN QKVZ, PP2
            (8192, 5120),  # GDN QKVZ, TP2
            (14336, 5120),  # Full-attention QKV including Q gate, PP2
            (7168, 5120),  # Full-attention QKV, TP2
            (34816, 5120),  # MLP gate/up, PP2
            (17408, 5120),  # MLP gate/up, TP2
            (5120, 17408),  # MLP down, PP2
            (5120, 8704),  # MLP down, TP2 (output dimension stays replicated)
            (5120, 6144),  # Attention output, PP2
            (5120, 3072),  # Attention output, TP2
            (5120, 10240),  # MTP fusion, PP2
            (2560, 10240),  # MTP fusion, TP2 (column parallel + gather)
        ]:
            weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.01
            codes, scale = quantize_channel(weight)
            signed = (codes.to(torch.int16) - 128).to(torch.int8)
            packed, packed_scale, _ = vops.allspark_repack_weight(
                codes.T.contiguous(), scale[None].contiguous(), None, False
            )
            for m in (1, 2, 4, 16, 256, 1024):
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                a = torch.empty_like(x, dtype=torch.int8)
                sa = torch.empty(m, device="cuda")
                out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                workspace = torch.empty(m, n, device="cuda", dtype=torch.int32)

                native = partial(
                    _quantized_mm,
                    ops.channel_mm,
                    x,
                    a,
                    sa,
                    codes,
                    scale,
                    out,
                )
                allspark = partial(
                    vops.allspark_w8a16_gemm,
                    a=x,
                    b_qweight=packed,
                    b_scales=packed_scale,
                    b_qzeros=None,
                    n=n,
                    group_size=-1,
                    sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
                    sm_version=86,
                    CUBLAS_M_THRESHOLD=1024,
                    has_zp=False,
                    n32k16_reorder=True,
                )
                cutlass = partial(
                    _quantized_mm,
                    ops.cutlass_channel_mm,
                    x,
                    a,
                    sa,
                    signed,
                    scale,
                    workspace,
                    out,
                )

                record(
                    {
                        "op": "linear",
                        "m": m,
                        "n": n,
                        "k": k,
                        "native_w8a8": time_us(native),
                        "cutlass_w8a8": time_us(cutlass),
                        "allspark_w8a16": time_us(allspark),
                        **(
                            {
                                "canonical_gemv_w8a16": time_us(
                                    partial(ops.channel_gemv, x, signed, scale, out)
                                )
                            }
                            if m <= 8
                            else {}
                        ),
                    }
                )
            del weight, codes, scale, packed, packed_scale


if __name__ == "__main__":
    main()
