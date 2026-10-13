# SPDX-License-Identifier: Apache-2.0
"""Measure activation quantization on Qwen TP2 and PP2 projection widths."""

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

import torch
from microbench import time_us

from consumer_int8 import ops


@torch.compile(fullgraph=True, dynamic=True)
def compiled_swiglu(x):
    k = x.shape[-1] // 2
    return torch.nn.functional.silu(x[:, :k]) * x[:, k:]


def benchmark_swiglu(args, result):
    for m in args.rows:
        for k in (8704, 17408):
            x = torch.randn(m, 2 * k, device="cuda", dtype=torch.bfloat16) * 3
            codes = torch.empty(m, k, device="cuda", dtype=torch.int8)
            scales = torch.empty(m, device="cuda")
            ref_codes, ref_scales = torch.empty_like(codes), torch.empty_like(scales)

            def reference(x=x, ref_codes=ref_codes, ref_scales=ref_scales):
                ops.quantize_rows(compiled_swiglu(x), ref_codes, ref_scales)

            candidate = partial(ops.silu_mul_quantize, x, codes, scales)
            reference()
            candidate()
            torch.testing.assert_close(codes, ref_codes, atol=0, rtol=0)
            torch.testing.assert_close(scales, ref_scales, atol=0, rtol=0)
            before = time_us(reference, repeats=5, launches=30)
            fused = time_us(candidate, repeats=5, launches=30)
            after = time_us(reference, repeats=5, launches=30)
            row = dict(
                m=m,
                k=k,
                codes_equal=True,
                scales_equal=True,
                before=before,
                fused=fused,
                after=after,
                speedup=(before["median_us"] + after["median_us"])
                / (2 * fused["median_us"]),
            )
            result["rows"].append(row)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1024, 4096, 8192])
    parser.add_argument("--silu", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(741)
    result = dict(gpu=torch.cuda.get_device_name(), rows=[])
    if args.silu:
        benchmark_swiglu(args, result)
        return
    for m in args.rows:
        for k in (3072, 5120, 6144, 8704, 17408):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            codes = torch.empty_like(x, dtype=torch.int8)
            scales = torch.empty(m, device="cuda")
            call = partial(ops.quantize_rows, x, codes, scales)
            call()
            expected_scale = (x.float().abs().amax(-1) / 127).clamp_min(1e-20)
            expected = (x.float() / expected_scale[:, None]).round().clamp(-127, 127)
            torch.testing.assert_close(scales, expected_scale, rtol=2e-7, atol=0)
            torch.testing.assert_close(codes, expected.to(torch.int8), rtol=0, atol=1)
            row = dict(
                m=m,
                k=k,
                codes_sha256=hashlib.sha256(codes.cpu().numpy().tobytes()).hexdigest(),
                scales_sha256=hashlib.sha256(
                    scales.cpu().numpy().tobytes()
                ).hexdigest(),
                **time_us(call, launches=30),
            )
            result["rows"].append(row)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
