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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1024, 4096, 8192])
    args = parser.parse_args()
    torch.manual_seed(741)
    result = dict(gpu=torch.cuda.get_device_name(), rows=[])
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
