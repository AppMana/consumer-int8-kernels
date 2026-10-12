# SPDX-License-Identifier: Apache-2.0
"""Compare the chunked CUDA and FLA paths on Qwen TP2/PP2 head counts."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from microbench import time_us

from consumer_int8 import ops


def main():
    from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20261011)
    result = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, rows=[])
    for hq, hv in ((8, 24), (16, 48)):
        for tokens in (256, 1024, 8192, 32768):
            q = torch.randn(tokens, hq, 128, device="cuda")
            k = torch.randn_like(q)
            q = (q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()).bfloat16()
            k = (k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()).bfloat16()
            v = torch.randn(tokens, hv, 128, device="cuda", dtype=torch.bfloat16)
            g = -torch.rand(tokens, hv, device="cuda") * 0.05
            beta = torch.rand_like(g)
            initial = torch.zeros(hv, 128, 128, device="cuda")
            final = torch.empty_like(initial)
            out = torch.empty_like(v)
            cumulative = torch.empty_like(g)
            w, u, v_new = (torch.empty_like(v) for _ in range(3))
            h_chunk = torch.empty(
                tokens // 64, hv, 128, 128, device="cuda", dtype=torch.bfloat16
            )
            native = partial(
                ops.gdn_prefill,
                q,
                k,
                v,
                g,
                beta,
                initial,
                final,
                out,
                cumulative,
                w,
                u,
                v_new,
                h_chunk,
                torch.cuda.get_device_properties(0).multi_processor_count,
            )
            reference = partial(
                chunk_gated_delta_rule,
                q[None],
                k[None],
                v[None],
                g[None],
                beta[None],
                initial_state=initial[None],
                output_final_state=True,
                use_qk_l2norm_in_kernel=False,
            )
            native()
            expected, expected_state = reference()
            errors = {}
            for name, actual, target in (
                ("output", out, expected[0]),
                ("state", final, expected_state[0]),
            ):
                error = (actual.float() - target.float()).norm() / target.float().norm()
                errors[name + "_relative_l2"] = error.item()
            row = dict(tokens=tokens, key_heads=hq, value_heads=hv, **errors)
            row["native"] = time_us(native, launches=10)
            row["fla"] = time_us(reference, launches=10)
            result["rows"].append(row)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(row), flush=True)
            del native, reference, expected, expected_state
    assert all(
        row[key] < 0.007
        for row in result["rows"]
        for key in ("output_relative_l2", "state_relative_l2")
    )


if __name__ == "__main__":
    main()
