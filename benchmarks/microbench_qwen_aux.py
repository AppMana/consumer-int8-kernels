# SPDX-License-Identifier: Apache-2.0
"""Compare native CUDA Qwen normalization/preparation with compiled references."""

import argparse
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
from microbench import time_us

from consumer_int8 import ops


def norm_reference(x, weight, residual):
    summed = x.float() + residual.float() if residual is not None else x.float()
    y = summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-6)
    return (y * (weight.float() + 1)).bfloat16(), summed.bfloat16()


def main():
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
    from vllm.third_party.flash_linear_attention.ops import fused_post_conv_prep
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 16, 1024, 4096, 8192]
    )
    parser.add_argument("--convolution", action="store_true")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1])
    args = parser.parse_args()
    torch.manual_seed(927)
    compiled_norm = torch.compile(norm_reference, dynamic=True, fullgraph=True)
    result = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, rows=[])

    def measure(kind, tokens, native, reference, **metadata):
        native()
        reference()
        row = dict(kind=kind, tokens=tokens, **metadata)
        row["reference"] = time_us(reference, launches=20)
        row["native"] = time_us(native, launches=20)
        row["speedup"] = row["reference"]["median_us"] / row["native"]["median_us"]
        result["rows"].append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(row), flush=True)

    def conv_reference(
        x, weight, state, offsets, slots, initial, metadata, a, b, log_a, dt, hq
    ):
        conv = causal_conv1d_fn(
            x.T,
            weight,
            None,
            state,
            offsets,
            slots,
            initial,
            metadata=metadata,
        ).T
        return fused_post_conv_prep(conv, a, b, log_a, dt, hq, 128, 128)

    for tokens in args.tokens:
        for with_residual in (False, True):
            x = torch.randn(tokens, 5120, device="cuda", dtype=torch.bfloat16)
            weight = torch.randn(5120, device="cuda", dtype=x.dtype)
            residual = torch.randn_like(x) if with_residual else None
            out = torch.empty_like(x)
            residual_out = torch.empty_like(x) if with_residual else None
            measure(
                "gemma_norm",
                tokens,
                partial(ops.gemma_norm, x, weight, residual, out, residual_out, 1e-6),
                partial(compiled_norm, x, weight, residual),
                residual=with_residual,
            )
        for hq, hv in ((8, 24), (16, 48)):
            width = (2 * hq + hv) * 128
            x = torch.randn(tokens, width, device="cuda", dtype=torch.bfloat16)
            ab = torch.randn(tokens, hv * 2, device="cuda", dtype=x.dtype)
            a, b = ab[:, hv:], ab[:, :hv]
            log_a, dt = (torch.randn(hv, device="cuda") for _ in range(2))
            q, k = (
                torch.empty(tokens, hq, 128, device="cuda", dtype=x.dtype)
                for _ in range(2)
            )
            v = torch.empty(tokens, hv, 128, device="cuda", dtype=x.dtype)
            g, beta = (torch.empty(tokens, hv, device="cuda") for _ in range(2))
            measure(
                "gdn_post_conv",
                tokens,
                partial(ops.gdn_post_conv, x, a, b, log_a, dt, q, k, v, g, beta),
                partial(fused_post_conv_prep, x, a, b, log_a, dt, hq, 128, 128),
                key_heads=hq,
                value_heads=hv,
            )
            for concurrency in args.concurrency if args.convolution else []:
                if tokens < concurrency:
                    continue
                weight = torch.randn(width, 4, device="cuda", dtype=x.dtype) * 0.2
                state = torch.randn(
                    concurrency + 1, width, 3, device="cuda", dtype=x.dtype
                )
                reference_state = state.clone()
                offsets_cpu = (
                    torch.arange(concurrency + 1, dtype=torch.int32)
                    * tokens
                    // concurrency
                )
                max_query = int(offsets_cpu.diff().max())
                offsets = offsets_cpu.cuda()
                slots = torch.arange(
                    1, concurrency + 1, device="cuda", dtype=torch.int32
                )
                initial = torch.ones(concurrency, device="cuda", dtype=torch.bool)
                nums, batches, chunks = compute_causal_conv1d_metadata(
                    offsets_cpu, device=x.device
                )
                metadata = SimpleNamespace(
                    nums_dict=nums, batch_ptr=batches, token_chunk_offset_ptr=chunks
                )

                reference = partial(
                    conv_reference,
                    x,
                    weight,
                    reference_state,
                    offsets,
                    slots,
                    initial,
                    metadata,
                    a,
                    b,
                    log_a,
                    dt,
                    hq,
                )

                native = partial(
                    ops.gdn_conv,
                    x,
                    a,
                    b,
                    log_a,
                    dt,
                    q,
                    k,
                    v,
                    g,
                    beta,
                    weight,
                    state,
                    offsets,
                    slots,
                    initial,
                    max_query,
                )
                native()
                expected = reference()
                for actual, target in zip((q, k, v, g, beta), expected, strict=True):
                    torch.testing.assert_close(actual, target, atol=0.004, rtol=0.008)
                torch.testing.assert_close(state, reference_state, atol=0, rtol=0)
                measure(
                    "gdn_conv",
                    tokens,
                    native,
                    reference,
                    key_heads=hq,
                    value_heads=hv,
                    concurrency=concurrency,
                )


if __name__ == "__main__":
    main()
