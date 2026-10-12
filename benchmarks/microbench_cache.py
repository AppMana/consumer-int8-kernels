# SPDX-License-Identifier: Apache-2.0
"""Compare packed INT8 KV writers with Qwen's strided QKV projection outputs."""

import argparse
import json
from functools import partial

import torch
from microbench import time_us
from vllm.v1.attention.backends.utils import KVQuantMode
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_per_token_head_quant,
)

from consumer_int8 import ops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(20261011)
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "rows": [],
    }
    for heads in (2, 4):
        for layout in ("HND", "NHD"):
            for tokens in (1, 2, 4, 8, 256, 1024):
                # Q has 6x as many heads as K, and also carries its output gate.
                packed = torch.randn(
                    tokens, 14 * heads, 256, device="cuda", dtype=torch.bfloat16
                )
                key, value = packed[:, 12 * heads : 13 * heads], packed[:, 13 * heads :]
                shape = (
                    (64, heads, 32, 520) if layout == "HND" else (64, 32, heads, 520)
                )
                native_cache = torch.full(shape, 55, device="cuda", dtype=torch.int8)
                reference = native_cache.clone()
                if layout == "NHD":
                    native_cache, reference = (
                        native_cache.transpose(1, 2),
                        reference.transpose(1, 2),
                    )
                slots = torch.arange(tokens, device="cuda", dtype=torch.int64)
                # Include a padded token without altering the launch geometry.
                if tokens > 1:
                    slots[-1] = -1
                views = reference.transpose(1, 2)
                key_cache, value_cache = views[..., :256], views[..., 260:516]
                k_scale = views[..., 256:260].view(torch.float32).squeeze(-1)
                v_scale = views[..., 516:520].view(torch.float32).squeeze(-1)
                native = partial(ops.kv_cache_update, key, value, native_cache, slots)
                triton = partial(
                    triton_reshape_and_cache_flash_per_token_head_quant,
                    key,
                    value,
                    key_cache,
                    value_cache,
                    k_scale,
                    v_scale,
                    slots,
                    kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
                )
                native()
                triton()
                torch.testing.assert_close(native_cache, reference, atol=0, rtol=0)
                row = dict(
                    tokens=tokens,
                    heads=heads,
                    head_dim=256,
                    layout=layout,
                    exact_cache_match=True,
                    native=time_us(native),
                    triton=time_us(triton),
                )
                result["rows"].append(row)
                with open(args.output, "w") as stream:
                    json.dump(result, stream, indent=2)
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
