# SPDX-License-Identifier: Apache-2.0
"""Explicit diagnostic hook for recording eager serving MoE inputs.

Call install(directory) in the serving worker before model load. Create directory/
ARMED only after warmup. This synchronizes and copies inputs to CPU: never use
its request timings as benchmark results. CUDA graph capture is skipped.
"""

import functools
import re
from pathlib import Path

import torch


def install(directory, layers=(2, 3, 4, 30), counts=(128, 512)):
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.model_executor.layers.quantization.b12x_iq import B12xIQMoEMethod

    original = B12xIQMoEMethod.apply
    if getattr(original, "_captures_serving_inputs", False):
        raise RuntimeError("Serving capture hook already installed")
    directory = Path(directory)
    saved = set()

    @functools.wraps(original)
    def apply(self, layer, x, weights, ids, shared_experts, shared_experts_input):
        count = x.shape[0]
        match = re.match(r"blk\.(\d+)\.", self.sources[0].name)
        number = int(match[1]) if match else -1
        key = (number, count)
        if (
            count in counts
            and number in layers
            and key not in saved
            and not torch.cuda.is_current_stream_capturing()
            and (directory / "ARMED").exists()
        ):
            rank = get_tensor_model_parallel_rank()
            path = directory / f"rank{rank}-layer{number}-tokens{count}.pt"
            if path.exists():
                raise FileExistsError(path)
            fixture = dict(
                revision="60ad3651641b5c97679c807e064ab1d7a3ca1455",
                layer=number,
                tokens=count,
                rank=rank,
                x=x.detach().cpu(),
                ids=ids.detach().cpu(),
                weights=weights.detach().cpu(),
            )
            temporary = path.with_suffix(".tmp")
            torch.save(fixture, temporary)
            temporary.rename(path)
            saved.add(key)
        return original(
            self, layer, x, weights, ids, shared_experts, shared_experts_input
        )

    apply._captures_serving_inputs = True
    B12xIQMoEMethod.apply = apply
