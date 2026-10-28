# SPDX-License-Identifier: Apache-2.0
"""Explicit serving experiment: M512 IQ2_XXS layers, b12x for other shapes.

Install before model loading. Requires the qualified prebuilt register-N128
C ABI library and the benchmark helper on PYTHONPATH. Not a production backend.
"""

import functools
import re
from dataclasses import replace
from pathlib import Path

import torch


def install(library):
    from iq2xxs_int8_moe import Int8MoE, repack
    from vllm.model_executor.layers.quantization.b12x_iq import B12xIQMoEMethod
    from vllm.v1.worker.workspace import (
        current_workspace_manager,
        is_workspace_manager_initialized,
    )

    from consumer_int8.gguf_moe import prepare_experts

    library = Path(library).resolve(strict=True)
    original_load = B12xIQMoEMethod.process_weights_after_loading
    original_apply = B12xIQMoEMethod.apply
    if getattr(original_load, "_integer_prefill_experiment", False):
        raise RuntimeError("Integer prefill experiment already installed")

    def scratch(plan):
        if not is_workspace_manager_initialized():
            raise RuntimeError("Integer serving experiment requires worker workspace")
        return tuple(
            current_workspace_manager().get_simultaneous(*plan.scratch_specs())
        )

    @functools.wraps(original_load)
    def load(self, layer):
        original_load(self, layer)
        match = re.match(r"blk\.(\d+)\.", self.sources[0].name)
        number = int(match[1]) if match else -1
        if number not in (2, 3, 4):
            return
        if (
            any(s.quant_type != "IQ2_XXS" for s in self.sources)
            or self.moe.swiglu_limit != 10.0
            or self.moe.swiglu_alpha not in (None, 1.0)
            or self.moe.swiglu_beta not in (None, 0.0)
            or self.moe.max_num_tokens < 512
            or self.moe.experts_per_token != 6
        ):
            raise ValueError("Integer experiment requires qualified IQ2 geometry")
        mapping = layer.expert_map
        if mapping is None:
            ids = tuple(range(self.moe.num_experts))
        else:
            ordered = sorted(
                (local, global_id)
                for global_id, local in enumerate(mapping.cpu().tolist())
                if local >= 0
            )
            if [local for local, _ in ordered] != list(
                range(self.moe.num_local_experts)
            ):
                raise ValueError("Expert map must cover each local slot once")
            ids = tuple(global_id for _, global_id in ordered)
        native = prepare_experts(
            *self.sources, ids, device=layer._b12x_iq_device.device
        )
        if (native.hidden_size, native.intermediate_size) != (5120, 2304):
            raise ValueError("Unqualified integer expert dimensions")
        packed = (repack(native.gate_up), repack(native.down))
        # INT8 uses repacked descriptors; retain only geometry and the expert map.
        native = replace(native, gate_up=(), down=())
        plan = Int8MoE(
            native,
            packed,
            library,
            512,
            6,
            16,
            register_decode=True,
            fused_swiglu=True,
            masked_swiglu=True,
            external_scratch=True,
        )
        device = native.device
        x = torch.zeros((512, 5120), dtype=torch.bfloat16, device=device)
        routes = torch.full((512, 6), ids[0], dtype=torch.int32, device=device)
        weights = torch.full((512, 6), 1 / 6, dtype=torch.float32, device=device)
        plan.run(x, routes, weights, self._output[:512], scratch(plan))
        self._integer_prefill_plan = plan
        print(f"Integer IQ2 prefill prepared layer={number} M=512 N128", flush=True)

    @functools.wraps(original_apply)
    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        plan = getattr(self, "_integer_prefill_plan", None)
        if plan is None or x.shape[0] != 512:
            return original_apply(
                self,
                layer,
                x,
                topk_weights,
                topk_ids,
                shared_experts,
                shared_experts_input,
            )
        if topk_ids.dtype != torch.int32 or topk_weights.dtype != torch.float32:
            raise ValueError("Integer IQ2 requires int32 IDs and float32 weights")
        output = self._output[:512]
        plan.run(x, topk_ids, topk_weights, output, scratch(plan))
        return output

    load._integer_prefill_experiment = True
    B12xIQMoEMethod.process_weights_after_loading = load
    B12xIQMoEMethod.apply = apply
