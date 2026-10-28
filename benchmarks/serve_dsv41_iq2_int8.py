# SPDX-License-Identifier: Apache-2.0
"""Explicit serving experiment: selected prefill shapes use integer IQ MMA.

Install before model loading. Requires the qualified prebuilt register-N128
C ABI library and the benchmark helper on PYTHONPATH. Not a production backend.
"""

import ctypes
import functools
import re
from dataclasses import replace
from pathlib import Path

import torch


def install(library, *, enabled=True, borrow_b12x=False, prefill_tokens=512):
    """Use enabled=False for a matched control; borrow_b12x avoids weight copies."""
    from iq2xxs_int8_moe import Int8MoE, repack
    from vllm.model_executor.layers.quantization.b12x_iq import B12xIQMoEMethod
    from vllm.v1.worker.workspace import (
        current_workspace_manager,
        is_workspace_manager_initialized,
    )

    from consumer_int8.gguf_moe import PreparedExperts, prepare_experts

    if prefill_tokens not in (512, 1024, 2048) or (
        prefill_tokens != 512 and not borrow_b12x
    ):
        raise ValueError("Larger prefill windows require borrowed weights")
    library = Path(library).resolve(strict=True)
    cdll = ctypes.CDLL(str(library))
    has_hybrid_fused = hasattr(cdll, "iq2xxs_routed_int8_b12x_hybrid1632_swiglu")
    has_register_fused = hasattr(cdll, "iq2xxs_routed_int8_b12x_register_swiglu")
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
        selected = (*range(2, 15), *range(30, 40)) if borrow_b12x else (2, 3, 4)
        if number not in selected:
            return
        if (
            any(
                s.quant_type
                not in (("IQ2_XXS", "IQ3_XXS") if borrow_b12x else ("IQ2_XXS",))
                for s in self.sources
            )
            or self.moe.swiglu_limit != 10.0
            or self.moe.swiglu_alpha not in (None, 1.0)
            or self.moe.swiglu_beta not in (None, 0.0)
            or self.moe.max_num_tokens < prefill_tokens
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
        if borrow_b12x:
            gate, up, down = self.sources
            if gate.quant_type != up.quant_type:
                raise ValueError("Gate/up formats must match")
            device = layer._b12x_iq_device.device
            if mapping is None:
                mapping = torch.arange(
                    self.moe.num_experts, device=device, dtype=torch.int32
                )
            native = PreparedExperts(
                gate_up=(),
                down=(),
                expert_map=mapping,
                first_format=gate.quant_type,
                second_format=down.quant_type,
                hidden_size=gate.columns,
                intermediate_size=gate.rows,
                expert_ids=ids,
            )
            impl = self.experts._impl
            packed = (
                (impl.w1_fp4.view(torch.int32), impl.w1_blockscale.view(torch.uint8)),
                (impl.w2_fp4.view(torch.int32), impl.w2_blockscale.view(torch.uint8)),
            )
        else:
            native = prepare_experts(
                *self.sources, ids, device=layer._b12x_iq_device.device
            )
            packed = (repack(native.gate_up), repack(native.down))
            native = replace(native, gate_up=(), down=())
        if (native.hidden_size, native.intermediate_size) != (5120, 2304):
            raise ValueError("Unqualified integer expert dimensions")
        count = 128 if borrow_b12x and number >= 30 else prefill_tokens
        use_hybrid = bool(borrow_b12x and has_hybrid_fused and count >= 512)
        use_fused_fc1 = bool(
            borrow_b12x and (use_hybrid or has_register_fused)
        )
        tile_m = 32 if use_hybrid else 16
        plan = Int8MoE(
            native,
            packed,
            library,
            count,
            6,
            tile_m,
            register_decode=True,
            fused_swiglu=True,
            masked_swiglu=True,
            hybrid1632=use_hybrid,
            external_scratch=True,
            b12x_packed=borrow_b12x,
            fused_fc1=use_fused_fc1,
        )
        device = native.device
        x = torch.zeros((count, 5120), dtype=torch.bfloat16, device=device)
        routes = torch.full((count, 6), ids[0], dtype=torch.int32, device=device)
        weights = torch.full((count, 6), 1 / 6, dtype=torch.float32, device=device)
        plan.run(x, routes, weights, self._output[:count], scratch(plan))
        self._integer_prefill_plan = plan
        self._integer_prefill_count = count
        print(
            f"Integer prefill prepared layer={number} M={count} "
            f"N128 borrowed={borrow_b12x}",
            flush=True,
        )

    @functools.wraps(original_apply)
    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        plan = getattr(self, "_integer_prefill_plan", None)
        if not enabled or plan is None or x.shape[0] != self._integer_prefill_count:
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
        output = self._output[: x.shape[0]]
        plan.run(x, topk_ids, topk_weights, output, scratch(plan))
        return output

    load._integer_prefill_experiment = True
    B12xIQMoEMethod.process_weights_after_loading = load
    B12xIQMoEMethod.apply = apply
