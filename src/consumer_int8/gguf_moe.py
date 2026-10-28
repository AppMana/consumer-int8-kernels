# SPDX-License-Identifier: Apache-2.0
"""Bounded GGUF expert loading and caller-owned native CUDA MoE execution."""

import math
from dataclasses import dataclass

import torch

from . import gguf

_PAIRS = {
    ("IQ2_XS", "MXFP4"),
    ("IQ2_XXS", "IQ2_XXS"),
    ("IQ2_XXS", "IQ3_XXS"),
    ("IQ2_S", "MXFP4"),
    ("IQ2_S", "IQ3_S"),
    ("IQ3_XXS", "IQ3_S"),
    ("IQ3_XXS", "IQ3_XXS"),
}


def _pack(raw, kind):
    if kind == "MXFP4":
        return gguf.pack_mxfp4(raw)
    if kind == "IQ2_XS":
        return gguf.pack_iq2_xs(raw)
    if kind.endswith("_XXS"):
        return gguf.pack_iq_xxs(raw, int(kind[2]))
    return gguf.pack_iq_s(raw, int(kind[2]))


@dataclass(frozen=True)
class PreparedExperts:
    gate_up: tuple[torch.Tensor, ...]
    down: tuple[torch.Tensor, ...]
    expert_map: torch.Tensor
    first_format: str
    second_format: str
    hidden_size: int
    intermediate_size: int
    expert_ids: tuple[int, ...]

    @property
    def device(self):
        return self.expert_map.device

    @property
    def storage_bytes(self):
        return sum(
            t.numel() * t.element_size()
            for t in (*self.gate_up, *self.down, self.expert_map)
        )


def prepare_experts(gate, up, down, expert_ids, *, device):
    """Stream descriptors with read_expert(id) into compact local storage.

    Source descriptors expose experts/rows/columns/quant_type and return CPU
    uint8 [N,K/block,bytes] blocks. Only one expert triple and its packed copy
    are staged at a time. Global expert IDs follow the supplied local order.
    """
    ids = tuple(int(e) for e in expert_ids)
    if (
        not ids
        or len(set(ids)) != len(ids)
        or any(e < 0 or e >= gate.experts for e in ids)
    ):
        raise ValueError("Expert IDs must be nonempty, unique and in range")
    h, intermediate = gate.columns, gate.rows
    if (up.experts, up.rows, up.columns, up.quant_type) != (
        gate.experts,
        intermediate,
        h,
        gate.quant_type,
    ) or (down.experts, down.rows, down.columns) != (gate.experts, h, intermediate):
        raise ValueError("Gate/up/down geometry or formats disagree")
    if (
        (gate.quant_type, down.quant_type) not in _PAIRS
        or h % 256
        or intermediate % 256
    ):
        raise ValueError("Unsupported native IQ pair or geometry")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Native IQ preparation requires CUDA")
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Expert preparation must precede graph capture")
        if torch.cuda.get_device_capability(device) < (8, 6):
            raise ValueError("Native IQ requires SM86 or later")
        resident13 = resident2 = None
        for local, global_id in enumerate(ids):
            raw13 = torch.cat((gate.read_expert(global_id), up.read_expert(global_id)))
            raw2 = down.read_expert(global_id)
            one13 = _pack(raw13.unsqueeze(0).to(device), gate.quant_type)
            one2 = _pack(raw2.unsqueeze(0).to(device), down.quant_type)
            if resident13 is None:
                resident13 = tuple(
                    torch.empty((len(ids), *t.shape[1:]), dtype=t.dtype, device=device)
                    for t in one13
                )
                resident2 = tuple(
                    torch.empty((len(ids), *t.shape[1:]), dtype=t.dtype, device=device)
                    for t in one2
                )
            for destination, source in zip(
                (*resident13, *resident2), (*one13, *one2), strict=True
            ):
                destination[local : local + 1].copy_(source)
            del raw13, raw2, one13, one2
        mapping = torch.full((gate.experts,), -1, dtype=torch.int32)
        mapping[list(ids)] = torch.arange(len(ids), dtype=torch.int32)
        return PreparedExperts(
            resident13,
            resident2,
            mapping.to(device),
            gate.quant_type,
            down.quant_type,
            h,
            intermediate,
            ids,
        )


@dataclass(frozen=True)
class ExecutionPlan:
    experts: PreparedExperts
    max_tokens: int
    topk: int
    tile_rows: int = 16
    swiglu_limit: float = 10.0

    def __post_init__(self):
        if self.max_tokens <= 0 or self.topk <= 0 or self.tile_rows not in (16, 32):
            raise ValueError("Invalid native IQ capacity/topk/tile")
        if not math.isfinite(self.swiglu_limit) or self.swiglu_limit <= 0:
            raise ValueError("SwiGLU limit must be finite and positive")
        if (self.route_capacity + self.tile_rows - 1) // self.tile_rows > 65535:
            raise ValueError("Native IQ task capacity exceeds CUDA grid")

    @property
    def route_capacity(self):
        return self.max_tokens * self.topk + len(self.experts.expert_ids) * (
            self.tile_rows - 1
        )

    def scratch_specs(self):
        """Independent simultaneous views; output storage belongs to the runner."""
        r, e, k = self.route_capacity, len(self.experts.expert_ids), self.topk
        h, intermediate = self.experts.hidden_size, self.experts.intermediate_size
        return (
            ((e, 2), torch.int32),
            ((r,), torch.int32),
            ((r,), torch.float32),
            ((self.max_tokens, k), torch.int32),
            (((r + self.tile_rows - 1) // self.tile_rows, 2), torch.int32),
            ((r,), torch.int32),
            ((r, 2 * intermediate), torch.bfloat16),
            ((r, intermediate), torch.bfloat16),
            ((r, h), torch.bfloat16),
        )

    def allocate_scratch(self):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Scratch allocation must precede capture")
        return tuple(
            torch.empty(shape, dtype=dtype, device=self.experts.device)
            for shape, dtype in self.scratch_specs()
        )

    def run(self, x, topk_ids, topk_weights, out, scratch):
        """Use fresh arena views; no cached workspace addresses or host sync.

        All scratch views must be disjoint from inputs/weights/output and each
        other. Identity indices are refreshed every invocation because other
        sequential operators may have overwritten the shared workspace arena.
        """
        m = x.shape[0]
        if x.shape != (m, self.experts.hidden_size) or m > self.max_tokens:
            raise ValueError("Activation shape exceeds native IQ plan")
        if topk_ids.shape != (m, self.topk) or topk_weights.shape != topk_ids.shape:
            raise ValueError("Routing shape differs from native IQ plan")
        if (
            out.shape != x.shape
            or x.dtype != torch.bfloat16
            or out.dtype != torch.bfloat16
        ):
            raise ValueError("Native IQ input/output must match BF16 [T,H]")
        if len(scratch) != len(self.scratch_specs()):
            raise ValueError("Invalid native IQ scratch count")
        for t, (shape, dtype) in zip(scratch, self.scratch_specs(), strict=True):
            if (
                tuple(t.shape) != shape
                or t.dtype != dtype
                or t.device != self.experts.device
                or not t.is_contiguous()
            ):
                raise ValueError("Invalid native IQ scratch view")
        (
            info,
            tokens,
            weights,
            inverse,
            tasks,
            identity,
            projected,
            hidden,
            contributions,
        ) = scratch
        torch.arange(
            self.route_capacity, device=identity.device, dtype=torch.int32, out=identity
        )
        gguf.plan_routes(
            topk_ids,
            topk_weights,
            self.experts.expert_map,
            info,
            tokens,
            weights,
            inverse[:m],
            tasks,
            self.tile_rows,
        )
        gguf.mixed_iq_moe(
            x,
            self.experts.gate_up,
            self.experts.down,
            tokens,
            identity,
            tasks,
            weights,
            inverse[:m],
            projected,
            hidden,
            contributions,
            out,
            tile_rows=self.tile_rows,
            swiglu_limit=self.swiglu_limit,
            first_format=self.experts.first_format,
            second_format=self.experts.second_format,
        )
