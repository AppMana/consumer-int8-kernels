# SPDX-License-Identifier: Apache-2.0
"""Benchmark-only complete IQ2_XXS chain using prebuilt INT8 C ABI kernels."""

import ctypes

import torch

from consumer_int8 import gguf
from consumer_int8.gguf_moe import ExecutionPlan


def repack(packed):
    codes, bases, metadata = packed
    e, n, code_bytes = codes.shape
    k = code_bytes * 8
    chunks = codes.reshape(e, n, k // 128, 4, 4)
    words = metadata.contiguous().view(torch.uint8).reshape_as(chunks)
    descriptors = torch.cat((chunks, words), -1).permute(0, 2, 1, 3, 4).contiguous()
    scales = bases[..., : k // 256].transpose(1, 2).contiguous()
    assert descriptors.numel() + scales.numel() * 2 == e * n * (k // 256) * 66
    return descriptors, scales


class Int8MoE:
    def __init__(
        self,
        native,
        packed,
        library,
        max_tokens,
        topk,
        tile,
        activation_block=32,
        down_activation_block=None,
        register_decode=False,
        fused_swiglu=False,
        masked_swiglu=False,
        accumulate256=False,
        hybrid1632=False,
    ):
        if (native.first_format, native.second_format) != ("IQ2_XXS", "IQ2_XXS"):
            raise ValueError("Complete INT8 candidate only supports IQ2_XXS pairs")
        if hybrid1632 and not (
            tile == 32
            and register_decode
            and fused_swiglu
            and masked_swiglu
            and not accumulate256
            and activation_block == 32
            and down_activation_block in (None, 32)
        ):
            raise ValueError("Hybrid16/32 requires masked fused per32 path and tile32")
        self.native = native
        self.packed = packed
        # Benchmark C ABI uses tile=0 for disjoint sparse8/dense16 task arrays.
        if tile in (0, 8) and not (register_decode and fused_swiglu and masked_swiglu):
            raise ValueError("Small-row experiment requires masked fused register path")
        self.tile = tile
        self.plan = ExecutionPlan(
            native, max_tokens, topk, tile_rows=16 if tile in (0, 8) else tile
        )
        self.scratch = self.plan.allocate_scratch()
        r = self.plan.route_capacity
        self.quant = tuple(
            (
                torch.empty(rows, width, device=native.device, dtype=torch.int8),
                torch.empty(rows, width // 32, device=native.device),
            )
            for rows, width in (
                (max_tokens, native.hidden_size),
                (r, native.intermediate_size),
            )
        )
        self.lib = ctypes.CDLL(str(library.resolve()))
        self.small_tasks = None
        if tile in (0, 8) or hybrid1632:
            tasks = self.scratch[4]
            if tasks.shape[0] > 32767:
                raise ValueError("Retiled task capacity exceeds CUDA grid")
            self.small_tasks = torch.empty(
                tasks.shape[0] * 2, 2, device=native.device, dtype=torch.int32
            )
            self.retile = (
                self.lib.iq2xxs_classify1632
                if hybrid1632
                else self.lib.iq2xxs_retile_mixed
                if tile == 0
                else self.lib.iq2xxs_retile8
            )
            self.retile.argtypes = (
                [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
            )
            self.retile.restype = ctypes.c_int
        if down_activation_block is None:
            down_activation_block = activation_block
        if activation_block not in (32, 256) or down_activation_block not in (32, 256):
            raise ValueError("activation_block must be 32 or 256")
        if accumulate256 and not (
            register_decode and activation_block == 256 and down_activation_block == 32
        ):
            raise ValueError(
                "Integer scale accumulation requires register FC1 per256 and FC2 per32"
            )
        if (
            register_decode
            and not accumulate256
            and (activation_block != 32 or down_activation_block != 32)
        ):
            raise ValueError("register decoder requires per32 activation quantization")
        suffix = (
            "_register_acc256"
            if accumulate256
            else "_register"
            if register_decode
            else "_fold256"
            if activation_block == 256
            else ""
        )
        self.fc1 = getattr(self.lib, "iq2xxs_routed_int8_slab" + suffix)
        down_suffix = (
            "_register"
            if register_decode
            else "_fold256"
            if down_activation_block == 256
            else ""
        )
        if fused_swiglu and down_activation_block != 32:
            raise ValueError("fused SwiGLU requires per32 down quantization")
        if masked_swiglu and not fused_swiglu:
            raise ValueError("masked SwiGLU requires fused SwiGLU")
        self.masked_swiglu = masked_swiglu
        self.fused_quant = (
            getattr(
                self.lib,
                "iq2xxs_swiglu_quantize32" + ("_masked" if masked_swiglu else ""),
            )
            if fused_swiglu
            else None
        )
        if self.fused_quant is not None:
            self.fused_quant.argtypes = (
                [ctypes.c_void_p] * (4 if masked_swiglu else 3)
                + [ctypes.c_int] * 2
                + [ctypes.c_float, ctypes.c_void_p]
            )
            self.fused_quant.restype = ctypes.c_int
        self.fc2 = getattr(
            self.lib,
            "iq2xxs_routed_int8_slab"
            + down_suffix
            + "_weighted"
            + ("_quantized" if fused_swiglu else ""),
        )
        if hybrid1632:
            self.fc1 = self.lib.iq2xxs_routed_int8_slab_hybrid1632
            self.fc2 = self.lib.iq2xxs_routed_int8_slab_hybrid1632_weighted_quantized
        for fn, pointers in ((self.fc1, 8), (self.fc2, 9)):
            fn.argtypes = (
                [ctypes.c_void_p] * pointers + [ctypes.c_int] * 7 + [ctypes.c_void_p]
            )
            fn.restype = ctypes.c_int

    def run(self, x, ids, top_weights, out):
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
        ) = self.scratch
        m = x.shape[0]
        r = self.plan.route_capacity
        torch.arange(r, device=x.device, dtype=torch.int32, out=identity)
        gguf.plan_routes(
            ids,
            top_weights,
            self.native.expert_map,
            info,
            tokens,
            weights,
            inverse[:m],
            tasks,
            self.plan.tile_rows,
        )
        q1, s1 = self.quant[0]
        q2, s2 = self.quant[1]
        if self.small_tasks is not None:
            assert (
                self.retile(
                    tasks.data_ptr(),
                    tokens.data_ptr(),
                    self.small_tasks.data_ptr(),
                    tasks.shape[0],
                    r,
                    torch.cuda.current_stream().cuda_stream,
                )
                == 0
            )
            tasks = self.small_tasks
        h, i = self.native.hidden_size, self.native.intermediate_size
        tail = (
            len(self.native.expert_ids),
            tasks.shape[0],
            self.tile,
            torch.cuda.current_stream().cuda_stream,
        )
        assert (
            self.fc1(
                *[
                    t.data_ptr()
                    for t in (x, q1, s1, *self.packed[0], tokens, tasks, projected)
                ],
                m,
                r,
                2 * i,
                h,
                *tail,
            )
            == 0
        )
        if self.fused_quant is None:
            gguf.swiglu(projected, hidden, self.plan.swiglu_limit)
        else:
            assert (
                self.fused_quant(
                    projected.data_ptr(),
                    q2.data_ptr(),
                    s2.data_ptr(),
                    *([tokens.data_ptr()] if self.masked_swiglu else []),
                    r,
                    i,
                    self.plan.swiglu_limit,
                    torch.cuda.current_stream().cuda_stream,
                )
                == 0
            )
        assert (
            self.fc2(
                *[
                    t.data_ptr()
                    for t in (
                        hidden,
                        q2,
                        s2,
                        *self.packed[1],
                        identity,
                        tasks,
                        weights,
                        contributions,
                    )
                ],
                r,
                r,
                h,
                i,
                *tail,
            )
            == 0
        )
        gguf.route_reduce(contributions, inverse[:m], out)
