# SPDX-License-Identifier: Apache-2.0
"""Benchmark-only complete IQ2_XXS/IQ3_XXS chain using prebuilt INT8 C ABI kernels."""

import ctypes

import torch

from consumer_int8 import gguf
from consumer_int8.gguf_moe import ExecutionPlan


def repack(packed, bits=2):
    if bits not in (2, 3):
        raise ValueError("XXS repack requires bits2/3")
    codes, bases, metadata = packed
    e, n, code_bytes = codes.shape
    k = code_bytes * (8 if bits == 2 else 4)
    chunks = codes.reshape(e, n, k // 128, 4, 4 if bits == 2 else 8)
    words = metadata.contiguous().view(torch.uint8).reshape(e, n, k // 128, 4, 4)
    descriptors = torch.cat((chunks, words), -1).permute(0, 2, 1, 3, 4).contiguous()
    scales = bases[..., : k // 256].transpose(1, 2).contiguous()
    assert descriptors.numel() + scales.numel() * 2 == e * n * (k // 256) * (
        66 if bits == 2 else 98
    )
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
        down_register_decode=None,
        down_library=None,
        iq3_adaptive=False,
        external_scratch=False,
        first_tile16=False,
        b12x_packed=False,
        fused_fc1=False,
    ):
        if (down_register_decode is not None or down_library is not None) and not (
            tile in (16, 32, 64)
            and not hybrid1632
            and not accumulate256
            and activation_block == 32
            and down_activation_block in (None, 32)
        ):
            raise ValueError("Independent down decoder requires ordinary per32 tiles")
        if down_register_decode is None:
            down_register_decode = register_decode
        if b12x_packed:
            if not (
                register_decode == down_register_decode
                and tile in (16, 32)
                and activation_block == 32
                and down_activation_block in (None, 32)
                and not (first_tile16 or accumulate256 or iq3_adaptive)
            ):
                raise ValueError("b12x storage requires ordinary per32 M16/32 or hybrid1632")
            for pair, n, k, fmt in zip(
                packed,
                (2 * native.intermediate_size, native.hidden_size),
                (native.hidden_size, native.intermediate_size),
                (native.first_format, native.second_format),
                strict=True,
            ):
                if n % 64 or k % 256 or len(pair) != 2:
                    raise ValueError("Invalid b12x tiled geometry")
                values, metadata = pair
                blocks = len(native.expert_ids) * n * (k // 256)
                width = 32 if fmt == "IQ2_XXS" else 64
                if (
                    values.dtype != torch.int32
                    or metadata.dtype != torch.uint8
                    or values.numel() * 4 != blocks * width
                    or metadata.numel() != blocks * 34
                    or any(
                        t.device != native.device or not t.is_contiguous() for t in pair
                    )
                ):
                    raise ValueError("Invalid b12x tiled descriptors/metadata storage")
        storage_suffix = "_routed_int8_b12x" if b12x_packed else "_routed_int8_slab"
        if fused_fc1 and not (
            b12x_packed
            and fused_swiglu
            and masked_swiglu
            and (
                (tile == 16 and register_decode and not hybrid1632)
                or (tile == 32 and hybrid1632)
            )
        ):
            raise ValueError("Fused FC1 requires borrowed M16 or hybrid1632 masked fused path")
        self.fused_fc1 = fused_fc1
        if first_tile16 and not (
            tile == 32
            and native.first_format == "IQ2_XXS"
            and register_decode
            and not down_register_decode
            and fused_swiglu
            and masked_swiglu
            and activation_block == 32
            and down_activation_block in (None, 32)
            and not (hybrid1632 or accumulate256 or iq3_adaptive)
        ):
            raise ValueError("FC1 tile16 requires register/shared fused per32 tile32")
        self.first_tile16 = first_tile16
        if any(
            f not in ("IQ2_XXS", "IQ3_XXS")
            for f in (native.first_format, native.second_format)
        ):
            raise ValueError("INT8 candidate requires IQ2_XXS/IQ3_XXS projections")
        if "IQ3_XXS" in (native.first_format, native.second_format) and (
            (hybrid1632 and not b12x_packed)
            or accumulate256
            or activation_block != 32
            or down_activation_block not in (None, 32)
            or (
                tile not in (16, 32)
                and (
                    (native.first_format == "IQ3_XXS" and register_decode)
                    or (native.second_format == "IQ3_XXS" and down_register_decode)
                )
            )
        ):
            raise ValueError(
                "IQ3 requires per32 scales; register decoding supports M16/32"
            )
        first_prefix = "iq3xxs" if native.first_format == "IQ3_XXS" else "iq2xxs"
        down_prefix = "iq3xxs" if native.second_format == "IQ3_XXS" else "iq2xxs"
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
        if iq3_adaptive and not (
            native.first_format == "IQ3_XXS"
            and tile == 32
            and activation_block == 32
            and down_activation_block in (None, 32)
            and not register_decode
            and not down_register_decode
            and fused_swiglu
            and masked_swiglu
            and not hybrid1632
            and not accumulate256
            and down_library is None
        ):
            raise ValueError("Adaptive IQ3 FC1 requires shared fused per32 tile32")
        self.iq3_adaptive = iq3_adaptive
        self.native = native
        self.packed = packed
        # Benchmark C ABI uses tile=0 for disjoint sparse8/dense16 task arrays.
        if tile in (0, 8) and not (register_decode and fused_swiglu and masked_swiglu):
            raise ValueError("Small-row experiment requires masked fused register path")
        self.tile = tile
        self.plan = ExecutionPlan(
            native, max_tokens, topk, tile_rows=16 if tile in (0, 8) else tile
        )
        r = self.plan.route_capacity
        self._scratch_specs = self.plan.scratch_specs() + (
            ((max_tokens, native.hidden_size), torch.int8),
            ((max_tokens, native.hidden_size // 32), torch.float32),
            ((r, native.intermediate_size), torch.int8),
            ((r, native.intermediate_size // 32), torch.float32),
        )
        self.lib = ctypes.CDLL(str(library.resolve()))
        self.down_lib = (
            ctypes.CDLL(str(down_library.resolve()))
            if down_library is not None
            else self.lib
        )
        self.retile = None
        if tile in (0, 8) or hybrid1632 or iq3_adaptive or first_tile16:
            task_count = self.plan.scratch_specs()[4][0][0]
            if task_count > 32767:
                raise ValueError("Retiled task capacity exceeds CUDA grid")
            self._scratch_specs += (((task_count * 2, 2), torch.int32),)
            self.retile = (
                self.lib.iq2xxs_retile16
                if first_tile16
                else self.lib.iq3xxs_classify_full32
                if iq3_adaptive
                else self.lib.iq2xxs_classify1632
                if hybrid1632
                else self.lib.iq2xxs_retile_mixed
                if tile == 0
                else self.lib.iq2xxs_retile8
            )
            self.retile.argtypes = (
                [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
            )
            self.retile.restype = ctypes.c_int
        self.scratch = None if external_scratch else self.allocate_scratch()
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
        down_suffix = (
            "_register"
            if down_register_decode
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
        if hybrid1632:
            self.fc1 = getattr(
                self.lib,
                first_prefix
                + storage_suffix
                + "_hybrid1632"
                + ("_swiglu" if fused_fc1 else ""),
            )
            self.fc2 = getattr(
                self.down_lib,
                down_prefix + storage_suffix + "_hybrid1632_weighted_quantized",
            )
        else:
            self.fc1 = getattr(
                self.lib,
                first_prefix + storage_suffix + suffix + ("_swiglu" if fused_fc1 else ""),
            )
            self.fc2 = getattr(
                self.down_lib,
                down_prefix
                + storage_suffix
                + down_suffix
                + "_weighted"
                + ("_quantized" if fused_swiglu else ""),
            )
        self.pass_tokens_to_fc2 = (
            self.masked_swiglu
            and getattr(self.down_lib, "iq_fc2_uses_route_tokens", None) is not None
        )
        if iq3_adaptive:
            self.fc1 = self.lib.iq3xxs_routed_int8_slab_adaptive
        for fn, pointers in ((self.fc1, 8), (self.fc2, 9)):
            fn.argtypes = (
                [ctypes.c_void_p] * pointers + [ctypes.c_int] * 7 + [ctypes.c_void_p]
            )
            fn.restype = ctypes.c_int
        if fused_fc1:
            self.fc1.argtypes = (
                [ctypes.c_void_p] * 9
                + [ctypes.c_int] * 7
                + [ctypes.c_float, ctypes.c_void_p]
            )

    def scratch_specs(self):
        """Simultaneous disjoint views, suitable for a shared worker arena."""
        return self._scratch_specs

    def allocate_scratch(self):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Scratch allocation must precede capture")
        return tuple(
            torch.empty(shape, dtype=dtype, device=self.native.device)
            for shape, dtype in self.scratch_specs()
        )

    def run(self, x, ids, top_weights, out, scratch=None):
        """Accept fresh arena views; never retain caller-owned workspace pointers."""
        scratch = self.scratch if scratch is None else scratch
        if scratch is None or len(scratch) != len(self.scratch_specs()):
            raise ValueError("Supply all INT8 MoE scratch views")
        for t, (shape, dtype) in zip(scratch, self.scratch_specs(), strict=True):
            if (
                tuple(t.shape) != shape
                or t.dtype != dtype
                or t.device != self.native.device
                or not t.is_contiguous()
            ):
                raise ValueError("Invalid INT8 MoE scratch view")
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
        ) = scratch[:9]
        m = x.shape[0]
        r = self.plan.route_capacity
        if not self.pass_tokens_to_fc2:
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
        q1, s1, q2, s2 = scratch[9:13]
        if self.retile is not None:
            small_tasks = scratch[13]
            assert (
                self.retile(
                    tasks.data_ptr(),
                    tokens.data_ptr(),
                    small_tasks.data_ptr(),
                    tasks.shape[0],
                    r,
                    torch.cuda.current_stream().cuda_stream,
                )
                == 0
            )
            tasks = small_tasks
        h, i = self.native.hidden_size, self.native.intermediate_size
        tail = (
            len(self.native.expert_ids),
            tasks.shape[0],
            16 if self.first_tile16 else self.tile,
            torch.cuda.current_stream().cuda_stream,
        )
        assert (
            self.fc1(
                *[
                    t.data_ptr()
                    for t in (
                        x,
                        q1,
                        s1,
                        *self.packed[0],
                        tokens,
                        tasks,
                        q2 if self.fused_fc1 else projected,
                        *([s2] if self.fused_fc1 else []),
                    )
                ],
                m,
                r,
                2 * i,
                h,
                *tail[:-1],
                *([self.plan.swiglu_limit] if self.fused_fc1 else []),
                tail[-1],
            )
            == 0
        )
        if self.fused_fc1:
            pass  # FC1 already produced q2/s2, with both BF16 rounding boundaries.
        elif self.fused_quant is None:
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
        if self.iq3_adaptive or self.first_tile16:
            tasks = scratch[4]
            tail = (tail[0], tasks.shape[0], self.tile, tail[3])
        fc2_tokens = tokens if self.pass_tokens_to_fc2 else identity
        assert (
            self.fc2(
                *[
                    t.data_ptr()
                    for t in (
                        hidden,
                        q2,
                        s2,
                        *self.packed[1],
                        fc2_tokens,
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
