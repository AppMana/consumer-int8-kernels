# SPDX-License-Identifier: Apache-2.0
"""Experimental native GGUF operators; no serving policy selects these yet.

Q8 accepts BF16 x[M,G,K], raw uint8 weight[G,N,K/32,34], and caller-owned
contiguous BF16 out[M,G,N]. Input row/group strides are independent. Decode
uses the checkpoint's FP16 scale times signed INT8, rounded to BF16 before
FP32 tensor-core accumulation. No activation quantization or global expanded
weight/scratch allocation occurs. Output must not overlap either input.

Upstream comparison (vllm-gguf-plugin e454affe): CUDA dense MMQ supports Q8
and K-quants, but IQ2/IQ3 matrix operations fall back to Triton. CUDA GEMV
has broader IQ coverage. Thus Q8 CUDA itself is not a new capability; this
addition is a BF16-activation, directly strided grouped tensor-core operator.
Our deployed b12x path uses CuTe DSL and separate group gathers/GEMMs/cat;
this operator instead reads and writes group strides in one launch.
"""

import torch

from . import _gguf  # noqa: F401

q8_mm = torch.ops.consumer_int8.gguf_q8_mm


@torch.library.register_fake("consumer_int8::gguf_q8_mm")
def _q8_fake(x, weight, out):
    return None


q8_packed_mm = torch.ops.consumer_int8.gguf_q8_packed_mm


@torch.library.register_fake("consumer_int8::gguf_q8_packed_mm")
def _q8_packed_fake(x, codes, scales, out):
    return None


def pack_q8(raw):
    """Losslessly separate raw [G,N,K/32,34] GGUF bytes before serving.

    Original FP16 bases and signed INT8 codes are retained. K must align to
    64 for asynchronous copies. The caller can release raw after preparation.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Q8 packing must precede graph capture")
    if not raw.is_cuda or raw.dtype != torch.uint8 or raw.ndim != 4:
        raise ValueError("Expected CUDA uint8 raw Q8 blocks [G,N,K/32,34]")
    if raw.shape[-1] != 34 or raw.shape[-2] % 2 or min(raw.shape) <= 0:
        raise ValueError("Packed Q8 requires positive dimensions and K divisible by 64")
    scales = raw[..., :2].contiguous().view(torch.float16).squeeze(-1)
    if not torch.isfinite(scales).all().item():
        raise ValueError("Q8 scale bases must be finite")
    codes = raw[..., 2:].contiguous().view(torch.int8)
    return codes.reshape(*raw.shape[:2], raw.shape[2] * 32), scales


iq2_xs_mm = torch.ops.consumer_int8.gguf_iq2_xs_mm


@torch.library.register_fake("consumer_int8::gguf_iq2_xs_mm")
def _iq2_xs_fake(x, codes, bases, scales, out):
    return None


def pack_iq2_xs(raw):
    """Losslessly separate IQ2_XS [G,N,K/256,74] before graph capture.

    Bases are padded to an even count for aligned copies; codeword and nibble
    storage retain their original size. No expanded floating weights are kept.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("IQ2_XS packing must precede graph capture")
    if not raw.is_cuda or raw.dtype != torch.uint8 or raw.ndim != 4:
        raise ValueError("Expected CUDA uint8 raw IQ2_XS blocks [G,N,K/256,74]")
    if raw.shape[-1] != 74 or min(raw.shape) <= 0:
        raise ValueError("Invalid IQ2_XS block shape")
    bases = raw[..., :2].contiguous().view(torch.float16).squeeze(-1)
    if not torch.isfinite(bases).all().item():
        raise ValueError("IQ2_XS scale bases must be finite")
    if bases.shape[-1] % 2:
        bases = torch.nn.functional.pad(bases, (0, 1))
    codes = raw[..., 2:66].contiguous().view(torch.int16).flatten(2)
    scales = raw[..., 66:].contiguous().flatten(2)
    return codes, bases, scales


# tokens[R] maps expert-major route rows to x[T,K]; negative entries are padding.
# tasks[L,2] holds (expert, route-row start), with 16 or 32 rows per task.
# Tasks must write disjoint row ranges. Expert -1 disables unused graph slots.
# Output rows not covered by an active task remain untouched. No top-k weights,
# SiLU or scatter are applied: this is a projection in the routed domain.
iq2_xs_routed_mm = torch.ops.consumer_int8.gguf_iq2_xs_routed_mm


@torch.library.register_fake("consumer_int8::gguf_iq2_xs_routed_mm")
def _iq2_xs_routed_fake(x, codes, bases, scales, tokens, tasks, out, tile_rows=16):
    return None
