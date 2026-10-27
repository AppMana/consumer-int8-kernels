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

q6_mm = torch.ops.consumer_int8.gguf_q6_mm
q6_expand = torch.ops.consumer_int8.gguf_q6_expand


@torch.library.register_fake("consumer_int8::gguf_q6_mm")
def _q6_fake(x, raw, out, tile_rows=64):
    return None


@torch.library.register_fake("consumer_int8::gguf_q6_expand")
def _q6_expand_fake(raw, out):
    return None


def q6_expand_mm(x, raw, weight_workspace, out):
    """Expand raw Q6_K [G,N,K/256,210] into shared scratch, then BF16 GEMM.

    Keeps original signed scales and codes, without activation quantization.
    The caller owns scratch [G,N,K] and serializes concurrent users. This is
    an experimental shape-dependent candidate, not a default serving policy.
    """
    if x.ndim != 3 or raw.ndim != 4 or out.ndim != 3:
        raise ValueError("Expected x[M,G,K], raw[G,N,K/256,210], out[M,G,N]")
    if x.shape[1:] != (raw.shape[0], raw.shape[2] * 256):
        raise ValueError("Q6 activation shape does not match weights")
    if out.shape != (x.shape[0], raw.shape[0], raw.shape[1]):
        raise ValueError("Q6 output shape does not match activations/weights")
    q6_expand(raw, weight_workspace)
    torch.bmm(
        x.transpose(0, 1), weight_workspace.transpose(1, 2), out=out.transpose(0, 1)
    )


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


# Expand into caller-owned, reusable BF16 workspace before a dense GEMM.
q8_expand = torch.ops.consumer_int8.gguf_q8_expand


@torch.library.register_fake("consumer_int8::gguf_q8_expand")
def _q8_expand_fake(codes, scales, out):
    return None


def q8_expand_mm(x, codes, scales, weight_workspace, out):
    """Decode Q8 into reusable workspace, then run grouped BF16 cuBLAS.

    x[M,G,K], codes[G,N,K], scales[G,N,K/32], workspace[G,N,K], out[M,G,N].
    Allocate the weight workspace once and share it across sequential layers;
    simultaneous streams require separate workspaces or explicit ordering.
    Callers select this path from device/shape measurements, not quant alone.
    """
    if x.ndim != 3 or out.ndim != 3:
        raise ValueError("Expected x[M,G,K] and out[M,G,N]")
    if x.shape[1:] != (codes.shape[0], codes.shape[2]):
        raise ValueError("Q8 activation shape does not match weights")
    if out.shape != (x.shape[0], codes.shape[0], codes.shape[1]):
        raise ValueError("Q8 output shape does not match activations/weights")
    q8_expand(codes, scales, weight_workspace)
    torch.bmm(
        x.transpose(0, 1), weight_workspace.transpose(1, 2), out=out.transpose(0, 1)
    )


mxfp4_routed_mm = torch.ops.consumer_int8.gguf_mxfp4_routed_mm


@torch.library.register_fake("consumer_int8::gguf_mxfp4_routed_mm")
def _mxfp4_routed_fake(x, codes, scales, tokens, tasks, out, tile_rows=16):
    return None


def pack_mxfp4(raw):
    """Separate original GGUF MXFP4 [G,N,K/32,17] bytes without requantizing.

    Low/high nibbles retain the GGUF split-half layout, not adjacent E2M1
    ordering. This is MXFP4 with E8M0 scales, not NVFP4 with E4M3 scales.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("MXFP4 packing must precede graph capture")
    if not raw.is_cuda or raw.dtype != torch.uint8 or raw.ndim != 4:
        raise ValueError("Expected CUDA uint8 MXFP4 blocks [G,N,K/32,17]")
    if raw.shape[-1] != 17 or min(raw.shape) <= 0 or raw.shape[-2] % 4:
        raise ValueError("MXFP4 requires positive dimensions and K divisible by 128")
    return raw[..., 1:].contiguous().flatten(2), raw[..., 0].contiguous()


mxfp4_routed_weighted_mm = torch.ops.consumer_int8.gguf_mxfp4_routed_weighted_mm
swiglu = torch.ops.consumer_int8.gguf_swiglu
route_reduce = torch.ops.consumer_int8.gguf_route_reduce


@torch.library.register_fake("consumer_int8::gguf_mxfp4_routed_weighted_mm")
def _mxfp4_weighted_fake(x, codes, scales, tokens, tasks, weights, out, tile_rows=16):
    return None


@torch.library.register_fake("consumer_int8::gguf_swiglu")
def _swiglu_fake(x, out, limit=10.0):
    return None


@torch.library.register_fake("consumer_int8::gguf_route_reduce")
def _route_reduce_fake(contributions, rows, out):
    return None


def mixed_iq_moe(
    x,
    packed_gate_up,
    packed_down,
    tokens,
    identity_rows,
    tasks,
    weights,
    inverse_routes,
    projected,
    hidden,
    contributions,
    out,
    *,
    tile_rows=16,
    swiglu_limit=10.0,
    first_format="IQ2_XS",
    second_format="MXFP4",
):
    """Execute one local mixed expert partition using caller-owned buffers.

    Gate/up rows are concatenated in that order. Projected BF16 values feed
    FP32 clipped SwiGLU with one BF16 intermediate rounding. Down accumulators
    are weighted in FP32 before BF16 contribution rounding; inverse_routes
    [tokens,topk] selects contributions for deterministic FP32 summation.
    Negative inverse entries omit remote experts. No collective is performed.

    Tasks and tokens follow iq2_xs_routed_mm's disjoint padded-row contract.
    identity_rows maps each down-input route row to itself. All scratch and
    outputs must be disjoint and prepared before capture. Routing construction
    is separate; this function performs no allocation or host synchronization.
    """
    if first_format not in {
        "IQ2_XS",
        "IQ2_XXS",
        "IQ3_XXS",
        "IQ2_S",
        "IQ3_S",
    } or second_format not in {
        "MXFP4",
        "IQ2_XXS",
        "IQ3_XXS",
        "IQ2_S",
        "IQ3_S",
    }:
        raise ValueError("Unsupported native mixed IQ formats")
    if first_format == "IQ2_XS":
        iq2_xs_routed_mm(x, *packed_gate_up, tokens, tasks, projected, tile_rows)
    else:
        op = iq_s_routed_mm if first_format.endswith("_S") else iq_xxs_routed_mm
        op(
            x,
            *packed_gate_up,
            tokens,
            tasks,
            weights,
            projected,
            int(first_format[2]),
            tile_rows,
            False,
        )
    swiglu(projected, hidden, swiglu_limit)
    if second_format == "MXFP4":
        mxfp4_routed_weighted_mm(
            hidden,
            *packed_down,
            identity_rows,
            tasks,
            weights,
            contributions,
            tile_rows,
        )
    else:
        op = iq_s_routed_mm if second_format.endswith("_S") else iq_xxs_routed_mm
        op(
            hidden,
            *packed_down,
            identity_rows,
            tasks,
            weights,
            contributions,
            int(second_format[2]),
            tile_rows,
            True,
        )
    route_reduce(contributions, inverse_routes, out)


mixed_iq2_xs_mxfp4 = mixed_iq_moe


# Build stable expert-major routes entirely on-device. expert_map[global_id]
# gives the local weight index, or -1 for a remote expert. info[E,2] receives
# (count, padded offset). Other outputs follow mixed_iq2_xs_mxfp4's contracts.
# Allocate >=T*topk + E*(tile_rows-1) route rows and ceil(that/tile_rows)
# tasks. Every output buffer must be disjoint. Unused entries reset each call.
plan_routes = torch.ops.consumer_int8.gguf_plan_routes


@torch.library.register_fake("consumer_int8::gguf_plan_routes")
def _plan_routes_fake(
    ids, top_weights, expert_map, info, tokens, weights, inverse, tasks, tile_rows=16
):
    return None


iq_xxs_routed_mm = torch.ops.consumer_int8.gguf_iq_xxs_routed_mm


@torch.library.register_fake("consumer_int8::gguf_iq_xxs_routed_mm")
def _iq_xxs_fake(
    x,
    codes,
    bases,
    metadata,
    tokens,
    tasks,
    weights,
    out,
    bits,
    tile_rows=16,
    weighted=False,
):
    return None


def pack_iq_xxs(raw, bits):
    """Losslessly separate IQ2_XXS/IQ3_XXS without expanding descriptors.

    Codes remain uint8, bases FP16 padded to an even block count, and original
    packed sign/scale words int32. Only one extra FP16 per odd-block row is added.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("XXS packing must precede graph capture")
    if (
        bits not in (2, 3)
        or not raw.is_cuda
        or raw.dtype != torch.uint8
        or raw.ndim != 4
    ):
        raise ValueError("Expected CUDA uint8 IQ2_XXS/IQ3_XXS blocks and bits2/3")
    if min(raw.shape) <= 0 or raw.shape[-1] != (66 if bits == 2 else 98):
        raise ValueError("Invalid XXS block shape")
    bases = raw[..., :2].contiguous().view(torch.float16).squeeze(-1)
    if not torch.isfinite(bases).all().item():
        raise ValueError("XXS scale bases must be finite")
    if bases.shape[-1] % 2:
        bases = torch.nn.functional.pad(bases, (0, 1))
    if bits == 2:
        chunks = raw[..., 2:].reshape(*raw.shape[:3], 8, 8)
        codes = chunks[..., :4].contiguous().flatten(2)
        metadata = chunks[..., 4:].contiguous().view(torch.int32).flatten(2)
    else:
        codes = raw[..., 2:66].contiguous().flatten(2)
        metadata = raw[..., 66:].contiguous().view(torch.int32).flatten(2)
    return codes, bases, metadata


iq_s_routed_mm = torch.ops.consumer_int8.gguf_iq_s_routed_mm


@torch.library.register_fake("consumer_int8::gguf_iq_s_routed_mm")
def _iq_s_fake(
    x,
    codes,
    bases,
    metadata,
    tokens,
    tasks,
    weights,
    out,
    bits,
    tile_rows=16,
    weighted=False,
):
    return None


def pack_iq_s(raw, bits):
    """Losslessly separate IQ2_S/IQ3_S codes, bases and compact metadata."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("IQ_S packing must precede graph capture")
    if (
        bits not in (2, 3)
        or not raw.is_cuda
        or raw.dtype != torch.uint8
        or raw.ndim != 4
    ):
        raise ValueError("Expected CUDA uint8 IQ2_S/IQ3_S blocks and bits2/3")
    if min(raw.shape) <= 0 or raw.shape[-1] != (82 if bits == 2 else 110):
        raise ValueError("Invalid IQ_S block shape")
    bases = raw[..., :2].contiguous().view(torch.float16).squeeze(-1)
    if not torch.isfinite(bases).all().item():
        raise ValueError("IQ_S scale bases must be finite")
    if bases.shape[-1] % 2:
        bases = torch.nn.functional.pad(bases, (0, 1))
    if bits == 2:
        codes = raw[..., 2:34].contiguous().flatten(2)
        highscale = torch.stack((raw[..., 66:74], raw[..., 74:82]), -1).flatten(-2)
        metadata = torch.cat((raw[..., 34:66], highscale), -1).flatten(2)
    else:
        codes = raw[..., 2:66].contiguous().flatten(2)
        metadata = torch.cat(
            (raw[..., 74:106], raw[..., 66:74], raw[..., 106:110]), -1
        ).flatten(2)
    return codes, bases, metadata
