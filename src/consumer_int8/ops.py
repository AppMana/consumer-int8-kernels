# SPDX-License-Identifier: Apache-2.0
"""AOT operators. Buffers belong to the caller; importing never compiles code.

GDN states are FP32 [slot, value head, value dim, key dim]. Slot zero and
negative slots are padding. Active slots must be unique within one launch.
Inputs q/k are BF16 and normalized inside the kernel. g is the FP32 natural
log decay and beta is the FP32 sigmoid gate, both computed by the caller.

quantize_grouped4096 reads contiguous BF16[M,G,4096] and writes contiguous
INT8[G,M,4096] codes plus FP32[G,M] scales. Outputs must not overlap inputs
or each other; empty M is supported. Quantization matches quantize_rows.
"""

import torch

from . import _C, _fa2  # noqa: F401

silu_mul = torch.ops.consumer_int8.silu_mul
silu_mul_quantize = torch.ops.consumer_int8.silu_mul_quantize
quantize_rows = torch.ops.consumer_int8.quantize_rows
quantize_grouped4096 = torch.ops.consumer_int8.quantize_grouped4096
channel_mm = torch.ops.consumer_int8.channel_mm
channel_gemv = torch.ops.consumer_int8.channel_gemv
unpack_allspark = torch.ops.consumer_int8.unpack_allspark
gdn_decode = torch.ops.consumer_int8.gdn_decode
cutlass_channel_mm = torch.ops.consumer_int8.cutlass_channel_mm
kv_cache_update = torch.ops.consumer_int8.kv_cache_update
fa2_prefill = torch.ops.consumer_int8.fa2_prefill
paged_decode = torch.ops.consumer_int8.paged_decode
paged_attention = torch.ops.consumer_int8.paged_attention
unpack_kv_cache = torch.ops.consumer_int8.unpack_kv_cache
fused_channel_mm = torch.ops.consumer_int8.fused_channel_mm
gemma_norm = torch.ops.consumer_int8.gemma_norm
gdn_post_conv = torch.ops.consumer_int8.gdn_post_conv


@torch.library.register_fake("consumer_int8::gemma_norm")
def _gemma_norm_fake(x, weight, residual, out, residual_out, epsilon):
    return None


@torch.library.register_fake("consumer_int8::gdn_post_conv")
def _gdn_post_conv_fake(x, a, b, log_a, dt_bias, q, k, v, g, beta):
    return None


@torch.library.register_fake("consumer_int8::fused_channel_mm")
def _fused_channel_mm_fake(a, w, a_scale, w_scale, out, schedule=0):
    return None


@torch.library.register_fake("consumer_int8::unpack_kv_cache")
def _unpack_kv_cache_fake(cache, table, lengths, key, value):
    return None


@torch.library.register_fake("consumer_int8::fa2_prefill")
def _fa2_prefill_fake(
    query, key, value, offsets, lengths, out, lse, max_query, scale, schedule=0
):
    return None


@torch.library.register_fake("consumer_int8::paged_attention")
def _paged_attention_fake(query, cache, table, offsets, lengths, out, max_query, scale):
    return None


@torch.library.register_fake("consumer_int8::quantize_rows")
def _quantize_rows_fake(input, codes, scales):
    return None


@torch.library.register_fake("consumer_int8::quantize_grouped4096")
def _quantize_grouped4096_fake(input, codes, scales):
    return None


@torch.library.register_fake("consumer_int8::channel_mm")
def _channel_mm_fake(a, w, a_scale, w_scale, out):
    return None


@torch.library.register_fake("consumer_int8::channel_gemv")
def _channel_gemv_fake(x, w, scale, out):
    return None


@torch.library.register_fake("consumer_int8::unpack_allspark")
def _unpack_allspark_fake(packed, packed_scales, weight, scales):
    return None


@torch.library.register_fake("consumer_int8::gdn_decode")
def _gdn_decode_fake(q, k, v, g, beta, state, slots, out):
    return None


@torch.library.register_fake("consumer_int8::cutlass_channel_mm")
def _cutlass_channel_mm_fake(a, w, a_scale, w_scale, workspace, out):
    return None


@torch.library.register_fake("consumer_int8::kv_cache_update")
def _kv_cache_update_fake(key, value, cache, slots):
    return None


gdn_packed_decode = torch.ops.consumer_int8.gdn_packed_decode
gdn_prefill = torch.ops.consumer_int8.gdn_prefill


@torch.library.register_fake("consumer_int8::gdn_packed_decode")
def _gdn_packed_decode_fake(qkv, a, b, A_log, dt_bias, state, slots, out):
    return None


@torch.library.register_fake("consumer_int8::gdn_prefill")
def _gdn_prefill_fake(
    q,
    k,
    v,
    g,
    beta,
    initial,
    final,
    out,
    cumulative,
    w,
    u,
    v_new,
    h_chunk,
    num_sms,
):
    return None


gdn_conv = torch.ops.consumer_int8.gdn_conv


@torch.library.register_fake("consumer_int8::gdn_conv")
def _gdn_conv_fake(
    x,
    a,
    b,
    log_a,
    dt_bias,
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
):
    return None


def _gdn_buffers(x, num_k_heads, num_v_heads):
    tokens = x.shape[0]
    q = torch.empty(tokens, num_k_heads, 128, dtype=x.dtype, device=x.device)
    k = torch.empty_like(q)
    v = torch.empty(tokens, num_v_heads, 128, dtype=x.dtype, device=x.device)
    g = torch.empty(tokens, num_v_heads, dtype=torch.float32, device=x.device)
    return q, k, v, g, torch.empty_like(g)


def gdn_post_conv_prep(
    conv_output,
    a,
    b,
    A_log,
    dt_bias,
    num_k_heads,
    head_k_dim,
    head_v_dim,
    apply_l2norm=True,
    output_g_exp=False,
):
    """D128 BF16 preparation with the same buffers as FLA's prefill API."""
    if head_k_dim != 128 or head_v_dim != 128 or not apply_l2norm or output_g_exp:
        raise ValueError("CUDA preparation requires normalized D128 Q/K and log decay")
    buffers = _gdn_buffers(conv_output, num_k_heads, A_log.shape[0])
    gdn_post_conv(conv_output, a, b, A_log, dt_bias, *buffers)
    return buffers


def gdn_conv_prep(
    x,
    a,
    b,
    A_log,
    dt_bias,
    weight,
    state,
    offsets,
    slots,
    initial,
    num_k_heads,
    max_query,
):
    """Fuse width-4 SiLU convolution and D128 preparation without prefix caching.

    Active state slots must be unique. Offsets partition all input tokens;
    max_query bounds the largest partition. Slot zero is reserved for padding.
    Only the first three history positions are used; speculative tails are preserved.
    """
    buffers = _gdn_buffers(x, num_k_heads, A_log.shape[0])
    gdn_conv(
        x,
        a,
        b,
        A_log,
        dt_bias,
        *buffers,
        weight,
        state,
        offsets,
        slots,
        initial,
        max_query,
    )
    return buffers


@torch.library.register_fake("consumer_int8::silu_mul")
def _silu_mul_fake(input, out):
    return None


@torch.library.register_fake("consumer_int8::silu_mul_quantize")
def _silu_mul_quantize_fake(input, codes, scales):
    return None


@torch.library.register_fake("consumer_int8::paged_decode")
def _paged_decode_fake(
    query, cache, table, offsets, lengths, out, max_query, scale, scratch
):
    return None
