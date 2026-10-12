# SPDX-License-Identifier: Apache-2.0
"""AOT operators. Buffers belong to the caller; importing never compiles code.

GDN states are FP32 [slot, value head, value dim, key dim]. Slot zero and
negative slots are padding. Active slots must be unique within one launch.
Inputs q/k are BF16 and normalized inside the kernel. g is the FP32 natural
log decay and beta is the FP32 sigmoid gate, both computed by the caller.
"""

import torch

from . import _C  # noqa: F401

quantize_rows = torch.ops.consumer_int8.quantize_rows
channel_mm = torch.ops.consumer_int8.channel_mm
gdn_decode = torch.ops.consumer_int8.gdn_decode
cutlass_channel_mm = torch.ops.consumer_int8.cutlass_channel_mm
kv_cache_update = torch.ops.consumer_int8.kv_cache_update


@torch.library.register_fake("consumer_int8::quantize_rows")
def _quantize_rows_fake(input, codes, scales):
    return None


@torch.library.register_fake("consumer_int8::channel_mm")
def _channel_mm_fake(a, w, a_scale, w_scale, out):
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
