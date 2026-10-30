// SPDX-License-Identifier: Apache-2.0
#include "kernels.h"
#include <cuda_bf16.h>

namespace {
// One warp converts a complete token/head row. Each sequence becomes one
// contiguous BF16 page suitable for an existing paged FlashAttention kernel.
template <bool Int4 = false>
__global__ void unpack_kv_cache_kernel(
    const int8_t* cache, const int32_t* table, const int32_t* lengths,
    __nv_bfloat16* key, __nv_bfloat16* value, int heads, int capacity,
    int block_size, int table_stride, int64_t cb, int64_t ch, int64_t ct) {
  const int seq = blockIdx.y;
  const int item = blockIdx.x * 8 + threadIdx.x / 32;
  const int pos = item / heads, head = item % heads;
  if (pos >= lengths[seq] || pos >= capacity) return;
  const int lane = threadIdx.x % 32;
  const int page = table[seq * table_stride + pos / block_size];
  const int8_t* row = cache + page * cb + head * ch + (pos % block_size) * ct;
  const int64_t dst = (int64_t(seq) * capacity * heads + item) * 256;
  if constexpr (Int4) {
    const uint32_t ks_bits = *reinterpret_cast<const uint32_t*>(row + 128);
    const uint32_t vs_bits = *reinterpret_cast<const uint32_t*>(row + 260);
    const int kzp = int(ks_bits & 0xFu);
    const int vzp = int(vs_bits & 0xFu);
    const float ks = __uint_as_float(ks_bits & ~0xFu);
    const float vs = __uint_as_float(vs_bits & ~0xFu);
    const int d = lane * 8;
    const uint32_t k = *reinterpret_cast<const uint32_t*>(row + lane * 4);
    const uint32_t v = *reinterpret_cast<const uint32_t*>(row + 132 + lane * 4);
#pragma unroll
    for (int j = 0; j < 8; j += 2) {
      *reinterpret_cast<__nv_bfloat162*>(key + dst + d + j) =
          __floats2bfloat162_rn(float(int((k >> (j * 4)) & 0xFu) - kzp) * ks,
                               float(int((k >> ((j + 1) * 4)) & 0xFu) - kzp) * ks);
      *reinterpret_cast<__nv_bfloat162*>(value + dst + d + j) =
          __floats2bfloat162_rn(float(int((v >> (j * 4)) & 0xFu) - vzp) * vs,
                               float(int((v >> ((j + 1) * 4)) & 0xFu) - vzp) * vs);
    }
  } else {
    const float ks = *reinterpret_cast<const float*>(row + 256);
    const float vs = *reinterpret_cast<const float*>(row + 516);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int d = lane * 4 + half * 128;
      const uint32_t k = *reinterpret_cast<const uint32_t*>(row + d);
      const uint32_t v = *reinterpret_cast<const uint32_t*>(row + 260 + d);
#pragma unroll
      for (int j = 0; j < 4; j += 2) {
        *reinterpret_cast<__nv_bfloat162*>(key + dst + d + j) =
            __floats2bfloat162_rn(float(int8_t(k >> (j * 8))) * ks,
                                 float(int8_t(k >> ((j + 1) * 8))) * ks);
        *reinterpret_cast<__nv_bfloat162*>(value + dst + d + j) =
            __floats2bfloat162_rn(float(int8_t(v >> (j * 8))) * vs,
                                 float(int8_t(v >> ((j + 1) * 8))) * vs);
      }
    }
  }
}
} // namespace

void unpack_kv_cache(const int8_t* cache, const int32_t* table,
                     const int32_t* lengths, void* key, void* value,
                     int heads, int capacity, int sequences, int block_size,
                     int table_stride, int64_t cb, int64_t ch, int64_t ct,
                     cudaStream_t stream, bool int4) {
  if (int4) {
    unpack_kv_cache_kernel<true><<<dim3((capacity * heads + 7) / 8, sequences), 256, 0, stream>>>(
        cache, table, lengths, static_cast<__nv_bfloat16*>(key),
        static_cast<__nv_bfloat16*>(value), heads, capacity, block_size,
        table_stride, cb, ch, ct);
  } else {
    unpack_kv_cache_kernel<false><<<dim3((capacity * heads + 7) / 8, sequences), 256, 0, stream>>>(
        cache, table, lengths, static_cast<__nv_bfloat16*>(key),
        static_cast<__nv_bfloat16*>(value), heads, capacity, block_size,
        table_stride, cb, ch, ct);
  }
}
