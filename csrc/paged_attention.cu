// SPDX-License-Identifier: Apache-2.0
#include "kernels.h"
#include <cmath>
#include <cuda_bf16.h>
#include <stdexcept>

namespace {
using bf16 = __nv_bfloat16;
struct AttentionShared {
  union {
    bf16 qs[64][264];
    bf16 ks[64][264];
  };
  bf16 vs[64][264], ps[64][72];
  float scores[64][64], kscale[64], vscale[64];
  float rowmax[64], rowsum[64], alpha[64];
};

static_assert(sizeof(AttentionShared) <= 101376, "SM86 shared-memory limit");

__device__ __forceinline__ uint32_t pair(const bf16 *p) {
  return *reinterpret_cast<const uint32_t *>(p);
}
__device__ __forceinline__ void mma(float *c, const uint32_t *a,
                                    const uint32_t *b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                 "r"(b[1]));
}

// One head and 64 query tokens per CTA. Codes are exactly representable in
// BF16; K scales apply after QK, V scales before PV, as in the serving kernel.
// No expanded KV cache or persistent model workspace is required.
__global__ void paged_attention_kernel(
    const bf16 *query, const int8_t *cache, const int32_t *table,
    const int32_t *offsets, const int32_t *lengths, bf16 *out, int heads,
    int kv_heads, int block_size, int table_stride, int64_t qt, int64_t qh,
    int64_t cb, int64_t ch, int64_t ct, int64_t ot, int64_t oh, float scale) {
  const int seq = blockIdx.z, head = blockIdx.y;
  const int start = offsets[seq], count = offsets[seq + 1] - start;
  const int first = blockIdx.x * 64;
  if (first >= count)
    return;
  const int length = lengths[seq], context = length - count;
  const int key_head = head / (heads / kv_heads);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int group = lane >> 2, tid = lane & 3;
  extern __shared__ __align__(16) char storage[];
  auto &shared = *reinterpret_cast<AttentionShared *>(storage);
  auto &qs = shared.qs;
  auto &ks = shared.ks;
  auto &vs = shared.vs;
  auto &ps = shared.ps;
  auto &scores = shared.scores;
  auto &kscale = shared.kscale;
  auto &vscale = shared.vscale;
  auto &rowmax = shared.rowmax;
  auto &rowsum = shared.rowsum;
  auto &alpha = shared.alpha;
  const int row_base = (warp / 2) * 16, col_base = (warp % 2) * 32;
  for (int i = threadIdx.x; i < 64 * 256; i += 256) {
    const int r = i / 256, d = i % 256;
    qs[r][d] = first + r < count
                   ? query[(start + first + r) * qt + head * qh + d]
                   : __float2bfloat16(0.f);
  }
  if (threadIdx.x < 64) {
    rowmax[threadIdx.x] = -INFINITY;
    rowsum[threadIdx.x] = 0.f;
  }
  float accum[16][4] = {};
  const int stop = min(length, context + first + 64);
  __syncthreads();
  uint32_t query_fragments[16][4];
#pragma unroll
  for (int t = 0; t < 16; ++t) {
    const int d = t * 16 + tid * 2;
    query_fragments[t][0] = pair(&qs[row_base + group][d]);
    query_fragments[t][1] = pair(&qs[row_base + group + 8][d]);
    query_fragments[t][2] = pair(&qs[row_base + group][d + 8]);
    query_fragments[t][3] = pair(&qs[row_base + group + 8][d + 8]);
  }
  __syncthreads();
  for (int base = 0; base < stop; base += 64) {
#pragma unroll
    for (int r = warp; r < 64; r += 8) {
      const int pos = base + r;
      const int page =
          pos < length ? table[seq * table_stride + pos / block_size] : 0;
      const int8_t *src =
          cache + page * cb + key_head * ch + (pos % block_size) * ct;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int d = lane * 4 + half * 128;
        const uint32_t kc =
            pos < length ? *reinterpret_cast<const uint32_t *>(src + d) : 0;
        const uint32_t vc =
            pos < length ? *reinterpret_cast<const uint32_t *>(src + 260 + d)
                         : 0;
#pragma unroll
        for (int j = 0; j < 4; j += 2) {
          *reinterpret_cast<__nv_bfloat162 *>(&ks[r][d + j]) =
              __floats2bfloat162_rn(float(int8_t(kc >> (j * 8))),
                                    float(int8_t(kc >> ((j + 1) * 8))));
          *reinterpret_cast<__nv_bfloat162 *>(&vs[r][d + j]) =
              __floats2bfloat162_rn(float(int8_t(vc >> (j * 8))),
                                    float(int8_t(vc >> ((j + 1) * 8))));
        }
      }
      if (!lane) {
        kscale[r] =
            pos < length ? *reinterpret_cast<const float *>(src + 256) : 1.f;
        vscale[r] =
            pos < length ? *reinterpret_cast<const float *>(src + 516) : 1.f;
      }
    }
    __syncthreads();
#pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
      float score[4] = {};
#pragma unroll
      for (int d = 0; d < 256; d += 16) {
        uint32_t b[2] = {
            pair(&ks[col_base + tile * 8 + group][d + tid * 2]),
            pair(&ks[col_base + tile * 8 + group][d + tid * 2 + 8])};
        mma(score, query_fragments[d / 16], b);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int r = row_base + group + (j / 2) * 8,
                  col = col_base + tile * 8 + tid * 2 + j % 2;
        scores[r][col] =
            base + col <= context + first + r && base + col < length
                ? __fmul_rn(score[j], __fmul_rn(scale, kscale[col]))
                : -INFINITY;
      }
    }
    __syncthreads();
    // Eight lanes cooperate on each row's online softmax.
    for (int row = threadIdx.x / 8; row < 64; row += 32) {
      const int sublane = threadIdx.x % 8;
      float maximum = -INFINITY;
#pragma unroll
      for (int j = 0; j < 8; ++j)
        maximum = fmaxf(maximum, scores[row][sublane + j * 8]);
      for (int mask = 4; mask; mask >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, mask));
      maximum = fmaxf(maximum, rowmax[row]);
      const float decay = expf(rowmax[row] - maximum);
      float sum = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int col = sublane + j * 8;
        const float p = expf(scores[row][col] - maximum);
        sum += p;
        ps[row][col] = __float2bfloat16(p * vscale[col]);
      }
      for (int mask = 4; mask; mask >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, mask);
      if (!sublane) {
        alpha[row] = decay;
        rowmax[row] = maximum;
        rowsum[row] = rowsum[row] * decay + sum;
      }
    }
    __syncthreads();
#pragma unroll
    for (int tile = 0; tile < 16; ++tile) {
#pragma unroll
      for (int j = 0; j < 4; ++j)
        accum[tile][j] *= alpha[row_base + group + (j / 2) * 8];
#pragma unroll
      for (int d = 0; d < 64; d += 16) {
        uint32_t a[4] = {pair(&ps[row_base + group][d + tid * 2]),
                         pair(&ps[row_base + group + 8][d + tid * 2]),
                         pair(&ps[row_base + group][d + tid * 2 + 8]),
                         pair(&ps[row_base + group + 8][d + tid * 2 + 8])};
        const int col = (warp % 2) * 128 + tile * 8 + group;
        uint32_t b[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int r = d + tid * 2 + j * 8;
          b[j] = uint32_t(__bfloat16_as_ushort(vs[r][col])) |
                 (uint32_t(__bfloat16_as_ushort(vs[r + 1][col])) << 16);
        }
        mma(accum[tile], a, b);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int tile = 0; tile < 16; ++tile) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int row = row_base + group + (j / 2) * 8;
      const int col = (warp % 2) * 128 + tile * 8 + tid * 2 + j % 2;
      if (first + row < count)
        out[(start + first + row) * ot + head * oh + col] =
            __float2bfloat16(accum[tile][j] / rowsum[row]);
    }
  }
}
} // namespace

void paged_attention(const void *query, const int8_t *cache,
                     const int32_t *table, const int32_t *offsets,
                     const int32_t *lengths, void *out, int heads, int kv_heads,
                     int block_size, int table_stride, int sequences,
                     int max_query, int64_t qt, int64_t qh, int64_t cb,
                     int64_t ch, int64_t ct, int64_t ot, int64_t oh,
                     float scale, cudaStream_t stream) {
  constexpr int shared_bytes = sizeof(AttentionShared);
  const auto status = cudaFuncSetAttribute(
      paged_attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      shared_bytes);
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
  paged_attention_kernel<<<dim3((max_query + 63) / 64, heads, sequences), 256,
                           shared_bytes, stream>>>(
      static_cast<const bf16 *>(query), cache, table, offsets, lengths,
      static_cast<bf16 *>(out), heads, kv_heads, block_size, table_stride, qt,
      qh, cb, ch, ct, ot, oh, scale);
}
