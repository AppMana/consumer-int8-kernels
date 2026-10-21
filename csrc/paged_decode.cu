// SPDX-License-Identifier: Apache-2.0
#include "kernels.h"
#include <cmath>
#include <cuda_bf16.h>

namespace {
using bf16 = __nv_bfloat16;
template <int Rows> struct Shared {
  union {
    bf16 q[Rows][264];
    bf16 k[32][264];
    uint32_t packed[32][132];
  };
  bf16 v[32][264];
  bf16 p[Rows][40], p_residual[Rows][40];
  float score[Rows][32], ks[32], vs[32];
  float maximum[Rows], sum[Rows], alpha[Rows];
};
static_assert(sizeof(Shared<32>) < 48 * 1024);

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
__device__ __forceinline__ void expand(uint32_t code, bf16 *dst) {
  *reinterpret_cast<__nv_bfloat162 *>(dst) =
      __floats2bfloat162_rn(float(int8_t(code)), float(int8_t(code >> 8)));
  *reinterpret_cast<__nv_bfloat162 *>(dst + 2) = __floats2bfloat162_rn(
      float(int8_t(code >> 16)), float(int8_t(code >> 24)));
}

template <int Rows>
__device__ __forceinline__ void
stage(Shared<Rows> &s, const int8_t *cache, const int32_t *table, int seq,
      int kh, int base, int length, int block_size, int table_stride,
      int64_t cb, int64_t ch, int64_t ct) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  // Packed rows are eight-byte aligned even though the V codes start at 260.
  for (int row = warp; row < 32; row += Rows / 4) {
    const int pos = base + row;
    const bool valid = pos < length;
    const int page = valid ? table[seq * table_stride + pos / block_size] : 0;
    const int8_t *src = cache + page * cb + kh * ch + (pos % block_size) * ct;
#pragma unroll
    for (int word = lane; word < 65; word += 32) {
      const uint32_t dst = static_cast<uint32_t>(
          __cvta_generic_to_shared(&s.packed[row][word * 2]));
      asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" ::"r"(dst),
                   "l"(src + word * 8), "r"(valid ? 8 : 0)
                   : "memory");
    }
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

// Six query heads share one packed KV tile. Split contexts independently of
// query count so C=1 decode can occupy every SM. No host metadata reads.
template <int Rows>
__global__ __launch_bounds__(Rows * 8, 2)
void split_kernel(const bf16 *q, const int8_t *cache,
                  const int32_t *table, const int32_t *offsets,
                  const int32_t *lengths, float *scratch, int heads,
                  int kv_heads, int parts, int block_size,
                  int table_stride, int64_t qt, int64_t qh,
                  int64_t cb, int64_t ch, int64_t ct, float scale) {
  const int seq = blockIdx.z, kh = blockIdx.y / parts,
            part = blockIdx.y % parts;
  const int start = offsets[seq], count = offsets[seq + 1] - start;
  constexpr int Threads = Rows * 8;
  const int first = blockIdx.x * Rows;
  if (first >= count * 6)
    return;
  const int length = lengths[seq], context = length - count;
  const int span = ((length + parts * 32 - 1) / (parts * 32)) * 32;
  const int begin = part * span, end = min(length, begin + span);
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int group = lane / 4, tid = lane % 4;
  const int row_base = (warp / 4) * 16;
  __shared__ Shared<Rows> s;
  for (int i = threadIdx.x; i < Rows * 256; i += Threads) {
    const int row = i / 256, d = i % 256, r = first + row;
    s.q[row][d] = r < count * 6
                      ? q[(start + r / 6) * qt + (kh * 6 + r % 6) * qh + d]
                      : __float2bfloat16(0.f);
  }
  if (threadIdx.x < Rows) {
    s.maximum[threadIdx.x] = -INFINITY;
    s.sum[threadIdx.x] = 0.f;
  }
  __syncthreads();
  uint32_t a[16][4];
#pragma unroll
  for (int t = 0; t < 16; ++t) {
    const int d = t * 16 + tid * 2;
    a[t][0] = pair(&s.q[row_base + group][d]);
    a[t][1] = pair(&s.q[row_base + group + 8][d]);
    a[t][2] = pair(&s.q[row_base + group][d + 8]);
    a[t][3] = pair(&s.q[row_base + group + 8][d + 8]);
  }
  float accum[8][4] = {};
  __syncthreads();
  if (begin < end)
    stage(s, cache, table, seq, kh, begin, length, block_size, table_stride, cb,
          ch, ct);
  for (int base = begin; base < end; base += 32) {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    uint32_t k[32 * 64 / Threads], v[32 * 64 / Threads];
#pragma unroll
    for (int j = 0; j < 32 * 64 / Threads; ++j) {
      const int i = threadIdx.x + j * Threads, row = i / 64, col = i % 64;
      k[j] = s.packed[row][col];
      v[j] = s.packed[row][65 + col];
    }
    if (threadIdx.x < 32) {
      s.ks[threadIdx.x] = __uint_as_float(s.packed[threadIdx.x][64]);
      s.vs[threadIdx.x] = __uint_as_float(s.packed[threadIdx.x][129]);
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 32 * 64 / Threads; ++j) {
      const int i = threadIdx.x + j * Threads;
      expand(k[j], &s.k[i / 64][(i % 64) * 4]);
      expand(v[j], &s.v[i / 64][(i % 64) * 4]);
    }
    __syncthreads();
    float score[4] = {};
#pragma unroll
    for (int t = 0; t < 16; ++t) {
      uint32_t b[2] = {
          pair(&s.k[(warp % 4) * 8 + group][t * 16 + tid * 2]),
          pair(&s.k[(warp % 4) * 8 + group][t * 16 + tid * 2 + 8])};
      mma(score, a[t], b);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int row = row_base + group + (j / 2) * 8,
                col = (warp % 4) * 8 + tid * 2 + j % 2;
      s.score[row][col] =
          base + col < length && base + col <= context + (first + row) / 6
              ? __fmul_rn(score[j], __fmul_rn(scale, s.ks[col]))
              : -INFINITY;
    }
    __syncthreads();
    if (base + 32 < end)
      stage(s, cache, table, seq, kh, base + 32, length, block_size,
            table_stride, cb, ch, ct);
    const int row = threadIdx.x / 8, sub = threadIdx.x % 8;
    float maximum = -INFINITY;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      maximum = fmaxf(maximum, s.score[row][sub + j * 8]);
#pragma unroll
    for (int mask = 4; mask; mask >>= 1)
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, mask));
    maximum = fmaxf(maximum, s.maximum[row]);
    const float decay =
        isfinite(maximum) ? expf(s.maximum[row] - maximum) : 0.f;
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int col = sub + j * 8;
      const float p =
          isfinite(maximum) ? expf(s.score[row][col] - maximum) : 0.f;
      sum += p;
      // A single BF16 rounding of P * V_scale introduces split-dependent
      // error before PV. Preserve its residual in a second tensor-core dot.
      const float scaled_p = p * s.vs[col];
      const bf16 hi = __float2bfloat16(scaled_p);
      s.p[row][col] = hi;
      s.p_residual[row][col] =
          __float2bfloat16(scaled_p - __bfloat162float(hi));
    }
#pragma unroll
    for (int mask = 4; mask; mask >>= 1)
      sum += __shfl_xor_sync(0xffffffff, sum, mask);
    if (!sub) {
      s.alpha[row] = decay;
      s.maximum[row] = maximum;
      s.sum[row] = s.sum[row] * decay + sum;
    }
    __syncthreads();
#pragma unroll
    for (int tile = 0; tile < 8; ++tile) {
#pragma unroll
      for (int j = 0; j < 4; ++j)
        accum[tile][j] *= s.alpha[row_base + group + (j / 2) * 8];
#pragma unroll
      for (int d = 0; d < 32; d += 16) {
        uint32_t p[4] = {pair(&s.p[row_base + group][d + tid * 2]),
                         pair(&s.p[row_base + group + 8][d + tid * 2]),
                         pair(&s.p[row_base + group][d + tid * 2 + 8]),
                         pair(&s.p[row_base + group + 8][d + tid * 2 + 8])};
        uint32_t residual[4] = {
            pair(&s.p_residual[row_base + group][d + tid * 2]),
            pair(&s.p_residual[row_base + group + 8][d + tid * 2]),
            pair(&s.p_residual[row_base + group][d + tid * 2 + 8]),
            pair(&s.p_residual[row_base + group + 8][d + tid * 2 + 8])};
        const int col = (warp % 4) * 64 + tile * 8 + group;
        uint32_t b[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int r = d + tid * 2 + j * 8;
          b[j] = uint32_t(__bfloat16_as_ushort(s.v[r][col])) |
                 (uint32_t(__bfloat16_as_ushort(s.v[r + 1][col])) << 16);
        }
        mma(accum[tile], residual, b);
        mma(accum[tile], p, b);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int tile = 0; tile < 8; ++tile) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int row = row_base + group + (j / 2) * 8, r = first + row;
      if (r < count * 6) {
        float *dst =
            scratch +
            (((int64_t(start + r / 6) * heads + kh * 6 + r % 6) * parts +
              part) *
             258);
        const int col = (warp % 4) * 64 + tile * 8 + tid * 2 + j % 2;
        dst[col] = accum[tile][j];
        if (!col) {
          dst[256] = s.maximum[row];
          dst[257] = s.sum[row];
        }
      }
    }
  }
}

__global__ void merge_kernel(const float *scratch, bf16 *out,
                             const int32_t *offsets, int sequences, int heads,
                             int parts, int64_t ot, int64_t oh) {
  const int token = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
  if (token >= offsets[sequences])
    return;
  const float *src = scratch + (int64_t(token) * heads + head) * parts * 258;
  float maximum = -INFINITY;
  for (int p = 0; p < parts; ++p)
    maximum = fmaxf(maximum, src[p * 258 + 256]);
  float sum = 0.f, value = 0.f;
  for (int p = 0; p < parts; ++p) {
    const float alpha = expf(src[p * 258 + 256] - maximum);
    sum += alpha * src[p * 258 + 257];
    value += alpha * src[p * 258 + d];
  }
  out[token * ot + head * oh + d] = __float2bfloat16(value / sum);
}
} // namespace

void paged_decode(const void *query, const int8_t *cache, const int32_t *table,
                  const int32_t *offsets, const int32_t *lengths, void *out,
                  float *scratch, int tokens, int heads, int kv_heads,
                  int parts, int block_size, int table_stride, int sequences,
                  int max_query, int64_t qt, int64_t qh, int64_t cb, int64_t ch,
                  int64_t ct, int64_t ot, int64_t oh, float scale,
                  cudaStream_t stream) {
  if (max_query <= 2) {
    split_kernel<16>
        <<<dim3((max_query * 6 + 15) / 16, kv_heads * parts, sequences), 128, 0,
           stream>>>(static_cast<const bf16 *>(query), cache, table, offsets,
                     lengths, scratch, heads, kv_heads, parts, block_size,
                     table_stride, qt, qh, cb, ch, ct, scale);
  } else {
    split_kernel<32>
        <<<dim3((max_query * 6 + 31) / 32, kv_heads * parts, sequences), 256, 0,
           stream>>>(static_cast<const bf16 *>(query), cache, table, offsets,
                     lengths, scratch, heads, kv_heads, parts, block_size,
                     table_stride, qt, qh, cb, ch, ct, scale);
  }
  merge_kernel<<<dim3(tokens, heads), 256, 0, stream>>>(
      scratch, static_cast<bf16 *>(out), offsets, sequences, heads, parts, ot,
      oh);
}
