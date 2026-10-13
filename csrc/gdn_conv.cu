// SPDX-License-Identifier: Apache-2.0
#include "kernels.h"
#include <cuda_bf16.h>

namespace {
using bf16 = __nv_bfloat16;
constexpr int Tile = 16;
struct alignas(8) Pack4 {
  bf16 values[4];
};

__device__ __forceinline__ float sum_warp(float x) {
#pragma unroll
  for (int shift = 16; shift; shift >>= 1)
    x += __shfl_xor_sync(0xffffffff, x, shift);
  return x;
}

__device__ __forceinline__ void copy_async(bf16 *dst, const bf16 *src,
                                           bool valid) {
  const uint32_t shared = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(shared),
               "l"(src), "r"(valid ? 16 : 0)
               : "memory");
}

template <bool QK, bool FloatBias>
__device__ __forceinline__ void
conv_head(const bf16 *x, const bf16 *weight, bf16 *state,
          const int32_t *offsets, const int32_t *slots, const bool *initial,
          const bf16 *a, const bf16 *b, const float *log_a, const void *dt,
          bf16 *q, bf16 *k, bf16 *v, float *g, float *beta, int hq, int hv,
          int num_slots, int64_t xs, int64_t as, int64_t bs, int64_t ss,
          int64_t sd, int64_t st, int64_t slot_stride, int head,
          bf16 *shared_input) {
  constexpr int Planes = QK ? 2 : 1;
  const int seq = blockIdx.z;
  const int begin = offsets[seq], length = offsets[seq + 1] - begin;
  const int first = blockIdx.x * Tile, slot = slots[seq * slot_stride];
  if (first >= length || slot <= 0 || slot >= num_slots)
    return;
  const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  const int channel = QK ? head * 128 : (2 * hq + head) * 128;
  auto input = reinterpret_cast<bf16(*)[Tile + 3][128]>(shared_input);

#pragma unroll
  for (int plane = 0; plane < Planes; ++plane) {
    const int base_channel = channel + plane * hq * 128;
    for (int i = threadIdx.x; i < (Tile + 3) * 16; i += 128) {
      const int row = i / 16, col = (i % 16) * 8;
      const int pos = first + row - 3;
      if (pos >= 0)
        copy_async(&input[plane][row][col],
                   x + (begin + min(pos, length - 1)) * xs + base_channel + col,
                   pos < length);
    }
    if (first == 0) {
      const bool has_state = initial[seq];
      for (int i = threadIdx.x; i < 3 * 128; i += 128) {
        const int row = i / 128, col = i % 128;
        input[plane][row][col] =
            has_state ? state[slot * ss + (base_channel + col) * sd + row * st]
                      : __float2bfloat16(0.f);
      }
    }
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
  float w[Planes][4][4];
#pragma unroll
  for (int plane = 0; plane < Planes; ++plane) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const auto packed = *reinterpret_cast<const Pack4 *>(
          weight + (channel + plane * hq * 128 + lane * 4 + j) * 4);
#pragma unroll
      for (int tap = 0; tap < 4; ++tap)
        w[plane][j][tap] = __bfloat162float(packed.values[tap]);
    }
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();

  // Only the first tile touches recurrent history. All readers of that
  // head's old history are in this CTA and have completed their loads.
  if (first == 0) {
#pragma unroll
    for (int plane = 0; plane < Planes; ++plane) {
      const int base_channel = channel + plane * hq * 128;
      for (int i = threadIdx.x; i < 3 * 128; i += 128) {
        const int row = i / 128, col = i % 128;
        const int pos = length - 3 + row;
        const bf16 value = pos >= 0 ? x[(begin + pos) * xs + base_channel + col]
                                    : input[plane][pos + 3][col];
        state[slot * ss + (base_channel + col) * sd + row * st] = value;
      }
    }
  }

  for (int row = warp; row < min(Tile, length - first); row += 4) {
    const int token = begin + first + row;
    float result[Planes][4], inv[Planes];
#pragma unroll
    for (int plane = 0; plane < Planes; ++plane) {
      float values[4] = {};
#pragma unroll
      for (int tap = 0; tap < 4; ++tap) {
        const auto packed = *reinterpret_cast<const Pack4 *>(
            &input[plane][row + tap][lane * 4]);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          // Match Triton's BF16-rounded products and FP32 accumulation.
          const float product =
              __fmul_rn(__bfloat162float(packed.values[j]), w[plane][j][tap]);
          values[j] += __bfloat162float(__float2bfloat16(product));
        }
      }
      float squares = 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float value = __fdividef(values[j], 1.f + __expf(-values[j]));
        result[plane][j] = __bfloat162float(__float2bfloat16(value));
        squares += __fmul_rn(result[plane][j], result[plane][j]);
      }
      if constexpr (QK)
        inv[plane] = __fdividef(1.f, sqrtf(sum_warp(squares) + 1e-6f));
    }
    Pack4 first_out, second_out;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      if constexpr (QK) {
        first_out.values[j] = __float2bfloat16(result[0][j] * inv[0]);
        second_out.values[j] = __float2bfloat16(result[1][j] * inv[1]);
      } else {
        first_out.values[j] = __float2bfloat16(result[0][j]);
      }
    }
    if constexpr (QK) {
      *reinterpret_cast<Pack4 *>(q + (token * hq + head) * 128 + lane * 4) =
          first_out;
      *reinterpret_cast<Pack4 *>(k + (token * hq + head) * 128 + lane * 4) =
          second_out;
    } else {
      *reinterpret_cast<Pack4 *>(v + (token * hv + head) * 128 + lane * 4) =
          first_out;
    }
    if constexpr (!QK) {
      if (lane == 0) {
        const float bias =
            FloatBias ? static_cast<const float *>(dt)[head]
                      : __bfloat162float(static_cast<const bf16 *>(dt)[head]);
        const float value = __bfloat162float(a[token * as + head]) + bias;
        const float sp = value > 20.f ? value
                                      : fmaxf(value, 0.f) +
                                            __logf(1.f + __expf(-fabsf(value)));
        g[token * hv + head] = -__expf(log_a[head]) * sp;
        beta[token * hv + head] = __fdividef(
            1.f, 1.f + __expf(-__bfloat162float(b[token * bs + head])));
      }
    }
  }
}
template <bool FloatBias>
__global__ void gdn_conv_kernel(
    const bf16 *x, const bf16 *weight, bf16 *state, const int32_t *offsets,
    const int32_t *slots, const bool *initial, const bf16 *a, const bf16 *b,
    const float *log_a, const void *dt, bf16 *q, bf16 *k, bf16 *v, float *g,
    float *beta, int hq, int hv, int num_slots, int64_t xs, int64_t as,
    int64_t bs, int64_t ss, int64_t sd, int64_t st, int64_t slot_stride) {
  __shared__ __align__(16) bf16 input[2 * (Tile + 3) * 128];
#define HEAD(QK, Head)                                                         \
  conv_head<QK, FloatBias>(x, weight, state, offsets, slots, initial, a, b,    \
                           log_a, dt, q, k, v, g, beta, hq, hv, num_slots, xs, \
                           as, bs, ss, sd, st, slot_stride, Head, input)
  if (blockIdx.y < hq) {
    HEAD(true, blockIdx.y);
  } else {
    HEAD(false, blockIdx.y - hq);
  }
#undef HEAD
}
} // namespace

void gdn_conv(const void *x, const void *weight, void *state,
              const int32_t *offsets, const int32_t *slots, const bool *initial,
              const void *a, const void *b, const float *log_a, const void *dt,
              void *q, void *k, void *v, float *g, float *beta, int sequences,
              int max_query, int hq, int hv, int num_slots, int64_t xs,
              int64_t as, int64_t bs, int64_t ss, int64_t sd, int64_t st,
              int64_t slot_stride, bool float_bias, cudaStream_t stream) {
  if (!sequences || !max_query)
    return;
#define LAUNCH(Bias)                                                           \
  gdn_conv_kernel<Bias>                                                        \
      <<<dim3((max_query + Tile - 1) / Tile, hq + hv, sequences), 128, 0,      \
         stream>>>(static_cast<const bf16 *>(x),                               \
                   static_cast<const bf16 *>(weight),                          \
                   static_cast<bf16 *>(state), offsets, slots, initial,        \
                   static_cast<const bf16 *>(a), static_cast<const bf16 *>(b), \
                   log_a, dt, static_cast<bf16 *>(q), static_cast<bf16 *>(k),  \
                   static_cast<bf16 *>(v), g, beta, hq, hv, num_slots, xs, as, \
                   bs, ss, sd, st, slot_stride)
  if (float_bias) {
    LAUNCH(true);
  } else {
    LAUNCH(false);
  }
#undef LAUNCH
}
