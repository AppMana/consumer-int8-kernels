// SPDX-License-Identifier: Apache-2.0
#include "kernels.h"
#include <cuda_bf16.h>

namespace {
using bf16 = __nv_bfloat16;

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int shift = 16; shift; shift >>= 1)
    x += __shfl_xor_sync(0xffffffff, x, shift);
  return x;
}

struct alignas(16) Pack8 {
  bf16 x[8];
};
struct alignas(8) Pack4 {
  bf16 x[4];
};

template <int Width, bool Residual>
__global__ void gemma_norm_kernel(const bf16 *x, const bf16 *residual,
                                  const bf16 *weight, bf16 *out,
                                  bf16 *residual_out, float epsilon) {
  constexpr int Threads = 128;
  constexpr int Packs = Width / (Threads * 8);
  float values[Packs][8], squares = 0.f;
  const int64_t row = int64_t(blockIdx.x) * Width;
#pragma unroll
  for (int i = 0; i < Packs; ++i) {
    const int col = (i * Threads + threadIdx.x) * 8;
    const Pack8 input = *reinterpret_cast<const Pack8 *>(x + row + col);
    Pack8 res{};
    if constexpr (Residual)
      res = *reinterpret_cast<const Pack8 *>(residual + row + col);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float value = __bfloat162float(input.x[j]);
      if constexpr (Residual)
        value += __bfloat162float(res.x[j]);
      values[i][j] = value;
      squares += __fmul_rn(value, value);
    }
  }
  __shared__ float sums[4];
  squares = warp_sum(squares);
  if ((threadIdx.x & 31) == 0)
    sums[threadIdx.x / 32] = squares;
  __syncthreads();
  squares = threadIdx.x < 4 ? sums[threadIdx.x] : 0.f;
  squares = warp_sum(squares);
  if (threadIdx.x == 0)
    sums[0] = rsqrtf(__fadd_rn(squares / Width, epsilon));
  __syncthreads();
  const float inv = sums[0];
#pragma unroll
  for (int i = 0; i < Packs; ++i) {
    const int col = (i * Threads + threadIdx.x) * 8;
    const Pack8 ws = *reinterpret_cast<const Pack8 *>(weight + col);
    Pack8 result, res;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float w = __bfloat162float(ws.x[j]) + 1.f;
      result.x[j] = __float2bfloat16(__fmul_rn(values[i][j] * inv, w));
      if constexpr (Residual)
        res.x[j] = __float2bfloat16(values[i][j]);
    }
    *reinterpret_cast<Pack8 *>(out + row + col) = result;
    if constexpr (Residual)
      *reinterpret_cast<Pack8 *>(residual_out + row + col) = res;
  }
}

// One warp handles one D128 head. Keep Q/K in registers: staging these
// single-use vectors in shared memory would add a barrier without reuse.
template <bool FloatBias>
__global__ void post_conv_kernel(const bf16 *x, const bf16 *a, const bf16 *b,
                                 const float *log_a, const void *dt, bf16 *q,
                                 bf16 *k, bf16 *v, float *g, float *beta,
                                 int tokens, int hq, int hv, int64_t xs,
                                 int64_t as, int64_t bs) {
  const int job = blockIdx.x * 4 + threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int token = job / (hq + hv), head = job % (hq + hv);
  if (token >= tokens)
    return;
  if (head < hq) {
    const Pack4 qp = *reinterpret_cast<const Pack4 *>(x + token * xs +
                                                      head * 128 + lane * 4);
    const Pack4 kp = *reinterpret_cast<const Pack4 *>(
        x + token * xs + (hq + head) * 128 + lane * 4);
    float qr[4], kr[4], qsum = 0.f, ksum = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      qr[j] = __bfloat162float(qp.x[j]);
      kr[j] = __bfloat162float(kp.x[j]);
      qsum += __fmul_rn(qr[j], qr[j]);
      ksum += __fmul_rn(kr[j], kr[j]);
    }
    const float qi = __fdividef(1.f, sqrtf(warp_sum(qsum) + 1e-6f));
    const float ki = __fdividef(1.f, sqrtf(warp_sum(ksum) + 1e-6f));
    Pack4 qo, ko;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      qo.x[j] = __float2bfloat16(qr[j] * qi);
      ko.x[j] = __float2bfloat16(kr[j] * ki);
    }
    *reinterpret_cast<Pack4 *>(q + (token * hq + head) * 128 + lane * 4) = qo;
    *reinterpret_cast<Pack4 *>(k + (token * hq + head) * 128 + lane * 4) = ko;
  } else {
    const int vh = head - hq;
    *reinterpret_cast<Pack4 *>(v + (token * hv + vh) * 128 + lane * 4) =
        *reinterpret_cast<const Pack4 *>(x + token * xs + (2 * hq + vh) * 128 +
                                         lane * 4);
    if (lane == 0) {
      const float bias =
          FloatBias ? static_cast<const float *>(dt)[vh]
                    : __bfloat162float(static_cast<const bf16 *>(dt)[vh]);
      const float value = __bfloat162float(a[token * as + vh]) + bias;
      const float softplus =
          value > 20.f
              ? value
              : fmaxf(value, 0.f) + __logf(1.f + __expf(-fabsf(value)));
      g[token * hv + vh] = -__expf(log_a[vh]) * softplus;
      beta[token * hv + vh] =
          __fdividef(1.f, 1.f + __expf(-__bfloat162float(b[token * bs + vh])));
    }
  }
}
} // namespace

void gemma_norm(const void *x, const void *residual, const void *weight,
                void *out, void *residual_out, int rows, float epsilon,
                cudaStream_t stream) {
  if (!rows)
    return;
  if (residual)
    gemma_norm_kernel<5120, true><<<rows, 128, 0, stream>>>(
        static_cast<const bf16 *>(x), static_cast<const bf16 *>(residual),
        static_cast<const bf16 *>(weight), static_cast<bf16 *>(out),
        static_cast<bf16 *>(residual_out), epsilon);
  else
    gemma_norm_kernel<5120, false>
        <<<rows, 128, 0, stream>>>(static_cast<const bf16 *>(x), nullptr,
                                   static_cast<const bf16 *>(weight),
                                   static_cast<bf16 *>(out), nullptr, epsilon);
}

void gdn_post_conv(const void *x, const void *a, const void *b,
                   const float *log_a, const void *dt, void *q, void *k,
                   void *v, float *g, float *beta, int tokens, int hq, int hv,
                   int64_t xs, int64_t as, int64_t bs, bool float_bias,
                   cudaStream_t stream) {
  if (!tokens)
    return;
  const int blocks = (tokens * (hq + hv) + 3) / 4;
#define LAUNCH(Bias)                                                           \
  post_conv_kernel<Bias><<<blocks, 128, 0, stream>>>(                          \
      static_cast<const bf16 *>(x), static_cast<const bf16 *>(a),              \
      static_cast<const bf16 *>(b), log_a, dt, static_cast<bf16 *>(q),         \
      static_cast<bf16 *>(k), static_cast<bf16 *>(v), g, beta, tokens, hq, hv, \
      xs, as, bs)
  if (float_bias) {
    LAUNCH(true);
  } else {
    LAUNCH(false);
  }
#undef LAUNCH
}
