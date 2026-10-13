// SPDX-License-Identifier: Apache-2.0
#include "silu_quant.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cmath>
namespace {
struct alignas(16) Bf16Pack8 { __nv_bfloat16 values[8]; };
struct alignas(8) Int8Pack8 { int8_t values[8]; };
template <int K, int Threads = 256>
__global__ void silu_quant_kernel(const __nv_bfloat16* x, int8_t* q,
                                  float* scales) {
  constexpr int Packs = (K + Threads * 8 - 1) / (Threads * 8);
  __shared__ float maxima[Threads / 32];
  const int64_t row = int64_t(blockIdx.x) * K;
  float values[Packs][8], maximum = 0.f;
#pragma unroll
  for (int i = 0; i < Packs; ++i) {
    const int col = (i * Threads + threadIdx.x) * 8;
    Bf16Pack8 packed{}, up{};
    if (col < K) {
      packed = *reinterpret_cast<const Bf16Pack8*>(x + row * 2 + col);
      up = *reinterpret_cast<const Bf16Pack8*>(x + row * 2 + K + col);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float gate = __bfloat162float(packed.values[j]);
      const float u = __bfloat162float(up.values[j]);
      const float denom = 1.f + expf(-gate);
      const float silu = __fdividef(gate, denom);
      values[i][j] = __bfloat162float(__float2bfloat16_rn(silu * u));
      maximum = fmaxf(maximum, fabsf(values[i][j]));
    }
  }
#pragma unroll
  for (int mask = 16; mask; mask >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, mask));
  if (!(threadIdx.x & 31)) maxima[threadIdx.x / 32] = maximum;
  __syncthreads();
  maximum = threadIdx.x < Threads / 32 ? maxima[threadIdx.x] : 0;
#pragma unroll
  for (int mask = 16; mask; mask >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, mask));
  if (!threadIdx.x) maxima[0] = fmaxf(maximum / 127.f, 1e-20f);
  __syncthreads();
  const float scale = maxima[0];
  if (!threadIdx.x) scales[blockIdx.x] = scale;
#pragma unroll
  for (int i = 0; i < Packs; ++i) {
    const int col = (i * Threads + threadIdx.x) * 8;
    if (col < K) {
      Int8Pack8 result;
#pragma unroll
      for (int j = 0; j < 8; ++j)
        result.values[j] = static_cast<int8_t>(max(-127, min(127,
            __float2int_rn(values[i][j] / scale))));
      *reinterpret_cast<Int8Pack8*>(q + row + col) = result;
    }
  }
}


__global__ void silu_mul_kernel(const __nv_bfloat16* x, __nv_bfloat16* out,
                                int64_t count, int k) {
  const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const int64_t row = index / k;
  const int col = index % k;
  const float gate = __bfloat162float(x[row * (2 * k) + col]);
  const float up = __bfloat162float(x[row * (2 * k) + k + col]);
  out[index] = __float2bfloat16_rn(__fdividef(gate, 1.f + expf(-gate)) * up);
}
}  // namespace

void silu_mul_quantize(const void* x, int8_t* codes, float* scales,
                       int m, int k, cudaStream_t stream) {
  if (m == 0) return;
  const auto* input = static_cast<const __nv_bfloat16*>(x);
  if (k == 8704) silu_quant_kernel<8704><<<m, 256, 0, stream>>>(input, codes, scales);
  else silu_quant_kernel<17408><<<m, 256, 0, stream>>>(input, codes, scales);
}

void silu_mul(const void* x, void* out, int m, int k, cudaStream_t stream) {
  const int64_t count = int64_t(m) * k;
  if (count == 0) return;
  silu_mul_kernel<<<(count + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count, k);
}
