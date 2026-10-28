// SPDX-License-Identifier: Apache-2.0
// Strided grouped variant of csrc/kernels.cu vector row quantization.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
namespace {
struct alignas(16) Bf16Pack8 { __nv_bfloat16 values[8]; };
struct alignas(8) Int8Pack8 { int8_t values[8]; };
template <int K, int Threads = 256>
__global__ void quantize_rows_vector_kernel(const __nv_bfloat16* x, int8_t* q,
                                           float* scales, int m, int groups) {
  constexpr int Packs = (K + Threads * 8 - 1) / (Threads * 8);
  __shared__ float maxima[Threads / 32];
  const int64_t row = int64_t(blockIdx.x % m * groups + blockIdx.x / m) * K;
  const int64_t out_row = int64_t(blockIdx.x) * K;
  float values[Packs][8], maximum = 0.f;
#pragma unroll
  for (int i = 0; i < Packs; ++i) {
    const int col = (i * Threads + threadIdx.x) * 8;
    Bf16Pack8 packed{};
    if (col < K) packed = *reinterpret_cast<const Bf16Pack8*>(x + row + col);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      values[i][j] = __bfloat162float(packed.values[j]);
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
      *reinterpret_cast<Int8Pack8*>(q + out_row + col) = result;
    }
  }
}

}
extern "C" int grouped_quant4096(const void* x, void* q, float* scales,
                                 int m, int groups, cudaStream_t stream) {
  if (m <= 0 || groups <= 0) return cudaErrorInvalidValue;
  quantize_rows_vector_kernel<4096><<<m * groups, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<int8_t*>(q), scales, m, groups);
  return cudaGetLastError();
}
