// SPDX-License-Identifier: Apache-2.0
#include "kernels.h"
#include <cuda_bf16.h>

namespace {
// One warp owns one output channel. The small batch shares each weight load,
// keeping the canonical signed row-major representation used by CUTLASS.
template <int Rows>
__global__ void channel_gemv_kernel(const __nv_bfloat16* x, const int8_t* weight,
    const __nv_bfloat16* scales, __nv_bfloat16* out, int m, int n, int k) {
  const int col = blockIdx.x * 4 + threadIdx.x / 32;
  if (col >= n) return;
  const int lane = threadIdx.x & 31;
  const float scale = __bfloat162float(scales[col]);
  float accum[Rows] = {};
  for (int base = lane * 16; base < k; base += 512) {
    // K is a multiple of 16, so the final vector never crosses a row boundary.
    const uint4 packed = *reinterpret_cast<const uint4*>(weight + col * k + base);
    const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
    float w[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int8_t code = static_cast<int8_t>(words[j / 4] >> ((j % 4) * 8));
      // Match AllSpark's BF16 dequantized multiplicand before FP32 accumulation.
      w[j] = __bfloat162float(__float2bfloat16(float(code) * scale));
    }
    #pragma unroll
    for (int row = 0; row < Rows; ++row) {
      if (row < m) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
          const auto pair = *reinterpret_cast<const __nv_bfloat162*>(x + row * k + base + j * 2);
          const float2 values = __bfloat1622float2(pair);
          accum[row] = fmaf(values.x, w[j * 2], accum[row]);
          accum[row] = fmaf(values.y, w[j * 2 + 1], accum[row]);
        }
      }
    }
  }
  #pragma unroll
  for (int row = 0; row < Rows; ++row) {
    #pragma unroll
    for (int delta = 16; delta; delta >>= 1)
      accum[row] += __shfl_xor_sync(0xffffffffu, accum[row], delta);
    if (lane == 0 && row < m) out[row * n + col] = __float2bfloat16(accum[row]);
  }
}
}  // namespace

void channel_gemv(const void* x, const int8_t* weight, const void* scales,
    void* out, int m, int n, int k, cudaStream_t stream) {
  const auto* input = static_cast<const __nv_bfloat16*>(x);
  const auto* s = static_cast<const __nv_bfloat16*>(scales);
  auto* output = static_cast<__nv_bfloat16*>(out);
  if (m == 1) channel_gemv_kernel<1><<<(n + 3) / 4, 128, 0, stream>>>(input, weight, s, output, m, n, k);
  else if (m == 2) channel_gemv_kernel<2><<<(n + 3) / 4, 128, 0, stream>>>(input, weight, s, output, m, n, k);
  else if (m <= 4) channel_gemv_kernel<4><<<(n + 3) / 4, 128, 0, stream>>>(input, weight, s, output, m, n, k);
  else channel_gemv_kernel<8><<<(n + 3) / 4, 128, 0, stream>>>(input, weight, s, output, m, n, k);
}
