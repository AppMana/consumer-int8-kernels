// SPDX-License-Identifier: Apache-2.0
#include "kernels.h"
#include <cuda_bf16.h>

namespace {
// Invert AllSpark's N32K16 permutation without dequantizing the weights.
// Each thread restores 16 contiguous K elements in a canonical output row.
__global__ void unpack_kernel(const uint8_t* packed, const __nv_bfloat16* scales,
    int8_t* weight, __nv_bfloat16* restored_scales, int n, int k) {
  const int vector = blockIdx.x * blockDim.x + threadIdx.x;
  if (vector >= n * (k / 16)) return;
  const int row = vector / (k / 16);
  const int tile = vector % (k / 16);
  const int packed_row = (row / 32) * 8 + row % 8;
  const int offset = packed_row * k * 4 + tile * 64
      + ((row % 32) / 16) * 8 + ((row % 16) / 8) * 4;
  const auto* source = reinterpret_cast<const uint32_t*>(packed + offset);
  const uint32_t a = source[0], b = source[4], c = source[8], d = source[12];
  uint4 restored;
  restored.x = ((a & 0xffffu) | (b << 16)) ^ 0x80808080u;
  restored.y = ((c & 0xffffu) | (d << 16)) ^ 0x80808080u;
  restored.z = ((a >> 16) | (b & 0xffff0000u)) ^ 0x80808080u;
  restored.w = ((c >> 16) | (d & 0xffff0000u)) ^ 0x80808080u;
  reinterpret_cast<uint4*>(weight)[vector] = restored;
  if (tile == 0)
    restored_scales[row] = scales[(row / 32) * 32 + (row % 8) * 4 + (row % 32) / 8];
}
}  // namespace

void unpack_allspark(const uint8_t* packed, const void* scales, int8_t* weight,
    void* restored_scales, int n, int k, cudaStream_t stream) {
  unpack_kernel<<<(n * (k / 16) + 255) / 256, 256, 0, stream>>>(packed,
      static_cast<const __nv_bfloat16*>(scales), weight,
      static_cast<__nv_bfloat16*>(restored_scales), n, k);
}
