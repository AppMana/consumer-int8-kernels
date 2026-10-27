// SPDX-License-Identifier: Apache-2.0
// Experimental plain-CUDA Q8_0 W8A8 candidate. No weight requantization.
// Activation scales and integer accumulation reset every 32 K elements.
// Fragment mapping follows channel_mm_kernel in csrc/kernels.cu.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
__global__ void quantize(const __nv_bfloat16* x, int8_t* q, float* scales,
                         int blocks) {
  const int lane = threadIdx.x & 31;
  const int block = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  if (block >= blocks) return;
  float v = __bfloat162float(x[block * 32 + lane]);
  float a = fabsf(v);
  for (int delta = 16; delta; delta /= 2)
    a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, delta));
  float s = fmaxf(a / 127.f, 1.e-20f);
  q[block * 32 + lane] = static_cast<int8_t>(__float2int_rn(v / s));
  if (!lane) scales[block] = s;
}

__global__ void matmul(const int8_t* a, const int8_t* w, const float* sa,
                       const __half* sw, __nv_bfloat16* y, int m, int n, int k) {
  const int lane = threadIdx.x & 31, group = lane >> 2, tid = lane & 3;
  const int rb = blockIdx.y * 16;
  const int cb = blockIdx.x * 32 + (threadIdx.x >> 5) * 8;
  float sum[4] = {};
  for (int offset = 0; offset < k; offset += 32) {
    uint32_t av[4], bv[2];
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      int row = rb + group + (r & 1) * 8;
      int col = offset + tid * 4 + (r >> 1) * 16;
      av[r] = row < m ? *reinterpret_cast<const uint32_t*>(a + row * k + col) : 0;
    }
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
      int col = cb + group;
      bv[r] = col < n ? *reinterpret_cast<const uint32_t*>(w + col * k + offset + tid * 4 + r * 16) : 0;
    }
    int acc[4] = {};
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3])
        : "r"(av[0]), "r"(av[1]), "r"(av[2]), "r"(av[3]), "r"(bv[0]), "r"(bv[1]));
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      int row = rb + group + (r >> 1) * 8;
      int col = cb + tid * 2 + (r & 1);
      if (row < m && col < n)
        sum[r] += float(acc[r]) * (sa[row * (k / 32) + offset / 32] *
                                  __half2float(sw[col * (k / 32) + offset / 32]));
    }
  }
  #pragma unroll
  for (int r = 0; r < 4; ++r) {
    int row = rb + group + (r >> 1) * 8;
    int col = cb + tid * 2 + (r & 1);
    if (row < m && col < n) y[row * n + col] = __float2bfloat16(sum[r]);
  }
}
}  // namespace

// Benchmark-only raw-pointer entry point; caller validates contiguous tensor
// storage and K % 32 == 0. Caller owns scratch and outputs, including in graphs.
extern "C" int q8_block_int8(const void* x, void* a, float* sa,
                             const void* w, const void* sw, void* y,
                             int m, int n, int k, cudaStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0 || k % 32) return cudaErrorInvalidValue;
  quantize<<<(m * (k / 32) + 7) / 8, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<int8_t*>(a), sa, m * (k / 32));
  matmul<<<dim3((n + 31) / 32, (m + 15) / 16), 128, 0, stream>>>(
      static_cast<const int8_t*>(a), static_cast<const int8_t*>(w), sa,
      static_cast<const __half*>(sw), static_cast<__nv_bfloat16*>(y), m, n, k);
  return cudaGetLastError();
}
