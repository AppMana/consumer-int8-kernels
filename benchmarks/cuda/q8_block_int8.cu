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
// A CTA reuses a 128-wide K slab across 32x64 outputs. Four warps each
// compute 16x32, keeping the four N fragments in registers. The first
// schedule deliberately uses synchronous staging as a tiling control.
template<int BM>
__global__ void tiled(const int8_t* a, const int8_t* w, const float* sa,
                     const __half* sw, __nv_bfloat16* y, int m, int n, int k) {
  constexpr int Rows = BM + 64, Threads = BM * 4;
  __shared__ int8_t tile[Rows][144];
  __shared__ float scales[Rows][4];
  const int lane = threadIdx.x & 31, group = lane >> 2, tid = lane & 3;
  const int warp = threadIdx.x >> 5;
  const int rb = blockIdx.y * BM, cb = blockIdx.x * 64;
  const int wr = (warp / 2) * 16, wc = (warp % 2) * 32;
  float sum[4][4] = {};
  for (int base = 0; base < k; base += 128) {
    for (int index = threadIdx.x; index < Rows * 8; index += Threads) {
      int row = index / 8, col = (index % 8) * 16;
      int global = row < BM ? rb + row : cb + row - BM;
      const int8_t* src = row < BM ? a : w;
      int limit = row < BM ? m : n;
      int4 value = {};
      if (global < limit && base + col < k)
        value = *reinterpret_cast<const int4*>(src + global * k + base + col);
      *reinterpret_cast<int4*>(&tile[row][col]) = value;
    }
    for (int index = threadIdx.x; index < Rows * 4; index += Threads) {
      int row = index / 4, block = index % 4;
      int global = row < BM ? rb + row : cb + row - BM;
      float value = 0;
      if (base / 32 + block < k / 32) {
        if (row < BM && global < m) value = sa[global * (k / 32) + base / 32 + block];
        if (row >= BM && global < n) value = __half2float(sw[global * (k / 32) + base / 32 + block]);
      }
      scales[row][block] = value;
    }
    __syncthreads();
    #pragma unroll
    for (int block = 0; block < 4; ++block) {
      uint32_t av[4];
      #pragma unroll
      for (int r = 0; r < 4; ++r)
        av[r] = *reinterpret_cast<const uint32_t*>(&tile[wr + group + (r & 1) * 8][block * 32 + tid * 4 + (r >> 1) * 16]);
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        uint32_t bv[2];
        #pragma unroll
        for (int r = 0; r < 2; ++r)
          bv[r] = *reinterpret_cast<const uint32_t*>(&tile[BM + wc + j * 8 + group][block * 32 + tid * 4 + r * 16]);
        int acc[4] = {};
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3])
            : "r"(av[0]), "r"(av[1]), "r"(av[2]), "r"(av[3]), "r"(bv[0]), "r"(bv[1]));
        #pragma unroll
        for (int r = 0; r < 4; ++r)
          sum[j][r] += float(acc[r]) * (scales[wr + group + (r >> 1) * 8][block] *
                                      scales[BM + wc + j * 8 + tid * 2 + (r & 1)][block]);
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      int row = rb + wr + group + (r >> 1) * 8;
      int col = cb + wc + j * 8 + tid * 2 + (r & 1);
      if (row < m && col < n) y[row * n + col] = __float2bfloat16(sum[j][r]);
    }
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

extern "C" int q8_block_int8_tiled(const void* x, void* a, float* sa,
                                   const void* w, const void* sw, void* y,
                                   int m, int n, int k, cudaStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0 || k % 32) return cudaErrorInvalidValue;
  quantize<<<(m * (k / 32) + 7) / 8, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<int8_t*>(a), sa, m * (k / 32));
  tiled<32><<<dim3((n + 63) / 64, (m + 31) / 32), 128, 0, stream>>>(
      static_cast<const int8_t*>(a), static_cast<const int8_t*>(w), sa,
      static_cast<const __half*>(sw), static_cast<__nv_bfloat16*>(y), m, n, k);
  return cudaGetLastError();
}

extern "C" int q8_block_int8_tiled64(const void* x, void* a, float* sa,
                                     const void* w, const void* sw, void* y,
                                     int m, int n, int k, cudaStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0 || k % 32) return cudaErrorInvalidValue;
  quantize<<<(m * (k / 32) + 7) / 8, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<int8_t*>(a), sa, m * (k / 32));
  tiled<64><<<dim3((n + 63) / 64, (m + 63) / 64), 256, 0, stream>>>(
      static_cast<const int8_t*>(a), static_cast<const int8_t*>(w), sa,
      static_cast<const __half*>(sw), static_cast<__nv_bfloat16*>(y), m, n, k);
  return cudaGetLastError();
}
