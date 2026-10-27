// SPDX-License-Identifier: Apache-2.0
// Native GGUF Q8_0 -> BF16 tensor-core GEMM, including strided grouped WO-A.
// Unlike dequantize-then-GEMM, weights only expand in CTA shared memory.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

namespace {
using namespace nvcuda;
constexpr int BM = 32, BN = 64, BK = 32, LD = 40;
__global__ void q8_mm(const __nv_bfloat16* x, const uint8_t* w,
                      __nv_bfloat16* y, int m, int n, int k, int groups,
                      int64_t xs, int64_t xg) {
  __shared__ __align__(32) __nv_bfloat16 a[BM * LD];
  __shared__ __align__(32) __nv_bfloat16 b[BN * LD];
  __shared__ __align__(32) float c[BM * BN];
  const int tid = threadIdx.x, warp = tid / 32;
  const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
  const int group = blockIdx.z;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fill_fragment(acc, 0.0f);
  for (int start = 0; start < k; start += BK) {
    for (int i = tid; i < BM * BK; i += 256) {
      const int row = i / BK, col = i % BK;
      a[row * LD + col] = row0 + row < m
          ? x[int64_t(row0 + row) * xs + int64_t(group) * xg + start + col]
          : __float2bfloat16(0.0f);
    }
    for (int i = tid; i < BN * BK; i += 256) {
      const int row = i / BK, col = i % BK;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (col0 + row < n) {
        const int64_t block = ((int64_t(group) * n + col0 + row) * (k / 32) + start / 32) * 34;
        const float scale = __half2float(*reinterpret_cast<const __half*>(w + block));
        value = __float2bfloat16_rn(scale * float(static_cast<int8_t>(w[block + 2 + col])));
      }
      b[row * LD + col] = value;
    }
    __syncthreads();
    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
      wmma::load_matrix_sync(af, a + (warp / 4) * 16 * LD + kk, LD);
      wmma::load_matrix_sync(bf, b + (warp % 4) * 16 * LD + kk, LD);
      wmma::mma_sync(acc, af, bf, acc);
    }
    __syncthreads();
  }
  wmma::store_matrix_sync(c + (warp / 4) * 16 * BN + (warp % 4) * 16,
                          acc, BN, wmma::mem_row_major);
  __syncthreads();
  for (int i = tid; i < BM * BN; i += 256) {
    const int row = row0 + i / BN, col = col0 + i % BN;
    if (row < m && col < n)
      y[(int64_t(row) * groups + group) * n + col] = __float2bfloat16_rn(c[i]);
  }
}
}

void gguf_q8_mm(const void* x, const uint8_t* w, void* y,
                int m, int n, int k, int groups, int64_t xs, int64_t xg,
                cudaStream_t stream) {
  if (!m || !n || !groups) return;
  q8_mm<<<dim3((n + BN - 1) / BN, (m + BM - 1) / BM, groups), 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), w, static_cast<__nv_bfloat16*>(y),
      m, n, k, groups, xs, xg);
}
