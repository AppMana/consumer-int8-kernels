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

namespace {
// Q6_K: 128 low-nibble bytes, 64 high-bit bytes, 16 signed scales,
// then the original FP16 base. Preserve FP32 base*scale*code rounding.
__device__ __forceinline__ __nv_bfloat162 q6_pair(const uint8_t* block, int col) {
  int half = col / 128, within = col % 128;
  uint16_t lo = *reinterpret_cast<const uint16_t*>(block + half*64 + within%64);
  uint16_t hi = *reinterpret_cast<const uint16_t*>(block + 128 + half*32 + within%32);
  int shift = (within/64)*4, high_shift = (within/32)*2;
  int q0 = ((lo >> shift)&15) | (((hi >> high_shift)&3)<<4);
  int q1 = ((lo >> (shift+8))&15) | (((hi >> (high_shift+8))&3)<<4);
  float d = __half2float(*reinterpret_cast<const __half*>(block+208));
  float scale = d * float(static_cast<int8_t>(block[192+col/16]));
  return __floats2bfloat162_rn(scale*float(q0-32), scale*float(q1-32));
}
__global__ void q6_expand(const uint8_t* raw, __nv_bfloat16* out, int64_t pairs) {
  for (int64_t pair=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
       pair<pairs; pair+=int64_t(gridDim.x)*blockDim.x) {
    *reinterpret_cast<__nv_bfloat162*>(out+pair*2) =
        q6_pair(raw+(pair/128)*210, (pair%128)*2);
  }
}
template<int Rows>
__global__ void q6_mm(const __nv_bfloat16* x, const uint8_t* raw,
    __nv_bfloat16* y, int m, int n, int k, int groups, int64_t xs, int64_t xg) {
  constexpr int Columns=64, Inner=64, Stride=72;
  __shared__ __align__(32) union {
    struct { __nv_bfloat16 a[Rows*Stride], b[Columns*Stride]; } tile;
    float output[Rows*Columns];
  } s;
  int tid=threadIdx.x, warp=tid/32;
  int rowbase=blockIdx.y*Rows, colbase=blockIdx.x*Columns, group=blockIdx.z;
  wmma::fragment<wmma::accumulator,16,16,16,float> acc[Rows/32];
  #pragma unroll
  for (int r=0;r<Rows/32;++r) wmma::fill_fragment(acc[r],0.f);
  for (int start=0;start<k;start+=Inner) {
    for (int i=tid;i<Rows*Inner/2;i+=256) {
      int row=i/(Inner/2), col=(i%(Inner/2))*2;
      __nv_bfloat162 value=__floats2bfloat162_rn(0.f,0.f);
      if (rowbase+row<m) value=*reinterpret_cast<const __nv_bfloat162*>(
          x+int64_t(rowbase+row)*xs+int64_t(group)*xg+start+col);
      *reinterpret_cast<__nv_bfloat162*>(s.tile.a+row*Stride+col)=value;
    }
    for (int i=tid;i<Columns*Inner/2;i+=256) {
      int row=i/(Inner/2), col=(i%(Inner/2))*2;
      __nv_bfloat162 value=__floats2bfloat162_rn(0.f,0.f);
      if (colbase+row<n) value=q6_pair(
          raw+((int64_t(group)*n+colbase+row)*(k/256)+start/256)*210,
          start%256+col);
      *reinterpret_cast<__nv_bfloat162*>(s.tile.b+row*Stride+col)=value;
    }
    __syncthreads();
    #pragma unroll
    for (int kk=0;kk<Inner;kk+=16) {
      wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> bf;
      wmma::load_matrix_sync(bf,s.tile.b+(warp%4)*16*Stride+kk,Stride);
      #pragma unroll
      for (int r=0;r<Rows/32;++r) {
        wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> af;
        wmma::load_matrix_sync(af,s.tile.a+((warp/4)*16+r*32)*Stride+kk,Stride);
        wmma::mma_sync(acc[r],af,bf,acc[r]);
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int r=0;r<Rows/32;++r)
    wmma::store_matrix_sync(s.output+((warp/4)*16+r*32)*Columns+(warp%4)*16,
        acc[r],Columns,wmma::mem_row_major);
  __syncthreads();
  for (int i=tid;i<Rows*Columns;i+=256) {
    int row=rowbase+i/Columns, col=colbase+i%Columns;
    if (row<m && col<n) y[(int64_t(row)*groups+group)*n+col]=__float2bfloat16_rn(s.output[i]);
  }
}
}
void gguf_q6_expand(const uint8_t* raw, void* out, int64_t elements, cudaStream_t stream) {
  if (!elements) return;
  int64_t blocks=(elements/2+255)/256;
  if (blocks>65535) blocks=65535;
  q6_expand<<<blocks,256,0,stream>>>(raw,static_cast<__nv_bfloat16*>(out),elements/2);
}
void gguf_q6_mm(const void* x,const uint8_t* raw,void* y,int m,int n,int k,int groups,
    int64_t xs,int64_t xg,int rows,cudaStream_t stream) {
  if (!m || !n || !groups) return;
  dim3 grid((n+63)/64,(m+rows-1)/rows,groups);
  if (rows==64) q6_mm<64><<<grid,256,0,stream>>>(static_cast<const __nv_bfloat16*>(x),raw,
      static_cast<__nv_bfloat16*>(y),m,n,k,groups,xs,xg);
  else q6_mm<32><<<grid,256,0,stream>>>(static_cast<const __nv_bfloat16*>(x),raw,
      static_cast<__nv_bfloat16*>(y),m,n,k,groups,xs,xg);
}
