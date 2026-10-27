// SPDX-License-Identifier: Apache-2.0
// Original signed codes and FP16 scales, independently contiguous. No requantization.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
namespace {
using namespace nvcuda;
constexpr int BM=32, BN=64, BK=64, LD=72;
__device__ __forceinline__ void copy16(void* dst, const void* src, bool valid) {
  auto address = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
      "r"(address), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void copy4(void* dst, const void* src, bool valid) {
  auto address = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::
      "r"(address), "l"(src), "r"(valid ? 4 : 0));
}
struct Storage {
  __nv_bfloat16 a[2][BM*LD];
  int8_t q[2][BN*BK];
  __half scales[2][BN*2];
  union { __nv_bfloat16 b[BN*LD]; float c[BM*BN]; } tile;
};
__device__ __forceinline__ void load(Storage& s, int stage, int start,
    const __nv_bfloat16* x, const int8_t* q, const __half* scales,
    int m, int n, int k, int64_t xs, int64_t xg) {
  int t=threadIdx.x, row=blockIdx.y*BM+t/8, col=(t%8)*8;
  auto xp = row < m ? x+int64_t(row)*xs+int64_t(blockIdx.z)*xg+start+col : x;
  copy16(s.a[stage]+(t/8)*LD+col, xp, row<m);
  row=blockIdx.x*BN+t/4; col=(t%4)*16;
  auto qp = row<n ? q+(int64_t(blockIdx.z)*n+row)*k+start+col : q;
  copy16(s.q[stage]+(t/4)*BK+col, qp, row<n);
  if (t<BN) {
    row=blockIdx.x*BN+t;
    auto sp = row<n ? scales+(int64_t(blockIdx.z)*n+row)*(k/32)+start/32 : scales;
    copy4(s.scales[stage]+t*2, sp, row<n);
  }
  asm volatile("cp.async.commit_group;");
}
__global__ void packed_q8(const __nv_bfloat16* x, const int8_t* q,
    const __half* scales, __nv_bfloat16* y, int m, int n, int k, int groups,
    int64_t xs, int64_t xg) {
  __shared__ __align__(32) Storage s;
  int t=threadIdx.x, warp=t/32;
  wmma::fragment<wmma::accumulator,16,16,16,float> acc;
  wmma::fill_fragment(acc,0.f);
  load(s,0,0,x,q,scales,m,n,k,xs,xg);
  asm volatile("cp.async.wait_group 0;");
  __syncthreads();
  for (int start=0, stage=0; start<k; start+=BK, stage^=1) {
    if (start+BK<k) load(s,stage^1,start+BK,x,q,scales,m,n,k,xs,xg);
    for (int i=t; i<BN*BK/2; i+=256) {
      int row=i/(BK/2), col=(i%(BK/2))*2;
      auto pair=*reinterpret_cast<const uint16_t*>(s.q[stage]+row*BK+col);
      float scale=__half2float(s.scales[stage][row*2+col/32]);
      auto value=__floats2bfloat162_rn(float(int8_t(pair&255))*scale,
                                     float(int8_t(pair>>8))*scale);
      *reinterpret_cast<__nv_bfloat162*>(s.tile.b+row*LD+col)=value;
    }
    __syncthreads();
    #pragma unroll
    for (int kk=0; kk<BK; kk+=16) {
      wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
      wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b;
      wmma::load_matrix_sync(a,s.a[stage]+(warp/4)*16*LD+kk,LD);
      wmma::load_matrix_sync(b,s.tile.b+(warp%4)*16*LD+kk,LD);
      wmma::mma_sync(acc,a,b,acc);
    }
    __syncthreads();
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
  }
  wmma::store_matrix_sync(s.tile.c+(warp/4)*16*BN+(warp%4)*16,acc,BN,wmma::mem_row_major);
  __syncthreads();
  for (int i=t;i<BM*BN;i+=256) {
    int row=blockIdx.y*BM+i/BN,col=blockIdx.x*BN+i%BN;
    if (row<m && col<n)
      y[(int64_t(row)*groups+blockIdx.z)*n+col]=__float2bfloat16_rn(s.tile.c[i]);
  }
}
}
void gguf_q8_packed_mm(const void* x,const int8_t* q,const void* scales,void* y,
    int m,int n,int k,int groups,int64_t xs,int64_t xg,cudaStream_t stream) {
  if (!m || !n || !groups) return;
  packed_q8<<<dim3((n+BN-1)/BN,(m+BM-1)/BM,groups),256,0,stream>>>(
      static_cast<const __nv_bfloat16*>(x),q,static_cast<const __half*>(scales),
      static_cast<__nv_bfloat16*>(y),m,n,k,groups,xs,xg);
}
