// SPDX-License-Identifier: Apache-2.0
// Original signed codes and FP16 scales, independently contiguous. No requantization.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "gguf_iq_tables.cuh"
namespace {
constexpr int BM=32, BN=64, BK=64, LD=64;
__device__ __forceinline__ int swizzle(int row, int col) {
  return row*LD + (col ^ ((row & 7)*8));
}
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
template<bool IQ> struct Storage {
  __nv_bfloat16 a[2][BM*LD];
  int8_t q[2][BN*(IQ ? BK/4 : BK)];
  uint8_t iq_scales[2][BN*4];
  __half scales[2][BN*2];
  __nv_bfloat16 b[BN*LD];
};
template<bool IQ>
__device__ __forceinline__ void load(Storage<IQ>& s, int stage, int start,
    const __nv_bfloat16* x, const int8_t* q, const __half* scales,
    const uint8_t* iq_scales, int m, int n, int k, int64_t xs, int64_t xg) {
  int t=threadIdx.x, row=blockIdx.y*BM+t/8, col=(t%8)*8;
  auto xp = row < m ? x+int64_t(row)*xs+int64_t(blockIdx.z)*xg+start+col : x;
  copy16(s.a[stage]+swizzle(t/8,col), xp, row<m);
  if constexpr (IQ) {
    if (t<BN) {
      row=blockIdx.x*BN+t;
      auto qp=row<n ? q+(int64_t(blockIdx.z)*n+row)*(k/4)+start/4 : q;
      copy16(s.q[stage]+t*(BK/4),qp,row<n);
      const int dstride=((k/256)+1)&~1;
      auto dp=row<n ? scales+(int64_t(blockIdx.z)*n+row)*dstride+((start/256)&~1) : scales;
      copy4(s.scales[stage]+t*2,dp,row<n);
      auto sp=row<n ? iq_scales+(int64_t(blockIdx.z)*n+row)*(k/32)+(start/128)*4 : iq_scales;
      copy4(s.iq_scales[stage]+t*4,sp,row<n);
    }
  } else {
    row=blockIdx.x*BN+t/4; col=(t%4)*16;
    auto qp = row<n ? q+(int64_t(blockIdx.z)*n+row)*k+start+col : q;
    copy16(s.q[stage]+(t/4)*BK+col, qp, row<n);
    if (t<BN) {
      row=blockIdx.x*BN+t;
      auto sp = row<n ? scales+(int64_t(blockIdx.z)*n+row)*(k/32)+start/32 : scales;
      copy4(s.scales[stage]+t*2, sp, row<n);
    }
  }
  asm volatile("cp.async.commit_group;");
}
template<bool IQ>
__global__ void packed_mm(const __nv_bfloat16* x, const int8_t* q,
    const __half* scales, const uint8_t* iq_scales, __nv_bfloat16* y, int m, int n, int k, int groups,
    int64_t xs, int64_t xg) {
  __shared__ __align__(32) Storage<IQ> s;
  __shared__ uint32_t codebook[IQ ? 2048 : 1];
  int t=threadIdx.x, warp=t/32;
  if constexpr (IQ) {
    for (int i=t;i<2048;i+=256) {
      uint64_t grid=iq2xs_grid[i/4];
      uint32_t pair=uint32_t(grid>>((i%4)*16));
      codebook[i]=(__float_as_uint(float(pair&255))>>16) |
                   (__float_as_uint(float((pair>>8)&255))&0xffff0000u);
    }
  }
  float acc[2][4] = {};
  const int lane=t%32;
  load(s,0,0,x,q,scales,iq_scales,m,n,k,xs,xg);
  asm volatile("cp.async.wait_group 0;");
  __syncthreads();
  for (int start=0, stage=0; start<k; start+=BK, stage^=1) {
    if (start+BK<k) load(s,stage^1,start+BK,x,q,scales,iq_scales,m,n,k,xs,xg);
    for (int i=t; i<BN*BK/2; i+=256) {
      int row=i/(BK/2), col=(i%(BK/2))*2;
      __nv_bfloat162 value;
      if constexpr (IQ) {
        uint16_t code=*reinterpret_cast<const uint16_t*>(s.q[stage]+row*(BK/4)+(col/8)*2);
        uint32_t signs=code>>9;
        signs|=(__popc(signs)&1)<<7;
        uint32_t pair=codebook[(code&511)*4+(col%8)/2];
        uint8_t sb=s.iq_scales[stage][row*4+((start/64)%2)*2+col/32];
        int scale4=(sb>>(((col%32)/16)*4))&15;
        float d=__half2float(s.scales[stage][row*2+((start/256)%2)])*(0.5f+scale4)*0.25f;
        float v0=d*__uint_as_float(pair<<16);
        float v1=d*__uint_as_float(pair&0xffff0000u);
        value=__floats2bfloat162_rn(v0,v1);
        uint32_t signpair=signs>>(col%8);
        uint32_t bits=(*reinterpret_cast<uint32_t*>(&value)) ^ ((signpair*0x40008000u)&0x80008000u);
        *reinterpret_cast<uint32_t*>(&value)=bits;
      } else {
        auto pair=*reinterpret_cast<const uint16_t*>(s.q[stage]+row*BK+col);
        float scale=__half2float(s.scales[stage][row*2+col/32]);
        value=__floats2bfloat162_rn(float(int8_t(pair&255))*scale,
                                  float(int8_t(pair>>8))*scale);
      }
      *reinterpret_cast<__nv_bfloat162*>(s.b+swizzle(row,col))=value;
    }
    __syncthreads();
    #pragma unroll
    for (int kk=0; kk<BK; kk+=16) {
      uint32_t a[4];
      uint32_t ap=static_cast<uint32_t>(__cvta_generic_to_shared(
          s.a[stage]+swizzle((warp/4)*16+lane%16,kk+(lane/16)*8)));
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
          : "=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]) : "r"(ap));
      #pragma unroll
      for (int nn=0;nn<2;++nn) {
        uint32_t b[2];
        uint32_t bp=static_cast<uint32_t>(__cvta_generic_to_shared(
            s.b+swizzle((warp%4)*16+nn*8+lane%8,kk+((lane/8)%2)*8)));
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
            : "=r"(b[0]),"=r"(b[1]) : "r"(bp));
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(acc[nn][0]),"+f"(acc[nn][1]),"+f"(acc[nn][2]),"+f"(acc[nn][3])
            : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
      }
    }
    __syncthreads();
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
  }
  #pragma unroll
  for (int nn=0;nn<2;++nn) {
    #pragma unroll
    for (int r=0;r<2;++r) {
      int row=blockIdx.y*BM+(warp/4)*16+lane/4+r*8;
      int col=blockIdx.x*BN+(warp%4)*16+nn*8+(lane%4)*2;
      if (row<m && col<n) {
        int64_t dst=(int64_t(row)*groups+blockIdx.z)*n+col;
        if (col+1<n && (dst%2)==0)
          *reinterpret_cast<__nv_bfloat162*>(y+dst)=
              __floats2bfloat162_rn(acc[nn][r*2],acc[nn][r*2+1]);
        else {
          y[dst]=__float2bfloat16_rn(acc[nn][r*2]);
          if (col+1<n) y[dst+1]=__float2bfloat16_rn(acc[nn][r*2+1]);
        }
      }
    }
  }

}
}
void gguf_q8_packed_mm(const void* x,const int8_t* q,const void* scales,void* y,
    int m,int n,int k,int groups,int64_t xs,int64_t xg,cudaStream_t stream) {
  if (!m || !n || !groups) return;
  packed_mm<false><<<dim3((n+BN-1)/BN,(m+BM-1)/BM,groups),256,0,stream>>>(
      static_cast<const __nv_bfloat16*>(x),q,static_cast<const __half*>(scales),nullptr,
      static_cast<__nv_bfloat16*>(y),m,n,k,groups,xs,xg);
}

void gguf_iq2_xs_mm(const void* x,const int8_t* q,const void* bases,
    const uint8_t* scales,void* y,int m,int n,int k,int groups,
    int64_t xs,int64_t xg,cudaStream_t stream) {
  if (!m || !n || !groups) return;
  packed_mm<true><<<dim3((n+BN-1)/BN,(m+BM-1)/BM,groups),256,0,stream>>>(
      static_cast<const __nv_bfloat16*>(x),q,static_cast<const __half*>(bases),scales,
      static_cast<__nv_bfloat16*>(y),m,n,k,groups,xs,xg);
}
