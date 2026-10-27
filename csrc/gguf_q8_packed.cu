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
template<bool IQ, int TM, bool MX=false> struct Storage {
  __nv_bfloat16 a[2][TM*LD];
  int8_t q[2][BN*(IQ ? BK/4 : MX ? BK/2 : BK)];
  uint8_t iq_scales[2][BN*4];
  __half scales[2][BN*2];
  __nv_bfloat16 b[TM==16 ? 1 : BN*LD];
};
template<bool IQ, int TM, bool Routed, bool MX=false>
__device__ __forceinline__ void load(Storage<IQ,TM,MX>& s, int stage, int start,
    const __nv_bfloat16* x, const int8_t* q, const __half* scales,
    const uint8_t* iq_scales, int m, int n, int k, int64_t xs, int64_t xg,
    int group, int rowbase, const int32_t* tokens, int xrows) {
  int t=threadIdx.x, row=rowbase+t/8, col=(t%8)*8;
  bool valid=row<m;
  int inputrow=row;
  if constexpr (Routed) {
    inputrow=valid ? tokens[row] : -1;
    valid=valid && inputrow>=0 && inputrow<xrows;
  }
  auto xp = valid ? x+int64_t(inputrow)*xs+int64_t(Routed ? 0 : group)*xg+start+col : x;
  copy16(s.a[stage]+swizzle(t/8,col), xp, valid);
  if constexpr (IQ) {
    if (t<BN) {
      row=blockIdx.x*BN+t;
      auto qp=row<n ? q+(int64_t(group)*n+row)*(k/4)+start/4 : q;
      copy16(s.q[stage]+t*(BK/4),qp,row<n);
      const int dstride=((k/256)+1)&~1;
      auto dp=row<n ? scales+(int64_t(group)*n+row)*dstride+((start/256)&~1) : scales;
      copy4(s.scales[stage]+t*2,dp,row<n);
      auto sp=row<n ? iq_scales+(int64_t(group)*n+row)*(k/32)+(start/128)*4 : iq_scales;
      copy4(s.iq_scales[stage]+t*4,sp,row<n);
    }
  } else if constexpr (MX) {
    if (t<BN*2) {
      row=blockIdx.x*BN+t/2; col=(t%2)*16;
      auto qp=row<n ? q+(int64_t(group)*n+row)*(k/2)+start/2+col : q;
      copy16(s.q[stage]+(t/2)*(BK/2)+col,qp,row<n);
    }
    if (t<BN) {
      row=blockIdx.x*BN+t;
      auto sp=row<n ? iq_scales+(int64_t(group)*n+row)*(k/32)+(start/128)*4 : iq_scales;
      copy4(s.iq_scales[stage]+t*4,sp,row<n);
    }
  } else {
    row=blockIdx.x*BN+t/4; col=(t%4)*16;
    auto qp = row<n ? q+(int64_t(group)*n+row)*k+start+col : q;
    copy16(s.q[stage]+(t/4)*BK+col, qp, row<n);
    if (t<BN) {
      row=blockIdx.x*BN+t;
      auto sp = row<n ? scales+(int64_t(group)*n+row)*(k/32)+start/32 : scales;
      copy4(s.scales[stage]+t*2, sp, row<n);
    }
  }
  asm volatile("cp.async.commit_group;");
}
template<int TM>
__device__ __forceinline__ uint32_t decode_iq_pair(Storage<true,TM>& s,
    const uint32_t* codebook,int stage,int start,int row,int col) {
  uint16_t code=*reinterpret_cast<const uint16_t*>(s.q[stage]+row*(BK/4)+(col/8)*2);
  uint32_t signs=code>>9;
  signs|=(__popc(signs)&1)<<7;
  uint32_t pair=codebook[(code&511)*4+(col%8)/2];
  uint8_t sb=s.iq_scales[stage][row*4+((start/64)%2)*2+col/32];
  int scale4=(sb>>(((col%32)/16)*4))&15;
  float d=__half2float(s.scales[stage][row*2+((start/256)%2)])*(0.5f+scale4)*0.25f;
  float v0=d*__uint_as_float(pair<<16);
  float v1=d*__uint_as_float(pair&0xffff0000u);
  auto value=__floats2bfloat162_rn(v0,v1);
  uint32_t signpair=signs>>(col%8);
  return (*reinterpret_cast<uint32_t*>(&value)) ^ ((signpair*0x40008000u)&0x80008000u);
}
template<int TM>
__device__ __forceinline__ uint32_t decode_mx_pair(Storage<false,TM,true>& s,
    int stage,int start,int row,int col) {
  const uint8_t* q=reinterpret_cast<const uint8_t*>(s.q[stage])+row*(BK/2);
  uint16_t pair=*reinterpret_cast<const uint16_t*>(q+(col/32)*16+col%16);
  int shift=((col%32)/16)*4;
  uint32_t c0=(pair>>shift)&15, c1=(pair>>(shift+8))&15;
  uint32_t e=s.iq_scales[stage][row*4+((start/64)%2)*2+col/32];
  uint32_t bits=e<2 ? 0x00200000u<<e : (e-1)<<23;
  float d=__uint_as_float(bits);
  float v0=d*float((0xc8643210u>>((c0&7)*4))&15);
  float v1=d*float((0xc8643210u>>((c1&7)*4))&15);
  auto value=__floats2bfloat162_rn(v0,v1);
  return (*reinterpret_cast<uint32_t*>(&value)) ^ ((c0&8)<<12) ^ ((c1&8)<<28);
}
template<bool IQ, int TM=32, bool Routed=false, bool MX=false, bool Weighted=false>
__global__ void packed_mm(const __nv_bfloat16* x, const int8_t* q,
    const __half* scales, const uint8_t* iq_scales, __nv_bfloat16* y, int m, int n, int k, int groups,
    int64_t xs, int64_t xg, const int32_t* tokens=nullptr,
    const int32_t* tasks=nullptr, int xrows=0, int experts=0,
    const float* route_weights=nullptr) {
  const int group=Routed ? tasks[blockIdx.y*2] : int(blockIdx.z);
  const int rowbase=Routed ? tasks[blockIdx.y*2+1] : int(blockIdx.y)*TM;
  if constexpr (Routed) {
    if (group<0 || group>=experts || rowbase<0 || rowbase>=m) return;
  }
  __shared__ __align__(32) Storage<IQ,TM,MX> s;
  __shared__ uint32_t codebook[IQ ? 2048 : 1];
  int t=threadIdx.x, warp=t/32;
  if constexpr (IQ) {
    for (int i=t;i<2048;i+=TM*8) {
      uint64_t grid=iq2xs_grid[i/4];
      uint32_t pair=uint32_t(grid>>((i%4)*16));
      codebook[i]=(__float_as_uint(float(pair&255))>>16) |
                   (__float_as_uint(float((pair>>8)&255))&0xffff0000u);
    }
  }
  float acc[2][4] = {};
  const int lane=t%32;
  load<IQ,TM,Routed,MX>(s,0,0,x,q,scales,iq_scales,m,n,k,xs,xg,group,rowbase,tokens,xrows);
  asm volatile("cp.async.wait_group 0;");
  __syncthreads();
  for (int start=0, stage=0; start<k; start+=BK, stage^=1) {
    if (start+BK<k) load<IQ,TM,Routed,MX>(s,stage^1,start+BK,x,q,scales,iq_scales,m,n,k,xs,xg,group,rowbase,tokens,xrows);
    if constexpr (TM!=16) {
    for (int i=t; i<BN*BK/2; i+=TM*8) {
      int row=i/(BK/2), col=(i%(BK/2))*2;
      __nv_bfloat162 value;
      if constexpr (IQ) {
        uint32_t bits=decode_iq_pair(s,codebook,stage,start,row,col);
        *reinterpret_cast<uint32_t*>(&value)=bits;
      } else if constexpr (MX) {
        uint32_t bits=decode_mx_pair(s,stage,start,row,col);
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
    }
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
        if constexpr (TM==16) {
          int row=(warp%4)*16+nn*8+lane/4;
          int col=kk+(lane%4)*2;
          if constexpr (MX) {
            b[0]=decode_mx_pair(s,stage,start,row,col);
            b[1]=decode_mx_pair(s,stage,start,row,col+8);
          } else {
            b[0]=decode_iq_pair(s,codebook,stage,start,row,col);
            b[1]=decode_iq_pair(s,codebook,stage,start,row,col+8);
          }
        } else {
        uint32_t bp=static_cast<uint32_t>(__cvta_generic_to_shared(
            s.b+swizzle((warp%4)*16+nn*8+lane%8,kk+((lane/8)%2)*8)));
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
            : "=r"(b[0]),"=r"(b[1]) : "r"(bp));
        }
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
      int row=rowbase+(warp/4)*16+lane/4+r*8;
      int col=blockIdx.x*BN+(warp%4)*16+nn*8+(lane%4)*2;
      if (row<m && col<n) {
        int64_t dst=(int64_t(row)*groups+(Routed ? 0 : group))*n+col;
        float v0=acc[nn][r*2], v1=acc[nn][r*2+1];
        if constexpr (Weighted) {
          float weight=route_weights[row];
          v0*=weight; v1*=weight;
        }
        if (col+1<n && (dst%2)==0)
          *reinterpret_cast<__nv_bfloat162*>(y+dst)=
              __floats2bfloat162_rn(v0,v1);
        else {
          y[dst]=__float2bfloat16_rn(v0);
          if (col+1<n) y[dst+1]=__float2bfloat16_rn(v1);
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

// Tasks hold (expert, padded row start), one tile of 16 or 32 rows per task.
// -1 expert denotes an inactive graph-capacity slot. Tokens map route rows to
// original activation rows; negative tokens are padding and produce zeros.
void gguf_iq2_xs_routed_mm(const void* x,const int8_t* q,const void* bases,
    const uint8_t* scales,const int32_t* tokens,const int32_t* tasks,void* y,
    int rows,int n,int k,int experts,int taskcount,int xrows,int tile_rows,cudaStream_t stream) {
  if (!rows || !n || !taskcount) return;
  if (tile_rows==32) {
    packed_mm<true,32,true><<<dim3((n+BN-1)/BN,taskcount),256,0,stream>>>(
        static_cast<const __nv_bfloat16*>(x),q,static_cast<const __half*>(bases),scales,
        static_cast<__nv_bfloat16*>(y),rows,n,k,1,k,0,tokens,tasks,xrows,experts);
    return;
  }
  packed_mm<true,16,true><<<dim3((n+BN-1)/BN,taskcount),128,0,stream>>>(
      static_cast<const __nv_bfloat16*>(x),q,static_cast<const __half*>(bases),scales,
      static_cast<__nv_bfloat16*>(y),rows,n,k,1,k,0,tokens,tasks,xrows,experts);
}

namespace {
__global__ void q8_expand(const int8_t* q,const __half* scales,
    __nv_bfloat16* out,int64_t pairs) {
  for (int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
       i<pairs;i+=int64_t(gridDim.x)*blockDim.x) {
    uint16_t codes=*reinterpret_cast<const uint16_t*>(q+i*2);
    float scale=__half2float(scales[i/16]);
    *reinterpret_cast<__nv_bfloat162*>(out+i*2)=__floats2bfloat162_rn(
        float(int8_t(codes&255))*scale,float(int8_t(codes>>8))*scale);
  }
}
}
void gguf_q8_expand(const int8_t* q,const void* scales,void* out,int64_t elements,
    cudaStream_t stream) {
  if (!elements) return;
  int64_t blocks=(elements/2+255)/256;
  if (blocks>65535) blocks=65535;
  q8_expand<<<blocks,256,0,stream>>>(q,static_cast<const __half*>(scales),
      static_cast<__nv_bfloat16*>(out),elements/2);
}

void gguf_mxfp4_routed_mm(const void* x,const uint8_t* q,const uint8_t* scales,
    const int32_t* tokens,const int32_t* tasks,void* y,int rows,int n,int k,
    int experts,int taskcount,int xrows,int tile_rows,cudaStream_t stream) {
  if (!rows || !n || !taskcount) return;
  if (tile_rows==32) {
    packed_mm<false,32,true,true><<<dim3((n+BN-1)/BN,taskcount),256,0,stream>>>(
        static_cast<const __nv_bfloat16*>(x),reinterpret_cast<const int8_t*>(q),nullptr,scales,
        static_cast<__nv_bfloat16*>(y),rows,n,k,1,k,0,tokens,tasks,xrows,experts);
  } else {
    packed_mm<false,16,true,true><<<dim3((n+BN-1)/BN,taskcount),128,0,stream>>>(
        static_cast<const __nv_bfloat16*>(x),reinterpret_cast<const int8_t*>(q),nullptr,scales,
        static_cast<__nv_bfloat16*>(y),rows,n,k,1,k,0,tokens,tasks,xrows,experts);
  }
}

void gguf_mxfp4_routed_weighted_mm(const void* x,const uint8_t* q,const uint8_t* scales,
    const int32_t* tokens,const int32_t* tasks,const float* weights,void* y,
    int rows,int n,int k,int experts,int taskcount,int xrows,int tile_rows,cudaStream_t stream) {
  if (!rows || !n || !taskcount) return;
  if (tile_rows==32) {
    packed_mm<false,32,true,true,true><<<dim3((n+BN-1)/BN,taskcount),256,0,stream>>>(
        static_cast<const __nv_bfloat16*>(x),reinterpret_cast<const int8_t*>(q),nullptr,scales,
        static_cast<__nv_bfloat16*>(y),rows,n,k,1,k,0,tokens,tasks,xrows,experts,weights);
  } else {
    packed_mm<false,16,true,true,true><<<dim3((n+BN-1)/BN,taskcount),128,0,stream>>>(
        static_cast<const __nv_bfloat16*>(x),reinterpret_cast<const int8_t*>(q),nullptr,scales,
        static_cast<__nv_bfloat16*>(y),rows,n,k,1,k,0,tokens,tasks,xrows,experts,weights);
  }
}
namespace {
__global__ void swiglu(const __nv_bfloat16* x,__nv_bfloat16* out,int64_t count,int width,float limit) {
  for (int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;
       i+=int64_t(gridDim.x)*blockDim.x) {
    int64_t source=(i/width)*(2*int64_t(width))+i%width;
    float gate=fminf(__bfloat162float(x[source]),limit);
    float up=fminf(fmaxf(__bfloat162float(x[source+width]),-limit),limit);
    out[i]=__float2bfloat16_rn((gate/(1.f+expf(-gate)))*up);
  }
}
__global__ void route_reduce(const __nv_bfloat16* contributions,const int32_t* rows,
    __nv_bfloat16* out,int64_t count,int width,int topk,int routes) {
  for (int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;
       i+=int64_t(gridDim.x)*blockDim.x) {
    int64_t token=i/width;
    int col=i%width;
    float sum=0;
    for (int slot=0;slot<topk;++slot) {
      int row=rows[token*topk+slot];
      if (row>=0 && row<routes) sum+=__bfloat162float(contributions[int64_t(row)*width+col]);
    }
    out[i]=__float2bfloat16_rn(sum);
  }
}
int post_blocks(int64_t count) { return int((count+255)/256>65535 ? 65535 : (count+255)/256); }
}
void gguf_swiglu(const void* x,void* out,int64_t rows,int width,float limit,cudaStream_t stream) {
  if (!rows || !width) return;
  swiglu<<<post_blocks(rows*width),256,0,stream>>>(static_cast<const __nv_bfloat16*>(x),
      static_cast<__nv_bfloat16*>(out),rows*width,width,limit);
}
void gguf_route_reduce(const void* x,const int32_t* indices,void* out,
    int64_t tokens,int width,int topk,int routes,cudaStream_t stream) {
  if (!tokens || !width) return;
  route_reduce<<<post_blocks(tokens*width),256,0,stream>>>(static_cast<const __nv_bfloat16*>(x),
      indices,static_cast<__nv_bfloat16*>(out),tokens*width,width,topk,routes);
}
