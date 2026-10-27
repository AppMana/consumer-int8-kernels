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
template<bool IQ, int TM, bool MX=false, int XX=0> struct Storage {
  __nv_bfloat16 a[2][TM*LD];
  int8_t q[2][BN*(IQ ? BK/4 : MX ? BK/2 : BK)];
  uint8_t iq_scales[2][BN*(XX==4 ? 12 : XX==5 ? 16 : XX ? 8 : 4)];
  __half scales[2][BN*2];
  __nv_bfloat16 b[TM==16 ? 1 : BN*LD];
};
template<bool IQ, int TM, bool Routed, bool MX=false, int XX=0>
__device__ __forceinline__ void load(Storage<IQ,TM,MX,XX>& s, int stage, int start,
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
      auto qp=row<n ? q+(int64_t(group)*n+row)*((XX==2 || XX==4) ? k/8 : k/4)+
          ((XX==2 || XX==4) ? (start/128)*16 : start/4) : q;
      copy16(s.q[stage]+t*(BK/4),qp,row<n);
      const int dstride=((k/256)+1)&~1;
      auto dp=row<n ? scales+(int64_t(group)*n+row)*dstride+((start/256)&~1) : scales;
      copy4(s.scales[stage]+t*2,dp,row<n);
      if constexpr (XX==4 || XX==5) {
        constexpr int stride=XX==4 ? 12 : 16;
        constexpr int meta=XX==4 ? 48 : 44;
        auto base=row<n ? iq_scales+((int64_t(group)*n+row)*(k/256)+start/256)*meta : iq_scales;
        auto signs=row<n ? base+(start%256)/8 : base;
        copy4(s.iq_scales[stage]+t*stride,signs,row<n);
        copy4(s.iq_scales[stage]+t*stride+4,row<n ? signs+4 : signs,row<n);
        auto extra=row<n ? base+32+(XX==4 ? (start%256)/16 : ((start%256)/128)*4) : base;
        copy4(s.iq_scales[stage]+t*stride+8,extra,row<n);
        if constexpr (XX==5) copy4(s.iq_scales[stage]+t*stride+12,row<n ? base+40 : base,row<n);
      } else if constexpr (XX) {
        auto sp=row<n ? iq_scales+((int64_t(group)*n+row)*(k/32)+start/32)*4 : iq_scales;
        copy4(s.iq_scales[stage]+t*8,sp,row<n);
        copy4(s.iq_scales[stage]+t*8+4,row<n ? sp+4 : sp,row<n);
      } else {
      auto sp=row<n ? iq_scales+(int64_t(group)*n+row)*(k/32)+(start/128)*4 : iq_scales;
      copy4(s.iq_scales[stage]+t*4,sp,row<n);
      }
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
template<int TM, int XX=0>
__device__ __forceinline__ uint32_t decode_iq_pair(Storage<true,TM,false,XX>& s,
    const uint32_t* codebook,int stage,int start,int row,int col) {
  uint32_t signs,pair;
  int scale4;
  if constexpr (XX==4 || XX==5) {
    constexpr int stride=XX==4 ? 12 : 16;
    signs=s.iq_scales[stage][row*stride+col/8];
    int index;
    if constexpr (XX==4) {
      uint16_t highscale=*reinterpret_cast<const uint16_t*>(s.iq_scales[stage]+row*stride+8+(col/32)*2);
      int high=(highscale>>(((col%32)/8)*2))&3;
      scale4=(highscale>>(8+((col%32)/16)*4))&15;
      index=static_cast<uint8_t>(s.q[stage][row*(BK/4)+((start/64)%2)*8+col/8]) | (high<<8);
      pair=codebook[index*4+(col%8)/2];
    } else {
      uint8_t high=s.iq_scales[stage][row*stride+8+((start/64)%2)*2+col/32];
      high=(high>>((col%32)/4))&1;
      scale4=(s.iq_scales[stage][row*stride+12+(start%256)/64]>>((col/32)*4))&15;
      index=static_cast<uint8_t>(s.q[stage][row*(BK/4)+col/4]) | (int(high)<<8);
      pair=codebook[index*2+(col%4)/2];
    }
  } else if constexpr (XX) {
    uint32_t metadata=*reinterpret_cast<const uint32_t*>(s.iq_scales[stage]+row*8+(col/32)*4);
    signs=(metadata>>(((col%32)/8)*7))&127;
    scale4=metadata>>28;
    int index=static_cast<uint8_t>(s.q[stage][row*(BK/4)+
        (XX==2 ? ((start/64)%2)*8+col/8 : col/4)]);
    pair=codebook[index*(XX==2 ? 4 : 2)+(XX==2 ? (col%8)/2 : (col%4)/2)];
  } else {
    uint16_t code=*reinterpret_cast<const uint16_t*>(s.q[stage]+row*(BK/4)+(col/8)*2);
    signs=code>>9;
    pair=codebook[(code&511)*4+(col%8)/2];
    uint8_t sb=s.iq_scales[stage][row*4+((start/64)%2)*2+col/32];
    scale4=(sb>>(((col%32)/16)*4))&15;
  }
  if constexpr (XX!=4 && XX!=5) signs|=(__popc(signs)&1)<<7;
  float d=__half2float(s.scales[stage][row*2+((start/256)%2)])*
      (XX==5 ? float(1+2*scale4) : (0.5f+scale4)*(XX==3 ? 0.5f : 0.25f));
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
template<bool IQ, int TM=32, bool Routed=false, bool MX=false, bool Weighted=false, int XX=0>
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
  __shared__ __align__(32) Storage<IQ,TM,MX,XX> s;
  __shared__ uint32_t codebook[IQ ? (XX==4 ? 4096 : XX==5 ? 1024 : XX==2 ? 1024 : XX==3 ? 512 : 2048) : 1];
  int t=threadIdx.x, warp=t/32;
  if constexpr (IQ) {
    for (int i=t;i<(XX==4 ? 4096 : XX==5 ? 1024 : XX==2 ? 1024 : XX==3 ? 512 : 2048);i+=TM*8) {
      uint64_t grid;
      if constexpr (XX==4) grid=iq2s_grid[i/4];
      else if constexpr (XX==5) grid=iq3s_grid[i/2];
      else if constexpr (XX==2) grid=iq2xxs_grid[i/4];
      else if constexpr (XX==3) grid=iq3xxs_grid[i/2];
      else grid=iq2xs_grid[i/4];
      uint32_t pair=uint32_t(grid>>((i%((XX==3 || XX==5) ? 2 : 4))*16));
      codebook[i]=(__float_as_uint(float(pair&255))>>16) |
                   (__float_as_uint(float((pair>>8)&255))&0xffff0000u);
    }
  }
  float acc[2][4] = {};
  const int lane=t%32;
  load<IQ,TM,Routed,MX,XX>(s,0,0,x,q,scales,iq_scales,m,n,k,xs,xg,group,rowbase,tokens,xrows);
  asm volatile("cp.async.wait_group 0;");
  __syncthreads();
  for (int start=0, stage=0; start<k; start+=BK, stage^=1) {
    if (start+BK<k) load<IQ,TM,Routed,MX,XX>(s,stage^1,start+BK,x,q,scales,iq_scales,m,n,k,xs,xg,group,rowbase,tokens,xrows);
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

namespace {
__device__ __forceinline__ int local_expert(int global,const int32_t* map,int globals,int experts) {
  int local=(global>=0 && global<globals) ? map[global] : -1;
  return local>=0 && local<experts ? local : -1;
}
__global__ void route_count_init(const int32_t* ids,const int32_t* map,int32_t* info,
    int32_t* tokens,float* weights,int32_t* inverse,int32_t* tasks,
    int pairs,int globals,int experts,int capacity,int task_capacity) {
  int t=threadIdx.x,e=blockIdx.x;
  int index=e*256+t, stride=gridDim.x*256;
  for (int64_t i=index;i<capacity;i+=stride) { tokens[i]=-1; weights[i]=0; }
  for (int64_t i=index;i<pairs;i+=stride) inverse[i]=-1;
  for (int64_t i=index;i<task_capacity*2;i+=stride) tasks[i]=-1;
  int count=0;
  for (int i=t;i<pairs;i+=256) count+=local_expert(ids[i],map,globals,experts)==e;
  for (int offset=16;offset;offset/=2) count+=__shfl_down_sync(0xffffffff,count,offset);
  __shared__ int sums[8];
  if (t%32==0) sums[t/32]=count;
  __syncthreads();
  if (t==0) {
    int total=0;
    for (int w=0;w<8;++w) total+=sums[w];
    info[e*2]=total;
  }
}
__global__ void route_offsets(int32_t* info,int32_t* tasks,int experts,int tile) {
  if (threadIdx.x==0) {
    int offset=0;
    for (int e=0;e<experts;++e) {
      info[e*2+1]=offset;
      offset+=((info[e*2]+tile-1)/tile)*tile;
    }
  }
  __syncthreads();
  for (int e=threadIdx.x;e<experts;e+=blockDim.x) {
    int offset=info[e*2+1];
    for (int r=0;r<info[e*2];r+=tile) {
      int task=(offset+r)/tile;
      tasks[task*2]=e;
      tasks[task*2+1]=offset+r;
    }
  }
}
__global__ void route_assign(const int32_t* ids,const float* top_weights,const int32_t* map,
    const int32_t* info,int32_t* tokens,float* weights,int32_t* inverse,
    int pairs,int topk,int globals,int experts) {
  int t=threadIdx.x,e=blockIdx.x,warp=t/32,lane=t%32;
  if (!info[e*2]) return;
  __shared__ int warp_offsets[8], cursor;
  if (t==0) cursor=0;
  __syncthreads();
  for (int start=0;start<pairs;start+=256) {
    int i=start+t;
    bool match=i<pairs && local_expert(ids[i],map,globals,experts)==e;
    uint32_t ballot=__ballot_sync(0xffffffff,match);
    if (lane==0) warp_offsets[warp]=__popc(ballot);
    __syncthreads();
    if (t==0) {
      int running=cursor;
      for (int w=0;w<8;++w) {
        int count=warp_offsets[w];
        warp_offsets[w]=running;
        running+=count;
      }
      cursor=running;
    }
    __syncthreads();
    if (match) {
      uint32_t lower=(uint32_t(1)<<lane)-1;
      int row=info[e*2+1]+warp_offsets[warp]+__popc(ballot&lower);
      tokens[row]=i/topk;
      weights[row]=top_weights[i];
      inverse[i]=row;
    }
    __syncthreads();
  }
}
}
void gguf_plan_routes(const int32_t* ids,const float* top_weights,const int32_t* map,
    int32_t* info,int32_t* tokens,float* weights,int32_t* inverse,int32_t* tasks,
    int pairs,int topk,int globals,int experts,int capacity,int task_capacity,int tile,
    cudaStream_t stream) {
  route_count_init<<<experts,256,0,stream>>>(ids,map,info,tokens,weights,inverse,tasks,
      pairs,globals,experts,capacity,task_capacity);
  route_offsets<<<1,256,0,stream>>>(info,tasks,experts,tile);
  route_assign<<<experts,256,0,stream>>>(ids,top_weights,map,info,tokens,weights,inverse,
      pairs,topk,globals,experts);
}

namespace {
template<int XX,int TM,bool Weighted>
void launch_xxs(const void* x,const uint8_t* q,const void* bases,const uint8_t* metadata,
    const int32_t* tokens,const int32_t* tasks,const float* weights,void* out,
    int rows,int n,int k,int experts,int taskcount,int xrows,cudaStream_t stream) {
  packed_mm<true,TM,true,false,Weighted,XX><<<dim3((n+BN-1)/BN,taskcount),TM*8,0,stream>>>(
      static_cast<const __nv_bfloat16*>(x),reinterpret_cast<const int8_t*>(q),
      static_cast<const __half*>(bases),metadata,static_cast<__nv_bfloat16*>(out),
      rows,n,k,1,k,0,tokens,tasks,xrows,experts,weights);
}
template<int XX,int TM>
void launch_xxs_weight(const void* x,const uint8_t* q,const void* bases,const uint8_t* metadata,
    const int32_t* tokens,const int32_t* tasks,const float* weights,void* out,
    int rows,int n,int k,int experts,int taskcount,int xrows,bool weighted,cudaStream_t stream) {
  if (weighted) launch_xxs<XX,TM,true>(x,q,bases,metadata,tokens,tasks,weights,out,rows,n,k,experts,taskcount,xrows,stream);
  else launch_xxs<XX,TM,false>(x,q,bases,metadata,tokens,tasks,weights,out,rows,n,k,experts,taskcount,xrows,stream);
}
}
void gguf_iq_xxs_routed_mm(const void* x,const uint8_t* q,const void* bases,const uint8_t* metadata,
    const int32_t* tokens,const int32_t* tasks,const float* weights,void* out,
    int rows,int n,int k,int experts,int taskcount,int xrows,int bits,int tile,bool weighted,cudaStream_t stream) {
  if (!rows || !n || !taskcount) return;
  if (bits==4 && tile==16) launch_xxs_weight<4,16>(x,q,bases,metadata,tokens,tasks,weights,out,rows,n,k,experts,taskcount,xrows,weighted,stream);
  else if (bits==4) launch_xxs_weight<4,32>(x,q,bases,metadata,tokens,tasks,weights,out,rows,n,k,experts,taskcount,xrows,weighted,stream);
  else if (bits==5 && tile==16) launch_xxs_weight<5,16>(x,q,bases,metadata,tokens,tasks,weights,out,rows,n,k,experts,taskcount,xrows,weighted,stream);
  else if (bits==5) launch_xxs_weight<5,32>(x,q,bases,metadata,tokens,tasks,weights,out,rows,n,k,experts,taskcount,xrows,weighted,stream);
  else if (bits==2 && tile==16) launch_xxs_weight<2,16>(x,q,bases,metadata,tokens,tasks,weights,out,rows,n,k,experts,taskcount,xrows,weighted,stream);
  else if (bits==2) launch_xxs_weight<2,32>(x,q,bases,metadata,tokens,tasks,weights,out,rows,n,k,experts,taskcount,xrows,weighted,stream);
  else if (tile==16) launch_xxs_weight<3,16>(x,q,bases,metadata,tokens,tasks,weights,out,rows,n,k,experts,taskcount,xrows,weighted,stream);
  else launch_xxs_weight<3,32>(x,q,bases,metadata,tokens,tasks,weights,out,rows,n,k,experts,taskcount,xrows,weighted,stream);
}
