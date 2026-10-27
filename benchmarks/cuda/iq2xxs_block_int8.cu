// SPDX-License-Identifier: Apache-2.0
// Experimental IQ2_XXS compressed weights -> shared INT8 -> INT8 MMA.
// Original codebook values and per32 scales; only activations are quantized.
// Reuse the block32 activation quantizer and fragment mapping control.
#include "q8_block_int8.cu"
#include "../../csrc/gguf_iq_tables.cuh"
namespace {
__device__ __forceinline__ uint32_t signed_bytes(uint32_t v, uint32_t signs) {
  uint32_t mask = (signs & 1) * 0xffu | ((signs >> 1) & 1) * 0xff00u |
                  ((signs >> 2) & 1) * 0xff0000u | ((signs >> 3) & 1) * 0xff000000u;
  return __vsub4(v ^ mask, mask);
}
template<int BM, int BN=64, bool Packed=false, bool SharedLUT=false>
__global__ void iq2xxs_tiled(const int8_t* a, const uint8_t* w, const float* sa, __nv_bfloat16* y, int m, int n, int k) {
  constexpr int Rows = BM + BN, Threads = (BM / 16) * (BN / 32) * 32;
  __shared__ int8_t tile[Rows][144];
  __shared__ float scales[Rows][4];
  const int lane = threadIdx.x & 31, group = lane >> 2, tid = lane & 3;
  const int warp = threadIdx.x >> 5;
  const int rb = blockIdx.y * BM, cb = blockIdx.x * BN;
  const int wr = (warp / (BN / 32)) * 16, wc = (warp % (BN / 32)) * 32;
  __shared__ uint64_t lut[SharedLUT ? 256 : 1];
  if constexpr (SharedLUT) {
    for (int i = threadIdx.x; i < 256; i += Threads) lut[i] = iq2xxs_grid[i];
    __syncthreads();
  }
  float sum[4][4] = {};
  for (int base = 0; base < k; base += 128) {
    for (int index = threadIdx.x; index < BM * 8; index += Threads) {
      int row = index / 8, col = (index % 8) * 16;
      int4 value = {};
      if (rb + row < m)
        value = *reinterpret_cast<const int4*>(a + (rb + row) * k + base + col);
      *reinterpret_cast<int4*>(&tile[row][col]) = value;
    }
    for (int index = threadIdx.x; index < BM * 4; index += Threads) {
      int row = index / 4, block = index % 4;
      scales[row][block] = rb + row < m ? sa[(rb + row) * (k / 32) + base / 32 + block] : 0;
    }
    for (int index = threadIdx.x; index < BN * 4; index += Threads) {
      int row = index / 4, block = index % 4;
      uint64_t values[4] = {};
      float scale = 0;
      if (cb + row < n) {
        const uint8_t* raw = w + (int64_t(cb + row) * (k / 256) + base / 256) * 66;
        const uint8_t* sub = raw + 2 + ((base % 256) / 32 + block) * 8;
        uint32_t metadata = uint32_t(sub[4]) | (uint32_t(sub[5]) << 8) |
                            (uint32_t(sub[6]) << 16) | (uint32_t(sub[7]) << 24);
        scale = __half2float(*reinterpret_cast<const __half*>(raw)) *
                (0.5f + float(metadata >> 28)) * 0.25f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint32_t signs = (metadata >> (j * 7)) & 127;
          signs |= (__popc(signs) & 1) << 7;
          uint64_t grid = SharedLUT ? lut[sub[j]] : iq2xxs_grid[sub[j]];
          if constexpr (Packed) {
            values[j] = uint64_t(signed_bytes(uint32_t(grid), signs)) |
                        (uint64_t(signed_bytes(uint32_t(grid >> 32), signs >> 4)) << 32);
          } else {
          #pragma unroll
          for (int t = 0; t < 8; ++t) {
            int v = int((grid >> (t * 8)) & 255);
            if (signs & (1 << t)) v = -v;
            values[j] |= uint64_t(uint8_t(v)) << (t * 8);
          }
          }
        }
      }
      scales[BM + row][block] = scale;
      #pragma unroll
      for (int j = 0; j < 4; ++j)
        *reinterpret_cast<uint64_t*>(&tile[BM + row][block * 32 + j * 8]) = values[j];
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
extern "C" int iq2xxs_block_int8(const void* x, void* a, float* sa,
                                const void* raw, void* y, int m, int n, int k,
                                int tile_m, cudaStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0 || k % 256 || (tile_m != 16 && tile_m != 32 && tile_m != 64))
    return cudaErrorInvalidValue;
  quantize<<<(m * (k / 32) + 7) / 8, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<int8_t*>(a), sa, m * (k / 32));
  if (tile_m == 16)
    iq2xxs_tiled<16,32><<<dim3((n + 31) / 32, (m + 15) / 16), 32, 0, stream>>>(
        static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa,
        static_cast<__nv_bfloat16*>(y), m, n, k);
  else if (tile_m == 32)
    iq2xxs_tiled<32><<<dim3((n + 63) / 64, (m + 31) / 32), 128, 0, stream>>>(
        static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa,
        static_cast<__nv_bfloat16*>(y), m, n, k);
  else
    iq2xxs_tiled<64><<<dim3((n + 63) / 64, (m + 63) / 64), 256, 0, stream>>>(
        static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa,
        static_cast<__nv_bfloat16*>(y), m, n, k);
  return cudaGetLastError();
}

extern "C" int iq2xxs_block_int8_packed(const void* x, void* a, float* sa,
                                const void* raw, void* y, int m, int n, int k,
                                int tile_m, cudaStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0 || k % 256 || (tile_m != 16 && tile_m != 32 && tile_m != 64))
    return cudaErrorInvalidValue;
  quantize<<<(m * (k / 32) + 7) / 8, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<int8_t*>(a), sa, m * (k / 32));
  if (tile_m == 16)
    iq2xxs_tiled<16,32,true><<<dim3((n + 31) / 32, (m + 15) / 16), 32, 0, stream>>>(
        static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa,
        static_cast<__nv_bfloat16*>(y), m, n, k);
  else if (tile_m == 32)
    iq2xxs_tiled<32,64,true><<<dim3((n + 63) / 64, (m + 31) / 32), 128, 0, stream>>>(
        static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa,
        static_cast<__nv_bfloat16*>(y), m, n, k);
  else
    iq2xxs_tiled<64,64,true><<<dim3((n + 63) / 64, (m + 63) / 64), 256, 0, stream>>>(
        static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa,
        static_cast<__nv_bfloat16*>(y), m, n, k);
  return cudaGetLastError();
}

extern "C" int iq2xxs_block_int8_lut(const void* x, void* a, float* sa,
                                const void* raw, void* y, int m, int n, int k,
                                int tile_m, cudaStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0 || k % 256 || (tile_m != 16 && tile_m != 32 && tile_m != 64))
    return cudaErrorInvalidValue;
  quantize<<<(m * (k / 32) + 7) / 8, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<int8_t*>(a), sa, m * (k / 32));
  if (tile_m == 16)
    iq2xxs_tiled<16,32,true,true><<<dim3((n + 31) / 32, (m + 15) / 16), 32, 0, stream>>>(
        static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa,
        static_cast<__nv_bfloat16*>(y), m, n, k);
  else if (tile_m == 32)
    iq2xxs_tiled<32,64,true,true><<<dim3((n + 63) / 64, (m + 31) / 32), 128, 0, stream>>>(
        static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa,
        static_cast<__nv_bfloat16*>(y), m, n, k);
  else
    iq2xxs_tiled<64,64,true,true><<<dim3((n + 63) / 64, (m + 63) / 64), 256, 0, stream>>>(
        static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa,
        static_cast<__nv_bfloat16*>(y), m, n, k);
  return cudaGetLastError();
}
