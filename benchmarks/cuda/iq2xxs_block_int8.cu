// SPDX-License-Identifier: Apache-2.0
// Experimental IQ2_XXS/IQ3_XXS compressed weights -> shared INT8 -> INT8 MMA.
// Exact original weights; default per32 or experimental folded per256 INT8.
// Only activations are quantized; folded weights use two signed INT8 components.
// Reuse the block32 activation quantizer and fragment mapping control.
#include "q8_block_int8.cu"
#include "../../csrc/gguf_iq_tables.cuh"
#include <type_traits>
// Column-major shared descriptors win the paired GB10 routed benchmark.
// SM86 counters do not confirm fewer shared-load bank conflicts.
// Compile with -DIQ2XXS_TRANSPOSE_DESC=0 to reproduce the row-major control.
#ifndef IQ2XXS_TRANSPOSE_DESC
#define IQ2XXS_TRANSPOSE_DESC 1
#endif
// Benchmark wider output tiling without changing the C ABI or arithmetic.
#ifndef IQ2XXS_REGISTER_N
#define IQ2XXS_REGISTER_N 64
#endif
// Group adjacent routed row tasks at each output tile to test L2 weight reuse.
// Default preserves the original column-fast block order.
#ifndef IQ2XXS_TASK_GROUP
#define IQ2XXS_TASK_GROUP 1
#endif
static_assert(IQ2XXS_TASK_GROUP >= 1 && IQ2XXS_TASK_GROUP <= 16);
// Independently tune output width for shared-expanded per32 IQ decoding.
#ifndef IQ2XXS_SHARED_N
#define IQ2XXS_SHARED_N 64
#endif
static_assert(IQ2XXS_SHARED_N == 32 || IQ2XXS_SHARED_N == 64 || IQ2XXS_SHARED_N == 128);
#ifndef IQ2XXS_REGISTER_N8
#define IQ2XXS_REGISTER_N8 IQ2XXS_REGISTER_N
#endif
static_assert(IQ2XXS_REGISTER_N8 == 64 || IQ2XXS_REGISTER_N8 == 128);
// Increase staging depth independently of the per32 quantization/MMA groups.
// K256 loses paired complete-MoE timings; keep K128 as the default.
#ifndef IQ2XXS_REGISTER_K
#define IQ2XXS_REGISTER_K 128
#endif
static_assert(IQ2XXS_REGISTER_K == 128 || IQ2XXS_REGISTER_K == 256);
// Rejected benchmark variant: lower resource use, slower complete MoE on both
// SM86 and SM121. Retain the shared per32 scale tile by default.
#ifndef IQ2XXS_SHUFFLE_SCALE
#define IQ2XXS_SHUFFLE_SCALE 0
#endif
#ifndef IQ2XXS_PIPELINE
#define IQ2XXS_PIPELINE 0
#endif
#ifndef IQ2XXS_PIPELINE_STAGES
#define IQ2XXS_PIPELINE_STAGES 2
#endif
#ifndef IQ2XXS_COMPACT_LUT
#define IQ2XXS_COMPACT_LUT 0
#endif
#ifndef IQ2XXS_SHARED_COMPACT_LUT
#define IQ2XXS_SHARED_COMPACT_LUT 0
#endif
// Experimental padding-free layout; paired full-MoE trials regress on GB10.
#ifndef IQ2XXS_SHARED_SWIZZLE
#define IQ2XXS_SHARED_SWIZZLE 0
#endif
// IQ3 lifetime experiments: paired whole-MoE trials do not justify defaults.
#ifndef IQ3XXS_STREAM_STORE
#define IQ3XXS_STREAM_STORE 0
#endif
#ifndef IQ3XXS_DECODE_UNROLL
#define IQ3XXS_DECODE_UNROLL 4
#endif
static_assert(IQ3XXS_DECODE_UNROLL == 1 || IQ3XXS_DECODE_UNROLL == 4);
#ifndef IQ3XXS_COOPERATIVE_DECODE
#define IQ3XXS_COOPERATIVE_DECODE 0
#endif
static_assert(!IQ3XXS_COOPERATIVE_DECODE || IQ3XXS_STREAM_STORE);
#ifndef IQ2XXS_SHARED_MIN_BLOCKS
#define IQ2XXS_SHARED_MIN_BLOCKS 1
#endif
static_assert(IQ2XXS_SHARED_MIN_BLOCKS >= 1 && IQ2XXS_SHARED_MIN_BLOCKS <= 8);
#ifndef IQ3XXS_MIN_BLOCKS
#define IQ3XXS_MIN_BLOCKS 1
#endif
static_assert(IQ3XXS_MIN_BLOCKS >= 1 && IQ3XXS_MIN_BLOCKS <= 6);
#ifndef IQ2XXS_SHARED_PREFETCH_L2
#define IQ2XXS_SHARED_PREFETCH_L2 0
#endif
static_assert(IQ2XXS_SHARED_PREFETCH_L2 >= 0 && IQ2XXS_SHARED_PREFETCH_L2 <= 8);
#ifndef IQ2XXS_PREFETCH_L2
#define IQ2XXS_PREFETCH_L2 0
#endif
static_assert(IQ2XXS_PREFETCH_L2 >= 0 && IQ2XXS_PREFETCH_L2 <= 8);
static_assert(IQ2XXS_PIPELINE_STAGES == 2 || IQ2XXS_PIPELINE_STAGES == 3);
static_assert(IQ2XXS_REGISTER_N == 64 || IQ2XXS_REGISTER_N == 128);
namespace {
// Match gguf_swiglu's clipping, expf/division and BF16 product rounding before
// the existing block32 quantizer. Each warp owns exactly one quantization block.
template<bool Masked=false>
__global__ void swiglu_quantize32(const __nv_bfloat16* x, int8_t* q,
                                float* scales, int blocks, int width, float limit, const int32_t* tokens=nullptr) {
  const int lane = threadIdx.x & 31;
  const int block = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  if (block >= blocks) return;
  const int64_t index = int64_t(block) * 32 + lane;
  if constexpr (Masked) {
    // Routing initializes the entire capacity to -1 on each replay.
    // Padded projected rows are mathematically zero; unused rows are not read.
    if (tokens[index / width] < 0) {
      q[index] = 0;
      if (!lane) scales[block] = 1.e-20f;
      return;
    }
  }
  const int64_t source = (index / width) * (2 * int64_t(width)) + index % width;
  const float gate = fminf(__bfloat162float(x[source]), limit);
  const float up = fminf(fmaxf(__bfloat162float(x[source + width]), -limit), limit);
  const float v = __bfloat162float(__float2bfloat16_rn((gate / (1.f + expf(-gate))) * up));
  float maximum = fabsf(v);
  for (int delta = 16; delta; delta /= 2)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, delta));
  const float scale = fmaxf(maximum / 127.f, 1.e-20f);
  q[index] = static_cast<int8_t>(__float2int_rn(v / scale));
  if (!lane) scales[block] = scale;
}
// Eight values per lane; repeat scales to retain existing benchmark scratch ABI.
__global__ void quantize256(const __nv_bfloat16* x, int8_t* q, float* scales, int blocks) {
  int lane = threadIdx.x & 31;
  int block = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  if (block >= blocks) return;
  float v[8], a = 0;
  #pragma unroll
  for (int j = 0; j < 8; ++j) {
    v[j] = __bfloat162float(x[int64_t(block) * 256 + j * 32 + lane]);
    a = fmaxf(a, fabsf(v[j]));
  }
  for (int delta = 16; delta; delta /= 2)
    a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, delta));
  float scale = fmaxf(a / 127.f, 1.e-20f);
  #pragma unroll
  for (int j = 0; j < 8; ++j)
    q[int64_t(block) * 256 + j * 32 + lane] = static_cast<int8_t>(__float2int_rn(v[j] / scale));
  if (lane < 8) scales[int64_t(block) * 8 + lane] = scale;
}
__device__ __forceinline__ uint32_t signed_bytes(uint32_t v, uint32_t signs) {
  uint32_t mask = (signs & 1) * 0xffu | ((signs >> 1) & 1) * 0xff00u |
                  ((signs >> 2) & 1) * 0xff0000u | ((signs >> 3) & 1) * 0xff000000u;
  return __vsub4(v ^ mask, mask);
}
template<int Bytes>
__device__ __forceinline__ void iq_async_copy(void* dst, const void* src, bool valid) {
  const uint32_t shared = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;" ::
      "r"(shared), "l"(src), "n"(Bytes), "r"(valid ? Bytes : 0) : "memory");
}
template<int BM, int BN=64, bool Packed=false, bool SharedLUT=false, bool Routed=false, bool HalfLoads=false, bool Slab=false, bool Weighted=false, bool Fold256=false, bool RegisterDecode=false, bool Acc256=false, bool IQ3=false, int IQ3MinBlocks=IQ3XXS_MIN_BLOCKS>
__global__ __launch_bounds__(((BM + 15) / 16) * (BN / 32) * 32,
    (Routed && Slab && !Fold256 && !RegisterDecode && BM == 32 && BN == 64) ? (IQ3 ? IQ3MinBlocks : IQ2XXS_SHARED_MIN_BLOCKS) : 1)
void iq2xxs_tiled(const int8_t* a, const uint8_t* w, const float* sa, __nv_bfloat16* y, int m, int n, int k,
    const int32_t* tokens=nullptr, const int32_t* tasks=nullptr, int xrows=0, int experts=1, const __half* wb=nullptr, const float* route_weights=nullptr) {
  static_assert(!IQ3 || (Slab && SharedLUT && Routed && !RegisterDecode && !Fold256));
  constexpr int DecodeUnroll = IQ3 ? IQ3XXS_DECODE_UNROLL : 4;
  constexpr bool Cooperative = IQ3 && IQ3XXS_COOPERATIVE_DECODE;
  static_assert(!RegisterDecode || (Slab && SharedLUT && !Fold256));
  static_assert(BM != 8 || RegisterDecode);
  static_assert(!Acc256 || (RegisterDecode && !Weighted));
  constexpr bool Shuffle = RegisterDecode && IQ2XXS_SHUFFLE_SCALE && !Acc256;
  constexpr int Lookahead = RegisterDecode ? IQ2XXS_PREFETCH_L2 :
      (Slab && !Fold256 ? IQ2XXS_SHARED_PREFETCH_L2 : 0);
  constexpr int Outputs = BM == 8 ? 2 : 4;
  constexpr int Rows = BM + BN, Threads = ((BM + 15) / 16) * (BN / 32) * 32;
  constexpr int BK = RegisterDecode ? IQ2XXS_REGISTER_K : 128;
  constexpr int Blocks = BK / 32;
  constexpr bool Pipeline = RegisterDecode && IQ2XXS_PIPELINE && !Acc256;
  constexpr int Stages = Pipeline ? IQ2XXS_PIPELINE_STAGES : 1;
  static_assert(!Pipeline || (BK == 128 && !IQ2XXS_SHUFFLE_SCALE));
  constexpr bool Swizzle = Slab && !Fold256 && !RegisterDecode && IQ2XXS_SHARED_SWIZZLE;
  // XOR 16-byte sectors by row: preserve vector alignment without row padding.
  auto tile_column = [](int row, int col) {
    return Swizzle ? col ^ ((row & 7) * 16) : col;
  };
  __shared__ int8_t tile_storage[Stages][RegisterDecode ? BM : Rows][BK + (Swizzle ? 0 : 16)];
  __shared__ uint2 descriptor_storage[Stages][RegisterDecode ? BN * Blocks : 1];
  __shared__ int8_t high_tile[Fold256 ? BN : 1][144];
  __shared__ float scale_storage[Stages][Shuffle ? BM : Rows][Blocks];
  __shared__ int subscales[Acc256 ? BN : 1][Blocks];
  __shared__ float pipeline_bases[Stages][Pipeline ? BN : 1];
  __shared__ float weight_bases[Shuffle ? BN : 1];
  const int lane = threadIdx.x & 31, group = lane >> 2, tid = lane & 3;
  const int warp = threadIdx.x >> 5;
  int task = blockIdx.y, column = blockIdx.x;
  if constexpr (Routed && IQ2XXS_TASK_GROUP > 1) {
    const int linear = blockIdx.y * gridDim.x + blockIdx.x;
    const int group = linear / (IQ2XXS_TASK_GROUP * gridDim.x);
    const int first = group * IQ2XXS_TASK_GROUP;
    const int size = min(IQ2XXS_TASK_GROUP, int(gridDim.y) - first);
    const int offset = linear - first * gridDim.x;
    task = first + offset % size;
    column = offset / size;
  }
  int rb = task * BM;
  if constexpr (Routed) {
    const int expert = tasks[task * 2];
    rb = tasks[task * 2 + 1];
    if (expert < 0 || expert >= experts || rb < 0 || rb >= m) return;
    w += int64_t(expert) * n * (k / 256) * (Slab ? (IQ3 ? 96 : 64) : 66);
    if constexpr (Slab) wb += int64_t(expert) * n * (k / 256);
  }
  const int cb = column * BN;
  const int wr = (warp / (BN / 32)) * 16, wc = (warp % (BN / 32)) * 32;
  constexpr bool CompactLUT = !IQ3 && ((RegisterDecode && IQ2XXS_COMPACT_LUT) ||
      (!RegisterDecode && Slab && !Fold256 && IQ2XXS_SHARED_COMPACT_LUT));
  using LUTWord = typename std::conditional<CompactLUT || IQ3, uint32_t, uint64_t>::type;
  __shared__ LUTWord lut[SharedLUT ? 256 : 1];
  if constexpr (SharedLUT) {
    for (int i = threadIdx.x; i < 256; i += Threads) {
      uint64_t grid = IQ3 ? uint64_t(iq3xxs_grid[i]) : iq2xxs_grid[i];
      if constexpr (CompactLUT) {
        uint32_t selectors = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
          uint32_t value = (grid >> (j * 8)) & 255;
          selectors |= (value == 8 ? 0u : value == 25 ? 1u : 2u) << (j * 4);
        }
        lut[i] = selectors;
      } else lut[i] = grid;
    }
    __syncthreads();
  }
  float sum[4][4] = {};
  int low_acc[4][4] = {}, high_acc[4][4] = {};
  auto prefetch = [&](int base, int stage) {
    if constexpr (Pipeline) {
      if (base >= k) {
        // Empty groups preserve the same wait depth while draining the tail.
        asm volatile("cp.async.commit_group;" ::: "memory");
        return;
      }
      for (int index = threadIdx.x; index < BM * 8; index += Threads) {
        int row = index / 8, col = (index % 8) * 16;
        int input = rb + row;
        if constexpr (Routed) input = input < m ? tokens[input] : -1;
        bool valid = input >= 0 && input < (Routed ? xrows : m);
        iq_async_copy<16>(&tile_storage[stage][row][col],
            a + (valid ? int64_t(input) * k + base + col : 0), valid);
      }
      for (int index = threadIdx.x; index < BM * 4; index += Threads) {
        int row = index / 4, block = index % 4;
        int input = rb + row;
        if constexpr (Routed) input = input < m ? tokens[input] : -1;
        bool valid = input >= 0 && input < (Routed ? xrows : m);
        iq_async_copy<4>(&scale_storage[stage][row][block],
            sa + (valid ? int64_t(input) * (k / 32) + base / 32 + block : 0), valid);
      }
      for (int index = threadIdx.x; index < BN * 4; index += Threads) {
        int row = index / 4, block = index % 4;
        bool valid = cb + row < n;
        iq_async_copy<8>(&descriptor_storage[stage][IQ2XXS_TRANSPOSE_DESC ? block * BN + row : index],
            reinterpret_cast<const uint2*>(w) +
                (valid ? (int64_t(base / 128) * n + cb + row) * 4 + block : 0), valid);
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
      for (int row = threadIdx.x; row < BN; row += Threads)
        pipeline_bases[stage][row] = cb + row < n ?
            __half2float(wb[int64_t(base / 256) * n + cb + row]) : 0;
    }
  };
  if constexpr (Pipeline)
    for (int stage = 0; stage < Stages - 1; ++stage) prefetch(stage * BK, stage);
  for (int base = 0; base < k; base += BK) {
    if constexpr (Slab && Lookahead > 0 && !Pipeline) {
      // Cache-only hint: retain all normal loads and the original arithmetic.
      const int future = base + Lookahead * BK;
      if (future < k) {
        for (int row = threadIdx.x; row < BN; row += Threads) {
          if (cb + row < n) {
            const uint8_t* next = w +
                (int64_t(future / 128) * n + cb + row) * 4 * (IQ3 ? 12 : 8);
            asm volatile("prefetch.global.L2 [%0];" :: "l"(next));
          }
        }
      }
    }
    const int stage = Pipeline ? (base / BK) % Stages : 0;
    auto& tile = tile_storage[stage];
    auto& descriptors = descriptor_storage[stage];
    auto& scales = scale_storage[stage];
    if constexpr (Pipeline) {
      asm volatile("cp.async.wait_group %0;" :: "n"(Stages - 2) : "memory");
      __syncthreads();
      prefetch(base + (Stages - 1) * BK, (stage + Stages - 1) % Stages);
      for (int index = threadIdx.x; index < BN * Blocks; index += Threads) {
        int row = index / Blocks, block = index % Blocks;
        uint32_t metadata = descriptors[IQ2XXS_TRANSPOSE_DESC ? block * BN + row : index].y;
        scales[BM + row][block] = pipeline_bases[stage][row] *
            (0.5f + float(metadata >> 28)) * 0.25f;
      }
    } else {
    for (int index = threadIdx.x; index < BM * (BK / 16); index += Threads) {
      int row = index / (BK / 16), col = (index % (BK / 16)) * 16;
      int4 value = {};
      int input = rb + row;
      bool valid = input < m;
      if constexpr (Routed) {
        input = valid ? tokens[input] : -1;
        valid = input >= 0 && input < xrows;
      }
      if (valid)
        value = *reinterpret_cast<const int4*>(a + int64_t(input) * k + base + col);
      *reinterpret_cast<int4*>(&tile[row][tile_column(row, col)]) = value;
    }
    for (int index = threadIdx.x; index < BM * Blocks; index += Threads) {
      int row = index / Blocks, block = index % Blocks;
      int input = rb + row;
      bool valid = input < m;
      if constexpr (Routed) {
        input = valid ? tokens[input] : -1;
        valid = input >= 0 && input < xrows;
      }
      scales[row][block] = valid ? sa[int64_t(input) * (k / 32) + base / 32 + block] : 0;
    }
    for (int index = threadIdx.x; index < BN * Blocks * (Cooperative ? 4 : 1); index += Threads) {
      const int descriptor_index_in_tile = index / (Cooperative ? 4 : 1);
      int row = descriptor_index_in_tile / Blocks, block = descriptor_index_in_tile % Blocks;
      const int first_part = Cooperative ? index % 4 : 0;
      const int end_part = Cooperative ? first_part + 1 : 4;
      uint64_t values[4] = {}, high_values[4] = {};
      float scale = 0;
      if (cb + row < n) {
        const uint8_t* sub = nullptr;
        uint32_t metadata, indices = 0, indices_high = 0;
        float base_scale;
        if constexpr (Slab) {
          uint2 descriptor;
          const int64_t descriptor_index =
              (int64_t(base / 128 + block / 4) * n + cb + row) * 4 + block % 4;
          if constexpr (IQ3) {
            const uint32_t* words = reinterpret_cast<const uint32_t*>(w) + descriptor_index * 3;
            descriptor = make_uint2(words[0], words[2]);
            indices_high = words[1];
          } else descriptor = reinterpret_cast<const uint2*>(w)[descriptor_index];
          indices = descriptor.x;
          metadata = descriptor.y;
          if constexpr (RegisterDecode) descriptors[IQ2XXS_TRANSPOSE_DESC ? block * BN + row : row * Blocks + block] = descriptor;
          base_scale = __half2float(wb[int64_t(base / 256) * n + cb + row]);
          if constexpr (Shuffle)
            if (block == 0) weight_bases[row] = base_scale;
        } else {
          const uint8_t* raw = w + (int64_t(cb + row) * (k / 256) + base / 256) * 66;
          sub = raw + 2 + ((base % 256) / 32 + block) * 8;
          if constexpr (HalfLoads) {
            const uint16_t* words = reinterpret_cast<const uint16_t*>(sub);
            indices = uint32_t(words[0]) | (uint32_t(words[1]) << 16);
            metadata = uint32_t(words[2]) | (uint32_t(words[3]) << 16);
          } else {
            metadata = uint32_t(sub[4]) | (uint32_t(sub[5]) << 8) |
                       (uint32_t(sub[6]) << 16) | (uint32_t(sub[7]) << 24);
          }
          base_scale = __half2float(*reinterpret_cast<const __half*>(raw));
        }
        if constexpr (Acc256) subscales[row][block] = 2 * int(metadata >> 28) + 1;
        scale = (Fold256 || Acc256) ? base_scale * 0.125f : base_scale * (0.5f + float(metadata >> 28)) * (IQ3 ? 0.5f : 0.25f);
        #pragma unroll (DecodeUnroll)
        for (int j = first_part; j < end_part; ++j) {
          uint32_t signs = (metadata >> (j * 7)) & 127;
          signs |= (__popc(signs) & 1) << 7;
          uint32_t code = (HalfLoads || Slab) ? (indices >> (j * 8)) & 255 : sub[j];
          uint64_t grid = SharedLUT ? lut[code] : iq2xxs_grid[code];
          if constexpr (IQ3) {
            const uint32_t codes = j < 2 ? indices : indices_high;
            const int shift = (j & 1) * 16;
            grid = uint64_t(lut[(codes >> shift) & 255]) |
                (uint64_t(lut[(codes >> (shift + 8)) & 255]) << 32);
          }
          if constexpr (CompactLUT && !RegisterDecode) {
            const uint32_t selectors = lut[code];
            grid = uint64_t(__byte_perm(0x002b1908u, 0, selectors)) |
                (uint64_t(__byte_perm(0x002b1908u, 0, selectors >> 16)) << 32);
          }
          if constexpr (RegisterDecode) {
            // Decode only the consuming lane's four bytes in the MMA loop.
          } else if constexpr (Fold256) {
            // Fold the odd IQ scale exactly: signed v = lo + 64*hi.
            // Max magnitude 43*31=1333; both components fit signed INT8.
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
              int v = int((grid >> (t * 8)) & 255) * (2 * int(metadata >> 28) + 1);
              int lo = v & 63, hi = v >> 6;
              if (signs & (1 << t)) { lo = -lo; hi = -hi; }
              values[j] |= uint64_t(uint8_t(lo)) << (t * 8);
              high_values[j] |= uint64_t(uint8_t(hi)) << (t * 8);
            }
          } else if constexpr (Packed) {
            const uint64_t decoded = uint64_t(signed_bytes(uint32_t(grid), signs)) |
                        (uint64_t(signed_bytes(uint32_t(grid >> 32), signs >> 4)) << 32);
            if constexpr (IQ3 && IQ3XXS_STREAM_STORE)
              *reinterpret_cast<uint64_t*>(&tile[BM + row][tile_column(BM + row, block * 32 + j * 8)]) = decoded;
            else values[j] = decoded;
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
      if constexpr (Acc256)
        if (cb + row >= n) subscales[row][block] = 0;
      if constexpr (RegisterDecode) {
        if (cb + row >= n) descriptors[IQ2XXS_TRANSPOSE_DESC ? block * BN + row : row * Blocks + block] = make_uint2(0, 0);
        if constexpr (Shuffle)
          if (cb + row >= n && block == 0) weight_bases[row] = 0;
      }
      if constexpr (!Shuffle)
        if (!Cooperative || first_part == 0) scales[BM + row][block] = scale;
      if constexpr (!RegisterDecode) {
      #pragma unroll
      for (int j = first_part; j < end_part; ++j) {
        if constexpr (IQ3 && IQ3XXS_STREAM_STORE) {
          if (cb + row >= n)
            *reinterpret_cast<uint64_t*>(&tile[BM + row][tile_column(BM + row, block * 32 + j * 8)]) = 0;
        } else *reinterpret_cast<uint64_t*>(&tile[BM + row][tile_column(BM + row, block * 32 + j * 8)]) = values[j];
        if constexpr (Fold256)
          *reinterpret_cast<uint64_t*>(&high_tile[row][block * 32 + j * 8]) = high_values[j];
      }
      }
    }
    }
    __syncthreads();
    #pragma unroll
    for (int block = 0; block < Blocks; ++block) {
      uint32_t av[4];
      #pragma unroll
      for (int r = 0; r < 4; ++r)
        av[r] = BM == 8 && (r & 1) ? 0 :
            *reinterpret_cast<const uint32_t*>(&tile[wr + group + (r & 1) * 8][tile_column(wr + group + (r & 1) * 8, block * 32 + tid * 4 + (r >> 1) * 16)]);
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        uint32_t bv[2];
        float weight_scale = 0;
        if constexpr (RegisterDecode) {
          const uint2 descriptor = descriptors[IQ2XXS_TRANSPOSE_DESC ? block * BN + wc + j * 8 + group : (wc + j * 8 + group) * Blocks + block];
          if constexpr (Shuffle)
            weight_scale = weight_bases[wc + j * 8 + group] *
                           (0.5f + float(descriptor.y >> 28)) * 0.25f;
          #pragma unroll
          for (int r = 0; r < 2; ++r) {
            int part = tid / 2 + r * 2;
            uint32_t code = (descriptor.x >> (part * 8)) & 255;
            uint32_t signs = (descriptor.y >> (part * 7)) & 127;
            signs |= (__popc(signs) & 1) << 7;
            uint32_t values;
            if constexpr (CompactLUT) {
              uint32_t selectors = lut[code] >> ((tid & 1) * 16);
              values = __byte_perm(0x002b1908u, 0, selectors);
            } else {
              uint64_t grid = lut[code];
              values = uint32_t(grid >> ((tid & 1) * 32));
            }
            bv[r] = signed_bytes(values, signs >> ((tid & 1) * 4));
          }
        } else {
          #pragma unroll
          for (int r = 0; r < 2; ++r)
            bv[r] = *reinterpret_cast<const uint32_t*>(&tile[BM + wc + j * 8 + group][tile_column(BM + wc + j * 8 + group, block * 32 + tid * 4 + r * 16)]);
        }
        int acc[4] = {};
        if constexpr (Fold256) {
          #pragma unroll
          for (int r = 0; r < 4; ++r) acc[r] = low_acc[j][r];
        }
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3])
            : "r"(av[0]), "r"(av[1]), "r"(av[2]), "r"(av[3]), "r"(bv[0]), "r"(bv[1]));
        if constexpr (Fold256) {
          #pragma unroll
          for (int r = 0; r < 4; ++r) low_acc[j][r] = acc[r];
          #pragma unroll
          for (int r = 0; r < 2; ++r)
            bv[r] = *reinterpret_cast<const uint32_t*>(&high_tile[wc + j * 8 + group][block * 32 + tid * 4 + r * 16]);
          asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
              : "+r"(high_acc[j][0]), "+r"(high_acc[j][1]), "+r"(high_acc[j][2]), "+r"(high_acc[j][3])
              : "r"(av[0]), "r"(av[1]), "r"(av[2]), "r"(av[3]), "r"(bv[0]), "r"(bv[1]));
          if (base % 256 == 128 && block == 3) {
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
              // Bound 127*1333*256 < 2^31; multiply avoids shifting negatives.
              sum[j][r] += float(low_acc[j][r] + 64 * high_acc[j][r]) *
                  (scales[wr + group + (r >> 1) * 8][block] *
                   scales[BM + wc + j * 8 + tid * 2 + (r & 1)][block]);
              low_acc[j][r] = high_acc[j][r] = 0;
            }
          }
        } else if constexpr (Acc256) {
          #pragma unroll
          for (int r = 0; r < Outputs; ++r) {
            // Bound 127*43*31*256=43338496 fits int32. One MMA per K32.
            low_acc[j][r] += acc[r] * subscales[wc + j * 8 + tid * 2 + (r & 1)][block];
            if ((base / 32 + block) % 8 == 7) {
              sum[j][r] += float(low_acc[j][r]) *
                  (scales[wr + group + (r >> 1) * 8][block] *
                   scales[BM + wc + j * 8 + tid * 2 + (r & 1)][block]);
              low_acc[j][r] = 0;
            }
          }
        } else if constexpr (Shuffle) {
          // Four lanes decode one B row. Output columns select two of the
          // eight row groups; preserve the original per32 FP32 operation order.
          float output_scales[2];
          #pragma unroll
          for (int r = 0; r < 2; ++r)
            output_scales[r] = __shfl_sync(0xffffffff, weight_scale, (tid * 2 + r) * 4);
          #pragma unroll
          for (int r = 0; r < Outputs; ++r)
            sum[j][r] += float(acc[r]) *
                (scales[wr + group + (r >> 1) * 8][block] * output_scales[r & 1]);
        } else {
          #pragma unroll
          for (int r = 0; r < Outputs; ++r)
            sum[j][r] += float(acc[r]) * (scales[wr + group + (r >> 1) * 8][block] *
                                        scales[BM + wc + j * 8 + tid * 2 + (r & 1)][block]);
        }
      }
    }
    // Pipeline readers rendezvous at the next iteration's post-wait barrier
    // before any thread can prefetch into this buffer again.
    if constexpr (!Pipeline) __syncthreads();
  }
  if constexpr (Pipeline) asm volatile("cp.async.wait_group 0;" ::: "memory");
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    #pragma unroll
    for (int r = 0; r < Outputs; ++r) {
      int row = rb + wr + group + (r >> 1) * 8;
      int col = cb + wc + j * 8 + tid * 2 + (r & 1);
      if (row < m && col < n) {
        float value = sum[j][r];
        if constexpr (Weighted) value *= route_weights[row];
        y[row * n + col] = __float2bfloat16(value);
      }
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

// Benchmark-only routed projection: caller validates task spans and buffers.
// Quantize each input once; routes gather codes/scales inside the MMA kernel.
extern "C" int iq2xxs_routed_int8(const void* x, void* a, float* sa,
    const void* raw, const int32_t* tokens, const int32_t* tasks, void* y,
    int xrows, int rows, int n, int k, int experts, int taskcount,
    int tile_m, cudaStream_t stream) {
  if (xrows <= 0 || rows <= 0 || n <= 0 || k <= 0 || k % 256 || experts <= 0 ||
      taskcount <= 0 || (tile_m != 16 && tile_m != 32 && tile_m != 64))
    return cudaErrorInvalidValue;
  quantize<<<(xrows * (k / 32) + 7) / 8, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<int8_t*>(a), sa, xrows * (k / 32));
#define ROUTED_LAUNCH(M,N) \
  iq2xxs_tiled<M,N,true,true,true><<<dim3((n+N-1)/N,taskcount), (M/16)*(N/32)*32, 0, stream>>>( \
      static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa, \
      static_cast<__nv_bfloat16*>(y), rows, n, k, tokens, tasks, xrows, experts)
  if (tile_m == 16) { ROUTED_LAUNCH(16,64); }
  else if (tile_m == 32) { ROUTED_LAUNCH(32,64); }
  else { ROUTED_LAUNCH(64,64); }
#undef ROUTED_LAUNCH
  return cudaGetLastError();
}

extern "C" int iq2xxs_routed_int8_half(const void* x, void* a, float* sa,
    const void* raw, const int32_t* tokens, const int32_t* tasks, void* y,
    int xrows, int rows, int n, int k, int experts, int taskcount,
    int tile_m, cudaStream_t stream) {
  if (xrows <= 0 || rows <= 0 || n <= 0 || k <= 0 || k % 256 || experts <= 0 ||
      taskcount <= 0 || (tile_m != 16 && tile_m != 32 && tile_m != 64))
    return cudaErrorInvalidValue;
  quantize<<<(xrows * (k / 32) + 7) / 8, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<int8_t*>(a), sa, xrows * (k / 32));
#define ROUTED_LAUNCH(M,N) \
  iq2xxs_tiled<M,N,true,true,true,true><<<dim3((n+N-1)/N,taskcount), (M/16)*(N/32)*32, 0, stream>>>( \
      static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa, \
      static_cast<__nv_bfloat16*>(y), rows, n, k, tokens, tasks, xrows, experts)
  if (tile_m == 16) { ROUTED_LAUNCH(16,64); }
  else if (tile_m == 32) { ROUTED_LAUNCH(32,64); }
  else { ROUTED_LAUNCH(64,64); }
#undef ROUTED_LAUNCH
  return cudaGetLastError();
}

// Lossless descriptors[E,K/128,N,4,8] and bases[E,K/256,N].
extern "C" int iq2xxs_routed_int8_slab(const void* x, void* a, float* sa,
    const void* raw, const void* bases, const int32_t* tokens, const int32_t* tasks, void* y,
    int xrows, int rows, int n, int k, int experts, int taskcount,
    int tile_m, cudaStream_t stream) {
  if (xrows <= 0 || rows <= 0 || n <= 0 || k <= 0 || k % 256 || experts <= 0 ||
      taskcount <= 0 || (tile_m != 16 && tile_m != 32 && tile_m != 64))
    return cudaErrorInvalidValue;
  quantize<<<(xrows * (k / 32) + 7) / 8, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<int8_t*>(a), sa, xrows * (k / 32));
#define ROUTED_LAUNCH(M,N) \
  iq2xxs_tiled<M,N,true,true,true,false,true><<<dim3((n+N-1)/N,taskcount), (M/16)*(N/32)*32, 0, stream>>>( \
      static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa, \
      static_cast<__nv_bfloat16*>(y), rows, n, k, tokens, tasks, xrows, experts, static_cast<const __half*>(bases))
  if (tile_m == 16) { ROUTED_LAUNCH(16,IQ2XXS_SHARED_N); }
  else if (tile_m == 32) { ROUTED_LAUNCH(32,IQ2XXS_SHARED_N); }
  else { ROUTED_LAUNCH(64,IQ2XXS_SHARED_N); }
#undef ROUTED_LAUNCH
  return cudaGetLastError();
}

extern "C" int iq2xxs_routed_int8_slab_weighted(const void* x, void* a, float* sa,
    const void* raw, const void* bases, const int32_t* tokens, const int32_t* tasks, const float* weights, void* y,
    int xrows, int rows, int n, int k, int experts, int taskcount,
    int tile_m, cudaStream_t stream) {
  if (xrows <= 0 || rows <= 0 || n <= 0 || k <= 0 || k % 256 || experts <= 0 ||
      taskcount <= 0 || (tile_m != 16 && tile_m != 32 && tile_m != 64))
    return cudaErrorInvalidValue;
  quantize<<<(xrows * (k / 32) + 7) / 8, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<int8_t*>(a), sa, xrows * (k / 32));
#define ROUTED_LAUNCH(M,N) \
  iq2xxs_tiled<M,N,true,true,true,false,true,true><<<dim3((n+N-1)/N,taskcount), (M/16)*(N/32)*32, 0, stream>>>( \
      static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa, \
      static_cast<__nv_bfloat16*>(y), rows, n, k, tokens, tasks, xrows, experts, static_cast<const __half*>(bases), weights)
  if (tile_m == 16) { ROUTED_LAUNCH(16,IQ2XXS_SHARED_N); }
  else if (tile_m == 32) { ROUTED_LAUNCH(32,IQ2XXS_SHARED_N); }
  else { ROUTED_LAUNCH(64,IQ2XXS_SHARED_N); }
#undef ROUTED_LAUNCH
  return cudaGetLastError();
}

// Experimental per256 activations and exact two-component folded IQ weights.
extern "C" int iq2xxs_routed_int8_slab_fold256(const void* x, void* a, float* sa,
    const void* raw, const void* bases, const int32_t* tokens, const int32_t* tasks, void* y,
    int xrows, int rows, int n, int k, int experts, int taskcount,
    int tile_m, cudaStream_t stream) {
  if (xrows <= 0 || rows <= 0 || n <= 0 || k <= 0 || k % 256 || experts <= 0 ||
      taskcount <= 0 || (tile_m != 16 && tile_m != 32 && tile_m != 64))
    return cudaErrorInvalidValue;
  quantize256<<<(xrows * (k / 256) + 7) / 8, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<int8_t*>(a), sa, xrows * (k / 256));
#define ROUTED_LAUNCH(M,N) \
  iq2xxs_tiled<M,N,true,true,true,false,true,false,true><<<dim3((n+N-1)/N,taskcount), (M/16)*(N/32)*32, 0, stream>>>( \
      static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa, \
      static_cast<__nv_bfloat16*>(y), rows, n, k, tokens, tasks, xrows, experts, static_cast<const __half*>(bases))
  if (tile_m == 16) { ROUTED_LAUNCH(16,64); }
  else if (tile_m == 32) { ROUTED_LAUNCH(32,64); }
  else { ROUTED_LAUNCH(64,64); }
#undef ROUTED_LAUNCH
  return cudaGetLastError();
}

extern "C" int iq2xxs_routed_int8_slab_fold256_weighted(const void* x, void* a, float* sa,
    const void* raw, const void* bases, const int32_t* tokens, const int32_t* tasks, const float* weights, void* y,
    int xrows, int rows, int n, int k, int experts, int taskcount,
    int tile_m, cudaStream_t stream) {
  if (xrows <= 0 || rows <= 0 || n <= 0 || k <= 0 || k % 256 || experts <= 0 ||
      taskcount <= 0 || (tile_m != 16 && tile_m != 32 && tile_m != 64))
    return cudaErrorInvalidValue;
  quantize256<<<(xrows * (k / 256) + 7) / 8, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<int8_t*>(a), sa, xrows * (k / 256));
#define ROUTED_LAUNCH(M,N) \
  iq2xxs_tiled<M,N,true,true,true,false,true,true,true><<<dim3((n+N-1)/N,taskcount), (M/16)*(N/32)*32, 0, stream>>>( \
      static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa, \
      static_cast<__nv_bfloat16*>(y), rows, n, k, tokens, tasks, xrows, experts, static_cast<const __half*>(bases), weights)
  if (tile_m == 16) { ROUTED_LAUNCH(16,64); }
  else if (tile_m == 32) { ROUTED_LAUNCH(32,64); }
  else { ROUTED_LAUNCH(64,64); }
#undef ROUTED_LAUNCH
  return cudaGetLastError();
}

// Decode compressed IQ descriptors straight into signed INT8 MMA registers.
extern "C" int iq2xxs_routed_int8_slab_register(const void* x, void* a, float* sa,
    const void* raw, const void* bases, const int32_t* tokens, const int32_t* tasks, void* y,
    int xrows, int rows, int n, int k, int experts, int taskcount,
    int tile_m, cudaStream_t stream) {
  if (xrows <= 0 || rows <= 0 || n <= 0 || k <= 0 || k % 256 || experts <= 0 ||
      taskcount <= 0 || (tile_m == 0 && (taskcount % 2 || taskcount > 65534)) || (tile_m != 0 && tile_m != 8 && tile_m != 16 && tile_m != 32 && tile_m != 64))
    return cudaErrorInvalidValue;
  quantize<<<(xrows * (k / 32) + 7) / 8, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<int8_t*>(a), sa, xrows * (k / 32));
#define ROUTED_LAUNCH(M,N) \
  iq2xxs_tiled<M,N,true,true,true,false,true,false,false,true><<<dim3((n+N-1)/N,taskcount), ((M+15)/16)*(N/32)*32, 0, stream>>>( \
      static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa, \
      static_cast<__nv_bfloat16*>(y), rows, n, k, tokens, tasks, xrows, experts, static_cast<const __half*>(bases))
  // tile_m=0: two equal disjoint task arrays, sparse8 followed by dense16.
  if (tile_m == 0) {
    taskcount /= 2;
    ROUTED_LAUNCH(8,IQ2XXS_REGISTER_N8);
    tasks += taskcount * 2;
    ROUTED_LAUNCH(16,IQ2XXS_REGISTER_N);
  }
  else if (tile_m == 8) { ROUTED_LAUNCH(8,IQ2XXS_REGISTER_N8); }
  else if (tile_m == 16) { ROUTED_LAUNCH(16,IQ2XXS_REGISTER_N); }
  else if (tile_m == 32) { ROUTED_LAUNCH(32,IQ2XXS_REGISTER_N); }
  else { ROUTED_LAUNCH(64,IQ2XXS_REGISTER_N); }
#undef ROUTED_LAUNCH
  return cudaGetLastError();
}

extern "C" int iq2xxs_routed_int8_slab_register_weighted(const void* x, void* a, float* sa,
    const void* raw, const void* bases, const int32_t* tokens, const int32_t* tasks, const float* weights, void* y,
    int xrows, int rows, int n, int k, int experts, int taskcount,
    int tile_m, cudaStream_t stream) {
  if (xrows <= 0 || rows <= 0 || n <= 0 || k <= 0 || k % 256 || experts <= 0 ||
      taskcount <= 0 || (tile_m == 0 && (taskcount % 2 || taskcount > 65534)) || (tile_m != 0 && tile_m != 8 && tile_m != 16 && tile_m != 32 && tile_m != 64))
    return cudaErrorInvalidValue;
  quantize<<<(xrows * (k / 32) + 7) / 8, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<int8_t*>(a), sa, xrows * (k / 32));
#define ROUTED_LAUNCH(M,N) \
  iq2xxs_tiled<M,N,true,true,true,false,true,true,false,true><<<dim3((n+N-1)/N,taskcount), ((M+15)/16)*(N/32)*32, 0, stream>>>( \
      static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa, \
      static_cast<__nv_bfloat16*>(y), rows, n, k, tokens, tasks, xrows, experts, static_cast<const __half*>(bases), weights)
  // tile_m=0: two equal disjoint task arrays, sparse8 followed by dense16.
  if (tile_m == 0) {
    taskcount /= 2;
    ROUTED_LAUNCH(8,IQ2XXS_REGISTER_N8);
    tasks += taskcount * 2;
    ROUTED_LAUNCH(16,IQ2XXS_REGISTER_N);
  }
  else if (tile_m == 8) { ROUTED_LAUNCH(8,IQ2XXS_REGISTER_N8); }
  else if (tile_m == 16) { ROUTED_LAUNCH(16,IQ2XXS_REGISTER_N); }
  else if (tile_m == 32) { ROUTED_LAUNCH(32,IQ2XXS_REGISTER_N); }
  else { ROUTED_LAUNCH(64,IQ2XXS_REGISTER_N); }
#undef ROUTED_LAUNCH
  return cudaGetLastError();
}

// Benchmark-only C ABI; caller owns correctly sized, non-overlapping buffers.
extern "C" int iq2xxs_swiglu_quantize32(const void* x, void* q, float* scales,
                                      int rows, int width, float limit, cudaStream_t stream) {
  if (rows < 0 || width <= 0 || width % 32 || !(limit > 0.f) ||
      // Keep the flattened uint32 CUDA thread index within its range.
      !isfinite(limit) || int64_t(rows) * (width / 32) > 134217728)
    return cudaErrorInvalidValue;
  if (!rows) return cudaSuccess;
  const int blocks = rows * (width / 32);
  swiglu_quantize32<><<<(int64_t(blocks) + 7) / 8, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<int8_t*>(q), scales, blocks, width, limit);
  return cudaGetLastError();
}

extern "C" int iq2xxs_routed_int8_slab_register_weighted_quantized(const void* x, void* a, float* sa,
    const void* raw, const void* bases, const int32_t* tokens, const int32_t* tasks, const float* weights, void* y,
    int xrows, int rows, int n, int k, int experts, int taskcount,
    int tile_m, cudaStream_t stream) {
  if (xrows <= 0 || rows <= 0 || n <= 0 || k <= 0 || k % 256 || experts <= 0 ||
      taskcount <= 0 || (tile_m == 0 && (taskcount % 2 || taskcount > 65534)) || (tile_m != 0 && tile_m != 8 && tile_m != 16 && tile_m != 32 && tile_m != 64))
    return cudaErrorInvalidValue;
  // x is unused: a/sa already contain the fused quantizer output.
#define ROUTED_LAUNCH(M,N) \
  iq2xxs_tiled<M,N,true,true,true,false,true,true,false,true><<<dim3((n+N-1)/N,taskcount), ((M+15)/16)*(N/32)*32, 0, stream>>>( \
      static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa, \
      static_cast<__nv_bfloat16*>(y), rows, n, k, tokens, tasks, xrows, experts, static_cast<const __half*>(bases), weights)
  // tile_m=0: two equal disjoint task arrays, sparse8 followed by dense16.
  if (tile_m == 0) {
    taskcount /= 2;
    ROUTED_LAUNCH(8,IQ2XXS_REGISTER_N8);
    tasks += taskcount * 2;
    ROUTED_LAUNCH(16,IQ2XXS_REGISTER_N);
  }
  else if (tile_m == 8) { ROUTED_LAUNCH(8,IQ2XXS_REGISTER_N8); }
  else if (tile_m == 16) { ROUTED_LAUNCH(16,IQ2XXS_REGISTER_N); }
  else if (tile_m == 32) { ROUTED_LAUNCH(32,IQ2XXS_REGISTER_N); }
  else { ROUTED_LAUNCH(64,IQ2XXS_REGISTER_N); }
#undef ROUTED_LAUNCH
  return cudaGetLastError();
}

extern "C" int iq2xxs_swiglu_quantize32_masked(const void* x, void* q, float* scales, const int32_t* tokens,
                                      int rows, int width, float limit, cudaStream_t stream) {
  if (rows < 0 || width <= 0 || width % 32 || !(limit > 0.f) ||
      // Keep the flattened uint32 CUDA thread index within its range.
      !isfinite(limit) || int64_t(rows) * (width / 32) > 134217728)
    return cudaErrorInvalidValue;
  if (!rows) return cudaSuccess;
  const int blocks = rows * (width / 32);
  swiglu_quantize32<true><<<(int64_t(blocks) + 7) / 8, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<int8_t*>(q), scales, blocks, width, limit, tokens);
  return cudaGetLastError();
}

namespace {
__global__ void iq_retile8(const int32_t* input, const int32_t* tokens,
                           int32_t* output, int count, int rows) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count * 2) return;
  int expert = input[(index / 2) * 2];
  int start = input[(index / 2) * 2 + 1];
  int row = start + (index % 2) * 8;
  // The native planner produces a contiguous valid-token prefix per expert.
  bool valid = expert >= 0 && start >= 0 && row < rows && tokens[row] >= 0;
  output[index * 2] = valid ? expert : -1;
  output[index * 2 + 1] = valid ? row : -1;
}
}
extern "C" int iq2xxs_retile8(const int32_t* input, const int32_t* tokens,
                             int32_t* output, int count, int rows, cudaStream_t stream) {
  if (count <= 0 || count > 32767 || rows <= 0) return cudaErrorInvalidValue;
  iq_retile8<<<(count * 2 + 255) / 256, 256, 0, stream>>>(input, tokens, output, count, rows);
  return cudaGetLastError();
}

namespace {
template<int Threshold=8>
__global__ void iq_classify_rows(const int32_t* input, const int32_t* tokens,
                                int32_t* output, int count, int rows) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  int expert = input[index * 2], row = input[index * 2 + 1];
  bool valid = expert >= 0 && row >= 0 && row < rows && tokens[row] >= 0;
  bool dense = valid && row + Threshold < rows && tokens[row + Threshold] >= 0;
  output[index * 2] = valid && !dense ? expert : -1;
  output[index * 2 + 1] = valid && !dense ? row : -1;
  output[(count + index) * 2] = dense ? expert : -1;
  output[(count + index) * 2 + 1] = dense ? row : -1;
}
}
// Input must be native16-row route tasks; output is disjoint [2*count,2].
extern "C" int iq2xxs_retile_mixed(const int32_t* input, const int32_t* tokens,
                                  int32_t* output, int count, int rows, cudaStream_t stream) {
  if (count <= 0 || count > 32767 || rows <= 0) return cudaErrorInvalidValue;
  iq_classify_rows<><<<(count + 255) / 256, 256, 0, stream>>>(input, tokens, output, count, rows);
  return cudaGetLastError();
}

// FC1-only per256 activations; scale integer MMA accumulators, not weights.
extern "C" int iq2xxs_routed_int8_slab_register_acc256(const void* x, void* a, float* sa,
    const void* raw, const void* bases, const int32_t* tokens, const int32_t* tasks, void* y,
    int xrows, int rows, int n, int k, int experts, int taskcount,
    int tile_m, cudaStream_t stream) {
  if (xrows <= 0 || rows <= 0 || n <= 0 || k <= 0 || k % 256 || experts <= 0 ||
      taskcount <= 0 || (tile_m == 0 && (taskcount % 2 || taskcount > 65534)) || (tile_m != 0 && tile_m != 8 && tile_m != 16 && tile_m != 32 && tile_m != 64))
    return cudaErrorInvalidValue;
  quantize256<<<(xrows * (k / 256) + 7) / 8, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<int8_t*>(a), sa, xrows * (k / 256));
#define ROUTED_LAUNCH(M,N) \
  iq2xxs_tiled<M,N,true,true,true,false,true,false,false,true,true><<<dim3((n+N-1)/N,taskcount), ((M+15)/16)*(N/32)*32, 0, stream>>>( \
      static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa, \
      static_cast<__nv_bfloat16*>(y), rows, n, k, tokens, tasks, xrows, experts, static_cast<const __half*>(bases))
  // tile_m=0: two equal disjoint task arrays, sparse8 followed by dense16.
  if (tile_m == 0) {
    taskcount /= 2;
    ROUTED_LAUNCH(8,IQ2XXS_REGISTER_N8);
    tasks += taskcount * 2;
    ROUTED_LAUNCH(16,IQ2XXS_REGISTER_N);
  }
  else if (tile_m == 8) { ROUTED_LAUNCH(8,IQ2XXS_REGISTER_N8); }
  else if (tile_m == 16) { ROUTED_LAUNCH(16,IQ2XXS_REGISTER_N); }
  else if (tile_m == 32) { ROUTED_LAUNCH(32,IQ2XXS_REGISTER_N); }
  else { ROUTED_LAUNCH(64,IQ2XXS_REGISTER_N); }
#undef ROUTED_LAUNCH
  return cudaGetLastError();
}

extern "C" int iq2xxs_routed_int8_slab_weighted_quantized(const void* x, void* a, float* sa,
    const void* raw, const void* bases, const int32_t* tokens, const int32_t* tasks, const float* weights, void* y,
    int xrows, int rows, int n, int k, int experts, int taskcount,
    int tile_m, cudaStream_t stream) {
  if (xrows <= 0 || rows <= 0 || n <= 0 || k <= 0 || k % 256 || experts <= 0 ||
      taskcount <= 0 || (tile_m != 16 && tile_m != 32 && tile_m != 64))
    return cudaErrorInvalidValue;
  // x is unused; the fused SwiGLU quantizer already populated a/sa.
#define ROUTED_LAUNCH(M,N) \
  iq2xxs_tiled<M,N,true,true,true,false,true,true><<<dim3((n+N-1)/N,taskcount), (M/16)*(N/32)*32, 0, stream>>>( \
      static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa, \
      static_cast<__nv_bfloat16*>(y), rows, n, k, tokens, tasks, xrows, experts, static_cast<const __half*>(bases), weights)
  if (tile_m == 16) { ROUTED_LAUNCH(16,IQ2XXS_SHARED_N); }
  else if (tile_m == 32) { ROUTED_LAUNCH(32,IQ2XXS_SHARED_N); }
  else { ROUTED_LAUNCH(64,IQ2XXS_SHARED_N); }
#undef ROUTED_LAUNCH
  return cudaGetLastError();
}

// Input native32-row tasks; sparse16 list followed by shared32 list.
extern "C" int iq2xxs_classify1632(const int32_t* input, const int32_t* tokens,
                                 int32_t* output, int count, int rows, cudaStream_t stream) {
  if (count <= 0 || count > 32767 || rows <= 0) return cudaErrorInvalidValue;
  iq_classify_rows<16><<<(count + 255) / 256, 256, 0, stream>>>(input, tokens, output, count, rows);
  return cudaGetLastError();
}

extern "C" int iq2xxs_routed_int8_slab_hybrid1632(const void* x, void* a, float* sa,
    const void* raw, const void* bases, const int32_t* tokens, const int32_t* tasks, void* y,
    int xrows, int rows, int n, int k, int experts, int taskcount,
    int tile_m, cudaStream_t stream) {
  if (xrows <= 0 || rows <= 0 || n <= 0 || k <= 0 || k % 256 || experts <= 0 ||
      taskcount <= 0 || taskcount % 2 || taskcount > 65534 || tile_m != 32)
    return cudaErrorInvalidValue;
  quantize<<<(xrows * (k / 32) + 7) / 8, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<int8_t*>(a), sa, xrows * (k / 32));
  taskcount /= 2;
  iq2xxs_tiled<16,IQ2XXS_REGISTER_N,true,true,true,false,true,false,false,true><<<dim3((n+IQ2XXS_REGISTER_N-1)/IQ2XXS_REGISTER_N,taskcount), (16/16)*(IQ2XXS_REGISTER_N/32)*32, 0, stream>>>(
      static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa,
      static_cast<__nv_bfloat16*>(y), rows, n, k, tokens, tasks, xrows, experts,
      static_cast<const __half*>(bases));
  tasks += taskcount * 2;
  iq2xxs_tiled<32,IQ2XXS_SHARED_N,true,true,true,false,true,false,false,false><<<dim3((n+IQ2XXS_SHARED_N-1)/IQ2XXS_SHARED_N,taskcount), (32/16)*(IQ2XXS_SHARED_N/32)*32, 0, stream>>>(
      static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa,
      static_cast<__nv_bfloat16*>(y), rows, n, k, tokens, tasks, xrows, experts,
      static_cast<const __half*>(bases));

  return cudaGetLastError();
}

extern "C" int iq2xxs_routed_int8_slab_hybrid1632_weighted_quantized(const void* x, void* a, float* sa,
    const void* raw, const void* bases, const int32_t* tokens, const int32_t* tasks, const float* weights, void* y,
    int xrows, int rows, int n, int k, int experts, int taskcount,
    int tile_m, cudaStream_t stream) {
  if (xrows <= 0 || rows <= 0 || n <= 0 || k <= 0 || k % 256 || experts <= 0 ||
      taskcount <= 0 || taskcount % 2 || taskcount > 65534 || tile_m != 32)
    return cudaErrorInvalidValue;
  // x is unused: a/sa already contain the fused quantizer output.
  taskcount /= 2;
  iq2xxs_tiled<16,IQ2XXS_REGISTER_N,true,true,true,false,true,true,false,true><<<dim3((n+IQ2XXS_REGISTER_N-1)/IQ2XXS_REGISTER_N,taskcount), (16/16)*(IQ2XXS_REGISTER_N/32)*32, 0, stream>>>(
      static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa,
      static_cast<__nv_bfloat16*>(y), rows, n, k, tokens, tasks, xrows, experts,
      static_cast<const __half*>(bases), weights);
  tasks += taskcount * 2;
  iq2xxs_tiled<32,IQ2XXS_SHARED_N,true,true,true,false,true,true,false,false><<<dim3((n+IQ2XXS_SHARED_N-1)/IQ2XXS_SHARED_N,taskcount), (32/16)*(IQ2XXS_SHARED_N/32)*32, 0, stream>>>(
      static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa,
      static_cast<__nv_bfloat16*>(y), rows, n, k, tokens, tasks, xrows, experts,
      static_cast<const __half*>(bases), weights);

  return cudaGetLastError();
}

// IQ3_XXS: lossless12-byte descriptors (8 indices + packed signs/scales)
// per32 weights, plus the original FP16 base per256. Magnitudes fit INT8.
extern "C" int iq3xxs_routed_int8_slab(const void* x, void* a, float* sa,
    const void* raw, const void* bases, const int32_t* tokens, const int32_t* tasks, void* y,
    int xrows, int rows, int n, int k, int experts, int taskcount,
    int tile_m, cudaStream_t stream) {
  if (xrows <= 0 || rows <= 0 || n <= 0 || k <= 0 || k % 256 || experts <= 0 ||
      taskcount <= 0 || taskcount > 65535 || (tile_m != 16 && tile_m != 32 && tile_m != 64))
    return cudaErrorInvalidValue;
  quantize<<<(xrows * (k / 32) + 7) / 8, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<int8_t*>(a), sa, xrows * (k / 32));
  if (tile_m == 16)
    iq2xxs_tiled<16,IQ2XXS_SHARED_N,true,true,true,false,true,false,false,false,false,true><<<dim3((n+IQ2XXS_SHARED_N-1)/IQ2XXS_SHARED_N,taskcount), (16/16)*(IQ2XXS_SHARED_N/32)*32, 0, stream>>>(
        static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa,
        static_cast<__nv_bfloat16*>(y), rows, n, k, tokens, tasks, xrows, experts,
        static_cast<const __half*>(bases));
  else if (tile_m == 32)
    iq2xxs_tiled<32,IQ2XXS_SHARED_N,true,true,true,false,true,false,false,false,false,true><<<dim3((n+IQ2XXS_SHARED_N-1)/IQ2XXS_SHARED_N,taskcount), (32/16)*(IQ2XXS_SHARED_N/32)*32, 0, stream>>>(
        static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa,
        static_cast<__nv_bfloat16*>(y), rows, n, k, tokens, tasks, xrows, experts,
        static_cast<const __half*>(bases));
  else if (tile_m == 64)
    iq2xxs_tiled<64,IQ2XXS_SHARED_N,true,true,true,false,true,false,false,false,false,true><<<dim3((n+IQ2XXS_SHARED_N-1)/IQ2XXS_SHARED_N,taskcount), (64/16)*(IQ2XXS_SHARED_N/32)*32, 0, stream>>>(
        static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa,
        static_cast<__nv_bfloat16*>(y), rows, n, k, tokens, tasks, xrows, experts,
        static_cast<const __half*>(bases));
  return cudaGetLastError();
}
extern "C" int iq3xxs_routed_int8_slab_weighted(const void* x, void* a, float* sa,
    const void* raw, const void* bases, const int32_t* tokens, const int32_t* tasks, const float* weights, void* y,
    int xrows, int rows, int n, int k, int experts, int taskcount,
    int tile_m, cudaStream_t stream) {
  if (xrows <= 0 || rows <= 0 || n <= 0 || k <= 0 || k % 256 || experts <= 0 ||
      taskcount <= 0 || taskcount > 65535 || (tile_m != 16 && tile_m != 32 && tile_m != 64))
    return cudaErrorInvalidValue;
  quantize<<<(xrows * (k / 32) + 7) / 8, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<int8_t*>(a), sa, xrows * (k / 32));
  if (tile_m == 16)
    iq2xxs_tiled<16,IQ2XXS_SHARED_N,true,true,true,false,true,true,false,false,false,true><<<dim3((n+IQ2XXS_SHARED_N-1)/IQ2XXS_SHARED_N,taskcount), (16/16)*(IQ2XXS_SHARED_N/32)*32, 0, stream>>>(
        static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa,
        static_cast<__nv_bfloat16*>(y), rows, n, k, tokens, tasks, xrows, experts,
        static_cast<const __half*>(bases), weights);
  else if (tile_m == 32)
    iq2xxs_tiled<32,IQ2XXS_SHARED_N,true,true,true,false,true,true,false,false,false,true><<<dim3((n+IQ2XXS_SHARED_N-1)/IQ2XXS_SHARED_N,taskcount), (32/16)*(IQ2XXS_SHARED_N/32)*32, 0, stream>>>(
        static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa,
        static_cast<__nv_bfloat16*>(y), rows, n, k, tokens, tasks, xrows, experts,
        static_cast<const __half*>(bases), weights);
  else if (tile_m == 64)
    iq2xxs_tiled<64,IQ2XXS_SHARED_N,true,true,true,false,true,true,false,false,false,true><<<dim3((n+IQ2XXS_SHARED_N-1)/IQ2XXS_SHARED_N,taskcount), (64/16)*(IQ2XXS_SHARED_N/32)*32, 0, stream>>>(
        static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa,
        static_cast<__nv_bfloat16*>(y), rows, n, k, tokens, tasks, xrows, experts,
        static_cast<const __half*>(bases), weights);
  return cudaGetLastError();
}
extern "C" int iq3xxs_routed_int8_slab_weighted_quantized(const void* x, void* a, float* sa,
    const void* raw, const void* bases, const int32_t* tokens, const int32_t* tasks, const float* weights, void* y,
    int xrows, int rows, int n, int k, int experts, int taskcount,
    int tile_m, cudaStream_t stream) {
  if (xrows <= 0 || rows <= 0 || n <= 0 || k <= 0 || k % 256 || experts <= 0 ||
      taskcount <= 0 || taskcount > 65535 || (tile_m != 16 && tile_m != 32 && tile_m != 64))
    return cudaErrorInvalidValue;
  if (tile_m == 16)
    iq2xxs_tiled<16,IQ2XXS_SHARED_N,true,true,true,false,true,true,false,false,false,true><<<dim3((n+IQ2XXS_SHARED_N-1)/IQ2XXS_SHARED_N,taskcount), (16/16)*(IQ2XXS_SHARED_N/32)*32, 0, stream>>>(
        static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa,
        static_cast<__nv_bfloat16*>(y), rows, n, k, tokens, tasks, xrows, experts,
        static_cast<const __half*>(bases), weights);
  else if (tile_m == 32)
    iq2xxs_tiled<32,IQ2XXS_SHARED_N,true,true,true,false,true,true,false,false,false,true><<<dim3((n+IQ2XXS_SHARED_N-1)/IQ2XXS_SHARED_N,taskcount), (32/16)*(IQ2XXS_SHARED_N/32)*32, 0, stream>>>(
        static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa,
        static_cast<__nv_bfloat16*>(y), rows, n, k, tokens, tasks, xrows, experts,
        static_cast<const __half*>(bases), weights);
  else if (tile_m == 64)
    iq2xxs_tiled<64,IQ2XXS_SHARED_N,true,true,true,false,true,true,false,false,false,true><<<dim3((n+IQ2XXS_SHARED_N-1)/IQ2XXS_SHARED_N,taskcount), (64/16)*(IQ2XXS_SHARED_N/32)*32, 0, stream>>>(
        static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa,
        static_cast<__nv_bfloat16*>(y), rows, n, k, tokens, tasks, xrows, experts,
        static_cast<const __half*>(bases), weights);
  return cudaGetLastError();
}

// Native32 tasks: partial (<32 valid rows) first, full32 second. Every slot
// is overwritten on replay; route planner guarantees valid-prefix tiles.
extern "C" int iq3xxs_classify_full32(const int32_t* input, const int32_t* tokens,
    int32_t* output, int count, int rows, cudaStream_t stream) {
  if (count <= 0 || count > 32767 || rows <= 0) return cudaErrorInvalidValue;
  iq_classify_rows<31><<<(count + 255) / 256, 256, 0, stream>>>(input, tokens, output, count, rows);
  return cudaGetLastError();
}

// Quantize once, then select FC1 register allocation without host readback.
extern "C" int iq3xxs_routed_int8_slab_adaptive(const void* x, void* a, float* sa,
    const void* raw, const void* bases, const int32_t* tokens, const int32_t* tasks, void* y,
    int xrows, int rows, int n, int k, int experts, int taskcount,
    int tile_m, cudaStream_t stream) {
  if (xrows <= 0 || rows <= 0 || n <= 0 || k <= 0 || k % 256 || experts <= 0 ||
      taskcount <= 0 || taskcount % 2 || taskcount > 65534 || tile_m != 32)
    return cudaErrorInvalidValue;
  quantize<<<(xrows * (k / 32) + 7) / 8, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<int8_t*>(a), sa, xrows * (k / 32));
  taskcount /= 2;
#define IQ3_ADAPTIVE_LAUNCH(BOUND) \
  iq2xxs_tiled<32,64,true,true,true,false,true,false,false,false,false,true,BOUND><<<dim3((n+63)/64,taskcount),128,0,stream>>>( \
      static_cast<const int8_t*>(a), static_cast<const uint8_t*>(raw), sa, \
      static_cast<__nv_bfloat16*>(y), rows, n, k, tokens, tasks, xrows, experts, static_cast<const __half*>(bases))
  IQ3_ADAPTIVE_LAUNCH(IQ3XXS_MIN_BLOCKS);
  tasks += taskcount * 2;
  IQ3_ADAPTIVE_LAUNCH(1);
#undef IQ3_ADAPTIVE_LAUNCH
  return cudaGetLastError();
}
