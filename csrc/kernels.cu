// SPDX-License-Identifier: Apache-2.0
// MMA fragment mapping follows AppMana/FlashMLA debug_imma_sm80.cu.
// GDN's four value rows per warp follow Don-Chad/ninfer-3090 recurrent.cuh.
#include "kernels.h"
#include <cuda_bf16.h>
#include <cmath>

namespace {
__device__ float warp_sum(float x) {
  for (int mask = 16; mask; mask >>= 1) x += __shfl_xor_sync(0xffffffff, x, mask);
  return x;
}
__global__ void quantize_rows_kernel(const __nv_bfloat16* x, int8_t* q,
                                     float* scales, int k) {
  __shared__ float maxima[8];
  float maximum = 0;
  const int row = blockIdx.x;
  for (int col = threadIdx.x; col < k; col += blockDim.x)
    maximum = fmaxf(maximum, fabsf(__bfloat162float(x[row * k + col])));
  for (int mask = 16; mask; mask >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, mask));
  if (!(threadIdx.x & 31)) maxima[threadIdx.x / 32] = maximum;
  __syncthreads();
  maximum = threadIdx.x < 8 ? maxima[threadIdx.x] : 0;
  for (int mask = 16; mask; mask >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, mask));
  if (!threadIdx.x) maxima[0] = fmaxf(maximum / 127.f, 1e-20f);
  __syncthreads();
  const float scale = maxima[0];
  if (!threadIdx.x) scales[row] = scale;
  for (int col = threadIdx.x; col < k; col += blockDim.x) {
    float value = __bfloat162float(x[row * k + col]) / scale;
    q[row * k + col] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(value))));
  }
}

// Keep each BF16 row in registers across the maximum reduction. Qwen's
// fixed widths permit vector loads/stores without a second global read.
struct alignas(16) Bf16Pack8 { __nv_bfloat16 values[8]; };
struct alignas(8) Int8Pack8 { int8_t values[8]; };
template <int K, int Threads = 256, bool Grouped = false>
__global__ void quantize_rows_vector_kernel(const __nv_bfloat16* x, int8_t* q,
                                           float* scales, int m = 1, int groups = 1) {
  constexpr int Packs = (K + Threads * 8 - 1) / (Threads * 8);
  __shared__ float maxima[Threads / 32];
  const int64_t out_row = int64_t(blockIdx.x) * K;
  const int64_t row = Grouped
      ? int64_t(blockIdx.x % m * groups + blockIdx.x / m) * K : out_row;
  float values[Packs][8], maximum = 0.f;
#pragma unroll
  for (int i = 0; i < Packs; ++i) {
    const int col = (i * Threads + threadIdx.x) * 8;
    Bf16Pack8 packed{};
    if (col < K) packed = *reinterpret_cast<const Bf16Pack8*>(x + row + col);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      values[i][j] = __bfloat162float(packed.values[j]);
      maximum = fmaxf(maximum, fabsf(values[i][j]));
    }
  }
#pragma unroll
  for (int mask = 16; mask; mask >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, mask));
  if (!(threadIdx.x & 31)) maxima[threadIdx.x / 32] = maximum;
  __syncthreads();
  maximum = threadIdx.x < Threads / 32 ? maxima[threadIdx.x] : 0;
#pragma unroll
  for (int mask = 16; mask; mask >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, mask));
  if (!threadIdx.x) maxima[0] = fmaxf(maximum / 127.f, 1e-20f);
  __syncthreads();
  const float scale = maxima[0];
  if (!threadIdx.x) scales[blockIdx.x] = scale;
#pragma unroll
  for (int i = 0; i < Packs; ++i) {
    const int col = (i * Threads + threadIdx.x) * 8;
    if (col < K) {
      Int8Pack8 result;
#pragma unroll
      for (int j = 0; j < 8; ++j)
        result.values[j] = static_cast<int8_t>(max(-127, min(127,
            __float2int_rn(values[i][j] / scale))));
      *reinterpret_cast<Int8Pack8*>(q + out_row + col) = result;
    }
  }
}

__global__ void channel_mm_kernel(const int8_t* a, const uint8_t* w,
                                  const float* sa, const __nv_bfloat16* sw,
                                  __nv_bfloat16* out, int m, int n, int k) {
  const int lane = threadIdx.x & 31, group = lane >> 2, tid = lane & 3;
  const int row_base = blockIdx.y * 16;
  const int col_base = blockIdx.x * 32 + (threadIdx.x >> 5) * 8;
  int accum[4] = {0, 0, 0, 0};
  for (int offset = 0; offset < k; offset += 32) {
    uint32_t av[4], bv[2];
    #pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
      const int row = row_base + group + ((reg & 1) ? 8 : 0);
      const int col = offset + tid * 4 + ((reg >> 1) * 16);
      av[reg] = row < m ? *reinterpret_cast<const uint32_t*>(a + row * k + col) : 0;
    }
    #pragma unroll
    for (int reg = 0; reg < 2; ++reg) {
      const int col = col_base + group;
      const int kk = offset + tid * 4 + reg * 16;
      bv[reg] = col < n ? (*reinterpret_cast<const uint32_t*>(w + col * k + kk) ^ 0x80808080u) : 0;
    }
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(accum[0]), "+r"(accum[1]), "+r"(accum[2]), "+r"(accum[3])
        : "r"(av[0]), "r"(av[1]), "r"(av[2]), "r"(av[3]), "r"(bv[0]), "r"(bv[1]));
  }
  #pragma unroll
  for (int reg = 0; reg < 4; ++reg) {
    const int row = row_base + group + ((reg >> 1) * 8);
    const int col = col_base + tid * 2 + (reg & 1);
    if (row < m && col < n)
      out[row * n + col] = __float2bfloat16(float(accum[reg]) * sa[row] * __bfloat162float(sw[col]));
  }
}

template <bool Packed, typename Slot>
__global__ void gdn_decode_kernel(const __nv_bfloat16* q, const __nv_bfloat16* k,
                                  const __nv_bfloat16* v, const float* g,
                                  const float* beta, float* state, const Slot* slots,
                                  __nv_bfloat16* out, int hq, int hv, int num_slots,
                                  int64_t q_stride, int64_t v_stride, int64_t state_stride,
                                  const __nv_bfloat16* a, const __nv_bfloat16* raw_b,
                                  const float* A_log, const void* dt_bias, bool dt_float,
                                  int64_t a_stride, int64_t b_stride, int64_t slot_stride) {
  const int batch = blockIdx.z, head = blockIdx.y;
  const int lane = threadIdx.x & 31;
  const int v_base = blockIdx.x * 16 + (threadIdx.x >> 5) * 4;
  const int64_t slot = slots[batch * slot_stride];
  if (slot <= 0 || slot >= num_slots) {
    if (lane < 4) out[(batch * hv + head) * 128 + v_base + lane] = __float2bfloat16(0);
    return;
  }
  const int key_head = head / (hv / hq);
  float qr[4], kr[4], qs = 0, ks = 0;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int64_t idx = batch * q_stride + key_head * 128 + lane * 4 + j;
    qr[j] = __bfloat162float(q[idx]); kr[j] = __bfloat162float(k[idx]);
    qs += qr[j] * qr[j]; ks += kr[j] * kr[j];
  }
  qs = rsqrtf(warp_sum(qs) + 1e-6f) * 0.08838834764831844f;
  ks = rsqrtf(warp_sum(ks) + 1e-6f);
  #pragma unroll
  for (int j = 0; j < 4; ++j) { qr[j] *= qs; kr[j] *= ks; }
  float decay, b;
  if constexpr (Packed) {
    const float bias = dt_float ? static_cast<const float*>(dt_bias)[head]
        : __bfloat162float(static_cast<const __nv_bfloat16*>(dt_bias)[head]);
    const float x = __bfloat162float(a[batch * a_stride + head]) + bias;
    const float softplus = x <= 20.f ? log1pf(expf(x)) : x;
    decay = expf(-expf(A_log[head]) * softplus);
    b = 1.f / (1.f + expf(-__bfloat162float(raw_b[batch * b_stride + head])));
  } else {
    decay = expf(g[batch * hv + head]);
    b = beta[batch * hv + head];
  }
  #pragma unroll
  for (int r = 0; r < 4; ++r) {
    const int64_t offset = slot * state_stride + (head * 128 + v_base + r) * 128 + lane * 4;
    float4 packed = *reinterpret_cast<const float4*>(state + offset);
    float s[4] = {packed.x * decay, packed.y * decay, packed.z * decay, packed.w * decay};
    float dot = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) dot += s[j] * kr[j];
    dot = warp_sum(dot);
    const float delta = b * (__bfloat162float(v[batch * v_stride + head * 128 + v_base + r]) - dot);
    float output = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) { s[j] += delta * kr[j]; output += s[j] * qr[j]; }
    *reinterpret_cast<float4*>(state + offset) = make_float4(s[0], s[1], s[2], s[3]);
    output = warp_sum(output);
    if (!lane) out[(batch * hv + head) * 128 + v_base + r] = __float2bfloat16(output);
  }
}

// Match Triton's per-token-head cache encoding, including scale rounding.
__device__ __forceinline__ float cache_divide(float x, float y) {
  float result;
  asm("div.full.f32 %0, %1, %2;" : "=f"(result) : "f"(x), "f"(y));
  return result;
}

__global__ void kv_cache_update_kernel(
    const __nv_bfloat16* key, const __nv_bfloat16* value, int8_t* cache,
    const int64_t* slots, int tokens, int heads, int blocks, int block_size,
    int64_t kt, int64_t kh, int64_t vt, int64_t vh,
    int64_t cb, int64_t ch, int64_t ct) {
  const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (row >= tokens * heads) return;
  const int token = row / heads, head = row % heads, lane = threadIdx.x & 31;
  const int64_t slot = slots[token];
  if (slot < 0 || slot >= int64_t(blocks) * block_size) return;
  float k[8], v[8], km = 0, vm = 0;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    k[i] = __bfloat162float(key[token * kt + head * kh + lane + i * 32]);
    v[i] = __bfloat162float(value[token * vt + head * vh + lane + i * 32]);
    km = fmaxf(km, fabsf(k[i])); vm = fmaxf(vm, fabsf(v[i]));
  }
  for (int mask = 16; mask; mask >>= 1) {
    km = fmaxf(km, __shfl_xor_sync(0xffffffff, km, mask));
    vm = fmaxf(vm, __shfl_xor_sync(0xffffffff, vm, mask));
  }
  const float ks = fmaxf(cache_divide(km, 127.f), 1e-6f);
  const float vs = fmaxf(cache_divide(vm, 127.f), 1e-6f);
  const float ki = cache_divide(1.f, ks), vi = cache_divide(1.f, vs);
  int8_t* destination = cache + (slot / block_size) * cb + head * ch + (slot % block_size) * ct;
  if (!lane) {
    *reinterpret_cast<float*>(destination + 256) = ks;
    *reinterpret_cast<float*>(destination + 516) = vs;
  }
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float kq = k[i] * ki, vq = v[i] * vi;
    destination[lane + i * 32] = static_cast<int8_t>(
        fminf(127.f, fmaxf(-128.f, kq + (kq >= 0 ? 0.5f : -0.5f))));
    destination[260 + lane + i * 32] = static_cast<int8_t>(
        fminf(127.f, fmaxf(-128.f, vq + (vq >= 0 ? 0.5f : -0.5f))));
  }
}
}  // namespace

void quantize_rows(const void* x, int8_t* codes, float* scales, int m, int k, cudaStream_t stream) {
  if (m >= 1024 && reinterpret_cast<uintptr_t>(x) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(codes) % 8 == 0) {
#define QUANT(K) case K: quantize_rows_vector_kernel<K><<<m, 256, 0, stream>>>( \
    static_cast<const __nv_bfloat16*>(x), codes, scales); return
    switch (k) { QUANT(3072); QUANT(5120); QUANT(6144); QUANT(8704); QUANT(17408); }
#undef QUANT
  }
  quantize_rows_kernel<<<m, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(x), codes, scales, k);
}
void quantize_grouped4096(const void* x, int8_t* codes, float* scales,
                          int m, int groups, cudaStream_t stream) {
  if (!m) return;
  quantize_rows_vector_kernel<4096, 256, true><<<m * groups, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), codes, scales, m, groups);
}
void channel_mm(const int8_t* a, const uint8_t* w, const float* sa,
                const void* sw, void* out, int m, int n, int k, cudaStream_t stream) {
  channel_mm_kernel<<<dim3((n + 31) / 32, (m + 15) / 16), 128, 0, stream>>>(
      a, w, sa, static_cast<const __nv_bfloat16*>(sw), static_cast<__nv_bfloat16*>(out), m, n, k);
}
void gdn_decode(const void* q, const void* k, const void* v, const float* g,
                const float* beta, float* state, const int64_t* slots, void* out,
                int batch, int hq, int hv, int num_slots, cudaStream_t stream) {
  gdn_decode_kernel<false, int64_t><<<dim3(8, hv, batch), 128, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(q), static_cast<const __nv_bfloat16*>(k),
      static_cast<const __nv_bfloat16*>(v), g, beta, state, slots,
      static_cast<__nv_bfloat16*>(out), hq, hv, num_slots,
      hq * 128, hv * 128, int64_t(hv) * 128 * 128,
      nullptr, nullptr, nullptr, nullptr, false, 0, 0, 1);
}

void kv_cache_update(const void* key, const void* value, int8_t* cache,
                      const int64_t* slots, int tokens, int heads, int blocks,
                      int block_size, int64_t kt, int64_t kh, int64_t vt, int64_t vh,
                      int64_t cb, int64_t ch, int64_t ct, cudaStream_t stream) {
  kv_cache_update_kernel<<<(tokens * heads + 3) / 4, 128, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(key), static_cast<const __nv_bfloat16*>(value),
      cache, slots, tokens, heads, blocks, block_size, kt, kh, vt, vh, cb, ch, ct);
}

void gdn_packed_decode(const void* qkv, const void* a, const void* b,
    const float* A_log, const void* dt_bias, float* state, const void* slots,
    void* out, int batch, int hq, int hv, int num_slots, int64_t qkv_stride,
    int64_t a_stride, int64_t b_stride, int64_t state_stride, int64_t slot_stride,
    bool dt_float, bool slots_int64, cudaStream_t stream) {
  const auto* q = static_cast<const __nv_bfloat16*>(qkv);
  const auto* k = q + hq * 128;
  const auto* v = k + hq * 128;
  if (slots_int64) {
    gdn_decode_kernel<true, int64_t><<<dim3(8, hv, batch), 128, 0, stream>>>(
        q, k, v, nullptr, nullptr, state, static_cast<const int64_t*>(slots),
        static_cast<__nv_bfloat16*>(out), hq, hv, num_slots,
        qkv_stride, qkv_stride, state_stride,
        static_cast<const __nv_bfloat16*>(a), static_cast<const __nv_bfloat16*>(b),
        A_log, dt_bias, dt_float, a_stride, b_stride, slot_stride);
  } else {
    gdn_decode_kernel<true, int32_t><<<dim3(8, hv, batch), 128, 0, stream>>>(
        q, k, v, nullptr, nullptr, state, static_cast<const int32_t*>(slots),
        static_cast<__nv_bfloat16*>(out), hq, hv, num_slots,
        qkv_stride, qkv_stride, state_stride,
        static_cast<const __nv_bfloat16*>(a), static_cast<const __nv_bfloat16*>(b),
        A_log, dt_bias, dt_float, a_stride, b_stride, slot_stride);
  }
}
