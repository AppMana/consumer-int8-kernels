// SPDX-License-Identifier: Apache-2.0
#include "kernels.h"
#define FLASHATTENTION_DISABLE_DROPOUT
#include "flash.h"
#include "flash_fwd_kernel.h"
#include <stdexcept>

namespace consumer_fa2 {
template <typename Traits>
__global__ __launch_bounds__(Traits::kNThreads) void fa2_prefill_kernel(
    const flash::Flash_fwd_params params) {
  flash::compute_attn<Traits, false, true, false, false, false, true, false,
                      false>(params);
}

template <int M, int N, int Warps, bool Qregs = false, bool Share = false>
void launch(flash::Flash_fwd_params &p, cudaStream_t stream) {
  using Traits = Flash_fwd_kernel_traits<256, M, N, Warps, Qregs, Share,
                                         cutlass::bfloat16_t>;
  static_assert(Traits::kSmemSize <= 101376, "SM86 shared-memory limit");
  auto kernel = fa2_prefill_kernel<Traits>;
  if constexpr (Traits::kSmemSize >= 49152) {
    const auto status = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Traits::kSmemSize);
    if (status != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(status));
  }
  kernel<<<dim3((p.seqlen_q + M - 1) / M, p.b, p.h), Traits::kNThreads,
           Traits::kSmemSize, stream>>>(p);
}
} // namespace consumer_fa2

void fa2_prefill(const void *q, const void *k, const void *v,
                 const int32_t *offsets, const int32_t *lengths, void *out,
                 float *lse, int tokens, int heads, int kv_heads, int sequences,
                 int capacity, int max_query, int64_t qt, int64_t qh,
                 int64_t ot, int64_t oh, float scale, int schedule,
                 cudaStream_t stream) {
  flash::Flash_fwd_params p{};
  p.q_ptr = const_cast<void *>(q);
  p.k_ptr = const_cast<void *>(k);
  p.v_ptr = const_cast<void *>(v);
  p.o_ptr = out;
  p.softmax_lse_ptr = lse;
  p.q_row_stride = qt;
  p.q_head_stride = qh;
  p.k_row_stride = p.v_row_stride = int64_t(kv_heads) * 256;
  p.k_head_stride = p.v_head_stride = 256;
  p.k_batch_stride = p.v_batch_stride = int64_t(capacity) * kv_heads * 256;
  p.o_row_stride = ot;
  p.o_head_stride = oh;
  p.h = heads;
  p.h_k = kv_heads;
  p.h_h_k_ratio = heads / kv_heads;
  p.b = sequences;
  p.seqlen_q = max_query;
  p.seqlen_k = capacity;
  p.seqlen_q_rounded = (max_query + 127) / 128 * 128;
  p.seqlen_k_rounded = (capacity + 127) / 128 * 128;
  p.d = p.d_rounded = 256;
  p.total_q = tokens;
  p.cu_seqlens_q = const_cast<int32_t *>(offsets);
  p.seqused_k = const_cast<int32_t *>(lengths);
  p.scale_softmax = p.scale_softmax_rp_dropout = scale;
  p.scale_softmax_log2 = scale * 1.4426950408889634f;
  p.p_dropout = p.rp_dropout = 1.f;
  p.p_dropout_in_uint8_t = 255;
  p.window_size_left = capacity;
  p.window_size_right = 0;
  p.is_bf16 = p.is_causal = p.unpadded_lse = true;
  if (schedule == 0) {
    int device, sms;
    auto status = cudaGetDevice(&device);
    if (status == cudaSuccess)
      status =
          cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    if (status != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(status));
    const int64_t blocks128 =
        int64_t((max_query + 127) / 128) * sequences * heads;
    // Avoid a mostly empty second wave and short causal triangles. Keep
    // unmeasured SM counts on the standard FA2 tile.
    schedule = sms == 64 && blocks128 >= 2 * sms &&
                       int64_t(capacity) >= 4 * int64_t(max_query)
                   ? 8
                   : 1;
  }
  switch (schedule) {
  case 1:
    consumer_fa2::launch<64, 64, 4>(p, stream);
    break;
  case 2:
    consumer_fa2::launch<64, 32, 4>(p, stream);
    break;
  case 3:
    consumer_fa2::launch<128, 32, 8>(p, stream);
    break;
  case 4:
    consumer_fa2::launch<32, 32, 2>(p, stream);
    break;
  case 5:
    consumer_fa2::launch<32, 64, 2>(p, stream);
    break;
  case 6:
    consumer_fa2::launch<64, 64, 4, true, true>(p, stream);
    break;
  case 7:
    consumer_fa2::launch<128, 64, 8, true, true>(p, stream);
    break;
  case 8:
    consumer_fa2::launch<128, 32, 8, true, true>(p, stream);
    break;
  default:
    throw std::invalid_argument("Unknown FA2 schedule");
  }
}
