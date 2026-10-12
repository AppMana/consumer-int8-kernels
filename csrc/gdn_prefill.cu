// SPDX-License-Identifier: Apache-2.0
// AOT SM86 launch adapter for pinned Apache-2.0 NInfer chunked GDN kernels.
#include "kernels.h"
#include "ops/linear_attention/gated_delta_net/chunked/prepare_wy_wu.cuh"
#include "ops/linear_attention/gated_delta_net/chunked/state_passing.cuh"
#include "ops/linear_attention/gated_delta_net/chunked/output.cuh"

namespace gdn = ninfer::ops::detail::gated_delta_net;
namespace chunk = gdn::chunked;
namespace {
template <int KPanel, int WPanel, int Warps>
cudaError_t prepare(const __nv_bfloat16* k, const __nv_bfloat16* v,
    const float* g, const float* beta, __nv_bfloat16* w, __nv_bfloat16* u,
    float* cumulative, gdn::head_map map, int chunks, cudaStream_t stream) {
  constexpr int shared = chunk::prepare_wy_wu::kernel_dims<KPanel, WPanel>::SMEM_FLOATS * 4;
  static_assert(shared <= 101376, "SM86 shared-memory limit");
  const auto kernel = chunk::prepare_wy_wu::prepare_wy_wu_kernel<KPanel, WPanel, Warps>;
  auto status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
  if (status != cudaSuccess) return status;
  kernel<<<dim3(chunks, map.H_v), Warps * 32, shared, stream>>>(
      k, v, g, beta, w, u, cumulative, map);
  return cudaGetLastError();
}

template <int Strip>
cudaError_t propagate(const __nv_bfloat16* w, const __nv_bfloat16* u,
    const __nv_bfloat16* k, const float* cumulative, const float* initial,
    __nv_bfloat16* v_new, __nv_bfloat16* h_chunk, float* final,
    gdn::head_map map, int chunks, cudaStream_t stream) {
  using D = chunk::state_passing::kernel_dims<Strip>;
  constexpr int shared = chunk::state_passing::smem_layout<Strip>::SMEM_FLOATS * 4;
  static_assert(shared <= 101376, "SM86 shared-memory limit");
  const auto kernel = chunk::state_passing::state_passing_kernel<Strip>;
  auto status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
  if (status != cudaSuccess) return status;
  kernel<<<map.H_v * D::D_STRIPS, D::THREADS, shared, stream>>>(
      w, u, k, cumulative, initial, v_new, h_chunk, final, map, chunks);
  return cudaGetLastError();
}
}  // namespace

cudaError_t gdn_prefill(const void* q, const void* k, const void* v,
    const float* g, const float* beta, const float* initial, float* final,
    void* out, float* cumulative, void* w, void* u, void* v_new, void* h_chunk,
    int tokens, int hq, int hv, int num_sms, cudaStream_t stream) {
  const auto map = gdn::head_map::of(hq, hv);
  const int chunks = tokens / 64;
  const auto* kb = static_cast<const __nv_bfloat16*>(k);
  const auto* vb = static_cast<const __nv_bfloat16*>(v);
  auto* wb = static_cast<__nv_bfloat16*>(w);
  auto* ub = static_cast<__nv_bfloat16*>(u);
  auto* vnb = static_cast<__nv_bfloat16*>(v_new);
  auto* hcb = static_cast<__nv_bfloat16*>(h_chunk);
  cudaError_t status;
  if (hv >= 48) {
    status = prepare<64, 32, 8>(kb, vb, g, beta, wb, ub, cumulative, map, chunks, stream);
  } else {
    status = prepare<32, 16, 4>(kb, vb, g, beta, wb, ub, cumulative, map, chunks, stream);
  }
  if (status != cudaSuccess) return status;
  if (hv >= 48) {
    status = propagate<16>(wb, ub, kb, cumulative, initial, vnb, hcb, final, map, chunks, stream);
  } else {
    status = propagate<32>(wb, ub, kb, cumulative, initial, vnb, hcb, final, map, chunks, stream);
  }
  if (status != cudaSuccess) return status;

  constexpr int shared = chunk::output::kernel_dims::SMEM_BYTES;
  static_assert(shared <= 101376, "SM86 shared-memory limit");
  // Use the actual GPU's SM count (A5000 and 3090 differ), not NInfer's
  // fixed 5090 launch grid. Each CTA strides through its assigned chunks.
  const int jobs = (chunks * hv + num_sms * 4 - 1) / (num_sms * 4);
  const int grid_chunks = (chunks + jobs - 1) / jobs;
  const auto kernel = chunk::output::output_kernel<true>;
  status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
  if (status != cudaSuccess) return status;
  kernel<<<dim3(grid_chunks, hv), chunk::output::THREADS, shared, stream>>>(
      static_cast<const __nv_bfloat16*>(q), kb, vnb, cumulative, hcb,
      static_cast<__nv_bfloat16*>(out), map, 0.08838834764831844f, chunks);
  return cudaGetLastError();
}
