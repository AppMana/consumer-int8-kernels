// SPDX-License-Identifier: Apache-2.0
#include "kernels.h"
#include <cuda_bf16.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <stdexcept>

namespace {
template<int M, int N, int K, int WM, int WN, int Stages>
void run(const int8_t* a, const int8_t* w, int32_t* workspace,
          int m, int n, int k, cudaStream_t stream) {
  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      int32_t, cutlass::layout::RowMajor, int32_t,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<M, N, K>, cutlass::gemm::GemmShape<WM, WN, 64>,
      cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::LinearCombination<int32_t, 4, int32_t, int32_t>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages>;
  static_assert(sizeof(typename Gemm::GemmKernel::SharedStorage) <= 101376,
                "SM86 shared memory limit");
  typename Gemm::Arguments args({m, n, k}, {a, k}, {w, k},
                                {workspace, n}, {workspace, n}, {1, 0});
  Gemm gemm;
  auto status = gemm(args, nullptr, stream);
  if (status != cutlass::Status::kSuccess)
    throw std::runtime_error(cutlassGetStatusString(status));
}
__global__ void rescale(const int32_t* accum, const float* sa,
                        const __nv_bfloat16* sw, __nv_bfloat16* out,
                        int m, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m * n;
       i += blockDim.x * gridDim.x)
    out[i] = __float2bfloat16(float(accum[i]) * sa[i / n] * __bfloat162float(sw[i % n]));
}
}  // namespace

void cutlass_channel_mm(const int8_t* a, const int8_t* w, const float* sa,
                         const void* sw, int32_t* workspace, void* out,
                         int m, int n, int k, cudaStream_t stream) {
  // Explicit SM86 schedules: no M64 path may request SM80's 120 KiB shared memory.
  if (m <= 16) run<16, 64, 128, 16, 64, 5>(a, w, workspace, m, n, k, stream);
  else if (m <= 32) run<32, 64, 128, 32, 64, 4>(a, w, workspace, m, n, k, stream);
  else if (m <= 128) run<64, 128, 64, 32, 64, 4>(a, w, workspace, m, n, k, stream);
  else run<128, 128, 64, 64, 64, 5>(a, w, workspace, m, n, k, stream);
  rescale<<<(m * n + 255) / 256, 256, 0, stream>>>(
      workspace, sa, static_cast<const __nv_bfloat16*>(sw),
      static_cast<__nv_bfloat16*>(out), m, n);
}
