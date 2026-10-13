// SPDX-License-Identifier: Apache-2.0
// CUTLASS SM80 visitor construction follows vLLM's scaled_mm_c2x.cuh.
// This variant preserves BF16 channel scales and row-first scaling, and
// uses ordinary tiled scheduling without an INT32 output workspace.
#include "kernels.h"
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <stdexcept>

namespace {
using namespace cute;
template<int M, int N, int K, int WM, int WN, int Stages, int Swizzle = 1>
void fused_run(const int8_t* a, const int8_t* w, const float* sa,
               const void* sw, void* out, int m, int n, int k,
               cudaStream_t stream) {
  using BF16 = cutlass::bfloat16_t;
  using Tile = cutlass::gemm::GemmShape<M, N, K>;
  using Warp = cutlass::gemm::GemmShape<WM, WN, 64>;
  using Map = cutlass::epilogue::threadblock::OutputTileThreadLayout<Tile, Warp, float, 4, 1>;
  using Acc = cutlass::epilogue::threadblock::VisitorAccFetch;
  using RowScale = cutlass::epilogue::threadblock::VisitorColBroadcast<
      Map, float, Stride<_1, _0, _0>>;
  using ChannelScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      Map, BF16, Stride<_0, _1, _0>>;
  using MulFloat = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using MulBF16 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, BF16, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using First = cutlass::epilogue::threadblock::Sm80EVT<MulFloat, Acc, RowScale>;
  using Second = cutlass::epilogue::threadblock::Sm80EVT<MulBF16, First, ChannelScale>;
  using Store = cutlass::epilogue::threadblock::VisitorAuxStore<
      Map, BF16, cutlass::FloatRoundStyle::round_to_nearest, Stride<int64_t, _1, _0>>;
  using EVT = cutlass::epilogue::threadblock::Sm80EVT<Store, Second>;
  using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16,
      int8_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16,
      float, cutlass::layout::RowMajor, 4, int32_t, float,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      Tile, Warp, cutlass::gemm::GemmShape<16, 8, 32>, EVT,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swizzle>,
      Stages, cutlass::arch::OpMultiplyAddSaturate, 1>::GemmKernel;
  static_assert(sizeof(typename Kernel::SharedStorage) <= 101376,
                "SM86 shared-memory limit");
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  typename First::Arguments first{{}, {sa}, {}};
  typename Second::Arguments second{first, {static_cast<const BF16*>(sw)}, {}};
  typename EVT::Arguments epilogue{
      second, {static_cast<BF16*>(out), {int64_t(n), _1{}, _0{}}}};
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k}, 1,
      epilogue, a, w, nullptr, nullptr, 0, 0, 0, 0, k, k, n, n};
  Gemm gemm;
  if (gemm.get_workspace_size(args) != 0)
    throw std::runtime_error("Fused channel GEMM unexpectedly requires workspace");
  const auto status = gemm(args, nullptr, stream);
  if (status != cutlass::Status::kSuccess)
    throw std::runtime_error(cutlassGetStatusString(status));
}
} // namespace

void fused_channel_mm(const int8_t* a, const int8_t* w, const float* sa,
                       const void* sw, void* out, int m, int n, int k,
                       int schedule, cudaStream_t stream) {
  // Eight warps with a wider N tile improve large prefills on SM86. Retain
  // the smaller tiles below 2048 rows; their extra CTAs reduce tail overhead.
  if (schedule == 0) {
    if (m == 1024) {
      if (k == 5120) {
        if (n == 16384 || n == 34816) schedule = 3;
        else if (n == 14336) schedule = 2;
        else if (n == 8192 || n == 7168) schedule = 1;
      } else if (n == 5120 && (k == 3072 || k == 6144 || k == 8704)) {
        schedule = 1;
      }
    } else if (m >= 2048) {
      schedule = 4;
    }
  }
  if (schedule == 1) return fused_run<128, 128, 64, 64, 64, 3>(a, w, sa, sw, out, m, n, k, stream);
  if (schedule == 2) return fused_run<128, 256, 64, 64, 64, 3>(a, w, sa, sw, out, m, n, k, stream);
  if (schedule == 3) return fused_run<256, 128, 64, 64, 64, 3>(a, w, sa, sw, out, m, n, k, stream);
  if (schedule == 4) return fused_run<128, 256, 64, 64, 64, 4>(a, w, sa, sw, out, m, n, k, stream);
  if (schedule == 6) return fused_run<128, 256, 64, 64, 64, 4, 8>(a, w, sa, sw, out, m, n, k, stream);
  if (m <= 16) fused_run<16, 64, 128, 16, 64, 5>(a, w, sa, sw, out, m, n, k, stream);
  else if (m <= 32) fused_run<32, 64, 128, 32, 64, 4>(a, w, sa, sw, out, m, n, k, stream);
  else if (m <= 128) fused_run<64, 128, 64, 32, 64, 4>(a, w, sa, sw, out, m, n, k, stream);
  else fused_run<128, 128, 64, 64, 64, 5>(a, w, sa, sw, out, m, n, k, stream);
}
