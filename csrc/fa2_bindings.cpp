// SPDX-License-Identifier: Apache-2.0
#include "kernels.h"
#include <Python.h>
#include <limits>
#include <optional>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

namespace {
void check(const Tensor &t, ScalarType dtype, int device) {
  STD_TORCH_CHECK(t.is_cuda() && t.get_device_index() == device,
                  "All tensors must be on the same CUDA device");
  STD_TORCH_CHECK(t.is_contiguous() && t.scalar_type() == dtype,
                  "Unexpected dtype or noncontiguous tensor");
}
cudaStream_t stream_for(int device) {
  void *stream = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device, &stream));
  return reinterpret_cast<cudaStream_t>(stream);
}
void launch_check() {
  const auto error = cudaGetLastError();
  STD_TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
}
void boxed_fa2_prefill(StableIValue *stack, uint64_t, uint64_t) {
  auto q = to<Tensor>(stack[0]), k = to<Tensor>(stack[1]),
       v = to<Tensor>(stack[2]);
  auto offsets = to<Tensor>(stack[3]), lengths = to<Tensor>(stack[4]);
  auto out = to<Tensor>(stack[5]), lse = to<Tensor>(stack[6]);
  const int64_t max_query = to<int64_t>(stack[7]);
  const double scale = to<double>(stack[8]);
  const int64_t schedule = to<int64_t>(stack[9]);
  const int device = q.get_device_index();
  for (const auto &t : {q, out}) {
    STD_TORCH_CHECK(t.is_cuda() && t.get_device_index() == device &&
                        t.scalar_type() == ScalarType::BFloat16 &&
                        t.dim() == 3 && t.size(2) == 256 && t.stride(2) == 1 &&
                        t.stride(1) >= 256 &&
                        t.stride(0) >= t.size(1) * t.stride(1) &&
                        t.stride(0) % 8 == 0 && t.stride(1) % 8 == 0 &&
                        reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                    "Expected aligned BF16 [T,H,256] query/output");
  }
  for (const auto &t : {k, v}) {
    check(t, ScalarType::BFloat16, device);
    STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                    "FA2 cache requires 16-byte alignment");
  }
  for (const auto &t : {offsets, lengths})
    check(t, ScalarType::Int, device);
  check(lse, ScalarType::Float, device);
  STD_TORCH_CHECK(q.sizes() == out.sizes() && q.size(0) > 0 &&
                      (q.size(1) == 12 || q.size(1) == 24) && k.dim() == 4 &&
                      k.sizes() == v.sizes() && k.size(0) > 0 &&
                      k.size(0) <= 65535 && k.size(1) > 0 &&
                      k.size(2) * 6 == q.size(1) && k.size(3) == 256,
                  "Expected Qwen GQA6 with BF16 [B,capacity,Hkv,256] cache");
  STD_TORCH_CHECK(offsets.dim() == 1 && lengths.dim() == 1 &&
                      lengths.size(0) == k.size(0) &&
                      offsets.size(0) == k.size(0) + 1 && lse.dim() == 2 &&
                      lse.size(0) == q.size(1) && lse.size(1) == q.size(0) &&
                      max_query > 0 && max_query <= q.size(0) &&
                      schedule >= 0 && schedule <= 8 &&
                      q.size(0) <= std::numeric_limits<int>::max() - 127 &&
                      k.size(1) <= std::numeric_limits<int>::max() - 127,
                  "Invalid sequence metadata, LSE output or FA2 schedule");
  torch::stable::accelerator::DeviceGuard guard(device);
  fa2_prefill(q.data_ptr(), k.data_ptr(), v.data_ptr(),
              static_cast<int32_t *>(offsets.data_ptr()),
              static_cast<int32_t *>(lengths.data_ptr()), out.data_ptr(),
              static_cast<float *>(lse.data_ptr()), q.size(0), q.size(1),
              k.size(2), k.size(0), k.size(1), max_query, q.stride(0),
              q.stride(1), out.stride(0), out.stride(1), scale, schedule,
              stream_for(device));
  launch_check();
}
} // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(consumer_int8, m) {
  m.def("fa2_prefill(Tensor query, Tensor key, Tensor value, Tensor offsets, "
        "Tensor lengths, Tensor(a!) out, Tensor(b!) lse, int max_query, float "
        "scale, int schedule=0) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(consumer_int8, CUDA, m) {
  m.impl("fa2_prefill", &boxed_fa2_prefill);
}
extern "C" PyObject *PyInit__fa2() {
  static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_fa2", nullptr, -1,
                               nullptr};
  return PyModule_Create(&module);
}
