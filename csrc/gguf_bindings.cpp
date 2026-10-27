// SPDX-License-Identifier: Apache-2.0
#include <Python.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/headeronly/util/Exception.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
using torch::stable::Tensor;
using torch::headeronly::ScalarType;
void gguf_q8_mm(const void*, const uint8_t*, void*, int, int, int, int,
                int64_t, int64_t, cudaStream_t);
void gguf_q8_packed_mm(const void*, const int8_t*, const void*, void*, int, int, int, int,
                       int64_t, int64_t, cudaStream_t);
namespace {
void boxed_q8(StableIValue* stack, uint64_t, uint64_t) {
  auto x = to<Tensor>(stack[0]), w = to<Tensor>(stack[1]), y = to<Tensor>(stack[2]);
  STD_TORCH_CHECK(x.is_cuda() && w.is_cuda() && y.is_cuda() &&
      x.get_device_index() == w.get_device_index() && x.get_device_index() == y.get_device_index(),
      "Q8 tensors must share a CUDA device");
  STD_TORCH_CHECK(x.scalar_type() == ScalarType::BFloat16 &&
      w.scalar_type() == ScalarType::Byte && y.scalar_type() == ScalarType::BFloat16,
      "Q8 requires BF16 activations/output and raw uint8 GGUF weights");
  STD_TORCH_CHECK(x.dim() == 3 && w.dim() == 4 && y.dim() == 3,
      "Expected x[M,G,K], raw w[G,N,K/32,34], y[M,G,N]");
  STD_TORCH_CHECK(x.stride(2) == 1 && x.stride(0) > 0 && x.stride(1) > 0 &&
      w.is_contiguous() && y.is_contiguous() && x.size(2) > 0 && x.size(2) % 32 == 0 &&
      w.size(0) == x.size(1) && w.size(2) == x.size(2) / 32 && w.size(3) == 34 &&
      y.size(0) == x.size(0) && y.size(1) == x.size(1) && y.size(2) == w.size(1),
      "Invalid Q8 shape or strides");
  STD_TORCH_CHECK(x.size(0) <= 65535 * 32 && x.size(1) <= 65535 &&
      x.size(2) <= std::numeric_limits<int>::max() && w.size(1) <= 2147483584,
      "Q8 launch dimensions exceed supported grid");
  STD_TORCH_CHECK(x.data_ptr() != y.data_ptr() && w.data_ptr() != y.data_ptr(),
      "Q8 output must not alias its inputs");
  const int device = x.get_device_index();
  torch::stable::accelerator::DeviceGuard guard(device);
  void* stream = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device, &stream));
  gguf_q8_mm(x.data_ptr(), static_cast<const uint8_t*>(w.data_ptr()), y.data_ptr(),
      x.size(0), w.size(1), x.size(2), x.size(1), x.stride(0), x.stride(1),
      reinterpret_cast<cudaStream_t>(stream));
  auto error = cudaGetLastError();
  STD_TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
}
void boxed_q8_packed(StableIValue* stack, uint64_t, uint64_t) {
  auto x=to<Tensor>(stack[0]), q=to<Tensor>(stack[1]);
  auto scales=to<Tensor>(stack[2]), out=to<Tensor>(stack[3]);
  STD_TORCH_CHECK(x.is_cuda() && q.is_cuda() && scales.is_cuda() && out.is_cuda() &&
      x.get_device_index()==q.get_device_index() && x.get_device_index()==scales.get_device_index() &&
      x.get_device_index()==out.get_device_index(), "Packed Q8 tensors must share a CUDA device");
  STD_TORCH_CHECK(x.scalar_type()==ScalarType::BFloat16 && q.scalar_type()==ScalarType::Char &&
      scales.scalar_type()==ScalarType::Half && out.scalar_type()==ScalarType::BFloat16,
      "Packed Q8 requires BF16 input/output, INT8 codes and original FP16 scales");
  STD_TORCH_CHECK(x.dim()==3 && q.dim()==3 && scales.dim()==3 && out.dim()==3,
      "Expected x[M,G,K], q[G,N,K], scales[G,N,K/32], out[M,G,N]");
  STD_TORCH_CHECK(q.is_contiguous() && scales.is_contiguous() && out.is_contiguous() &&
      x.stride(2)==1 && x.stride(0)>0 && x.stride(1)>0 &&
      x.size(2)>0 && x.size(2)%64==0 && q.size(0)==x.size(1) && q.size(2)==x.size(2) &&
      scales.size(0)==q.size(0) && scales.size(1)==q.size(1) && scales.size(2)==x.size(2)/32 &&
      out.size(0)==x.size(0) && out.size(1)==x.size(1) && out.size(2)==q.size(1),
      "Invalid packed Q8 shape/strides; K must be a multiple of 64");
  STD_TORCH_CHECK(x.size(0)<=65535*32 && x.size(1)<=65535 &&
      x.size(2)<=2147483584 && q.size(1)<=2147483584,"Packed Q8 grid too large");
  STD_TORCH_CHECK(x.stride(0)%8==0 && x.stride(1)%8==0 &&
      reinterpret_cast<uintptr_t>(x.data_ptr())%16==0 &&
      reinterpret_cast<uintptr_t>(q.data_ptr())%16==0 &&
      reinterpret_cast<uintptr_t>(scales.data_ptr())%4==0,"Packed Q8 copy alignment violation");
  if (!out.numel()) return;
  STD_TORCH_CHECK(out.data_ptr()!=x.data_ptr() && out.data_ptr()!=q.data_ptr() &&
      out.data_ptr()!=scales.data_ptr(),"Packed Q8 output must not alias inputs");
  int device=x.get_device_index();
  torch::stable::accelerator::DeviceGuard guard(device);
  void* stream=nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device,&stream));
  gguf_q8_packed_mm(x.data_ptr(),static_cast<const int8_t*>(q.data_ptr()),scales.data_ptr(),
      out.data_ptr(),x.size(0),q.size(1),x.size(2),x.size(1),x.stride(0),x.stride(1),
      reinterpret_cast<cudaStream_t>(stream));
  auto error=cudaGetLastError();
  STD_TORCH_CHECK(error==cudaSuccess,cudaGetErrorString(error));
}

}
STABLE_TORCH_LIBRARY_FRAGMENT(consumer_int8, m) {
  m.def("gguf_q8_packed_mm(Tensor x, Tensor codes, Tensor scales, Tensor(a!) out) -> ()");
  m.def("gguf_q8_mm(Tensor x, Tensor weight, Tensor(a!) out) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(consumer_int8, CUDA, m) {
  m.impl("gguf_q8_packed_mm", &boxed_q8_packed);
  m.impl("gguf_q8_mm", &boxed_q8);
}
extern "C" PyObject* PyInit__gguf() {
  static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_gguf", nullptr, -1, nullptr};
  return PyModule_Create(&module);
}
