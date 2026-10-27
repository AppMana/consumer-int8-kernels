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
void gguf_iq2_xs_mm(const void*, const int8_t*, const void*, const uint8_t*, void*,
    int, int, int, int, int64_t, int64_t, cudaStream_t);
void gguf_iq2_xs_routed_mm(const void*, const int8_t*, const void*, const uint8_t*,
    const int32_t*, const int32_t*, void*, int, int, int, int, int, int, int, cudaStream_t);
void gguf_q8_expand(const int8_t*,const void*,void*,int64_t,cudaStream_t);
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

void boxed_iq2_xs(StableIValue* stack, uint64_t, uint64_t) {
  auto x=to<Tensor>(stack[0]), q=to<Tensor>(stack[1]), d=to<Tensor>(stack[2]);
  auto s=to<Tensor>(stack[3]), out=to<Tensor>(stack[4]);
  STD_TORCH_CHECK(x.is_cuda() && q.is_cuda() && d.is_cuda() && s.is_cuda() && out.is_cuda() &&
      x.get_device_index()==q.get_device_index() && x.get_device_index()==d.get_device_index() &&
      x.get_device_index()==s.get_device_index() && x.get_device_index()==out.get_device_index(),
      "IQ2_XS tensors must share a CUDA device");
  STD_TORCH_CHECK(x.scalar_type()==ScalarType::BFloat16 && q.scalar_type()==ScalarType::Short &&
      d.scalar_type()==ScalarType::Half && s.scalar_type()==ScalarType::Byte &&
      out.scalar_type()==ScalarType::BFloat16, "Invalid IQ2_XS dtype");
  STD_TORCH_CHECK(x.dim()==3 && q.dim()==3 && d.dim()==3 && s.dim()==3 && out.dim()==3,
      "Expected x[M,G,K], codes[G,N,K/8], bases[G,N,even_ceil(K/256)], scales[G,N,K/32]");
  STD_TORCH_CHECK(x.size(2)>0 && x.size(2)%256==0 && x.stride(2)==1 &&
      x.stride(0)>0 && x.stride(1)>0 && q.is_contiguous() && d.is_contiguous() &&
      s.is_contiguous() && out.is_contiguous() && q.size(0)==x.size(1) &&
      q.size(2)==x.size(2)/8 && d.size(0)==q.size(0) && d.size(1)==q.size(1) &&
      d.size(2)==((x.size(2)/256+1)&~int64_t(1)) && s.size(0)==q.size(0) &&
      s.size(1)==q.size(1) && s.size(2)==x.size(2)/32 && out.size(0)==x.size(0) &&
      out.size(1)==x.size(1) && out.size(2)==q.size(1), "Invalid IQ2_XS shape or strides");
  STD_TORCH_CHECK(x.size(0)<=65535*32 && x.size(1)<=65535 && x.size(2)<=2147483392 &&
      q.size(1)<=2147483584, "IQ2_XS grid too large");
  STD_TORCH_CHECK(x.stride(0)%8==0 && x.stride(1)%8==0 &&
      reinterpret_cast<uintptr_t>(x.data_ptr())%16==0 &&
      reinterpret_cast<uintptr_t>(q.data_ptr())%16==0 &&
      reinterpret_cast<uintptr_t>(d.data_ptr())%4==0 &&
      reinterpret_cast<uintptr_t>(s.data_ptr())%4==0, "IQ2_XS copy alignment violation");
  if (!out.numel()) return;
  STD_TORCH_CHECK(out.data_ptr()!=x.data_ptr() && out.data_ptr()!=q.data_ptr() &&
      out.data_ptr()!=d.data_ptr() && out.data_ptr()!=s.data_ptr(), "IQ2_XS output must not alias inputs");
  int device=x.get_device_index();
  torch::stable::accelerator::DeviceGuard guard(device);
  void* stream=nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device,&stream));
  gguf_iq2_xs_mm(x.data_ptr(),static_cast<const int8_t*>(q.data_ptr()),d.data_ptr(),
      static_cast<const uint8_t*>(s.data_ptr()),out.data_ptr(),x.size(0),q.size(1),x.size(2),
      x.size(1),x.stride(0),x.stride(1),reinterpret_cast<cudaStream_t>(stream));
  auto error=cudaGetLastError();
  STD_TORCH_CHECK(error==cudaSuccess,cudaGetErrorString(error));
}

void boxed_iq2_xs_routed(StableIValue* stack, uint64_t, uint64_t) {
  int64_t tile_rows=to<int64_t>(stack[7]);
  STD_TORCH_CHECK(tile_rows==16 || tile_rows==32, "Routed IQ2_XS tile must be 16 or 32");
  auto x=to<Tensor>(stack[0]), q=to<Tensor>(stack[1]), d=to<Tensor>(stack[2]);
  auto s=to<Tensor>(stack[3]), tokens=to<Tensor>(stack[4]);
  auto tasks=to<Tensor>(stack[5]), out=to<Tensor>(stack[6]);
  STD_TORCH_CHECK(x.is_cuda(), "Routed IQ2_XS requires CUDA");
  for (const auto& tensor : {q,d,s,tokens,tasks,out}) {
    STD_TORCH_CHECK(tensor.is_cuda() && tensor.get_device_index()==x.get_device_index() &&
        tensor.is_contiguous(), "Routed IQ2_XS tensors must be contiguous on one CUDA device");
  }
  STD_TORCH_CHECK(x.is_contiguous() && x.scalar_type()==ScalarType::BFloat16 &&
      q.scalar_type()==ScalarType::Short && d.scalar_type()==ScalarType::Half &&
      s.scalar_type()==ScalarType::Byte && tokens.scalar_type()==ScalarType::Int &&
      tasks.scalar_type()==ScalarType::Int && out.scalar_type()==ScalarType::BFloat16,
      "Invalid routed IQ2_XS dtype or activation layout");
  STD_TORCH_CHECK(x.dim()==2 && q.dim()==3 && d.dim()==3 && s.dim()==3 &&
      tokens.dim()==1 && tasks.dim()==2 && out.dim()==2, "Invalid routed IQ2_XS dimensions");
  STD_TORCH_CHECK(x.size(1)>0 && x.size(1)%256==0 && q.size(2)==x.size(1)/8 &&
      d.size(0)==q.size(0) && d.size(1)==q.size(1) &&
      d.size(2)==((x.size(1)/256+1)&~int64_t(1)) && s.size(0)==q.size(0) &&
      s.size(1)==q.size(1) && s.size(2)==x.size(1)/32 && tasks.size(1)==2 &&
      out.size(0)==tokens.size(0) && out.size(1)==q.size(1), "Invalid routed IQ2_XS shapes");
  STD_TORCH_CHECK(tasks.size(0)<=65535 && x.size(0)<=2147483647 &&
      tokens.size(0)<=2147483632 && x.size(1)<=2147483392 && q.size(0)<=2147483647 &&
      q.size(1)<=2147483584, "Routed IQ2_XS launch too large");
  STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr())%16==0 &&
      reinterpret_cast<uintptr_t>(q.data_ptr())%16==0 &&
      reinterpret_cast<uintptr_t>(d.data_ptr())%4==0 &&
      reinterpret_cast<uintptr_t>(s.data_ptr())%4==0, "Routed IQ2_XS alignment violation");
  if (!out.numel() || !tasks.numel()) return;
  for (const auto& tensor : {x,q,d,s,tokens,tasks})
    STD_TORCH_CHECK(tensor.data_ptr()!=out.data_ptr(), "Routed output must not alias inputs");
  int device=x.get_device_index();
  torch::stable::accelerator::DeviceGuard guard(device);
  void* stream=nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device,&stream));
  gguf_iq2_xs_routed_mm(x.data_ptr(),static_cast<const int8_t*>(q.data_ptr()),d.data_ptr(),
      static_cast<const uint8_t*>(s.data_ptr()),static_cast<const int32_t*>(tokens.data_ptr()),
      static_cast<const int32_t*>(tasks.data_ptr()),out.data_ptr(),tokens.size(0),q.size(1),
      x.size(1),q.size(0),tasks.size(0),x.size(0),tile_rows,reinterpret_cast<cudaStream_t>(stream));
  auto error=cudaGetLastError();
  STD_TORCH_CHECK(error==cudaSuccess,cudaGetErrorString(error));
}

void boxed_q8_expand(StableIValue* stack,uint64_t,uint64_t) {
  auto q=to<Tensor>(stack[0]), s=to<Tensor>(stack[1]), out=to<Tensor>(stack[2]);
  STD_TORCH_CHECK(q.is_cuda() && s.is_cuda() && out.is_cuda() &&
      q.get_device_index()==s.get_device_index() && q.get_device_index()==out.get_device_index(),
      "Q8 expansion requires one CUDA device");
  STD_TORCH_CHECK(q.scalar_type()==ScalarType::Char && s.scalar_type()==ScalarType::Half &&
      out.scalar_type()==ScalarType::BFloat16 && q.is_contiguous() &&
      s.is_contiguous() && out.is_contiguous(), "Invalid Q8 expansion dtype/layout");
  STD_TORCH_CHECK(q.dim()==3 && s.dim()==3 && out.dim()==3,
      "Q8 expansion expects codes/out[G,N,K], scales[G,N,K/32]");
  STD_TORCH_CHECK(q.size(2)>0 && q.size(2)%32==0 && s.size(0)==q.size(0) &&
      s.size(1)==q.size(1) && s.size(2)==q.size(2)/32 && out.size(0)==q.size(0) &&
      out.size(1)==q.size(1) && out.size(2)==q.size(2), "Invalid Q8 expansion shape");
  if (!out.numel()) return;
  STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr())%2==0 &&
      reinterpret_cast<uintptr_t>(out.data_ptr())%4==0 &&
      out.data_ptr()!=q.data_ptr() && out.data_ptr()!=s.data_ptr(),
      "Q8 expansion alignment/alias violation");
  int device=q.get_device_index();
  torch::stable::accelerator::DeviceGuard guard(device);
  void* stream=nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device,&stream));
  gguf_q8_expand(static_cast<const int8_t*>(q.data_ptr()),s.data_ptr(),out.data_ptr(),
      q.numel(),reinterpret_cast<cudaStream_t>(stream));
  auto error=cudaGetLastError();
  STD_TORCH_CHECK(error==cudaSuccess,cudaGetErrorString(error));
}

}
STABLE_TORCH_LIBRARY_FRAGMENT(consumer_int8, m) {
  m.def("gguf_q8_expand(Tensor codes, Tensor scales, Tensor(a!) out) -> ()");
  m.def("gguf_iq2_xs_routed_mm(Tensor x, Tensor codes, Tensor bases, Tensor scales, Tensor tokens, Tensor tasks, Tensor(a!) out, int tile_rows=16) -> ()");
  m.def("gguf_iq2_xs_mm(Tensor x, Tensor codes, Tensor bases, Tensor scales, Tensor(a!) out) -> ()");
  m.def("gguf_q8_packed_mm(Tensor x, Tensor codes, Tensor scales, Tensor(a!) out) -> ()");
  m.def("gguf_q8_mm(Tensor x, Tensor weight, Tensor(a!) out) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(consumer_int8, CUDA, m) {
  m.impl("gguf_q8_expand", &boxed_q8_expand);
  m.impl("gguf_iq2_xs_routed_mm", &boxed_iq2_xs_routed);
  m.impl("gguf_iq2_xs_mm", &boxed_iq2_xs);
  m.impl("gguf_q8_packed_mm", &boxed_q8_packed);
  m.impl("gguf_q8_mm", &boxed_q8);
}
extern "C" PyObject* PyInit__gguf() {
  static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_gguf", nullptr, -1, nullptr};
  return PyModule_Create(&module);
}
