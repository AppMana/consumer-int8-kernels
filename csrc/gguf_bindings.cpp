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
void gguf_mxfp4_routed_mm(const void*,const uint8_t*,const uint8_t*,const int32_t*,
    const int32_t*,void*,int,int,int,int,int,int,int,cudaStream_t);
void gguf_mxfp4_routed_weighted_mm(const void*,const uint8_t*,const uint8_t*,const int32_t*,
    const int32_t*,const float*,void*,int,int,int,int,int,int,int,cudaStream_t);
void gguf_swiglu(const void*,void*,int64_t,int,float,cudaStream_t);
void gguf_route_reduce(const void*,const int32_t*,void*,int64_t,int,int,int,cudaStream_t);
void gguf_plan_routes(const int32_t*,const float*,const int32_t*,int32_t*,int32_t*,float*,
    int32_t*,int32_t*,int,int,int,int,int,int,int,cudaStream_t);
void gguf_iq_xxs_routed_mm(const void*,const uint8_t*,const void*,const uint8_t*,const int32_t*,
    const int32_t*,const float*,void*,int,int,int,int,int,int,int,int,bool,cudaStream_t);
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
void boxed_iq_xxs_routed(StableIValue* stack, uint64_t, uint64_t) {
  int64_t tile_rows=to<int64_t>(stack[9]);
  int64_t bits=to<int64_t>(stack[8]);
  bool weighted=to<bool>(stack[10]);
  STD_TORCH_CHECK(bits==2 || bits==3, "XXS bits must be2 or3");
  STD_TORCH_CHECK(tile_rows==16 || tile_rows==32, "Routed IQ_XXS tile must be 16 or 32");
  auto x=to<Tensor>(stack[0]), q=to<Tensor>(stack[1]), d=to<Tensor>(stack[2]);
  auto s=to<Tensor>(stack[3]), tokens=to<Tensor>(stack[4]);
  auto tasks=to<Tensor>(stack[5]), weights=to<Tensor>(stack[6]), out=to<Tensor>(stack[7]);
  STD_TORCH_CHECK(x.is_cuda(), "Routed IQ_XXS requires CUDA");
  for (const auto& tensor : {q,d,s,tokens,tasks,weights,out}) {
    STD_TORCH_CHECK(tensor.is_cuda() && tensor.get_device_index()==x.get_device_index() &&
        tensor.is_contiguous(), "Routed IQ_XXS tensors must be contiguous on one CUDA device");
  }
  STD_TORCH_CHECK(x.is_contiguous() && x.scalar_type()==ScalarType::BFloat16 &&
      q.scalar_type()==ScalarType::Byte && d.scalar_type()==ScalarType::Half &&
      s.scalar_type()==ScalarType::Int && weights.scalar_type()==ScalarType::Float && tokens.scalar_type()==ScalarType::Int &&
      tasks.scalar_type()==ScalarType::Int && out.scalar_type()==ScalarType::BFloat16,
      "Invalid routed IQ_XXS dtype or activation layout");
  STD_TORCH_CHECK(x.dim()==2 && q.dim()==3 && d.dim()==3 && s.dim()==3 &&
      tokens.dim()==1 && weights.dim()==1 && weights.numel()==tokens.numel() && tasks.dim()==2 && out.dim()==2, "Invalid routed IQ_XXS dimensions");
  STD_TORCH_CHECK(x.size(1)>0 && x.size(1)%256==0 && q.size(2)==x.size(1)/(bits==2 ? 8 : 4) &&
      d.size(0)==q.size(0) && d.size(1)==q.size(1) &&
      d.size(2)==((x.size(1)/256+1)&~int64_t(1)) && s.size(0)==q.size(0) &&
      s.size(1)==q.size(1) && s.size(2)==x.size(1)/32 && tasks.size(1)==2 &&
      out.size(0)==tokens.size(0) && out.size(1)==q.size(1), "Invalid routed IQ_XXS shapes");
  STD_TORCH_CHECK(tasks.size(0)<=65535 && x.size(0)<=2147483647 &&
      tokens.size(0)<=2147483632 && x.size(1)<=2147483392 && q.size(0)<=2147483647 &&
      q.size(1)<=2147483584, "Routed IQ_XXS launch too large");
  STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr())%16==0 &&
      reinterpret_cast<uintptr_t>(q.data_ptr())%16==0 &&
      reinterpret_cast<uintptr_t>(d.data_ptr())%4==0 &&
      reinterpret_cast<uintptr_t>(s.data_ptr())%4==0, "Routed IQ_XXS alignment violation");
  if (!out.numel() || !tasks.numel()) return;
  for (const auto& tensor : {x,q,d,s,tokens,tasks,weights})
    STD_TORCH_CHECK(tensor.data_ptr()!=out.data_ptr(), "Routed output must not alias inputs");
  int device=x.get_device_index();
  torch::stable::accelerator::DeviceGuard guard(device);
  void* stream=nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device,&stream));
  gguf_iq_xxs_routed_mm(x.data_ptr(),static_cast<const uint8_t*>(q.data_ptr()),d.data_ptr(),
      static_cast<const uint8_t*>(s.data_ptr()),static_cast<const int32_t*>(tokens.data_ptr()),
      static_cast<const int32_t*>(tasks.data_ptr()),static_cast<const float*>(weights.data_ptr()),out.data_ptr(),tokens.size(0),q.size(1),
      x.size(1),q.size(0),tasks.size(0),x.size(0),bits,tile_rows,weighted,reinterpret_cast<cudaStream_t>(stream));
  auto error=cudaGetLastError();
  STD_TORCH_CHECK(error==cudaSuccess,cudaGetErrorString(error));
}
template<bool Weighted>
void boxed_mxfp4_routed(StableIValue* stack, uint64_t, uint64_t) {
  int64_t tile_rows=to<int64_t>(stack[Weighted ? 7 : 6]);
  STD_TORCH_CHECK(tile_rows==16 || tile_rows==32, "Routed MXFP4 tile must be 16 or 32");
  auto x=to<Tensor>(stack[0]), q=to<Tensor>(stack[1]), s=to<Tensor>(stack[2]);
  auto tokens=to<Tensor>(stack[3]), tasks=to<Tensor>(stack[4]), out=to<Tensor>(stack[Weighted ? 6 : 5]);
  STD_TORCH_CHECK(x.is_cuda(), "Routed MXFP4 requires CUDA");
  for (const auto& tensor : {q,s,tokens,tasks,out}) {
    STD_TORCH_CHECK(tensor.is_cuda() && tensor.get_device_index()==x.get_device_index() &&
        tensor.is_contiguous(), "Routed MXFP4 tensors must be contiguous on one CUDA device");
  }
  STD_TORCH_CHECK(x.is_contiguous() && x.scalar_type()==ScalarType::BFloat16 &&
      q.scalar_type()==ScalarType::Byte &&
      s.scalar_type()==ScalarType::Byte && tokens.scalar_type()==ScalarType::Int &&
      tasks.scalar_type()==ScalarType::Int && out.scalar_type()==ScalarType::BFloat16,
      "Invalid routed MXFP4 dtype or activation layout");
  STD_TORCH_CHECK(x.dim()==2 && q.dim()==3 && s.dim()==3 &&
      tokens.dim()==1 && tasks.dim()==2 && out.dim()==2, "Invalid routed MXFP4 dimensions");
  STD_TORCH_CHECK(x.size(1)>0 && x.size(1)%128==0 && q.size(2)==x.size(1)/2 && s.size(0)==q.size(0) &&
      s.size(1)==q.size(1) && s.size(2)==x.size(1)/32 && tasks.size(1)==2 &&
      out.size(0)==tokens.size(0) && out.size(1)==q.size(1), "Invalid routed MXFP4 shapes");
  STD_TORCH_CHECK(tasks.size(0)<=65535 && x.size(0)<=2147483647 &&
      tokens.size(0)<=2147483632 && x.size(1)<=2147483392 && q.size(0)<=2147483647 &&
      q.size(1)<=2147483584, "Routed MXFP4 launch too large");
  STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr())%16==0 &&
      reinterpret_cast<uintptr_t>(q.data_ptr())%16==0 &&
      reinterpret_cast<uintptr_t>(s.data_ptr())%4==0, "Routed MXFP4 alignment violation");
  if (!out.numel() || !tasks.numel()) return;
  for (const auto& tensor : {x,q,s,tokens,tasks})
    STD_TORCH_CHECK(tensor.data_ptr()!=out.data_ptr(), "Routed output must not alias inputs");
  int device=x.get_device_index();
  torch::stable::accelerator::DeviceGuard guard(device);
  void* stream=nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device,&stream));
  if constexpr (Weighted) {
    auto weights=to<Tensor>(stack[5]);
    STD_TORCH_CHECK(weights.is_cuda() && weights.get_device_index()==device &&
        weights.scalar_type()==ScalarType::Float && weights.dim()==1 &&
        weights.numel()==tokens.numel() && weights.is_contiguous() &&
        weights.data_ptr()!=out.data_ptr(), "Invalid routed MXFP4 weights");
    gguf_mxfp4_routed_weighted_mm(x.data_ptr(),static_cast<const uint8_t*>(q.data_ptr()),
        static_cast<const uint8_t*>(s.data_ptr()),static_cast<const int32_t*>(tokens.data_ptr()),
        static_cast<const int32_t*>(tasks.data_ptr()),static_cast<const float*>(weights.data_ptr()),
        out.data_ptr(),tokens.size(0),q.size(1),x.size(1),q.size(0),tasks.size(0),
        x.size(0),tile_rows,reinterpret_cast<cudaStream_t>(stream));
  } else {
  gguf_mxfp4_routed_mm(x.data_ptr(),static_cast<const uint8_t*>(q.data_ptr()),
      static_cast<const uint8_t*>(s.data_ptr()),static_cast<const int32_t*>(tokens.data_ptr()),
      static_cast<const int32_t*>(tasks.data_ptr()),out.data_ptr(),tokens.size(0),q.size(1),
      x.size(1),q.size(0),tasks.size(0),x.size(0),tile_rows,reinterpret_cast<cudaStream_t>(stream));
  }
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

void boxed_swiglu(StableIValue* stack,uint64_t,uint64_t) {
  auto x=to<Tensor>(stack[0]), out=to<Tensor>(stack[1]);
  double limit=to<double>(stack[2]);
  STD_TORCH_CHECK(x.is_cuda() && out.is_cuda() && x.get_device_index()==out.get_device_index() &&
      x.scalar_type()==ScalarType::BFloat16 && out.scalar_type()==ScalarType::BFloat16 &&
      x.is_contiguous() && out.is_contiguous() && x.dim()==2 && out.dim()==2,
      "SwiGLU requires contiguous BF16 matrices on one CUDA device");
  STD_TORCH_CHECK(x.size(0)==out.size(0) && x.size(1)==2*out.size(1) &&
      out.size(1)>0 && out.size(1)<=2147483647 && limit>0 && limit<=1e30,
      "Invalid SwiGLU shape or limit");
  if (!out.numel()) return;
  STD_TORCH_CHECK(x.data_ptr()!=out.data_ptr(), "SwiGLU output must not alias input");
  int device=x.get_device_index();
  torch::stable::accelerator::DeviceGuard guard(device);
  void* stream=nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device,&stream));
  gguf_swiglu(x.data_ptr(),out.data_ptr(),out.size(0),out.size(1),limit,reinterpret_cast<cudaStream_t>(stream));
  auto error=cudaGetLastError();
  STD_TORCH_CHECK(error==cudaSuccess,cudaGetErrorString(error));
}
void boxed_route_reduce(StableIValue* stack,uint64_t,uint64_t) {
  auto x=to<Tensor>(stack[0]), indices=to<Tensor>(stack[1]), out=to<Tensor>(stack[2]);
  STD_TORCH_CHECK(x.is_cuda() && indices.is_cuda() && out.is_cuda() &&
      x.get_device_index()==indices.get_device_index() && x.get_device_index()==out.get_device_index() &&
      x.scalar_type()==ScalarType::BFloat16 && indices.scalar_type()==ScalarType::Int &&
      out.scalar_type()==ScalarType::BFloat16 && x.is_contiguous() && indices.is_contiguous() &&
      out.is_contiguous() && x.dim()==2 && indices.dim()==2 && out.dim()==2,
      "Invalid route reduction tensors");
  STD_TORCH_CHECK(indices.size(0)==out.size(0) && x.size(1)==out.size(1) &&
      x.size(0)<=2147483647 && x.size(1)>0 && x.size(1)<=2147483647 &&
      indices.size(1)>0 && indices.size(1)<=2147483647, "Invalid route reduction shape");
  if (!out.numel()) return;
  STD_TORCH_CHECK(x.data_ptr()!=out.data_ptr() && indices.data_ptr()!=out.data_ptr(),
      "Route reduction output must not alias inputs");
  int device=x.get_device_index();
  torch::stable::accelerator::DeviceGuard guard(device);
  void* stream=nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device,&stream));
  gguf_route_reduce(x.data_ptr(),static_cast<const int32_t*>(indices.data_ptr()),out.data_ptr(),
      out.size(0),out.size(1),indices.size(1),x.size(0),reinterpret_cast<cudaStream_t>(stream));
  auto error=cudaGetLastError();
  STD_TORCH_CHECK(error==cudaSuccess,cudaGetErrorString(error));
}

void boxed_plan_routes(StableIValue* stack,uint64_t,uint64_t) {
  auto ids=to<Tensor>(stack[0]), top_weights=to<Tensor>(stack[1]), map=to<Tensor>(stack[2]);
  auto info=to<Tensor>(stack[3]), tokens=to<Tensor>(stack[4]), weights=to<Tensor>(stack[5]);
  auto inverse=to<Tensor>(stack[6]), tasks=to<Tensor>(stack[7]);
  int64_t tile=to<int64_t>(stack[8]);
  STD_TORCH_CHECK(ids.is_cuda() && ids.is_contiguous() && ids.scalar_type()==ScalarType::Int &&
      ids.dim()==2 && ids.size(1)>0 && ids.size(1)<=2147483647 && ids.numel()<=2147480000,
      "Routing ids must be contiguous CUDA int32 [T,topk]");
  int device=ids.get_device_index();
  for (const auto& tensor : {top_weights,map,info,tokens,weights,inverse,tasks})
    STD_TORCH_CHECK(tensor.is_cuda() && tensor.get_device_index()==device && tensor.is_contiguous(),
        "Routing tensors must be contiguous on one CUDA device");
  STD_TORCH_CHECK(top_weights.scalar_type()==ScalarType::Float && weights.scalar_type()==ScalarType::Float &&
      map.scalar_type()==ScalarType::Int && info.scalar_type()==ScalarType::Int &&
      tokens.scalar_type()==ScalarType::Int && inverse.scalar_type()==ScalarType::Int &&
      tasks.scalar_type()==ScalarType::Int, "Invalid routing dtype");
  STD_TORCH_CHECK(top_weights.dim()==2 && inverse.dim()==2 && map.dim()==1 && info.dim()==2 &&
      tokens.dim()==1 && weights.dim()==1 && tasks.dim()==2, "Invalid routing dimensions");
  STD_TORCH_CHECK(top_weights.size(0)==ids.size(0) && top_weights.size(1)==ids.size(1) &&
      inverse.size(0)==ids.size(0) && inverse.size(1)==ids.size(1) && info.size(1)==2 &&
      info.size(0)>0 && info.size(0)<=65535 && map.numel()<=2147483647 &&
      tasks.size(1)==2 && tasks.size(0)<=65535 && (tile==16 || tile==32),
      "Invalid routing shape, expert count, or tile");
  int64_t required=ids.numel()+info.size(0)*(tile-1);
  STD_TORCH_CHECK(tokens.numel()>=required && tokens.numel()<=2147480000 &&
      weights.numel()==tokens.numel() && tasks.size(0)>=(required+tile-1)/tile,
      "Routing capacity must cover T*topk + experts*(tile-1) rows and ceil(rows/tile) tasks");
  for (const auto& output : {info,tokens,weights,inverse,tasks}) {
    for (const auto& input : {ids,top_weights,map})
      STD_TORCH_CHECK(!output.numel() || !input.numel() || output.data_ptr()!=input.data_ptr(),
          "Routing outputs must not alias inputs");
  }
  // All output buffers must also have disjoint storage ranges (caller contract).
  torch::stable::accelerator::DeviceGuard guard(device);
  void* stream=nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device,&stream));
  gguf_plan_routes(static_cast<const int32_t*>(ids.data_ptr()),static_cast<const float*>(top_weights.data_ptr()),
      static_cast<const int32_t*>(map.data_ptr()),static_cast<int32_t*>(info.data_ptr()),
      static_cast<int32_t*>(tokens.data_ptr()),static_cast<float*>(weights.data_ptr()),
      static_cast<int32_t*>(inverse.data_ptr()),static_cast<int32_t*>(tasks.data_ptr()),
      ids.numel(),ids.size(1),map.numel(),info.size(0),tokens.numel(),tasks.size(0),tile,
      reinterpret_cast<cudaStream_t>(stream));
  auto error=cudaGetLastError();
  STD_TORCH_CHECK(error==cudaSuccess,cudaGetErrorString(error));
}

}
STABLE_TORCH_LIBRARY_FRAGMENT(consumer_int8, m) {
  m.def("gguf_iq_xxs_routed_mm(Tensor x, Tensor codes, Tensor bases, Tensor metadata, Tensor tokens, Tensor tasks, Tensor weights, Tensor(a!) out, int bits, int tile_rows=16, bool weighted=False) -> ()");
  m.def("gguf_plan_routes(Tensor ids, Tensor top_weights, Tensor expert_map, Tensor(a!) info, Tensor(b!) tokens, Tensor(c!) weights, Tensor(d!) inverse, Tensor(e!) tasks, int tile_rows=16) -> ()");
  m.def("gguf_mxfp4_routed_weighted_mm(Tensor x, Tensor codes, Tensor scales, Tensor tokens, Tensor tasks, Tensor weights, Tensor(a!) out, int tile_rows=16) -> ()");
  m.def("gguf_swiglu(Tensor x, Tensor(a!) out, float limit=10.) -> ()");
  m.def("gguf_route_reduce(Tensor contributions, Tensor rows, Tensor(a!) out) -> ()");
  m.def("gguf_mxfp4_routed_mm(Tensor x, Tensor codes, Tensor scales, Tensor tokens, Tensor tasks, Tensor(a!) out, int tile_rows=16) -> ()");
  m.def("gguf_q8_expand(Tensor codes, Tensor scales, Tensor(a!) out) -> ()");
  m.def("gguf_iq2_xs_routed_mm(Tensor x, Tensor codes, Tensor bases, Tensor scales, Tensor tokens, Tensor tasks, Tensor(a!) out, int tile_rows=16) -> ()");
  m.def("gguf_iq2_xs_mm(Tensor x, Tensor codes, Tensor bases, Tensor scales, Tensor(a!) out) -> ()");
  m.def("gguf_q8_packed_mm(Tensor x, Tensor codes, Tensor scales, Tensor(a!) out) -> ()");
  m.def("gguf_q8_mm(Tensor x, Tensor weight, Tensor(a!) out) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(consumer_int8, CUDA, m) {
  m.impl("gguf_iq_xxs_routed_mm", &boxed_iq_xxs_routed);
  m.impl("gguf_plan_routes", &boxed_plan_routes);
  m.impl("gguf_mxfp4_routed_weighted_mm", &boxed_mxfp4_routed<true>);
  m.impl("gguf_swiglu", &boxed_swiglu);
  m.impl("gguf_route_reduce", &boxed_route_reduce);
  m.impl("gguf_mxfp4_routed_mm", &boxed_mxfp4_routed<false>);
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
