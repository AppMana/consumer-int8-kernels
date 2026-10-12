// SPDX-License-Identifier: Apache-2.0
#include <Python.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/headeronly/util/Exception.h>
#include <limits>
#include "kernels.h"

using torch::stable::Tensor;
using torch::headeronly::ScalarType;

namespace {
void check(const Tensor& t, ScalarType dtype, int device) {
  STD_TORCH_CHECK(t.is_cuda() && t.get_device_index() == device,
                  "All tensors must be on the same CUDA device");
  STD_TORCH_CHECK(t.is_contiguous() && t.scalar_type() == dtype,
                  "Unexpected dtype or noncontiguous tensor");
}
cudaStream_t stream_for(int device) {
  void* stream = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device, &stream));
  return reinterpret_cast<cudaStream_t>(stream);
}
void launch_check() {
  const auto error = cudaGetLastError();
  STD_TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
}
void boxed_quant(StableIValue* stack, uint64_t, uint64_t) {
  auto x = to<Tensor>(stack[0]);
  auto codes = to<Tensor>(stack[1]);
  auto scale = to<Tensor>(stack[2]);
  const int device = x.get_device_index();
  check(x, ScalarType::BFloat16, device);
  check(codes, ScalarType::Char, device);
  check(scale, ScalarType::Float, device);
  STD_TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0,
                  "Expected a nonempty BF16 matrix");
  STD_TORCH_CHECK(codes.sizes() == x.sizes() && scale.dim() == 1 &&
                  scale.size(0) == x.size(0), "Wrong output shapes");
  STD_TORCH_CHECK(x.numel() <= std::numeric_limits<int>::max(), "Matrix too large");
  torch::stable::accelerator::DeviceGuard guard(device);
  quantize_rows(x.data_ptr(), static_cast<int8_t*>(codes.data_ptr()),
                static_cast<float*>(scale.data_ptr()), x.size(0), x.size(1),
                stream_for(device));
  launch_check();
}
void boxed_mm(StableIValue* stack, uint64_t, uint64_t) {
  auto a = to<Tensor>(stack[0]);
  auto w = to<Tensor>(stack[1]);
  auto sa = to<Tensor>(stack[2]);
  auto sw = to<Tensor>(stack[3]);
  auto out = to<Tensor>(stack[4]);
  const int device = a.get_device_index();
  check(a, ScalarType::Char, device);
  check(w, ScalarType::Byte, device);
  check(sa, ScalarType::Float, device);
  check(sw, ScalarType::BFloat16, device);
  check(out, ScalarType::BFloat16, device);
  STD_TORCH_CHECK(a.dim() == 2 && w.dim() == 2 && out.dim() == 2,
                  "Expected matrices");
  const int64_t m = a.size(0), n = w.size(0), k = a.size(1);
  STD_TORCH_CHECK(m > 0 && n > 0 && k > 0 && k % 32 == 0 &&
                  k <= 65536 && w.size(1) == k, "Invalid matrix geometry");
  STD_TORCH_CHECK(sa.dim() == 1 && sa.size(0) == m && sw.dim() == 1 &&
                  sw.size(0) == n && out.size(0) == m && out.size(1) == n,
                  "Wrong scale or output shapes");
  STD_TORCH_CHECK(a.numel() <= std::numeric_limits<int>::max() &&
                  w.numel() <= std::numeric_limits<int>::max(), "Matrix too large");
  torch::stable::accelerator::DeviceGuard guard(device);
  channel_mm(static_cast<int8_t*>(a.data_ptr()), static_cast<uint8_t*>(w.data_ptr()),
              static_cast<float*>(sa.data_ptr()), sw.data_ptr(), out.data_ptr(),
              m, n, k, stream_for(device));
  launch_check();
}
void boxed_unpack_allspark(StableIValue* stack, uint64_t, uint64_t) {
  auto packed = to<Tensor>(stack[0]), packed_scales = to<Tensor>(stack[1]);
  auto weight = to<Tensor>(stack[2]), scales = to<Tensor>(stack[3]);
  const int device = packed.get_device_index();
  check(packed, ScalarType::Byte, device);
  check(packed_scales, ScalarType::BFloat16, device);
  check(weight, ScalarType::Char, device);
  check(scales, ScalarType::BFloat16, device);
  STD_TORCH_CHECK(packed.dim() == 2 && weight.dim() == 2 &&
      packed_scales.dim() == 1 && scales.dim() == 1, "Wrong tensor ranks");
  const int64_t n = weight.size(0), k = weight.size(1), padded_n = (n + 31) / 32 * 32;
  STD_TORCH_CHECK(n > 0 && n % 16 == 0 && k > 0 && k <= 65536 && k % 16 == 0 &&
      packed.size(0) == padded_n && packed.size(1) == k &&
      packed_scales.size(0) == padded_n && scales.size(0) == n,
      "Expected AllSpark N32K16 weights and channel scales");
  STD_TORCH_CHECK(packed.numel() <= std::numeric_limits<int>::max(), "Weight too large");
  STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(packed.data_ptr()) % 4 == 0 &&
      reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0, "Unaligned weights");
  torch::stable::accelerator::DeviceGuard guard(device);
  unpack_allspark(static_cast<uint8_t*>(packed.data_ptr()), packed_scales.data_ptr(),
      static_cast<int8_t*>(weight.data_ptr()), scales.data_ptr(), n, k, stream_for(device));
  launch_check();
}
void boxed_gemv(StableIValue* stack, uint64_t, uint64_t) {
  auto x = to<Tensor>(stack[0]), w = to<Tensor>(stack[1]);
  auto scale = to<Tensor>(stack[2]), out = to<Tensor>(stack[3]);
  const int device = x.get_device_index();
  check(x, ScalarType::BFloat16, device);
  check(w, ScalarType::Char, device);
  check(scale, ScalarType::BFloat16, device);
  check(out, ScalarType::BFloat16, device);
  STD_TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && out.dim() == 2 && scale.dim() == 1,
                  "Expected input/weight/output matrices and channel scales");
  const int64_t m = x.size(0), n = w.size(0), k = x.size(1);
  STD_TORCH_CHECK(m > 0 && m <= 8 && n > 0 && k > 0 && k <= 65536 && k % 16 == 0 &&
      w.size(1) == k && scale.size(0) == n && out.size(0) == m && out.size(1) == n,
      "Unsupported W8A16 GEMV geometry");
  STD_TORCH_CHECK(w.numel() <= std::numeric_limits<int>::max(), "Weight matrix too large");
  STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(x.data_ptr()) % 4 == 0, "Unaligned vector inputs");
  torch::stable::accelerator::DeviceGuard guard(device);
  channel_gemv(x.data_ptr(), static_cast<int8_t*>(w.data_ptr()), scale.data_ptr(),
                out.data_ptr(), m, n, k, stream_for(device));
  launch_check();
}
void boxed_gdn(StableIValue* stack, uint64_t, uint64_t) {
  auto q = to<Tensor>(stack[0]), k = to<Tensor>(stack[1]);
  auto v = to<Tensor>(stack[2]), g = to<Tensor>(stack[3]);
  auto beta = to<Tensor>(stack[4]), state = to<Tensor>(stack[5]);
  auto slots = to<Tensor>(stack[6]), out = to<Tensor>(stack[7]);
  const int device = q.get_device_index();
  for (const auto& t : {q, k, v, out}) check(t, ScalarType::BFloat16, device);
  for (const auto& t : {g, beta, state}) check(t, ScalarType::Float, device);
  check(slots, ScalarType::Long, device);
  STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(state.data_ptr()) % 16 == 0,
                  "GDN state requires 16-byte alignment");
  STD_TORCH_CHECK(q.dim() == 3 && v.dim() == 3 && state.dim() == 4 &&
                  k.sizes() == q.sizes() && out.sizes() == v.sizes(),
                  "Invalid GDN tensor ranks or shapes");
  const int64_t b = q.size(0), hq = q.size(1), hv = v.size(1);
  STD_TORCH_CHECK(b > 0 && hq > 0 && hv > 0 && hv % hq == 0 &&
                  q.size(2) == 128 && v.size(2) == 128 && v.size(0) == b &&
                  state.size(1) == hv && state.size(2) == 128 &&
                  state.size(3) == 128, "GDN requires head dimension 128");
  STD_TORCH_CHECK(g.dim() == 2 && g.size(0) == b && g.size(1) == hv &&
                  beta.sizes() == g.sizes() && slots.dim() == 1 &&
                  slots.size(0) == b, "Wrong GDN gates or slots");
  torch::stable::accelerator::DeviceGuard guard(device);
  gdn_decode(q.data_ptr(), k.data_ptr(), v.data_ptr(),
              static_cast<float*>(g.data_ptr()), static_cast<float*>(beta.data_ptr()),
              static_cast<float*>(state.data_ptr()), static_cast<int64_t*>(slots.data_ptr()),
              out.data_ptr(), b, hq, hv, state.size(0), stream_for(device));
  launch_check();
}
void boxed_gdn_packed(StableIValue* stack, uint64_t, uint64_t) {
  auto qkv = to<Tensor>(stack[0]), a = to<Tensor>(stack[1]);
  auto b = to<Tensor>(stack[2]), A_log = to<Tensor>(stack[3]);
  auto dt = to<Tensor>(stack[4]), state = to<Tensor>(stack[5]);
  auto slots = to<Tensor>(stack[6]), out = to<Tensor>(stack[7]);
  const int device = qkv.get_device_index();
  for (const auto& t : {qkv, a, b}) {
    STD_TORCH_CHECK(t.is_cuda() && t.get_device_index() == device &&
        t.scalar_type() == ScalarType::BFloat16 && t.dim() == 2 &&
        t.stride(1) == 1, "Expected strided CUDA BF16 packed inputs");
  }
  check(A_log, ScalarType::Float, device);
  check(out, ScalarType::BFloat16, device);
  STD_TORCH_CHECK(dt.is_cuda() && dt.get_device_index() == device &&
      dt.is_contiguous() && dt.dim() == 1 &&
      (dt.scalar_type() == ScalarType::Float || dt.scalar_type() == ScalarType::BFloat16),
      "Expected BF16 or FP32 dt_bias");
  STD_TORCH_CHECK(state.is_cuda() && state.get_device_index() == device &&
      state.scalar_type() == ScalarType::Float && state.dim() == 4 &&
      state.size(2) == 128 && state.size(3) == 128 && state.stride(3) == 1 &&
      state.stride(2) == 128 && state.stride(1) == 128 * 128 &&
      state.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(state.data_ptr()) % 16 == 0,
      "Expected aligned FP32 [slot,Hv,128,128] GDN state");
  STD_TORCH_CHECK(slots.is_cuda() && slots.get_device_index() == device &&
      slots.dim() == 1 &&
      (slots.scalar_type() == ScalarType::Long || slots.scalar_type() == ScalarType::Int),
      "Expected INT32 or INT64 state slots");
  const int64_t batch = qkv.size(0), hv = state.size(1);
  const int64_t qk_width = qkv.size(1) - hv * 128;
  STD_TORCH_CHECK(batch > 0 && batch <= 65535 && hv > 0 && hv <= 65535 &&
      qk_width > 0 && qk_width % 256 == 0, "Invalid packed GDN dimensions");
  const int64_t hq = qk_width / 256;
  STD_TORCH_CHECK(hv % hq == 0 && a.size(0) == batch && a.size(1) == hv &&
      b.sizes() == a.sizes() && A_log.dim() == 1 && A_log.size(0) == hv &&
      dt.size(0) == hv && slots.size(0) == batch && out.dim() == 3 &&
      out.size(0) == batch && out.size(1) == hv && out.size(2) == 128 &&
      state.size(0) > 0, "Mismatched GDN shapes");
  torch::stable::accelerator::DeviceGuard guard(device);
  gdn_packed_decode(qkv.data_ptr(), a.data_ptr(), b.data_ptr(),
      static_cast<float*>(A_log.data_ptr()), dt.data_ptr(),
      static_cast<float*>(state.data_ptr()), slots.data_ptr(), out.data_ptr(),
      batch, hq, hv, state.size(0), qkv.stride(0), a.stride(0), b.stride(0),
      state.stride(0), slots.stride(0), dt.scalar_type() == ScalarType::Float,
      slots.scalar_type() == ScalarType::Long, stream_for(device));
  launch_check();
}
void boxed_cutlass_mm(StableIValue* stack, uint64_t, uint64_t) {
  auto a = to<Tensor>(stack[0]), w = to<Tensor>(stack[1]);
  auto sa = to<Tensor>(stack[2]), sw = to<Tensor>(stack[3]);
  auto workspace = to<Tensor>(stack[4]), out = to<Tensor>(stack[5]);
  const int device = a.get_device_index();
  check(a, ScalarType::Char, device); check(w, ScalarType::Char, device);
  check(sa, ScalarType::Float, device); check(sw, ScalarType::BFloat16, device);
  check(workspace, ScalarType::Int, device); check(out, ScalarType::BFloat16, device);
  STD_TORCH_CHECK(a.dim() == 2 && w.dim() == 2 && out.dim() == 2 &&
                  workspace.sizes() == out.sizes(), "Expected matrices and matching workspace");
  const int64_t m = a.size(0), n = w.size(0), k = a.size(1);
  STD_TORCH_CHECK(m > 0 && n > 0 && k > 0 && n % 4 == 0 && k % 32 == 0 &&
                  k <= 65536 && w.size(1) == k && out.size(0) == m &&
                  out.size(1) == n && sa.dim() == 1 && sw.dim() == 1 &&
                  sa.size(0) == m && sw.size(0) == n, "Invalid CUTLASS geometry");
  STD_TORCH_CHECK(a.numel() <= std::numeric_limits<int>::max() &&
                  w.numel() <= std::numeric_limits<int>::max() &&
                  out.numel() <= std::numeric_limits<int>::max(), "Matrix too large");
  torch::stable::accelerator::DeviceGuard guard(device);
  cutlass_channel_mm(static_cast<int8_t*>(a.data_ptr()), static_cast<int8_t*>(w.data_ptr()),
      static_cast<float*>(sa.data_ptr()), sw.data_ptr(), static_cast<int32_t*>(workspace.data_ptr()),
      out.data_ptr(), m, n, k, stream_for(device));
  launch_check();
}
void boxed_kv_cache_update(StableIValue* stack, uint64_t, uint64_t) {
  auto key = to<Tensor>(stack[0]), value = to<Tensor>(stack[1]);
  auto cache = to<Tensor>(stack[2]), slots = to<Tensor>(stack[3]);
  const int device = key.get_device_index();
  for (const auto& t : {key, value}) {
    STD_TORCH_CHECK(t.is_cuda() && t.get_device_index() == device &&
        t.scalar_type() == ScalarType::BFloat16 && t.dim() == 3 &&
        t.size(2) == 256 && t.stride(2) == 1, "Expected CUDA BF16 [T,H,256] keys and values");
  }
  check(slots, ScalarType::Long, device);
  STD_TORCH_CHECK(value.sizes() == key.sizes() && slots.dim() == 1 &&
      slots.size(0) == key.size(0), "Key, value and slot shapes must match");
  STD_TORCH_CHECK(cache.is_cuda() && cache.get_device_index() == device &&
      cache.scalar_type() == ScalarType::Char && cache.dim() == 4 &&
      cache.size(1) == key.size(1) && cache.size(3) == 520 && cache.stride(3) == 1 &&
      cache.size(0) > 0 && cache.size(2) > 0, "Expected packed INT8 cache [blocks,H,block_size,520]");
  STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(cache.data_ptr()) % 4 == 0 &&
      cache.stride(0) % 4 == 0 && cache.stride(1) % 4 == 0 && cache.stride(2) % 4 == 0,
      "Cache scales require four-byte alignment");
  STD_TORCH_CHECK(key.size(1) > 0 && key.numel() <= std::numeric_limits<int>::max(),
      "Invalid key geometry");
  if (key.size(0) == 0) return;
  torch::stable::accelerator::DeviceGuard guard(device);
  kv_cache_update(key.data_ptr(), value.data_ptr(), static_cast<int8_t*>(cache.data_ptr()),
      static_cast<int64_t*>(slots.data_ptr()), key.size(0), key.size(1), cache.size(0),
      cache.size(2), key.stride(0), key.stride(1), value.stride(0), value.stride(1),
      cache.stride(0), cache.stride(1), cache.stride(2), stream_for(device));
  launch_check();
}
void boxed_gdn_prefill(StableIValue* stack, uint64_t, uint64_t) {
  auto q = to<Tensor>(stack[0]), k = to<Tensor>(stack[1]), v = to<Tensor>(stack[2]);
  auto g = to<Tensor>(stack[3]), beta = to<Tensor>(stack[4]);
  auto initial = to<Tensor>(stack[5]), final = to<Tensor>(stack[6]);
  auto out = to<Tensor>(stack[7]), cumulative = to<Tensor>(stack[8]);
  auto w = to<Tensor>(stack[9]), u = to<Tensor>(stack[10]);
  auto v_new = to<Tensor>(stack[11]), h_chunk = to<Tensor>(stack[12]);
  const int64_t num_sms = to<int64_t>(stack[13]);
  const int device = q.get_device_index();
  for (const auto& t : {q, k, v, out, w, u, v_new, h_chunk}) {
    check(t, ScalarType::BFloat16, device);
    STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                    "Prefill BF16 buffers require 16-byte alignment");
  }
  for (const auto& t : {g, beta, initial, final, cumulative}) check(t, ScalarType::Float, device);
  STD_TORCH_CHECK(q.dim() == 3 && v.dim() == 3 && k.sizes() == q.sizes(),
                  "Expected normalized Q/K and V as [T,H,128]");
  const int64_t tokens = q.size(0), hq = q.size(1), hv = v.size(1);
  STD_TORCH_CHECK(tokens > 0 && tokens <= 32768 && tokens % 64 == 0 &&
      q.size(2) == 128 && v.size(2) == 128 && v.size(0) == tokens &&
      ((hq == 8 && hv == 24) || (hq == 16 && hv == 48)) &&
      num_sms > 0 && num_sms <= 1024, "Unsupported Qwen GDN prefill geometry");
  STD_TORCH_CHECK(g.dim() == 2 && g.size(0) == tokens && g.size(1) == hv &&
      beta.sizes() == g.sizes() && cumulative.sizes() == g.sizes(), "Invalid gate/scan workspace shape");
  STD_TORCH_CHECK(initial.dim() == 3 && initial.size(0) == hv &&
      initial.size(1) == 128 && initial.size(2) == 128 && final.sizes() == initial.sizes(),
      "Expected FP32 [Hv,V128,K128] initial and final states");
  STD_TORCH_CHECK(out.sizes() == v.sizes() && w.sizes() == v.sizes() &&
      u.sizes() == v.sizes() && v_new.sizes() == v.sizes() && h_chunk.dim() == 4 &&
      h_chunk.size(0) == tokens / 64 && h_chunk.size(1) == hv &&
      h_chunk.size(2) == 128 && h_chunk.size(3) == 128, "Invalid prefill workspace shape");
  torch::stable::accelerator::DeviceGuard guard(device);
  const auto status = gdn_prefill(q.data_ptr(), k.data_ptr(), v.data_ptr(),
      static_cast<float*>(g.data_ptr()), static_cast<float*>(beta.data_ptr()),
      static_cast<float*>(initial.data_ptr()), static_cast<float*>(final.data_ptr()),
      out.data_ptr(), static_cast<float*>(cumulative.data_ptr()), w.data_ptr(), u.data_ptr(),
      v_new.data_ptr(), h_chunk.data_ptr(), tokens, hq, hv, num_sms, stream_for(device));
  STD_TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}
void boxed_unpack_kv_cache(StableIValue* stack, uint64_t, uint64_t) {
  auto cache = to<Tensor>(stack[0]), table = to<Tensor>(stack[1]);
  auto lengths = to<Tensor>(stack[2]), key = to<Tensor>(stack[3]);
  auto value = to<Tensor>(stack[4]);
  const int device = cache.get_device_index();
  check(key, ScalarType::BFloat16, device);
  check(value, ScalarType::BFloat16, device);
  check(lengths, ScalarType::Int, device);
  STD_TORCH_CHECK(cache.is_cuda() && cache.scalar_type() == ScalarType::Char &&
      cache.dim() == 4 && cache.size(0) > 0 &&
      (cache.size(1) == 2 || cache.size(1) == 4) && cache.size(2) > 0 &&
      cache.size(3) == 520 && cache.stride(3) == 1 &&
      cache.stride(0) % 4 == 0 && cache.stride(1) % 4 == 0 && cache.stride(2) % 4 == 0 &&
      reinterpret_cast<uintptr_t>(cache.data_ptr()) % 4 == 0,
      "Expected aligned packed INT8 [pages,Hkv,block_size,520] cache");
  STD_TORCH_CHECK(lengths.dim() == 1 && lengths.size(0) > 0 && lengths.size(0) <= 65535 &&
      table.is_cuda() && table.get_device_index() == device &&
      table.scalar_type() == ScalarType::Int && table.dim() == 2 &&
      table.stride(1) == 1 && table.size(0) == lengths.size(0) && table.size(1) > 0,
      "Invalid page/sequence metadata");
  STD_TORCH_CHECK(key.dim() == 4 && key.sizes() == value.sizes() &&
      key.size(0) == lengths.size(0) && key.size(1) > 0 &&
      key.size(1) * cache.size(1) <= std::numeric_limits<int>::max() &&
      key.size(2) == cache.size(1) && key.size(3) == 256,
      "Expected contiguous BF16 [sequences,capacity,Hkv,256] outputs");
  torch::stable::accelerator::DeviceGuard guard(device);
  unpack_kv_cache(static_cast<int8_t*>(cache.data_ptr()),
      static_cast<int32_t*>(table.data_ptr()), static_cast<int32_t*>(lengths.data_ptr()),
      key.data_ptr(), value.data_ptr(), cache.size(1), key.size(1), lengths.size(0),
      cache.size(2), table.stride(0), cache.stride(0), cache.stride(1), cache.stride(2),
      stream_for(device));
  launch_check();
}
void boxed_paged_attention(StableIValue* stack, uint64_t, uint64_t) {
  auto q = to<Tensor>(stack[0]), cache = to<Tensor>(stack[1]);
  auto table = to<Tensor>(stack[2]), offsets = to<Tensor>(stack[3]);
  auto lengths = to<Tensor>(stack[4]), out = to<Tensor>(stack[5]);
  const int64_t max_query = to<int64_t>(stack[6]);
  const double scale = to<double>(stack[7]);
  const int device = q.get_device_index();
  for (const auto& t : {q, out}) {
    STD_TORCH_CHECK(t.is_cuda() && t.get_device_index() == device &&
        t.scalar_type() == ScalarType::BFloat16 && t.dim() == 3 &&
        t.size(2) == 256 && t.stride(2) == 1 && t.stride(1) >= 256 &&
        t.stride(0) >= t.size(1) * t.stride(1), "Expected BF16 [T,H,256] query/output");
  }
  STD_TORCH_CHECK(q.size(0) > 0 && q.sizes() == out.sizes() &&
      (q.size(1) == 12 || q.size(1) == 24) && max_query > 0 && max_query <= q.size(0),
      "Unsupported Qwen query geometry");
  STD_TORCH_CHECK(cache.is_cuda() && cache.get_device_index() == device &&
      cache.scalar_type() == ScalarType::Char && cache.dim() == 4 &&
      cache.size(0) > 0 && cache.size(1) * 6 == q.size(1) && cache.size(2) > 0 &&
      cache.size(3) == 520 && cache.stride(3) == 1 &&
      cache.stride(0) % 4 == 0 && cache.stride(1) % 4 == 0 && cache.stride(2) % 4 == 0 &&
      reinterpret_cast<uintptr_t>(cache.data_ptr()) % 4 == 0,
      "Expected aligned packed INT8 [pages,Hkv,block_size,520] cache");
  check(offsets, ScalarType::Int, device);
  check(lengths, ScalarType::Int, device);
  STD_TORCH_CHECK(table.is_cuda() && table.get_device_index() == device &&
      table.scalar_type() == ScalarType::Int && table.dim() == 2 && table.stride(1) == 1 &&
      lengths.dim() == 1 && offsets.dim() == 1 && lengths.size(0) > 0 && lengths.size(0) <= 65535 &&
      table.size(0) == lengths.size(0) && table.size(1) > 0 &&
      offsets.size(0) == lengths.size(0) + 1, "Invalid page/sequence metadata");
  torch::stable::accelerator::DeviceGuard guard(device);
  paged_attention(q.data_ptr(), static_cast<int8_t*>(cache.data_ptr()),
      static_cast<int32_t*>(table.data_ptr()), static_cast<int32_t*>(offsets.data_ptr()),
      static_cast<int32_t*>(lengths.data_ptr()), out.data_ptr(),
      q.size(1), cache.size(1), cache.size(2), table.stride(0), lengths.size(0), max_query,
      q.stride(0), q.stride(1), cache.stride(0), cache.stride(1), cache.stride(2),
      out.stride(0), out.stride(1), scale, stream_for(device));
  launch_check();
}
}  // namespace

STABLE_TORCH_LIBRARY(consumer_int8, m) {
  m.def("unpack_kv_cache(Tensor cache, Tensor table, Tensor lengths, Tensor(a!) key, Tensor(b!) value) -> ()");
  m.def("paged_attention(Tensor query, Tensor cache, Tensor table, Tensor offsets, Tensor lengths, Tensor(a!) out, int max_query, float scale) -> ()");
  m.def("quantize_rows(Tensor input, Tensor(a!) codes, Tensor(b!) scales) -> ()");
  m.def("channel_mm(Tensor a, Tensor w, Tensor a_scale, Tensor w_scale, Tensor(a!) out) -> ()");
  m.def("channel_gemv(Tensor x, Tensor w, Tensor scale, Tensor(a!) out) -> ()");
  m.def("unpack_allspark(Tensor packed, Tensor packed_scales, Tensor(a!) weight, Tensor(b!) scales) -> ()");
  m.def("gdn_decode(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor(a!) state, Tensor slots, Tensor(b!) out) -> ()");
  m.def("gdn_packed_decode(Tensor qkv, Tensor a, Tensor b, Tensor A_log, Tensor dt_bias, Tensor(a!) state, Tensor slots, Tensor(b!) out) -> ()");
  m.def("cutlass_channel_mm(Tensor a, Tensor w, Tensor a_scale, Tensor w_scale, Tensor(a!) workspace, Tensor(b!) out) -> ()");
  m.def("kv_cache_update(Tensor key, Tensor value, Tensor(a!) cache, Tensor slots) -> ()");
  m.def("gdn_prefill(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor initial, Tensor(a!) final, Tensor(b!) out, Tensor(c!) cumulative, Tensor(d!) w, Tensor(e!) u, Tensor(f!) v_new, Tensor(g!) h_chunk, int num_sms) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(consumer_int8, CUDA, m) {
  m.impl("unpack_kv_cache", &boxed_unpack_kv_cache);
  m.impl("paged_attention", &boxed_paged_attention);
  m.impl("quantize_rows", &boxed_quant);
  m.impl("channel_mm", &boxed_mm);
  m.impl("channel_gemv", &boxed_gemv);
  m.impl("unpack_allspark", &boxed_unpack_allspark);
  m.impl("gdn_decode", &boxed_gdn);
  m.impl("gdn_packed_decode", &boxed_gdn_packed);
  m.impl("cutlass_channel_mm", &boxed_cutlass_mm);
  m.impl("kv_cache_update", &boxed_kv_cache_update);
  m.impl("gdn_prefill", &boxed_gdn_prefill);
}
extern "C" PyObject* PyInit__C() {
  static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_C", nullptr, -1, nullptr};
  return PyModule_Create(&module);
}
