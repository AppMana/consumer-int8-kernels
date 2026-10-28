// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

void gdn_conv(const void* x, const void* weight, void* state,
    const int32_t* offsets, const int32_t* slots, const bool* initial,
    const void* a, const void* b, const float* log_a, const void* dt,
    void* q, void* k, void* v, float* g, float* beta,
    int sequences, int max_query, int hq, int hv, int num_slots,
    int64_t xs, int64_t as, int64_t bs, int64_t ss, int64_t sd, int64_t st,
    int64_t slot_stride, bool float_bias, cudaStream_t stream);

void gemma_norm(const void* x, const void* residual, const void* weight,
    void* out, void* residual_out, int rows, float epsilon, cudaStream_t stream);
void gdn_post_conv(const void* x, const void* a, const void* b,
    const float* log_a, const void* dt, void* q, void* k, void* v,
    float* g, float* beta, int tokens, int hq, int hv,
    int64_t xs, int64_t as, int64_t bs, bool float_bias, cudaStream_t stream);

void fa2_prefill(const void* q, const void* k, const void* v,
    const int32_t* offsets, const int32_t* lengths, void* out, float* lse,
    int tokens, int heads, int kv_heads, int sequences, int capacity, int max_query,
    int64_t qt, int64_t qh, int64_t ot, int64_t oh, float scale, int schedule,
    cudaStream_t stream);

void fused_channel_mm(const int8_t* a, const int8_t* w, const float* sa,
    const void* sw, void* out, int m, int n, int k, int schedule, cudaStream_t stream);

void unpack_kv_cache(const int8_t* cache, const int32_t* table,
    const int32_t* lengths, void* key, void* value, int heads, int capacity,
    int sequences, int block_size, int table_stride,
    int64_t cb, int64_t ch, int64_t ct, cudaStream_t stream);

void paged_attention(const void* query, const int8_t* cache,
    const int32_t* table, const int32_t* offsets, const int32_t* lengths, void* out,
    int heads, int kv_heads, int block_size, int table_stride, int sequences, int max_query,
    int64_t qt, int64_t qh, int64_t cb, int64_t ch, int64_t ct,
    int64_t ot, int64_t oh, float scale, cudaStream_t stream);

cudaError_t gdn_prefill(const void* q, const void* k, const void* v,
    const float* g, const float* beta, const float* initial, float* final,
    void* out, float* cumulative, void* w, void* u, void* v_new, void* h_chunk,
    int tokens, int hq, int hv, int num_sms, cudaStream_t stream);

void quantize_rows(const void* input, int8_t* codes, float* scales,
                   int rows, int cols, cudaStream_t stream);
void quantize_grouped4096(const void* input, int8_t* codes, float* scales,
                          int rows, int groups, cudaStream_t stream);
void unpack_allspark(const uint8_t* packed, const void* scales, int8_t* weight,
                     void* restored_scales, int n, int k, cudaStream_t stream);
void channel_gemv(const void* x, const int8_t* weight, const void* scales,
                  void* out, int m, int n, int k, cudaStream_t stream);
void channel_mm(const int8_t* a, const uint8_t* w, const float* a_scale,
                const void* w_scale, void* out, int m, int n, int k,
                cudaStream_t stream);
void gdn_decode(const void* q, const void* k, const void* v, const float* g,
                const float* beta, float* state, const int64_t* slots,
                void* out, int batch, int hq, int hv, int num_slots,
                cudaStream_t stream);
void cutlass_channel_mm(const int8_t* a, const int8_t* w, const float* a_scale,
                         const void* w_scale, int32_t* workspace, void* out,
                         int m, int n, int k, cudaStream_t stream);
void kv_cache_update(const void* key, const void* value, int8_t* cache,
                      const int64_t* slots, int tokens, int heads, int blocks,
                      int block_size, int64_t key_token_stride, int64_t key_head_stride,
                      int64_t value_token_stride, int64_t value_head_stride,
                      int64_t cache_block_stride, int64_t cache_head_stride,
                      int64_t cache_token_stride, cudaStream_t stream);

void gdn_packed_decode(const void* qkv, const void* a, const void* b,
    const float* A_log, const void* dt_bias, float* state, const void* slots,
    void* out, int batch, int hq, int hv, int num_slots, int64_t qkv_stride,
    int64_t a_stride, int64_t b_stride, int64_t state_stride, int64_t slot_stride,
    bool dt_float, bool slots_int64, cudaStream_t stream);

void paged_decode(const void* query, const int8_t* cache, const int32_t* table,
    const int32_t* offsets, const int32_t* lengths, void* out, float* scratch,
    int tokens, int heads, int kv_heads, int parts, int block_size,
    int table_stride, int sequences, int max_query, int64_t qt, int64_t qh,
    int64_t cb, int64_t ch, int64_t ct, int64_t ot, int64_t oh, float scale,
    cudaStream_t stream);
