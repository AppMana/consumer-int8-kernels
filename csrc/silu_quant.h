// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

void silu_mul_quantize(const void* x, int8_t* codes, float* scales,
                       int m, int k, cudaStream_t stream);
void silu_mul(const void* x, void* out, int m, int k, cudaStream_t stream);
