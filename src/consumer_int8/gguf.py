# SPDX-License-Identifier: Apache-2.0
"""Experimental native GGUF operators; no serving policy selects these yet.

Q8 accepts BF16 x[M,G,K], raw uint8 weight[G,N,K/32,34], and caller-owned
contiguous BF16 out[M,G,N]. Input row/group strides are independent. Decode
uses the checkpoint's FP16 scale times signed INT8, rounded to BF16 before
FP32 tensor-core accumulation. No activation quantization or global expanded
weight/scratch allocation occurs. Output must not overlap either input.

Upstream comparison (vllm-gguf-plugin e454affe): CUDA dense MMQ supports Q8
and K-quants, but IQ2/IQ3 matrix operations fall back to Triton. CUDA GEMV
has broader IQ coverage. Thus Q8 CUDA itself is not a new capability; this
addition is a BF16-activation, directly strided grouped tensor-core operator.
Our deployed b12x path uses CuTe DSL and separate group gathers/GEMMs/cat;
this operator instead reads and writes group strides in one launch.
"""

import torch

from . import _gguf  # noqa: F401

q8_mm = torch.ops.consumer_int8.gguf_q8_mm


@torch.library.register_fake("consumer_int8::gguf_q8_mm")
def _q8_fake(x, weight, out):
    return None
