# Consumer INT8 kernels

An independent AOT CUDA package for Qwen3.8-27B on SM86 GPUs, with torch stable
ABI bindings targeting torch 2.10 and Python 3.10 or newer. It does not import
vLLM. The serving engine owns allocation, cache scheduling and backend selection.

This is an implementation in progress. The checkpoint encoder and initial
matmul/GDN correctness tests have run; serving integration, model quality,
performance selection and release validation are pending. An encoded checkpoint
is marked `encoded_unvalidated` until those gates pass.

## Build and test

With a CUDA-enabled torch environment and CUDA toolkit installed:

```sh
git submodule update --init --recursive
TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=2 uv pip install -e . --no-build-isolation --no-deps
python -m pytest tests
```

Importing `consumer_int8.ops` loads the compiled extension; it never invokes a
compiler. All operators write caller-owned outputs. Quantization scales,
workspaces and graph buffers must be allocated before graph capture.

## Checkpoint format

```sh
consumer-int8-encode /path/to/pinned/Qwen3.8-27B /path/to/qwen3.8-27b-int8
```

The encoder preserves tensor names, shard ownership, vision, MTP, tokenizer and
processor files. Language/MTP projection matrices use DSV4's signed codes plus
128 in UINT8 storage and one BF16 scale per output channel. Vision linear
weights use signed INT8 with FP32 scales per 32 input channels, including tails.
Embeddings, the language-model head, small gates, norms and convolutions retain
source precision. The recurrent GDN state remains FP32.

The output records source revision, tensor inventory, per-tensor quantization
error, and output shard SHA256s. Config is written last so an interrupted
conversion cannot be mistaken for a complete checkpoint. This encoding needs
the vLLM `qwen_int8` integration before serving; it is not a generic Transformers
quantization format.

## Kernel contracts

| Operator | Inputs | Computation / storage |
| --- | --- | --- |
| `quantize_rows` | BF16 `[M,K]` | Dynamic signed INT8 activations and FP32 row scales |
| `channel_mm` | Signed INT8 activations, biased UINT8 weights | `m16n8k32` integer MMA, BF16 output; diagnostic candidate |
| `cutlass_channel_mm` | Signed INT8 activations/weights | SM86 tiled integer MMA, caller-owned INT32 workspace, BF16 output |
| `gdn_decode` | BF16 Q/K/V, FP32 log decay and sigmoid gates | Head dimensions 128, normalized Q/K, FP32 recurrent update |
| `gdn_packed_decode` | Strided BF16 QKV and raw gates | Fused gating and FP32 recurrence, INT32/INT64 slot indices |
| `kv_cache_update` | BF16 K/V with head dimension 256 | Packed INT8 cache and inline FP32 scales, HND or NHD strides |

`gdn_decode` accepts `[batch,key_heads,128]` Q/K,
`[batch,value_heads,128]` V, and FP32
`[slot,value_heads,128,128]` states ordered value-dimension then key-dimension.
Active state slots must be unique; slots at or below zero are padding.
`gdn_packed_decode` also accepts strided slot indices and padding between state
slots. `A_log` is FP32; `dt_bias` may be BF16 or FP32. State rows require
16-byte alignment. It matches vLLM's non-speculative packed decode contract;
multi-token speculative verification continues to use the engine's existing
implementation until separately validated.

`kv_cache_update` writes logical `[blocks,kv_heads,block_size,520]` cache rows
containing `[K256, K_scale4, V256, V_scale4]` bytes. Slot zero is valid; negative
slots are padding. Slots must be unique and in range. Input token/head strides
are honored. Quantization uses a scale floor of `1e-6`, round-half-away-from-zero
and saturation to `[-128,127]`, matching vLLM's Triton cache writer.

GDN decode is floating-point arithmetic. INT8 cache storage does not imply
integer attention arithmetic. The two matmul operators actually use integer
tensor-core instructions. No candidate is selected for serving solely because
it uses CUDA or integer arithmetic; numerical and performance gates apply.

`benchmarks/microbench.py` compares CUDA-graph timings with the installed vLLM
FLA and AllSpark implementations. Performance measurements require exclusive
GPUs; records taken alongside another GPU workload must be discarded.

`benchmarks/needle_matrix.py` targets one explicitly configured TP/PP/MTP
server per invocation. It runs C=1/2/4 and 8k/16k/32k prompts with a 1k-token
needle, one warmup and three measured cohorts per cell. Early/middle/late
needle depths rotate even at C=1. Prompts include the chat-template token
overhead, and output usage is recorded as actually returned (the 1k-token
passage has a 128-token output allowance).

The server must expose `metrics.engine_phase_timing` with same-host engine
monotonic timestamps and have prefix caching disabled. Prefill is scheduled
execution through the first output token; decode is first through last output
token. Aggregate rates use interval unions, and decode counts exclude each
request's first output token. These are elapsed request-phase rates, including
scheduling gaps inside a phase, rather than isolated GPU kernel throughput.
Client TTFT, end-to-end latency and stream chunk gaps are recorded separately.
The harness fails closed if engine timing or actual token usage is missing.
