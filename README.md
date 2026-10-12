# Consumer INT8 kernels

An independent AOT CUDA package for Qwen3.8-27B on SM86 GPUs, with torch stable
ABI bindings targeting torch 2.10 and Python 3.10 or newer. It does not import
vLLM. The serving engine owns allocation, cache scheduling and backend selection.

This is an implementation in progress. The checkpoint encoder, CUDA kernels and
vLLM adapter are implemented. Initial text and vision checks pass with AllSpark
linears, including the optional native GDN/cache path. Long-context retrieval,
MTP, serving performance selection and release validation are pending. An encoded checkpoint
is marked `encoded_unvalidated` until those gates pass.

## Build and test

With a CUDA-enabled torch environment and CUDA toolkit installed:

```sh
git submodule update --init --recursive
TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=2 uv pip install -e . --no-build-isolation --no-deps
.venv/bin/python -m pytest tests
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
| `channel_gemv` | BF16 activations, canonical signed INT8 weights | Small-batch W8A16 with BF16 dequantized multiplicands and FP32 accumulation |
| `unpack_allspark` | AllSpark N32K16 UINT8 weights and permuted BF16 scales | Lossless restoration of signed row-major INT8 weights and channel scales |
| `gdn_decode` | BF16 Q/K/V, FP32 log decay and sigmoid gates | Head dimensions 128, normalized Q/K, FP32 recurrent update |
| `gdn_packed_decode` | Strided BF16 QKV and raw gates | Fused gating and FP32 recurrence, INT32/INT64 slot indices |
| `gdn_prefill` | Normalized BF16 Q/K, BF16 V, FP32 gates and initial state | Chunked BF16/TF32 attention with FP32 final state |
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

`gdn_prefill` accepts one contiguous sequence of 64–32768 tokens, in multiples
of 64, with `(key_heads,value_heads)` equal to `(8,24)` or `(16,48)`. Initial
and final states have shape `[value_heads,128,128]`, ordered value then key.
The caller provides all intermediate buffers and the device's actual SM count.
It wraps the pinned NInfer preparation, state propagation and output kernels;
it does not yet support variable-length batches or unaligned tails. CUDA-graph
correctness against FLA has passed for 64, 256 and 1024 tokens on an RTX A5000.
Serving dispatch and model-level quality validation remain pending.

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

On one idle RTX A5000, the 2026-10-11 sweep of twelve Qwen projection shapes
found CUTLASS W8A8 1.68x faster than AllSpark at 1024 rows (geometric mean).
Restoring AllSpark's packed weights on every call reduces that gain to 1.41x,
with all twelve shapes still faster. At 256 rows, restoration erases the gain.
This supports a candidate dispatch that retains AllSpark for small batches
and restores one transient layer for 1024-row prefill, without a persistent
second weight copy. These are CUDA-graph kernel timings, not serving rates.

The first complete 65,504-token quality run measured BF16 mean NLL 1.951675.
AllSpark INT8 increased NLL by 0.005659 and scored 85/100 vision fixtures at
both C=1 and C=4, with no newly incorrect cases against BF16's 82/100.
AllSpark plus native GDN/cache also passed these checks. Applying W8A8 prefill
to all language projections passed NLL but regressed one vision case at C=4;
that mode remains experimental. The `consumer_int8_mlp_prefill` candidate
restricts activation quantization to MLP gate/up projections, preserving BF16
activations in attention, MLP down and MTP input projections. The MLP-wide
candidate still regressed the same C=4 vision case. Restricting W8A8 to gate/up
projections also regressed that case, despite passing NLL (+0.005424). All
tested W8A8 prefill modes remain experimental. The selected serving candidate
uses AllSpark projections with native GDN decode and native cache insertion;
its NLL increase was 0.004887, with 85/100 vision cases at both concurrencies
and no new failures against BF16.

`benchmarks/microbench_cache.py` compares the native paged INT8 cache writer
against vLLM's Triton writer using Qwen's strided projection output, two/four
KV heads, HND/NHD layouts, and padded slots. All 24 A5000 cases match the
entire cache byte-for-byte, including FP32 scales. Small-batch writes are
1.05–1.08x faster; 256-row and 1024-row writes are about 2.0x and 3.2x faster.
The native writer uses the same PTX division rounding as Triton. These
measurements cover cache insertion only, not attention or model throughput.

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

`benchmarks/run_matrix.py` launches four fresh servers for TP=2/PP=1 and
TP=1/PP=2, each with MTP off and on, then runs all nine cells on each server.
Run it with the vLLM checkout's `.venv/bin/python` and provide `--vllm-repo`,
`--model` and a new `--output-dir`. Both source repositories must be committed.
It saves launch commands, package versions, checkpoint configuration and GPU
telemetry. Foreign compute processes invalidate performance selection. The
optional `--kv-cache-memory-bytes` fixes the per-GPU cache reservation; validate
peak memory and C=4 capacity with the same model, topology and MTP settings
before using it. It overrides automatic cache sizing from memory utilization.
The optional `benchmark` dependencies cover HTTP clients and corpus preparation;
serving still requires the separately built vLLM integration.
Both launchers retain the vision tower and allow one image per request, with
a 1024-by-768 dummy image for memory profiling; video is disabled for these
text/image evaluations. The 100 vision fixtures are at most 800 by 548 pixels.

`benchmarks/quality_nll.py` prepares 32 fixed, disjoint 2048-token windows from
the pinned WikiText-2 raw test split, scores their teacher-forced likelihoods
through `/v1/completions`, and compares mean NLL with a BF16 reference.
The default allowed increase is 0.02 nats per scored token. The first token
of each window is excluded; token identity, likelihood completeness and corpus
checksums are checked before comparison. Save each server's launch manifest as
the scoring run's provenance. This text check does not replace the vision,
needle-retrieval or MTP checks.

`benchmarks/run_quality.py` owns a TP=2 server, records GPU interference, runs
the fixed NLL corpus and optionally the AppMana 100-image fixture set. Use
`--reference --cpu-offload-gb 12` with the original BF16 snapshot for the
reference on two 24-GB GPUs. Quantized candidates use the same corpus and
vision fixtures. The runner fingerprints image contents as well as prompts;
`benchmarks/compare_vision.py` rejects every newly incorrect answer, even if
other improvements leave aggregate accuracy unchanged.

## Existing full-attention implementations

The full-attention layers have no sparse indexer. Their serving baseline is
vLLM's paged Triton attention, which supports the packed INT8 per-token/head
cache. The GDN recurrent layers have separate kernels and keep FP32 state.

At FlashAttention revision
[`94e22c9066`](https://github.com/Dao-AILab/flash-attention/tree/94e22c906678e5483fa0e9e24d8e787bc2c0ed4c),
the FA3 build includes an SM80 FP16/BF16 forward fallback, including paged and
split variants. FA4's CuTe implementation also dispatches to SM80, but that
path explicitly rejects paged KV and split-KV. These are Ampere-compatible
paths, not implementations of this package's INT8 cache format.

The existing AppMana SageAttention stable-ABI fork at
[`30a3583bb6`](https://github.com/AppMana/forks-sageattention-stable-abi/tree/30a3583bb66a62f5c3805520782d8ba488020a78)
supports SM86 and head dimensions through 256, using INT8 QK with FP16 PV.
Its documented causal mode requires equal Q and KV sequence lengths, so it
does not directly replace paged attention or continuation-prefill attention.
Backend selection still requires shape, cache-layout, quality and serving
measurements; the presence of an SM80 build alone is insufficient.
