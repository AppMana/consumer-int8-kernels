# Consumer INT8 kernels

An independent AOT CUDA package for Qwen3.8-27B on SM86 GPUs and
`unsloth/DeepSeek-V4.1-Flash-1017-GGUF` (`UD-Q2_K_XL`) on SM86 and DGX Spark
GB10 (`SM121`) GPUs, with torch stable ABI bindings targeting torch 2.10 and
Python 3.10 or newer. It does not import vLLM. The serving engine owns
allocation, cache scheduling and backend selection.

## Build and test

With a CUDA-enabled torch environment and CUDA toolkit installed:

```sh
git submodule update --init --recursive
TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=2 uv pip install -e . --no-build-isolation --no-deps
.venv/bin/python -m pytest tests
```

Importing `consumer_int8.ops` or `consumer_int8.gguf` loads the compiled
extensions (`_C.abi3.so`, `_fa2.abi3.so`, `_gguf.abi3.so`); it never invokes a
compiler at import time. All operators write caller-owned outputs. Quantization
scales, workspaces and graph buffers must be allocated before graph capture.

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
| `quantize_grouped_rows` | BF16 `[M,K]` (`K` multiple of 4096) | Dynamic signed INT8 activations and FP32 scales per 4096-channel group |
| `channel_mm` | Signed INT8 activations, biased UINT8 weights | `m16n8k32` integer MMA, BF16 output; diagnostic candidate |
| `cutlass_channel_mm` | Signed INT8 activations/weights | SM86 tiled integer MMA, caller-owned INT32 workspace, BF16 output |
| `fused_channel_mm` | Signed INT8 activations/weights | SM86 integer MMA with fused row/channel scaling to BF16, no INT32 output workspace |
| `channel_gemv` | BF16 activations, canonical signed INT8 weights | Small-batch W8A16 with BF16 dequantized multiplicands and FP32 accumulation |
| `unpack_allspark` | AllSpark N32K16 UINT8 weights and permuted BF16 scales | Lossless restoration of signed row-major INT8 weights and channel scales |
| `gdn_decode` | BF16 Q/K/V, FP32 log decay and sigmoid gates | Head dimensions 128, normalized Q/K, FP32 recurrent update |
| `gdn_packed_decode` | Strided BF16 QKV and raw gates | Fused gating and FP32 recurrence, INT32/INT64 slot indices |
| `gdn_prefill` | Normalized BF16 Q/K, BF16 V, FP32 gates and initial state | Chunked BF16/TF32 attention with FP32 final state |
| `kv_cache_update` / `kv_cache_update_int4` | BF16 K/V with head dimension 256 | Packed INT8 (`520 B`) or INT4 (`264 B`) cache and inline FP32 scales, HND or NHD strides |
| `paged_attention` / `paged_attention_int4` | BF16 Q, packed INT8/INT4 paged K/V, device sequence metadata | Experimental D256/GQA6 causal prefill with BF16 tensor cores and FP32 softmax |
| `paged_decode` / `paged_decode_int4` | BF16 Q (`max_query <= 8`), packed INT8/INT4 paged K/V, device sequence metadata | Native SM86 D256/GQA6 split-KV decode and MTP verification with high-plus-residual BF16 PV accumulation and 2-CTA/SM occupancy |
| `unpack_kv_cache` / `unpack_kv_cache_int4` | Packed INT8/INT4 K/V, page table and device sequence lengths | BF16 active cache rows in caller-owned contiguous scratch pages |
| `consumer_int8.gguf` | Packed GGUF `Q8_0` / `Q6_K` blocks | Native GPU unpacking, grouped W8A8 integer GEMM (`wo_a`), and shared transient BF16 expansion workspace |
| `consumer_int8.gguf_moe` | Packed GGUF `IQ2_XXS` / `IQ3_XXS` / `MXFP4` expert blocks | AOT CUDA mixed-IQ routed MoE expert kernels and borrowed-weight `dp4a` INT8 prefill (`iq2xxs_routed_moe_b12x_int8`) |

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
it does not support variable-length batches or unaligned tails. The vLLM
adapter retains FLA for serving prefill.

`kv_cache_update` writes logical `[blocks,kv_heads,block_size,520]` cache rows
containing `[K256, K_scale4, V256, V_scale4]` bytes. Slot zero is valid; negative
slots are padding. Slots must be unique and in range. Input token/head strides
are honored. Quantization uses a scale floor of `1e-6`, round-half-away-from-zero
and saturation to `[-128,127]`, matching vLLM's Triton cache writer.

`paged_attention` reads this packed cache directly, without expanding a BF16
pool. It accepts 12/24 query heads and 2/4 KV heads, INT32 page tables and
cumulative query offsets, and device sequence lengths. Metadata must describe
valid pages and disjoint output rows; each query length must be at most the
provided `max_query`, with a KV length at least as large. Strided BF16 queries
and HND/NHD caches are supported. Its 64-query, 64-key tiles use 94,464 bytes
of shared memory on SM86. K scales apply after QK, and V scales apply before
PV, preserving the serving kernel's scale placement. The vLLM FA2 adapter
uses this implementation as its scratch-limit fallback.

`unpack_kv_cache` produces `[sequences,capacity,kv_heads,256]` BF16 K/V buffers
for existing attention implementations. It honors HND/NHD cache strides and
device page-table/length updates during CUDA graph replay. Only rows below
each sequence's length are written; the caller must provide valid pages and
capacity at least as large as every sequence. It applies each FP32 scale
before rounding to BF16, which differs numerically from scaling QK/PV.

The vLLM `attention_backend=consumer_fa2` adapter combines this converter
with the installed FlashAttention 2. It limits transient K/V scratch to
512 MiB per layer/rank and uses the direct packed CUDA kernel above that
bound. Decode and unsupported attention modes retain Triton.
`benchmarks/microbench_attention.py` compares these attention implementations.

`fused_channel_mm` retains row-first FP32 scaling and BF16 channel scales.
It matches the separate INT32 GEMM plus rescale bit for bit, including CUDA
graph replay and partial tiles. It avoids the separate INT32 output buffer.
The automatic schedule uses a 128x256x64 tile with eight warps and four
pipeline stages at M>=2048; smaller inputs retain the original schedules.
`microbench.py --fused-schedule 5` selects the 128x128x64 five-stage
schedule for comparison; 0 selects automatic dispatch. Values 1–4 expose
alternative schedules.

GDN decode is floating-point arithmetic. INT8 cache storage does not imply
integer attention arithmetic. The three matmul operators actually use integer
tensor-core instructions. No candidate is selected for serving solely because
it uses CUDA or integer arithmetic; numerical and performance gates apply.

`benchmarks/microbench.py` compares CUDA-graph timings with the installed vLLM
FLA and AllSpark implementations. Performance measurements require exclusive
GPUs; records taken alongside another GPU workload must be discarded.

The vLLM `consumer_int8_prefill` projection backend uses W8A8 for at least
1024 rows and AllSpark for smaller requests. Dispatch occurs inside a runtime
custom operator so reused AOT graphs select the implementation from the actual
row count. Each W8A8 call restores one transient layer from AllSpark's packed
weights without retaining a second persistent weight copy.

`benchmarks/microbench_cache.py` compares the native paged INT8 cache writer
against vLLM's Triton writer using Qwen's strided projection output, two/four
KV heads, HND/NHD layouts, and padded slots. It checks the entire cache
byte-for-byte, including FP32 scales. The native writer uses the same PTX
division rounding as Triton. The benchmark covers cache insertion only.

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

`benchmarks/run_quality.py` owns a TP=2 server by default, records GPU interference, runs
the fixed NLL corpus and optionally the AppMana 100-image fixture set. Use
`--reference --cpu-offload-gb 12` with the original BF16 snapshot for the
reference on two 24-GB GPUs. Quantized candidates use the same corpus and
vision fixtures. The runner fingerprints image contents as well as prompts;
`benchmarks/compare_vision.py` rejects every newly incorrect answer, even if
other improvements leave aggregate accuracy unchanged.
Use `--topology pp2` and `--mtp on` to validate pipeline parallelism and MTP
with the same corpus and fixtures before promoting those serving configurations.

## Native SM86 split-KV decode and MTP verification (`paged_decode`)

`paged_decode` executes short-query (`max_query <= 8`) D256 GQA6 causal
attention directly over the packed `[blocks,kv_heads,block_size,520]` INT8
cache. It splits the KV sequence across caller-selected partitions (`4..128`)
and reduces partial `[token,head,parts,258]` FP32 accumulators (`256` output
channels plus `maximum` and `sum`) in `merge_kernel`.

- **Scaled-probability residual PV accumulation:** a single BF16 rounding of
  `P * V_scale` introduces split-dependent rounding differences before PV.
  `split_kernel` decomposes `scaled_p = hi + residual` into two BF16 tiles and
  accumulates both tensor-core `m16n8k16` products into FP32.
- **Two-CTA-per-SM register bounding:** `__launch_bounds__(Rows * 8, 2)` bounds
  the 32-row verification variant (`max_query in {3..8}`) from 168 to 128
  registers per thread, raising achieved SM86 occupancy from `16.66%` (`7.99`
  active warps/SM) to `31.49%` (`15.12` active warps/SM) when paired with the
  `2 * SMs` partition target in `vllm/v1/attention/backends/triton_attn.py`.

CUDA-graph component timings on RTX A5000 (`SM86`, 64 SMs, 7 repeats of 20
launches, `benchmarks/microbench_decode_attention.py`):

| `Hkv` | `q` | `C` | Context | Triton `unified_attention` (us) | Native 1-CTA target (us / parts) | Native 2-CTA target (us / parts) | 2-CTA vs 1-CTA | Native vs Triton |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 1 | 8,000 | 108.18 | 42.09 (64) | 41.88 (64) | 1.005x | 2.58x |
| 2 | 1 | 2 | 32,000 | 436.12 | 176.23 (32) | 177.05 (32) | 0.995x | 2.46x |
| 2 | 2 | 2 | 32,000 | 3,742.31 | 181.20 (32) | 185.45 (32) | 0.977x | 20.18x |
| 2 | 4 | 2 | 32,000 | 3,712.56 | 259.38 (16) | 236.60 (32) | 1.096x | 15.69x |
| 2 | 4 | 4 | 32,000 | 3,720.45 | 524.24 (8) | 439.50 (16) | 1.193x | 8.47x |
| 2 | 8 | 1 | 32,000 | 3,696.90 | 261.27 (16) | 229.93 (32) | 1.136x | 16.08x |
| 2 | 8 | 2 | 32,000 | 3,699.10 | 505.75 (8) | 448.41 (16) | 1.128x | 8.25x |
| 2 | 8 | 4 | 32,000 | 3,707.90 | 1,020.88 (4) | 889.04 (8) | 1.148x | 4.17x |
| 4 | 4 | 2 | 32,000 | 3,740.67 | 525.57 (8) | 460.44 (16) | 1.141x | 8.12x |
| 4 | 4 | 4 | 32,000 | 3,783.22 | 1,075.51 (4) | 955.03 (8) | 1.126x | 3.96x |

## Native GGUF kernels for `unsloth/DeepSeek-V4.1-Flash-1017-GGUF` (`UD-Q2_K_XL`)

`consumer_int8.gguf` and `consumer_int8.gguf_moe` provide AOT CUDA kernels for
serving `unsloth/DeepSeek-V4.1-Flash-1017-GGUF` (`UD-Q2_K_XL`, revision
`60ad3651641b5c97679c807e064ab1d7a3ca1455`) across two DGX Spark GB10 (`SM121`)
nodes (`TP=2/EP=2`) or SM86 GPUs:

- **Grouped `Q8_0` W8A8 (`attn.wo_a`):** `quantize_grouped_rows` +
  `consumer_int8.gguf` grouped W8A8 integer GEMM over 4,096-channel groups, and
  shared live-row transient BF16 expansion workspaces for `Q8_0` / `Q6_K`
  projections.
- **Borrowed-weight `IQ2_XXS` INT8 `dp4a` MoE prefill (`benchmarks/cuda/iq2xxs_block_int8.cu`):**
  reads `b12x`'s existing packed `IQ2_XXS` expert storage directly with zero
  extra persistent weight memory, using:
  - Vectorized `int4` (`8 x int16_t`) coalesced writes in `route_quantize_kernel`
    plus compact separated `M=16` (`E_t <= 16`) and `M=32` (`E_t > 16`) task
    lists;
  - Hybrid `M=16` (`16x64`) and `M=32` (`32x64`) `dp4a` (`__dp4a`) tile
    dispatch with padded shared-memory activation/scale/epilogue layouts to
    eliminate shared-memory bank conflicts;
  - Fused `iq2xxs_w13_fused_s8_kernel` computing `gate` (`w1`) and `up` (`w3`)
    simultaneously against a single shared-memory staging of routed INT8
    activations `a_s` and writing `bf16(silu(gate) * up)` + dynamic per-row
    `int8` FC2 input in one pass.

### 192-expert `TP=2` MoE prefill microbenchmark on DGX Spark GB10 (`SM121`)

Measured across 5 captured real-routing fixtures from
`unsloth/DeepSeek-V4.1-Flash-1017-GGUF` (`UD-Q2_K_XL`, `E=192, K=5120, N=1152,
top_k=8`, 0 `compute-sanitizer` errors, exact bitwise output parity):

| Fixture (`layer_M`) | `b12x` W4A16 (ms) | Old Borrowed `M=16` (ms) | Hybrid Fused `M=16/M=32` (ms) | Speedup vs Old | Speedup vs `b12x` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `layer3_m512` | 3.743 | 3.600 | **2.435** | **1.48x** | **1.54x** |
| `layer39_m512` | 3.770 | 3.562 | **2.371** | **1.50x** | **1.59x** |
| `layer3_m1024` | 4.862 | 5.713 | **3.611** | **1.58x** | **1.35x** |
| `layer3_m2048` | 7.640 | 9.914 | **5.736** | **1.73x** | **1.33x** |
| `layer39_m2048` | 7.456 | 9.939 | **5.792** | **1.72x** | **1.29x** |

### Needle-in-haystack benchmarks (`C = 1, 2, 4, 8, 32` across `8k/1k`, `242k/1k`, `950k/1k`)

#### `Qwen3.8-27B` INT8 (`qwen_int8`), 2x RTX A5000 (`SM86`, 24 GB/GPU)

Measured in vLLM (`consumer_int8_prefill`, `consumer_fa2` with 64k-token
bucketed BF16 unpack, `paged_decode` / `paged_decode_int4` 2-CTA/SM split-KV,
`int8` KV cache for `C=1..8` at `8k/1k` and `int4_per_token_head` KV cache for
`242k/1k` and `C=32`, `2.0–3.2 GiB` KV reservation on `MTP=0` and `3.35 GiB` on
`MTP=7`, cold prefix cache; `242,000 / 1,000` at `C >= 2` exceeds the
`2.98 GiB` post-weight per-GPU VRAM ceiling on 2x 24 GB; `950,000 / 1,000`
exceeds `max_position_embeddings = 262,144`):

| Topology | Input / Output tokens | C | Prefill tok/s per stream | Prefill tok/s aggregate | Decode tok/s per stream | Decode tok/s aggregate | Verbatim |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TP=2` (`MTP=0`) | 8,000 / 1,000 | 1 | 4,280 | 4,280 | 36.7 | 36.7 | 3/3 |
| `TP=2` (`MTP=0`) | 8,000 / 1,000 | 2 | 3,496 | 4,181 | 32.8 | 63.5 | 5/6 |
| `TP=2` (`MTP=0`) | 8,000 / 1,000 | 4 | 2,078 | 4,145 | 29.4 | 108.0 | 12/12 |
| `TP=2` (`MTP=0`) | 8,000 / 1,000 | 8 | 1,827 | 3,722 | 22.7 | 153.8 | 24/24 |
| `TP=2` (`MTP=0`, `int4_per_token_head`) | 8,000 / 1,000 | 32 | 1,981 | 3,759 | 12.9 | 216.0 | 31/32 |
| `TP=2` (`MTP=0`, `int4_per_token_head`) | 242,000 / 1,000 | 1 | 1,743 | 1,743 | 26.2 | 26.2 | 1/1 |
| `TP=2` (`MTP=0`, `int4_per_token_head`) | 242,000 / 1,000 | 2, 4, 8, 32 | OOM | OOM | OOM | OOM | OOM |
| `TP=2` (`MTP=0`) | 950,000 / 1,000 | 1, 2, 4, 8, 32 | N/A | N/A | N/A | N/A | N/A |
| `TP=2` (`MTP=7`) | 8,000 / 1,000 | 1 | 4,114 | 4,114 | 142.1 | 142.1 | 3/3 |
| `TP=2` (`MTP=7`) | 8,000 / 1,000 | 2 | 3,034 | 4,048 | 122.7 | 215.7 | 5/6 |
| `TP=2` (`MTP=7`) | 8,000 / 1,000 | 4 | 2,003 | 4,000 | 89.3 | 272.6 | 12/12 |
| `TP=2` (`MTP=7`) | 8,000 / 1,000 | 8, 32 | OOM | OOM | OOM | OOM | OOM |
| `TP=2` (`MTP=7`) | 242,000 / 1,000 | 1, 2, 4, 8, 32 | OOM | OOM | OOM | OOM | OOM |
| `TP=2` (`MTP=7`) | 950,000 / 1,000 | 1, 2, 4, 8, 32 | N/A | N/A | N/A | N/A | N/A |
| `PP=2` (`MTP=0`) | 8,000 / 1,000 | 1 | 3,310 | 3,310 | 20.0 | 20.0 | 3/3 |
| `PP=2` (`MTP=0`) | 8,000 / 1,000 | 2 | 2,836 | 3,260 | 17.9 | 35.0 | 5/6 |
| `PP=2` (`MTP=0`) | 8,000 / 1,000 | 4 | 1,895 | 4,251 | 17.7 | 67.7 | 12/12 |
| `PP=2` (`MTP=0`) | 8,000 / 1,000 | 8 | 1,675 | 3,851 | 14.5 | 105.1 | 24/24 |
| `PP=2` (`MTP=0`, `int4_per_token_head`) | 8,000 / 1,000 | 32 | 1,725 | 3,873 | 9.6 | 161.6 | 31/32 |
| `PP=2` (`MTP=0`, `int4_per_token_head`) | 242,000 / 1,000 | 1 | 903 | 903 | 14.8 | 14.8 | 1/1 |
| `PP=2` (`MTP=0`, `int4_per_token_head`) | 242,000 / 1,000 | 2, 4, 8, 32 | OOM | OOM | OOM | OOM | OOM |
| `PP=2` (`MTP=0`) | 950,000 / 1,000 | 1, 2, 4, 8, 32 | N/A | N/A | N/A | N/A | N/A |

#### `unsloth/DeepSeek-V4.1-Flash-1017-GGUF` (`UD-Q2_K_XL`), 2x DGX Spark GB10 (`TP=2/EP=2`)

Measured on `leaderworkerset/deepseek-v4-int4-int8` (`spark-2ab3` +
`spark-5867`, `quant_method=dsv41_gguf`, `UD-Q2_K_XL` rev
`60ad3651641b5c97679c807e064ab1d7a3ca1455`, `int8_ds_mla` KV cache + INT8
Lightning Indexer, `1.0–2.0 GiB` per rank = up to `1,276,734` tokens capacity
with `94.24–95.11 GiB` weights per rank inside the `121 GiB` unified memory
limit, bounded-slab `streaming_prefill_topk`, PIECEWISE CUDA graphs
`[1,2,4,8]` / `[1,2,4,8,16,32]`, cold prefix cache per cell,
`tools/ampere/dsv4_needle_bench.py`):

| Input / Output tokens | C | TTFT s | Prefill tok/s per stream | Prefill tok/s aggregate | Output tok/s per stream | Output tok/s aggregate | Verbatim |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,000 / 1,000 | 1 | 4.0 | 1,988 | 1,988 | 19.3 | 17.9 | 1/1 |
| 8,000 / 1,000 | 2 | 6.2 | 1,429 | 1,943 | 16.3 | 29.6 | 2/2 |
| 8,000 / 1,000 | 4 | 10.7 | 783 | 1,918 | 13.2 | 46.1 | 3/4 |
| 8,000 / 1,000 | 8 | 19.2 | 422 | 1,907 | 10.2 | 67.8 | 8/8 |
| 8,000 / 1,000 | 32 | 98.6 | 81 | 1,346 | 4.1 | 92.6 | 31/32 |
| 242,000 / 1,000 | 1 | 196.5 | 1,232 | 1,232 | 18.4 | 4.0 | 1/1 |
| 242,000 / 1,000 | 2, 4, 8, 32 | OOM | OOM | OOM | OOM | OOM | OOM |
| 950,000 / 1,000 | 1 | 952.2 | 998 | 998 | 16.0 | 1.0 | 1/1 |
| 950,000 / 1,000 | 2, 4, 8, 32 | OOM | OOM | OOM | OOM | OOM | OOM |

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

[HyperQwen's INT8 prefill patch](https://github.com/syv-ai/HyperQwen/blob/95ee3b64b0df7014ee2626529524d6b99c3dcdd1/patches/prefill-attn-int8.patch)
is another useful reference: INT8 QK and BF16 PV over paged BF16 KV. Its
serving gate is one sequence with 24 query heads, four KV heads and dimension
256. Adapting it would require this package's packed INT8 cache and TP=2 head
geometry, followed by matching quality and serving measurements. Its INT4
weight recipes and optional FP16 recurrent state are different formats.
