# SPDX-License-Identifier: Apache-2.0
"""Check integer products, recurrent state transitions and CUDA graph replay."""

import pytest
import torch

from consumer_int8 import ops


@pytest.mark.parametrize("hq,hv", [(8, 24), (16, 48)])
@pytest.mark.parametrize("layout", ["DS", "SD"])
@pytest.mark.parametrize("bias_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("long_sequence", [65, 4097])
def test_fused_conv_preserves_ragged_history_and_padded_slots(
    hq, hv, layout, bias_dtype, long_sequence
):
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
    from vllm.third_party.flash_linear_attention.ops import fused_post_conv_prep

    torch.manual_seed(631)
    lengths = [1, 2, 17, long_sequence, 0]
    tokens, width = sum(lengths), (2 * hq + hv) * 128
    x = torch.randn(tokens, width + 128, device="cuda", dtype=torch.bfloat16)[:, :width]
    weight = torch.randn(width, 4, device="cuda", dtype=x.dtype) * 0.2
    state = torch.randn(7, width, 3, device="cuda", dtype=x.dtype)
    if layout == "SD":
        state = state.transpose(1, 2).contiguous().transpose(1, 2)
    original = state.clone()
    expected_state = state.clone()
    offsets = torch.tensor(
        [0, 1, 3, 20, tokens, tokens], device="cuda", dtype=torch.int32
    )
    slot_storage = torch.tensor(
        [4, 0, 2, 0, 1, 0, 3, 0, 0, 0], device="cuda", dtype=torch.int32
    )
    slots = slot_storage[::2]
    initial = torch.tensor([True, True, False, True, False], device="cuda")
    gates = torch.randn(tokens, hv * 2, device="cuda", dtype=x.dtype)
    a, b = gates[:, hv:], gates[:, :hv]
    log_a = torch.randn(hv, device="cuda")
    dt = torch.randn(hv, device="cuda", dtype=bias_dtype)
    q, k = (
        torch.empty(tokens, hq, 128, device="cuda", dtype=x.dtype) for _ in range(2)
    )
    v = torch.empty(tokens, hv, 128, device="cuda", dtype=x.dtype)
    g, beta = (torch.empty(tokens, hv, device="cuda") for _ in range(2))

    def run():
        ops.gdn_conv(
            x,
            a,
            b,
            log_a,
            dt,
            q,
            k,
            v,
            g,
            beta,
            weight,
            state,
            offsets,
            slots,
            initial,
            max(lengths),
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    state.copy_(original)
    for step in range(3):
        if step == 2:
            initial.fill_(True)
            x.normal_()
            # Same graph must consume changed device metadata.
            offsets[3] = 19
        graph.replay()
        conv = causal_conv1d_fn(
            x.T, weight, None, expected_state, offsets, slots, initial
        ).T
        expected = fused_post_conv_prep(conv, a, b, log_a, dt, hq, 128, 128)
        for actual, target in zip((q, k, v, g, beta), expected, strict=True):
            torch.testing.assert_close(actual, target, atol=0.004, rtol=0.008)
        torch.testing.assert_close(state, expected_state, atol=0, rtol=0)
        for slot in (0, 5, 6):
            torch.testing.assert_close(state[slot], original[slot], atol=0, rtol=0)


@pytest.mark.parametrize("tokens,hq,hv", [(0, 8, 24), (17, 8, 24), (1024, 16, 48)])
@pytest.mark.parametrize("bias_dtype", [torch.bfloat16, torch.float32])
def test_post_conv_preserves_strided_inputs_and_graph_replay(
    tokens, hq, hv, bias_dtype
):
    from vllm.third_party.flash_linear_attention.ops import fused_post_conv_prep

    torch.manual_seed(717)
    width = (2 * hq + hv) * 128
    x = torch.randn(tokens, width + 128, device="cuda", dtype=torch.bfloat16)[:, :width]
    gates = torch.randn(tokens, hv * 2, device="cuda", dtype=torch.bfloat16)
    a, b = gates[:, hv:], gates[:, :hv]
    log_a = torch.randn(hv, device="cuda")
    dt = torch.linspace(-30, 30, hv, device="cuda", dtype=bias_dtype)
    q = torch.empty(tokens, hq, 128, device="cuda", dtype=torch.bfloat16)
    k, v = (
        torch.empty_like(q),
        torch.empty(tokens, hv, 128, device="cuda", dtype=q.dtype),
    )
    g = torch.empty(tokens, hv, device="cuda")
    beta = torch.empty_like(g)

    def run():
        ops.gdn_post_conv(x, a, b, log_a, dt, q, k, v, g, beta)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for scale in (0, 0.7, 3):
        x.normal_().mul_(scale)
        gates.normal_().mul_(scale)
        graph.replay()
        expected = fused_post_conv_prep(x, a, b, log_a, dt, hq, 128, 128)
        for actual, target in zip((q, k, v, g, beta), expected, strict=True):
            if actual.dtype == torch.bfloat16:
                torch.testing.assert_close(actual, target, atol=0.002, rtol=0.008)
            else:
                torch.testing.assert_close(actual, target, atol=2e-5, rtol=2e-6)
        torch.testing.assert_close(v, expected[2], atol=0, rtol=0)


@pytest.mark.parametrize("tokens", [0, 1, 17, 4096])
@pytest.mark.parametrize("with_residual", [False, True])
def test_gemma_norm_preserves_unrounded_sum_and_bf16_residual(tokens, with_residual):
    torch.manual_seed(927)
    x = torch.randn(tokens, 5120, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x) if with_residual else None
    weight = torch.randn(5120, device="cuda", dtype=x.dtype)
    out = torch.empty_like(x)
    residual_out = torch.empty_like(x) if with_residual else None

    def run():
        ops.gemma_norm(x, weight, residual, out, residual_out, 1e-6)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for scale in (0, 1, 100):
        x.normal_().mul_(scale)
        graph.replay()
        summed = x.float() + residual.float() if with_residual else x.float()
        normalized = summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-6)
        expected = (normalized * (weight.float() + 1)).bfloat16()
        torch.testing.assert_close(out, expected, atol=0.001, rtol=0.008)
        if with_residual:
            torch.testing.assert_close(residual_out, summed.bfloat16(), atol=0, rtol=0)


@pytest.mark.parametrize("batch,hq,hv", [(1, 16, 48), (2, 8, 24), (4, 16, 48)])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("bias_dtype", [torch.bfloat16, torch.float32])
def test_packed_gdn_matches_serving_kernel_with_strided_inputs(
    batch, hq, hv, slot_dtype, bias_dtype
):
    from vllm.third_party.flash_linear_attention.ops import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )

    torch.manual_seed(872)
    width = (2 * hq + hv) * 128
    qkv = torch.randn(batch, width + 128, device="cuda", dtype=torch.bfloat16)
    qkv = qkv[:, :width]
    ba = torch.randn(batch, 2 * hv, device="cuda", dtype=torch.bfloat16)
    b, a = ba[:, :hv], ba[:, hv:]
    log_a = torch.randn(hv, device="cuda", dtype=torch.float32) - 1
    bias = torch.randn(hv, device="cuda", dtype=bias_dtype)
    # The leading state stride may include padding from a cache allocation.
    state = torch.randn(batch + 1, hv + 1, 128, 128, device="cuda")[:, :hv]
    original = state.clone()
    reference = state.clone()
    slot_storage = torch.zeros(batch * 2, device="cuda", dtype=slot_dtype)
    slots = slot_storage[::2]
    slots.copy_(torch.arange(batch, 0, -1, device="cuda", dtype=slot_dtype))
    output = torch.empty(batch, hv, 128, device="cuda", dtype=torch.bfloat16)
    expected = torch.empty_like(output)
    ops.gdn_packed_decode(qkv, a, b, log_a, bias, state, slots, output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.gdn_packed_decode(qkv, a, b, log_a, bias, state, slots, output)
    state.copy_(original)
    for step in range(3):
        if step == 2:
            slots[-1] = 0
        graph.replay()
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=qkv,
            a=a,
            b=b,
            A_log=log_a,
            dt_bias=bias,
            scale=128**-0.5,
            initial_state=reference,
            out=expected.unsqueeze(1),
            ssm_state_indices=slots,
            use_qk_l2norm_in_kernel=True,
        )
        torch.testing.assert_close(state, reference, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(output, expected, atol=0.002, rtol=0.008)
    torch.testing.assert_close(state[0], original[0], atol=0, rtol=0)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "query_lens,seq_lens,heads,decode",
    [
        ([33], [97], 2, False),
        ([1, 17, 49, 4], [65, 255, 511, 99], 4, False),
        ([1], [8193], 2, True),
        ([1, 2], [1, 3], 2, True),
        ([1, 3, 8, 0], [257, 2049, 8001, 0], 4, True),
    ],
)
@pytest.mark.parametrize("layout", ["HND", "NHD"])
def test_paged_attention_handles_gqa_tails_and_graph_metadata(
    query_lens, seq_lens, heads, decode, layout
):
    """D256 INT8 attention must preserve causal masking and changing page tables."""
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    from vllm.v1.kv_cache_interface import KVQuantMode

    torch.manual_seed(761)
    tokens, block_size = sum(query_lens), 32
    pages_per_seq = (max(seq_lens) + block_size - 1) // block_size
    blocks = len(seq_lens) * pages_per_seq
    shape = (blocks, heads, block_size, 520)
    cache = torch.empty(
        shape if layout == "HND" else (blocks, block_size, heads, 520),
        device="cuda",
        dtype=torch.int8,
    )
    if layout == "NHD":
        cache = cache.transpose(1, 2)
    key = torch.randn(
        blocks * block_size, heads, 256, device="cuda", dtype=torch.bfloat16
    )
    value = torch.randn_like(key)
    ops.kv_cache_update(key, value, cache, torch.arange(key.shape[0], device="cuda"))
    table = torch.randperm(blocks, device="cuda", dtype=torch.int32).reshape(
        len(seq_lens), -1
    )
    offsets = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    lengths = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)
    # Qwen's query is a strided view into its gated projection.
    query = torch.randn(tokens, heads * 12, 256, device="cuda", dtype=torch.bfloat16)[
        :, : heads * 6
    ]
    out, expected = torch.empty_like(query), torch.empty_like(query)
    views = cache.transpose(1, 2)

    scratch = torch.empty((tokens, heads * 6, 32, 258), device="cuda")

    def native():
        if decode:
            ops.paged_decode(
                query,
                cache,
                table,
                offsets,
                lengths,
                out,
                max(query_lens),
                1 / 16,
                scratch,
            )
        else:
            ops.paged_attention(
                query, cache, table, offsets, lengths, out, max(query_lens), 1 / 16
            )

    native()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        native()
    for _ in range(2):
        out.fill_(float("nan"))
        graph.replay()
        unified_attention(
            q=query,
            k=views[..., :256],
            v=views[..., 260:516],
            out=expected,
            cu_seqlens_q=offsets,
            max_seqlen_q=max(query_lens),
            seqused_k=lengths,
            max_seqlen_k=max(seq_lens),
            softmax_scale=1 / 16,
            causal=True,
            window_size=(-1, -1),
            block_table=table,
            softcap=0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
            k_scale_cache=views[..., 256:260].view(torch.float32).squeeze(-1),
            v_scale_cache=views[..., 516:520].view(torch.float32).squeeze(-1),
        )
        torch.testing.assert_close(out, expected, atol=0.002, rtol=0.008)
        table.copy_(table.flip(1))
        query.mul_(0.75)


@pytest.mark.parametrize("n,k", [(16, 16), (48, 80), (128, 512), (5120, 3072)])
def test_allspark_unpack_preserves_every_code_and_scale_in_graph(n, k):
    from vllm import _custom_ops as vops

    generator = torch.Generator(device="cuda").manual_seed(319)
    codes = torch.randint(
        0, 256, (n, k), device="cuda", dtype=torch.uint8, generator=generator
    )
    scales = torch.randn(n, device="cuda", dtype=torch.bfloat16, generator=generator)
    packed, packed_scales, _ = vops.allspark_repack_weight(
        codes.T.contiguous(), scales[None], None, False
    )
    restored = torch.empty_like(codes, dtype=torch.int8)
    restored_scales = torch.empty_like(scales)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.unpack_allspark(packed, packed_scales.flatten(), restored, restored_scales)
    for _ in range(2):
        restored.zero_()
        restored_scales.zero_()
        graph.replay()
        torch.testing.assert_close(
            restored, (codes.short() - 128).char(), atol=0, rtol=0
        )
        torch.testing.assert_close(restored_scales, scales, atol=0, rtol=0)
        codes.bitwise_xor_(255)
        scales.neg_()
        changed, changed_scales, _ = vops.allspark_repack_weight(
            codes.T.contiguous(), scales[None], None, False
        )
        packed.copy_(changed)
        packed_scales.copy_(changed_scales)


@pytest.mark.parametrize("m,n,k", [(1, 5120, 5120), (4, 2560, 3072), (17, 37, 128)])
def test_channel_mm_matches_integer_product_with_output_tails(m, n, k):
    generator = torch.Generator(device="cuda").manual_seed(17)
    a = torch.randint(
        -127, 128, (m, k), device="cuda", dtype=torch.int8, generator=generator
    )
    w = torch.randint(
        1, 256, (n, k), device="cuda", dtype=torch.uint8, generator=generator
    )
    sa = torch.rand(m, device="cuda", generator=generator) / 127
    sw = (torch.rand(n, device="cuda", generator=generator) / 127).bfloat16()
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    ops.channel_mm(a, w, sa, sw, out)
    # CPU integer matmul is independent of the CUDA MMA lane mapping.
    integer = a.cpu().int() @ (w.cpu().int() - 128).T
    expected = (integer.float().cuda() * sa[:, None] * sw.float()[None, :]).bfloat16()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("width", [4, 3072, 5120, 6144, 8704, 17408])
@pytest.mark.parametrize("rows,misaligned", [(2, False), (1024, False), (1024, True)])
def test_quantize_rows_preserves_zero_rows_and_round_to_even(width, rows, misaligned):
    torch.manual_seed(742)
    storage = torch.randn(rows * width + 1, device="cuda", dtype=torch.bfloat16)
    x = storage[int(misaligned) : rows * width + int(misaligned)].view(rows, width)
    x.mul_(100).clamp_(-127, 127)
    x[:, 0] = -127
    x[0, :4] = torch.tensor([0, -127, 63.5, 127], device="cuda")
    x[1].zero_()
    q = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(rows, device="cuda")
    ops.quantize_rows(x, q, scales)
    expected_scale = torch.ones_like(scales)
    expected_scale[1] = 1e-20
    torch.testing.assert_close(scales, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(q, x.float().round().to(torch.int8), rtol=0, atol=0)


@pytest.mark.parametrize("batch,hq,hv", [(1, 16, 48), (2, 8, 24), (4, 16, 48)])
def test_gdn_updates_fp32_state_for_reordered_slots_and_graph_replay(batch, hq, hv):
    generator = torch.Generator(device="cuda").manual_seed(28)

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=generator)

    q, k = randn(batch, hq, 128), randn(batch, hq, 128)
    v = randn(batch, hv, 128)
    g = -randn(batch, hv, dtype=torch.float32).abs()
    beta = randn(batch, hv, dtype=torch.float32).sigmoid()
    state = randn(batch + 2, hv, 128, 128, dtype=torch.float32)
    slots = torch.arange(batch, 0, -1, device="cuda", dtype=torch.int64)
    out = torch.empty_like(v)
    expected_state = state.clone()
    # Warm up the op before capture, then restore state before both executions.
    original = state.clone()
    ops.gdn_decode(q, k, v, g, beta, state, slots, out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.gdn_decode(q, k, v, g, beta, state, slots, out)
    state.copy_(original)
    for step in range(3):
        if step == 2:
            # Read new metadata and zero padded rows without writing slot 0.
            slots[-1] = 0
        graph.replay()
        normalized_q = (
            q.float() / (q.float().square().sum(-1, keepdim=True) + 1e-6).sqrt()
        )
        normalized_k = (
            k.float() / (k.float().square().sum(-1, keepdim=True) + 1e-6).sqrt()
        )
        normalized_q = normalized_q.repeat_interleave(hv // hq, 1) / (128**0.5)
        normalized_k = normalized_k.repeat_interleave(hv // hq, 1)
        expected = torch.zeros_like(v)
        for row, slot in enumerate(slots.cpu().tolist()):
            if slot == 0:
                continue
            s = expected_state[slot] * g[row].exp()[:, None, None]
            delta = (
                v[row].float() - (s * normalized_k[row, :, None, :]).sum(-1)
            ) * beta[row, :, None]
            s = s + delta[..., None] * normalized_k[row, :, None, :]
            expected_state[slot] = s
            expected[row] = (s * normalized_q[row, :, None, :]).sum(-1).bfloat16()
        torch.testing.assert_close(state, expected_state, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(out, expected, atol=0.002, rtol=0.008)


def test_channel_mm_rejects_unaligned_k_before_launch():
    with pytest.raises(RuntimeError, match="geometry"):
        ops.channel_mm(
            torch.zeros(1, 33, dtype=torch.int8, device="cuda"),
            torch.zeros(4, 33, dtype=torch.uint8, device="cuda"),
            torch.ones(1, device="cuda"),
            torch.ones(4, dtype=torch.bfloat16, device="cuda"),
            torch.empty(1, 4, dtype=torch.bfloat16, device="cuda"),
        )


@pytest.mark.parametrize("m", [1, 17, 64, 256, 1024, 1041, 2048, 2057])
@pytest.mark.parametrize("fused", [None, 0, 1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("n,k", [(128, 512), (132, 544)])
def test_gemm_schedules_match_integer_products_and_capture(m, fused, n, k):
    a = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
    w = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    sa = torch.rand(m, device="cuda") / 127
    sw = (torch.rand(n, device="cuda") / 127).bfloat16()
    workspace = torch.empty(m, n, device="cuda", dtype=torch.int32)
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

    def run():
        if fused is not None:
            ops.fused_channel_mm(a, w, sa, sw, out, fused)
        else:
            ops.cutlass_channel_mm(a, w, sa, sw, workspace, out)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    integer = a.cpu().int() @ w.cpu().int().T
    if fused is None:
        torch.testing.assert_close(workspace.cpu(), integer, atol=0, rtol=0)
    expected = (integer.float().cuda() * sa[:, None] * sw.float()[None, :]).bfloat16()
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("heads", [2, 4])
def test_cache_writer_matches_per_head_encoding_and_preserves_padding(layout, heads):
    backing = torch.randn(7, 3, heads, 256, device="cuda", dtype=torch.bfloat16)
    key, value = backing[:, 0], backing[:, 2]  # noncontiguous token strides
    key[0].zero_()
    value[0].zero_()
    shape = (4, heads, 4, 520) if layout == "HND" else (4, 4, heads, 520)
    cache = torch.full(shape, 55, device="cuda", dtype=torch.int8)
    if layout == "NHD":
        cache = cache.transpose(1, 2)
    slots = torch.tensor([0, 15, 3, -1, 7, 12, 8], device="cuda", dtype=torch.int64)
    ops.kv_cache_update(key, value, cache, slots)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.kv_cache_update(key, value, cache, slots)
    graph.replay()
    touched = set()
    for row, slot in enumerate(slots.cpu().tolist()):
        if slot < 0:
            continue
        block, offset = divmod(slot, 4)
        touched.add((block, offset))
        for source, start in [(key, 0), (value, 260)]:
            target = cache[block, :, offset, start : start + 260]
            scale = source[row].float().abs().amax(-1).div(127).clamp_min(1e-6)
            stored_scale = target[:, 256:].view(torch.float32).squeeze(-1)
            torch.testing.assert_close(stored_scale, scale, rtol=1e-7, atol=1e-9)
            restored = target[:, :256].float() * stored_scale[:, None]
            error = (restored - source[row].float()).abs()
            assert (error <= stored_scale[:, None] * 0.5001 + 1e-6).all()
    for block in range(4):
        for offset in range(4):
            if (block, offset) not in touched:
                assert (cache[block, :, offset] == 55).all()


@pytest.mark.parametrize("heads,layout", [(2, "HND"), (4, "NHD")])
def test_unpack_kv_cache_preserves_scales_pages_and_graph_updates(heads, layout):
    """FlashAttention's BF16 scratch pages must reflect current device metadata."""
    torch.manual_seed(181)
    block_size, blocks, capacity = 32, 32, 512
    shape = (blocks, heads, block_size, 520)
    cache = torch.empty(
        shape if layout == "HND" else (blocks, block_size, heads, 520),
        device="cuda",
        dtype=torch.int8,
    )
    if layout == "NHD":
        cache = cache.transpose(1, 2)
    source_key = torch.randn(
        blocks * block_size, heads, 256, device="cuda", dtype=torch.bfloat16
    )
    source_value = torch.randn_like(source_key)
    ops.kv_cache_update(
        source_key,
        source_value,
        cache,
        torch.arange(blocks * block_size, device="cuda"),
    )
    table = torch.randperm(blocks, device="cuda", dtype=torch.int32).reshape(2, 16)
    lengths = torch.tensor([97, 499], device="cuda", dtype=torch.int32)
    key = torch.full(
        (2, capacity, heads, 256), -100.0, device="cuda", dtype=torch.bfloat16
    )
    value = torch.full_like(key, -100.0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.unpack_kv_cache(cache, table, lengths, key, value)
    for seq_lengths in ([97, 499], [33, 255]):
        lengths.copy_(torch.tensor(seq_lengths, device="cuda", dtype=torch.int32))
        key.fill_(-100.0)
        value.fill_(-100.0)
        graph.replay()
        for seq, length in enumerate(seq_lengths):
            packed = (
                cache[table[seq].long()]
                .transpose(1, 2)
                .reshape(-1, heads, 520)[:length]
            )
            for got, codes, scale in (
                (key[seq, :length], packed[..., :256], packed[..., 256:260]),
                (value[seq, :length], packed[..., 260:516], packed[..., 516:520]),
            ):
                expected = (
                    codes.float() * scale.contiguous().view(torch.float32)
                ).bfloat16()
                torch.testing.assert_close(got, expected, atol=0, rtol=0)
            assert (key[seq, length:] == -100.0).all()
            assert (value[seq, length:] == -100.0).all()
        table.copy_(table.flip(1))


@pytest.mark.parametrize("heads", [2, 4])
@pytest.mark.parametrize("schedule", range(11))
def test_fa2_prefill_preserves_ragged_causality_and_graph_metadata(heads, schedule):
    """Tile changes must preserve bottom-right masks, striding and device lengths."""
    torch.manual_seed(1971)
    query_lens, seq_lens = [7, 65, 129], [19, 133, 257]
    tokens, capacity = sum(query_lens), 576
    q = torch.randn(tokens, heads * 12, 256, device="cuda", dtype=torch.bfloat16)[
        :, : heads * 6
    ]
    k = torch.randn(3, capacity, heads, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    out = torch.empty(tokens, heads * 12, 256, device="cuda", dtype=torch.bfloat16)[
        :, : heads * 6
    ]
    lse = torch.empty(heads * 6, tokens, device="cuda", dtype=torch.float32)
    offsets = torch.tensor([0, 7, 72, tokens], device="cuda", dtype=torch.int32)
    lengths = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)

    def call():
        ops.fa2_prefill(q, k, v, offsets, lengths, out, lse, 136, 1 / 16, schedule)

    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for counts, lens in [(query_lens, seq_lens), ([0, 65, 136], [0, 147, 281])]:
        offsets.copy_(
            torch.tensor(
                [0, counts[0], sum(counts[:2]), tokens],
                device="cuda",
                dtype=torch.int32,
            )
        )
        lengths.copy_(torch.tensor(lens, device="cuda", dtype=torch.int32))
        out.fill_(float("nan"))
        lse.fill_(float("nan"))
        graph.replay()
        if schedule in (9, 10):
            # Scheduling alone must not alter the attention arithmetic.
            reference = torch.empty_like(out)
            reference_lse = torch.empty_like(lse)
            ops.fa2_prefill(
                q,
                k,
                v,
                offsets,
                lengths,
                reference,
                reference_lse,
                136,
                1 / 16,
                1 if schedule == 9 else 8,
            )
            torch.testing.assert_close(out, reference, atol=0, rtol=0)
            torch.testing.assert_close(lse, reference_lse, atol=0, rtol=0)
        start = 0
        for i, (count, length) in enumerate(zip(counts, lens, strict=True)):
            if not count:
                continue
            queries = q[start : start + count].float().transpose(0, 1)
            keys = k[i, :length].float().repeat_interleave(6, dim=1).transpose(0, 1)
            values = v[i, :length].float().repeat_interleave(6, dim=1).transpose(0, 1)
            scores = queries @ keys.transpose(-1, -2) / 16
            causal = (
                torch.arange(length, device="cuda")[None, :]
                <= (length - count + torch.arange(count, device="cuda"))[:, None]
            )
            scores.masked_fill_(~causal, -float("inf"))
            expected = (scores.softmax(-1) @ values).transpose(0, 1).bfloat16()
            torch.testing.assert_close(
                out[start : start + count], expected, atol=0.004, rtol=0.008
            )
            torch.testing.assert_close(
                lse[:, start : start + count],
                scores.logsumexp(-1),
                atol=2e-5,
                rtol=2e-5,
            )
            start += count


@torch.compile(fullgraph=True, dynamic=True)
def _compiled_swiglu(x):
    k = x.shape[-1] // 2
    return torch.nn.functional.silu(x[:, :k]) * x[:, k:]


@pytest.mark.parametrize("rows", [0, 1, 4, 1024, 3904, 4096])
@pytest.mark.parametrize("width", [8704, 17408])
def test_swiglu_quantization_preserves_compiled_bf16_rounding(rows, width):
    torch.manual_seed(775 + rows + width)
    x = torch.randn(rows, 2 * width, device="cuda", dtype=torch.bfloat16) * 3
    if rows:
        x[0].zero_()
    activated = torch.empty(rows, width, device="cuda", dtype=x.dtype)
    codes = torch.empty(rows, width, device="cuda", dtype=torch.int8)
    scales = torch.empty(rows, device="cuda")
    ops.silu_mul(x, activated)
    ops.silu_mul_quantize(x, codes, scales)
    if not rows:
        return
    expected = _compiled_swiglu(x)
    reference_codes = torch.empty_like(codes)
    reference_scales = torch.empty_like(scales)
    ops.quantize_rows(expected, reference_codes, reference_scales)
    torch.testing.assert_close(activated, expected, atol=0, rtol=0)
    torch.testing.assert_close(codes, reference_codes, atol=0, rtol=0)
    torch.testing.assert_close(scales, reference_scales, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.silu_mul(x, activated)
        ops.silu_mul_quantize(x, codes, scales)
    x.mul_(0.75)
    graph.replay()
    expected = _compiled_swiglu(x)
    ops.quantize_rows(expected, reference_codes, reference_scales)
    torch.testing.assert_close(activated, expected, atol=0, rtol=0)
    torch.testing.assert_close(codes, reference_codes, atol=0, rtol=0)
    torch.testing.assert_close(scales, reference_scales, atol=0, rtol=0)


def test_swiglu_quantization_rejects_unaligned_input_before_vector_load():
    width = 8704
    x = torch.empty(2 * width + 1, device="cuda", dtype=torch.bfloat16)[1:]
    codes = torch.empty(1, width, device="cuda", dtype=torch.int8)
    scales = torch.empty(1, device="cuda")
    with pytest.raises(RuntimeError, match="aligned buffers"):
        ops.silu_mul_quantize(x.view(1, -1), codes, scales)
