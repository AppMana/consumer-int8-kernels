# SPDX-License-Identifier: Apache-2.0
"""Check integer products, recurrent state transitions and CUDA graph replay."""

import pytest
import torch

from consumer_int8 import ops


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


def test_quantize_rows_preserves_zero_rows_and_round_to_even():
    x = torch.tensor(
        [[0, -127, 63.5, 127], [0, 0, 0, 0]], device="cuda", dtype=torch.bfloat16
    )
    q = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(2, device="cuda")
    ops.quantize_rows(x, q, scales)
    assert q.cpu().tolist() == [[0, -127, 64, 127], [0, 0, 0, 0]]
    assert scales[0].item() == 1
    assert torch.isfinite(scales).all() and (scales > 0).all()


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


@pytest.mark.parametrize("m", [1, 17, 64, 256])
def test_cutlass_schedules_match_integer_products_and_capture(m):
    a = torch.randint(-127, 128, (m, 512), device="cuda", dtype=torch.int8)
    w = torch.randint(-127, 128, (128, 512), device="cuda", dtype=torch.int8)
    sa = torch.rand(m, device="cuda") / 127
    sw = (torch.rand(128, device="cuda") / 127).bfloat16()
    workspace = torch.empty(m, 128, device="cuda", dtype=torch.int32)
    out = torch.empty(m, 128, device="cuda", dtype=torch.bfloat16)
    ops.cutlass_channel_mm(a, w, sa, sw, workspace, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.cutlass_channel_mm(a, w, sa, sw, workspace, out)
    graph.replay()
    integer = a.cpu().int() @ w.cpu().int().T
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
