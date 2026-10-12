# SPDX-License-Identifier: Apache-2.0
"""Chunked CUDA recurrence versus vLLM FLA, including nonzero initial state."""

import pytest
import torch

from consumer_int8 import ops


@pytest.mark.parametrize("tokens", [64, 256, 1024])
@pytest.mark.parametrize("hq,hv", [(8, 24), (16, 48)])
def test_chunked_gdn_output_and_state_match_fla(tokens, hq, hv):
    from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule

    torch.manual_seed(732)
    # Qwen's fused post-convolution preparation already normalizes Q/K and
    # rounds them to BF16 before the chunked attention operator.
    q = torch.randn(tokens, hq, 128, device="cuda")
    k = torch.randn_like(q)
    q = (q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()).bfloat16()
    k = (k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()).bfloat16()
    v = torch.randn(tokens, hv, 128, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(tokens, hv, device="cuda") * 0.05
    beta = torch.rand_like(g)
    initial = torch.randn(hv, 128, 128, device="cuda") * 0.01
    initial_copy = initial.clone()
    final = torch.empty_like(initial)
    out = torch.empty_like(v)
    cumulative = torch.empty_like(g)
    w, u, v_new = (torch.empty_like(v) for _ in range(3))
    h_chunk = torch.empty(
        tokens // 64, hv, 128, 128, device="cuda", dtype=torch.bfloat16
    )
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    def run():
        ops.gdn_prefill(
            q,
            k,
            v,
            g,
            beta,
            initial,
            final,
            out,
            cumulative,
            w,
            u,
            v_new,
            h_chunk,
            num_sms,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for zero_state in (False, True):
        if zero_state:
            initial.zero_()
        graph.replay()
        expected, expected_state = chunk_gated_delta_rule(
            q[None],
            k[None],
            v[None],
            g[None],
            beta[None],
            initial_state=initial[None],
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
        torch.testing.assert_close(out, expected[0], atol=0.002, rtol=0.02)
        torch.testing.assert_close(final, expected_state[0], atol=0.01, rtol=0.02)
        for actual, reference in ((out, expected[0]), (final, expected_state[0])):
            relative_error = (
                actual.float() - reference.float()
            ).norm() / reference.float().norm()
            assert relative_error.item() < 0.007
        torch.testing.assert_close(
            initial,
            torch.zeros_like(initial) if zero_state else initial_copy,
            atol=0,
            rtol=0,
        )
