# SPDX-License-Identifier: Apache-2.0
import pytest

from consumer_int8.benchmark import interval_union_seconds, phase_throughput


def request(start, first, last, prompt=8000, output=1000):
    return {
        "engine_phase_timing": {
            "scheduled_s": start,
            "first_token_s": first,
            "last_token_s": last,
        },
        "prompt_tokens": prompt,
        "completion_tokens": output,
    }


def test_overlapping_phases_exclude_queue_and_cohort_wall():
    result = phase_throughput([request(10, 12, 37), request(11, 15, 40)], 50)
    assert result["prefill_tok_s_single_stream_median"] == 3000
    assert result["prefill_tok_s_aggregate"] == 3200
    assert result["decode_tok_s_single_stream_median"] == 999 / 25
    assert result["decode_tok_s_aggregate"] == 1998 / 28
    assert result["output_tok_s_over_cohort_wall"] == 40


def test_union_handles_nested_disjoint_and_unsorted_intervals():
    assert interval_union_seconds([(9, 11), (2, 6), (3, 5), (5, 7)]) == 7


def test_single_stream_aggregate_equals_single_stream_rate():
    result = phase_throughput([request(10, 12, 37)], 100)
    for phase in ("prefill", "decode"):
        assert (
            result[f"{phase}_tok_s_aggregate"]
            == result[f"{phase}_tok_s_single_stream_median"]
        )


@pytest.mark.parametrize("change", ["missing", "cached", "preempted", "unordered"])
def test_incomplete_or_contaminated_cohorts_fail_closed(change):
    row = request(10, 12, 37)
    if change == "missing":
        del row["engine_phase_timing"]
    elif change == "cached":
        row["cached_tokens"] = 64
    elif change == "preempted":
        row["engine_phase_timing"]["num_preemptions"] = 1
    else:
        row["engine_phase_timing"]["first_token_s"] = 9
    with pytest.raises(ValueError):
        phase_throughput([row], 50)
