# SPDX-License-Identifier: Apache-2.0
import pytest

from consumer_int8.benchmark import (
    audit_gpu_timing,
    interval_union_seconds,
    phase_throughput,
)


def gpu_sample(time):
    return dict(monotonic_s=time, compute_processes="", process_activity="", devices="")


def test_compute_only_recording_cannot_establish_idle_graphics():
    samples = [gpu_sample(t) for t in range(9, 24, 2)]
    for sample in samples:
        del sample["process_activity"]
    assert audit_gpu_timing(samples, [(10, 20)], []) == [
        {"error": "Incomplete compute/graphics GPU telemetry"}
    ]


def test_gpu_audit_rejects_missing_activity_during_request():
    samples = [gpu_sample(9), gpu_sample(11), gpu_sample(21)]
    assert audit_gpu_timing(samples, [(10, 20)], []) == [
        {"error": "GPU telemetry gap overlaps a request"}
    ]


def test_gpu_audit_uses_request_windows_and_keeps_interference_margin():
    samples = [gpu_sample(t) for t in range(9, 24, 2)]
    startup = dict(monotonic_s=2, process="desktop renderer")
    nearby = dict(monotonic_s=8, process="foreign CUDA process")
    assert audit_gpu_timing(samples, [(10, 20)], [startup]) == []
    assert audit_gpu_timing(samples, [(10, 20)], [startup, nearby]) == [nearby]


def test_gpu_audit_requires_samples_even_for_short_requests():
    assert audit_gpu_timing([], [(10, 11)], [])
    assert audit_gpu_timing([gpu_sample(1)], [(10, 11)], [])


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
