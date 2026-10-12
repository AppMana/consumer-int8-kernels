# SPDX-License-Identifier: Apache-2.0
"""Request-phase throughput accounting; no client TTFT proxies or chunk counts."""

import math
import statistics


def interval_union_seconds(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    for start, end in intervals:
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError("Invalid phase interval")
    ordered = sorted(intervals)
    start, end = ordered[0]
    duration = 0.0
    for next_start, next_end in ordered[1:]:
        if next_start > end:
            duration += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    return duration + end - start


def phase_throughput(results: list[dict], wall_seconds: float) -> dict:
    """Summarize a same-host cohort with prefix cache disabled.

    O-1 excludes each request's first output token from decode. Aggregate
    rates divide token totals by the union of engine phase intervals, so
    concurrent requests' elapsed time is counted once. Reject incomplete
    metrics instead of substituting client TTFT or end-to-end wall time.
    """
    if not results or not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise ValueError("A nonempty cohort and positive wall time are required")
    prefill_intervals, decode_intervals = [], []
    prefill_rates, decode_rates = [], []
    prompt_total = output_total = decode_total = 0
    for result in results:
        timing = result.get("engine_phase_timing")
        if not timing:
            raise ValueError("Missing engine phase metrics; enable per-request metrics")
        scheduled, first, last = (
            timing[key] for key in ("scheduled_s", "first_token_s", "last_token_s")
        )
        if not all(math.isfinite(t) for t in (scheduled, first, last)) or not (
            0 < scheduled < first < last
        ):
            raise ValueError("Incomplete or unordered engine phase timestamps")
        if timing.get("num_preemptions", 0):
            raise ValueError(
                "Preempted request: repeat with a sufficient memory budget"
            )
        if result.get("cached_tokens", 0):
            raise ValueError(
                "Prefix cache hits invalidate this uncached-prefill cohort"
            )
        prompt, output = result.get("prompt_tokens"), result.get("completion_tokens")
        if not isinstance(prompt, int) or prompt <= 0:
            raise ValueError("Actual prompt usage is required")
        if not isinstance(output, int) or output <= 1:
            raise ValueError("At least two actual output tokens are required")
        prompt_total += prompt
        output_total += output
        decode_total += output - 1
        prefill_intervals.append((scheduled, first))
        decode_intervals.append((first, last))
        prefill_rates.append(prompt / (first - scheduled))
        decode_rates.append((output - 1) / (last - first))
    prefill_union = interval_union_seconds(prefill_intervals)
    decode_union = interval_union_seconds(decode_intervals)
    return {
        "timing_source": "same_host_engine_monotonic_request_phases",
        "prefill_tok_s_single_stream_median": statistics.median(prefill_rates),
        "prefill_tok_s_aggregate": prompt_total / prefill_union,
        "decode_tok_s_single_stream_median": statistics.median(decode_rates),
        "decode_tok_s_aggregate": decode_total / decode_union,
        "prefill_phase_union_s": prefill_union,
        "decode_phase_union_s": decode_union,
        "prompt_tokens_total": prompt_total,
        "completion_tokens_total": output_total,
        "decode_tokens_total": decode_total,
        "output_tok_s_over_cohort_wall": output_total / wall_seconds,
        "wall_s": wall_seconds,
    }
