# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced token likelihoods with strict prompt identity checks."""

import math


def prompt_nll(response: dict, token_ids: list[int]) -> list[float]:
    choices = response.get("choices", [])
    if len(choices) != 1 or len(token_ids) < 2:
        raise ValueError("Expected one completion and at least two prompt tokens")
    choice = choices[0]
    if choice.get("prompt_token_ids") != token_ids:
        raise ValueError("Server changed the evaluation token IDs")
    if response.get("usage", {}).get("prompt_tokens") != len(token_ids):
        raise ValueError("Server prompt usage differs from the evaluation window")
    logprobs = choice.get("prompt_logprobs")
    if not isinstance(logprobs, list) or len(logprobs) != len(token_ids):
        raise ValueError("Missing or incomplete prompt logprobs")
    losses = []
    # The first token has no conditioning context and is intentionally unscored.
    for token, row in zip(token_ids[1:], logprobs[1:], strict=True):
        if not isinstance(row, dict):
            raise ValueError("Missing prompt token likelihood")
        entry = row.get(str(token), row.get(token))
        value = entry.get("logprob") if isinstance(entry, dict) else None
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value > 0:
            raise ValueError("Invalid likelihood for the actual prompt token")
        losses.append(-float(value))
    return losses


def compare_nll(reference: dict, candidate: dict, limit: float = 0.02) -> dict:
    if not math.isfinite(limit) or limit < 0:
        raise ValueError("The NLL degradation limit must be finite and nonnegative")
    if reference.get("corpus_sha256") != candidate.get("corpus_sha256"):
        raise ValueError("Reference and candidate used different evaluation windows")
    if not reference.get("corpus_sha256"):
        raise ValueError("Missing evaluation corpus identity")
    ref_rows, got_rows = reference.get("rows", []), candidate.get("rows", [])
    for result in (reference, candidate):
        expected = result.get("expected_windows")
        if (
            result.get("complete") is not True
            or not isinstance(expected, int)
            or expected < 2
            or len(result.get("rows", [])) != expected
        ):
            raise ValueError("Incomplete evaluation: every planned window must finish")
    if not ref_rows or len(ref_rows) != len(got_rows):
        raise ValueError("Both models must score every evaluation window")
    totals = [0.0, 0.0]
    count = 0
    for ref, got in zip(ref_rows, got_rows, strict=True):
        if ref["token_ids_sha256"] != got["token_ids_sha256"]:
            raise ValueError("Evaluation window token IDs differ")
        if ref["scored_tokens"] != got["scored_tokens"] or ref["scored_tokens"] <= 0:
            raise ValueError("Evaluation window token counts differ or are empty")
        count += ref["scored_tokens"]
        for index, row in enumerate((ref, got)):
            if not math.isfinite(row["nll_sum"]) or row["nll_sum"] < 0:
                raise ValueError("Invalid NLL sum")
            totals[index] += row["nll_sum"]
    ref_nll, got_nll = (total / count for total in totals)
    delta = got_nll - ref_nll
    return dict(
        scored_tokens=count,
        reference_mean_nll=ref_nll,
        candidate_mean_nll=got_nll,
        mean_nll_delta=delta,
        mean_nll_delta_limit=limit,
        passed=delta <= limit,
    )
