# SPDX-License-Identifier: Apache-2.0
import copy

import pytest

from consumer_int8.quality import compare_nll, prompt_nll


def response():
    return dict(
        usage=dict(prompt_tokens=3),
        choices=[
            dict(
                prompt_token_ids=[7, 8, 9],
                prompt_logprobs=[
                    None,
                    {"8": {"logprob": -0.2}, "99": {"logprob": -0.1}},
                    {"9": {"logprob": -0.6}},
                ],
            )
        ],
    )


def test_score_actual_prompt_tokens_and_exclude_unconditioned_first_token():
    assert prompt_nll(response(), [7, 8, 9]) == [0.2, 0.6]


@pytest.mark.parametrize("fault", ["token_ids", "missing", "nan"])
def test_fail_closed_on_changed_or_missing_evaluation_likelihoods(fault):
    value = response()
    choice = value["choices"][0]
    if fault == "token_ids":
        choice["prompt_token_ids"][1] = 99
    elif fault == "missing":
        del choice["prompt_logprobs"][1]["8"]
    else:
        choice["prompt_logprobs"][1]["8"]["logprob"] = float("nan")
    with pytest.raises(ValueError):
        prompt_nll(value, [7, 8, 9])


def test_weight_by_scored_tokens_and_reject_mismatched_windows():
    reference = dict(
        corpus_sha256="corpus",
        rows=[
            dict(token_ids_sha256="a", scored_tokens=1, nll_sum=0.5),
            dict(token_ids_sha256="b", scored_tokens=9, nll_sum=9.0),
        ],
    )
    candidate = copy.deepcopy(reference)
    candidate["rows"][0]["nll_sum"] += 0.1
    result = compare_nll(reference, candidate)
    assert result["mean_nll_delta"] == pytest.approx(0.01)
    assert result["passed"]
    assert not compare_nll(reference, candidate, limit=0.005)["passed"]
    candidate["rows"][1]["token_ids_sha256"] = "changed"
    with pytest.raises(ValueError):
        compare_nll(reference, candidate)
