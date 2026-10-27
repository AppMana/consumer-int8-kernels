# SPDX-License-Identifier: Apache-2.0
"""Cold single-request prefill measurements with server-counter validation."""

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen


def fetch(base, path, body=None):
    request = Request(base.rstrip("/") + path)
    if body is not None:
        request.data = json.dumps(body).encode()
        request.add_header("Content-Type", "application/json")
    with urlopen(request, timeout=600) as response:
        return response.read().decode()


def counter(metrics, name):
    values = re.findall(
        r"^vllm:" + re.escape(name) + r"\{[^\n]*\} ([0-9.e+-]+)$", metrics, re.M
    )
    if len(values) != 1:
        raise ValueError(f"Expected one engine/model series for {name}")
    return float(values[0])


def main(args):
    with args.output.open("x") as output:
        for trial in range(args.trials):
            for length in args.lengths:
                rng = random.Random(args.seed + trial * 100000 + length)
                prompt = [rng.randrange(1000, 20000) for _ in range(length)]
                before = fetch(args.base_url, "/metrics")
                for name in ("num_requests_running", "num_requests_waiting"):
                    if counter(before, name):
                        raise RuntimeError("Endpoint is not idle")
                body = dict(
                    model=args.model,
                    prompt=prompt,
                    temperature=0,
                    max_tokens=1,
                    ignore_eos=True,
                    seed=args.seed,
                    logprobs=5,
                )
                started = time.perf_counter()
                result = json.loads(fetch(args.base_url, "/v1/completions", body))
                elapsed = time.perf_counter() - started
                after = fetch(args.base_url, "/metrics")

                def delta(name, after=after, before=before):
                    return counter(after, name) - counter(before, name)

                if delta("request_prefill_time_seconds_count") != 1:
                    raise RuntimeError(
                        "Concurrent traffic or reset invalidates the trial"
                    )
                if any(
                    delta(name)
                    for name in (
                        "prefix_cache_hits_total",
                        "external_prefix_cache_hits_total",
                    )
                ):
                    raise RuntimeError("Prefix reuse invalidates the cold trial")
                prefill = delta("request_prefill_time_seconds_sum")
                assert prefill > 0
                assert result["usage"]["prompt_tokens"] == length
                assert result["usage"]["completion_tokens"] == 1
                record = dict(
                    trial=trial,
                    length=length,
                    seed=args.seed,
                    prompt_sha256=hashlib.sha256(
                        json.dumps(prompt).encode()
                    ).hexdigest(),
                    prefill_seconds=prefill,
                    input_tokens_per_second=length / prefill,
                    elapsed_seconds=elapsed,
                    response=result,
                    metrics_before=before,
                    metrics_after=after,
                )
                output.write(json.dumps(record) + "\n")
                output.flush()
                print(
                    json.dumps(
                        {
                            k: record[k]
                            for k in (
                                "trial",
                                "length",
                                "prefill_seconds",
                                "input_tokens_per_second",
                            )
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="deepseek-v4.1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=728194)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192, 20000])
    main(parser.parse_args())
