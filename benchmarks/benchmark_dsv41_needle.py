# SPDX-License-Identifier: Apache-2.0
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from benchmark_dsv41_prefill import counter, fetch


def main(args):
    base = args.base_url.rstrip("/")

    def request(path, body=None):
        r = Request(
            base + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(r, timeout=600) as response:
            raw = response.read()
            return json.loads(raw) if raw else None

    for i in range(180):
        try:
            request("/health")
            break
        except (URLError, TimeoutError):
            if i % 6 == 0:
                print("waiting for model readiness", flush=True)
            time.sleep(10)
    else:
        raise RuntimeError("readiness timeout")

    def tokenize(text, chat=False):
        body = {"model": args.model}
        if chat:
            body.update(
                messages=[{"role": "user", "content": text}],
                chat_template_kwargs={"thinking": False},
            )
        else:
            body.update(prompt=text, add_special_tokens=False)
        return request("/tokenize", body)["tokens"]

    def prompt(batch, index):
        rng = random.Random(args.seed + batch * 100 + index)
        key = f"archive-{batch}-{index}-{rng.randrange(100000, 999999)}"
        value = f"ORCHID-{rng.randrange(10000000, 99999999)}"
        # Keep chat framing and question intact; insert exactly enough filler tokens
        # in the middle of the user message for an 8192-token total prompt.
        marker = "NEEDLE_INSERTION_MARKER"
        frame = tokenize(
            f"Run {key}. Find the exact access code for {key} in the archive below.\n"
            f"{marker}\nEnd archive. What is the access code for {key}? "
            "Start with the exact code, then explain how you found it.",
            True,
        )
        mark = tokenize(marker)
        positions = [
            i
            for i in range(len(frame) - len(mark) + 1)
            if frame[i : i + len(mark)] == mark
        ]
        assert len(positions) == 1, positions
        pos = positions[0]
        prefix = frame[:pos]
        suffix = frame[pos + len(mark) :]
        needle = tokenize(f"\nThe access code for {key} is {value}.\n")
        filler = tokenize(
            "".join(
                f"Record {j}: cabinet {rng.randrange(1000, 9999)} contains routine "
                "weather observations and maintenance notes.\n"
                for j in range(1000)
            )
        )
        n = 8192 - len(prefix) - len(suffix) - len(needle)
        depth = (0.1, 0.5, 0.9)[index % 3]
        split = int(n * depth)
        tokens = prefix + filler[:split] + needle + filler[split:n] + suffix
        assert len(tokens) == 8192
        return tokens, value, depth

    all_correct = True
    print("ready; building 8k/1k needle workloads", flush=True)
    with args.output.open("x") as output:
        for batch in args.batches:
            cases = [prompt(batch, i) for i in range(batch)]
            before = fetch(base, "/metrics")
            for name in ("num_requests_running", "num_requests_waiting"):
                if counter(before, name):
                    raise RuntimeError("Endpoint is not idle")
            if args.profile:
                request("/start_profile", {})
            active = args.profile
            body = dict(
                model=args.model,
                prompt=[c[0] for c in cases],
                temperature=0,
                max_tokens=1024,
                ignore_eos=True,
                stream=True,
                stream_options={"include_usage": True},
            )
            r = Request(
                base + "/v1/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            texts = [""] * batch
            events = [0] * batch
            first = [None] * batch
            usage = None
            start = time.perf_counter()
            try:
                with urlopen(r, timeout=1800) as response:
                    for raw in response:
                        if not raw.startswith(b"data: "):
                            continue
                        data = raw[6:].strip()
                        if data == b"[DONE]":
                            break
                        item = json.loads(data)
                        if item.get("usage"):
                            usage = item["usage"]
                        for c in item.get("choices", []):
                            i = c["index"]
                            text = c.get("text", "")
                            texts[i] += text
                            if text:
                                events[i] += 1
                                if first[i] is None:
                                    first[i] = time.perf_counter() - start
                        # Capture prefill and early decode, then finish the
                        # 1024-token workload without further tracing.
                        if active and min(events) >= 32:
                            request("/stop_profile", {})
                            active = False
                            print(
                                f"batch {batch}: prefill and early decode captured",
                                flush=True,
                            )
            finally:
                if active:
                    request("/stop_profile", {})
            elapsed = time.perf_counter() - start
            after = fetch(base, "/metrics")
            completed = counter(after, "request_prefill_time_seconds_count") - counter(
                before, "request_prefill_time_seconds_count"
            )
            if completed != batch:
                raise RuntimeError(
                    "Concurrent traffic or counter reset invalidates run"
                )
            cache_hits = sum(
                counter(after, name) - counter(before, name)
                for name in (
                    "prefix_cache_hits_total",
                    "external_prefix_cache_hits_total",
                )
            )
            assert (
                usage
                and usage["prompt_tokens"] == 8192 * batch
                and usage["completion_tokens"] == 1024 * batch
            ), usage
            row = dict(
                batch=batch,
                prompt_tokens_each=8192,
                output_tokens_each=1024,
                seconds=elapsed,
                ttft_seconds=first,
                usage=usage,
                profiled=args.profile,
                prefix_cache_hits=cache_hits,
                metrics_before=before,
                metrics_after=after,
                seed=args.seed,
                needle_found=[c[1] in t for c, t in zip(cases, texts, strict=True)],
                depths=[c[2] for c in cases],
                expected=[c[1] for c in cases],
                responses=texts,
                prompt_sha256=[
                    hashlib.sha256(json.dumps(c[0]).encode()).hexdigest() for c in cases
                ],
            )
            all_correct &= all(row["needle_found"])
            output.write(json.dumps(row) + "\n")
            output.flush()
            if cache_hits:
                raise RuntimeError(
                    "Prefix reuse invalidates cold timing; rerun with a fresh --seed"
                )
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in (
                            "batch",
                            "seconds",
                            "ttft_seconds",
                            "usage",
                            "needle_found",
                        )
                    }
                ),
                flush=True,
            )
    print("all needle workloads complete", flush=True)

    if not all_correct:
        raise RuntimeError("Needle retrieval failed; see recorded responses")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exact 8k/1k needle retrieval and optional CUDA capture windows."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="deepseek-v4.1")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batches", nargs="+", type=int, default=list(range(1, 8)))
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--seed", type=int, default=827100)
    args = parser.parse_args()
    if any(b < 1 or b > 8 for b in args.batches):
        parser.error("batches must be in 1..8")
    main(args)
