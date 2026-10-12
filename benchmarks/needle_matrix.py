# SPDX-License-Identifier: Apache-2.0
"""Run nine needle cells against one TP/PP/MTP server configuration.

Prompt construction follows vLLM tools/ampere/dsv4_needle_bench.py. Each cell
has one discarded warmup cohort and three measured cohorts. Server request
metrics are mandatory; streamed chunks are never counted as output tokens.
"""

import argparse
import asyncio
import difflib
import json
import random
import statistics
import time
from pathlib import Path

from openai import AsyncOpenAI
from transformers import AutoTokenizer

from consumer_int8.benchmark import phase_throughput

WORDS = [
    "quartz",
    "lattice",
    "ember",
    "violet",
    "copper",
    "meadow",
    "harbor",
    "drift",
    "anchor",
    "tide",
    "signal",
    "ledger",
    "canyon",
    "summit",
    "prairie",
    "tundra",
    "glacier",
    "basalt",
    "marble",
    "cobalt",
    "saffron",
    "indigo",
    "crimson",
    "umber",
    "juniper",
    "cedar",
    "willow",
    "aspen",
    "falcon",
    "heron",
    "plover",
    "osprey",
    "kestrel",
    "gannet",
    "petrel",
    "skua",
    "tern",
    "brant",
    "keel",
    "hull",
    "mast",
    "spar",
    "boom",
    "tiller",
    "rudder",
    "winch",
    "cleat",
    "davit",
    "gudgeon",
]
INSTRUCTION = (
    "The document above contains exactly one passage delimited by the "
    "markers BEGIN_NEEDLE and END_NEEDLE. Reproduce that passage verbatim, "
    "without the markers and without any commentary before or after it."
)


def token_text(tokenizer, rng, count):
    text = " ".join(rng.choice(WORDS) for _ in range(count + 32))
    ids = tokenizer.encode(text, add_special_tokens=False)
    while len(ids) < count:
        text += " " + " ".join(rng.choice(WORDS) for _ in range(count - len(ids) + 32))
        ids = tokenizer.encode(text, add_special_tokens=False)
    result = tokenizer.decode(ids[:count])
    if len(tokenizer.encode(result, add_special_tokens=False)) != count:
        raise ValueError("Needle/filler tokenizer round trip changed its length")
    return result


def make_prompt(tokenizer, target, needle_tokens, seed, index, depth):
    rng = random.Random(0xD5F4 + seed * 0x10001 + index)
    needle = token_text(tokenizer, rng, needle_tokens)
    # A unique salt at the start also prevents accidental cross-cohort hits.
    salt = f"run {seed:08x}, document {index:03d}. "
    filler = token_text(tokenizer, rng, target)
    filler_ids = tokenizer.encode(filler, add_special_tokens=False)
    budget = target - needle_tokens - 128
    seen_budgets = set()

    def chat_length(text):
        return len(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            )
        )

    for _ in range(32):
        if budget in seen_budgets:
            # BPE boundaries can make adjacent filler lengths oscillate by
            # two tokens. Leave a short margin and fill it with whole words.
            budget -= 8
        seen_budgets.add(budget)
        if budget < 1 or budget > len(filler_ids):
            raise ValueError("Prompt template exceeds the requested token budget")
        cut = int(budget * depth)
        before = tokenizer.decode(filler_ids[:cut])
        after = tokenizer.decode(filler_ids[cut:budget])
        prompt = (
            f"{salt}{before}\nBEGIN_NEEDLE\n{needle}\nEND_NEEDLE\n"
            f"{after}\n\n{INSTRUCTION}"
        )
        measured = chat_length(prompt)
        if measured == target:
            return prompt, needle
        if 0 < target - measured <= 32:
            for word in (" a", " x", " 0", ".", "x", "\n."):
                for count in range(1, target - measured + 3):
                    padded = prompt.removesuffix("\n\n" + INSTRUCTION)
                    padded += word * count + "\n\n" + INSTRUCTION
                    if chat_length(padded) == target:
                        return padded, needle
        budget += target - measured
    raise ValueError("Could not construct the exact requested chat prompt length")


async def run_request(client, model, prompt, needle, max_tokens):
    start = time.perf_counter()
    first = previous = None
    text, usage, metrics, gaps = "", {}, {}, []
    finish = None
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    async for chunk in stream:
        raw = chunk.model_dump()
        if raw.get("metrics"):
            metrics = raw["metrics"]
        if chunk.usage:
            usage = chunk.usage.model_dump()
        for choice in chunk.choices:
            delta = choice.delta.content if choice.delta else None
            if delta:
                now = time.perf_counter()
                if first is None:
                    first = now
                if previous is not None:
                    gaps.append(now - previous)
                previous = now
                text += delta
            if choice.finish_reason:
                finish = choice.finish_reason
    details = usage.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": details.get("cached_tokens", 0),
        "engine_phase_timing": metrics.get("engine_phase_timing"),
        "server_metrics": metrics,
        "ttft_s": first - start if first is not None else None,
        "e2e_s": time.perf_counter() - start,
        "stream_chunk_gaps_s": gaps,
        "finish_reason": finish,
        "needle_verbatim": needle in text,
        "match_ratio": difflib.SequenceMatcher(
            None, needle.split(), text.split(), autojunk=False
        ).ratio(),
        "needle": needle,
        "text": text,
    }


def write_json(path, value):
    pending = path.with_suffix(".json.tmp")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    pending.replace(path)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tp", required=True, type=int, choices=[1, 2])
    parser.add_argument("--pp", required=True, type=int, choices=[1, 2])
    parser.add_argument("--mtp", required=True, choices=["off", "on"])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument(
        "--input-lengths", type=int, nargs="+", default=[8000, 16000, 32000]
    )
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()
    if args.tp * args.pp != 2:
        parser.error("This matrix requires TP2/PP1 or TP1/PP2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    manifest = vars(args).copy()
    manifest["output_dir"] = str(args.output_dir)
    manifest.update(needle_tokens=1000, max_output_tokens=1128, measured_cohorts=3)
    write_json(args.output_dir / "manifest.json", manifest)
    failure = False
    rows = []
    async with AsyncOpenAI(
        base_url=args.base_url.rstrip("/") + "/v1",
        api_key="none",
        timeout=3600,
        max_retries=0,
    ) as client:
        for length in args.input_lengths:
            for concurrency in args.concurrencies:
                for cohort in range(-1, 3):
                    warmup = cohort == -1
                    seed = args.seed + length * 10 + concurrency * 100 + cohort
                    depths = [
                        (0.1, 0.5, 0.9)[(cohort + i) % 3] for i in range(concurrency)
                    ]
                    prompts = [
                        make_prompt(tokenizer, length, 1000, seed, i, depths[i])
                        for i in range(concurrency)
                    ]
                    start = time.perf_counter()
                    results = await asyncio.gather(
                        *[
                            run_request(client, args.model, prompt, needle, 1128)
                            for prompt, needle in prompts
                        ]
                    )
                    elapsed = time.perf_counter() - start
                    record = {
                        "tp": args.tp,
                        "pp": args.pp,
                        "mtp": args.mtp,
                        "target_input_tokens": length,
                        "concurrency": concurrency,
                        "cohort": cohort,
                        "warmup": warmup,
                        "seed": seed,
                        "depths": depths,
                        "results": results,
                    }
                    try:
                        if any(r["prompt_tokens"] != length for r in results):
                            raise ValueError(
                                "Server prompt usage differs from the tokenized target"
                            )
                        record["summary"] = phase_throughput(results, elapsed)
                    except ValueError as exc:
                        record["invalid_reason"] = str(exc)
                        failure = True
                    passed = sum(r["needle_verbatim"] for r in results)
                    record["needle_verbatim_passed"] = passed
                    if not warmup and passed != concurrency:
                        failure = True
                    name = (
                        f"tp{args.tp}-pp{args.pp}-mtp{args.mtp}"
                        f"-{length}-c{concurrency}-{cohort}"
                    )
                    write_json(args.output_dir / f"{name}.json", record)
                    rows.append(record)
                    print(
                        name,
                        f"verbatim={passed}/{concurrency}",
                        record.get("summary", record.get("invalid_reason")),
                        flush=True,
                    )
    columns = [
        "TP",
        "PP",
        "MTP",
        "Input",
        "C",
        "Prefill tok/s stream",
        "Prefill tok/s aggregate",
        "Decode tok/s stream",
        "Decode tok/s aggregate",
        "TTFT s",
        "E2E s",
        "Verbatim",
    ]
    report_lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---:"] * len(columns)) + " |",
    ]
    for length in args.input_lengths:
        for concurrency in args.concurrencies:
            cells = [
                r
                for r in rows
                if not r["warmup"]
                and r["target_input_tokens"] == length
                and r["concurrency"] == concurrency
            ]
            if any("summary" not in r for r in cells):
                continue
            values = [
                statistics.median(r["summary"][key] for r in cells)
                for key in (
                    "prefill_tok_s_single_stream_median",
                    "prefill_tok_s_aggregate",
                    "decode_tok_s_single_stream_median",
                    "decode_tok_s_aggregate",
                )
            ]
            requests = [request for r in cells for request in r["results"]]
            values += [
                statistics.median(r[key] for r in requests)
                for key in ("ttft_s", "e2e_s")
            ]
            passed = sum(r["needle_verbatim_passed"] for r in cells)
            formatted = " | ".join(f"{value:.2f}" for value in values)
            report_lines.append(
                f"| {args.tp} | {args.pp} | {args.mtp} | {length} | {concurrency} "
                f"| {formatted} | {passed}/{3 * concurrency} |"
            )
    (args.output_dir / "report.md").write_text("\n".join(report_lines) + "\n")
    return int(failure)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
