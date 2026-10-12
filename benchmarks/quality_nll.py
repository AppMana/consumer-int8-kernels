# SPDX-License-Identifier: Apache-2.0
"""Prepare fixed WikiText test windows, score each server, then compare NLL."""

import argparse
import hashlib
import json
import math
import urllib.request
from pathlib import Path

from consumer_int8.quality import compare_nll, prompt_nll

DATASET = "Salesforce/wikitext"
REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
PARQUET = "wikitext-2-raw-v1/test-00000-of-00001.parquet"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def prepare(args):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    if args.windows < 2 or args.window_tokens < 2:
        raise ValueError("Use at least two windows of at least two tokens")
    source = hf_hub_download(DATASET, PARQUET, repo_type="dataset", revision=REVISION)
    text = "\n\n".join(pq.read_table(source)["text"].to_pylist())
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < args.windows * args.window_tokens:
        raise ValueError("Requested windows would overlap")
    starts = [
        i * (len(tokens) - args.window_tokens) // (args.windows - 1)
        for i in range(args.windows)
    ]
    windows = [tokens[start : start + args.window_tokens] for start in starts]
    corpus = dict(
        dataset=DATASET,
        revision=REVISION,
        file=PARQUET,
        license="CC-BY-SA-3.0 / GFDL; https://huggingface.co/datasets/Salesforce/wikitext",
        tokenizer=str(Path(args.tokenizer).resolve()),
        corpus_sha256=fingerprint(windows),
        starts=starts,
        windows=windows,
        method="Disjoint evenly spaced raw test windows; no chat template or BOS",
    )
    args.output.write_text(json.dumps(corpus) + "\n")
    print(f"Prepared {len(windows)} windows; {len(tokens)} corpus tokens")


def score(args):
    corpus = json.loads(args.corpus.read_text())
    if fingerprint(corpus["windows"]) != corpus["corpus_sha256"]:
        raise ValueError("Evaluation corpus checksum mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(
        corpus_sha256=corpus["corpus_sha256"],
        model=args.model,
        server=args.base_url,
        provenance=json.loads(args.provenance.read_text()),
        expected_windows=len(corpus["windows"]),
        complete=False,
        rows=[],
    )
    for index, tokens in enumerate(corpus["windows"]):
        payload = dict(
            model=args.model,
            prompt=tokens,
            add_special_tokens=False,
            prompt_logprobs=1,
            max_tokens=1,
            temperature=0,
            seed=42,
            return_token_ids=True,
            stream=False,
        )
        request = urllib.request.Request(
            args.base_url.rstrip("/") + "/v1/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=1800) as stream:
            response = json.load(stream)
        (args.output / f"response-{index:03}.json").write_text(
            json.dumps(response) + "\n"
        )
        losses = prompt_nll(response, tokens)
        row = dict(
            index=index,
            token_ids_sha256=fingerprint(tokens),
            scored_tokens=len(losses),
            nll_sum=math.fsum(losses),
            mean_nll=math.fsum(losses) / len(losses),
        )
        result["rows"].append(row)
        result["complete"] = len(result["rows"]) == len(corpus["windows"])
        (args.output / "scores.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(row), flush=True)


def compare(args):
    reference, candidate = (
        json.loads(p.read_text()) for p in (args.reference, args.candidate)
    )
    if not reference.get("complete") or not candidate.get("complete"):
        raise ValueError("Cannot compare incomplete evaluation runs")
    result = compare_nll(reference, candidate, args.limit)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--windows", type=int, default=32)
    p.add_argument("--window-tokens", type=int, default=2048)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(run=prepare)
    p = sub.add_parser("score")
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument(
        "--provenance", type=Path, required=True, help="Saved server launch manifest"
    )
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(run=score)
    p = sub.add_parser("compare")
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--limit", type=float, default=0.02)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(run=compare)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
