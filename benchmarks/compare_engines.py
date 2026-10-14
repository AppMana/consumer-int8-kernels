# SPDX-License-Identifier: Apache-2.0
"""Matched uncached needle cohorts for local vLLM and llama.cpp servers.

llama.cpp must expose its existing monotonic phase timestamps in timings:
prompt_start_us, prompt_end_us, generation_end_us. Durations alone cannot
measure aggregate throughput when request phases overlap.
"""

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import httpx
from needle_matrix import make_prompt
from run_matrix import GpuMonitor, stop_server, write_json
from transformers import AutoTokenizer

from consumer_int8.benchmark import audit_gpu_timing, phase_throughput


def source(repo):
    return {
        "head": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "diff": subprocess.check_output(
            ["git", "-C", str(repo), "diff", "HEAD"], text=True
        ),
    }


async def request(client, engine, model, ids, needle, max_tokens):
    if engine == "llama":
        path = "/completion"
        body = dict(
            prompt=ids,
            n_predict=max_tokens,
            temperature=0,
            seed=42,
            stream=True,
            cache_prompt=False,
            return_tokens=True,
        )
    else:
        path = "/v1/completions"
        body = dict(
            model=model,
            prompt=ids,
            max_tokens=max_tokens,
            temperature=0,
            seed=42,
            stream=True,
            stream_options={"include_usage": True},
            return_token_ids=True,
        )
    start = time.monotonic()
    first = None
    text, token_ids, metrics, usage, timing = "", [], {}, {}, {}
    finish = None
    async with client.stream("POST", path, json=body) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            row = json.loads(line[5:])
            if row.get("error"):
                raise RuntimeError(row["error"])
            if engine == "llama":
                delta = row.get("content", "")
                token_ids.extend(row.get("tokens", []))
                if row.get("timings"):
                    timing = row["timings"]
                if row.get("stop"):
                    finish = row.get("stop_type")
                    metrics = row
            else:
                choices = row.get("choices", [])
                delta = "".join(c.get("text", "") for c in choices)
                for choice in choices:
                    token_ids.extend(choice.get("token_ids") or [])
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
                if row.get("metrics"):
                    metrics = row["metrics"]
                if row.get("usage"):
                    usage = row["usage"]
            if delta:
                first = first or time.monotonic()
                text += delta
    end = time.monotonic()
    if engine == "llama":
        phase = {
            key: timing[src] / 1e6
            for key, src in (
                ("scheduled_s", "prompt_start_us"),
                ("first_token_s", "prompt_end_us"),
                ("last_token_s", "generation_end_us"),
            )
        }
        prompt, output, cached = (
            timing["prompt_n"],
            timing["predicted_n"],
            timing["cache_n"],
        )
    else:
        phase = metrics.get("engine_phase_timing")
        prompt, output = usage.get("prompt_tokens"), usage.get("completion_tokens")
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    if prompt != len(ids):
        raise ValueError(
            f"Input usage differs from supplied IDs: {prompt} != {len(ids)}"
        )
    if not phase or not (
        start - 1 < phase["scheduled_s"] < phase["last_token_s"] < end + 1
    ):
        raise ValueError("Server timestamps do not share the client monotonic clock")
    return dict(
        prompt_tokens=prompt,
        completion_tokens=output,
        cached_tokens=cached,
        engine_phase_timing=phase,
        server_metrics=metrics,
        llama_timings=timing,
        client_start_s=start,
        client_first_s=first,
        client_end_s=end,
        ttft_s=first - start if first else None,
        e2e_s=end - start,
        finish_reason=finish,
        needle=needle,
        text=text,
        token_ids=token_ids,
        needle_verbatim=needle in text,
        needle_trimmed=needle.strip() == text.strip(),
        prompt_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
    )


async def measure(args, monitor, out):
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    attempts, selected = [], []
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{args.port}", timeout=1800
    ) as client:
        for length in args.lengths:
            for concurrency in args.concurrencies:
                for cohort in range(-args.warmups, args.repeats):
                    seed = 20261011 + length * 10 + concurrency * 100 + cohort
                    prompts = []
                    for i in range(concurrency):
                        prompt, needle = make_prompt(
                            tokenizer,
                            length,
                            1000,
                            seed,
                            i,
                            (0.1, 0.5, 0.9)[(cohort + i) % 3],
                        )
                        ids = tokenizer.apply_chat_template(
                            [dict(role="user", content=prompt)],
                            tokenize=True,
                            add_generation_prompt=True,
                            enable_thinking=False,
                            return_dict=False,
                        )
                        assert len(ids) == length
                        prompts.append((ids, needle))
                    for attempt in range(1, 31):
                        quiet = (15, 45, 90, 180)[min(attempt - 1, 3)]
                        while any(
                            p.get("sample_end_monotonic_s", p.get("monotonic_s", 0))
                            > time.monotonic() - quiet
                            for p in monitor.problems
                        ):
                            await asyncio.sleep(3)
                        start = time.monotonic()
                        results = await asyncio.gather(
                            *[
                                request(
                                    client,
                                    args.engine,
                                    "qwen3.8-27b-int8",
                                    *p,
                                    args.max_tokens,
                                )
                                for p in prompts
                            ]
                        )
                        end = time.monotonic()
                        if args.max_tokens == 1128 and any(
                            not 990 <= r["completion_tokens"] <= 1010 for r in results
                        ):
                            write_json(out / "unexpected-output-length.json", results)
                            raise ValueError(
                                "Needle completion is outside the 1k output budget"
                            )
                        row = dict(
                            length=length,
                            concurrency=concurrency,
                            cohort=cohort,
                            seed=seed,
                            attempt=attempt,
                            results=results,
                            summary=phase_throughput(results, end - start),
                        )
                        await asyncio.sleep(6)
                        samples = [
                            json.loads(s)
                            for s in (out / "gpu.jsonl").read_text().splitlines()
                        ]
                        intervals = [
                            (
                                r["engine_phase_timing"]["scheduled_s"],
                                r["engine_phase_timing"]["last_token_s"],
                            )
                            for r in results
                        ]
                        row["interference"] = audit_gpu_timing(
                            samples, intervals, monitor.problems
                        )
                        attempts.append(row)
                        write_json(out / "attempts.json", attempts)
                        print(
                            json.dumps(
                                dict(
                                    length=length,
                                    concurrency=concurrency,
                                    cohort=cohort,
                                    attempt=attempt,
                                    interference=row["interference"],
                                    strict=sum(r["needle_verbatim"] for r in results),
                                    trimmed=sum(r["needle_trimmed"] for r in results),
                                    summary=row["summary"],
                                )
                            ),
                            flush=True,
                        )
                        if cohort < 0 or not row["interference"]:
                            selected.append(row)
                            write_json(out / "selected.json", selected)
                            break
                    else:
                        raise RuntimeError(
                            "Thirty interfered attempts for a fixed cohort"
                        )
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["llama", "vllm"], required=True)
    parser.add_argument("--topology", choices=["tp2", "pp2"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--llama-repo", type=Path, required=True)
    parser.add_argument("--vllm-manifest", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[8000, 16000, 32000])
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--ubatch", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=1128)
    parser.add_argument("--mtp", type=int, default=0)
    parser.add_argument("--quality-corpus", type=Path)
    parser.add_argument("--vision-repo", type=Path)
    parser.add_argument("--port", type=int, default=18085)
    parser.add_argument("--nccl-simple32", action="store_true")
    parser.add_argument("--triton-decode", action="store_true")
    args = parser.parse_args()
    if args.engine != "vllm" and (args.quality_corpus or args.vision_repo):
        parser.error("The existing NLL/vision quality clients require vLLM")
    if args.slots < max(args.concurrencies):
        parser.error("Slots must cover the maximum requested concurrency")
    args.output.mkdir(parents=True, exist_ok=False)
    old = json.loads(args.vllm_manifest.read_text())
    env = {k: v for k, v in os.environ.items() if not k.startswith("NCCL_")}
    settings = dict(
        CUDA_VISIBLE_DEVICES="0,1", CUDA_DEVICE_ORDER="PCI_BUS_ID", NCCL_DEBUG="INFO"
    )
    if args.nccl_simple32:
        settings.update(
            NCCL_PROTO="allreduce:Simple",
            NCCL_MIN_NCHANNELS="32",
            NCCL_MAX_NCHANNELS="32",
        )
    if args.engine == "llama":
        command = [
            str(args.llama_repo / "build-sm86/bin/llama-server"),
            "-m",
            str(args.gguf),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "-ngl",
            "all",
            "--fit",
            "off",
            "-sm",
            "tensor" if args.topology == "tp2" else "layer",
            "-ts",
            "1,1",
            "-fa",
            "on",
            "-ctk",
            "q8_0",
            "-ctv",
            "q8_0",
            "-c",
            str(33536 * args.slots),
            "-np",
            str(args.slots),
            "-b",
            str(args.batch),
            "-ub",
            str(args.ubatch),
            "--no-context-shift",
            "--cache-ram",
            "0",
            "--metrics",
            "-t",
            "16",
            "-tb",
            "16",
        ]
        if args.mtp:
            command += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(args.mtp)]
        settings["GGML_CUDA_P2P"] = "1"
        cwd = args.llama_repo
    else:
        command = list(old["command"])
        if args.triton_decode:
            index = command.index("--hf-overrides") + 1
            overrides = json.loads(command[index])
            overrides["quantization_config"]["attention_decode_backend"] = "triton"
            command[index] = json.dumps(overrides)
        for flag, value in (
            ("--port", args.port),
            ("--max-num-batched-tokens", args.batch),
            ("--max-num-seqs", args.slots),
            ("--tensor-parallel-size", 2 if args.topology == "tp2" else 1),
            ("--pipeline-parallel-size", 2 if args.topology == "pp2" else 1),
        ):
            command[command.index(flag) + 1] = str(value)
        if args.mtp:
            command += [
                "--speculative-config",
                json.dumps(dict(method="mtp", num_speculative_tokens=args.mtp)),
            ]
        settings.update(
            {k: v for k, v in old["environment"].items() if not k.startswith("NCCL_")}
        )
        cwd = Path(command[0]).parents[2]
    env.update(settings)
    state = dict(
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        command=command,
        environment=settings,
        status="starting",
        sources={str(p): source(p) for p in (cwd, Path(__file__).resolve().parents[1])},
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (args.output / "harness.py").write_bytes(Path(__file__).read_bytes())
    write_json(args.output / "manifest.json", state)
    with (args.output / "server.log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        state["server_pid"] = process.pid
        monitor = GpuMonitor(process.pid, args.output / "gpu.jsonl")
        monitor.start()
        try:
            deadline = time.monotonic() + 1800
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited with {process.returncode}")
                if time.monotonic() > deadline:
                    raise TimeoutError("Server startup")
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{args.port}/health", timeout=2
                    ):
                        break
                except OSError:
                    time.sleep(2)
            rows = asyncio.run(measure(args, monitor, args.output))
            state["status"] = "complete"
            state["cells"] = []
            for length in args.lengths:
                for c in args.concurrencies:
                    cell = [
                        r
                        for r in rows
                        if r["length"] == length
                        and r["concurrency"] == c
                        and r["cohort"] >= 0
                    ]
                    state["cells"].append(
                        dict(
                            length=length,
                            concurrency=c,
                            rates={
                                k: statistics.median(r["summary"][k] for r in cell)
                                for k in cell[0]["summary"]
                                if k != "timing_source"
                            },
                            strict=sum(
                                q["needle_verbatim"] for r in cell for q in r["results"]
                            ),
                            trimmed=sum(
                                q["needle_trimmed"] for r in cell for q in r["results"]
                            ),
                            total=len(cell) * c,
                        )
                    )
            if args.quality_corpus:
                score = [
                    sys.executable,
                    str(Path(__file__).with_name("quality_nll.py")),
                    "score",
                    "--base-url",
                    f"http://127.0.0.1:{args.port}",
                    "--model",
                    "qwen3.8-27b-int8",
                    "--corpus",
                    str(args.quality_corpus),
                    "--provenance",
                    str(args.output / "manifest.json"),
                    "--output",
                    str(args.output / "nll"),
                ]
                with (args.output / "nll.log").open("w") as output:
                    subprocess.run(
                        score,
                        env=env,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            if args.vision_repo:
                from run_quality import vision_fixture_digest

                digest = vision_fixture_digest(args.vision_repo)
                for concurrency in (1, 4):
                    cmd = [
                        sys.executable,
                        "-c",
                        "from appmana_inference_benchmarks.vision.evaluate "
                        "import main; main()",
                        "--base-url",
                        f"http://127.0.0.1:{args.port}",
                        "--model",
                        "qwen3.8-27b-int8",
                        "--phase",
                        f"vision-c{concurrency}",
                        "--output-dir",
                        str(args.output),
                        "--concurrency",
                        str(concurrency),
                    ]
                    with (args.output / f"vision-c{concurrency}.log").open(
                        "w"
                    ) as output:
                        subprocess.run(
                            cmd,
                            env=dict(env, PYTHONPATH=str(args.vision_repo / "src")),
                            stdout=output,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
                assert vision_fixture_digest(args.vision_repo) == digest
                state["vision_fixture_sha256"] = digest
        except BaseException as exc:
            state.update(status="failed", error=repr(exc))
            raise
        finally:
            monitor.done.set()
            monitor.join(timeout=15)
            stop_server(process)
            state["interference"] = monitor.problems
            write_json(args.output / "manifest.json", state)


if __name__ == "__main__":
    main()
