# SPDX-License-Identifier: Apache-2.0
"""Screen prefill token budgets, then qualify finalists with the 1k-output matrix.

This uses short outputs to isolate prefill. It is not a retrieval-quality gate
and its decode rates must not substitute for the full needle matrix.
"""

import argparse
import asyncio
import hashlib
import json
import os
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from needle_matrix import make_prompt, run_request
from openai import AsyncOpenAI
from run_matrix import GpuMonitor, git_head, stop_server, write_json
from transformers import AutoTokenizer

from consumer_int8.benchmark import phase_throughput


async def screen(args, out):
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows = []
    async with AsyncOpenAI(
        base_url=f"http://127.0.0.1:{args.port}/v1",
        api_key="unused",
        timeout=900,
        max_retries=0,
    ) as client:
        for length in args.input_lengths:
            for concurrency in args.concurrencies:
                for repeat in range(-1, args.repeats):
                    prompts = [
                        make_prompt(
                            tokenizer,
                            length,
                            1000,
                            20261013 + repeat,
                            index,
                            (0.1, 0.5, 0.9)[index % 3],
                        )
                        for index in range(concurrency)
                    ]
                    start = time.monotonic()
                    requests = await asyncio.gather(
                        *[
                            run_request(
                                client, "qwen3.8-27b-int8", *prompt, args.output_tokens
                            )
                            for prompt in prompts
                        ]
                    )
                    elapsed = time.monotonic() - start
                    row = dict(
                        length=length,
                        concurrency=concurrency,
                        repeat=repeat,
                        started_monotonic_s=start,
                        ended_monotonic_s=time.monotonic(),
                        requests=requests,
                        metrics=phase_throughput(requests, elapsed),
                    )
                    if any(
                        r["prompt_tokens"] != length
                        or r["completion_tokens"] != args.output_tokens
                        or r["cached_tokens"]
                        or not r["engine_phase_timing"]
                        or r["engine_phase_timing"]["num_preemptions"]
                        for r in requests
                    ):
                        raise RuntimeError(
                            "Screening failed token/cache/preemption checks"
                        )
                    rows.append(row)
                    write_json(out / "requests.json", rows)
                    print(
                        json.dumps({k: v for k, v in row.items() if k != "requests"}),
                        flush=True,
                    )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--server-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--chunks", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384, 32768]
    )
    parser.add_argument("--input-lengths", type=int, nargs="+", default=[8000, 32000])
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--port", type=int, default=18084)
    parser.add_argument(
        "--gdn-preparation-backend", choices=["triton", "consumer_int8"]
    )
    args = parser.parse_args()
    if (
        args.repeats < 2
        or args.output_tokens < 2
        or min(args.chunks) < max(args.concurrencies)
    ):
        parser.error(
            "Use at least two repetitions/output tokens and sufficient chunk capacity"
        )
    package = Path(__file__).resolve().parents[1]
    if Path(sys.prefix).resolve() != (args.vllm_repo / ".venv").resolve():
        raise RuntimeError("Use the vLLM repository's .venv Python")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    commits = {}
    for repo in (args.vllm_repo, package):
        if subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"], text=True
        ):
            raise RuntimeError(f"Commit measured source first: {repo}")
        commits[str(repo)] = git_head(repo)
    binaries = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (package / "src/consumer_int8").glob("*.so")
    }
    summary = []
    for source in args.server_manifests:
        baseline = json.loads(source.read_text())
        if Path(baseline["checkpoint"]).resolve() != args.model.resolve():
            raise ValueError(
                "Server manifest and tokenizer must use the same checkpoint"
            )
        topology = f"tp{baseline['tp']}-pp{baseline['pp']}"
        for chunk in args.chunks:
            active = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            if any(
                line.strip() and "gnome-remote-desktop-daemon" not in line
                for line in active.splitlines()
            ):
                raise RuntimeError("Both GPUs must be free before serving measurements")
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", args.port))
            for repo, head in commits.items():
                if git_head(repo) != head or subprocess.check_output(
                    ["git", "-C", repo, "status", "--porcelain"], text=True
                ):
                    raise RuntimeError("Measured source changed during sweep")
            if any(
                hashlib.sha256(
                    (package / "src/consumer_int8" / name).read_bytes()
                ).hexdigest()
                != digest
                for name, digest in binaries.items()
            ):
                raise RuntimeError("Measured CUDA binary changed during sweep")
            out = args.output_dir / f"{topology}-chunk{chunk}"
            out.mkdir()
            command = list(baseline["command"])
            for flag, value in (
                ("--max-num-batched-tokens", chunk),
                ("--port", args.port),
            ):
                command[command.index(flag) + 1] = str(value)
            if args.gdn_preparation_backend:
                index = command.index("--hf-overrides") + 1
                override = json.loads(command[index])
                override["quantization_config"]["gdn_preparation_backend"] = (
                    args.gdn_preparation_backend
                )
                command[index] = json.dumps(override)
            env = dict(os.environ, **baseline["environment"], PYTHONUNBUFFERED="1")
            manifest = dict(
                purpose="Short-output prefill screening",
                source_manifest=str(source),
                commits=commits,
                binaries=binaries,
                topology=topology,
                chunk=chunk,
                command=command,
                environment=baseline["environment"],
                cells=[],
            )
            write_json(out / "server-manifest.json", manifest)
            with (out / "server.log").open("w") as log:
                process = subprocess.Popen(
                    command,
                    cwd=args.vllm_repo,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                monitor = GpuMonitor(process.pid, out / "gpu-load.jsonl")
                monitor.start()
                try:
                    deadline = time.monotonic() + 1800
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError(f"Server exited: {process.returncode}")
                        if time.monotonic() > deadline:
                            raise TimeoutError("Server startup exceeded 1800 seconds")
                        try:
                            with urllib.request.urlopen(
                                f"http://127.0.0.1:{args.port}/v1/models", timeout=2
                            ) as response:
                                if any(
                                    m["id"] == "qwen3.8-27b-int8"
                                    for m in json.load(response)["data"]
                                ):
                                    break
                        except (OSError, ValueError):
                            pass
                        time.sleep(2)
                    rows = asyncio.run(screen(args, out))
                    for length in args.input_lengths:
                        for concurrency in args.concurrencies:
                            measured = [
                                r
                                for r in rows
                                if r["length"] == length
                                and r["concurrency"] == concurrency
                                and r["repeat"] >= 0
                            ]
                            cell = dict(
                                topology=topology,
                                chunk=chunk,
                                length=length,
                                concurrency=concurrency,
                            )
                            for name in measured[0]["metrics"]:
                                if name == "timing_source":
                                    cell[name] = measured[0]["metrics"][name]
                                else:
                                    cell[name] = statistics.median(
                                        r["metrics"][name] for r in measured
                                    )
                            manifest["cells"].append(cell)
                except Exception as exc:
                    manifest["error"] = repr(exc)
                finally:
                    stop_server(process)
                    monitor.done.set()
                    monitor.join(timeout=15)
                    manifest.update(
                        interference=monitor.problems,
                        server_returncode=process.returncode,
                    )
                    manifest["valid_for_selection"] = (
                        not monitor.problems and "error" not in manifest
                    )
                    write_json(out / "server-manifest.json", manifest)
            summary.append({k: v for k, v in manifest.items() if k != "command"})
            write_json(args.output_dir / "summary.json", summary)
            print(
                json.dumps(
                    dict(
                        topology=topology,
                        chunk=chunk,
                        valid=manifest["valid_for_selection"],
                        error=manifest.get("error"),
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
