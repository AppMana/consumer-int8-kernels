# SPDX-License-Identifier: Apache-2.0
"""Own an isolated two-GPU server for fixed-corpus NLL and vision validation."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from run_matrix import VISION_LIMITS, GpuMonitor, git_head, stop_server, write_json


async def capacity_smoke(base_url, model, output):
    from needle_matrix import make_prompt, run_request
    from openai import AsyncOpenAI
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    async with AsyncOpenAI(
        base_url=base_url + "/v1", api_key="none", timeout=900, max_retries=0
    ) as client:
        prompts = [
            make_prompt(tokenizer, 32000, 1000, 20261012, i, (0.1, 0.5, 0.9)[i % 3])
            for i in range(4)
        ]
        rows = await asyncio.gather(
            *[run_request(client, "qwen-quality", *prompt, 16) for prompt in prompts]
        )
    write_json(
        output,
        dict(
            purpose="C=4 x 32k capacity/scheduling check; not a retrieval or throughput gate",
            results=rows,
        ),
    )
    for row in rows:
        if (
            row["prompt_tokens"] != 32000
            or row["cached_tokens"]
            or not row["engine_phase_timing"]
            or row["engine_phase_timing"]["num_preemptions"]
        ):
            raise RuntimeError("Capacity smoke failed token/cache/preemption checks")


def vision_fixture_digest(repo):
    fixtures = repo / "src/appmana_inference_benchmarks/vision/fixtures"
    cases = json.loads((fixtures / "cases.json").read_text())[:100]
    if len(cases) != 100:
        raise ValueError("Expected all 100 vision fixtures")
    files = {"cases.json"} | {case["image"] for case in cases}
    hashes = {
        name: hashlib.sha256((fixtures / name).read_bytes()).hexdigest()
        for name in sorted(files)
    }
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--topology", choices=["tp2", "pp2"], default="tp2")
    parser.add_argument("--mtp", choices=["off", "on"], default="off")
    parser.add_argument("--cpu-offload-gb", type=float, default=12)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=1073741824)
    parser.add_argument("--capacity-smoke", action="store_true")
    parser.add_argument(
        "--linear-backend",
        choices=["allspark", "consumer_int8_prefill", "consumer_int8_mlp_prefill"],
        default="allspark",
    )
    parser.add_argument(
        "--gdn-backend", choices=["triton", "consumer_int8"], default="triton"
    )
    parser.add_argument(
        "--kv-writer", choices=["triton", "consumer_int8"], default="triton"
    )
    parser.add_argument("--vision-repo", type=Path)
    parser.add_argument(
        "--attention-backend",
        choices=["triton", "consumer_int8", "consumer_fa2"],
        default="triton",
    )
    parser.add_argument("--vision-concurrency", type=int, nargs="+", default=[1])
    parser.add_argument("--port", type=int, default=18083)
    args = parser.parse_args()
    if args.capacity_smoke and (
        args.reference or args.max_model_len < 32016 or max(args.vision_concurrency) < 4
    ):
        parser.error("Capacity smoke requires a resident model, length >=32016 and C=4")
    tp, pp = (2, 1) if args.topology == "tp2" else (1, 2)
    repo = args.vllm_repo.resolve()
    model = args.model.resolve()
    package = Path(__file__).resolve().parents[1]
    if Path(sys.prefix).resolve() != repo / ".venv":
        raise RuntimeError("Use the vLLM repository's .venv Python")
    for checkout in (repo, package):
        if subprocess.check_output(
            ["git", "-C", str(checkout), "status", "--porcelain"], text=True
        ):
            raise RuntimeError(f"Commit measured source first: {checkout}")
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
        raise RuntimeError("Quality validation requires otherwise idle GPUs")
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", args.port))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config_bytes = (model / "config.json").read_bytes()
    config = json.loads(config_bytes)
    overrides = {}
    if not args.reference:
        quantization = dict(config["quantization_config"])
        quantization.update(
            linear_backend=args.linear_backend,
            gdn_decode_backend=args.gdn_backend,
            kv_cache_backend=args.kv_writer,
            attention_backend=args.attention_backend,
        )
        overrides["quantization_config"] = quantization
    elif config.get("quantization_config"):
        raise ValueError("Reference must contain the original BF16 weights")
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(model),
        "--served-model-name",
        "qwen-quality",
        "--tensor-parallel-size",
        str(tp),
        "--pipeline-parallel-size",
        str(pp),
        "--distributed-executor-backend",
        "mp",
        "--safetensors-load-strategy",
        "lazy",
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "auto" if args.reference else "int8_per_token_head",
        "--mamba-ssm-cache-dtype",
        "float32",
        "--mm-encoder-tp-mode",
        "data",
        "--limit-mm-per-prompt",
        json.dumps(VISION_LIMITS),
        "--attention-backend",
        "TRITON_ATTN",
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(1 if args.reference else max(args.vision_concurrency)),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--enable-chunked-prefill",
        "--gpu-memory-utilization",
        "0.85",
        "--kv-cache-memory-bytes",
        str(args.kv_cache_memory_bytes),
        "--no-enable-prefix-caching",
        "--enable-per-request-metrics",
        "--async-scheduling",
        "--seed",
        "42",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
    ]
    if args.reference:
        command += ["--cpu-offload-gb", str(args.cpu_offload_gb), "--enforce-eager"]
    else:
        command += ["--hf-overrides", json.dumps(overrides)]
    if args.mtp == "on":
        command += [
            "--speculative-config",
            json.dumps(dict(method="mtp", num_speculative_tokens=1)),
        ]
    manifest = dict(
        vllm_commit=git_head(repo),
        kernels_commit=git_head(package),
        checkpoint=str(model),
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        hf_overrides=overrides,
        reference=args.reference,
        tp=tp,
        pp=pp,
        mtp=args.mtp,
        command=command,
        cpu_offload_gb=args.cpu_offload_gb if args.reference else 0,
        python=sys.executable,
        gpu_devices="0,1",
        status="starting",
        packages={
            dist.metadata["Name"]: dist.version
            for dist in importlib.metadata.distributions()
            if dist.metadata["Name"]
        },
    )
    if args.vision_repo:
        manifest["vision_repository_commit"] = git_head(args.vision_repo)
        manifest["vision_fixture_sha256"] = vision_fixture_digest(args.vision_repo)
    manifest_path = args.output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="0,1",
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        VLLM_USE_V2_MODEL_RUNNER="1",
        VLLM_GDN_DECODE_KERNEL="cuda",
        PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        VIRTUAL_ENV=str(repo / ".venv"),
        PYTHONUNBUFFERED="1",
    )
    base_url = f"http://127.0.0.1:{args.port}"
    with (args.output_dir / "server.log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=repo,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        monitor = GpuMonitor(process.pid, args.output_dir / "gpu.jsonl")
        monitor.start()
        try:
            deadline = time.monotonic() + 2700
            while process.poll() is None and time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen(
                        base_url + "/v1/models", timeout=2
                    ) as response:
                        json.load(response)
                    break
                except Exception:
                    time.sleep(2)
            else:
                raise RuntimeError(f"Server not ready: exit={process.poll()}")
            manifest["status"] = "ready"
            write_json(manifest_path, manifest)
            score = [
                sys.executable,
                str(package / "benchmarks/quality_nll.py"),
                "score",
                "--base-url",
                base_url,
                "--model",
                "qwen-quality",
                "--corpus",
                str(args.corpus.resolve()),
                "--provenance",
                str(manifest_path.resolve()),
                "--output",
                str((args.output_dir / "nll").resolve()),
            ]
            with (args.output_dir / "nll.log").open("w") as nll_log:
                subprocess.run(
                    score, env=env, stdout=nll_log, stderr=subprocess.STDOUT, check=True
                )
            if args.vision_repo:
                vision_env = dict(
                    env, PYTHONPATH=str(args.vision_repo.resolve() / "src")
                )
                for concurrency in args.vision_concurrency:
                    if args.reference and concurrency != 1:
                        raise ValueError("CPU-offloaded reference vision uses C=1")
                    vision = [
                        sys.executable,
                        "-c",
                        "from appmana_inference_benchmarks.vision.evaluate "
                        "import main; main()",
                        "--base-url",
                        base_url,
                        "--model",
                        "qwen-quality",
                        "--phase",
                        f"vision-c{concurrency}",
                        "--output-dir",
                        str(args.output_dir.resolve()),
                        "--concurrency",
                        str(concurrency),
                    ]
                    with (args.output_dir / f"vision-c{concurrency}.log").open(
                        "w"
                    ) as vision_log:
                        subprocess.run(
                            vision,
                            env=vision_env,
                            stdout=vision_log,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
                    digest = vision_fixture_digest(args.vision_repo)
                    if digest != manifest["vision_fixture_sha256"]:
                        raise ValueError("Vision fixtures changed during evaluation")
                    report_path = (
                        args.output_dir / f"vision-c{concurrency}-evaluation.json"
                    )
                    report = json.loads(report_path.read_text())
                    report["fixture_sha256"] = digest
                    write_json(report_path, report)
            if args.capacity_smoke:
                asyncio.run(
                    capacity_smoke(
                        base_url, model, args.output_dir / "capacity-smoke.json"
                    )
                )
            manifest["status"] = "completed"
        except Exception as exc:
            manifest.update(status="failed", error=repr(exc))
        finally:
            stop_server(process)
            monitor.done.set()
            monitor.join(timeout=20)
            manifest["interference"] = monitor.problems
            if monitor.problems:
                manifest["status"] = "interference"
            write_json(manifest_path, manifest)
    print(
        json.dumps(
            {key: manifest[key] for key in ("status", "reference", "interference")}
        ),
        flush=True,
    )
    if manifest["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
