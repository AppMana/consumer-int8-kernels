# SPDX-License-Identifier: Apache-2.0
"""Own four fresh vLLM servers and run the complete two-A5000 needle matrix."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path


def write_json(path, value):
    pending = path.with_suffix(".json.tmp")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    pending.replace(path)


def git_head(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def stop_server(process):
    # The API parent can exit while a worker still owns CUDA allocations.
    # Always finish cleaning the session created for this server.
    def group_exists():
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return False
        return True

    for sig, timeout in ((signal.SIGINT, 45), (signal.SIGTERM, 15)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            process.wait()
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            process.poll()  # Reap the parent before checking its process group.
            if not group_exists():
                process.wait()
                return
            time.sleep(0.25)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


class GpuMonitor(threading.Thread):
    """Record device load and invalidate measurements if another job appears."""

    def __init__(self, server_pid, output):
        super().__init__(daemon=True)
        self.server_pid = server_pid
        self.output = output
        self.done = threading.Event()
        self.problems = []

    def run(self):
        with self.output.open("w") as stream:
            while not self.done.is_set():
                record = dict(monotonic_s=time.monotonic())
                try:
                    processes = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-compute-apps=pid,process_name,used_memory",
                            "--format=csv,noheader",
                        ],
                        text=True,
                        timeout=10,
                    )
                    record["compute_processes"] = processes
                    for line in processes.splitlines():
                        if not line.strip() or "gnome-remote-desktop-daemon" in line:
                            continue
                        pid = int(line.split(",", 1)[0])
                        try:
                            owned = os.getpgid(pid) == self.server_pid
                        except ProcessLookupError:
                            continue
                        if not owned:
                            self.problems.append(
                                dict(monotonic_s=time.monotonic(), process=line)
                            )
                    record["devices"] = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=index,uuid,memory.used,utilization.gpu,clocks.sm,power.draw,temperature.gpu",
                            "--format=csv,noheader",
                        ],
                        text=True,
                        timeout=10,
                    )
                except Exception as exc:
                    record["error"] = repr(exc)
                    self.problems.append(record)
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                self.done.wait(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--speculative-tokens", type=int, default=1)
    parser.add_argument(
        "--linear-backend",
        choices=[
            "allspark",
            "consumer_int8",
            "consumer_int8_hybrid",
            "consumer_int8_prefill",
        ],
    )
    parser.add_argument("--gdn-backend", choices=["triton", "consumer_int8"])
    parser.add_argument("--kv-writer", choices=["triton", "consumer_int8"])
    parser.add_argument("--gdn-spec-kernel", choices=["cuda", "triton"], default="cuda")
    parser.add_argument(
        "--topology", nargs="+", choices=["tp2", "pp2"], default=["tp2", "pp2"]
    )
    parser.add_argument(
        "--mtp", nargs="+", choices=["off", "on"], default=["off", "on"]
    )
    args = parser.parse_args()
    args.model = args.model.resolve()
    args.vllm_repo = args.vllm_repo.resolve()
    if Path(sys.prefix).resolve() != (args.vllm_repo / ".venv").resolve():
        raise RuntimeError("Run this launcher with the vLLM repository's .venv Python")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    package = Path(__file__).resolve().parents[1]
    for repository in (args.vllm_repo, package):
        changes = subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"], text=True
        )
        if changes:
            raise RuntimeError(
                f"Commit the measured source before benchmarking: {repository}"
            )
    config_bytes = (args.model / "config.json").read_bytes()
    config = json.loads(config_bytes)
    quantization_config = dict(config["quantization_config"])
    for key, value in (
        ("linear_backend", args.linear_backend),
        ("gdn_decode_backend", args.gdn_backend),
        ("kv_cache_backend", args.kv_writer),
    ):
        if value is not None:
            quantization_config[key] = value
    base_url = f"http://127.0.0.1:{args.port}"
    provenance = dict(
        vllm_commit=git_head(args.vllm_repo),
        kernels_commit=git_head(package),
        checkpoint=str(args.model),
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        quantization_config=config.get("quantization_config"),
        effective_quantization_config=quantization_config,
        python=sys.executable,
        python_version=sys.version,
        packages={
            dist.metadata["Name"]: dist.version
            for dist in importlib.metadata.distributions()
            if dist.metadata["Name"]
        },
    )
    failures = []
    reports = []
    for topology in args.topology:
        tp, pp = (2, 1) if topology == "tp2" else (1, 2)
        for mtp in args.mtp:
            name = f"{topology}-mtp-{mtp}"
            out = args.output_dir / name
            out.mkdir()
            # Compute workloads invalidate timings, even when enough VRAM remains.
            gpu_processes = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            if any(
                line.strip() and "gnome-remote-desktop-daemon" not in line
                for line in gpu_processes.splitlines()
            ):
                raise RuntimeError("Both GPUs must be free of other compute workloads")
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", args.port))
            hardware = subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True)
            (out / "gpu-before.xml").write_text(hardware)
            command = [
                sys.executable,
                "-m",
                "vllm.entrypoints.cli.main",
                "serve",
                str(args.model),
                "--served-model-name",
                "qwen3.8-27b-int8",
                "--hf-overrides",
                json.dumps({"quantization_config": quantization_config}),
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
                "int8_per_token_head",
                "--mamba-ssm-cache-dtype",
                "float32",
                "--mm-encoder-tp-mode",
                "data",
                "--attention-backend",
                "TRITON_ATTN",
                "--max-model-len",
                "33536",
                "--max-num-seqs",
                "4",
                "--enable-chunked-prefill",
                "--max-num-batched-tokens",
                str(args.max_num_batched_tokens),
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
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
            if mtp == "on":
                command += [
                    "--speculative-config",
                    json.dumps(
                        dict(
                            method="mtp",
                            num_speculative_tokens=args.speculative_tokens,
                        )
                    ),
                ]
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="0,1",
                CUDA_DEVICE_ORDER="PCI_BUS_ID",
                VLLM_USE_V2_MODEL_RUNNER="1",
                VLLM_GDN_DECODE_KERNEL=args.gdn_spec_kernel,
                PYTHONUNBUFFERED="1",
                PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
                PYTHONPATH=str(package / "src")
                + os.pathsep
                + os.environ.get("PYTHONPATH", ""),
            )
            manifest = dict(
                provenance,
                tp=tp,
                pp=pp,
                mtp=mtp,
                command=command,
                environment={
                    key: env[key]
                    for key in (
                        "CUDA_VISIBLE_DEVICES",
                        "CUDA_DEVICE_ORDER",
                        "VLLM_USE_V2_MODEL_RUNNER",
                        "VLLM_GDN_DECODE_KERNEL",
                    )
                },
            )
            with (out / "server.log").open("w") as stream:
                process = subprocess.Popen(
                    command,
                    cwd=args.vllm_repo,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                manifest["server_pid"] = process.pid
                write_json(out / "server-manifest.json", manifest)
                monitor = GpuMonitor(process.pid, out / "gpu-load.jsonl")
                monitor.start()
                try:
                    deadline = time.monotonic() + 1800
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError(
                                f"Server exited before ready: {process.returncode}"
                            )
                        if time.monotonic() > deadline:
                            raise TimeoutError("Server startup exceeded 1800 seconds")
                        try:
                            with urllib.request.urlopen(
                                base_url + "/v1/models", timeout=2
                            ) as response:
                                models = json.load(response)
                            if any(
                                m["id"] == "qwen3.8-27b-int8" for m in models["data"]
                            ):
                                break
                        except (OSError, ValueError):
                            pass
                        time.sleep(2)
                    write_json(out / "models.json", models)
                    bench = [
                        sys.executable,
                        str(package / "benchmarks/needle_matrix.py"),
                        "--base-url",
                        base_url,
                        "--model",
                        "qwen3.8-27b-int8",
                        "--tokenizer",
                        str(args.model),
                        "--tp",
                        str(tp),
                        "--pp",
                        str(pp),
                        "--mtp",
                        mtp,
                        "--output-dir",
                        str(out / "needle"),
                    ]
                    with (out / "benchmark.log").open("w") as log:
                        result = subprocess.run(
                            bench, env=env, stdout=log, stderr=subprocess.STDOUT
                        )
                    manifest["benchmark_returncode"] = result.returncode
                    if result.returncode:
                        failures.append(name)
                except Exception as exc:
                    failures.append(name)
                    manifest["error"] = repr(exc)
                finally:
                    monitor.done.set()
                    monitor.join(timeout=25)
                    manifest["gpu_monitor_problems"] = monitor.problems
                    manifest["valid_for_performance_selection"] = (
                        not monitor.problems
                        and manifest.get("benchmark_returncode") == 0
                        and "error" not in manifest
                    )
                    if monitor.problems and name not in failures:
                        failures.append(name)
                    stop_server(process)
                    manifest["server_returncode"] = process.returncode
                    write_json(out / "server-manifest.json", manifest)
            report = out / "needle/report.md"
            if report.exists() and manifest["valid_for_performance_selection"]:
                reports.append(report.read_text())
            print(name, "failed" if name in failures else "completed", flush=True)
    (args.output_dir / "report.md").write_text("\n\n".join(reports))
    write_json(args.output_dir / "status.json", dict(failed=failures))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
