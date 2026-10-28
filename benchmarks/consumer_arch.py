"""Correctness-gated dense INT8 comparison across consumer GPU architectures.

Synthetic weights; these results do not measure checkpoint quantization quality.
"""

import argparse
import json
import statistics
import subprocess
from functools import partial
from pathlib import Path

import torch

from consumer_int8 import ops


def metrics(actual, expected):
    a, b = actual.float().flatten(), expected.float().flatten()
    assert torch.isfinite(a).all() and a.norm() > 0
    relative = ((a - b).norm() / b.norm()).item()
    return {
        "relative_l2": relative,
        "cosine": torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
        "sqnr_db": -20 * torch.tensor(max(relative, 1e-20)).log10().item(),
    }


def measure(call):
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(20):
            captured = call()
    samples = []
    for _ in range(7):
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 20)
    return {"samples_us": samples, "median_us": statistics.median(samples)}, captured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vllm-probe", type=Path)
    parser.add_argument("--allspark", action="store_true")
    args = parser.parse_args()
    if args.vllm_probe:
        torch.ops.load_library(str(args.vllm_probe))
    torch.manual_seed(20261026)
    torch.backends.cuda.matmul.allow_tf32 = False
    props = torch.cuda.get_device_properties(0)
    result = dict(
        gpu=str(props),
        uuid=str(props.uuid),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        source="synthetic",
        rows=[],
        telemetry=subprocess.check_output(["nvidia-smi", "-q"], text=True),
    )
    for n, k in [(3072, 2560), (2560, 1536)]:
        w = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
        sw = (torch.rand(n, device="cuda") * 0.002 + 0.001).bfloat16()
        dequant = (w.float() * sw.float()[:, None]).bfloat16()
        if args.allspark:
            from vllm import _custom_ops as vops

            codes = (w.short() + 128).byte()
            packed, packed_scale, _ = vops.allspark_repack_weight(
                codes.T.contiguous(), sw[None], None, False
            )
        for m in (1, 4, 8, 32, 128, 512):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            a = torch.empty_like(x, dtype=torch.int8)
            sa = torch.empty(m, device="cuda")
            out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            ops.quantize_rows(x, a, sa)
            reference = x.float() @ dequant.float().T
            integer_ref = (
                (a.float() @ w.float().T) * sa[:, None] * sw.float()[None, :]
            ).bfloat16()

            def fused(x=x, a=a, sa=sa, w=w, sw=sw, out=out):
                ops.quantize_rows(x, a, sa)
                ops.fused_channel_mm(a, w, sa, sw, out, 0)

            calls = {"quantize_and_fused_mm": fused}
            if m <= 8:
                calls["channel_gemv"] = partial(ops.channel_gemv, x, w, sw, out)
            if args.vllm_probe:
                scales = sw.float()
                calls["vllm_cutlass_mm"] = partial(
                    torch.ops.cutlass_arch_probe.mm, out, a, w.T, sa, scales, None
                )
            if args.allspark:
                calls["allspark_w8a16"] = partial(
                    vops.allspark_w8a16_gemm,
                    x,
                    packed,
                    packed_scale,
                    None,
                    n,
                    -1,
                    props.multi_processor_count,
                    props.major * 10 + props.minor,
                    1024,
                    False,
                    True,
                )
            for name, call in calls.items():
                actual = call()
                actual = out if actual is None else actual
                integer_path = name in ("quantize_and_fused_mm", "vllm_cutlass_mm")
                if integer_path:
                    expected = integer_ref
                    if name == "vllm_cutlass_mm":
                        # vLLM's EVT applies channel scale before row scale.
                        expected = (
                            (a.float() @ w.float().T)
                            * sw.float()[None, :]
                            * sa[:, None]
                        ).bfloat16()
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                error = metrics(actual, reference)
                assert error["relative_l2"] < 0.02, error
                eager = actual.clone()
                timing, captured = measure(call)
                # Verify the captured path too, not only eager execution.
                torch.testing.assert_close(
                    out if captured is None else captured, eager, atol=0, rtol=0
                )
                row = dict(m=m, n=n, k=k, op=name, **error, **timing)
                row["weight_gb_per_s"] = (n * k + n * 2) / timing["median_us"] / 1000
                result["rows"].append(row)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
