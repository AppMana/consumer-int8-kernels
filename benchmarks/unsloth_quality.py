"""Compare exact Unsloth expert blocks to the matching original 1017 weights.

CPU GGUF decoding is a loader reference, not a native compressed GPU kernel.
INT8 conversion is measured separately so its additional loss stays visible.
"""

import argparse
import hashlib
import json
import time
from functools import partial
from pathlib import Path

import gguf
import numpy as np
import torch
import torch.nn.functional as F
from consumer_arch import measure, metrics
from safetensors import safe_open
from vllm import _custom_ops as vops

from consumer_int8 import ops
from consumer_int8.quantization import quantize_channel


def decode_original(packed, scales):
    """Reference MXFP4 decode independent of the installed vLLM version."""
    codes = packed.view(torch.uint8)
    nibbles = torch.stack((codes & 15, codes >> 4), dim=-1).flatten(-2)
    values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=packed.device)
    fp4 = values[(nibbles & 7).long()] * (1 - 2 * (nibbles >> 3).float())
    exponent = scales.view(torch.uint8)
    if torch.any(exponent == 255):
        raise ValueError("Nonfinite E8M0 scale in reference checkpoint")
    scale = torch.exp2(exponent.float() - 127)
    return (fp4.reshape(*fp4.shape[:-1], -1, 32) * scale[..., None]).flatten(-2)


def benchmark_projection(weight, projection, name):
    # TP2 after decoding; this does not assert packed-GGUF TP compatibility.
    w = (weight[:, :1152] if projection == "down" else weight[:1152]).contiguous()
    n, k = w.shape
    codes, scales = quantize_channel(w)
    signed = (codes.short() - 128).char()
    packed, packed_s, _ = vops.allspark_repack_weight(
        codes.T.contiguous(), scales[None], None, False
    )
    props = torch.cuda.get_device_properties(0)
    rows = []
    for m in (1, 4, 8, 32, 128, 512):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
        expected = x.float() @ w.T
        a = torch.empty_like(x, dtype=torch.int8)
        sa = torch.empty(m, device="cuda")
        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

        def fused(x=x, a=a, sa=sa, out=out):
            ops.quantize_rows(x, a, sa)
            ops.fused_channel_mm(a, signed, sa, scales, out, 0)

        calls = {
            "bf16": partial(torch.mm, x, w.bfloat16().T),
            "fp16": partial(torch.mm, x.half(), w.half().T),
            "int8_quantize_and_cutlass": fused,
            "allspark_from_unsloth": partial(
                vops.allspark_w8a16_gemm,
                x,
                packed,
                packed_s,
                None,
                n,
                -1,
                props.multi_processor_count,
                props.major * 10 + props.minor,
                1024,
                False,
                True,
            ),
        }
        if m <= 8:
            calls["channel_gemv_from_unsloth"] = partial(
                ops.channel_gemv, x, signed, scales, out
            )
        for label, call in calls.items():
            actual = call()
            actual = out if actual is None else actual
            error = metrics(actual, expected)
            assert error["relative_l2"] < 0.03, error
            eager = actual.clone()
            timing, captured = measure(call)
            torch.testing.assert_close(
                out if captured is None else captured, eager, atol=0, rtol=0
            )
            rows.append(
                dict(
                    tensor=name,
                    op=label,
                    m=m,
                    n=n,
                    k=k,
                    vs_decoded_unsloth=error,
                    **timing,
                )
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernels", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    torch.manual_seed(20261026)
    torch.backends.cuda.matmul.allow_tf32 = False
    result = dict(
        repo=manifest["repo"],
        revision=manifest["revision"],
        variant=manifest["variant"],
        rows=[],
        layers=[],
        kernel_rows=[],
        gpu=str(torch.cuda.get_device_properties(0)),
        reference=json.loads(args.reference.with_suffix(".json").read_text()),
        scope="expert 0; random BF16 activations; no model-level quality claim",
    )
    matrices = {}
    with safe_open(args.reference, framework="pt", device="cpu") as reference:
        for sample in manifest["samples"]:
            name = sample["name"]
            layer = int(name.split(".")[1])
            projection = name.split(".")[2].split("_")[1]
            key = (
                f"layers.{layer}.ffn.experts.0."
                + {"gate": "w1", "up": "w3", "down": "w2"}[projection]
            )
            packed = reference.get_tensor(key + ".weight").cuda()
            scale = reference.get_tensor(key + ".scale").cuda()
            original = decode_original(packed, scale)
            start = time.perf_counter()
            data = np.fromfile(
                args.manifest.parent / Path(sample["path"]).name, dtype=np.uint8
            )
            assert hashlib.sha256(data).hexdigest() == sample["sha256"]
            quant = gguf.GGMLQuantizationType[sample["type"]]
            block, size = gguf.GGML_QUANT_SIZES[quant]
            cols, rows, _ = sample["dims"]
            decoded = gguf.dequantize(data.reshape(rows, cols // block * size), quant)
            cpu_ms = (time.perf_counter() - start) * 1000
            weight = torch.from_numpy(decoded).cuda()
            assert weight.shape == original.shape
            torch.cuda.synchronize()
            start = time.perf_counter()
            codes, scales = quantize_channel(weight)
            int8 = (codes.float() - 128) * scales.float()[:, None]
            torch.cuda.synchronize()
            conversion_ms = (time.perf_counter() - start) * 1000
            row = dict(
                name=name,
                type=sample["type"],
                shape=list(weight.shape),
                cpu_read_and_decode_ms=cpu_ms,
                gpu_int8_conversion_ms=conversion_ms,
                unsloth_vs_original=metrics(weight, original),
                int8_from_unsloth_vs_original=metrics(int8, original),
                int8_from_unsloth_vs_unsloth=metrics(int8, weight),
            )
            result["rows"].append(row)
            matrices.setdefault(layer, {})[projection] = (original, weight, int8)
            print(json.dumps(row), flush=True)
            if args.kernels:
                result["kernel_rows"].extend(
                    benchmark_projection(weight, projection, name)
                )
                args.output.write_text(json.dumps(result, indent=2) + "\n")
        for layer, projections in matrices.items():
            torch.manual_seed(20261026 + layer)
            x = torch.randn(32, 5120, device="cuda", dtype=torch.bfloat16).float() * 0.1
            outputs = []
            for index in range(3):
                gate, up, down = (projections[p][index] for p in ("gate", "up", "down"))
                outputs.append((F.silu(x @ gate.T) * (x @ up.T)) @ down.T)
            row = dict(
                layer=layer,
                unsloth_vs_original=metrics(outputs[1], outputs[0]),
                int8_from_unsloth_vs_original=metrics(outputs[2], outputs[0]),
                int8_from_unsloth_vs_unsloth=metrics(outputs[2], outputs[1]),
            )
            result["layers"].append(row)
            print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
