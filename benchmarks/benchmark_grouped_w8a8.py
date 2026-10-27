# SPDX-License-Identifier: Apache-2.0
"""Compare separate and batched group activation quantization for actual Q8 WO-A.

This measures a channel-requantized candidate, not exact GGUF arithmetic.
"""

import argparse
import ctypes
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", type=int, default=1)
parser.add_argument("--library", type=Path, required=True)
grouped_args = parser.add_mutually_exclusive_group()
grouped_args.add_argument("--grouped-library", type=Path)
grouped_args.add_argument("--grouped-native", action="store_true")
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
report = args.output.open("x")
manifest = json.loads(args.manifest.read_text())
sample = manifest["samples"][0]
assert sample["dims"] == [4096, 8192] and sample["type"] == "Q8_0"
data = (args.manifest.parent / Path(sample["path"]).name).read_bytes()
assert hashlib.sha256(data).hexdigest() == sample["sha256"]
torch.cuda.set_device(args.device)
torch.manual_seed(137)
torch.ops.load_library(str(args.library.resolve()))
ops = torch.ops.consumer_int8
grouped_quant = None
if args.grouped_library:
    grouped_lib = ctypes.CDLL(str(args.grouped_library.resolve()))
    grouped_quant = grouped_lib.grouped_quant4096
    grouped_quant.argtypes = (
        [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
    )
    grouped_quant.restype = ctypes.c_int


def apply_grouped(x, qa, sa, m):
    if args.grouped_native:
        ops.quantize_grouped4096(x, qa, sa)
    else:
        assert (
            grouped_quant(
                x.data_ptr(),
                qa.data_ptr(),
                sa.data_ptr(),
                m,
                4,
                torch.cuda.current_stream().cuda_stream,
            )
            == 0
        )


raw = np.frombuffer(data, np.uint8).reshape(8192, 128, 34)[4096:].copy()
sc = np.ascontiguousarray(raw[..., :2]).view("<f2").astype(np.float32)
w = (
    torch.from_numpy(
        (sc * raw[..., 2:].view(np.int8).astype(np.float32)).reshape(4, 1024, 4096)
    )
    .cuda()
    .bfloat16()
)
sw = (w.float().abs().amax(-1) / 127).clamp_min(1e-20).bfloat16()
qw = (w.float() / sw.float()[..., None]).round().clamp(-127, 127).to(torch.int8)
flush = torch.empty(64 << 20, device="cuda", dtype=torch.uint8)
for m in (8, 128, 512, 2048):
    x = torch.randn(m, 4, 4096, device="cuda", dtype=torch.bfloat16) * 0.1

    def sequential(x=x, m=m):
        output = torch.empty(4, m, 1024, device="cuda", dtype=torch.bfloat16)
        qa = torch.empty(m, 4096, device="cuda", dtype=torch.int8)
        sa = torch.empty(m, device="cuda")
        for g in range(4):
            ops.quantize_rows(x[:, g].contiguous(), qa, sa)
            ops.fused_channel_mm(qa, qw[g], sa, sw[g], output[g], 0)
        return output.transpose(0, 1).contiguous()

    def batched(x=x, m=m):
        output = torch.empty(4, m, 1024, device="cuda", dtype=torch.bfloat16)
        qa = torch.empty(4, m, 4096, device="cuda", dtype=torch.int8)
        sa = torch.empty(4, m, device="cuda")
        ops.quantize_rows(
            x.transpose(0, 1).contiguous().view(-1, 4096),
            qa.view(-1, 4096),
            sa.flatten(),
        )
        for g in range(4):
            ops.fused_channel_mm(qa[g], qw[g], sa[g], sw[g], output[g], 0)
        return output.transpose(0, 1).contiguous()

    def strided(x=x, m=m):
        output = torch.empty(4, m, 1024, device="cuda", dtype=torch.bfloat16)
        qa = torch.empty(4, m, 4096, device="cuda", dtype=torch.int8)
        sa = torch.empty(4, m, device="cuda")
        apply_grouped(x, qa, sa, m)
        for g in range(4):
            ops.fused_channel_mm(qa[g], qw[g], sa[g], sw[g], output[g], 0)
        return output.transpose(0, 1).contiguous()

    candidates = [("sequential", sequential), ("batched", batched)]
    if grouped_quant is not None or args.grouped_native:
        qa = torch.empty(4, m, 4096, device="cuda", dtype=torch.int8)
        sa = torch.empty(4, m, device="cuda")
        apply_grouped(x, qa, sa, m)
        for g in range(4):
            reference_q = torch.empty_like(qa[g])
            reference_s = torch.empty_like(sa[g])
            ops.quantize_rows(x[:, g].contiguous(), reference_q, reference_s)
            assert torch.equal(qa[g], reference_q) and torch.equal(sa[g], reference_s)
        candidates.append(("strided", strided))
    graphs = {}
    ys = {}
    for name, fn in candidates:
        for _ in range(3):
            fn()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = fn()
        graphs[name] = g
        ys[name] = y
    x.neg_()
    for name, g in graphs.items():
        ys[name].fill_(float("nan"))
        g.replay()
    assert all(torch.equal(ys["sequential"], value) for value in ys.values())
    times = {name: [] for name in graphs}
    for trial in range(20):
        for name in list(graphs)[:: 1 if trial % 2 else -1]:
            flush.zero_()
            a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            a.record()
            graphs[name].replay()
            b.record()
            b.synchronize()
            times[name].append(a.elapsed_time(b))
    record = dict(
        m=m,
        groups=4,
        n=1024,
        k=4096,
        row_start=4096,
        medians_ms={name: statistics.median(t) for name, t in times.items()},
        milliseconds=times,
        checkpoint=manifest["repo"],
        revision=manifest["revision"],
        sample_sha256=sample["sha256"],
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
        grouped_library_sha256=(
            hashlib.sha256(args.grouped_library.read_bytes()).hexdigest()
            if args.grouped_library
            else None
        ),
        grouped_native=args.grouped_native,
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        device=str(torch.cuda.get_device_properties(args.device)),
        scope=(
            "channel-requantized W8A8; gather, activation quantization, "
            "GEMMs and output packing"
        ),
    )
    report.write(json.dumps(record) + "\n")
    report.flush()
    print(json.dumps(record), flush=True)
