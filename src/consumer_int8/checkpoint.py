# SPDX-License-Identifier: Apache-2.0
"""Encode a full Qwen3.8-27B snapshot, retaining vision, MTP and shard ownership."""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from . import __version__
from .quantization import (
    dequantize_channel,
    dequantize_group32,
    quantize_channel,
    quantize_group32,
)

SOURCE_REPO = "Qwen/Qwen3.8-27B"
SOURCE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
_LAYER = r"(?:model\.language_model|mtp)\.layers\.\d+\."
_CHANNEL = re.compile(
    _LAYER + r"(?:mlp\.(?:gate|up|down)_proj|self_attn\.(?:q|k|v|o)_proj|"
    r"linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj))\.weight$"
)
_VISION = re.compile(
    r"model\.visual\.(?:blocks\.\d+\.(?:attn\.(?:qkv|proj)|"
    r"mlp\.linear_fc[12])|merger\.linear_fc[12])\.weight$"
)


def encoding_for(name: str) -> str:
    if _CHANNEL.fullmatch(name) or name == "mtp.fc.weight":
        return "channel_uint8"
    if _VISION.fullmatch(name):
        return "group32_int8"
    return "passthrough"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    pending.replace(path)


def encode(source: Path, destination: Path, *, source_revision: str) -> dict:
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must differ")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(
            "Destination must be empty; partial conversions are not models"
        )
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "qwen3_5" or config.get("quantization_config"):
        raise ValueError("Expected an unquantized Qwen3.5 architecture checkpoint")
    text = config["text_config"]
    if (text["hidden_size"], text["num_hidden_layers"]) != (5120, 64):
        raise ValueError("This encoding manifest targets Qwen3.8-27B geometry")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    source_map = index["weight_map"]
    if not any(n.startswith("mtp.") for n in source_map):
        raise ValueError("The full checkpoint must contain MTP weights")
    if not any(n.startswith("model.visual.") for n in source_map):
        raise ValueError("The full checkpoint must contain vision weights")
    destination.mkdir(parents=True, exist_ok=True)
    report = {
        "format_version": 1,
        "encoder_version": __version__,
        "source_repo": SOURCE_REPO,
        "source_revision": source_revision,
        "status": "encoding",
        "tensors": {},
        "shards": {},
    }
    weight_map = {}
    total_size = 0
    for shard in sorted(set(source_map.values())):
        if Path(shard).name != shard:
            raise ValueError(f"Invalid shard filename: {shard}")
        output = {}
        with safe_open(source / shard, framework="pt", device="cpu") as handle:
            expected = {n for n, owner in source_map.items() if owner == shard}
            if set(handle.keys()) != expected:
                raise ValueError(f"Index and shard disagree: {shard}")
            for name in sorted(expected):
                weight = handle.get_tensor(name)
                encoding = encoding_for(name)
                record = {"shape": list(weight.shape), "encoding": encoding}
                if encoding == "passthrough":
                    output[name] = weight
                else:
                    if encoding == "channel_uint8":
                        codes, scales = quantize_channel(weight)
                        scale_name = name.removesuffix(".weight") + ".weight_scale_inv"
                        restored = dequantize_channel(codes, scales)
                    else:
                        codes, scales = quantize_group32(weight)
                        scale_name = name.removesuffix(".weight") + ".weight_scale"
                        restored = dequantize_group32(codes, scales)
                    if scale_name in source_map:
                        raise ValueError(f"Scale name already exists: {scale_name}")
                    output[name], output[scale_name] = codes, scales
                    error = restored - weight.float()
                    record["rmse"] = error.square().mean().sqrt().item()
                    record["max_abs_error"] = error.abs().max().item()
                    del restored, error
                record["stored_dtype"] = str(output[name].dtype)
                report["tensors"][name] = record
        pending = destination / (shard + ".tmp")
        save_file(output, pending, metadata={"format": "pt"})
        pending.replace(destination / shard)
        for name, tensor in output.items():
            weight_map[name] = shard
            total_size += tensor.numel() * tensor.element_size()
        report["shards"][shard] = {
            "sha256": _digest(destination / shard),
            "tensor_count": len(output),
        }
        del output
        _write_json(destination / "encoding_report.json", report)
        print(f"Encoded {shard}", flush=True)
    if set(report["tensors"]) != set(source_map):
        raise RuntimeError("Conversion did not preserve the source tensor inventory")
    for path in source.iterdir():
        if path.is_file() and path.suffix in {".json", ".txt", ".jinja"}:
            if path.name not in {
                "config.json",
                "model.safetensors.index.json",
                "crc32.txt",
            }:
                shutil.copy2(path, destination / path.name)
        elif path.name == "LICENSE":
            shutil.copy2(path, destination / path.name)
    config["quantization_config"] = {
        "quant_method": "qwen_int8",
        "format_version": 1,
        "weight_strategy": "channel_uint8",
        "vision_weight_strategy": "group32_int8",
        "source_repo": SOURCE_REPO,
        "source_revision": source_revision,
        "quantized_modules": {
            name.removesuffix(".weight"): record["encoding"]
            for name, record in report["tensors"].items()
            if record["encoding"] != "passthrough"
        },
    }
    _write_json(
        destination / "model.safetensors.index.json",
        {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        },
    )
    report["status"] = "encoded_unvalidated"
    report["total_tensor_bytes"] = total_size
    _write_json(destination / "encoding_report.json", report)
    # Publish config last: a interrupted conversion must not look loadable.
    _write_json(destination / "config.json", config)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    encode(args.source, args.destination, source_revision=args.source_revision)


if __name__ == "__main__":
    main()
