"""Fetch selected original 1017 expert tensors for Unsloth error comparisons."""

import argparse
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

from unsloth_gguf import read_range


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 20, 39])
    args = parser.parse_args()
    repo = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}") as r:
        revision = json.load(r)["sha"]
    base = f"https://huggingface.co/{repo}/resolve/{revision}"
    with urllib.request.urlopen(base + "/model.safetensors.index.json") as r:
        index = json.load(r)["weight_map"]
    headers, output, payloads, evidence = {}, {}, [], []
    offset = 0
    for layer in args.layers:
        for projection in ("w1", "w3", "w2"):
            for suffix in ("weight", "scale"):
                key = f"layers.{layer}.ffn.experts.0.{projection}.{suffix}"
                shard = index[key]
                url = base + "/" + shard
                if shard not in headers:
                    length = struct.unpack("<Q", read_range(url, 0, 8))[0]
                    headers[shard] = (length, json.loads(read_range(url, 8, length)))
                length, header = headers[shard]
                entry = header[key]
                lo, hi = entry["data_offsets"]
                data = read_range(url, 8 + length + lo, hi - lo)
                assert len(data) == hi - lo
                output[key] = entry | {"data_offsets": [offset, offset + len(data)]}
                offset += len(data)
                payloads.append(data)
                evidence.append(
                    dict(
                        key=key,
                        shard=shard,
                        metadata=entry,
                        sha256=hashlib.sha256(data).hexdigest(),
                    )
                )
                print(key, entry["shape"], entry["dtype"], flush=True)
    header = json.dumps(output).encode()
    header += b" " * (-len(header) % 8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        stream.write(struct.pack("<Q", len(header)))
        stream.write(header)
        for data in payloads:
            stream.write(data)
    args.output.with_suffix(".json").write_text(
        json.dumps(dict(repo=repo, revision=revision, tensors=evidence), indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
