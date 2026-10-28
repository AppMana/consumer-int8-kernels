"""Inspect pinned Unsloth GGUFs and fetch exact expert blocks using HTTP ranges.

Only headers and explicitly selected experts are downloaded. Packed blocks are
preserved byte-for-byte, with hashes and offsets for loader qualification.
"""

import argparse
import hashlib
import json
import math
import re
import struct
import urllib.request
from pathlib import Path

from gguf import GGML_QUANT_SIZES, GGMLQuantizationType


def read_range(url, start, size):
    request = urllib.request.Request(
        url + f"?range_start={start}&range_size={size}",
        headers={"Range": f"bytes={start}-{start + size - 1}"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        if response.status != 206:
            raise ValueError(f"Server ignored byte range: {response.status}")
        match = re.fullmatch(
            r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
        )
        if not match:
            raise ValueError("Content-Range does not match request")
        first, last, total = map(int, match.groups())
        if first != start or last != min(start + size, total) - 1:
            raise ValueError("Content-Range does not match request")
        expected_size = last - first + 1
        data = response.read(size + 1)
    if len(data) != expected_size:
        raise ValueError(f"Truncated range: expected {expected_size}, got {len(data)}")
    return data


class Header:
    def __init__(self, data):
        self.data = data
        self.offset = 0

    def unpack(self, fmt):
        result = struct.unpack_from("<" + fmt, self.data, self.offset)
        self.offset += struct.calcsize("<" + fmt)
        return result[0] if len(result) == 1 else result

    def string(self):
        size = self.unpack("Q")
        end = self.offset + size
        if end > len(self.data):
            raise ValueError("GGUF header exceeds downloaded range")
        value = self.data[self.offset : end].decode()
        self.offset = end
        return value

    def value(self, kind):
        formats = {
            0: "B",
            1: "b",
            2: "H",
            3: "h",
            4: "I",
            5: "i",
            6: "f",
            7: "?",
            10: "Q",
            11: "q",
            12: "d",
        }
        if kind in formats:
            return self.unpack(formats[kind])
        if kind == 8:
            return self.string()
        if kind == 9:
            element, count = self.unpack("IQ")
            for _ in range(count):
                self.value(element)
            return {"array_length": count}
        raise ValueError(f"Unsupported metadata type {kind}")

    def parse(self):
        magic, version, count, kv_count = self.unpack("4sIQQ")
        if magic != b"GGUF" or version != 3:
            raise ValueError("Expected GGUF version 3")
        metadata = {}
        for _ in range(kv_count):
            key = self.string()
            metadata[key] = self.value(self.unpack("I"))
        tensors = []
        for _ in range(count):
            name = self.string()
            ndim = self.unpack("I")
            dims = self.unpack("Q" * ndim)
            dims = [dims] if ndim == 1 else list(dims)
            kind, offset = self.unpack("IQ")
            quant = GGMLQuantizationType(kind)
            block, size = GGML_QUANT_SIZES[quant]
            elements = math.prod(dims)
            if elements % block:
                raise ValueError(f"Unaligned tensor {name}")
            tensors.append(
                dict(
                    name=name,
                    dims=dims,
                    type=quant.name,
                    offset=offset,
                    nbytes=elements // block * size,
                )
            )
        alignment = int(metadata.get("general.alignment", 32))
        data_start = (self.offset + alignment - 1) // alignment * alignment
        for tensor in tensors:
            tensor["offset"] += data_start
        return metadata, tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="unsloth/DeepSeek-V4.1-Flash-1017-GGUF")
    parser.add_argument(
        "--revision", help="Pinned Hub revision; defaults to current HEAD"
    )
    parser.add_argument(
        "--variant", choices=("UD-IQ3_XXS", "UD-Q2_K_XL"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="*", default=[])
    parser.add_argument("--experts", type=int, default=1)
    args = parser.parse_args()
    if args.experts < 1:
        parser.error("--experts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    api = f"https://huggingface.co/api/models/{args.repo}"
    if args.revision:
        api += f"/revision/{args.revision}"
    with urllib.request.urlopen(api) as r:
        info = json.load(r)
    revision = info["sha"]
    manifest = dict(
        repo=args.repo, revision=revision, variant=args.variant, shards=[], samples=[]
    )
    files = sorted(
        x["rfilename"]
        for x in info["siblings"]
        if x["rfilename"].startswith(args.variant + "/")
        and x["rfilename"].endswith(".gguf")
    )
    if not files:
        raise ValueError("No GGUF shards found for selected variant")
    for filename in files:
        url = f"https://huggingface.co/{args.repo}/resolve/{revision}/{filename}"
        data = read_range(url, 0, 8 * 1024 * 1024)
        metadata, tensors = Header(data).parse()
        manifest["shards"].append(
            dict(
                file=filename,
                metadata=metadata,
                tensors=tensors,
                header_sha256=hashlib.sha256(data).hexdigest(),
            )
        )
        print(filename, len(tensors), flush=True)
        for tensor in tensors:
            if tensor["name"] not in {
                f"blk.{layer}.ffn_{projection}_exps.weight"
                for layer in args.layers
                for projection in ("gate", "up", "down")
            }:
                continue
            if len(tensor["dims"]) != 3 or args.experts > tensor["dims"][2]:
                raise ValueError("Unexpected expert tensor geometry")
            size = tensor["nbytes"] // tensor["dims"][2] * args.experts
            blocks = read_range(url, tensor["offset"], size)
            path = args.output / (tensor["name"] + ".bin")
            path.write_bytes(blocks)
            sample = tensor | dict(
                file=filename,
                path=str(path.resolve()),
                experts=args.experts,
                downloaded_bytes=size,
                sha256=hashlib.sha256(blocks).hexdigest(),
            )
            manifest["samples"].append(sample)
            print(
                "sample",
                tensor["name"],
                tensor["type"],
                tensor["dims"],
                size,
                flush=True,
            )
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    expected = {
        f"blk.{layer}.ffn_{projection}_exps.weight"
        for layer in args.layers
        for projection in ("gate", "up", "down")
    }
    missing = expected - {sample["name"] for sample in manifest["samples"]}
    if missing:
        raise ValueError(f"Requested tensors not found: {sorted(missing)}")


if __name__ == "__main__":
    main()
