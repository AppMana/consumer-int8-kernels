# SPDX-License-Identifier: Apache-2.0
"""Compare every fixture's answer against the BF16 reference run."""

import argparse
import json
from pathlib import Path

from consumer_int8.quality import compare_vision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_vision(
        json.loads(args.reference.read_text()), json.loads(args.candidate.read_text())
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
