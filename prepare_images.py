"""Prepare paired 256px images for the H2O, Aria, and Assembly101 benchmark.

The Assembly101 command requires a manually aligned frame manifest. No timing
offset is inferred from equal frame numbers.
"""

import argparse
import csv
import io
from pathlib import Path

from preprocessing.aria import prepare_aria
from preprocessing.assembly101 import prepare_assembly
from preprocessing.h2o import prepare_h2o


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("h2o", "aria", "assembly101"))
    parser.add_argument("--input", type=Path, required=True, help="Downloaded H2O/Assembly101 root or aligned Aria root")
    parser.add_argument("--output", type=Path, required=True, help="Output root with training/ and testing/ image pairs")
    parser.add_argument("--setting", choices=("action", "object", "subject", "scene"), default="action", help="H2O setting")
    parser.add_argument("--alignment", type=Path, help="Assembly101 CSV of manually aligned frame pairs")
    parser.add_argument("--crops", type=Path, help="Assembly101 JSON crop rectangles, one per recording")
    parser.add_argument("--dry-run", action="store_true", help="Check counts without decoding or writing images")
    args = parser.parse_args()
    if not args.input.is_dir():
        parser.error(f"Input directory does not exist: {args.input}")
    if args.dataset != "h2o" and args.setting != "action":
        parser.error("Only H2O supports object, subject, or scene settings")
    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)
        index_path = args.output / "pairs.csv"
        if index_path.exists():
            parser.error(f"Refusing to overwrite {index_path}")
        handle = index_path.open("w", newline="")
    else:
        handle = io.StringIO()
    with handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "recording", "segment", "clip", "position", "split", "source_id", "exo_source", "ego_source"])
        if args.dataset == "h2o":
            counts = prepare_h2o(args, writer)
        elif args.dataset == "aria":
            counts = prepare_aria(args, writer)
        else:
            counts = prepare_assembly(args, writer)
    print(f"30-frame clips: training={counts['training']}, testing={counts['testing']}")
    if not counts:
        raise ValueError("No paired clips found")


if __name__ == "__main__":
    main()
