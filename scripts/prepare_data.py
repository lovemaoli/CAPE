"""Build the train-only U/W statistics consumed by CAPE.

Input: SID-level CSV splits in MiniOneRec format
(``user_id``, ``history_item_sid``, ``item_sid``) and the matching info file.
Output: a compressed stats.npz cache (no dataset files are copied).

Usage:
    python scripts/prepare_data.py \
        --name Industrial_and_Scientific \
        --train data/processed/Industrial_and_Scientific/train.csv \
        --info  data/processed/Industrial_and_Scientific/info.txt \
        --valid data/processed/Industrial_and_Scientific/valid.csv \
        --test  data/processed/Industrial_and_Scientific/test.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as "python scripts/prepare_data.py" from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cape.data import load_split
from cape.statistics import build_statistics, save_statistics


def _tail_targets(path: Path, stats) -> int:
    count = 0
    for sample in load_split(path):
        index = stats.lookup.get(sample.target)
        if index is not None and stats.tail_mask[index]:
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="dataset", help="dataset name, used for the default output path")
    parser.add_argument("--train", type=Path, required=True, help="training CSV")
    parser.add_argument("--info", type=Path, help="catalog info file (tab separated, SID first)")
    parser.add_argument("--valid", type=Path, help="validation CSV, for a statistics report")
    parser.add_argument("--test", type=Path, help="test CSV, for a statistics report")
    parser.add_argument("--out", type=Path, help="output .npz path")
    parser.add_argument("--tail-ratio", type=float, default=0.8)
    args = parser.parse_args()

    out = args.out or Path("data/processed") / args.name / "stats.npz"
    stats = build_statistics(args.train, args.info, args.tail_ratio)
    save_statistics(stats, out)

    print(f"dataset:      {args.name}")
    print(f"catalog:      {stats.size}")
    print(f"tail items:   {int(stats.tail_mask.sum())}")
    print(f"train rows:   {len(load_split(args.train))}")
    print(f"U entries:    {stats.U.data.size}")
    print(f"W entries:    {stats.W.data.size}")
    for label, path in (("valid", args.valid), ("test", args.test)):
        if path:
            print(f"{label} rows:    {len(load_split(path))}")
            print(f"{label} tail targets: {_tail_targets(path, stats)}")
    print(f"cache:        {out}")


if __name__ == "__main__":
    main()
