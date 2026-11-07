"""Evaluate CAPE on pre-computed beam outputs and teacher-forced likelihoods.

The frozen backbone is not run here: each input record already contains its
50-item beam, the original-vocabulary log-likelihoods of those items, and
teacher-forced log-likelihoods for candidates outside the beam (``extra_ll``).
CAPE performs pool expansion, evidence fusion and top-1 anchoring, then
reports HR@5/10 and NDCG@10 overall and on tail targets.

Usage:
    python scripts/run_cape.py --stats data/processed/stats.npz \
        --test data/processed/test.jsonl --case MiniOneRec_Office_Products_seed42
    python scripts/run_cape.py --stats ... --valid ... --test ... --select-lambda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cape.data import BeamRecord, load_beam_jsonl
from cape.metrics import hit, ndcg, paired_bootstrap_ci, rank_of, select_lambda, summarize
from cape.rerank import cape_rank
from cape.statistics import TrainingStatistics, load_statistics

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("overall_HR5", "overall_HR10", "overall_NDCG10", "tail_HR5", "tail_HR10", "tail_NDCG10")


def tail_flags(records: list[BeamRecord], stats: TrainingStatistics) -> list[bool]:
    flags = []
    for record in records:
        index = stats.lookup.get(record.target)
        flags.append(bool(index is not None and stats.tail_mask[index]))
    return flags


def ranking(record: BeamRecord, lam: float, stats: TrainingStatistics, budget: int):
    return cape_rank(
        record.history,
        record.candidates,
        record.ll,
        record.extra_ll,
        stats,
        lam=lam,
        budget=budget,
        anchor=record.anchor,
    )


def backbone_summary(records: list[BeamRecord], flags: list[bool]) -> dict:
    return summarize([rank_of(r.target, r.candidates) for r in records], flags)


def report(reference: dict, current: dict) -> None:
    for key in KEYS:
        before, after = reference[key], current[key]
        print(f"  {key:<16} {before * 100:8.2f}  {after * 100:8.2f}  {(after - before) * 100:+8.2f} pp")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats", type=Path, required=True, help="stats.npz from prepare_data.py")
    parser.add_argument("--test", type=Path, required=True, help="test beam JSONL")
    parser.add_argument("--valid", type=Path, help="validation beam JSONL (for --select-lambda)")
    parser.add_argument("--case", help="case name in configs/lambda.json")
    parser.add_argument("--lambda", dest="lam", type=float, help="fusion weight (overrides --case)")
    parser.add_argument("--select-lambda", action="store_true", help="select lambda on --valid")
    parser.add_argument("--budget", type=int, default=25, help="number of added tail candidates")
    parser.add_argument("--out", type=Path, default=Path("results/rankings.jsonl"))
    parser.add_argument("--bootstrap", action="store_true", help="paired user-cluster CIs for HR@10/NDCG@10")
    args = parser.parse_args()

    stats = load_statistics(args.stats)
    test = load_beam_jsonl(args.test)
    test_flags = tail_flags(test, stats)
    valid = load_beam_jsonl(args.valid) if args.valid else None

    lam: float | None = args.lam
    if args.select_lambda:
        if valid is None:
            parser.error("--select-lambda requires --valid")
        valid_flags = tail_flags(valid, stats)
        reference = backbone_summary(valid, valid_flags)["overall_HR10"]
        lam, feasible, grid = select_lambda(
            valid,
            valid_flags,
            lambda record, value: ranking(record, value, stats, args.budget).ranking,
            reference,
        )
        print(f"selected lambda={lam} (feasible={feasible}, grid={len(grid)} points)")
    if lam is None and args.case:
        with (ROOT / "configs" / "lambda.json").open("r", encoding="utf-8") as handle:
            lam = float(json.load(handle)["cases"][args.case])
    if lam is None:
        lam = 1.0
        print("no --lambda/--case given; using lam=1.0")

    results = [ranking(record, lam, stats, args.budget) for record in test]
    cape_ranks = [rank_of(record.target, result.ranking) for record, result in zip(test, results)]
    backbone_ranks = [rank_of(r.target, r.candidates) for r in test]
    before = summarize(backbone_ranks, test_flags)
    after = summarize(cape_ranks, test_flags)
    print(f"case: {args.case or 'custom'}   lambda: {lam}   test rows: {len(test)}   pool: {len(results[0].pool)}")
    print(f"  {'metric':<16} {'backbone':>8}  {'CAPE':>8}  {'delta':>12}")
    report(before, after)

    if args.bootstrap:
        users = [r.user_id for r in test]
        for group, mask in (("overall", [True] * len(test)), ("tail", test_flags)):
            keep = [i for i, flag in enumerate(mask) if flag]
            for name, fn, k in (("HR10", hit, 10), ("NDCG10", ndcg, 10)):
                first = [fn(cape_ranks[i], k) for i in keep]
                second = [fn(backbone_ranks[i], k) for i in keep]
                estimate, low, high = paired_bootstrap_ci([users[i] for i in keep], first, second)
                print(f"  {group}_{name}: delta {estimate * 100:+.2f} pp, 95% CI [{low * 100:+.2f}, {high * 100:+.2f}] pp")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record, result, rank, is_tail in zip(test, results, cape_ranks, test_flags):
            handle.write(
                json.dumps(
                    {
                        "sample_id": record.sample_id,
                        "user_id": record.user_id,
                        "target": record.target,
                        "tail": bool(is_tail),
                        "anchor": result.anchor,
                        "rank": rank,
                        "lambda": lam,
                        "pool_size": len(result.pool),
                        "added": list(result.added),
                        "ranking": list(result.ranking),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"rankings: {args.out}")


if __name__ == "__main__":
    main()
