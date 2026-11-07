"""Ranking metrics, group summaries, paired bootstrap CIs and lambda selection."""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

MISSING_RANK = 1_000_000
LAMBDA_GRID = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)


def rank_of(target: str, ranking: Sequence[str]) -> int:
    try:
        return list(ranking).index(target) + 1
    except ValueError:
        return MISSING_RANK


def hit(rank: int, k: int) -> int:
    return int(rank <= k)


def ndcg(rank: int, k: int) -> float:
    return 1.0 / math.log2(rank + 1) if rank <= k else 0.0


def summarize(ranks: Sequence[int], tail_flags: Sequence[bool]) -> dict[str, float]:
    """HR@5/10 and NDCG@10 over all cases and over tail targets."""
    ranks = np.asarray(ranks, dtype=np.int64)
    tail = np.asarray(tail_flags, dtype=bool)
    out: dict[str, float] = {"N": int(ranks.size)}
    for group, mask in (("overall", np.ones(ranks.size, dtype=bool)), ("tail", tail)):
        values = ranks[mask]
        out[f"{group}_N"] = int(values.size)
        for k in (5, 10):
            out[f"{group}_HR{k}"] = float(np.mean(values <= k)) if values.size else float("nan")
        out[f"{group}_NDCG10"] = (
            float(np.mean(np.where(values <= 10, 1.0 / np.log2(values + 1), 0.0)))
            if values.size
            else float("nan")
        )
    return out


def paired_bootstrap_ci(
    users: Sequence[str],
    first: Sequence[float],
    second: Sequence[float],
    reps: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Resample users with replacement; return the mean difference and its CI."""
    delta = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    unique, inverse = np.unique(np.asarray(users), return_inverse=True)
    sums = np.bincount(inverse, weights=delta)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, unique.size, size=(reps, unique.size))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(delta.mean()), float(low), float(high)


def select_lambda(
    valid_records: Sequence[Any],
    tail_flags: Sequence[bool],
    rank_fn: Callable[[Any, float], Sequence[str]],
    reference_overall_hr10: float,
    grid: Sequence[float] = LAMBDA_GRID,
) -> tuple[float, bool, list[dict[str, float]]]:
    """Select lambda on the full validation split (Section 4.4 of the paper).

    Among grid points whose overall HR@10 is at least the backbone's, take
    the largest tail HR@10, then overall NDCG@10, then the smaller lambda.
    If no point is feasible, return lambda=0 with ``feasible=False``.
    """
    entries: list[dict[str, float]] = []
    targets = [record.target for record in valid_records]
    for lam in grid:
        ranks = [rank_of(target, rank_fn(record, lam)) for record, target in zip(valid_records, targets)]
        entries.append({"lambda_value": float(lam), **summarize(ranks, tail_flags)})
    feasible = [e for e in entries if e["overall_HR10"] + 1e-12 >= reference_overall_hr10]
    best = (
        max(feasible, key=lambda e: (e["tail_HR10"], e["overall_NDCG10"], -e["lambda_value"]))
        if feasible
        else entries[0]
    )
    return float(best["lambda_value"]), bool(feasible), entries
