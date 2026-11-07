# CAPE inference
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .statistics import TrainingStatistics

EPS = 1e-12
DEFAULT_BUDGET = 25


def minmax(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """N(x)(c) = (x(c) - min x) / (ptp(x) + eps); constant x maps to zero."""
    x = np.asarray(values, dtype=np.float64)
    return (x - x.min()) / (np.ptp(x) + EPS)


def resolve_candidate_ll(sid: str, extra_ll: Any, stats: TrainingStatistics) -> float:
    """Read a teacher-forced likelihood from a mapping or a catalog-wide vector."""
    if isinstance(extra_ll, Mapping):
        if sid not in extra_ll:
            raise KeyError(
                f"Missing teacher-forced likelihood for {sid!r}. "
                "Score the memorization-channel candidates with the frozen generator."
            )
        return float(extra_ll[sid])
    array = np.asarray(extra_ll, dtype=np.float64)
    if array.ndim == 1 and array.size == stats.size:
        return float(array[stats.lookup[sid]])
    raise ValueError("extra_ll must be a SID->LL mapping or a catalog-sized vector")


def memorization_channel(
    history: Sequence[str],
    beam_candidates: Sequence[str],
    stats: TrainingStatistics,
    budget: int = DEFAULT_BUDGET,
) -> tuple[list[str], np.ndarray]:
    """Return the top-``budget`` tail items outside the beam by U support.

    Ties are broken by catalog index, and zero-support items still fill the
    remaining slots (Eq. 2). The full support vector is returned for auditing.
    """
    if budget < 0:
        raise ValueError("budget must be non-negative")
    support = stats.support(history)
    allowed = stats.tail_mask.copy()
    for sid in beam_candidates:
        if sid in stats.lookup:
            allowed[stats.lookup[sid]] = False
    ids = np.flatnonzero(allowed)
    if budget == 0 or ids.size == 0:
        return [], support
    order = ids[np.lexsort((ids, -support[ids]))]
    return [stats.catalog[i] for i in order[:budget]], support


@dataclass(frozen=True)
class CapeResult:
    ranking: tuple[str, ...]
    pool: tuple[str, ...]
    likelihood: tuple[float, ...]
    evidence: tuple[float, ...]
    scores: tuple[float, ...]
    anchor: str
    added: tuple[str, ...]


def cape_rank(
    history: Sequence[str],
    beam_candidates: Sequence[str],
    beam_ll: Sequence[float],
    extra_ll: Any,
    stats: TrainingStatistics,
    lam: float = 1.0,
    budget: int = DEFAULT_BUDGET,
    anchor: str | None = None,
) -> CapeResult:
    """Run one CAPE request and return the re-ranked candidate list.

    ``beam_candidates`` are the B=50 unique beam items and ``beam_ll`` their
    original-vocabulary log-likelihoods. ``extra_ll`` supplies the
    teacher-forced likelihoods of items added by the memorization channel.
    """
    if not beam_candidates or len(beam_candidates) != len(beam_ll):
        raise ValueError("beam_candidates and beam_ll must have the same non-zero length")
    if len(set(beam_candidates)) != len(beam_candidates):
        raise ValueError("beam candidates must be unique")
    if any(sid not in stats.lookup for sid in beam_candidates):
        raise ValueError("beam contains an item outside the catalog")
    if lam < 0:
        raise ValueError("lam must be non-negative")

    added, _ = memorization_channel(history, beam_candidates, stats, budget)
    added_ll = [resolve_candidate_ll(sid, extra_ll, stats) for sid in added]
    pool = tuple(beam_candidates) + tuple(added)
    likelihood = np.concatenate(
        [np.asarray(beam_ll, dtype=np.float64), np.asarray(added_ll, dtype=np.float64)]
    )
    evidence = stats.evidence(history, pool)
    scores = minmax(likelihood) + float(lam) * minmax(evidence)

    if anchor is None:
        anchor = beam_candidates[int(np.argmax(np.asarray(beam_ll)))]
    if anchor not in pool:
        raise ValueError("anchor is not in the candidate pool")
    order = sorted(range(len(pool)), key=lambda j: (-scores[j], j))
    ranking = [anchor] + [pool[j] for j in order if pool[j] != anchor]
    return CapeResult(
        ranking=tuple(ranking),
        pool=pool,
        likelihood=tuple(float(v) for v in likelihood),
        evidence=tuple(float(v) for v in evidence),
        scores=tuple(float(v) for v in scores),
        anchor=anchor,
        added=tuple(added),
    )
