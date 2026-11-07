from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .data import load_split, parse_catalog


@dataclass(frozen=True)
class SparseCounts:
    """Minimal square CSR matrix: data, column indices and row pointers."""

    data: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    size: int

    def rows_sum(self, row_ids: Sequence[int]) -> np.ndarray:
        """Sum the selected rows, counting a repeated row once per occurrence."""
        out = np.zeros(self.size, dtype=np.float64)
        for row in row_ids:
            start, end = int(self.indptr[row]), int(self.indptr[row + 1])
            # Each row has unique column indices, so in-place add is exact.
            out[self.indices[start:end]] += self.data[start:end]
        return out


def _csr_from_counts(counts: dict[tuple[str, str], int], lookup: dict[str, int], size: int) -> SparseCounts:
    rows = np.fromiter((lookup[j] for j, _ in counts), dtype=np.int32, count=len(counts))
    cols = np.fromiter((lookup[c] for _, c in counts), dtype=np.int32, count=len(counts))
    data = np.fromiter((float(v) for v in counts.values()), dtype=np.float64, count=len(counts))
    order = np.lexsort((cols, rows))
    rows, cols, data = rows[order], cols[order], data[order]
    indptr = np.zeros(size + 1, dtype=np.int64)
    np.add.at(indptr, rows.astype(np.int64) + 1, 1)
    np.cumsum(indptr, out=indptr)
    return SparseCounts(data=data, indices=cols, indptr=indptr, size=size)


@dataclass
class TrainingStatistics:
    catalog: tuple[str, ...]
    lookup: dict[str, int]
    frequencies: np.ndarray
    tail_mask: np.ndarray
    U: SparseCounts
    W: SparseCounts
    tail_ratio: float

    @property
    def size(self) -> int:
        return len(self.catalog)

    def history_indices(self, history: Sequence[str]) -> list[int]:
        try:
            return [self.lookup[sid] for sid in history]
        except KeyError as exc:
            raise ValueError(f"Unknown history SID: {exc}") from exc

    def support(self, history: Sequence[str]) -> np.ndarray:
        """S_u(c) = sum over the actual history positions of U[i_p, c]."""
        return self.U.rows_sum(self.history_indices(history))

    def evidence(self, history: Sequence[str], candidates: Sequence[str]) -> np.ndarray:
        """e_u(c) = 1[c in tail] * sum_p W[i_p, c] / max(1, n)."""
        values = self.W.rows_sum(self.history_indices(history)) / max(1, len(history))
        indices = [self.lookup[sid] for sid in candidates]
        return values[indices] * self.tail_mask[indices]


def build_statistics(
    train_file: str | Path,
    info_file: str | Path | None = None,
    tail_ratio: float = 0.8,
) -> TrainingStatistics:
    """Build all CAPE statistics from the training split and optional info file."""
    if not 0.0 < tail_ratio < 1.0:
        raise ValueError(f"tail_ratio must be in (0, 1), got {tail_ratio}")

    samples = load_split(train_file)
    frequencies: Counter[str] = Counter()
    catalog: list[str] = []
    seen: set[str] = set()
    w_counts: Counter[tuple[str, str]] = Counter()
    user_items: dict[str, set[str]] = defaultdict(set)

    for sample in samples:
        sequence = list(sample.history) + ([sample.target] if sample.target else [])
        frequencies.update(sequence)
        for sid in sequence:
            if sid not in seen:
                seen.add(sid)
                catalog.append(sid)
        # W: ordered pairs of distinct positions; repeated events are kept.
        for p, source in enumerate(sequence):
            for q, neighbor in enumerate(sequence):
                if p != q:
                    w_counts[(source, neighbor)] += 1
        # U: one vote per distinct training user and item pair.
        user_items[sample.user_id].update(sequence)

    for sid in parse_catalog(info_file):
        if sid not in seen:
            seen.add(sid)
            catalog.append(sid)

    lookup = {sid: i for i, sid in enumerate(catalog)}
    size = len(catalog)
    # Stable sort keeps first-seen order among equally frequent SIDs.
    observed = list(frequencies.keys())
    ordered = sorted(observed, key=lambda sid: frequencies[sid])
    cutoff = int(len(ordered) * tail_ratio)
    tail = set(ordered[:cutoff])

    u_counts: Counter[tuple[str, str]] = Counter()
    for items in user_items.values():
        for source in items:
            for neighbor in items:
                u_counts[(source, neighbor)] += 1

    return TrainingStatistics(
        catalog=tuple(catalog),
        lookup=lookup,
        frequencies=np.asarray([frequencies.get(sid, 0) for sid in catalog], dtype=np.float64),
        tail_mask=np.asarray([sid in tail for sid in catalog], dtype=bool),
        U=_csr_from_counts(u_counts, lookup, size),
        W=_csr_from_counts(w_counts, lookup, size),
        tail_ratio=tail_ratio,
    )


def save_statistics(stats: TrainingStatistics, path: str | Path) -> None:
    """Cache the statistics as a compressed .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            catalog=np.asarray(stats.catalog),
            frequencies=stats.frequencies,
            tail_mask=stats.tail_mask,
            tail_ratio=np.asarray(stats.tail_ratio),
            size=np.asarray(stats.size),
            u_data=stats.U.data,
            u_indices=stats.U.indices,
            u_indptr=stats.U.indptr,
            w_data=stats.W.data,
            w_indices=stats.W.indices,
            w_indptr=stats.W.indptr,
        )


def load_statistics(path: str | Path) -> TrainingStatistics:
    """Load a cache written by :func:`save_statistics`."""
    with np.load(path, allow_pickle=False) as packed:
        catalog = tuple(str(sid) for sid in packed["catalog"])
        size = int(packed["size"])
        U = SparseCounts(packed["u_data"], packed["u_indices"], packed["u_indptr"], size)
        W = SparseCounts(packed["w_data"], packed["w_indices"], packed["w_indptr"], size)
        return TrainingStatistics(
            catalog=catalog,
            lookup={sid: i for i, sid in enumerate(catalog)},
            frequencies=packed["frequencies"],
            tail_mask=packed["tail_mask"],
            U=U,
            W=W,
            tail_ratio=float(packed["tail_ratio"]),
        )
