"""CAPE: Co-occurrence-Aware Pool Expansion for generative recommenders."""

from .data import BeamRecord, Sample, load_beam_jsonl, load_split
from .metrics import LAMBDA_GRID, paired_bootstrap_ci, rank_of, select_lambda, summarize
from .rerank import CapeResult, cape_rank, memorization_channel, minmax
from .statistics import (
    TrainingStatistics,
    build_statistics,
    load_statistics,
    save_statistics,
)

__all__ = [
    "BeamRecord",
    "Sample",
    "load_beam_jsonl",
    "load_split",
    "TrainingStatistics",
    "build_statistics",
    "load_statistics",
    "save_statistics",
    "CapeResult",
    "cape_rank",
    "memorization_channel",
    "minmax",
    "LAMBDA_GRID",
    "rank_of",
    "summarize",
    "select_lambda",
    "paired_bootstrap_ci",
]
