"""I/O for SID-level interaction files and pre-computed backbone outputs.
"""
from __future__ import annotations

import ast
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping


def normalize_sid(value: Any) -> str:
    """Normalize a semantic item ID without changing its tokens."""
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        value = value[0]
    return str(value).strip().strip('"').strip()


def parse_sid_list(value: Any) -> list[str]:
    """Parse a stringified list of SIDs, falling back to comma splitting."""
    if isinstance(value, (list, tuple)):
        return [normalize_sid(item) for item in value]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = [part.strip() for part in text.split(",") if part.strip()]
    if not isinstance(parsed, (list, tuple)):
        parsed = [parsed]
    return [normalize_sid(item) for item in parsed]


def parse_catalog(info_file: str | Path | None) -> list[str]:
    """Read the catalog from a tab-separated info file (SID in the first column)."""
    if info_file is None:
        return []
    catalog: list[str] = []
    seen: set[str] = set()
    with Path(info_file).open("r", encoding="utf-8") as handle:
        for line in handle:
            sid = line.split("\t", 1)[0].strip()
            if sid and sid not in seen:
                seen.add(sid)
                catalog.append(sid)
    return catalog


@dataclass(frozen=True)
class Sample:
    sample_id: int
    user_id: str
    history: tuple[str, ...]
    target: str


def load_split(path: str | Path) -> list[Sample]:
    """Load a train/valid/test CSV split in MiniOneRec SID format."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"user_id", "history_item_sid", "item_sid"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        return [
            Sample(
                sample_id=i,
                user_id=str(row["user_id"]),
                history=tuple(parse_sid_list(row["history_item_sid"])),
                target=normalize_sid(row["item_sid"]),
            )
            for i, row in enumerate(reader)
        ]


@dataclass(frozen=True)
class BeamRecord:
    """One backbone request: its 50-item beam and frozen-model likelihoods.

    ``extra_ll`` holds teacher-forced log-likelihoods for tail candidates
    outside the beam, either as a ``{sid: value}`` mapping or as a dense
    vector over the whole catalog. CAPE only reads the values it needs.
    """

    sample_id: int
    user_id: str
    history: tuple[str, ...]
    target: str
    candidates: tuple[str, ...]
    ll: tuple[float, ...]
    extra_ll: Mapping[str, float] | Any
    anchor: str | None = None


def _json_lines(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        first = handle.readline().lstrip()
        handle.seek(0)
        if first.startswith("["):
            yield from json.load(handle)
        else:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def load_beam_jsonl(path: str | Path) -> list[BeamRecord]:
    """Load beam outputs from a JSON array or a JSONL file.

    Each record has ``sample_id``, ``user_id``, ``history``, ``target``,
    ``candidates`` (50 unique SIDs), ``ll`` (their log-likelihoods),
    optional ``anchor`` and ``extra_ll``.
    """
    records = []
    for raw in _json_lines(path):
        candidates = tuple(normalize_sid(c) for c in raw["candidates"])
        ll = tuple(float(v) for v in raw["ll"])
        if not candidates or len(candidates) != len(ll):
            raise ValueError(f"Candidate/LL length mismatch at sample {raw.get('sample_id')}")
        if len(set(candidates)) != len(candidates):
            raise ValueError(f"Duplicate beam candidates at sample {raw.get('sample_id')}")
        if not all(math.isfinite(v) for v in ll):
            raise ValueError(f"Non-finite beam LL at sample {raw.get('sample_id')}")
        records.append(
            BeamRecord(
                sample_id=int(raw.get("sample_id", len(records))),
                user_id=str(raw.get("user_id", "")),
                history=tuple(parse_sid_list(raw.get("history", []))),
                target=normalize_sid(raw["target"]),
                candidates=candidates,
                ll=ll,
                extra_ll=raw.get("extra_ll", {}),
                anchor=raw.get("anchor"),
            )
        )
    if not records:
        raise ValueError(f"No beam records in {path}")
    return records
