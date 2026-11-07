"""Download the public raw datasets used in the paper.

This script downloads the
public sources listed in configs/data_sources.json (or any URL you pass with
--url) into data/raw/, and unpacks .gz / .zip archives.

The raw reviews still need the MiniOneRec preprocessing and SID-construction
steps to become the SID-level CSV splits consumed by prepare_data.py.

Usage:
    python scripts/download_data.py --list
    python scripts/download_data.py --dataset Industrial_and_Scientific
    python scripts/download_data.py --all
    python scripts/download_data.py --url https://example.org/data.zip --name LastFM1K
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = ROOT / "configs" / "data_sources.json"
DEFAULT_OUTPUT = Path("data/raw")


def _sources(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def download(url: str, destination: Path, sha256: str | None = None, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        print(f"skip (exists): {destination}")
        return destination
    partial = destination.with_name(destination.name + ".part")
    print(f"download: {url}")
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out)
    if sha256:
        digest = hashlib.sha256(partial.read_bytes()).hexdigest()
        if digest != sha256:
            partial.unlink()
            raise ValueError(f"SHA256 mismatch for {destination.name}")
    partial.replace(destination)
    print(f"saved: {destination}")
    return destination


def unpack(path: Path) -> None:
    if path.name.endswith(".gz"):
        target = path.with_name(path.name[: -len(".gz")])
        if target.exists():
            return
        with gzip.open(path, "rb") as source, target.open("wb") as out:
            shutil.copyfileobj(source, out)
        print(f"unpacked: {target}")
    elif path.suffix == ".zip":
        target = path.with_name(path.stem)
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                member_path = Path(member)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError(f"Unsafe archive member: {member}")
            archive.extractall(target)
        print(f"unpacked: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES, help="URL manifest")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="download directory")
    parser.add_argument("--dataset", action="append", default=[], help="dataset name (repeatable)")
    parser.add_argument("--all", action="store_true", help="download every dataset with a URL")
    parser.add_argument("--url", help="download one arbitrary URL")
    parser.add_argument("--name", help="output filename for --url")
    parser.add_argument("--sha256", help="optional checksum for --url")
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    parser.add_argument("--no-extract", action="store_true", help="keep archives compressed")
    parser.add_argument("--list", action="store_true", help="list configured datasets and exit")
    args = parser.parse_args()

    config = _sources(args.sources)
    datasets = config["datasets"]

    if args.list:
        for name, entry in datasets.items():
            status = entry["url"] or entry.get("note", "no URL configured")
            print(f"{name}: {status}")
        return

    jobs: list[tuple[str, str, str | None]] = []
    if args.url:
        filename = args.name or args.url.rsplit("/", 1)[-1].split("?")[0] or "download"
        jobs.append((filename, args.url, args.sha256))
    else:
        names = list(datasets) if args.all else args.dataset
        if not names:
            parser.error("pass --dataset, --all, --url, or --list")
        for name in names:
            if name not in datasets:
                parser.error(f"unknown dataset {name!r}; use --list")
            entry = datasets[name]
            if not entry.get("url"):
                print(f"no URL configured for {name}: {entry.get('note', '')}", file=sys.stderr)
                continue
            jobs.append((entry.get("filename") or name, entry["url"], None))

    for filename, url, checksum in jobs:
        path = download(url, args.output_dir / filename, checksum, args.force)
        if not args.no_extract:
            unpack(path)


if __name__ == "__main__":
    main()
