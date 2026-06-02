"""Fetch + inspect the HKUST rooftop PV dataset (Dryad doi:10.5061/dryad.m37pvmd99).

The archive is a single ``Dataset.zip`` (~296 MB, CC0) plus a ``README.md``. Dryad
guards its download endpoints with a bot challenge, so a fully head-less pull is
not guaranteed; the reliable path is a one-click browser download of ``Dataset.zip``
that you drop into ``data/raw/`` (or point at with ``--zip``). This script then
extracts it, recursively unzips any nested archives, and PRINTS the discovered
layout (file tree + one example header per distinct CSV schema + the Brick .ttl).
That printed layout is the spec that ``pvfc.data.load_station`` reads against.

Usage:
    python experiments/get_data.py                      # find/extract data/raw/Dataset.zip, inspect
    python experiments/get_data.py --zip ~/Downloads/Dataset.zip
    python experiments/get_data.py --inspect-only       # re-print the layout, no extraction
    python experiments/get_data.py --download            # best-effort API pull (may hit the bot wall)
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

DRYAD_DOI = "10.5061/dryad.m37pvmd99"
DRYAD_LANDING = f"https://datadryad.org/dataset/doi:{DRYAD_DOI}"
# Best-effort direct file stream (subject to Dryad's bot challenge):
DATASET_ZIP_STREAM = "https://datadryad.org/api/v2/files/3462931/download"
DATASET_ZIP_NAME = "Dataset.zip"


def _manual_instructions() -> str:
    return (
        "\nCould not download automatically (Dryad guards downloads with a bot "
        "challenge).\nDownload it manually instead:\n"
        f"  1. Open {DRYAD_LANDING}\n"
        f"  2. Click 'Download dataset' (or the {DATASET_ZIP_NAME} file)\n"
        f"  3. Move the downloaded {DATASET_ZIP_NAME} into {RAW_DIR}/\n"
        "  4. Re-run:  python experiments/get_data.py\n"
    )


def try_download(dest: Path) -> bool:
    """Best-effort streamed download. Returns True only on a real zip."""
    import requests  # already in the env

    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    try:
        with requests.get(DATASET_ZIP_STREAM, headers=headers, stream=True, timeout=60) as r:
            ctype = r.headers.get("Content-Type", "")
            if r.status_code != 200 or "zip" not in ctype:
                print(f"  download returned HTTP {r.status_code}, content-type {ctype!r}")
                return False
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {done / 1e6:6.0f} / {total / 1e6:.0f} MB", end="", flush=True)
            print()
    except Exception as exc:  # noqa: BLE001
        print(f"  download error: {exc}")
        return False
    # A real zip starts with the 'PK' magic bytes.
    with open(dest, "rb") as fh:
        if fh.read(2) != b"PK":
            print("  downloaded file is not a zip (likely a challenge page).")
            dest.unlink(missing_ok=True)
            return False
    return True


def locate_zip(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.exists() else None
    for cand in (RAW_DIR / DATASET_ZIP_NAME, RAW_DIR / "_download" / DATASET_ZIP_NAME):
        if cand.exists():
            return cand
    return None


def extract(zip_path: Path, raw_dir: Path) -> None:
    """Extract ``zip_path`` into ``raw_dir``, then recursively unzip nested zips."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path.name} -> {raw_dir} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    # one level of recursive unzip (e.g. per-category nested archives)
    for nested in list(raw_dir.rglob("*.zip")):
        if nested.resolve() == zip_path.resolve():
            continue
        target = nested.with_suffix("")
        if target.exists():
            continue
        print(f"  unzipping nested {nested.relative_to(raw_dir)} ...")
        with zipfile.ZipFile(nested) as zf:
            zf.extractall(nested.parent)


def _read_header(path: Path) -> str:
    try:
        with open(path, "r", errors="replace") as fh:
            return fh.readline().strip()
    except Exception as exc:  # noqa: BLE001
        return f"<unreadable: {exc}>"


def inspect(raw_dir: Path) -> None:
    """Print the file tree, distinct CSV headers, and Brick .ttl files."""
    if not any(raw_dir.iterdir()) if raw_dir.exists() else True:
        print(f"Nothing extracted in {raw_dir}.")
        return
    files = sorted(p for p in raw_dir.rglob("*") if p.is_file())
    print(f"\n=== File tree under {raw_dir} ({len(files)} files) ===")
    for p in files[:200]:
        print(f"  {p.relative_to(raw_dir)}  ({p.stat().st_size} B)")
    if len(files) > 200:
        print(f"  ... and {len(files) - 200} more")

    csvs = [p for p in files if p.suffix.lower() == ".csv"]
    print(f"\n=== Distinct CSV header signatures ({len(csvs)} CSVs) ===")
    seen: dict[str, Path] = {}
    for p in csvs:
        h = _read_header(p)
        if h not in seen:
            seen[h] = p
    for h, example in seen.items():
        print(f"  [{example.relative_to(raw_dir)}]")
        print(f"     {h}")

    ttls = [p for p in files if p.suffix.lower() in (".ttl", ".rdf", ".owl")]
    if ttls:
        print(f"\n=== Brick / RDF metadata ({len(ttls)}) ===")
        for p in ttls:
            print(f"  {p.relative_to(raw_dir)}  ({p.stat().st_size} B)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", help="path to an already-downloaded Dataset.zip")
    ap.add_argument("--raw-dir", default=str(RAW_DIR))
    ap.add_argument("--inspect-only", action="store_true", help="print layout; do not extract")
    ap.add_argument("--download", action="store_true", help="attempt a head-less download first")
    ap.add_argument("--force", action="store_true", help="re-extract even if data already present")
    args = ap.parse_args()
    raw_dir = Path(args.raw_dir)

    if args.inspect_only:
        inspect(raw_dir)
        return

    already = raw_dir.exists() and any(p.suffix.lower() == ".csv" for p in raw_dir.rglob("*"))
    if already and not args.force:
        print(f"Raw CSVs already present in {raw_dir}; inspecting (use --force to re-extract).")
        inspect(raw_dir)
        return

    zip_path = locate_zip(args.zip)
    if zip_path is None and args.download:
        print("Attempting head-less download from Dryad ...")
        cand = raw_dir / "_download" / DATASET_ZIP_NAME
        if try_download(cand):
            zip_path = cand
    if zip_path is None:
        print(_manual_instructions())
        sys.exit(1)

    extract(zip_path, raw_dir)
    inspect(raw_dir)
    print("\nDone. Set configs/experiment.yaml `data.site_id` to one of the discovered "
          "station IDs (see pvfc.data.list_stations).")


if __name__ == "__main__":
    main()
