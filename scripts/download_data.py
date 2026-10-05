"""
scripts/download_data.py — Fetch and verify the real UCI Mammographic Mass dataset.

The verified raw file is committed (data/raw/mammographic_masses.data), so this is
only needed to re-create or re-verify it. Two independent official sources must agree:

  1. `ucimlrepo.fetch_ucirepo(id=161)` — UCI's dataset API (data + metadata)
  2. the original raw file from the UCI archive — pinned by SHA-256

The script saves the original raw file unmodified, and only if its checksum matches
and its contents equal the ucimlrepo data cell for cell (values and missing-value
positions). On any failure it exits non-zero and writes nothing — it never
generates substitute data.

Run:  python scripts/download_data.py           # verify / fetch if missing
      python scripts/download_data.py --force   # re-download and overwrite
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    MAMMO_UCI_ID, MAMMO_RAW_PATH, MAMMO_SOURCE_URL, MAMMO_SHA256, MAMMO_RAW_COLUMNS,
)

# Published by UCI for dataset 161 (variable info, "Missing Attribute Values",
# "Class Distribution").
EXPECTED_ROWS = 961
EXPECTED_MISSING = {"birads": 2, "age": 5, "shape": 31, "margin": 48,
                    "density": 76, "severity": 0}
EXPECTED_CLASSES = {0: 516, 1: 445}


def fetch_ucimlrepo() -> pd.DataFrame:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        sys.exit("ERROR: ucimlrepo is not installed. Run `pip install ucimlrepo`.")
    ds = fetch_ucirepo(id=MAMMO_UCI_ID)
    df = ds.data.original.copy()
    df.columns = MAMMO_RAW_COLUMNS
    print(f"ucimlrepo: '{ds.metadata['name']}' (DOI {ds.metadata['dataset_doi']}), "
          f"{len(df)} rows")
    return df


def parse_raw(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), header=None,
                       names=MAMMO_RAW_COLUMNS, na_values=["?"])


def check(raw: pd.DataFrame, api: pd.DataFrame) -> list[str]:
    """Return a list of problems; empty means the raw file is verified."""
    problems = []
    if raw.shape != api.shape:
        problems.append(f"shape {raw.shape} != ucimlrepo shape {api.shape}")
    else:
        if not (raw.isna().to_numpy() == api.isna().to_numpy()).all():
            problems.append("missing-value positions differ from ucimlrepo")
        if not np.array_equal(raw.fillna(-1).to_numpy(float), api.fillna(-1).to_numpy(float)):
            problems.append("values differ from ucimlrepo")
    if len(raw) != EXPECTED_ROWS:
        problems.append(f"{len(raw)} rows, UCI documents {EXPECTED_ROWS}")
    missing = {c: int(n) for c, n in raw.isna().sum().items()}
    if missing != EXPECTED_MISSING:
        problems.append(f"missing counts {missing} != documented {EXPECTED_MISSING}")
    classes = {int(k): int(v) for k, v in raw["severity"].value_counts().items()}
    if classes != EXPECTED_CLASSES:
        problems.append(f"class counts {classes} != documented {EXPECTED_CLASSES}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch and verify UCI dataset 161.")
    parser.add_argument("--force", action="store_true",
                        help="re-download the raw file even if a copy exists")
    args = parser.parse_args()

    api = fetch_ucimlrepo()

    if MAMMO_RAW_PATH.exists() and not args.force:
        data = MAMMO_RAW_PATH.read_bytes()
        source = f"existing {MAMMO_RAW_PATH}"
    else:
        print(f"Downloading original raw file: {MAMMO_SOURCE_URL}")
        with urllib.request.urlopen(MAMMO_SOURCE_URL, timeout=30) as resp:
            data = resp.read()
        source = "downloaded file"

    digest = hashlib.sha256(data).hexdigest()
    problems = [] if digest == MAMMO_SHA256 else [
        f"sha256 {digest} != pinned {MAMMO_SHA256}"]
    problems += check(parse_raw(data), api)

    if problems:
        print(f"ERROR: {source} failed verification:")
        for p in problems:
            print(f"  - {p}")
        if source.startswith("existing"):
            print("Re-run with --force to replace it with the official file.")
        print("Nothing was written.")
        return 1

    if source == "downloaded file":
        MAMMO_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        MAMMO_RAW_PATH.write_bytes(data)
        print(f"Saved {len(data):,} bytes to {MAMMO_RAW_PATH}")
    print(f"OK: {source} verified (sha256 match; identical to ucimlrepo; "
          f"{EXPECTED_ROWS} rows; documented missing and class counts).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
