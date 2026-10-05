"""
src/ml/data.py — Load, verify, clean and split the UCI Mammographic Mass data.

Training only runs on the official UCI file, verified by SHA-256 (see
data/README.md). If the file is missing or altered, loading stops with an
error; it never substitutes generated data.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    CLEANED_CSV, FEATURE_NAMES,
    SHAPE_MAP, MARGIN_MAP, DENSITY_MAP,
    MAMMO_UCI_ID, MAMMO_RAW_PATH, MAMMO_SOURCE_URL, MAMMO_SHA256,
    MAMMO_DOI, MAMMO_RAW_COLUMNS,
)

# Data split (Phase 1). The hold-out was re-drawn with a new seed so that no
# test row was ever used for any tuning decision.
SPLIT_SEED         = 2026
TEST_FRACTION      = 0.20   # of all rows    → final evaluation only
CONFORMAL_FRACTION = 0.25   # of the dev rows → conformal calibration only

# Documented UCI codes; anything else means the file is not what we expect.
_VALID_CODES = {
    "shape":    set(SHAPE_MAP),     # 1–4 (nominal)
    "margin":   set(MARGIN_MAP),    # 1–5 (nominal)
    "density":  set(DENSITY_MAP),   # 1–4 (ordinal)
    "severity": {0, 1},             # 0 = benign, 1 = malignant
}


def verify_raw_file(path: Path = MAMMO_RAW_PATH) -> str:
    """
    Return the SHA-256 of the raw file. Raises if the file is missing or is
    not byte-identical to the official UCI file.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Fetch the real UCI file with "
            "`python scripts/download_data.py` (training never substitutes generated data)."
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MAMMO_SHA256:
        raise ValueError(
            f"{path} is not the official UCI Mammographic Mass file "
            f"(sha256 {digest}, expected {MAMMO_SHA256}). Refusing to train on it. "
            "Re-fetch it with `python scripts/download_data.py --force`."
        )
    return digest


def read_raw_mammographic(path: Path = MAMMO_RAW_PATH) -> pd.DataFrame:
    """Read the headerless UCI file; '?' marks a missing value."""
    return pd.read_csv(path, header=None, names=MAMMO_RAW_COLUMNS, na_values=["?"])


def clean_mammographic(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Features: age (years), shape, margin, density (integer codes).
    Target: severity (0 = benign, 1 = malignant).
    BI-RADS is dropped: it is the radiologist's assessment, so using it as a
    feature would leak the conclusion into the model.
    Rows with any missing feature are dropped (codes cannot be meaningfully imputed).
    The raw row index is kept, so BI-RADS can still be looked up as a reference.
    Raises ValueError if any value falls outside the documented UCI codes.
    """
    df = raw.drop(columns=["birads"]).dropna()
    df = df.astype({"shape": int, "margin": int, "density": int, "severity": int})

    problems = [
        f"{col}: unexpected values {sorted(df.loc[~df[col].isin(valid), col].unique())}"
        for col, valid in _VALID_CODES.items()
        if not df[col].isin(valid).all()
    ]
    if not df["age"].between(1, 120).all():
        problems.append("age: values outside 1-120")
    if problems:
        raise ValueError("Mammographic data failed validation — " + "; ".join(problems))

    return df[FEATURE_NAMES + ["severity"]]


def load_mammographic(path: Path = MAMMO_RAW_PATH):
    """
    Verify, load and clean the UCI Mammographic Mass file; save cleaned.csv.
    Returns (cleaned DataFrame indexed by raw row, provenance dict).
    """
    digest = verify_raw_file(path)
    print(f"  Loading mammographic data from {path} (sha256 verified)")
    raw = read_raw_mammographic(path)
    df = clean_mammographic(raw)

    print(f"  Raw: {len(raw)} rows  |  Cleaned: {len(df)} rows  |  "
          f"Malignant: {int(df.severity.sum())}  |  Benign: {int((df.severity == 0).sum())}")
    df.to_csv(CLEANED_CSV, index=False)

    provenance = {
        "name":         "UCI Mammographic Mass",
        "uci_id":       MAMMO_UCI_ID,
        "doi":          MAMMO_DOI,
        "license":      "CC BY 4.0",
        "source_url":   MAMMO_SOURCE_URL,
        "raw_sha256":   digest,
        "raw_rows":     int(len(raw)),
        "raw_missing":  {col: int(n) for col, n in raw.isna().sum().items()},
        "clean_rows":   int(len(df)),
        "benign":       int((df.severity == 0).sum()),
        "malignant":    int(df.severity.sum()),
        "features":     list(FEATURE_NAMES),
        "excluded_columns": {"birads": "radiologist's assessment (label leakage)"},
        "target":       "severity: 0 = benign, 1 = malignant",
    }
    return df, provenance


def split_rows(df: pd.DataFrame, seed: int = SPLIT_SEED) -> dict[str, np.ndarray]:
    """
    Stratified three-way split of the cleaned rows (raw-row indices):
      test   — 20%: final evaluation only, touched once
      fit    — 60%: cross-validated model selection, out-of-fold calibration and
               threshold choice, and training the final models
      conformal — 20%: conformal calibration only
    """
    y = df["severity"].to_numpy()
    dev, test = train_test_split(df.index.to_numpy(), test_size=TEST_FRACTION,
                                 stratify=y, random_state=seed)
    fit, conformal = train_test_split(dev, test_size=CONFORMAL_FRACTION,
                                      stratify=df.loc[dev, "severity"], random_state=seed)
    return {"fit": np.sort(fit), "conformal": np.sort(conformal), "test": np.sort(test)}
