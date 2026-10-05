"""
tests/fixtures/synthetic_mammographic.py — SYNTHETIC TEST DATA ONLY.

Generates fake rows in the raw UCI Mammographic Mass format, so the cleaning
code can be unit-tested without touching the real file. This used to be a
silent fallback inside the training script; it now lives only in the test
suite and must never be imported by production code (src/, server.py, scripts/).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RAW_COLUMNS = ["birads", "age", "shape", "margin", "density", "severity"]


def make_synthetic_mammographic_raw(n: int = 961, seed: int = 42,
                                    missing_rate: float = 0.03) -> pd.DataFrame:
    """Uniformly random codes, logistic severity, ~3% missing values per feature."""
    rng = np.random.default_rng(seed)
    birads  = rng.integers(1, 6, n).astype(float)
    age     = rng.integers(18, 96, n).astype(float)
    shape   = rng.integers(1, 5, n).astype(float)
    margin  = rng.integers(1, 6, n).astype(float)
    density = rng.integers(1, 5, n).astype(float)
    logit = (-2.5 + 0.04 * (age - 50) + 0.3 * (shape - 2.5)
             + 0.7 * (margin - 3.0) - 0.2 * (density - 2.5))
    severity = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame({"birads": birads, "age": age, "shape": shape,
                       "margin": margin, "density": density, "severity": severity})
    for col in ["age", "shape", "margin", "density"]:
        df.loc[rng.uniform(size=n) < missing_rate, col] = np.nan
    return df[RAW_COLUMNS]


def to_raw_uci_bytes(df: pd.DataFrame) -> bytes:
    """Serialise like the UCI file: no header, '?' for missing values."""
    return df.to_csv(header=False, index=False, na_rep="?").encode()
