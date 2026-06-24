"""
src/audit/anonymizer.py — Data anonymization utilities for HIPAA-aware exports.

Provides age bucketing, session ID hashing, and full DataFrame anonymization
for safe data export and sharing.

Usage:
    from src.audit.anonymizer import anonymize_features, anonymize_export
    anon = anonymize_features({"age": 45, "shape": 3, "margin": 4, "density": 2})
    anon_df = anonymize_export(predictions_df)
"""
from __future__ import annotations

import hashlib
from typing import Any


def anonymize_age(age: int) -> str:
    """
    Bucket age into 10-year ranges for de-identification.

    18-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80+
    """
    if age < 30:
        return "18-29"
    elif age < 40:
        return "30-39"
    elif age < 50:
        return "40-49"
    elif age < 60:
        return "50-59"
    elif age < 70:
        return "60-69"
    elif age < 80:
        return "70-79"
    else:
        return "80+"


def hash_identifier(raw_id: str, salt: str = "breastcare_ai") -> str:
    """
    SHA-256 hash an identifier for de-identification.
    Returns first 16 hex characters for readability.
    """
    combined = f"{salt}:{raw_id}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def anonymize_features(features: dict) -> dict:
    """
    Anonymize a feature dictionary.

    - Age is bucketed into ranges
    - All other features pass through (they're ordinal codes, not PII)
    """
    anon = features.copy()
    if "age" in anon:
        anon["age_bucket"] = anonymize_age(int(anon["age"]))
        del anon["age"]
    return anon


def anonymize_prediction(record: dict) -> dict:
    """
    Anonymize a full prediction record from the audit database.

    - Age bucketed
    - Session ID hashed
    - Timestamps truncated to date only
    - SHAP values retained (non-PII)
    """
    anon = record.copy()

    # Age bucketing
    if "age" in anon:
        anon["age_bucket"] = anonymize_age(int(anon["age"]))
        del anon["age"]

    # Session ID hashing
    if "session_id" in anon:
        anon["session_id"] = hash_identifier(anon["session_id"])

    # Timestamp truncation (keep date only)
    for ts_field in ("timestamp", "session_start"):
        if ts_field in anon and anon[ts_field]:
            anon[ts_field] = anon[ts_field][:10]  # YYYY-MM-DD only

    return anon


def anonymize_export(records: list[dict]) -> list[dict]:
    """
    Anonymize a list of prediction records for safe export.
    """
    return [anonymize_prediction(r) for r in records]
