"""
src/risk/classical_risk.py — Classical breast cancer risk estimation models.

Implements simplified versions of validated clinical risk models:
- Gail Model (NCI): 5-year and lifetime risk from demographic/reproductive factors
- Tyrer-Cuzick (IBIS): Includes genetic and density information
- Population baseline: SEER age-specific incidence rates

These complement the ML prediction by providing established clinical benchmarks.

Usage:
    from src.risk.classical_risk import gail_risk, tyrer_cuzick_risk, population_baseline
    from src.risk.patient_profile import PatientProfile
    profile = PatientProfile(age=55, family_history=True)
    gail = gail_risk(profile)
"""
from __future__ import annotations

import math
import logging
from typing import Any
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

# ── SEER Age-Specific Incidence Rates (per 100,000 women/year) ───
# Source: SEER Cancer Statistics Review, 2015-2019
SEER_INCIDENCE = {
    20: 1.9, 25: 1.9, 30: 25.1, 35: 25.1,
    40: 122.5, 45: 122.5, 50: 229.9, 55: 229.9,
    60: 350.7, 65: 350.7, 70: 412.3, 75: 412.3,
    80: 390.0, 85: 390.0, 90: 350.0,
}

# Average US female lifetime risk: 12.9% (approximately 1 in 8)
BASELINE_LIFETIME_RISK = 0.129


def _age_bucket(age: int) -> int:
    """Round age down to nearest 5-year bucket."""
    return max(20, min(90, (age // 5) * 5))


def population_baseline(age: int) -> dict[str, Any]:
    """
    Return population-average risk for a given age.
    Based on SEER data for US women.
    """
    bucket = _age_bucket(age)
    annual_rate = SEER_INCIDENCE.get(bucket, 200.0)
    five_year_risk = 1 - (1 - annual_rate / 100000) ** 5
    remaining_years = max(90 - age, 1)
    lifetime_risk = 1 - (1 - annual_rate / 100000) ** remaining_years

    # Cap lifetime risk at reasonable bounds
    lifetime_risk = min(lifetime_risk, 0.30)

    return {
        "model": "Population Baseline (SEER)",
        "five_year_risk": round(five_year_risk, 4),
        "five_year_pct": round(five_year_risk * 100, 2),
        "lifetime_risk": round(lifetime_risk, 4),
        "lifetime_pct": round(lifetime_risk * 100, 2),
        "annual_incidence_per_100k": annual_rate,
        "risk_category": _categorize_risk(lifetime_risk),
    }


def gail_risk(profile: Any) -> dict[str, Any]:
    """
    Simplified Gail Model (NCI Breast Cancer Risk Assessment Tool).

    Estimates 5-year and lifetime risk using:
    - Age
    - Age at menarche
    - Age at first live birth
    - Number of first-degree relatives with breast cancer
    - Number of prior breast biopsies
    - Atypical hyperplasia status

    This is a simplified implementation for educational purposes.
    The full Gail model uses race-specific hazard rates and competing
    mortality — this version uses approximate relative risk multipliers.
    """
    age = getattr(profile, "age", 50)
    menarche = getattr(profile, "age_at_menarche", 12)
    first_birth = getattr(profile, "age_at_first_birth", 25)
    n_relatives = getattr(profile, "num_first_degree_relatives", 0)
    if getattr(profile, "family_history", False) and n_relatives == 0:
        n_relatives = 1
    n_biopsies = getattr(profile, "prior_biopsies", 0)
    atypia = getattr(profile, "atypical_hyperplasia", False)

    # Relative risk multipliers (approximate, from Gail 1989 / NCI literature)
    rr = 1.0

    # Age at menarche
    if menarche < 12:
        rr *= 1.10
    elif menarche > 14:
        rr *= 0.90

    # Age at first birth
    if first_birth == 0:  # nulliparous
        rr *= 1.25
    elif first_birth > 30:
        rr *= 1.20
    elif first_birth < 20:
        rr *= 0.85

    # Number of first-degree relatives
    if n_relatives == 1:
        rr *= 1.80
    elif n_relatives >= 2:
        rr *= 3.20

    # Number of biopsies
    if n_biopsies == 1:
        rr *= 1.30
    elif n_biopsies >= 2:
        rr *= 1.70

    # Atypical hyperplasia
    if atypia:
        rr *= 2.00

    # Baseline 5-year risk by age (approximate from BCRAT)
    bucket = _age_bucket(age)
    baseline_annual = SEER_INCIDENCE.get(bucket, 200.0) / 100000
    baseline_5yr = 1 - (1 - baseline_annual) ** 5

    # Apply relative risk
    five_year_risk = min(1 - (1 - baseline_5yr) ** rr, 0.50)

    # Lifetime risk (simplified: project forward from current age to 90)
    remaining_years = max(90 - age, 1)
    lifetime_risk = min(1 - (1 - baseline_annual * rr) ** remaining_years, 0.80)

    return {
        "model": "Gail Model (Simplified)",
        "five_year_risk": round(five_year_risk, 4),
        "five_year_pct": round(five_year_risk * 100, 2),
        "lifetime_risk": round(lifetime_risk, 4),
        "lifetime_pct": round(lifetime_risk * 100, 2),
        "relative_risk": round(rr, 2),
        "risk_category": _categorize_risk(lifetime_risk),
        "factors_used": {
            "age": age,
            "age_at_menarche": menarche,
            "age_at_first_birth": first_birth,
            "first_degree_relatives": n_relatives,
            "prior_biopsies": n_biopsies,
            "atypical_hyperplasia": atypia,
        },
    }


def tyrer_cuzick_risk(profile: Any) -> dict[str, Any]:
    """
    Simplified Tyrer-Cuzick (IBIS) Model.

    More comprehensive than Gail — additionally includes:
    - BRCA1/BRCA2 mutation status
    - Breast density category (ACR A-D)
    - Hormone replacement therapy
    - Second-degree family history (not implemented in this version)

    This is an approximation; the full TC model uses Monte Carlo simulation.
    """
    age = getattr(profile, "age", 50)
    brca = getattr(profile, "brca_status", "Unknown")
    density_cat = getattr(profile, "breast_density_category", "Unknown")
    hrt = getattr(profile, "hormone_replacement", False)

    # Start with Gail-like baseline
    gail_result = gail_risk(profile)
    rr = gail_result["relative_risk"]

    # BRCA status adjustment
    if brca == "BRCA1":
        rr *= 4.5  # ~55-72% lifetime risk
    elif brca == "BRCA2":
        rr *= 3.5  # ~45-69% lifetime risk

    # Breast density adjustment
    density_rr = {
        "A": 0.80,  # fatty — lower risk
        "B": 1.00,  # scattered — average
        "C": 1.20,  # heterogeneously dense
        "D": 2.00,  # extremely dense
    }
    if density_cat in density_rr:
        rr *= density_rr[density_cat]

    # HRT adjustment
    if hrt:
        rr *= 1.25

    # Compute risks
    bucket = _age_bucket(age)
    baseline_annual = SEER_INCIDENCE.get(bucket, 200.0) / 100000
    baseline_5yr = 1 - (1 - baseline_annual) ** 5

    five_year_risk = min(1 - (1 - baseline_5yr) ** rr, 0.60)
    remaining_years = max(90 - age, 1)
    lifetime_risk = min(1 - (1 - baseline_annual * rr) ** remaining_years, 0.90)

    return {
        "model": "Tyrer-Cuzick (IBIS, Simplified)",
        "five_year_risk": round(five_year_risk, 4),
        "five_year_pct": round(five_year_risk * 100, 2),
        "lifetime_risk": round(lifetime_risk, 4),
        "lifetime_pct": round(lifetime_risk * 100, 2),
        "relative_risk": round(rr, 2),
        "risk_category": _categorize_risk(lifetime_risk),
        "factors_used": {
            **gail_result["factors_used"],
            "brca_status": brca,
            "breast_density_category": density_cat,
            "hormone_replacement": hrt,
        },
    }


def compute_all_risks(profile: Any, ml_probability: float = 0.0) -> dict[str, Any]:
    """
    Compute all risk estimates for a patient profile.
    Returns a unified comparison dict.
    """
    pop = population_baseline(profile.age)
    gail = gail_risk(profile)
    tc = tyrer_cuzick_risk(profile)

    return {
        "population_baseline": pop,
        "gail_model": gail,
        "tyrer_cuzick": tc,
        "ml_estimate": {
            "model": "BreastCare AI (calibrated model estimate)",
            "probability": round(ml_probability, 4),
            "probability_pct": round(ml_probability * 100, 2),
            "risk_category": _categorize_ml_risk(ml_probability),
        },
        "risk_factors": profile.get_risk_factors_summary() if hasattr(profile, "get_risk_factors_summary") else [],
    }


def _categorize_risk(lifetime_risk: float) -> str:
    """Categorize lifetime risk per NCCN guidelines."""
    if lifetime_risk < 0.15:
        return "Average"
    elif lifetime_risk < 0.20:
        return "Moderate"
    else:
        return "High"


def _categorize_ml_risk(probability: float) -> str:
    """Categorize ML malignancy probability."""
    if probability < 0.35:
        return "Low"
    elif probability < 0.65:
        return "Moderate"
    else:
        return "Elevated"
