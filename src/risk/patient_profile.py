"""
src/risk/patient_profile.py — Extended patient profile data model.

Captures mammographic features (existing) plus clinical history for
classical risk model computation: family history, BRCA status, prior
biopsies, breast density category, and reproductive factors.

Usage:
    from src.risk.patient_profile import PatientProfile
    profile = PatientProfile(
        age=55, shape=4, margin=5, density=2,
        family_history=True, brca_status="Unknown",
    )
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class BrcaStatus(str, Enum):
    UNKNOWN = "Unknown"
    NEGATIVE = "Negative"
    BRCA1 = "BRCA1"
    BRCA2 = "BRCA2"


class DensityCategory(str, Enum):
    """ACR Breast Density Categories."""
    A = "A"  # Almost entirely fatty
    B = "B"  # Scattered fibroglandular
    C = "C"  # Heterogeneously dense
    D = "D"  # Extremely dense
    UNKNOWN = "Unknown"


@dataclass
class PatientProfile:
    """
    Extended patient profile for comprehensive risk assessment.

    Core mammographic features (required):
        age, shape, margin, density — same as the ML model input

    Extended clinical history (optional):
        family_history — first-degree relative with breast cancer
        brca_status — genetic mutation status
        prior_biopsies — number of previous breast biopsies
        breast_density_category — ACR density A/B/C/D
        age_at_menarche — age at first menstruation
        age_at_first_birth — age at first live birth (0 = nulliparous)
        hormone_replacement — currently using HRT
    """
    # Core features (required)
    age: int = 50
    shape: int = 2
    margin: int = 3
    density: int = 2

    # Extended clinical history (optional)
    family_history: bool = False
    num_first_degree_relatives: int = 0
    brca_status: str = "Unknown"
    prior_biopsies: int = 0
    atypical_hyperplasia: bool = False
    breast_density_category: str = "Unknown"
    age_at_menarche: int = 12
    age_at_first_birth: int = 25
    hormone_replacement: bool = False

    def get_core_features(self) -> dict[str, int]:
        """Return the 4 mammographic features for ML prediction."""
        return {
            "age": self.age,
            "shape": self.shape,
            "margin": self.margin,
            "density": self.density,
        }

    def has_extended_history(self) -> bool:
        """Check if any extended fields have been filled."""
        return (
            self.family_history
            or self.brca_status != "Unknown"
            or self.prior_biopsies > 0
            or self.breast_density_category != "Unknown"
            or self.hormone_replacement
        )

    def get_risk_factors_summary(self) -> list[str]:
        """Return a human-readable list of active risk factors."""
        factors = []
        if self.age >= 55:
            factors.append(f"Age {self.age} (post-menopausal range)")
        if self.family_history:
            rel = self.num_first_degree_relatives or 1
            factors.append(f"Family history ({rel} first-degree relative{'s' if rel > 1 else ''})")
        if self.brca_status in ("BRCA1", "BRCA2"):
            factors.append(f"Genetic mutation: {self.brca_status}")
        if self.prior_biopsies > 0:
            factors.append(f"{self.prior_biopsies} prior biopsy/biopsies")
        if self.atypical_hyperplasia:
            factors.append("Atypical ductal hyperplasia (ADH)")
        if self.breast_density_category in ("C", "D"):
            factors.append(f"Dense breasts (ACR category {self.breast_density_category})")
        if self.age_at_menarche < 12:
            factors.append(f"Early menarche (age {self.age_at_menarche})")
        if self.age_at_first_birth > 30 or self.age_at_first_birth == 0:
            factors.append("Late first birth or nulliparity")
        if self.hormone_replacement:
            factors.append("Hormone replacement therapy (HRT)")
        return factors

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PatientProfile":
        """Create from dictionary (ignores unknown keys)."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
