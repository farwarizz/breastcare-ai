"""
src/safety.py — Deterministic safety filter applied to every LLM response.

Architectural decision: this is CODE-LEVEL enforcement, not just a system prompt note.
It fires regardless of what the LLM outputs — even under adversarial prompting.
"""
from __future__ import annotations
import re
import logging

logger = logging.getLogger(__name__)

# ── Diagnostic pattern replacements ──────────────────────────────
# Each tuple: (pattern, replacement)
# All patterns are case-insensitive.
DIAGNOSTIC_PATTERNS = [
    (r"\byou have (?:breast )?cancer\b",
     "the features you entered are associated with elevated risk"),
    (r"\bthis is (?:breast )?cancer\b",
     "this pattern is associated with elevated risk"),
    (r"\byou are malignant\b",
     "the estimated risk is elevated"),
    (r"\bthe (?:mass|lesion|finding) is malignant\b",
     "the estimated malignancy probability is elevated"),
    (r"\byou need surgery\b",
     "a clinician should review these findings"),
    (r"\byou need (?:a )?biopsy\b",
     "a clinician may recommend biopsy based on these findings"),
    (r"\byou should have (?:a )?biopsy\b",
     "a clinician may recommend biopsy based on these findings"),
    (r"\bthis confirms (?:cancer|malignancy)\b",
     "this estimate suggests elevated risk"),
    (r"\bdiagnos(?:is|ed|ed with) (?:breast )?cancer\b",
     "an elevated risk estimate"),
    (r"\bI (?:can )?confirm (?:cancer|malignancy)\b",
     "I can confirm an elevated probability estimate"),
    (r"\bthis is definitely cancer\b",
     "this pattern is statistically associated with elevated risk"),
    (r"\byou are (?:very )?likely to have cancer\b",
     "the probability estimate is elevated"),
]

# ── Standard disclaimer (appended to ALL responses) ──────────────
STANDARD_DISCLAIMER = (
    "\n\n---\n"
    "⚕️ **Clinical Review Required** — This is a risk estimate based on structured "
    "mammographic features, not a clinical diagnosis. All findings must be reviewed by "
    "a qualified radiologist or physician before any clinical decision is made. "
    "This tool is not FDA-cleared and is not intended for clinical use."
)

# ── High-risk disclaimer (appended when probability ≥ 0.65) ──────
HIGH_RISK_DISCLAIMER = (
    "\n\n🔴 **High-Risk Estimate** — The model has flagged this pattern as elevated risk. "
    "This does not mean a diagnosis of cancer. Please ensure these findings are reviewed "
    "promptly by a qualified clinician."
)


def filter_response(raw_response: str, probability: float) -> dict:
    """
    Apply safety filter to a raw LLM response.

    Parameters
    ----------
    raw_response : str — raw text from Claude
    probability  : float — malignancy probability from the ML model

    Returns
    -------
    dict:
        main_text    str  — filtered response body (before disclaimer separator)
        disclaimer   str  — disclaimer text
        high_risk    bool — whether high-risk notice was added
        filters_fired int — number of diagnostic patterns replaced
    """
    filtered = raw_response
    filters_fired = 0

    for pattern, replacement in DIAGNOSTIC_PATTERNS:
        new_text, count = re.subn(pattern, replacement, filtered,
                                   flags=re.IGNORECASE)
        if count > 0:
            logger.warning(
                f"Safety filter fired: pattern '{pattern}' replaced {count} time(s)"
            )
            filters_fired += count
        filtered = new_text

    # Build disclaimer
    disclaimer = STANDARD_DISCLAIMER
    high_risk = False
    if probability >= 0.65:
        disclaimer = HIGH_RISK_DISCLAIMER + STANDARD_DISCLAIMER
        high_risk = True

    # Split on separator if already present (LLM sometimes adds one)
    if "\n\n---\n" in filtered:
        main_text = filtered.split("\n\n---\n")[0].strip()
    else:
        main_text = filtered.strip()

    return {
        "main_text":    main_text,
        "disclaimer":   disclaimer.strip(),
        "high_risk":    high_risk,
        "filters_fired": filters_fired,
        "full_text":    main_text + disclaimer,
    }
