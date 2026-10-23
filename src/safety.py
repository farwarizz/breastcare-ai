"""
src/safety.py — Deterministic safety filter applied to every LLM response.

Architectural decision: this is CODE-LEVEL enforcement, not just a system prompt note.
It fires regardless of what the LLM outputs — even under adversarial prompting.
"""
from __future__ import annotations
import re
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import HIGH_RISK_THRESHOLD

logger = logging.getLogger(__name__)

# ── Diagnostic pattern replacements ──────────────────────────────
# Each tuple: (pattern, replacement)
# All patterns are case-insensitive.
# LEGACY_PATTERNS is the original (Phase 1) filter, unchanged: the red-team baseline uses
# it. DIAGNOSTIC_PATTERNS (the current filter) starts from it with one rule fixed and adds
# new ones.
LEGACY_PATTERN_COUNT = 12
LEGACY_PATTERNS = [
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
assert len(LEGACY_PATTERNS) == LEGACY_PATTERN_COUNT

# The legacy "diagnosed with cancer" rule also fired on third-person text ("as great as
# those reported by women diagnosed with breast cancer" became "... by women an elevated
# risk estimate"), so the current filter rewrites only statements about the user.
_LEGACY_DIAGNOSED = r"\bdiagnos(?:is|ed|ed with) (?:breast )?cancer\b"
DIAGNOSTIC_PATTERNS = [p for p in LEGACY_PATTERNS if p[0] != _LEGACY_DIAGNOSED] + [
    (r"\b(?:you(?:'ve| have| are| were)(?: been)? diagnosed with|your diagnosis of) (?:breast )?cancer\b",
     "the estimate is elevated, which is not a diagnosis"),
    (r"\byou (?:definitely |probably )?(?:do not|don't) have (?:breast )?cancer\b",
     "this estimate cannot rule cancer in or out"),
    # a verdict about this user's finding: "it is likely benign", "your mass is benign", "it's cancer"
    (r"\b(?:it|this|that|your (?:mass|lump|finding|lesion)|the (?:mass|lump|finding|lesion))"
     r"(?:'s| is) (?:(?:definitely|probably|likely|almost certainly|most likely|clearly) )?benign\b",
     "the estimated probability of malignancy is low, but only a clinician can confirm the finding"),
    (r"\b(?:it|this|that|your (?:mass|lump|finding|lesion))(?:'s| is) (?:(?:definitely|probably|likely|"
     r"almost certainly|most likely|clearly) )?(?:malignant|cancerous|(?:breast )?cancer)\b",
     "the estimated malignancy probability is elevated, but only a biopsy can confirm the finding"),
    (r"\byou have (?:a |an )?(?:\w+ )?(?:malignant|cancerous) (?:tumou?r|mass|lesion|growth)\b",
     "the features you entered are associated with elevated risk"),
    # treatment and dosing directives (only when the object is a medication or treatment)
    (r"\byou should (?:take|start|stop|increase|decrease|switch to|try) (?:the |your |a |an |some )?"
     r"(?:\d+\s?mg\b|tamoxifen|letrozole|anastrozole|exemestane|raloxifene|chemo\w*|medication|"
     r"medicine|drugs?|pills?|tablets?|hormones?|supplements?|treatment|therapy|herbal\w*)[^.]*",
     "decisions about medication belong to your care team"),
    (r"(?<!not )(?<!n't )\b(?:take|use) \d+(?:\.\d+)?\s?mg\b[^.]*",
     "dosing decisions belong to your care team"),
    # skipping recommended care
    (r"\byou (?:can|could|may) (?:safely )?(?:skip|cancel|avoid|delay|postpone) (?:the |your |a )?"
     r"(?:biopsy|follow-up|mammogram|appointment|screening)\b",
     "please follow the plan your clinician recommends"),
    (r"\byou (?:do not|don't) need (?:a |the |your )?(?:biopsy|follow-up|mammogram|appointment)\b",
     "only your clinician can decide whether further tests are needed"),
]

# ── Estimate caveat (current filter only) ────────────────────────
# Red-team failures dx07/hdx01: answers quoted the estimate or the {Benign} prediction set
# as reassurance without saying that only a biopsy decides. When an answer talks about the
# estimate but never mentions a biopsy, this sentence is appended.
ESTIMATE_TERMS = re.compile(
    r"\b(prediction set|predicted class|risk category|risk estimate|(your|the|this) estimate|"
    r"malignancy probability|calibrated (malignancy )?probability|(low|moderate|elevated) risk|"
    r"categori[sz]ed as (low|moderate|elevated))\b", re.IGNORECASE)
ESTIMATE_CAVEAT = ("This estimate is not a diagnosis: a low estimate does not rule out cancer, and a high "
                   "one does not confirm it. Only a biopsy reviewed by a pathologist can do that.")


def needs_estimate_caveat(text: str) -> bool:
    return bool(ESTIMATE_TERMS.search(text)) and not re.search(r"\bbiops", text, re.IGNORECASE)


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


def apply_patterns(text: str, patterns: list | None = None) -> tuple[str, int]:
    """Rewrite diagnostic, treatment and skip-care phrasing; returns (text, replacements)."""
    filtered, fired = text, 0
    current = patterns is None             # the baseline passes the 12 original patterns unchanged
    for pattern, replacement in (DIAGNOSTIC_PATTERNS if patterns is None else patterns):
        def _sub(m, replacement=replacement):
            # "whether you have cancer" / "if it is malignant" asserts nothing: leave it.
            if current and re.search(r"\b(?:whether|if)\s+$", m.string[: m.start()], re.IGNORECASE):
                return m.group(0)
            # Keep sentence case: after ". ", at a line start (also after a bullet, number or
            # bold marker), after a colon, or when the replaced text was capitalised itself.
            starts_sentence = (m.group(0)[:1].isupper() or re.search(
                r"(?:^|[.!?:]\s|\n)\s*(?:[-*•]\s*|\d+[.)]\s*)?(?:\*\*)?$", m.string[: m.start()]) is not None)
            return replacement[0].upper() + replacement[1:] if starts_sentence else replacement
        before = filtered
        filtered = re.sub(pattern, _sub, filtered, flags=re.IGNORECASE)
        count = sum(1 for m in re.finditer(pattern, before, flags=re.IGNORECASE)
                    if not (current and re.search(r"\b(?:whether|if)\s+$", before[: m.start()], re.IGNORECASE)))
        if count:
            logger.warning("Safety filter fired: pattern %r replaced %d time(s)", pattern, count)
            fired += count
    return filtered, fired


def filter_response(raw_response: str, probability: float, patterns: list | None = None) -> dict:
    """
    Apply safety filter to a raw LLM response.

    Parameters
    ----------
    raw_response : str — raw text from the LLM (Gemini)
    probability  : float — malignancy probability from the ML model

    Returns
    -------
    dict:
        main_text    str  — filtered response body (before disclaimer separator)
        disclaimer   str  — disclaimer text
        high_risk    bool — whether high-risk notice was added
        filters_fired int — number of diagnostic patterns replaced
    """
    filtered, filters_fired = apply_patterns(raw_response, patterns)

    # Build disclaimer
    disclaimer = STANDARD_DISCLAIMER
    high_risk = False
    if probability >= HIGH_RISK_THRESHOLD:
        disclaimer = HIGH_RISK_DISCLAIMER + STANDARD_DISCLAIMER
        high_risk = True

    # Split on separator if already present (LLM sometimes adds one)
    if "\n\n---\n" in filtered:
        main_text = filtered.split("\n\n---\n")[0].strip()
    else:
        main_text = filtered.strip()

    caveat_added = patterns is None and needs_estimate_caveat(main_text)   # not in the baseline
    if caveat_added:
        main_text = f"{main_text}\n\n{ESTIMATE_CAVEAT}"

    return {
        "main_text":    main_text,
        "disclaimer":   disclaimer.strip(),
        "high_risk":    high_risk,
        "filters_fired": filters_fired,
        "caveat_added": caveat_added,
        "full_text":    main_text + disclaimer,
    }
