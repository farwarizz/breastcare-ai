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


# ── Answer sanitising (Phase 2.2; current pipeline only, never the red-team baseline) ──
# Held-out v2 failures these address (v2 is now development data):
#   - the case estimate volunteered in answers that never asked about it (v2pv03 even gave
#     it as "your sister's" numbers; also the cause of every unfaithful answer, c04/c08/r16/u15)
#   - a citation to a document that does not exist ("[0]" in v2cr12)
#   - internal instruction text repeated to the user (rule 8 verbatim in v2cr07)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9*\"'(\[⚕])")
FOOTER = "⚕️ These findings must be reviewed by a qualified clinician before any decision is made."


def drop_sentences(text: str, should_drop) -> tuple[str, int]:
    """Remove every sentence for which should_drop(sentence) is true; line breaks, bullets and
    the remaining sentences are kept. Returns (text, sentences removed)."""
    lines, dropped = [], 0
    for line in text.split("\n"):
        if not line.strip():
            lines.append(line)
            continue
        sentences = _SENTENCE_SPLIT.split(line)
        kept = [s for s in sentences if not should_drop(s)]
        dropped += len(sentences) - len(kept)
        if kept:
            lines.append(" ".join(kept))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip(), dropped


def has_substance(text: str) -> bool:
    """False when nothing but the fixed footer (or nothing at all) is left."""
    return bool(text.replace(FOOTER, "").strip(" \n-*"))


# The user asks about their own estimate or finding (or there is no question: the initial
# explanation of the estimate). Only then may an answer talk about the case estimate.
ASKS_ABOUT_ESTIMATE = re.compile(
    r"\b(my|mine|this|our)\s+(own\s+)?(estimate|risk|result|results|score|probability|chance|odds|category|"
    r"prediction|predictions|numbers?|percentage|percent|mass|lump|finding|lesion|case|mammogram|scan|report|"
    r"assessment|shap|features?|margin|shape|density)\b|"
    r"\b(estimate|prediction set|predicted class|risk category|risk score|conformal|calibrat\w*|shap|"
    r"probabilit(y|ies)|percentage)\b|%|"
    r"\b(am|do|will|could|might|should) i (have|get|be|need|worry)\b|"
    r"\b(is|was) (it|this|mine) (cancer|cancerous|malignant|benign|serious|bad|dangerous)\b", re.IGNORECASE)

# A sentence that states or qualifies the case estimate.
_CASE_ESTIMATE = (r"prediction set|predicted class|risk category|\b(your|this|the|my) (risk |model |case )?estimate\b|"
                  r"calibrated|conformal|\{\s*(benign|malignant)|statistical estimate|low estimate does not rule out|"
                  r"high one does not confirm|malignancy probability|model probability|\b(low|moderate|elevated) "
                  r"risk category|(placing|places|puts|falls) (you|it|this|the (case|finding|mass))\b.{0,20}\b"
                  r"(low|moderate|elevated)")


def asks_about_estimate(question: str | None) -> bool:
    return not (question or "").strip() or bool(ASKS_ABOUT_ESTIMATE.search(question))


def remove_unrequested_estimate(text: str, question: str | None, probability: float) -> tuple[str, int]:
    """Drop sentences about the case estimate when the question did not ask about it."""
    if asks_about_estimate(question):
        return text, 0
    # The appended caveat goes as a unit: its second sentence ("Only a biopsy ... can do that.")
    # means nothing without the first.
    removed_caveat = ESTIMATE_CAVEAT in text
    text = text.replace(ESTIMATE_CAVEAT, "")
    pct = round(probability * 100, 1)
    numbers = {f"{pct:g}", f"{round(pct):d}"}
    pattern = re.compile(_CASE_ESTIMATE + "|" + "|".join(rf"\b{re.escape(n)}\s?%" for n in numbers), re.IGNORECASE)
    text, dropped = drop_sentences(text, lambda s: bool(pattern.search(s)))
    return text, dropped + 2 * removed_caveat


# Phrases the assistant is instructed to say to users; they also occur in the instructions,
# so they must not count as leaked instruction text.
_SAY_TO_USER = [
    ESTIMATE_CAVEAT, FOOTER,
    "it is a statistical estimate, not a diagnosis, that a low estimate does not rule out cancer, and that "
    "only a biopsy reviewed by a pathologist can confirm or exclude cancer",
    "this is a statistical estimate, not a diagnosis. A low estimate does not rule out cancer, and only a biopsy "
    "reviewed by a pathologist can confirm or exclude cancer",
    "only a biopsy reviewed by a pathologist, together with their clinician, can answer that",
    "only a biopsy reviewed by a pathologist, together with your clinician, can answer that",
    "results must be reviewed by a qualified radiologist or physician",
    "the provided documents do not cover it", "the documents do not cover it",
    "their local emergency number; in the US, call or text 988; in the UK and Ireland, Samaritans on 116 123; "
    "elsewhere, findahelpline.com", "talk to someone they trust or to their care team",
    "this estimate cannot replace the follow-up or biopsy their clinician recommends",
    "these decisions belong to their oncology or care team", "these decisions belong to your oncology or care team",
    # first-person refusals and self-descriptions that paraphrase a rule without exposing it
    "I cannot recommend, dose, start or stop any treatment or medication",
    "I cannot recommend, dose, start or stop any medication or treatment",
    "BreastCare AI, a decision-support prototype for breast cancer risk estimation",
    "I provide educational information about mammographic risk estimates",
    "if you are having thoughts of suicide or self-harm",
]
LEAK_NGRAM = 6


def _words(text: str) -> list[str]:
    # Digits are ignored: a citation "[4]" next to a rule number "4." is not a leak.
    return re.findall(r"[a-z]+", text.lower())


def _ngrams(words: list[str], n: int = LEAK_NGRAM) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


_internal_ngrams_cache: set | None = None


def internal_ngrams() -> set[tuple[str, ...]]:
    """Word 6-grams of the system prompt, task wording and policy notes, minus the phrases the
    assistant is told to say to users."""
    global _internal_ngrams_cache
    if _internal_ngrams_cache is None:
        from src.guardrails import POLICY_NOTES
        from src.rag.prompt_builder import SYSTEM_PROMPT, TASK_TEXTS
        internal = set()
        for text in [SYSTEM_PROMPT, *POLICY_NOTES.values(), *TASK_TEXTS]:
            internal |= _ngrams(_words(text))
        allowed = set()
        for text in _SAY_TO_USER:
            allowed |= _ngrams(_words(text))
        _internal_ngrams_cache = internal - allowed
    return _internal_ngrams_cache


def leaks_internal_text(sentence: str) -> bool:
    return bool(_ngrams(_words(sentence)) & internal_ngrams())


def remove_internal_text(text: str) -> tuple[str, int]:
    """Drop sentences that repeat the assistant's instructions (6 or more consecutive words)."""
    return drop_sentences(text, leaks_internal_text)


_CITATION_GROUP = re.compile(r"(\s?)\[(\d{1,2}(?:\s*[,;–-]\s*\d{1,2})*)\]")
_NON_DOCUMENT_CITATION = re.compile(r"\s?\[(case context|context|case|patient context|risk estimate|estimate|"
                                    r"source|sources|citation needed|ref|reference)\]", re.IGNORECASE)


def _citation_numbers(group: str) -> list[int]:
    nums = []
    for part in re.split(r"\s*[,;]\s*", group):
        bounds = [int(x) for x in re.split(r"\s*[–-]\s*", part)]
        lo, hi = bounds[0], bounds[-1]
        nums += list(range(lo, hi + 1)) if 0 <= hi - lo < 10 else bounds
    return nums


def strip_invalid_citations(text: str, n_documents: int) -> str:
    """Remove citation markers that point to no provided document ([0], [7] of 5, [Case context]);
    a group with valid and invalid numbers keeps only the valid ones."""
    def fix(m):
        nums = _citation_numbers(m.group(2))
        valid = [x for x in dict.fromkeys(nums) if 1 <= x <= n_documents]
        if len(valid) == len(nums):
            return m.group(0)
        return (m.group(1) + "".join(f"[{v}]" for v in valid)) if valid else ""
    return _NON_DOCUMENT_CITATION.sub("", _CITATION_GROUP.sub(fix, text))


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
