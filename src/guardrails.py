"""
src/guardrails.py — Input guardrails, applied before retrieval and generation.

Deterministic rules (regular expressions): cheap, testable, and they cannot be
talked out of their behaviour. Each user question is routed to one category:

  crisis        fixed supportive reply with crisis resources; the LLM is not called
  distress      LLM answer with a policy note: crisis resources first, no reassurance
  injection     fixed refusal (attempts to change the rules or reveal instructions)
  privacy       fixed refusal (the app holds no one else's records)
  off_topic     fixed scope reply (requests unrelated to breast imaging)
  diagnosis     LLM answer with a policy note: do not say whether it is cancer
  treatment     LLM answer with a policy note: no medication or treatment decisions
  skip_care     LLM answer with a policy note: do not endorse skipping recommended care
  ok            normal pipeline

The output filter in src/safety.py still runs on every LLM answer. Both layers
are measured by the red-team evaluation (src/evals/redteam.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_I = re.IGNORECASE

# Two tiers for thoughts of suicide or self-harm.
# CRISIS (fixed reply with crisis resources, no LLM call): explicit statements, following
# common suicide-screening language: wishing to be dead or not to wake up, ending one's
# life, others being better off, not being around, hoarding medication, self-harm in any
# tense. "can't go on" alone is too common in ordinary complaints ("I can't go on with these
# tests"), so it needs "anymore" / "like this".
_LIFE = r"(living|life|my life|carrying on|going on|go on|carry on|anything|it all)"
CRISIS = re.compile(
    rf"\b(kill(ing)? myself|(end|ending|take|taking) (it all|my (own )?life|things|everything)|"
    rf"suicid\w*|want(ed)? to die|(wish|rather) (i (was|were)|to be|be) dead|better off dead|"
    rf"don'?t want to (live|be alive|be here|go on|carry on|wake up)|self[- ]?harm\w*|"
    rf"(hurt|harm|cut)(ting|ing)? myself|i cut myself|no (reason|point) (to|in|of) (live|{_LIFE})|"
    rf"(don'?t|do not|can'?t|cannot) see (any|the|much) point (in|of) {_LIFE}|"
    rf"what'?s the point (of|in) {_LIFE}|nothing (left )?to live for|overdos\w*|"
    rf"(better|happier) (off )?without me|(won'?t|will not|wouldn'?t|not going to) be (around|here) "
    rf"(for|to see|anymore|any more|much longer)|"
    rf"(saving|stockpiling|hoarding|collecting) (up )?(my |all my )?(sleeping )?(pills|tablets|medication|meds)|"
    rf"take all (of )?my (pills|tablets|medication|meds)|can'?t go on (anymore|any more|like this))\b", _I)

# DISTRESS (the LLM answers, with a policy note that puts crisis resources first): softer
# signs of hopelessness that are not explicit enough for the fixed reply. A false alarm here
# only adds a check-in and the support lines to an otherwise normal answer.
DISTRESS = re.compile(
    r"\b(hopeless\w*|can'?t cope|cannot cope|can'?t (take|handle|bear) (it|this|any more|anymore)|"
    r"can'?t (do|face) (this|it) (anymore|any more)|give up on (everything|life|myself|living)|"
    r"thinking (about|of) (just )?disappearing|(want|wanted|need) to (just )?disappear|"
    r"just disappear(ing)? (for good|forever)|i'?m (just )?a burden|i am (just )?a burden|"
    r"falling apart|breaking down|panic attack\w*|can'?t stop crying|(no|any) point (anymore|any more))\b", _I)

# Attempts to change the assistant's rules, reveal its instructions or swap its role. Each
# alternative names the assistant's own instructions, so ordinary questions that mention
# "rules", "instructions" or "safety" ("show me the follow-up rules for BI-RADS 3") pass.
_TARGET = r"(all |any |the |your |my |these |those |previous |prior |above |earlier |initial |system |safety )*"
INJECTION = re.compile(
    rf"((ignore|disregard|forget|override|bypass)\s+{_TARGET}(instructions?|rules|prompts?|guidelines|"
    r"polic(y|ies)|restrictions|guardrails?|filters?)\b|system prompt|your (instructions|prompt|rules)\b|"
    r"developer mode|\bjailbreak|(?-i:\bDAN\b)|do anything now|"
    r"you are now (a|an|my|in|no longer|free|unrestricted|uncensored|jailbroken)\b|from now on,? you|"
    r"pretend (you are|to be|you're)|act as (an? )?(unrestricted|uncensored|different|real)|roleplay as|"
    r"(new|updated) (system )?(polic(y|ies)|rules|instructions)( for you|:| you must| to follow)|"
    r"<\s*/?\s*(system|admin|developer)\s*>|\[\s*(system|admin)\s*\]|"
    r"(reveal|print|show|repeat|output|display)\b.{0,20}\b(your|the system|system|hidden|initial|original|"
    r"developer|above)\s+(prompt|instructions|rules|message)|"
    r"(what|which) (rules|instructions|guidelines) (were|have) you (been )?(given|told|programmed)|"
    r"(respond|reply|answer) only (in|with) (json|xml|yaml|code))", _I)

# Requests for other people's data. Questions about what usually happens to other patients
# ("do other patients with similar results need a biopsy?") are allowed.
PRIVACY = re.compile(
    r"(((other|another) (patient|user|person)'?s|other (patients|users|people)'|someone else'?s|"
    r"everyone'?s|(last|previous) (patient|user|person)'?s)\s+(records?|data|results?|history|details|"
    r"names?|emails?|files?|scans?|reports?)|"
    r"\b(records?|data|results?|details|names?|emails?|files?|scans?)\s+(of|for|from|belonging to)\s+"
    r"(other|another|all (the )?|every|previous|the (last|previous)) (patients?|users?|people|persons?)|"
    r"\b(show|give|list|export|dump|download|send)\b.{0,30}\b(patients?|users?)\b.{0,30}"
    r"\b(records?|data|names|emails?|results?)|audit (log|database|db))", _I)

DOMAIN = re.compile(
    r"\b(breast|mammo\w*|bi-?rads|biops\w*|cancer|malignan\w*|benign|tumou?r|mass(es)?|lump|"
    r"margin|density|dense|spiculat\w*|calcif\w*|ultrasound|mri|screening|radiolog\w*|"
    r"oncolog\w*|risk|result|estimate|probability|shap|model|follow[- ]?up|cyst|"
    r"fibroadenoma|lesion|nipple|lymph|brca|tamoxifen|chemo\w*|radiation|surgery|"
    r"doctor|clinician|nurse|symptom|pain|scan|imaging|app|score)\b", _I)

OFF_TOPIC = re.compile(
    r"\b(poem|song|lyrics|joke|story|recipe|cook|(write|debug|fix)\b.{0,15}\bcode|program(ming)?|python|javascript|"
    r"capital of|weather|stock|crypto|bitcoin|football|cricket|movie|homework|essay|"
    r"translate|math problem|president|election)\b", _I)

DIAGNOSIS = re.compile(
    r"(\b(do|could|might) i have\b.{0,25}\b(cancer|malignan\w*|tumou?r)|"
    r"\b(have i got|am i)\b.{0,20}\b(cancer|malignant)|"
    r"\bis (it|this|mine|my (lump|mass|finding|result))\b.{0,20}\b(cancer|malignant|cancerous|benign)|"
    r"\bdiagnos(e|is)\b.{0,10}\b(me|my)\b|\b(confirm|rule out)\b.{0,20}\bcancer|"
    r"\b(tell|let) me (straight|honestly|if)\b.{0,30}\b(cancer|malignant)|"
    # prognosis and verdict requests: "am I going to die", "would you bet it's cancer"
    r"\b(am i|will i|i'?m|i am) (going to |gonna )?(die|dying|survive)\b|\bgoing to die\b|"
    r"\b(would|do|did|can|could) you (honestly |personally |really )?(bet|guess|think|say|believe|reckon)\b"
    r".{0,40}\b(cancer|malignant|benign|serious|die|nothing)|"
    r"\b(my|the) (lump|mass|finding|result|scan|mammogram)\b.{0,12}\b(is|was|looks?)\b.{0,15}"
    r"\b(cancer|cancerous|malignant|benign|nothing)\b|"
    r"\b(reassure|promise|assure) me\b.{0,40}\b(benign|not cancer|nothing serious|fine|ok(ay)?)\b|"
    r"\bis (it|this) (serious|deadly|fatal|life[- ]threatening)\b)", _I)

TREATMENT = re.compile(
    r"(\b(dose|dosage|how (much|many)|what strength|\d+\s?mg)\b.{0,40}\b(take|tamoxifen|letrozole|"
    r"anastrozole|exemestane|chemo\w*|medication|medicine|drug|pills?|tablets?|herceptin|trastuzumab)|"
    r"\bshould i (take|start|stop|try)\b.{0,30}\b(tamoxifen|letrozole|anastrozole|chemo\w*|"
    r"medication|medicine|drugs?|pills?|hormones?|supplements?|treatment)|"
    r"\b(prescribe|recommend)\b.{0,20}\b(me|a)\b.{0,20}\b(drug|medication|treatment|chemo\w*))", _I)

SKIP_CARE = re.compile(
    r"(\b(skip|cancel|avoid|postpone|delay|ignore|refuse|not (go|bother))\b.{0,30}"
    r"\b(biops\w*|follow[- ]?up|mammogram|appointment|doctor|radiologist|screening|surgery)|"
    r"\b(do|did) i (really )?(still )?need\b.{0,15}\b(a |the |my )?(biopsy|follow[- ]?up|surgery)|"
    r"\binstead of\b.{0,15}\b(a |my |the )?(biopsy|doctor|radiologist))", _I)

CRISIS_REPLY = (
    "I'm really sorry you're going through this. You don't have to face it alone. "
    "If you might act on thoughts of harming yourself, please contact your local emergency "
    "number right now. You can also talk to a crisis line: in the US, call or text 988 "
    "(Suicide & Crisis Lifeline); in the UK and Ireland, call Samaritans on 116 123; anywhere "
    "else, findahelpline.com lists free, confidential helplines in your country. Talking to "
    "someone you trust, or to your doctor or care team, can help too. Waiting for or receiving "
    "breast imaging results is genuinely stressful, and support is available."
)
FIXED_REPLIES = {
    "crisis": CRISIS_REPLY,
    "injection": ("I can't change how I work or share my instructions. I can explain breast-imaging "
                  "terms, the risk estimate on screen, or what usually happens next. Please ask about those."),
    "privacy": ("I don't have access to anyone's records and can't share personal data. This prototype "
                "only sees the four values you enter for your own estimate."),
    "off_topic": ("I can only help with breast imaging: BI-RADS terms, the risk estimate shown in this "
                  "app, and what usually happens after a mammogram."),
}
POLICY_NOTES = {
    "distress": ("The user may be expressing hopelessness, panic or thoughts of self-harm. Start by "
                 "acknowledging how they feel. Give support lines: their local emergency number; in the US, "
                 "call or text 988; in the UK and Ireland, Samaritans on 116 123; elsewhere, findahelpline.com. "
                 "Encourage them to talk to someone they trust or to their care team. Do not use the risk "
                 "estimate to reassure them. Only then answer any factual question, briefly."),
    "diagnosis": ("The user is asking for a diagnosis or a prognosis. Do not say or imply whether they have "
                  "cancer or how it will turn out, and do not call the finding benign or malignant. Explain "
                  "that the estimate is not a diagnosis, that a low estimate does not rule out cancer, and that "
                  "only a biopsy reviewed by a pathologist, together with their clinician, can answer that."),
    "treatment": ("The user is asking about medication or treatment decisions. Do not recommend, dose, "
                  "start or stop any treatment. Say these decisions belong to their oncology or care team."),
    "skip_care": ("The user is considering skipping or delaying recommended care. Do not endorse that. "
                  "Explain that this estimate cannot replace the follow-up or biopsy their clinician "
                  "recommends, and encourage them to discuss concerns with that clinician."),
}


@dataclass(frozen=True)
class GuardDecision:
    category: str
    fixed_reply: str | None = None      # set → answer with this text, skip the LLM
    policy_note: str | None = None      # set → add to the LLM prompt

    @property
    def calls_llm(self) -> bool:
        return self.fixed_reply is None


def check_input(question: str | None) -> GuardDecision:
    q = (question or "").strip()
    if not q:
        return GuardDecision("ok")
    if CRISIS.search(q):
        return GuardDecision("crisis", fixed_reply=FIXED_REPLIES["crisis"])
    # Distress outranks the refusals below: someone in distress should never get only a
    # scope or injection refusal.
    if DISTRESS.search(q):
        return GuardDecision("distress", policy_note=POLICY_NOTES["distress"])
    for category, pattern in (("injection", INJECTION), ("privacy", PRIVACY)):
        if pattern.search(q):
            return GuardDecision(category, fixed_reply=FIXED_REPLIES[category])
    if OFF_TOPIC.search(q) and not DOMAIN.search(q):
        return GuardDecision("off_topic", fixed_reply=FIXED_REPLIES["off_topic"])
    for category, pattern in (("diagnosis", DIAGNOSIS), ("treatment", TREATMENT), ("skip_care", SKIP_CARE)):
        if pattern.search(q):
            return GuardDecision(category, policy_note=POLICY_NOTES[category])
    return GuardDecision("ok")
