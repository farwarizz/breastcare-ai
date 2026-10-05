"""
src/agents/safety_agent.py — Safety Review agent for output validation.

Role: Review the clinician's response for diagnostic language violations.
Input: clinician's response + original prediction
Output: approved (possibly edited) response + safety audit log.
"""
from __future__ import annotations
from typing import Any
from src.agents.base_agent import BaseAgent
from src.rag.prompt_builder import risk_summary_lines


class SafetyAgent(BaseAgent):
    role_name = "Safety Reviewer"

    system_prompt = """You are a medical AI safety reviewer. Your sole job is to ensure the clinician's response does NOT contain diagnostic language or claims.

Review the response and check for these violations:
1. DIAGNOSTIC CLAIMS: "you have cancer", "this is cancer", "you are malignant", "this confirms malignancy"
2. TREATMENT DIRECTIVES: "you need surgery", "you need a biopsy", "you should start chemotherapy"
3. FALSE CERTAINTY: "definitely", "certainly", "100%", "guaranteed"
4. MISSING DISCLAIMER: Response must end with a clinician review statement
5. EMOTIONAL MANIPULATION: Excessive fear-mongering or false reassurance

Output format (STRICT):
SAFETY_STATUS: PASS or FAIL
VIOLATIONS_FOUND: [list any violations, or "None"]
EDITED_RESPONSE: [if FAIL, provide the corrected response. If PASS, copy the original response exactly]
AUDIT_NOTES: [brief safety audit summary]"""

    def build_user_message(self, context: dict[str, Any]) -> str:
        clinician_output = context.get("clinician_output", {})
        prediction = context.get("prediction", {})

        response_text = clinician_output.get("patient_explanation", "")
        prob = prediction.get("probability", 0)
        risk_cat = prediction.get("risk_category", "Unknown")

        return f"""CLINICIAN'S RESPONSE TO REVIEW:
---
{response_text}
---

CONTEXT:
{risk_summary_lines(prediction)}
Risk category: {risk_cat}

Review this response for safety violations. Check all 5 categories."""

    def parse_response(self, raw_text: str) -> dict[str, Any]:
        result = {
            "safety_text": raw_text,
            "passed": "SAFETY_STATUS: PASS" in raw_text.upper() or "SAFETY_STATUS:PASS" in raw_text.upper(),
            "violations": [],
        }

        # Extract edited response
        if "EDITED_RESPONSE:" in raw_text:
            parts = raw_text.split("EDITED_RESPONSE:")
            if len(parts) > 1:
                edited = parts[1]
                if "AUDIT_NOTES:" in edited:
                    edited = edited.split("AUDIT_NOTES:")[0]
                result["edited_response"] = edited.strip()

        # Extract violations
        if "VIOLATIONS_FOUND:" in raw_text:
            parts = raw_text.split("VIOLATIONS_FOUND:")
            if len(parts) > 1:
                violations_section = parts[1].split("EDITED_RESPONSE:")[0] if "EDITED_RESPONSE:" in parts[1] else parts[1]
                if "none" not in violations_section.lower():
                    result["violations"] = [
                        v.strip("- ").strip()
                        for v in violations_section.strip().split("\n")
                        if v.strip() and v.strip() != "-"
                    ]

        return result
