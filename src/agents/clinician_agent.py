"""
src/agents/clinician_agent.py — Clinician agent for patient-facing explanations.

Role: Generate empathetic, plain-language explanations for patients.
Input: analyst's output + prediction + patient context
Output: patient-friendly explanation with appropriate urgency calibration.
"""
from __future__ import annotations
from typing import Any
from src.agents.base_agent import BaseAgent
from src.rag.prompt_builder import risk_summary_lines


class ClinicianAgent(BaseAgent):
    role_name = "Clinician"

    system_prompt = """You are a compassionate breast care clinician explaining mammographic findings to a patient.

Your task is to translate the data analyst's technical interpretation into a clear, empathetic explanation.

Requirements:
1. Use plain language — no medical jargon without explanation
2. NEVER diagnose: say "the estimate suggests" not "you have"
3. Calibrate urgency appropriately:
   - Low risk: reassuring but encourage continued screening
   - Moderate risk: balanced, explain follow-up options
   - Elevated risk: empathetic but clear, emphasize clinical review
4. Explain what typically happens next (follow-up imaging, biopsy discussion, etc.)
5. Acknowledge the patient's likely emotional response
6. Keep response under 200 words
7. End with: "These findings must be reviewed by a qualified clinician."

NEVER state "you have cancer", "this is cancer", or imply a diagnosis."""

    def build_user_message(self, context: dict[str, Any]) -> str:
        prediction = context.get("prediction", {})
        analyst_output = context.get("analyst_output", {})
        user_question = context.get("user_question", "")

        prob = prediction.get("probability", 0)
        risk_cat = prediction.get("risk_category", "Unknown")
        analysis = analyst_output.get("analysis_text", "No analysis available.")
        confidence = analyst_output.get("confidence_level", "Moderate")

        question_section = ""
        if user_question:
            question_section = f"\nPATIENT QUESTION: {user_question}\nAddress this question specifically."

        return f"""DATA ANALYST'S FINDINGS:
{analysis}

PREDICTION SUMMARY:
{risk_summary_lines(prediction)}
Risk category: {risk_cat}
Analyst confidence: {confidence}
{question_section}

Write a clear, empathetic explanation for the patient. Remember: you are explaining a risk estimate, not making a diagnosis."""

    def parse_response(self, raw_text: str) -> dict[str, Any]:
        return {
            "patient_explanation": raw_text,
            "word_count": len(raw_text.split()),
        }
