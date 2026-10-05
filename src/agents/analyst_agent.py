"""
src/agents/analyst_agent.py — Data Analyst agent for SHAP + literature interpretation.

Role: Interpret the SHAP explanation and retrieved medical literature.
Input: prediction, explanation, chunks
Output: structured analysis with key findings and confidence assessment.
"""
from __future__ import annotations
from typing import Any
from src.agents.base_agent import BaseAgent
from src.rag.prompt_builder import risk_summary_lines


class AnalystAgent(BaseAgent):
    role_name = "Analyst"

    system_prompt = """You are a medical data analyst specializing in mammographic risk assessment.

Your task is to interpret SHAP feature attributions and correlate them with retrieved medical literature.

Requirements:
1. Explain which features most influenced the prediction and why
2. Cite specific findings from the retrieved documents
3. Assess confidence level (high/moderate/low) based on feature alignment with medical literature
4. Identify any discrepancies between the model's reasoning and established clinical guidelines
5. Be precise and data-driven — quote SHAP values and document sources

Format your response as:
ANALYSIS:
[Your interpretation of the SHAP-driven prediction]

KEY FINDINGS:
- [Finding 1 with citation]
- [Finding 2 with citation]

CONFIDENCE: [High/Moderate/Low] — [brief justification]

CLINICAL ALIGNMENT: [How well the model's reasoning aligns with retrieved guidelines]"""

    def build_user_message(self, context: dict[str, Any]) -> str:
        prediction = context.get("prediction", {})
        explanation = context.get("explanation", {})
        chunks = context.get("chunks", [])

        prob = prediction.get("probability", 0)
        risk_cat = prediction.get("risk_category", "Unknown")
        shap_text = explanation.get("shap_text", "")
        dominant = explanation.get("dominant_feature", "")
        attrs = explanation.get("feature_attributions", {})

        chunks_text = "\n\n".join(
            f"[{c.get('source', 'unknown')}]: {c.get('text', '')[:300]}"
            for c in chunks[:4]
        )

        return f"""PREDICTION:
{risk_summary_lines(prediction)}
Risk category: {risk_cat}
Predicted class: {prediction.get('predicted_class', 'Unknown')}

SHAP EXPLANATION:
{shap_text}
Dominant feature: {dominant}
Feature attributions: {attrs}

RETRIEVED MEDICAL LITERATURE:
{chunks_text}

Analyze this prediction in the context of the retrieved medical literature."""

    def parse_response(self, raw_text: str) -> dict[str, Any]:
        result = {"analysis_text": raw_text}

        # Extract confidence level
        for level in ["High", "Moderate", "Low"]:
            if f"CONFIDENCE: {level}" in raw_text or f"CONFIDENCE:{level}" in raw_text:
                result["confidence_level"] = level
                break
        else:
            result["confidence_level"] = "Moderate"

        return result
