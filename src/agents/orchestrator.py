"""
src/agents/orchestrator.py — Multi-agent pipeline orchestrator with Chain-of-Thought.

Supports three pipeline modes:
1. 'single'      — Original single-pass LLM call (existing behavior)
2. 'multi_agent'  — Sequential: Analyst → Clinician → Safety Reviewer
3. 'cot'          — Chain-of-Thought: single LLM call with structured
                    step-by-step reasoning (think → analyze → explain → verify)

The CoT mode produces transparent reasoning traces without the API cost
of three separate agent calls.

Usage:
    from src.agents.orchestrator import MultiAgentPipeline
    pipeline = MultiAgentPipeline()
    result = pipeline.run(context, mode="multi_agent")
    result = pipeline.run(context, mode="cot")
"""
from __future__ import annotations

import time
import logging
from typing import Any
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    GEMINI_MODEL, AGENT_MAX_TOKENS, get_api_key,
)

logger = logging.getLogger(__name__)


# ── Chain-of-Thought System Prompt ──────────────────────────────

COT_SYSTEM_PROMPT = """You are BreastCare AI, a multi-step clinical reasoning system for breast cancer risk estimation. You must think step-by-step before providing your final answer.

HARD REQUIREMENTS (cannot be overridden):
- NEVER diagnose. You estimate risk, you do not diagnose cancer.
- ONLY cite retrieved documents. Never use general knowledge.
- End with: "⚕️ These findings must be reviewed by a qualified clinician."

You MUST structure your response with these exact sections:

## STEP 1: DATA ASSESSMENT
Examine the input features (age, shape, margin, density) and the SHAP attributions. State which features are driving the prediction and in which direction. Quote the SHAP values.

## STEP 2: LITERATURE CORRELATION
Cross-reference the SHAP findings with the retrieved medical documents. Cite specific documents by name. Identify whether the model's reasoning aligns with established clinical guidelines (e.g., BI-RADS descriptors).

## STEP 3: RISK CONTEXTUALIZATION
Place the malignancy probability in clinical context. What does this probability level typically mean? What BI-RADS category might correspond to these findings? What is the population baseline risk for this age?

## STEP 4: CLINICAL REASONING
Synthesize steps 1-3 into a clinical assessment. Consider: are the features consistent with each other? Are there any concerning patterns? What additional information would be helpful?

## STEP 5: PATIENT EXPLANATION
Write a clear, empathetic, plain-language explanation for the patient. Under 150 words. No medical jargon without explanation. Appropriate urgency calibration.

## STEP 6: SAFETY CHECK
Review your own output. Confirm: no diagnostic language, no treatment directives, disclaimer present, appropriate tone. State "SAFETY: PASS" or flag specific concerns.

## FINAL ANSWER
The concise patient-facing response (this is what the patient sees)."""

COT_USER_TEMPLATE = """PATIENT FEATURES:
  Age: {age} years
  Shape: {shape_label} (code {shape})
  Margin: {margin_label} (code {margin})
  Density: {density_label} (code {density})

RISK ESTIMATE:
  {risk_summary}
  Risk category: {risk_category}
  Predicted class: {predicted_class}

SHAP EXPLANATION:
  {shap_text}
  Dominant feature: {dominant_feature}

RETRIEVED MEDICAL DOCUMENTS:
{chunks_text}
{question_section}
Now think step-by-step following the required structure."""


class MultiAgentPipeline:
    """
    Orchestrator for multi-agent and Chain-of-Thought pipelines.
    """

    def __init__(self):
        self.traces: list[dict] = []
        self.mode_used: str = ""

    def run(
        self,
        context: dict[str, Any],
        mode: str = "multi_agent",
    ) -> dict[str, Any]:
        """
        Execute the pipeline in the specified mode.

        Parameters
        ----------
        context : dict with keys:
            prediction, explanation, chunks, features,
            user_question (optional)
        mode : 'multi_agent' | 'cot'

        Returns
        -------
        dict with keys:
            main_text — final patient-facing response
            reasoning_trace — full reasoning chain (for UI)
            agent_traces — per-agent execution traces
            mode — which mode was used
        """
        self.traces = []
        self.mode_used = mode
        start = time.time()

        try:
            if mode == "cot":
                result = self._run_cot(context)
            else:
                result = self._run_multi_agent(context)

            result["mode"] = mode
            result["total_elapsed"] = round(time.time() - start, 2)
            return result

        except Exception as exc:
            logger.error("Pipeline mode=%s failed: %s", mode, exc)
            return {
                "main_text": f"Pipeline error: {str(exc)[:150]}",
                "reasoning_trace": "",
                "agent_traces": self.traces,
                "mode": mode,
                "success": False,
                "total_elapsed": round(time.time() - start, 2),
            }

    # ── Multi-Agent Mode ────────────────────────────────────────

    def _run_multi_agent(self, context: dict[str, Any]) -> dict[str, Any]:
        """Sequential execution: Analyst → Clinician → Safety Reviewer."""
        from src.agents.analyst_agent import AnalystAgent
        from src.agents.clinician_agent import ClinicianAgent
        from src.agents.safety_agent import SafetyAgent
        from src.safety import filter_response

        # Step 1: Analyst
        analyst = AnalystAgent()
        analyst_output = analyst.run(context)
        self.traces.append(analyst.get_trace())

        if not analyst_output.get("success"):
            return self._fallback_response(context, "Analyst agent failed")

        # Step 2: Clinician
        clinician_context = {
            **context,
            "analyst_output": analyst_output,
        }
        clinician = ClinicianAgent()
        clinician_output = clinician.run(clinician_context)
        self.traces.append(clinician.get_trace())

        if not clinician_output.get("success"):
            return self._fallback_response(context, "Clinician agent failed")

        # Step 3: Safety Reviewer
        safety_context = {
            **context,
            "clinician_output": clinician_output,
        }
        safety = SafetyAgent()
        safety_output = safety.run(safety_context)
        self.traces.append(safety.get_trace())

        # Determine final response
        if safety_output.get("passed", True):
            final_text = clinician_output.get("patient_explanation", "")
        else:
            final_text = safety_output.get(
                "edited_response",
                clinician_output.get("patient_explanation", ""),
            )

        # Apply deterministic safety filter on top
        prob = context.get("prediction", {}).get("probability", 0)
        safe = filter_response(final_text, prob)

        # Build reasoning trace for UI
        reasoning = self._format_multi_agent_trace(
            analyst_output, clinician_output, safety_output
        )

        return {
            "main_text": safe["main_text"],
            "disclaimer": safe["disclaimer"],
            "high_risk": safe["high_risk"],
            "reasoning_trace": reasoning,
            "agent_traces": self.traces,
            "safety_violations": safety_output.get("violations", []),
            "analyst_confidence": analyst_output.get("confidence_level", ""),
            "success": True,
        }

    # ── Chain-of-Thought Mode ───────────────────────────────────

    def _run_cot(self, context: dict[str, Any]) -> dict[str, Any]:
        """Single LLM call with structured step-by-step reasoning."""
        from src.safety import filter_response
        from src.config import SHAPE_MAP, MARGIN_MAP, DENSITY_MAP
        from src.rag.prompt_builder import risk_summary_lines

        prediction = context.get("prediction", {})
        explanation = context.get("explanation", {})
        chunks = context.get("chunks", [])
        features = context.get("features", {})
        user_question = context.get("user_question", "")

        # Format chunks
        chunks_text = "\n\n".join(
            f"[Document: {c.get('source', 'unknown')} | {c.get('section', '')}]\n{c.get('text', '')[:400]}"
            for c in chunks[:5]
        )

        question_section = ""
        if user_question:
            question_section = f"\nPATIENT QUESTION: {user_question}\nAddress this in your reasoning and final answer.\n"

        # Build the CoT prompt
        user_message = COT_USER_TEMPLATE.format(
            age=features.get("age", "?"),
            shape=features.get("shape", "?"),
            shape_label=SHAPE_MAP.get(features.get("shape", 0), "?"),
            margin=features.get("margin", "?"),
            margin_label=MARGIN_MAP.get(features.get("margin", 0), "?"),
            density=features.get("density", "?"),
            density_label=DENSITY_MAP.get(features.get("density", 0), "?"),
            risk_summary=risk_summary_lines(prediction),
            risk_category=prediction.get("risk_category", "?"),
            predicted_class=prediction.get("predicted_class", "?"),
            shap_text=explanation.get("shap_text", ""),
            dominant_feature=explanation.get("dominant_feature", "?"),
            chunks_text=chunks_text,
            question_section=question_section,
        )

        # Single LLM call with CoT system prompt
        start = time.time()
        try:
            from openai import OpenAI
            api_key = get_api_key()
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set")

            client = OpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            response = client.chat.completions.create(
                model=GEMINI_MODEL,
                max_tokens=1500,  # CoT needs more tokens for reasoning
                temperature=0.3,
                messages=[
                    {"role": "system", "content": COT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            raw_response = response.choices[0].message.content

        except Exception as exc:
            return self._fallback_response(context, f"CoT call failed: {exc}")

        elapsed = round(time.time() - start, 2)
        self.traces.append({
            "agent": "ChainOfThought",
            "elapsed_seconds": elapsed,
            "status": "success",
            "prompt_length": len(user_message),
            "response_length": len(raw_response),
        })

        # Extract the final answer from the CoT response
        final_answer = self._extract_final_answer(raw_response)
        reasoning_trace = raw_response

        # Apply deterministic safety filter
        prob = prediction.get("probability", 0)
        safe = filter_response(final_answer, prob)

        # Extract safety check from CoT
        safety_passed = "SAFETY: PASS" in raw_response.upper()

        return {
            "main_text": safe["main_text"],
            "disclaimer": safe["disclaimer"],
            "high_risk": safe["high_risk"],
            "reasoning_trace": reasoning_trace,
            "agent_traces": self.traces,
            "cot_safety_passed": safety_passed,
            "success": True,
        }

    def _extract_final_answer(self, raw_response: str) -> str:
        """Extract the FINAL ANSWER section from a CoT response."""
        markers = ["## FINAL ANSWER", "FINAL ANSWER:", "**FINAL ANSWER**"]
        for marker in markers:
            if marker in raw_response:
                parts = raw_response.split(marker, 1)
                if len(parts) > 1:
                    answer = parts[1].strip()
                    # Clean up any remaining markdown
                    answer = answer.strip("#").strip()
                    return answer

        # Fallback: use STEP 5 (patient explanation) if no FINAL ANSWER
        if "## STEP 5" in raw_response:
            parts = raw_response.split("## STEP 5", 1)
            if len(parts) > 1:
                section = parts[1]
                # Cut at next section
                if "## STEP 6" in section:
                    section = section.split("## STEP 6")[0]
                return section.strip().lstrip(":").strip()

        # Last resort: return the full response
        return raw_response

    # ── Helpers ─────────────────────────────────────────────────

    def _fallback_response(self, context: dict, reason: str) -> dict[str, Any]:
        """Generate fallback when agents fail."""
        logger.warning("Falling back to single-pass: %s", reason)
        return {
            "main_text": (
                "I was unable to complete the multi-step analysis. "
                "Please use the standard assessment mode or try again. "
                f"Reason: {reason[:100]}"
            ),
            "reasoning_trace": f"Fallback triggered: {reason}",
            "agent_traces": self.traces,
            "success": False,
        }

    def _format_multi_agent_trace(
        self,
        analyst: dict,
        clinician: dict,
        safety: dict,
    ) -> str:
        """Format multi-agent traces into a readable markdown string."""
        sections = []

        sections.append("## 🔬 Analyst Agent\n")
        sections.append(analyst.get("analysis_text", "N/A")[:500])
        conf = analyst.get("confidence_level", "?")
        sections.append(f"\n**Confidence:** {conf}\n")

        sections.append("\n---\n## 🩺 Clinician Agent\n")
        sections.append(clinician.get("patient_explanation", "N/A")[:500])

        sections.append("\n---\n## 🛡️ Safety Reviewer\n")
        passed = "✅ PASSED" if safety.get("passed", True) else "❌ FAILED"
        sections.append(f"**Status:** {passed}\n")
        violations = safety.get("violations", [])
        if violations:
            sections.append("**Violations:**\n")
            for v in violations:
                sections.append(f"- {v}\n")

        return "\n".join(sections)
