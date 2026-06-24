"""
src/audit/pdf_report.py — Generate professional PDF reports for predictions.

Creates a branded PDF document containing the full assessment:
patient features, risk estimate, SHAP explanation, AI narrative, and disclaimers.

Usage:
    from src.audit.pdf_report import generate_pdf_report
    pdf_bytes = generate_pdf_report(prediction, explanation, features, llm_response)
"""
from __future__ import annotations

import io
import base64
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    SHAPE_MAP, MARGIN_MAP, DENSITY_MAP,
    REPORTS_DIR,
)

logger = logging.getLogger(__name__)


def generate_pdf_report(
    prediction: dict,
    explanation: dict,
    features: dict,
    llm_response: dict | None = None,
    session_id: str = "",
    patient_profile: dict | None = None,
) -> bytes:
    """
    Generate a professional PDF report for a risk assessment.

    Returns PDF as bytes (suitable for download or file write).
    """
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Header ──────────────────────────────────────────────────
    pdf.set_fill_color(14, 28, 47)  # navy
    pdf.rect(0, 0, 210, 38, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(15, 10)
    pdf.cell(0, 8, "BreastCare AI", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_xy(15, 20)
    pdf.cell(0, 5, "Risk Assessment Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_xy(15, 27)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pdf.cell(0, 5, f"Generated: {timestamp}  |  Session: {session_id[:8]}...",
             new_x="LMARGIN", new_y="NEXT")

    # Disclaimer bar
    pdf.set_fill_color(200, 68, 93)  # rose
    pdf.rect(0, 38, 210, 8, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 7)
    pdf.set_xy(15, 39)
    pdf.cell(0, 6, "NOT FOR CLINICAL USE  |  RESEARCH PROTOTYPE  |  NOT FDA-CLEARED  |  REQUIRES CLINICIAN REVIEW",
             new_x="LMARGIN", new_y="NEXT")

    pdf.set_y(52)

    # ── Section 1: Patient Features ─────────────────────────────
    _section_header(pdf, "1. Patient Features")

    shape_label = SHAPE_MAP.get(features.get("shape", 0), str(features.get("shape")))
    margin_label = MARGIN_MAP.get(features.get("margin", 0), str(features.get("margin")))
    density_label = DENSITY_MAP.get(features.get("density", 0), str(features.get("density")))

    feature_data = [
        ["Feature", "Value", "Type"],
        ["Age", f"{features.get('age')} years", "Continuous"],
        ["Mass Shape", f"{shape_label} (code {features.get('shape')})", "Ordinal 1-4"],
        ["Mass Margin", f"{margin_label} (code {features.get('margin')})", "Ordinal 1-5"],
        ["Mass Density", f"{density_label} (code {features.get('density')})", "Ordinal 1-4"],
    ]
    _render_table(pdf, feature_data)

    if patient_profile:
        pdf.ln(4)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(100, 100, 100)
        extras = []
        if patient_profile.get("family_history"):
            extras.append("Family history: Yes")
        if patient_profile.get("brca_status", "Unknown") != "Unknown":
            extras.append(f"BRCA: {patient_profile['brca_status']}")
        if extras:
            pdf.cell(0, 4, "Extended: " + " | ".join(extras), new_x="LMARGIN", new_y="NEXT")

    # ── Section 2: Risk Assessment ──────────────────────────────
    _section_header(pdf, "2. Risk Assessment")

    prob = prediction["probability"]
    pct = round(prob * 100, 1)
    risk_cat = prediction["risk_category"]

    # Risk color
    if risk_cat == "Low":
        pdf.set_fill_color(239, 246, 255)
        risk_rgb = (29, 78, 216)
    elif risk_cat == "Moderate":
        pdf.set_fill_color(255, 251, 235)
        risk_rgb = (180, 83, 9)
    else:
        pdf.set_fill_color(255, 241, 244)
        risk_rgb = (200, 68, 93)

    pdf.set_draw_color(*risk_rgb)
    pdf.rect(15, pdf.get_y(), 180, 20, "DF")
    pdf.set_text_color(*risk_rgb)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_xy(20, pdf.get_y() + 2)
    pdf.cell(80, 16, f"{pct}% Malignancy Probability")
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 16, f"{risk_cat} Risk", align="R")
    pdf.ln(22)

    risk_data = [
        ["Metric", "Value"],
        ["Predicted Class", prediction["predicted_class"]],
        ["Confidence Band", prediction.get("confidence_band", "N/A")],
        ["Dominant Feature", explanation.get("dominant_feature", "N/A").title()],
        ["Risk Direction", explanation.get("risk_direction", "N/A")],
    ]
    _render_table(pdf, risk_data)

    # ── Section 3: SHAP Explanation ─────────────────────────────
    _section_header(pdf, "3. Feature Attribution (SHAP)")

    attrs = explanation.get("feature_attributions", {})
    ordered = sorted(attrs.items(), key=lambda x: abs(x[1]), reverse=True)
    shap_data = [["Feature", "SHAP Value", "Direction"]]
    for feat, val in ordered:
        direction = "Increases risk" if val > 0 else "Decreases risk"
        shap_data.append([feat.title(), f"{val:+.4f}", direction])
    _render_table(pdf, shap_data)

    pdf.ln(3)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(80, 80, 80)
    shap_text = explanation.get("shap_text", "")
    pdf.multi_cell(180, 4, shap_text)

    # ── Section 4: AI Explanation ────────────────────────────────
    if llm_response:
        _section_header(pdf, "4. AI-Generated Explanation")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(40, 40, 40)
        main_text = llm_response.get("main_text", "")
        if main_text:
            pdf.multi_cell(180, 4.5, main_text)

        if llm_response.get("high_risk"):
            pdf.ln(3)
            pdf.set_fill_color(255, 241, 244)
            pdf.set_text_color(168, 48, 80)
            pdf.set_font("Helvetica", "B", 8)
            pdf.cell(180, 6, "HIGH-RISK ESTIMATE — Prompt clinical review recommended",
                     fill=True, new_x="LMARGIN", new_y="NEXT")

    # ── Section 5: Disclaimer ───────────────────────────────────
    pdf.ln(8)
    pdf.set_fill_color(240, 244, 248)
    y_start = pdf.get_y()
    pdf.set_draw_color(200, 200, 200)

    disclaimer_text = (
        "DISCLAIMER: This report is generated by BreastCare AI, a research and educational "
        "prototype. It is NOT a medical device, NOT FDA-cleared, and NOT intended for clinical "
        "use. All risk estimates are based on statistical models trained on research datasets "
        "and must be reviewed by a qualified radiologist or physician before any clinical "
        "decision is made. This tool does not diagnose cancer. The malignancy probability is "
        "a statistical estimate, not a diagnosis. No treatment decisions should be made based "
        "on this report alone."
    )

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(100, 30, 30)
    pdf.cell(180, 5, "Clinical Review Required", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(180, 3.5, disclaimer_text)

    # ── Footer ──────────────────────────────────────────────────
    pdf.set_y(-15)
    pdf.set_font("Helvetica", "I", 6)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 4,
             f"BreastCare AI v2.0  |  {timestamp}  |  Page {pdf.page_no()}",
             align="C")

    return bytes(pdf.output())


def save_pdf_report(pdf_bytes: bytes, session_id: str) -> Path:
    """Save PDF report to disk and return the file path."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"breastcare_report_{timestamp}_{session_id[:8]}.pdf"
    filepath = REPORTS_DIR / filename
    filepath.write_bytes(pdf_bytes)
    logger.info("PDF report saved: %s", filepath)
    return filepath


# ── Internal helpers ────────────────────────────────────────────

def _section_header(pdf, title: str) -> None:
    """Render a styled section header."""
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(14, 28, 47)
    pdf.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(200, 68, 93)
    pdf.line(15, pdf.get_y(), 60, pdf.get_y())
    pdf.ln(3)


def _render_table(pdf, data: list[list[str]]) -> None:
    """Render a simple table with header row."""
    if not data:
        return

    col_count = len(data[0])
    col_width = 180 / col_count

    # Header
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(245, 247, 250)
    pdf.set_text_color(40, 40, 40)
    for cell in data[0]:
        pdf.cell(col_width, 6, str(cell), border=1, fill=True)
    pdf.ln()

    # Rows
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(60, 60, 60)
    for row in data[1:]:
        for cell in row:
            pdf.cell(col_width, 5.5, str(cell), border=1)
        pdf.ln()
