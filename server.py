"""
server.py — FastAPI backend for BreastCare AI.

Serves the HTML frontend and exposes all pipeline API endpoints:
core assessment, chat, multi-agent/CoT, risk profiling, batch prediction,
analytics, audit trail, and PDF report generation.

Run:
    python server.py
    OR: uvicorn server:app --host 0.0.0.0 --port 8000 --reload

Core Endpoints:
    GET  /                     → serves app/index.html
    POST /api/assess           → ML + SHAP (no LLM)
    POST /api/chat             → full pipeline (ML + SHAP + RAG + Gemini)

Extended Endpoints:
    POST /api/assess-image     → EfficientNet image analysis + Grad-CAM
    POST /api/risk-profile     → extended patient risk scoring
    POST /api/compare-models   → A/B model comparison
    POST /api/batch-predict    → CSV batch prediction
    GET  /api/audit/history    → prediction audit trail
    GET  /api/audit/stats      → aggregate statistics
    GET  /api/report/{id}      → download PDF report
    GET  /api/graph-status     → knowledge graph stats
    GET  /api/drift-report     → model drift analysis
    GET  /api/global-shap      → dataset-wide SHAP analysis
    GET  /health               → system health check
"""
from __future__ import annotations

import io
import json
import sys
import os
import uuid
import base64
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

os.environ.setdefault("PYTHONUTF8", "1")

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="BreastCare AI",
    version="2.0.0",
    description="AI-powered breast cancer risk assessment with SHAP explainability",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded monitor singleton
_monitor = None

def _get_monitor():
    global _monitor
    if _monitor is None:
        from src.analytics.model_monitor import PredictionMonitor
        _monitor = PredictionMonitor()
    return _monitor


# ── Request Models ────────────────────────────────────────────────────────────

class AssessRequest(BaseModel):
    age:     int = Field(..., ge=18, le=96)
    shape:   int = Field(..., ge=1, le=4)
    margin:  int = Field(..., ge=1, le=5)
    density: int = Field(..., ge=1, le=4)


class ChatRequest(BaseModel):
    age:      int = Field(..., ge=18, le=96)
    shape:    int = Field(..., ge=1, le=4)
    margin:   int = Field(..., ge=1, le=5)
    density:  int = Field(..., ge=1, le=4)
    question: str = Field(..., min_length=1, max_length=500)
    mode:     str = Field(default="single", pattern="^(single|multi_agent|cot)$")
    session_id: str = Field(default="")
    use_hybrid_rag: bool = Field(default=False)


class RiskProfileRequest(BaseModel):
    age:                     int = Field(..., ge=18, le=96)
    shape:                   int = Field(..., ge=1, le=4)
    margin:                  int = Field(..., ge=1, le=5)
    density:                 int = Field(..., ge=1, le=4)
    family_history:          bool = False
    num_first_degree_relatives: int = 0
    brca_status:             str = "Unknown"
    prior_biopsies:          int = 0
    atypical_hyperplasia:    bool = False
    breast_density_category: str = "Unknown"
    age_at_menarche:         int = 12
    age_at_first_birth:      int = 25
    hormone_replacement:     bool = False


class CompareRequest(BaseModel):
    age:     int = Field(..., ge=18, le=96)
    shape:   int = Field(..., ge=1, le=4)
    margin:  int = Field(..., ge=1, le=5)
    density: int = Field(..., ge=1, le=4)


# ── Core Routes ───────────────────────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    html_path = Path(__file__).parent / "app" / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not built")
    return FileResponse(str(html_path), media_type="text/html")


@app.post("/api/assess")
def assess(req: AssessRequest):
    """Run ML prediction + SHAP explainability. No LLM call."""
    features = {"age": req.age, "shape": req.shape, "margin": req.margin, "density": req.density}
    try:
        from src.pipeline import run_pipeline_ml_only
        result = run_pipeline_ml_only(features)

        # Record for monitoring
        _get_monitor().record_prediction(features, result["prediction"]["probability"])

        return JSONResponse({
            "prediction":  result["prediction"],
            "explanation": result["explanation"],
            "sources":     result["sources_used"],
            "query":       result["query_used"],
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Run full pipeline: ML + SHAP + RAG + Gemini. Supports single/multi_agent/cot modes."""
    features = {"age": req.age, "shape": req.shape, "margin": req.margin, "density": req.density}
    try:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if api_key:
            from src.pipeline import run_pipeline
            result = run_pipeline(
                features,
                user_question=req.question,
                mode=req.mode,
                session_id=req.session_id,
                use_hybrid_rag=req.use_hybrid_rag,
            )
            lr = result.get("llm_response") or {}
            response_data = {
                "response":    lr.get("main_text", ""),
                "disclaimer":  lr.get("disclaimer", ""),
                "sources":     result.get("sources_used", []),
                "prediction":  result["prediction"],
                "explanation": result["explanation"],
                "mode":        result.get("mode", req.mode),
            }
            # Include reasoning trace for multi-agent/CoT
            if result.get("reasoning_trace"):
                response_data["reasoning_trace"] = result["reasoning_trace"]
            if result.get("agent_traces"):
                response_data["agent_traces"] = result["agent_traces"]

            return JSONResponse(response_data)
        else:
            from src.pipeline import run_pipeline_ml_only
            result = run_pipeline_ml_only(features)
            chunks = result.get("chunks", [])
            raw = "\n\n".join(c.get("text", "") for c in chunks[:2])
            return JSONResponse({
                "response":   f"(Gemini API key not set — showing retrieved context)\n\n{raw}",
                "disclaimer": "Always consult a qualified clinician.",
                "sources":    result.get("sources_used", []),
                "prediction":  result["prediction"],
                "explanation": result["explanation"],
                "mode":        "fallback",
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Image Analysis ────────────────────────────────────────────────────────────

@app.post("/api/assess-image")
async def assess_image(file: UploadFile = File(...)):
    """Upload mammogram → EfficientNet risk score + Grad-CAM heatmap."""
    from src.ml.image_model import is_available, predict_image, predict_image_demo

    allowed = {"image/jpeg", "image/png", "image/jpg"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Upload a JPEG or PNG mammogram image")

    try:
        from PIL import Image as PILImage
        data = await file.read()
        img = PILImage.open(io.BytesIO(data))

        if is_available():
            return JSONResponse(predict_image(img))
        else:
            # Demo mode — no PyTorch or model weights
            return JSONResponse(predict_image_demo(img))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Risk Profile ──────────────────────────────────────────────────────────────

@app.post("/api/risk-profile")
def risk_profile(req: RiskProfileRequest):
    """Compute extended risk profile with classical models (Gail, Tyrer-Cuzick)."""
    try:
        from src.risk.patient_profile import PatientProfile
        from src.risk.classical_risk import compute_all_risks
        from src.pipeline import run_pipeline_ml_only

        profile = PatientProfile(**req.model_dump())
        features = profile.get_core_features()
        ml_result = run_pipeline_ml_only(features)
        ml_prob = ml_result["prediction"]["probability"]

        risk_data = compute_all_risks(profile, ml_prob)
        risk_data["risk_factors"] = profile.get_risk_factors_summary()

        return JSONResponse(risk_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Model Comparison ──────────────────────────────────────────────────────────

@app.post("/api/compare-models")
def compare_models_endpoint(req: CompareRequest):
    """Run all models on same input for A/B comparison."""
    features = {"age": req.age, "shape": req.shape, "margin": req.margin, "density": req.density}
    try:
        from src.analytics.model_comparison import compare_models
        return JSONResponse(compare_models(features))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Batch Prediction ──────────────────────────────────────────────────────────

@app.post("/api/batch-predict")
async def batch_predict_endpoint(file: UploadFile = File(...)):
    """Upload CSV → batch predictions. Returns JSON with results + summary."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV file")
    try:
        import pandas as pd
        from src.analytics.batch_predict import batch_predict, batch_summary

        data = await file.read()
        df = pd.read_csv(io.BytesIO(data))
        result_df = batch_predict(df)
        summary = batch_summary(result_df)

        return JSONResponse({
            "results": result_df.to_dict(orient="records"),
            "summary": summary,
            "total_rows": len(result_df),
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Audit Trail ───────────────────────────────────────────────────────────────

@app.get("/api/audit/history")
def audit_history(limit: int = 50, offset: int = 0):
    """Get paginated prediction history."""
    try:
        from src.audit.database import AuditDB
        db = AuditDB()
        history = db.get_prediction_history(limit=limit, offset=offset)
        return JSONResponse({"history": history, "count": len(history)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/audit/stats")
def audit_stats():
    """Get aggregate audit statistics."""
    try:
        from src.audit.database import AuditDB
        db = AuditDB()
        return JSONResponse(db.get_stats())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/audit/session")
def create_session():
    """Create a new audit session. Returns session_id."""
    try:
        from src.audit.database import AuditDB
        db = AuditDB()
        session_id = db.create_session()
        return JSONResponse({"session_id": session_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/audit/export")
def export_audit():
    """Export all predictions as CSV."""
    try:
        from src.audit.database import AuditDB
        db = AuditDB()
        csv_data = db.export_predictions_csv()
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=breastcare_audit.csv"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── PDF Report ────────────────────────────────────────────────────────────────

@app.post("/api/report")
def generate_report(req: AssessRequest):
    """Generate and return a PDF report for a risk assessment."""
    features = {"age": req.age, "shape": req.shape, "margin": req.margin, "density": req.density}
    try:
        from src.pipeline import run_pipeline_ml_only
        from src.audit.pdf_report import generate_pdf_report

        result = run_pipeline_ml_only(features)
        session_id = str(uuid.uuid4())[:8]

        pdf_bytes = generate_pdf_report(
            prediction=result["prediction"],
            explanation=result["explanation"],
            features=features,
            llm_response=result.get("llm_response"),
            session_id=session_id,
        )

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=breastcare_report_{session_id}.pdf"
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Knowledge Graph ───────────────────────────────────────────────────────────

@app.get("/api/graph-status")
def graph_status():
    """Get knowledge graph statistics."""
    try:
        from src.rag.graph_builder import get_graph_stats
        return JSONResponse(get_graph_stats())
    except Exception as e:
        return JSONResponse({"available": False, "error": str(e)})


# ── Model Monitoring ──────────────────────────────────────────────────────────

@app.get("/api/drift-report")
def drift_report():
    """Get model drift analysis report."""
    try:
        report = _get_monitor().get_drift_report()
        return JSONResponse(report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Global SHAP ──────────────────────────────────────────────────────────────

@app.get("/api/global-shap")
def global_shap():
    """Get dataset-wide SHAP analysis."""
    try:
        from src.analytics.global_shap import compute_global_shap
        return JSONResponse(compute_global_shap())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    from src.ml.image_model import is_available as img_available
    from src.config import ENSEMBLE_MODEL_PATH, KNOWLEDGE_GRAPH_PATH, AUDIT_DB_PATH

    return {
        "status": "ok",
        "version": "2.0.0",
        "gemini_key_set": bool(os.environ.get("GEMINI_API_KEY")),
        "image_model": img_available(),
        "ensemble_model": ENSEMBLE_MODEL_PATH.exists(),
        "knowledge_graph": KNOWLEDGE_GRAPH_PATH.exists(),
        "audit_db": AUDIT_DB_PATH.exists(),
        "endpoints": {
            "core": ["/api/assess", "/api/chat"],
            "extended": [
                "/api/assess-image", "/api/risk-profile",
                "/api/compare-models", "/api/batch-predict",
                "/api/audit/history", "/api/audit/stats",
                "/api/report", "/api/graph-status",
                "/api/drift-report", "/api/global-shap",
            ],
        },
    }


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
