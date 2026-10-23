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
    POST /api/assess-image     → EfficientNet-B0 + Grad-CAM (experimental; 503 if the
                                 real weights/dependencies are missing)
    POST /api/risk-profile     → simplified Gail / Tyrer-Cuzick approximations
    POST /api/compare-models   → compare the trained models that are available
    POST /api/batch-predict    → CSV batch prediction
    GET  /api/audit/history    → prediction audit trail
    GET  /api/audit/stats      → aggregate statistics
    POST /api/report           → download PDF report
    GET  /api/drift-report     → drift report (placeholder baselines, illustrative)
    GET  /api/global-shap      → dataset-wide SHAP analysis
    GET  /api/model-info       → saved metrics + knowledge-base facts for the UI
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

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile, File, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger("breastcare.server")


def _server_error(exc: Exception) -> HTTPException:
    """Log the real error; tell the client only that something failed (no internal paths,
    provider messages or keys in responses)."""
    logger.exception("request failed: %s", exc)
    return HTTPException(status_code=500, detail="Internal error while processing the request. "
                                                 "The server log has the details.")

app = FastAPI(
    title="BreastCare AI",
    version="2.3.0",
    description="Research prototype: breast-mass risk estimate with conformal "
                "prediction sets, SHAP explanations and a retrieval-grounded assistant",
)

# ── Deployment limits (Phase 2.2; each one closes a blocker from the 2026-10-20 audit) ──
# CORS: the built-in UI is served by this app (same origin) and needs no CORS. A separate
# frontend must be listed in ALLOWED_ORIGINS (comma-separated); the default allows only local
# development servers. "*" is honoured only when set explicitly.
_DEFAULT_ORIGINS = ("http://localhost:5199,http://127.0.0.1:5199,http://localhost:5173,http://127.0.0.1:5173,"
                    "http://localhost:8080,http://127.0.0.1:8080")
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",") if o.strip()]
# /api/chat calls Gemini on a shared key: a per-client limit and a daily cap per server process.
CHAT_LIMIT_PER_MINUTE = int(os.environ.get("CHAT_RATE_LIMIT_PER_MINUTE", "10"))
CHAT_LIMIT_PER_DAY = int(os.environ.get("CHAT_DAILY_LIMIT", "300"))
# Uploads are read in chunks and refused past these sizes (HTTP 413).
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
MAX_CSV_BYTES = int(os.environ.get("MAX_CSV_BYTES", str(2 * 1024 * 1024)))
MAX_BATCH_ROWS = int(os.environ.get("MAX_BATCH_ROWS", "5000"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


class _RateLimiter:
    """In-memory sliding-window limit per client plus a daily total. It is per process: on a
    multi-instance or serverless deployment each instance counts separately, so a shared store
    (or the host's own rate limiting) is still needed there."""

    def __init__(self, per_minute: int, per_day: int):
        import threading
        self.per_minute, self.per_day = per_minute, per_day
        self._hits: dict[str, list[float]] = {}
        self._day, self._day_count = None, 0
        self._lock = threading.Lock()

    def check(self, client: str) -> int | None:
        """None if allowed (and counted), else seconds to wait."""
        import datetime
        import time
        now = time.time()
        with self._lock:
            today = datetime.date.today()
            if today != self._day:
                self._day, self._day_count = today, 0
            if self._day_count >= self.per_day:
                tomorrow = datetime.datetime.combine(today + datetime.timedelta(days=1), datetime.time())
                return max(1, int(tomorrow.timestamp() - now))
            hits = [h for h in self._hits.get(client, []) if now - h < 60]
            if len(hits) >= self.per_minute:
                self._hits[client] = hits
                return max(1, int(60 - (now - hits[0])) + 1)
            hits.append(now)
            self._hits[client] = hits
            self._day_count += 1
            return None


chat_limiter = _RateLimiter(CHAT_LIMIT_PER_MINUTE, CHAT_LIMIT_PER_DAY)


def _client_id(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def limit_chat(request: Request) -> None:
    wait = chat_limiter.check(_client_id(request))
    if wait is not None:
        raise HTTPException(status_code=429, detail="Too many questions; please wait and try again.",
                            headers={"Retry-After": str(wait)})


def require_audit_token(request: Request) -> None:
    """Audit endpoints expose logged assessments and chat questions. They are disabled unless
    AUDIT_API_TOKEN is set, and then need `Authorization: Bearer <token>`."""
    import hmac
    token = os.environ.get("AUDIT_API_TOKEN", "")
    if not token:
        raise HTTPException(status_code=403, detail="Audit endpoints are disabled on this server.")
    sent = request.headers.get("authorization", "")
    if not hmac.compare_digest(sent.encode(), f"Bearer {token}".encode()):
        raise HTTPException(status_code=401, detail="Audit endpoints need a valid token.",
                            headers={"WWW-Authenticate": "Bearer"})


async def _read_limited(file: UploadFile, max_bytes: int) -> bytes:
    """Read an upload in chunks; refuse it (413) as soon as it exceeds max_bytes."""
    chunks, size = [], 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        size += len(chunk)
        if size > max_bytes:
            raise HTTPException(status_code=413, detail=f"File too large (limit {max_bytes // (1024 * 1024)} MB).")
        chunks.append(chunk)

# Evaluation figures (generated by `python -m src.ml.train`), shown in the UI
_FIGURES = Path(__file__).parent / "reports" / "figures"
if _FIGURES.is_dir():
    app.mount("/figures", StaticFiles(directory=str(_FIGURES)), name="figures")

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
        raise _server_error(e)


@app.post("/api/chat", dependencies=[Depends(limit_chat)])
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
            )
            lr = result.get("llm_response") or {}
            response_data = {
                "response":    lr.get("main_text", ""),
                "disclaimer":  lr.get("disclaimer", ""),
                "sources":     result.get("sources_used", []),
                "prediction":  result["prediction"],
                "explanation": result["explanation"],
                "mode":        result.get("mode", req.mode),
                "citations":   result.get("citations", []),
                "guardrail":   result.get("guardrail"),
                "trace":       result.get("trace", {}),
                "requested_mode": result.get("requested_mode"),
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
        raise _server_error(e)


# ── Image Analysis ────────────────────────────────────────────────────────────

IMAGE_MODEL_UNAVAILABLE = (
    "Image model unavailable: this server does not have the trained CBIS-DDSM "
    "weights (models/efficientnet_cbis.pth) and/or the inference dependencies "
    "(torch, torchvision, timm). No image-based estimate was produced."
)


@app.post("/api/assess-image")
async def assess_image(file: UploadFile = File(...)):
    """
    Upload mammogram → EfficientNet-B0 score + Grad-CAM heatmap.
    Returns 503 when the real model cannot run; never substitutes a heuristic score.
    """
    from src.ml import image_model

    if not image_model.is_available():
        raise HTTPException(status_code=503, detail=IMAGE_MODEL_UNAVAILABLE)
    try:
        image_model.ensure_loaded()
    except Exception as exc:
        logger.exception("image model failed to load: %s", exc)
        raise HTTPException(status_code=503, detail="Image model unavailable: the weights failed to load.")

    allowed = {"image/jpeg", "image/png", "image/jpg"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Upload a JPEG or PNG mammogram image")

    from PIL import Image as PILImage, UnidentifiedImageError
    data = await _read_limited(file, MAX_IMAGE_BYTES)
    try:
        img = PILImage.open(io.BytesIO(data))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image")

    try:
        return JSONResponse(image_model.predict_image(img))
    except Exception as exc:
        raise _server_error(exc)


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
        raise _server_error(e)


# ── Model Comparison ──────────────────────────────────────────────────────────

@app.post("/api/compare-models")
def compare_models_endpoint(req: CompareRequest):
    """Run all models on same input for A/B comparison."""
    features = {"age": req.age, "shape": req.shape, "margin": req.margin, "density": req.density}
    try:
        from src.analytics.model_comparison import compare_models
        return JSONResponse(compare_models(features))
    except Exception as e:
        raise _server_error(e)


# ── Batch Prediction ──────────────────────────────────────────────────────────

@app.post("/api/batch-predict")
async def batch_predict_endpoint(file: UploadFile = File(...)):
    """Upload CSV → batch predictions. Returns JSON with results + summary."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV file")
    import pandas as pd
    from src.analytics.batch_predict import BatchInputError, batch_predict, batch_summary

    data = await _read_limited(file, MAX_CSV_BYTES)
    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception as exc:                      # parser messages are not shown to clients
        logger.info("unreadable CSV upload: %s", exc)
        raise HTTPException(status_code=400, detail="The file could not be read as a CSV with columns "
                                                    "age, shape, margin and density.")
    if len(df) > MAX_BATCH_ROWS:
        raise HTTPException(status_code=413, detail=f"Too many rows (limit {MAX_BATCH_ROWS}).")
    try:
        result_df = batch_predict(df)
        summary = batch_summary(result_df)

        return JSONResponse({
            "results": result_df.to_dict(orient="records"),
            "summary": summary,
            "total_rows": len(result_df),
        })
    except BatchInputError as e:                  # our own validation messages (columns, codes, ranges)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise _server_error(e)


# ── Audit Trail ───────────────────────────────────────────────────────────────

@app.get("/api/audit/history", dependencies=[Depends(require_audit_token)])
def audit_history(limit: int = 50, offset: int = 0):
    """Get paginated prediction history."""
    try:
        from src.audit.database import AuditDB
        db = AuditDB()
        history = db.get_prediction_history(limit=limit, offset=offset)
        return JSONResponse({"history": history, "count": len(history)})
    except Exception as e:
        raise _server_error(e)


@app.get("/api/audit/stats", dependencies=[Depends(require_audit_token)])
def audit_stats():
    """Get aggregate audit statistics."""
    try:
        from src.audit.database import AuditDB
        db = AuditDB()
        return JSONResponse(db.get_stats())
    except Exception as e:
        raise _server_error(e)


@app.post("/api/audit/session", dependencies=[Depends(require_audit_token)])
def create_session():
    """Create a new audit session. Returns session_id."""
    try:
        from src.audit.database import AuditDB
        db = AuditDB()
        session_id = db.create_session()
        return JSONResponse({"session_id": session_id})
    except Exception as e:
        raise _server_error(e)


@app.get("/api/audit/export", dependencies=[Depends(require_audit_token)])
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
        raise _server_error(e)


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
        raise _server_error(e)


# ── Model Monitoring ──────────────────────────────────────────────────────────

@app.get("/api/drift-report")
def drift_report():
    """Get model drift analysis report."""
    try:
        report = _get_monitor().get_drift_report()
        return JSONResponse(report)
    except Exception as e:
        raise _server_error(e)


# ── Global SHAP ──────────────────────────────────────────────────────────────

@app.get("/api/global-shap")
def global_shap():
    """Get dataset-wide SHAP analysis."""
    try:
        from src.analytics.global_shap import compute_global_shap
        return JSONResponse(compute_global_shap())
    except Exception as e:
        raise _server_error(e)


# ── Model Info (source of truth for numbers shown in the UI) ─────────────────

@app.get("/api/model-info")
def model_info():
    """Saved model metrics, dataset provenance and knowledge-base contents."""
    try:
        from src.model_info import get_model_info
        return JSONResponse(get_model_info())
    except Exception as e:
        raise _server_error(e)


# ── Health Check ──────────────────────────────────────────────────────────────

def _retrieval_mode() -> str | None:
    """The retrieval mode that will run, or None if the knowledge-base index is missing or invalid."""
    try:
        from src.rag.retriever import _index, resolve_mode
        _index()
        return resolve_mode()
    except Exception:
        return None


@app.get("/health")
def health():
    from src.ml.image_model import is_available as img_available
    from src.config import AUDIT_DB_PATH, SERVED_MODEL_PATH

    served = None
    try:
        from src.ml.artifacts import model_meta
        served = model_meta("served")["label"]
    except Exception:
        pass
    retrieval = _retrieval_mode()
    return {
        # "degraded": the risk model or the knowledge-base index is missing, so /api/assess
        # or /api/chat will fail.
        "status": "ok" if served and retrieval else "degraded",
        "version": "2.3.0",
        "served_model": served,
        "served_model_file": SERVED_MODEL_PATH.exists(),
        "gemini_key_set": bool(os.environ.get("GEMINI_API_KEY")),
        "image_model": img_available(),
        "retrieval": retrieval,
        "audit_db": AUDIT_DB_PATH.exists(),
        "endpoints": {
            "core": ["/api/assess", "/api/chat"],
            "extended": [
                "/api/assess-image", "/api/risk-profile",
                "/api/compare-models", "/api/batch-predict",
                "/api/audit/history", "/api/audit/stats",
                "/api/report",
                "/api/drift-report", "/api/global-shap",
                "/api/model-info",
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
