"""
server.py — FastAPI backend for BreastCare AI.
Replaces Streamlit. Serves the HTML frontend and exposes pipeline API endpoints.

Run:
    python server.py
    OR: uvicorn server:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /           → serves app/index.html
    POST /api/assess → ML + SHAP (no LLM)
    POST /api/chat   → full pipeline (ML + SHAP + RAG + Gemini)
"""
from __future__ import annotations
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Load env vars from farwa_env activation or .env file
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

os.environ.setdefault("PYTHONUTF8", "1")

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI(title="BreastCare AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ────────────────────────────────────────────────────────────

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


# ── Routes ────────────────────────────────────────────────────────────────────

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
    """Run full pipeline: ML + SHAP + RAG + Gemini AI explanation."""
    features = {"age": req.age, "shape": req.shape, "margin": req.margin, "density": req.density}
    try:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if api_key:
            from src.pipeline import run_pipeline
            result = run_pipeline(features, user_question=req.question)
            lr = result.get("llm_response") or {}
            return JSONResponse({
                "response":    lr.get("main_text", ""),
                "disclaimer":  lr.get("disclaimer", ""),
                "sources":     result.get("sources_used", []),
                "prediction":  result["prediction"],
                "explanation": result["explanation"],
            })
        else:
            # Fallback: return raw retrieved chunks when no API key
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
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/assess-image")
async def assess_image(file: UploadFile = File(...)):
    """
    Upload a mammogram PNG/JPEG → EfficientNet-B0 risk score + Grad-CAM heatmap.
    Returns base64-encoded Grad-CAM overlay PNG alongside the probability.
    """
    from src.ml.image_model import is_available, predict_image, IMG_MODEL_PATH
    if not is_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "Image model not ready. "
                f"Train on Kaggle and place efficientnet_cbis.pth at {IMG_MODEL_PATH}"
            ),
        )
    allowed = {"image/jpeg", "image/png", "image/jpg"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Upload a JPEG or PNG mammogram image")
    try:
        import io
        from PIL import Image as PILImage
        data = await file.read()
        img = PILImage.open(io.BytesIO(data))
        return JSONResponse(predict_image(img))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
def health():
    from src.ml.image_model import is_available
    return {
        "status": "ok",
        "gemini_key_set": bool(os.environ.get("GEMINI_API_KEY")),
        "image_model": is_available(),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
