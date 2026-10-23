"""
src/config.py — Central configuration for BreastCare AI.
All paths, thresholds, feature mappings, and extension settings live here.
"""
import os
from pathlib import Path

# ── Project root ────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent

# ── Data paths ──────────────────────────────────────────────────
DATA_DIR           = ROOT / "data"
DATA_RAW_DIR       = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
CLEANED_CSV        = DATA_PROCESSED_DIR / "cleaned.csv"

# ── UCI Mammographic Mass dataset (UCI id 161, CC BY 4.0) ────────
# The raw file is the original UCI file, pinned by SHA-256. Training refuses
# to run on any other file. Provenance: data/README.md
MAMMO_UCI_ID     = 161
MAMMO_RAW_PATH   = DATA_RAW_DIR / "mammographic_masses.data"
MAMMO_SOURCE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "mammographic-masses/mammographic_masses.data"
)
MAMMO_SHA256     = "072051c896fb14f8f25c64eed30ba1333290af41e6c9ffad4e1ca573daa865a8"
MAMMO_DOI        = "https://doi.org/10.24432/C53K6Z"
MAMMO_RAW_COLUMNS = ["birads", "age", "shape", "margin", "density", "severity"]

# ── Model artifacts (written by `python -m src.ml.train`) ────────
MODELS_DIR             = ROOT / "models"
SERVED_MODEL_PATH      = MODELS_DIR / "served_model.joblib"      # selected model
CHALLENGER_MODEL_PATH  = MODELS_DIR / "challenger_model.joblib"  # runner-up, for comparison
MODEL_METADATA_PATH    = MODELS_DIR / "model_metadata.json"      # metrics, calibration, conformal
SPLITS_PATH            = MODELS_DIR / "splits.json"              # raw-row ids of each split

# ── Evaluation reports (README/report figures; not the audit PDFs) ─
EVAL_REPORTS_DIR = ROOT / "reports"
FIGURES_DIR      = EVAL_REPORTS_DIR / "figures"

# ── Image model paths ────────────────────────────────────────────
IMG_MODEL_PATH       = MODELS_DIR / "efficientnet_cbis.pth"

# ── Knowledge base (built by `python scripts/build_knowledge_base.py`) ─
KB_DIR              = ROOT / "knowledge_base"
KB_SOURCES_DIR      = KB_DIR / "sources"
CHUNKS_JSONL        = KB_DIR / "chunks" / "chunks.jsonl"
KB_INDEX_DIR        = KB_DIR / "index"
KB_MANIFEST_PATH    = KB_INDEX_DIR / "manifest.json"   # embedders, dims, source hashes

# ── Evaluation sets and LLM response cache ───────────────────────
EVALS_DIR           = ROOT / "evals"

# ── Audit & reporting paths ──────────────────────────────────────
AUDIT_DB_PATH        = DATA_DIR / "audit.db"
# Serverless hosts (Vercel) have a read-only filesystem except /tmp: the audit log
# and downloaded model caches go there (per instance, not persistent).
if os.environ.get("VERCEL"):
    AUDIT_DB_PATH = Path("/tmp/audit.db")
    os.environ.setdefault("HF_HOME", "/tmp/hf")
    os.environ.setdefault("FASTEMBED_CACHE_PATH", "/tmp/fastembed")
REPORTS_DIR          = DATA_DIR / "reports"
AUDIT_ENABLED        = True

# ── Risk thresholds ──────────────────────────────────────────────
# Categories apply to the served probability (calibrated only if the calibration step
# chosen in training is not "none"). The binary decision threshold
# is chosen during training and stored in models/model_metadata.json.
LOW_RISK_THRESHOLD      = 0.35   # below = Low
HIGH_RISK_THRESHOLD     = 0.65   # at or above = Elevated
CHUNK_TOP_K             = 3      # retrieved chunks per query
# Retrieval mode served by the app: "bm25", "dense:<embedder>", "hybrid:<embedder>", each
# optionally "+rerank". Set by the fixed rule in src/evals/retrieval.py, never by hand
# (reports/rag_evaluation.md; tests/test_evals.py checks the two agree). BM25 was within
# the rule's 0.02 Hit@3 tolerance of the best mode and needs no embedding model at runtime.
RETRIEVAL_MODE          = "bm25"
# Query for a user question: "question" (the question alone) or "shap_prefix"
# (SHAP summary + question). Also set from the retrieval evaluation.
RETRIEVAL_QUERY         = "question"

# ── Mammographic feature mappings ────────────────────────────────
SHAPE_MAP = {
    1: "Round",
    2: "Oval",
    3: "Lobular",
    4: "Irregular",
}

MARGIN_MAP = {
    1: "Circumscribed",
    2: "Microlobulated",
    3: "Obscured",
    4: "Ill-defined",
    5: "Spiculated",
}

DENSITY_MAP = {
    1: "High density",
    2: "Iso density",
    3: "Low density",
    4: "Fat-containing",
}

FEATURE_NAMES = ["age", "shape", "margin", "density"]

# ── LLM (Google Gemini via its OpenAI-compatible endpoint) ───────
# Free-tier key from Google AI Studio. The primary model is the one the evaluations
# measure; gemini-2.5-flash's free tier allows only a few dozen requests a day, so it is
# a fallback rather than the default.
GEMINI_MODEL   = "gemini-3.5-flash-lite"
# Tried in order when the primary model is out of quota, overloaded or retired.
LLM_FALLBACK_MODELS = ["gemini-2.5-flash", "gemini-3.5-flash"]
# Gemini 2.5+/3.x models "think" by default and the hidden thinking tokens count
# against max_tokens: at reasoning_effort "low", gemini-3.5-flash-lite spent 370–800 of
# 900 tokens thinking and cut answers off mid-sentence. "minimal" (3.x) and "none"
# (2.5) use no hidden tokens; the client also records finish_reason "length".
LLM_MODEL_PARAMS = {
    "gemini-2.5-flash":      {"reasoning_effort": "none"},
    "gemini-3.5-flash-lite": {"reasoning_effort": "minimal"},
    "gemini-3.5-flash":      {"reasoning_effort": "minimal"},
}
JUDGE_MODEL    = "gemini-3.5-flash-lite"   # LLM-as-judge in the evaluations (see DESIGN §6 on bias)
LLM_MAX_TOKENS = 1500                      # the prompt asks for < 220 words; this is headroom

# ── Multi-agent pipeline ─────────────────────────────────────────
MULTI_AGENT_ENABLED  = False   # opt-in via UI toggle
AGENT_MAX_TOKENS     = 1200
AGENT_TEMPERATURE    = 0.3


# ── Google Gemini API key ────────────────────────────────────────
# Free key (no credit card): https://aistudio.google.com
def get_api_key() -> str:
    """Load Gemini API key from environment or .env file."""
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass
    return os.environ.get("GEMINI_API_KEY", "")
