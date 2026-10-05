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

# ── Knowledge base paths ─────────────────────────────────────────
KB_DIR              = ROOT / "knowledge_base"
KB_SOURCES_DIR      = KB_DIR / "sources"
KB_CHUNKS_DIR       = KB_DIR / "chunks"
KB_VECTORSTORE_DIR  = KB_DIR / "vectorstore"
CHUNKS_JSONL        = KB_CHUNKS_DIR / "chunks.jsonl"
FAISS_INDEX_PATH    = KB_VECTORSTORE_DIR / "faiss.index"
CHUNK_METADATA_PATH = KB_VECTORSTORE_DIR / "chunk_metadata.json"
TFIDF_PATH          = KB_VECTORSTORE_DIR / "tfidf_vectorizer.pkl"
SVD_PATH            = KB_VECTORSTORE_DIR / "svd_reducer.pkl"
EMBEDDER_FLAG_PATH  = KB_VECTORSTORE_DIR / "embedder_type.txt"

# ── GraphRAG paths ───────────────────────────────────────────────
KB_GRAPHSTORE_DIR    = KB_DIR / "graphstore"
KNOWLEDGE_GRAPH_PATH = KB_GRAPHSTORE_DIR / "knowledge_graph.graphml"
ENTITY_CACHE_PATH    = KB_GRAPHSTORE_DIR / "entity_cache.json"
COMMUNITY_CACHE_PATH = KB_GRAPHSTORE_DIR / "community_summaries.json"

# ── Audit & reporting paths ──────────────────────────────────────
AUDIT_DB_PATH        = DATA_DIR / "audit.db"
REPORTS_DIR          = DATA_DIR / "reports"
AUDIT_ENABLED        = True

# ── Risk thresholds ──────────────────────────────────────────────
# Categories apply to the calibrated probability. The binary decision threshold
# is chosen during training and stored in models/model_metadata.json.
LOW_RISK_THRESHOLD      = 0.35   # below = Low
HIGH_RISK_THRESHOLD     = 0.65   # at or above = Elevated
CHUNK_TOP_K             = 3      # retrieved chunks per query

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
GEMINI_MODEL   = "gemini-2.5-flash"   # free-tier key from Google AI Studio
LLM_MAX_TOKENS = 900

# ── Multi-agent pipeline ─────────────────────────────────────────
MULTI_AGENT_ENABLED  = False   # opt-in via UI toggle
AGENT_MAX_TOKENS     = 600
AGENT_TEMPERATURE    = 0.3

# ── Chunking ─────────────────────────────────────────────────────
CHUNK_MIN_LEN = 80
CHUNK_MAX_LEN = 1200

# ── SHAP embedding dims ──────────────────────────────────────────
EMBED_DIM_DENSE    = 384   # all-MiniLM-L6-v2
EMBED_DIM_FALLBACK = 128   # TF-IDF + SVD fallback

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
