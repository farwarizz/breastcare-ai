# BreastCare AI

> Multi-layer breast cancer risk estimation prototype.
> XGBoost · LightGBM · SHAP · FAISS RAG · Claude AI · Streamlit

**Research and educational prototype. Not for clinical use. Not FDA-cleared.**

---

## Setup (Windows PowerShell — run each command separately)

```powershell
# 1. Activate your virtual environment
& "path\to\farwa_env\Scripts\Activate.ps1"
```

```powershell
# 2. Install dependencies
pip install -r requirements.txt
```

```powershell
# 3. Copy .env.example to .env and add your Anthropic API key
copy .env.example .env
```

```powershell
# 4. Train both models (~60 seconds)
python -m src.ml.train
```

```powershell
# 5. Build knowledge base and FAISS index (~10 seconds)
python scripts/build_knowledge_base.py
```

```powershell
# 6. Launch the app
streamlit run app/streamlit_app.py
```

Open http://localhost:8501

---

## Architecture

```
User Input (4 mammographic features: age, shape, margin, density)
          │
          ▼
    pipeline.py  ← single orchestrator
          │
  ┌───────┼───────────────┐
  ▼       ▼               ▼
predict  explain        rag_query
(LightGBM) (SHAP)      (FAISS + TF-IDF)
          │
     shap_text → retrieval query → top-3 chunks
          │
     prompt_builder.py
          │
     Claude API (claude-sonnet-4)
          │
     safety.py  ← strips diagnostic language (code-level, guaranteed)
          │
     UI (Streamlit)
```

## What the system uses

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Dataset 1 | WDBC (569 cases, 30 features) | XGBoost research model |
| Dataset 2 | UCI Mammographic Masses (860 cases, 4 features) | LightGBM clinical tool |
| Knowledge base | 60 chunks from 4 ACR BI-RADS docs | RAG grounding |
| Explainability | SHAP TreeExplainer | Feature attribution |
| Retrieval | FAISS IndexFlatIP | Semantic search |
| LLM | Claude Sonnet | Grounded explanation |
| Safety | Regex filter in safety.py | Strips diagnostic language |

## API key

The Claude explanation panel requires `ANTHROPIC_API_KEY` in a `.env` file.
Without it, the ML prediction and SHAP explainability still work.
