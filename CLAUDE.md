# BreastCare AI: notes for coding assistants

Research and educational prototype (not a medical device): a LightGBM risk model trained on
the real UCI Mammographic Mass data, plus SHAP, FAISS retrieval and Google Gemini. It is
served by FastAPI (`server.py`) with one frontend (`app/index.html`).
Full design: `docs/DESIGN.md`. Dataset provenance: `data/README.md`.

## Commands

- Run the app: `uvicorn server:app --reload`, then open http://localhost:8000
- Tests: `pytest`
- Retrain: `python -m src.ml.train`, then `python scripts/update_readme_metrics.py`
- Verify the raw data: `python scripts/download_data.py` (needs internet and `ucimlrepo`)
- Rebuild the knowledge base: `python scripts/build_knowledge_base.py` (indexes every file in `knowledge_base/sources/`)
- On Windows, set `PYTHONUTF8=1` when piping script output (scripts print box-drawing characters).

## Rules

- Never add synthetic, heuristic or client-side fallbacks that produce a probability (data
  loading, image analysis, frontend). Fail loudly, or return HTTP 503.
- Training runs only on the checksum-pinned UCI file. The synthetic generator lives only in
  `tests/fixtures/`.
- BI-RADS is never a model feature (label leakage).
- Probabilities are uncalibrated and have no uncertainty interval. Label them that way and do
  not invent intervals.
- Doc metrics come only from `models/model_metadata.json` via `scripts/update_readme_metrics.py`.
  The UI reads `/api/model-info`. Never hand-type metrics.
- The LLM provider is Gemini (`GEMINI_API_KEY`, OpenAI-compatible endpoint in `src/llm/client.py`).

## Roadmap

Phase 0 (honesty cleanup) is done. Next: Phase 1, ML rigor (validation split, confidence
intervals, calibration, conformal prediction); Phase 2, RAG and safety evaluation; Phase 3,
packaging, tests and CI; Phase 4, presentation.
