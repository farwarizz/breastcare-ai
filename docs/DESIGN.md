# BreastCare AI: design document

This describes what the code does today, including the parts that are
experimental or not active. For a short overview see the [README](../README.md);
for dataset provenance see [data/README.md](../data/README.md).

## 1. Scope

BreastCare AI is a **research and educational prototype**. It estimates the
probability that a breast mass is malignant from four structured inputs (age,
and the BI-RADS descriptors shape, margin and density). It explains the
estimate with SHAP and answers follow-up questions with an LLM grounded in a
small knowledge base.

It is **not** a medical device. It is not FDA-cleared and not clinically
validated, it does not diagnose, and it does not recommend treatment. The risk
model works only on the structured descriptors, not on images. The separate,
experimental image model is described in §9.

Motivation, from the dataset documentation: the low positive predictive value
of breast biopsy after mammography leads to roughly 70% of biopsies having
benign outcomes. The UCI data was published to study whether descriptor-based
models can support the biopsy decision.

## 2. Data

| Dataset | Role | Rows (after cleaning) | Details |
|---|---|---|---|
| UCI Mammographic Mass (id 161) | Trains the model the app serves | 831 (428 benign / 403 malignant) | [data/README.md](../data/README.md) |
| WDBC (`sklearn.datasets.load_breast_cancer`) | Research benchmark only | 569 (357 benign / 212 malignant) | 30 nuclear features from fine-needle aspirates |

The raw UCI file is the original, checksum-pinned file, cross-checked against
`ucimlrepo`. Training refuses to run on anything else, and no production code
can generate substitute data. The BI-RADS assessment column is dropped because
it is the radiologist's conclusion: using it as a feature would leak the label.

## 3. Architecture and request flow

```
Browser: app/index.html  ──fetch──►  FastAPI: server.py
                                           │
             ┌─────────────────────────────┼──────────────────────────────┐
             ▼                             ▼                              ▼
    /api/assess, /api/report      /api/chat (full pipeline)      /api/model-info, /health
    run_pipeline_ml_only()        run_pipeline()                 saved metadata + KB facts
```

`src/pipeline.py`, `run_pipeline()` (used by `/api/chat`):

1. `predict()`: LightGBM gives the probability, the class at the saved Youden threshold, and a
   risk category (Low < 0.35 ≤ Moderate < 0.65 ≤ Elevated). The probability is uncalibrated
   and returned without an interval (`calibrated: false`).
2. `explain()`: SHAP TreeExplainer gives per-feature attributions, the dominant feature and a
   short text summary (`shap_text`).
3. `build_retrieval_query()`: dominant feature + direction + the start of `shap_text` + the
   user's question.
4. `rag_query()`: top-3 chunks from FAISS. GraphRAG is used only if a caller passes
   `use_hybrid_rag=true`. The UI never does, and without a built graph it falls back to FAISS.
5. If the probability is ≥ 0.65, the first chunk of the safety-disclaimer document is appended.
6. `build_prompt()` → `call_llm()`: Gemini, called through Google's OpenAI-compatible endpoint
   with the system prompt in `src/rag/prompt_builder.py`.
7. `filter_response()`: regex rewrite of diagnostic phrases, plus the standard (or high-risk) disclaimer.
8. If a session id is given, the prediction is written to the SQLite audit log.

`run_pipeline_ml_only()` (used by `/api/assess` and `/api/report`) runs steps 1–4 without an LLM call.

## 4. Component status

| Component | Where | Status |
|---|---|---|
| LightGBM risk model | `src/ml/train.py`, `predict.py` | Active |
| SHAP TreeExplainer | `src/ml/explain.py` | Active |
| FAISS dense retrieval | `src/rag/retriever.py` | Active; needs `sentence-transformers` at query time (§6) |
| Gemini generation | `src/llm/client.py` | Active in `/api/chat` with `GEMINI_API_KEY` |
| Regex safety filter | `src/safety.py` | Active on every LLM response |
| Chain-of-thought / multi-agent modes | `src/agents/` | Optional (chat mode selector); not evaluated |
| Audit log, PDF report, batch CSV | `src/audit/`, `src/analytics/batch_predict.py` | Active |
| Model comparison | `src/analytics/model_comparison.py` | Active, but only LightGBM is trained, so there is nothing to compare yet |
| EfficientNet-B0 image model | `src/ml/image_model.py` | Experimental; HTTP 503 unless the real weights and torch/timm are present |
| Simplified Gail / Tyrer-Cuzick | `src/risk/` | Active in `/api/risk-profile`; approximations, not the validated calculators |
| Drift monitor | `src/analytics/model_monitor.py` | Placeholder baselines, illustrative only |
| GraphRAG | `src/rag/graph_builder.py`, `graph_retriever.py` | Not active: no graph has been built |
| Stacking ensemble, calibration, Optuna | `src/ml/ensemble.py`, `calibration.py`, `hyperopt.py` | Not active: no trained artifacts, not in any request path |
| XGBoost on WDBC + 5-model CV | `src/ml/train.py` | Research benchmark; not used by the app |

## 5. Machine learning

**Features:** `age, shape, margin, density`, passed to the model as numbers. Shape and margin
are nominal codes; treating them as numeric is a simplification that Phase 1 revisits.

**Model:** `LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
min_child_samples=5)`, with early stopping after 30 rounds without improvement. A SHAP
`TreeExplainer` is saved alongside the model.

**Evaluation protocol (as implemented):** one stratified 80/20 split (`random_state=42`).
Early stopping and the Youden's-J threshold are both selected **on the test split**, so the
test metrics below are optimistically biased. No probability calibration, no confidence
intervals, no uncertainty estimates. These are Phase 1 work items; the exact protocol is
also stored in `models/model_metadata.json`.

**WDBC benchmark:** XGBoost (400 trees, depth 5), with SMOTE on the training split when
`imbalanced-learn` is installed (it is not in `requirements.txt`). The threshold is chosen
the same way. Five models are compared with 10-fold CV on the WDBC training split.

Current numbers, generated from `models/model_metadata.json`:

<!-- METRICS:START (generated by scripts/update_readme_metrics.py from models/model_metadata.json; do not edit by hand) -->
**Served model: LightGBM on UCI Mammographic Mass** (test split n = 167, train n = 664, decision threshold 0.368)

| Metric | Value |
|---|---|
| ROC-AUC | 0.8403 |
| Accuracy | 79.04% |
| Sensitivity (recall) | 87.65% |
| Specificity | 70.93% |
| Precision | 73.96% |
| F1 | 80.23% |
| False negatives / false positives | 10 / 25 |

**Evaluation protocol (from the metadata):** Single stratified 80/20 train/test split (random_state=42) per dataset. LightGBM early stopping and both Youden's-J decision thresholds are selected on the test split, so test metrics are optimistically biased (to be fixed in Phase 1 with a separate validation split). SMOTE, when imbalanced-learn is installed, is applied to the WDBC training split only. The 5-model comparison is stratified 10-fold CV on the WDBC training split. No probability calibration and no uncertainty intervals.

**Research benchmark: XGBoost on WDBC** (not used by the app; test n = 114): AUC 0.9957 · accuracy 97.37% · sensitivity 95.24% · specificity 98.61%

<details><summary>5-model stratified 10-fold CV on the WDBC training split</summary>

| Model | AUC (mean ± SD) | Accuracy | Recall |
|---|---|---|---|
| XGBoost | 0.9919 ± 0.0101 | 96.49% | 94.71% |
| LightGBM | 0.9909 ± 0.0127 | 96.27% | 93.53% |
| RandomForest | 0.9889 ± 0.0150 | 96.04% | 92.94% |
| SVM (RBF) | 0.9942 ± 0.0096 | 97.14% | 94.71% |
| LogisticRegression | 0.9952 ± 0.0084 | 97.58% | 95.88% |

</details>

**Data:** UCI Mammographic Mass (UCI id 161): 961 raw rows, 831 after dropping rows with missing values (428 benign / 403 malignant). **Knowledge base:** 32 chunks indexed from 4 of 10 source documents (384-d dense embeddings).
<!-- METRICS:END -->

## 6. Retrieval-augmented generation

- **Knowledge base:** Markdown files in `knowledge_base/sources/`, written for the project as
  summaries of BI-RADS terminology, follow-up guidance and safety notes. They do not yet cite
  primary sources. `scripts/build_knowledge_base.py` chunks every file and builds the index,
  but the committed index predates six of the documents, so only part of the corpus is
  indexed (counts in the generated block above).
- **Chunking:** split on blank lines; paragraphs longer than 1,200 characters are split at
  sentence boundaries; fragments under 80 characters (such as headings) are dropped.
- **Embeddings and index:** `all-MiniLM-L6-v2` (384-d), L2-normalised, FAISS `IndexFlatIP`
  (cosine similarity).
- **Known issue:** `sentence-transformers` is not in `requirements.txt`. Without it, the
  retriever falls back to a 128-d TF-IDF+SVD query padded with zeros to 384-d and compares it
  against the MiniLM vectors. The two spaces are unrelated, so results are not meaningful.
- **Not measured yet:** retrieval recall, answer faithfulness and citation accuracy (Phase 2).

## 7. LLM layer

- Google Gemini (`gemini-2.5-flash` by default) through the OpenAI-compatible endpoint,
  using the `openai` package. The key comes from `GEMINI_API_KEY`.
- On quota errors the client tries `gemini-2.0-flash`, then `gemini-2.0-flash-lite`. If all
  are exhausted, it returns the assembled prompt context (estimate, SHAP summary, retrieved
  chunks) instead of a generated answer.
- If no key is set, `/api/chat` returns the top retrieved passages with a notice.
- The system prompt requires an educational, non-diagnostic tone, use of the retrieved
  documents only, explicit uncertainty, and a closing reminder to see a clinician. Prompts
  label the probability as an uncalibrated model output with no uncertainty interval.

## 8. Safety layer

`src/safety.py` applies 12 case-insensitive regex rewrites to every LLM response (for example,
"you have cancer" becomes "the features you entered are associated with elevated risk"). It
appends a standard disclaimer, plus a high-risk notice when the probability is ≥ 0.65. The
filter is deterministic but easy to bypass with paraphrases. Its effectiveness has not been
measured; a red-team suite is planned for Phase 2.

## 9. Experimental and inactive modules

- **Image model:** EfficientNet-B0 (timm, 2 classes, 224×224, ImageNet normalisation) with a
  hook-based Grad-CAM on `conv_head`. The weights `models/efficientnet_cbis.pth` were trained
  on CBIS-DDSM on Kaggle. The training and evaluation run is not stored in this repo. Earlier
  project notes reported a validation AUC of about 0.62, which cannot be reproduced from the
  repo. `/api/assess-image` returns **HTTP 503** when the weights or torch/torchvision/timm
  are missing, or when the weights fail to load. It never substitutes a heuristic score.
  Scores are uncalibrated.
- **Chain-of-thought / multi-agent:** CoT is one LLM call with a fixed six-step structure.
  Multi-agent runs Analyst → Clinician → Safety reviewer as three sequential calls. Either one
  falls back to the single pipeline on error. Neither has been evaluated against the single pipeline.
- **Classical risk models:** a SEER-style age incidence table plus relative-risk multipliers
  (simplified Gail) and an approximate Tyrer-Cuzick. These are educational approximations.
- **Drift monitor:** a rolling window of 200 predictions. PSI against a normal reference for
  the probabilities, a KS test for age, chi-square for the coded features. The reference
  values are placeholders, not training statistics, and the API response says so.
- **GraphRAG, stacking ensemble, calibration, Optuna:** the code exists, but nothing has been
  built or trained, so none of it affects predictions.

## 10. Design decisions

1. **One served model.** LightGBM on the mammographic descriptors serves the app; WDBC is only a benchmark.
2. **BI-RADS excluded** to avoid label leakage.
3. **Rows with missing values dropped** rather than imputed: they are coded categories, and
   only 130 of 961 rows are affected.
4. **SHAP drives the retrieval query**, so retrieved text relates to what moved the estimate.
5. **Small curated knowledge base** rather than a large general corpus.
6. **Safety enforced in code** as well as in the prompt.
7. **Fail closed.** Missing data, missing image weights or an unreachable API produce an
   error, never a substitute estimate.
8. **Docs generated from artifacts.** README and DESIGN metrics come from
   `models/model_metadata.json` via `scripts/update_readme_metrics.py`, and the UI reads
   `/api/model-info`.
9. **Gemini** for its free API tier.

## 11. Limitations

1. Small single-institution datasets (Erlangen 2003–2006; Wisconsin), so results may not transfer.
2. Four coarse descriptors; real systems use many more, including calcifications and imaging.
3. Optimistic test metrics, uncalibrated probabilities, no confidence intervals (see §5).
4. Retrieval, answer quality and the safety filter are not yet evaluated (see §6 and §8).
5. Batch predictions classify at 0.50, while single assessments use the Youden threshold.
6. Not validated in any clinical setting.

## 12. Project structure

```
breastcare_ai/
├── server.py                 FastAPI app (serves app/index.html + API)
├── app/index.html            single-page frontend (the only frontend)
├── Procfile                  uvicorn server:app
├── requirements.txt
├── data/
│   ├── README.md             dataset provenance
│   ├── raw/                  original UCI file (checksum-pinned)
│   └── processed/cleaned.csv
├── models/                   LightGBM, SHAP, XGBoost, image weights, model_metadata.json
├── knowledge_base/           sources/, chunks/, vectorstore/ (FAISS)
├── scripts/
│   ├── download_data.py      fetch + verify UCI data (ucimlrepo + SHA-256)
│   ├── build_knowledge_base.py
│   └── update_readme_metrics.py
├── src/
│   ├── config.py  pipeline.py  safety.py  model_info.py
│   ├── ml/        train, predict, explain, image_model (+ inactive: ensemble, calibration, hyperopt)
│   ├── rag/       chunker, build_vectorstore, retriever, prompt_builder (+ inactive: graph_*)
│   ├── llm/       client.py (Gemini)
│   ├── agents/    CoT / multi-agent orchestration
│   ├── audit/     SQLite log, PDF report, anonymiser
│   ├── analytics/ batch prediction, model comparison, global SHAP, drift monitor
│   └── risk/      simplified classical risk models
├── tests/                    pytest suite (+ fixtures/synthetic_mammographic.py, test-only)
└── docs/DESIGN.md            this file
```

## 13. References

1. Elter, M., Schulz-Wendtland, R., & Wittenberg, T. (2007). The prediction of breast cancer biopsy
   outcomes using two CAD approaches that both emphasize an intelligible decision process. *Medical Physics*.
2. Elter, M. (2007). Mammographic Mass [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C53K6Z
3. Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear feature extraction for breast
   tumor diagnosis. *IS&T/SPIE Electronic Imaging*.
4. American College of Radiology (2013). *ACR BI-RADS Atlas*, 5th edition.
5. Ke, G. et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS*.
6. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*.
7. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
8. Youden, W. J. (1950). Index for rating diagnostic tests. *Cancer*.
