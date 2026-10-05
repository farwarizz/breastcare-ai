# BreastCare AI: design document

This describes what the code does today, including the parts that are experimental or not
active. For a short overview see the [README](../README.md); for intended use and limitations
see the [model card](MODEL_CARD.md); for every number see the
[evaluation report](../reports/evaluation.md).

## 1. Scope

BreastCare AI is a **research and educational prototype**. It estimates the probability that a
breast mass is malignant from four structured inputs (age, and the BI-RADS descriptors shape,
margin and density), attaches a conformal prediction set, explains the estimate with SHAP, and
answers follow-up questions with an LLM grounded in a small knowledge base.

It is **not** a medical device. It is not FDA-cleared and not clinically validated, it does not
diagnose, and it does not recommend treatment. The risk model works only on the structured
descriptors, not on images. The separate, experimental image model is described in §9.

Motivation, from the dataset documentation: the low positive predictive value of breast biopsy
after mammography leads to roughly 70% of biopsies having benign outcomes. The UCI data was
published to study whether descriptor-based models can support the biopsy decision.

## 2. Data

UCI Mammographic Mass (id 161): 961 raw cases, 831 after dropping rows with missing values
(428 benign / 403 malignant). The raw file is the original, checksum-pinned UCI file,
cross-checked against `ucimlrepo` ([provenance](../data/README.md)). Training refuses any other
file, and no production code can generate substitute data. The BI-RADS assessment column is
dropped from the inputs because it is the radiologist's conclusion (label leakage); it is kept
only as a reference for evaluation.

The classic WDBC dataset is no longer part of the pipeline. It lives in a reproducible appendix
(`experiments/wdbc_benchmark.py` → [reports/wdbc_benchmark.md](../reports/wdbc_benchmark.md)).

## 3. Architecture and request flow

```
Browser: app/index.html  ──fetch──►  FastAPI: server.py
                                           │
             ┌─────────────────────────────┼──────────────────────────────┐
             ▼                             ▼                              ▼
    /api/assess, /api/report      /api/chat (full pipeline)      /api/model-info, /figures
    run_pipeline_ml_only()        run_pipeline()                 saved metadata + figures
```

`src/pipeline.py`, `run_pipeline()` (used by `/api/chat`):

1. `predict()` (`src/ml/predict.py`): the served model's score passes through the stored
   calibration map. The result is the calibrated probability, the conformal prediction set,
   the class at the development-chosen threshold, and a risk category
   (Low < 0.35 ≤ Moderate < 0.65 ≤ Elevated).
2. `explain()` (`src/ml/explain.py`): per-feature SHAP contributions in log-odds space. These
   are exact for the linear model (interventional), or TreeSHAP for LightGBM. Also returns
   the dominant feature and a text summary.
3. `build_retrieval_query()`: dominant feature + direction + the start of the SHAP summary + the
   user's question.
4. `rag_query()`: top-3 chunks from FAISS. GraphRAG is used only if a caller passes
   `use_hybrid_rag=true`. The UI never does, and without a built graph it falls back to FAISS.
5. If the probability is ≥ 0.65, the first chunk of the safety-disclaimer document is appended.
6. `build_prompt()` → `call_llm()`: Gemini, called through Google's OpenAI-compatible endpoint.
7. `filter_response()`: regex rewrite of diagnostic phrases, plus the standard (or high-risk) disclaimer.
8. If a session id is given, the prediction is written to the SQLite audit log.

`run_pipeline_ml_only()` (used by `/api/assess` and `/api/report`) runs steps 1–4 without an LLM call.

## 4. Component status

| Component | Where | Status |
|---|---|---|
| Served model, calibration, conformal sets | `src/ml/predict.py`, `artifacts.py`, `calibration.py`, `conformal.py` | Active on every assessment |
| Training + evaluation pipeline | `src/ml/train.py`, `models.py`, `evaluation.py`, `figures.py` | Active (offline, `python -m src.ml.train`) |
| SHAP contributions | `src/ml/explain.py` | Active |
| Challenger model | `src/analytics/model_comparison.py` | Active in model comparison |
| FAISS dense retrieval | `src/rag/retriever.py` | Active; needs `sentence-transformers` at query time (§6) |
| Gemini generation | `src/llm/client.py` | Active in `/api/chat` with `GEMINI_API_KEY` |
| Regex safety filter | `src/safety.py` | Active on every LLM response |
| Chain-of-thought / multi-agent modes | `src/agents/` | Optional (chat mode selector); not evaluated |
| Audit log, PDF report, batch CSV | `src/audit/`, `src/analytics/batch_predict.py` | Active |
| Dataset-wide SHAP | `src/analytics/global_shap.py` | Active in `/api/global-shap` |
| EfficientNet-B0 image model | `src/ml/image_model.py` | Experimental; HTTP 503 unless the real weights and torch/timm are present |
| Simplified Gail / Tyrer-Cuzick | `src/risk/` | Active in `/api/risk-profile`; approximations, not the validated calculators |
| Drift monitor | `src/analytics/model_monitor.py` | Placeholder baselines, illustrative only |
| GraphRAG | `src/rag/graph_builder.py`, `graph_retriever.py` | Not active: no graph has been built |

## 5. Machine learning

### Protocol

Stratified split of the 831 cleaned rows (seed 2026; row ids in `models/splits.json`):

| Split | Share | Used for |
|---|---|---|
| fit | 60% | 5-fold CV model selection; out-of-fold predictions for calibration and thresholds; training the final models |
| conformal | 20% | conformal cut-offs only |
| test | 20% | final evaluation, used once |

The final models are trained on the fit split only, not refit on fit + conformal. Refitting would
invalidate the conformal guarantee, which needs calibration data the model has never seen.

### Candidates and selection (`src/ml/models.py`)

- **Logistic regression:** standardised age plus one-hot shape, margin and density (shape and margin
  are nominal codes). C ∈ {0.01, 0.1, 1, 10}.
- **LightGBM, monotone in age and shape:** risk may not decrease with age or with shape
  (round < oval < lobular < irregular). Margin is left unconstrained because its codes are not in risk
  order: obscured is less suspicious than microlobulated, as in BI-RADS and in the development data.
  Density is unconstrained because about 90% of masses share one value.
- **LightGBM, unconstrained.** Both LightGBM grids use num_leaves ∈ {4, 8, 16} and
  min_child_samples ∈ {10, 20, 40}; the tree count comes from early stopping inside CV folds.

**Rule**, fixed before looking at the test set: lowest mean 5-fold CV log-loss on the fit split; a
simpler family within one standard error of the best is preferred (logistic regression < monotone
LightGBM < LightGBM). The runner-up family is kept as the challenger. LightGBM's early stopping on
the CV folds makes its CV score slightly optimistic, which biases this rule *towards* LightGBM.

### Calibration and thresholds

Out-of-fold predictions on the fit split choose a calibration method (none / Platt sigmoid /
isotonic, by cross-validated Brier score, with ties going to the simpler method). The same
predictions choose the decision threshold (Youden's J) and a high-sensitivity threshold (highest
threshold with sensitivity ≥ 95%). This is the scheme of scikit-learn's
`CalibratedClassifierCV(ensemble=False)`. Calibrator parameters are plain numbers stored in the
metadata.

### Conformal prediction sets

Mondrian (class-conditional) split conformal prediction with the LAC score, at α = 0.10, fitted on
the conformal split. Benign enters the set when p ≤ q̂_benign, and Malignant when p ≥ 1 − q̂_malignant.
Each true class gets at least 90% coverage on average over random draws. The method is
unit-tested by simulation (`tests/test_uncertainty.py`).

### Evaluation (`src/ml/evaluation.py`)

On the test split, used once: AUROC, AUPRC, Brier, ECE, calibration intercept and slope, and
sensitivity / specificity / PPV / NPV at the chosen threshold. All come with 95% percentile
bootstrap CIs (2,000 resamples, the same indices for every model, so differences are paired).
Also reported: conformal coverage overall and per class, the high-sensitivity operating point, the
radiologists' BI-RADS assessment as a reference (rows with BI-RADS 1–5), the monotonic-constraint
ablation, age subgroups (< 50 vs ≥ 50), and a decision curve. Training is deterministic, and runs
are logged to MLflow when it is installed.

Current results (generated from `models/model_metadata.json`):

<!-- BEGIN GENERATED: results (generated by scripts/update_readme_metrics.py from models/model_metadata.json; do not edit by hand) -->
**Served model: Logistic regression**, chosen by a fixed rule. Lowest mean 5-fold CV log-loss on the fit split; a simpler family within one standard error of the best is preferred (logistic regression < monotone LightGBM < LightGBM).
Held-out test set: n = 167 (81 malignant), used once; 95% bootstrap CIs in brackets.

| Metric | Logistic regression (served) | LightGBM (monotone in age and shape) (challenger) |
|---|---|---|
| AUROC | 0.90 (0.84–0.94) | 0.90 (0.84–0.94) |
| AUPRC | 0.87 (0.79–0.93) | 0.87 (0.78–0.93) |
| Brier score (lower is better) | 0.132 (0.110–0.160) | 0.129 (0.104–0.159) |
| Calibration slope (ideal 1) | 1.63 (1.25–2.21) | 1.35 (1.02–1.86) |
| Calibration intercept (ideal 0) | -0.02 (-0.35 to 0.27) | -0.04 (-0.40 to 0.28) |
| Sensitivity at its threshold | 0.90 (0.83–0.96) | 0.91 (0.85–0.96) |
| Specificity at its threshold | 0.81 (0.72–0.89) | 0.77 (0.67–0.85) |
| Decision threshold (chosen on dev data) | 0.514 | 0.474 |

- **Served vs challenger:** AUROC difference +0.001 (-0.011 to +0.015): no significant difference in AUROC (the 95% CI includes 0).
- **Calibration:** the served model is underconfident on this test split (predictions less extreme than outcomes; slope > 1); intercept -0.02 (-0.35 to 0.27). Calibration step chosen on out-of-fold data: `none`.
- **90% conformal prediction sets:** coverage 0.87 (0.81–0.92) (consistent with the 90% target); benign 0.83 (0.74–0.90), malignant 0.91 (0.85–0.97). 89% of cases get a single label.
- **High-sensitivity operating point** (threshold 0.20, chosen on development data): sensitivity 0.99 (0.96–1.00), specificity 0.42 (0.31–0.52).
- **Reference, radiologists' BI-RADS assessment** (ordinal category, n = 164): AUROC 0.83 (0.78–0.88).
- **Decision curve:** the served model matches 'biopsy everyone' at the 2% BI-RADS biopsy threshold and adds net benefit over biopsying everyone from a threshold of 20%.
- **Subgroups (age, split at 50):** age < 50: sensitivity 0.71 (0.46–0.93), n malignant = 14; age ≥ 50: sensitivity 0.94 (0.88–0.99), n malignant = 67; sensitivity is 0.23 lower in the age < 50 group at the global threshold.
- **Cross-validation vs test:** 5-fold CV AUROC on the fit split 0.84; test AUROC 0.90. The CV figure is the more conservative estimate for new data.

<picture><source media="(prefers-color-scheme: dark)" srcset="../reports/figures/roc_pr_dark.png"><img src="../reports/figures/roc_pr_light.png" alt="ROC and precision-recall curves" width="100%"></picture>
<picture><source media="(prefers-color-scheme: dark)" srcset="../reports/figures/calibration_dark.png"><img src="../reports/figures/calibration_light.png" alt="Calibration curve" width="100%"></picture>
<picture><source media="(prefers-color-scheme: dark)" srcset="../reports/figures/decision_curve_dark.png"><img src="../reports/figures/decision_curve_light.png" alt="Decision curve" width="100%"></picture>
<picture><source media="(prefers-color-scheme: dark)" srcset="../reports/figures/conformal_sets_dark.png"><img src="../reports/figures/conformal_sets_light.png" alt="Conformal prediction sets" width="100%"></picture>

**Data:** UCI Mammographic Mass (UCI id 161): 831 cleaned cases. Split: fit 498 · conformal 166 · test 167. **Knowledge base:** 32 chunks from 4 of 10 documents.
Full details: [reports/evaluation.md](../reports/evaluation.md).
<!-- END GENERATED: results -->

## 6. Retrieval-augmented generation

- **Knowledge base:** Markdown files in `knowledge_base/sources/`, written for the project as
  summaries of BI-RADS terminology, follow-up guidance and safety notes. They do not yet cite primary
  sources. `scripts/build_knowledge_base.py` chunks every file and builds the index, but the committed
  index predates six of the documents, so only part of the corpus is indexed (counts above).
- **Chunking:** split on blank lines; paragraphs longer than 1,200 characters are split at sentence
  boundaries; fragments under 80 characters (such as headings) are dropped.
- **Embeddings and index:** `all-MiniLM-L6-v2` (384-d), L2-normalised, FAISS `IndexFlatIP` (cosine).
- **Known issue:** `sentence-transformers` is not in `requirements.txt` (it is in
  `requirements-dev.txt`). Without it, the retriever falls back to a 128-d TF-IDF+SVD query padded with
  zeros to 384-d and compares it against the MiniLM vectors. The two spaces are unrelated, so results
  are not meaningful.
- **Not measured yet:** retrieval recall, answer faithfulness and citation accuracy (Phase 2).

## 7. LLM layer

- Google Gemini (`gemini-2.5-flash` by default) through the OpenAI-compatible endpoint, using the
  `openai` package. The key comes from `GEMINI_API_KEY`.
- On quota errors the client tries `gemini-2.0-flash`, then `gemini-2.0-flash-lite`. If all are
  exhausted, it returns the assembled prompt context instead of a generated answer.
- If no key is set, `/api/chat` returns the top retrieved passages with a notice.
- Every prompt states the calibrated probability together with its conformal set, worded by one helper
  (`risk_summary_lines`). The system prompt requires an educational, non-diagnostic tone, use of the
  retrieved documents only, explicit uncertainty, and a closing reminder to see a clinician.

## 8. Safety layer

`src/safety.py` applies 12 case-insensitive regex rewrites to every LLM response (for example,
"you have cancer" becomes "the features you entered are associated with elevated risk"). It appends a
standard disclaimer, plus a high-risk notice when the probability is ≥ 0.65. The filter is
deterministic but easy to bypass with paraphrases. Its effectiveness has not been measured; a red-team
suite is planned for Phase 2.

## 9. Experimental and inactive modules

- **Image model:** EfficientNet-B0 (timm, 2 classes, 224×224, ImageNet normalisation) with a hook-based
  Grad-CAM on `conv_head`. The weights `models/efficientnet_cbis.pth` were trained on CBIS-DDSM on
  Kaggle. The training and evaluation run is not stored in this repo; earlier project notes reported a
  validation AUC of about 0.62, which cannot be reproduced from the repo. `/api/assess-image` returns
  **HTTP 503** when the weights or dependencies are missing or fail to load. It never substitutes a
  heuristic score.
- **Chain-of-thought / multi-agent:** CoT is one LLM call with a fixed six-step structure. Multi-agent
  runs Analyst → Clinician → Safety reviewer as three calls. Either one falls back to the single
  pipeline on error. Neither has been evaluated.
- **Classical risk models:** a SEER-style age incidence table plus relative-risk multipliers
  (simplified Gail) and an approximate Tyrer-Cuzick. These are educational approximations.
- **Drift monitor:** a rolling window of 200 predictions. The reference values are placeholders,
  not training statistics, and the API response says so.
- **GraphRAG:** the code exists, but no graph has been built, so it never affects answers.

## 10. Design decisions

1. **A selection rule, not a favourite model.** The served model is whatever the fixed rule picks; the
   runner-up is kept as a challenger with a paired comparison.
2. **Nominal codes one-hot encoded** for the linear model; **monotonic constraints only where clinical
   knowledge and data agree** (age, shape).
3. **Calibration learned out-of-fold**, so it reflects unseen-data behaviour without a separate split.
4. **Conformal sets instead of an invented interval**, class-conditional so malignant cases get their
   own guarantee.
5. **Test set used once.** Every threshold, calibrator and model choice comes from development data.
6. **Exact explanations where possible.** SHAP values are exact for the linear model; base value +
   contributions reproduces the model's log-odds (tested).
7. **BI-RADS excluded** from the inputs (leakage), but reported as a clinical reference.
8. **Fail closed.** Missing data, missing image weights or an unreachable API produce an error, never
   a substitute estimate.
9. **Docs generated from artifacts.** README, DESIGN, the model card and the evaluation report come
   from `models/model_metadata.json`. The UI reads `/api/model-info` and the figures, and a test fails
   on drift.
10. **Gemini** for its free API tier.

## 11. Limitations

1. Small single-institution data (Erlangen 2003–2006), so results may not transfer; test CIs are wide.
2. Four coarse, reader-dependent descriptors; real systems use far more information.
3. Calibration and conformal coverage are checked on one test split; re-check on new data.
4. Retrieval, answer quality and the safety filter are not yet evaluated (§6, §8).
5. Not validated in any clinical setting.

## 12. Project structure

```
breastcare_ai/
├── server.py                 FastAPI app (serves app/index.html, API, /figures)
├── app/index.html            single-page frontend (the only frontend)
├── Procfile                  uvicorn server:app
├── requirements.txt          runtime · requirements-dev.txt: training, tests, figures, MLflow
├── data/                     README (provenance) · raw/ (checksum-pinned UCI file) · processed/
├── models/                   served_model.joblib · challenger_model.joblib · model_metadata.json ·
│                             splits.json · efficientnet_cbis.pth (experimental image model)
├── reports/                  evaluation.md (generated) · figures/ (light + dark) · wdbc_benchmark.md
├── knowledge_base/           sources/, chunks/, vectorstore/ (FAISS)
├── experiments/              wdbc_benchmark.py (appendix)
├── scripts/                  download_data.py · build_knowledge_base.py · update_readme_metrics.py
├── src/
│   ├── config.py  pipeline.py  safety.py  model_info.py
│   ├── ml/        data · models · calibration · conformal · evaluation · figures · train ·
│   │              artifacts · predict · explain · image_model
│   ├── rag/       chunker, build_vectorstore, retriever, prompt_builder (+ inactive: graph_*)
│   ├── llm/       client.py (Gemini)
│   ├── agents/    CoT / multi-agent orchestration
│   ├── audit/     SQLite log, PDF report, anonymiser
│   ├── analytics/ batch prediction, model comparison, global SHAP, drift monitor
│   └── risk/      simplified classical risk models
├── tests/                    pytest suite (+ fixtures/synthetic_mammographic.py, test-only)
└── docs/                     DESIGN.md (this file) · MODEL_CARD.md
```

## 13. References

1. Elter, M., Schulz-Wendtland, R., & Wittenberg, T. (2007). The prediction of breast cancer biopsy
   outcomes using two CAD approaches that both emphasize an intelligible decision process. *Medical Physics*.
2. Elter, M. (2007). Mammographic Mass [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C53K6Z
3. American College of Radiology (2013). *ACR BI-RADS Atlas*, 5th edition.
4. Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
5. Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to conformal prediction and
   distribution-free uncertainty quantification. arXiv:2107.07511.
6. Van Calster, B., McLernon, D. J., van Smeden, M., Wynants, L., & Steyerberg, E. W. (2019).
   Calibration: the Achilles heel of predictive analytics. *BMC Medicine*, 17, 230.
7. Vickers, A. J., & Elkin, E. B. (2006). Decision curve analysis: a novel method for evaluating
   prediction models. *Medical Decision Making*, 26(6), 565–574.
8. Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized
   likelihood methods. *Advances in Large Margin Classifiers*.
9. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*, 2nd ed.
   (one-standard-error rule, §7.10).
10. Youden, W. J. (1950). Index for rating diagnostic tests. *Cancer*.
11. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
12. Ke, G. et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS*.
13. Mitchell, M. et al. (2019). Model cards for model reporting. *FAT\* '19*.
14. Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear feature extraction for breast
    tumor diagnosis. *IS&T/SPIE Electronic Imaging* (WDBC, appendix).
