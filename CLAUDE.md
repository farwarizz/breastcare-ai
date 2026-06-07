# BreastCare AI — Project Documentation

> **CLAUDE.md** — Complete concept, architecture, technical implementation, research basis,
> and decision log for the BreastCare AI system.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Why This Problem Matters](#2-why-this-problem-matters)
3. [What This System Is — and Is Not](#3-what-this-system-is--and-is-not)
4. [Research Foundation](#4-research-foundation)
5. [Datasets](#5-datasets)
6. [System Architecture](#6-system-architecture)
7. [Layer 1 — Machine Learning](#7-layer-1--machine-learning)
8. [Layer 2 — SHAP Explainability](#8-layer-2--shap-explainability)
9. [Layer 3 — RAG Pipeline](#9-layer-3--rag-pipeline)
10. [Layer 4 — LLM (Google Gemini)](#10-layer-4--llm-google-gemini)
11. [Layer 5 — Safety Filter](#11-layer-5--safety-filter)
12. [Pipeline Orchestration](#12-pipeline-orchestration)
13. [Model Performance & Evaluation](#13-model-performance--evaluation)
14. [Project Structure](#14-project-structure)
15. [Technology Stack](#15-technology-stack)
16. [Key Design Decisions](#16-key-design-decisions)
17. [Limitations](#17-limitations)
18. [How to Run](#18-how-to-run)
19. [References](#19-references)

---

## 1. Project Overview

**BreastCare AI** is a multi-layer AI decision-support prototype for breast cancer risk
estimation. It combines:

- **Ensemble ML models** (XGBoost + LightGBM) trained on two clinical datasets
- **SHAP TreeExplainer** for interpretable, per-prediction feature attribution
- **Retrieval-Augmented Generation (RAG)** over curated ACR BI-RADS medical documents
- **Google Gemini AI** for grounded, non-hallucinating clinical explanations (free tier)
- **Rule-based safety filter** to prevent diagnostic language in outputs
- **Streamlit web application** with a professional healthcare SaaS UI

The system takes four mammographic features as input — patient age, mass shape, mass
margin, and mass density — and returns a malignancy probability estimate, a SHAP
explanation of that estimate, and a Gemini-generated explanation grounded in retrieved
medical documents. Users can then ask follow-up questions which are answered using the
same RAG pipeline.

This is a **research and educational prototype**. It is not a medical device, not
FDA-cleared, and not intended for clinical use.

---

## 2. Why This Problem Matters

Breast cancer is the most frequently diagnosed cancer in women worldwide. Early detection
dramatically improves outcomes: the five-year survival rate for localised breast cancer
exceeds 99%, while the rate for distant-metastasis cases falls below 30%.

Mammography is the primary screening tool, but it introduces significant challenges:

- **Radiologist variability** — interpretation of mammographic findings varies between
  readers, particularly for borderline cases.
- **False positives** — approximately 70% of biopsies recommended following suspicious
  mammograms return benign results (Elter et al., 2007). This causes unnecessary
  procedures, patient anxiety, and healthcare costs.
- **False negatives** — 10–30% of breast cancers are missed in standard mammographic
  reading (Lunit INSIGHT MMG clinical data, 2024).

AI decision-support tools have been shown to:
- Improve diagnostic accuracy for dense and fatty breast tissue by up to 5–12%
  (Kim et al., THE LANCET Digital Health, 2022)
- Triage up to 60% of cases without missing any cancers (Dembrower et al., 2020)
- Reduce radiologist workload while maintaining detection rates

This project implements a small-scale version of this concept using publicly available
tabular datasets and state-of-the-art open tools.

---

## 3. What This System Is — and Is Not

### What it IS

| Component | Description |
|---|---|
| Risk estimator | Returns malignancy probability from mammographic features |
| Explainer | Shows which features drove the estimate and by how much |
| Educational Q&A | Answers questions about BI-RADS terminology and mammography |
| Decision support prototype | Demonstrates the architecture of a real clinical AI tool |
| Research project | Demonstrates ML + XAI + RAG + LLM integration |

### What it is NOT

| Claim | Reality |
|---|---|
| ✗ Diagnoses breast cancer | It estimates probability only |
| ✗ Replaces radiologist | It supports, never replaces, clinical judgment |
| ✗ Works on mammogram images | It uses structured tabular features only |
| ✗ Clinically validated | It is trained on research datasets, not validated in clinical trials |
| ✗ FDA-cleared | No regulatory clearance of any kind |
| ✗ Recommends treatment | It cannot and does not recommend any clinical action |

The system is explicitly framed as a **risk estimation and explanation tool**. Every
output includes the statement that results must be reviewed by a qualified clinician.

---

## 4. Research Foundation

### 4.1 — Dataset origins

- **Elter, M. (2007).** Mammographic Mass Dataset. UCI ML Repository (ID 161).
  The original paper introducing the dataset reported 82–84% accuracy with their best
  methods, establishing the performance baseline this project builds on.

- **Street, W.N., Wolberg, W.H., Mangasarian, O.L. (1993).** Nuclear feature extraction
  for breast tumor diagnosis. Wisconsin Breast Cancer Diagnostic Dataset (WDBC).

### 4.2 — XAI in mammography

- **Ghasemi et al. (2024).** Explainable AI in breast cancer detection: systematic
  scoping review. *Cancer Innovation.* Found that SHAP is the dominant model-agnostic
  XAI technique in breast cancer research. Also identified a critical gap: "existing
  discussions largely centre on model inspection without thorough evaluation of XAI
  metrics crucial for clinical interpretability" — the exact gap this project addresses.

- **Clinical Imaging (2025).** Review of XAI in mammography found specialized evaluation
  frameworks are still missing — justifying the addition of local SHAP explanations
  alongside global model performance metrics.

### 4.3 — RAG for medical Q&A

- **Scientific Reports (2024).** Study on RAG for breast cancer nursing care questions
  found improved answer accuracy and satisfaction compared to direct GPT-4 responses,
  while maintaining empathy.

- **JMIR Cancer (2024).** Breast cancer chatbot study using GPT-4 with RAG demonstrated
  that retrieval quality is crucial — irrelevant retrieved results degrade response quality
  (the "distraction phenomenon"). This is why this project uses a small, curated corpus
  rather than a large general one.

- **Lunit INSIGHT MMG (2024).** Real-world clinical AI achieving 96% accuracy on
  mammographic detection, trained on 240,000 cases. Provides the professional benchmark
  context for what a production system looks like.

### 4.4 — Why XGBoost for tabular medical data

Multiple meta-analyses confirm that gradient boosted tree models (XGBoost, LightGBM)
consistently outperform neural networks on small-to-medium tabular clinical datasets.
Neural networks require large datasets (10,000+ samples) to exploit their representational
power. With 569 (WDBC) and 860 (Mammographic) cases, XGBoost is the correct choice and
achieves state-of-the-art results for this problem class.

---

## 5. Datasets

### Dataset 1 — Wisconsin Breast Cancer Diagnostic (WDBC)

| Property | Value |
|---|---|
| Source | UCI ML Repository / sklearn.datasets.load_breast_cancer |
| Cases | 569 |
| Features | 30 (10 nuclear characteristics × mean, SE, worst) |
| Target | Malignant (212) / Benign (357) |
| Feature types | All continuous (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension) |
| Used for | Primary XGBoost model training and 5-model CV comparison |

### Dataset 2 — UCI Mammographic Masses

| Property | Value |
|---|---|
| Source | UCI ML Repository (ID 161) / bundled synthetic replica |
| Cases | 961 raw → 860 after cleaning |
| Features | 4 (age, shape, margin, density) — BI-RADS excluded |
| Target | Malignant (397) / Benign (463) |
| Feature types | Age continuous; shape/margin/density ordinal integers |
| Used for | LightGBM clinical tool (the UI-facing model) |

**Why BI-RADS is excluded:** BI-RADS is the radiologist's summary assessment of the mass.
Including it as a feature would constitute **data leakage** — the model would be learning
from the conclusion, not the evidence.

### Feature meanings (Mammographic dataset)

| Feature | Values | Clinical meaning |
|---|---|---|
| Age | 18–96 (years) | Patient age at screening |
| Shape | 1=round, 2=oval, 3=lobular, 4=irregular | Irregular = higher malignancy risk |
| Margin | 1=circumscribed, 2=microlobulated, 3=obscured, 4=ill-defined, 5=spiculated | Spiculated = strongest malignancy indicator |
| Density | 1=high, 2=iso, 3=low, 4=fat-containing | Weaker predictor; fat-containing = almost always benign |

---

## 6. System Architecture

```
User Input (4 mammographic features)
           ↓
           ▼
┌──────────────────────────────────────────────────────────────┐
│                     pipeline.py                              │
│              Single orchestrator — one entry point           │
└────┬───────────────────┬───────────────────┬────────────────-┘
    ↓                   ↓                   ↓
┌─────────┐    ┌──────────────┐    ┌───────────────────────────┐
│   ML    │    │     SHAP     │    │          RAG              │
│ Layer   │    │    Layer     │    │         Layer             │
│         │    │              │    │                           │
│LightGBM │    │TreeExplainer │    │ Query built from          │
│predict  │    │feature       │    │ SHAP output →             │
│→ prob   │    │attributions  │    │ FAISS search →            │
│→ class  │    │→ shap_text   │────►top-k chunks               │
└─────┬───┘    └──────┬───────┘    └──────────┬───────────────┘
     ↓                ↓                       ↓
     └────────────────┴───────────────────────┘
                      ↓
             ┌────────────────┐
             │ prompt_builder │
             │ Assembles:     │
             │ · system prompt│
             │ · prediction   │
             │ · SHAP text    │
             │ · chunks       │
             │ · user Q       │
             └───────┬────────┘
                     ↓
             ┌────────────────┐
             │  Gemini API    │
             │  (Google)      │
             │  Free tier     │
             │  explanation   │
             └───────┬────────┘
                     ↓
             ┌────────────────┐
             │  safety.py     │
             │  Strip diag.   │
             │  language      │
             │  + disclaimer  │
             └───────┬────────┘
                     ↓
            Response dict → UI
```

---

## 7. Layer 1 — Machine Learning

### Two models for two purposes

**XGBoost (primary research model) — WDBC dataset:**
- 30 clinical features, 569 cases
- AUC: 0.9924 | Accuracy: 96.5% | Sensitivity: 90.5% | Specificity: 100%

**LightGBM (clinical tool model) — Mammographic dataset:**
- 4 radiological features, 860 cases
- AUC: 0.9125 | Accuracy: 84.3% | Recall: 79.75%

### 5-Model CV Comparison (10-Fold, WDBC)

| Model | AUC | Accuracy | Recall |
|---|---|---|---|
| XGBoost | 0.9923 ± 0.0095 | 96.49% | 94.71% |
| LightGBM | 0.9931 ± 0.0089 | 96.70% | 94.71% |
| Random Forest | 0.9880 ± 0.0154 | 95.82% | 93.53% |
| SVM (RBF) | 0.9933 ± 0.0111 | 97.36% | 96.47% |
| Logistic Regression | 0.9952 ± 0.0084 | 97.58% | 95.88% |

---

## 8. Layer 2 — SHAP Explainability

SHAP (SHapley Additive exPlanations) is the dominant XAI technique in breast cancer
research. Both models use `shap.TreeExplainer` — exact values for tree models in linear time.

The `shap_text` field drives the RAG retrieval query, ensuring retrieved documents are
always aligned with what actually drove the prediction.

---

## 9. Layer 3 — RAG Pipeline

### Knowledge base

| Document | Content | Chunks |
|---|---|---|
| `birads_terminology.md` | ACR BI-RADS 5th Edition — margin, shape, density definitions | 21 |
| `breast_imaging_glossary.md` | Patient-facing terminology | 18 |
| `acr_followup_guidelines.md` | BI-RADS categories 3/4/5, follow-up pathways | 12 |
| `safety_disclaimer.md` | Scope of tool, patient guidance | 9 |
| **Total** | | **60 chunks** |

Embeddings: `all-MiniLM-L6-v2` (384-dim) → FAISS `IndexFlatIP`.
Offline fallback: TF-IDF + TruncatedSVD (128-dim).

---

## 10. Layer 4 — LLM (Google Gemini)

### Model

`gemini-2.0-flash` — Google's fast, free-tier model via the OpenAI-compatible endpoint.

**Free tier:** 1,500 requests/day, no credit card required.
**Get your key:** https://aistudio.google.com

### Why Gemini instead of Claude

The project was originally designed for Claude (Anthropic), but Anthropic has no free API
tier. Google Gemini offers an equivalent capability on a generous free tier, making it
accessible for students and researchers without a paid account.

The switch uses Google's OpenAI-compatible endpoint so the message format (`system` +
`user` messages) is identical — minimal code change.

### System prompt requirements (unchanged)

1. Role: decision-support prototype, not diagnostic tool
2. Citation: only use retrieved document content
3. Diagnostic prohibition: never diagnose
4. Tone: calm, empathetic, evidence-based
5. Length: under 220 words

---

## 11. Layer 5 — Safety Filter

`src/safety.py` is a deterministic post-processing step applied to every LLM response.
Diagnostic language is stripped via regex before the response reaches the UI — this is
guaranteed to fire regardless of what the LLM outputs.

Two disclaimer tiers: standard (all responses) and high-risk (probability ≥ 0.65).

---

## 12. Pipeline Orchestration

`src/pipeline.py` is the single entry point the UI calls. It executes 8 steps:
predict → explain → build retrieval query → retrieve chunks → inject safety chunks →
assemble prompt → call LLM → safety filter → return dict.

---

## 13. Model Performance & Evaluation

### XGBoost on WDBC (holdout test set, n=114)

| Metric | Value |
|---|---|
| Accuracy | 96.49% |
| Precision | 100.00% |
| Recall (Sensitivity) | 90.48% |
| F1 Score | 95.00% |
| ROC-AUC | 0.9924 |
| Specificity | 100.00% |

### LightGBM on Mammographic Masses (holdout test set)

| Metric | Value |
|---|---|
| Accuracy | 84.30% |
| ROC-AUC | 0.9125 |
| Recall | 79.75% |

---

## 14. Project Structure

```
breastcare_ai/
│
├── CLAUDE.md                          ← This file
├── README.md
├── requirements.txt
├── .env.example                       ← GEMINI_API_KEY placeholder
├── .streamlit/config.toml
│
├── data/
│   ├── raw/mammographic_masses.data
│   └── processed/cleaned.csv
│
├── knowledge_base/
│   ├── sources/                       ← 4 curated Markdown documents
│   ├── chunks/chunks.jsonl            ← 60 paragraph-level chunks
│   └── vectorstore/                   ← FAISS index + TF-IDF fallback
│
├── models/
│   ├── mammographic_lgbm.pkl          ← LightGBM — UI-facing clinical tool
│   ├── mammographic_shap.pkl
│   ├── wdbc_xgboost.pkl
│   └── wdbc_shap.pkl
│
├── src/
│   ├── config.py                      ← All paths, thresholds, feature mappings
│   ├── pipeline.py                    ← Single orchestrator entry point
│   ├── safety.py                      ← Post-processing filter + disclaimer
│   ├── ml/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── explain.py
│   ├── rag/
│   │   ├── chunker.py
│   │   ├── build_vectorstore.py
│   │   ├── retriever.py
│   │   └── prompt_builder.py
│   └── llm/
│       └── client.py                  ← Google Gemini via OpenAI-compatible API
│
├── app/
│   └── streamlit_app.py
│
└── scripts/
    └── build_knowledge_base.py
```

---

## 15. Technology Stack

| Layer | Technology | Version | Why |
|---|---|---|---|
| Primary ML | XGBoost | 3.2.0 | Best AUC on WDBC; SHAP-native |
| Clinical tool | LightGBM | 4.6.0 | Fast, accurate, SHAP-native |
| Explainability | SHAP | 0.52.0 | TreeExplainer exact values |
| Vector search | FAISS | 1.7.4 | Fast cosine similarity; no server |
| Embeddings | sentence-transformers | 5.5.1 | all-MiniLM-L6-v2 |
| LLM | Google Gemini 2.0 Flash | gemini-2.0-flash | Free tier, 1,500 req/day |
| LLM SDK | openai | ≥1.0.0 | Google's OpenAI-compatible endpoint |
| Web framework | Streamlit | 1.58.0 | Rapid ML app development |
| Visualisation | Plotly | 6.7.0 | Interactive charts |
| Config | python-dotenv | 1.0.0 | API key from .env |

---

## 16. Key Design Decisions

1. **Two models** — XGBoost/WDBC for performance demonstration; LightGBM/Mammographic for the actual UI tool.
2. **SHAP drives RAG query** — retrieved documents are always aligned with what drove the prediction.
3. **Small curated knowledge base** — 60 chunks beats thousands of irrelevant chunks.
4. **Safety filter as code** — guaranteed to fire regardless of LLM behaviour.
5. **Row deletion over imputation** — ordinal integers cannot be meaningfully imputed.
6. **Gemini over Claude** — free tier with equivalent capability for student use.

---

## 17. Limitations

1. Small datasets (569 and 860 cases) — wide confidence intervals on all metrics.
2. Four features only — real systems use 20+ descriptors.
3. Image model AUC 0.62 — EfficientNet-B0 on CBIS-DDSM (3,568 images); clinical-grade models train on 100k+ cases.
4. No external validation — single-institution datasets.
5. Fixed decision threshold of 0.50.
6. TF-IDF fallback has lower retrieval quality than dense embeddings.
7. Not validated in clinical trials — research prototype only.

---

## 18. How to Run

### Prerequisites

- Python 3.10+
- A virtual environment (e.g., `farwa_env`)
- Google Gemini API key (free from https://aistudio.google.com)

### Setup sequence

```powershell
# 1. Activate environment
& "path\to\farwa_env\Scripts\Activate.ps1"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add API key — edit .env:
#    GEMINI_API_KEY=your-key-here

# 4. Train models (~60 seconds)
python -m src.ml.train

# 5. Build knowledge base (~10 seconds)
python scripts/build_knowledge_base.py

# 6. Launch app
streamlit run app/streamlit_app.py
# Opens at http://localhost:8501
```

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes (for AI explanation panel) | Free key from https://aistudio.google.com |

Without the key, ML prediction and SHAP explanation still work fully.
Only the AI explanation panel requires the key.

---

## 19. References

1. Elter, M. (2007). Mammographic Mass Dataset. UCI ML Repository.
2. Street, W.N. et al. (1993). Nuclear feature extraction for breast tumor diagnosis. SPIE.
3. Ghasemi, F. et al. (2024). Explainable AI in breast cancer detection. *Cancer Innovation.*
4. Kim, H.E. et al. (2022). Changes in cancer detection using AI. *THE LANCET Digital Health.*
5. Dembrower, K. et al. (2020). Effect of AI-based triaging of mammograms. *THE LANCET Digital Health.*
6. ACR BI-RADS Atlas, 5th Edition (2013). American College of Radiology.
7. Chen, T. & Guestrin, C. (2016). XGBoost. *KDD 2016.*
8. Ke, G. et al. (2017). LightGBM. *NeurIPS 2017.*
9. Lundberg, S. & Lee, S.I. (2017). SHAP. *NeurIPS 2017.*

---

*Last updated: June 2026*
*Project: BreastCare AI — Multi-layer breast cancer risk estimation prototype*
*Stack: XGBoost · LightGBM · SHAP · FAISS · Google Gemini · Streamlit*
