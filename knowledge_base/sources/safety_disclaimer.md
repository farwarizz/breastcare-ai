# Safety Disclaimer — BreastCare AI Scope, Limitations, and Patient Guidance

## What This Tool Is

BreastCare AI is a research and educational prototype. It estimates the probability that a breast mass is malignant from four values: the patient's age and three BI-RADS descriptors of the mass (shape, margin and density). The estimate comes from a statistical model trained on the UCI Mammographic Mass dataset, a public collection of 961 masses (831 with complete descriptors) from one German radiology institute between 2003 and 2006. The model was chosen by a fixed rule from a logistic regression and two LightGBM candidates; the app shows which model is currently served.

## What the Outputs Mean

The tool shows a calibrated probability, which estimates how often masses with the same four descriptors were malignant in the study population. It also shows a 90% prediction set: {Benign}, {Malignant}, or both. A set containing both labels means the model cannot separate the two outcomes for these inputs. The feature contributions (SHAP values) show which of the four inputs pushed the estimate up or down. None of these outputs is a diagnosis.

## What This Tool Is Not

This tool is not a medical device. It has not been cleared or approved by the U.S. Food and Drug Administration (FDA) or any other regulator, and it has not been validated in a clinical trial. It does not diagnose cancer, does not replace a radiologist's reading or a biopsy, and does not recommend treatment.

The image-analysis feature is a separate, experimental model whose accuracy has not been measured. It is not part of the risk estimate and may be switched off on a given server.

## Clinical Limitations

The model sees only four coarse descriptors, which radiologists assign by eye, so it inherits differences between readers. It was trained and tested on data from one institution and may not generalise to other screening programmes, equipment or populations. Its test set was small, so its accuracy figures have wide uncertainty intervals, and its performance differs between age groups. The model card and evaluation report in the project's documentation give the measured figures.

## High-Risk Estimate Notice

When the estimated malignancy probability is 65% or higher, the tool displays an enhanced notice. A high estimate does not mean a diagnosis of cancer. It means that masses with the same descriptors were often malignant in the training data. Only a biopsy, examined by a pathologist, can confirm or exclude malignancy.

## What You Should Do

If you have received an abnormal mammography result, or have concerns about a breast finding, speak to a qualified radiologist or physician. Do not use this tool's output to make, delay or skip any clinical decision, including a recommended biopsy or follow-up appointment.

If you are feeling overwhelmed or having thoughts of harming yourself, contact your local emergency number or a crisis line. In the US, call or text 988; in the UK and Ireland, call Samaritans on 116 123; elsewhere, findahelpline.com lists free helplines.

## Privacy

The tool does not ask for your name or contact details. Questions typed into the chat are sent to Google's Gemini API to generate an answer. The server can keep an audit log of assessments (the four values entered and the estimate) when a session is started through the API; the web interface does not start one. Please do not type identifying personal information into the chat.

## Contact and Research Context

BreastCare AI was developed as an academic research prototype demonstrating how gradient-boosted and linear risk models, calibration, conformal prediction, SHAP explanations, retrieval-augmented generation with Google Gemini, and safety guardrails can be combined and evaluated for educational purposes in the context of breast cancer screening research.

## Sources

This document describes the tool itself; its statements are checked against the repository.

- Model, data split and evaluation: `models/model_metadata.json`, `reports/evaluation.md` and `docs/MODEL_CARD.md` in this repository.
- Dataset: Elter M. Mammographic Mass [dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C53K6Z (provenance in `data/README.md`).
- Chat processing and the audit log: `src/pipeline.py`, `src/llm/client.py` and `server.py` in this repository.
- Crisis lines: 988 Suicide & Crisis Lifeline, https://988lifeline.org; Samaritans, https://www.samaritans.org; Find A Helpline, https://findahelpline.com
