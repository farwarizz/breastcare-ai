# Safety Disclaimer — BreastCare AI Scope, Limitations, and Patient Guidance

## What This Tool Is

BreastCare AI is a research and educational prototype that estimates malignancy probability from four structured mammographic features: patient age, mass shape, mass margin, and mass density. It uses a LightGBM model trained on the UCI Mammographic Masses dataset (Elter, 2007) and produces SHAP-based feature attribution to explain its estimates.

## What This Tool Is Not

This tool is not a medical device. It has not been cleared or approved by the U.S. Food and Drug Administration (FDA) or any other regulatory authority. It is not intended for clinical diagnosis, treatment planning, or as a substitute for radiological interpretation.

This tool does not process mammogram images. It uses structured numeric features only. Its accuracy is limited by the small size of its training data (860 cases) and the sparsity of its feature set (4 features). Production clinical AI systems use 20 or more descriptors, calcification patterns, and full mammographic image analysis.

## Clinical Limitations

The model was trained on data from a single institution and may not generalise to other screening programmes, imaging equipment, or patient populations. It has not been validated in a prospective clinical trial. Its performance in real-world screening settings is unknown.

The model returns a point estimate without calibrated confidence intervals. The fixed decision threshold of 0.50 is not optimised for any clinical cost-benefit ratio.

## High-Risk Estimate Notice

When the estimated malignancy probability is 65% or higher, this tool displays an enhanced notice. A high probability estimate does not mean a diagnosis of cancer. It means the features you entered are statistically associated with a higher likelihood of malignancy in the training data. Only biopsy and pathological examination can confirm or exclude malignancy.

## What You Should Do

If you have received an abnormal mammography result, or if you have concerns about a breast finding, you should consult a qualified radiologist or physician. Do not use this tool's output to make or delay clinical decisions.

All estimates produced by this tool must be reviewed by a qualified clinician before any clinical action is taken.

## Privacy

This tool does not collect, store, or transmit personal health information. The feature values entered are used solely to compute and display a risk estimate within your session. No data is retained after the session ends.

## Contact and Research Context

BreastCare AI was developed as an academic research prototype demonstrating the integration of gradient boosted tree models, SHAP explainability, FAISS retrieval-augmented generation, and Google Gemini for educational purposes in the context of breast cancer screening research.
