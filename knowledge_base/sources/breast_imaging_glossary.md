# Breast Imaging Glossary — Patient and Clinician Reference

## What is a Mammographic Mass?

A mass is a three-dimensional finding with an outward-bulging contour, seen on two different mammographic projections (or on a single projection with tomosynthesis). Masses are described by their shape, margin and density. A finding seen on only one projection is called an asymmetry rather than a mass.

## What is a Mammogram?

A mammogram is an X-ray of the breast and the primary tool for breast cancer screening. Digital mammography records the X-rays electronically, so images can be enhanced and stored. Digital breast tomosynthesis (3D mammography) takes images from several angles and reconstructs thin slices through the breast. It improves cancer detection in many women with dense breasts, although a large US study found no gain at later screening rounds in extremely dense breasts.

## What is BI-RADS?

BI-RADS (Breast Imaging Reporting and Data System) is the American College of Radiology's standardised lexicon, report structure, assessment-category scheme and audit system for mammography, ultrasound, MRI and contrast-enhanced mammography. Its final assessment categories are:
- Category 0: Incomplete — additional imaging and/or comparison with prior images is needed
- Category 1: Negative — routine screening
- Category 2: Benign — routine screening
- Category 3: Probably benign — short-interval follow-up, usually at 6 months
- Category 4: Suspicious — tissue diagnosis (subdivided 4A, 4B, 4C)
- Category 5: Highly suggestive of malignancy — tissue diagnosis
- Category 6: Known biopsy-proven malignancy

## What is a Fibroadenoma?

A fibroadenoma is the most common benign breast tumour, typically seen as a well-defined, oval or round, circumscribed mass in young women. Coarse calcifications are typical of older, involuting fibroadenomas. Removal is considered when a fibroadenoma is large, growing, causing symptoms, or looks atypical on imaging.

## What is a Cyst?

A breast cyst is a fluid-filled sac. On ultrasound, a simple cyst is circumscribed, round or oval, free of internal echoes (anechoic) and thin-walled, often with posterior acoustic enhancement; it is benign. Complicated cysts are also usually benign and may be aspirated if they cause symptoms. Complex cystic and solid lesions should be biopsied.

## What is Fine Needle Aspiration (FNA)?

Fine needle aspiration is a minimally invasive procedure in which a thin needle draws cells from a mass for cytological examination. The Wisconsin Diagnostic Breast Cancer (WDBC) dataset's features were computed from digitised FNA images: 10 nuclear measurements, each summarised by its mean, standard error and worst value, giving 30 features.

## What is a Biopsy?

A breast biopsy removes a tissue sample from an abnormality for examination by a pathologist. An image-guided core needle biopsy is the preferred first procedure for a suspicious finding. A biopsy can confirm cancer; a benign result is reliable only when it agrees with the imaging findings (concordance). An AI risk estimate is not a substitute for a biopsy.

## What are Nuclear Features?

Nuclear features describe the cell nuclei seen under the microscope, such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension. The WDBC dataset contains 30 such features.

## What is Malignancy Probability?

A malignancy probability is a model's estimate of how often masses with the same features turned out to be cancer, expressed as a percentage. It reflects the cases the model learned from: in the UCI Mammographic Mass data, 445 of 961 masses were malignant, a far higher share than among masses found at routine screening. A probability is a statistical estimate, not a diagnosis.

## What is Sensitivity and Specificity?

Sensitivity is the share of people with the disease whom a test correctly flags: true positives divided by all people with the disease. Specificity is the share of people without the disease whom the test correctly clears: true negatives divided by all people without it. Screening programmes trade one against the other, and most women recalled after a screening mammogram do not have cancer.

## Sources

- American College of Radiology. BI-RADS v2025 Mammography Lexicon Summary Form. https://edge.sitecorecloud.io/americancoldf5f-acrorgf92a-productioncb02-3650/media/ACR/Files/RADS/BI-RADS/BI-RADS-Summary-Form-Mammography.pdf
- American College of Radiology. BI-RADS v2025 Ultrasound Lexicon Summary Form. https://edge.sitecorecloud.io/americancoldf5f-acrorgf92a-productioncb02-3650/media/ACR/Files/RADS/BI-RADS/BI-RADS-Summary-Form-Ultrasound.pdf
- National Cancer Institute. Breast Cancer Screening (PDQ), health professional version. https://www.cancer.gov/types/breast/hp/breast-screening-pdq
- National Cancer Institute. Mammograms fact sheet. https://www.cancer.gov/types/breast/mammograms-fact-sheet
- Lowry KP, et al. Screening performance of digital breast tomosynthesis vs digital mammography by breast density and age. JAMA Network Open. 2020.
- American Cancer Society. Fibroadenomas of the breast. https://www.cancer.org/cancer/types/breast-cancer/non-cancerous-breast-conditions/fibroadenomas-of-the-breast.html
- Berg WA, et al. Cystic lesions of the breast: sonographic-pathologic correlation. Radiology. 2003.
- UCI Machine Learning Repository. Breast Cancer Wisconsin (Diagnostic). https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- Elter M. Mammographic Mass [dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C53K6Z
