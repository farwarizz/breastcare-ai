# Data provenance

## UCI Mammographic Mass: the dataset the app's model is trained on

| | |
|---|---|
| Source | UCI Machine Learning Repository, dataset id 161: <https://archive.ics.uci.edu/dataset/161/mammographic+mass> |
| DOI | <https://doi.org/10.24432/C53K6Z> |
| License | CC BY 4.0 |
| Creator | Matthias Elter |
| Collection | Full-field digital mammograms, Institute of Radiology, University Erlangen-Nuremberg, 2003–2006 (per the UCI description) |
| Raw file | `data/raw/mammographic_masses.data`: the original UCI file, unmodified |
| Download URL | <https://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data> |
| SHA-256 | `072051c896fb14f8f25c64eed30ba1333290af41e6c9ffad4e1ca573daa865a8` |
| Retrieved | 2026-10-05 |
| Dataset citation | Elter, M. (2007). Mammographic Mass [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C53K6Z |
| Paper | Elter, M., Schulz-Wendtland, R., & Wittenberg, T. (2007). The prediction of breast cancer biopsy outcomes using two CAD approaches that both emphasize an intelligible decision process. *Medical Physics*. |

### How the file is verified

`python scripts/download_data.py` fetches the dataset through `ucimlrepo`
(`fetch_ucirepo(id=161)`, UCI's API) and accepts the raw file only if all of
these hold:

- its SHA-256 equals the pinned value above;
- its values and missing-value positions are identical, cell for cell, to the `ucimlrepo` data;
- it has 961 rows, the documented missing counts per column, and the documented class counts (516 benign / 445 malignant).

On any failure it exits with an error and writes nothing. Training
(`python -m src.ml.train`) re-checks the SHA-256 and refuses to run on any
other file. Production code has no synthetic-data fallback. A synthetic
generator exists only as a test fixture (`tests/fixtures/synthetic_mammographic.py`).

The file uses CRLF line endings. `.gitattributes` (`data/raw/** -text`) stops
git from converting them, so the checksum holds on every OS.

### Format

Headerless CSV with 6 columns. `?` marks a missing value (162 cells in total).

| Column | Values (UCI documentation) | Missing | Use |
|---|---|---|---|
| BI-RADS assessment | 1–5, ordinal. The file also contains 0 (5 rows), 6 (11 rows) and one 55, and no 1s | 2 | **Excluded.** It is the radiologist's assessment, so it would leak the label |
| Age | years (integer) | 5 | Feature |
| Shape | round=1, oval=2, lobular=3, irregular=4 (nominal) | 31 | Feature |
| Margin | circumscribed=1, microlobulated=2, obscured=3, ill-defined=4, spiculated=5 (nominal) | 48 | Feature |
| Density | high=1, iso=2, low=3, fat-containing=4 (ordinal) | 76 | Feature |
| Severity | benign=0, malignant=1 | 0 | **Target** |

### Cleaning (`clean_mammographic` in `src/ml/train.py`)

1. Drop the BI-RADS column.
2. Drop every row with a missing age, shape, margin or density (130 rows).
3. Check every value against the codes above, and raise an error on anything unexpected.

Result (`data/processed/cleaned.csv`): **831 rows, 428 benign and 403 malignant**.
Features: `age, shape, margin, density`. Target: `severity` (0 = benign, 1 = malignant).

Notes for modelling:

- Density carries little signal: 798 of the 885 masses with a recorded density are "low" (3).
- Shape and margin are nominal codes but are currently fed to the model as numbers.
- The features are coarse, so many different patients share an identical row (344 of the 831 cleaned rows repeat an earlier row). They are distinct cases, so they are kept.

## WDBC: research benchmark only

Loaded from `sklearn.datasets.load_breast_cancer`, a copy of the UCI Wisconsin
Diagnostic Breast Cancer dataset: 569 rows, 30 features, 212 malignant and 357
benign. It trains the XGBoost benchmark and the 5-model CV comparison. The app
does not use it.
