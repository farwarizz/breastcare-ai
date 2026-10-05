"""The training data is the real UCI file, and nothing can silently replace it."""
import hashlib
import json

import pandas as pd
import pytest

from src import config
from src.ml import data
from tests.fixtures.synthetic_mammographic import (
    make_synthetic_mammographic_raw, to_raw_uci_bytes,
)


def test_raw_file_is_the_official_uci_file():
    raw = config.MAMMO_RAW_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == config.MAMMO_SHA256


def test_real_dataset_matches_uci_documentation():
    raw = data.read_raw_mammographic()
    assert len(raw) == 961
    assert raw.isna().sum().to_dict() == {
        "birads": 2, "age": 5, "shape": 31, "margin": 48, "density": 76, "severity": 0}
    assert raw["severity"].value_counts().to_dict() == {0: 516, 1: 445}


def test_cleaning_real_data():
    clean = data.clean_mammographic(data.read_raw_mammographic())
    assert list(clean.columns) == ["age", "shape", "margin", "density", "severity"]
    assert "birads" not in clean.columns            # BI-RADS stays excluded (leakage)
    assert len(clean) == 831
    assert clean["severity"].value_counts().to_dict() == {0: 428, 1: 403}
    assert set(clean["shape"]) <= set(config.SHAPE_MAP)
    assert set(clean["margin"]) <= set(config.MARGIN_MAP)
    assert set(clean["density"]) <= set(config.DENSITY_MAP)


def test_saved_processed_csv_is_the_cleaned_real_data():
    expected = data.clean_mammographic(data.read_raw_mammographic()).reset_index(drop=True)
    saved = pd.read_csv(config.CLEANED_CSV)
    pd.testing.assert_frame_equal(saved, expected, check_dtype=False)


def test_model_metadata_records_real_data_provenance():
    md = json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    mammo = md["datasets"]["mammographic"]
    assert mammo["raw_sha256"] == config.MAMMO_SHA256
    assert (mammo["raw_rows"], mammo["clean_rows"]) == (961, 831)
    assert mammo["features"] == config.FEATURE_NAMES


def test_missing_raw_file_raises_instead_of_generating_data(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "CLEANED_CSV", tmp_path / "cleaned.csv")
    with pytest.raises(FileNotFoundError):
        data.load_mammographic(tmp_path / "absent.data")
    assert not (tmp_path / "cleaned.csv").exists()


def test_synthetic_file_is_rejected_by_training(tmp_path, monkeypatch):
    fake = tmp_path / "mammographic_masses.data"
    fake.write_bytes(to_raw_uci_bytes(make_synthetic_mammographic_raw()))
    monkeypatch.setattr(data, "CLEANED_CSV", tmp_path / "cleaned.csv")
    with pytest.raises(ValueError, match="not the official UCI"):
        data.load_mammographic(fake)
    assert not (tmp_path / "cleaned.csv").exists()


def test_cleaning_logic_on_synthetic_fixture():
    raw = make_synthetic_mammographic_raw()
    clean = data.clean_mammographic(raw)
    assert "birads" not in clean.columns
    assert not clean.isna().any().any()
    assert len(clean) == len(raw.drop(columns="birads").dropna())


def test_cleaning_rejects_undocumented_codes():
    raw = make_synthetic_mammographic_raw()
    raw.loc[raw["shape"].first_valid_index(), "shape"] = 7
    with pytest.raises(ValueError, match="shape"):
        data.clean_mammographic(raw)


def test_no_production_code_can_generate_synthetic_data():
    """No fixture imports, no 'synthetic' fallbacks, no RNG in data loading or analytics."""
    root = config.ROOT
    production = [root / "server.py", *(root / "src").rglob("*.py"), *(root / "scripts").glob("*.py")]
    for path in production:
        text = path.read_text(encoding="utf-8")
        assert "synthetic_mammographic" not in text, f"{path} imports the test fixture"
        assert "synthetic" not in text.lower(), f"{path} mentions a synthetic fallback"
    for rel in ("src/ml/data.py", "src/ml/train.py", "src/analytics/global_shap.py",
                "src/analytics/batch_predict.py"):
        text = (root / rel).read_text(encoding="utf-8")
        assert "default_rng" not in text and "RandomState" not in text and "randint" not in text, rel


def test_global_shap_reports_unavailable_without_data(tmp_path, monkeypatch):
    from src.analytics import global_shap
    monkeypatch.setattr(global_shap, "CLEANED_CSV", tmp_path / "missing.csv")
    out = global_shap.compute_global_shap()
    assert out["available"] is False      # never falls back to generated samples
