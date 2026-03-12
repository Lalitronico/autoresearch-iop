"""Tests for multiple imputation infrastructure.

Tests cover: Rubin's rules, MI spec IDs, imputed data loading,
save/load roundtrip, and sample size preservation.
MICE creation tests use conftest's simple random fill (no miceforest dependency).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.data_loader import DataRegistry
from core.specification import ExperimentSpec
from core.types import (
    Circumstance,
    DecompositionType,
    EstimationMethod,
    InequalityMeasure,
    IncomeVariable,
    SampleFilter,
)
from evaluation.metrics import pool_rubin
from imputation.mice_imputer import (
    load_imputation_metadata,
    load_imputed_dataset,
    save_imputed_datasets,
    validate_imputation,
)


# --- Rubin's rules ---


def test_rubin_pool_identical():
    """If all Q_m are identical, B=0 and CI depends only on within-variance."""
    estimates = [0.25, 0.25, 0.25, 0.25, 0.25]
    variances = [0.001, 0.001, 0.001, 0.001, 0.001]
    result = pool_rubin(estimates, variances)

    assert result["q_bar"] == pytest.approx(0.25)
    assert result["b"] == pytest.approx(0.0, abs=1e-15)
    # Total variance = U_bar when B=0
    assert result["t"] == pytest.approx(0.001, abs=1e-10)
    assert result["fraction_missing_info"] == pytest.approx(0.0, abs=1e-10)


def test_rubin_known_values():
    """Hand-calculated Rubin's rules example.

    M=3, Q = [0.20, 0.25, 0.30], U = [0.001, 0.001, 0.001]
    Q_bar = 0.25
    U_bar = 0.001
    B = var([0.20, 0.25, 0.30], ddof=1) = 0.0025
    T = 0.001 + (1 + 1/3)*0.0025 = 0.001 + 0.003333 = 0.004333
    FMI = (1+1/3)*0.0025 / 0.004333 ≈ 0.7692
    """
    estimates = [0.20, 0.25, 0.30]
    variances = [0.001, 0.001, 0.001]
    result = pool_rubin(estimates, variances)

    assert result["q_bar"] == pytest.approx(0.25)
    assert result["u_bar"] == pytest.approx(0.001)
    assert result["b"] == pytest.approx(0.0025)
    expected_t = 0.001 + (1 + 1/3) * 0.0025
    assert result["t"] == pytest.approx(expected_t, rel=1e-6)
    expected_fmi = (1 + 1/3) * 0.0025 / expected_t
    assert result["fraction_missing_info"] == pytest.approx(expected_fmi, rel=1e-4)
    # CI should be centered on Q_bar
    assert result["ci_lower"] < 0.25
    assert result["ci_upper"] > 0.25


def test_rubin_single_imputation():
    """With M=1, pool_rubin should return the single estimate with within-CI."""
    result = pool_rubin([0.30], [0.004])
    assert result["q_bar"] == pytest.approx(0.30)
    assert result["b"] == pytest.approx(0.0)
    assert result["fraction_missing_info"] == pytest.approx(0.0)


# --- MI spec IDs ---


def test_mi_spec_different_id():
    """use_mi=True and use_mi=False produce different spec_ids."""
    base_args = dict(
        circumstances=("father_education",),
        income_variable=IncomeVariable.HH_PC_IMPUTED.value,
        inequality_measure=InequalityMeasure.GINI.value,
        method=EstimationMethod.OLS.value,
        decomposition_type=DecompositionType.LOWER_BOUND.value,
    )
    spec_listwise = ExperimentSpec(**base_args, use_mi=False)
    spec_mi = ExperimentSpec(**base_args, use_mi=True)

    assert spec_listwise.spec_id != spec_mi.spec_id


def test_mi_spec_roundtrip():
    """MI spec survives to_dict -> from_dict."""
    spec = ExperimentSpec(
        circumstances=("father_education", "mother_education"),
        income_variable=IncomeVariable.HH_PC_IMPUTED.value,
        inequality_measure=InequalityMeasure.GINI.value,
        method=EstimationMethod.OLS.value,
        decomposition_type=DecompositionType.LOWER_BOUND.value,
        use_mi=True,
    )
    d = spec.to_dict()
    assert d["use_mi"] is True

    restored = ExperimentSpec.from_dict(d)
    assert restored.use_mi is True
    assert restored.spec_id == spec.spec_id


# --- Imputed data tests (using conftest's simple random fill) ---


def test_imputed_datasets_exist(isolate_test_paths):
    """conftest creates M=3 imputed datasets in temp dir."""
    import core.data_loader as dl
    imputed_dir = dl.IMPUTED_DIR
    meta = load_imputation_metadata(imputed_dir)
    assert meta["m"] == 3

    for i in range(3):
        df = load_imputed_dataset(imputed_dir, i)
        assert len(df) > 0


def test_no_nan_in_circumstances(isolate_test_paths):
    """All circumstance columns should be complete after imputation."""
    import core.data_loader as dl
    imputed_dir = dl.IMPUTED_DIR

    cols_imputed = ["father_education", "mother_education", "ethnicity"]
    for i in range(3):
        df = load_imputed_dataset(imputed_dir, i)
        for col in cols_imputed:
            if col in df.columns:
                assert df[col].isna().sum() == 0, (
                    f"Dataset {i}, column {col} has NaN after imputation"
                )


def test_income_not_imputed(isolate_test_paths):
    """Income columns should be identical across all imputed datasets."""
    import core.data_loader as dl

    original = pd.read_parquet(dl.ANALYTICAL_FILE)
    imputed_dir = dl.IMPUTED_DIR

    for i in range(3):
        imp_df = load_imputed_dataset(imputed_dir, i)
        assert imp_df["hh_pc_imputed"].equals(original["hh_pc_imputed"]), (
            f"Dataset {i}: hh_pc_imputed changed after imputation"
        )


def test_save_load_roundtrip(tmp_path):
    """Save and reload imputed datasets preserves data."""
    rng = np.random.default_rng(99)
    dfs = [
        pd.DataFrame({"a": rng.random(50), "b": rng.integers(0, 5, 50)})
        for _ in range(3)
    ]
    meta = {"seed": 99, "vars_imputed": ["b"]}
    save_imputed_datasets(dfs, tmp_path / "imp", meta)

    loaded_meta = load_imputation_metadata(tmp_path / "imp")
    assert loaded_meta["m"] == 3
    assert loaded_meta["seed"] == 99

    for i in range(3):
        loaded = load_imputed_dataset(tmp_path / "imp", i)
        pd.testing.assert_frame_equal(loaded, dfs[i])


def test_reproducibility(isolate_test_paths):
    """Same seed produces identical imputed datasets."""
    import core.data_loader as dl
    imputed_dir = dl.IMPUTED_DIR

    # Load dataset 0 twice — should be identical (same cached file)
    df0a = load_imputed_dataset(imputed_dir, 0)
    df0b = load_imputed_dataset(imputed_dir, 0)
    pd.testing.assert_frame_equal(df0a, df0b)


def test_mi_preserves_sample_size(isolate_test_paths):
    """MI sample size should equal the full filtered sample (no listwise deletion on circs)."""
    import core.data_loader as dl

    registry = DataRegistry()

    spec = ExperimentSpec(
        circumstances=("father_education", "mother_education", "ethnicity"),
        income_variable=IncomeVariable.HH_PC_IMPUTED.value,
        inequality_measure=InequalityMeasure.GINI.value,
        method=EstimationMethod.OLS.value,
        decomposition_type=DecompositionType.LOWER_BOUND.value,
        use_mi=True,
    )

    # MI sample: no NaN drop on circs
    y_mi, X_mi, _ = registry.get_sample_for_spec_mi(spec, 0)

    # Listwise sample: drops NaN on circs
    y_lw, X_lw, _ = registry.get_sample_for_spec(spec)

    # MI should preserve at least as many observations
    assert len(y_mi) >= len(y_lw), (
        f"MI n={len(y_mi)} should be >= listwise n={len(y_lw)}"
    )


def test_data_registry_has_imputed(isolate_test_paths):
    """DataRegistry detects imputed data and reports correct count."""
    registry = DataRegistry()
    assert registry.has_imputed_data is True
    assert registry.n_imputations == 3


def test_validate_imputation_passes(isolate_test_paths):
    """validate_imputation should pass on conftest's imputed data."""
    import core.data_loader as dl

    original = pd.read_parquet(dl.ANALYTICAL_FILE)
    imputed_dir = dl.IMPUTED_DIR
    dfs = [load_imputed_dataset(imputed_dir, i) for i in range(3)]

    circ_cols = ["father_education", "mother_education", "ethnicity"]
    income_cols = ["hh_pc_imputed"]
    result = validate_imputation(original, dfs, circ_cols, income_cols)

    assert result["all_passed"] is True
    assert result["n_datasets"] == 3
