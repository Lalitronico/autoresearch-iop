"""End-to-end tests using synthetic data.

Verifies the full pipeline: prepare -> spec -> run_experiment -> log.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.data_loader import DataRegistry, PROCESSED_DIR, CODEBOOK_PATH
from core.specification import ExperimentSpec
from core.types import (
    Circumstance,
    DecompositionType,
    EstimationMethod,
    ExperimentStatus,
    InequalityMeasure,
    IncomeVariable,
    SampleFilter,
)
from orchestration.experiment_log import JSONL_PATH, TSV_PATH
from prepare import create_synthetic_data, generate_codebook
from run_experiment import run_single_experiment, run_batch, ExperimentResult


@pytest.fixture(scope="module")
def synthetic_data():
    """Create and save synthetic data for tests."""
    df = create_synthetic_data(n=500, seed=42)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output = PROCESSED_DIR / "emovi_analytical.parquet"
    df.to_parquet(output, index=False)
    codebook = generate_codebook(df)
    with open(CODEBOOK_PATH, "w") as f:
        json.dump(codebook, f, indent=2, default=str)
    yield df
    # Cleanup is optional -- data stays for other tests


@pytest.fixture(autouse=True)
def clean_logs():
    """Clean experiment logs before each test."""
    for path in [JSONL_PATH, TSV_PATH]:
        if path.exists():
            path.unlink()
    yield
    for path in [JSONL_PATH, TSV_PATH]:
        if path.exists():
            path.unlink()


class TestEndToEnd:
    def test_parametric_ols(self, synthetic_data):
        """Full pipeline: OLS parametric with Gini."""
        spec = ExperimentSpec(
            circumstances=(
                Circumstance.FATHER_EDUCATION.value,
                Circumstance.MOTHER_EDUCATION.value,
            ),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.OLS.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
            bootstrap_n=10,  # Fast for testing
            seed=42,
        )
        registry = DataRegistry()
        result = run_single_experiment(spec, registry)

        assert result.status == ExperimentStatus.SUCCESS.value
        assert 0 < result.iop_share < 1
        assert result.ci_lower <= result.iop_share <= result.ci_upper
        assert result.n_obs > 0
        assert result.total_inequality > 0

    def test_nonparametric_tree(self, synthetic_data):
        """Full pipeline: decision tree with MLD ex-ante."""
        spec = ExperimentSpec(
            circumstances=(
                Circumstance.FATHER_EDUCATION.value,
                Circumstance.ETHNICITY.value,
            ),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.MLD.value,
            method=EstimationMethod.DECISION_TREE.value,
            decomposition_type=DecompositionType.EX_ANTE.value,
            bootstrap_n=10,
            seed=42,
        )
        registry = DataRegistry()
        result = run_single_experiment(spec, registry)

        assert result.status == ExperimentStatus.SUCCESS.value
        assert 0 <= result.iop_share <= 1
        assert result.n_types is not None
        assert result.n_types >= 2

    def test_invalid_spec_logged(self, synthetic_data):
        """Invalid spec should be logged with INVALID_SPEC status."""
        spec = ExperimentSpec(
            circumstances=(),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.OLS.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
        )
        result = run_single_experiment(spec)
        assert result.status == ExperimentStatus.INVALID_SPEC.value

    def test_batch_execution(self, synthetic_data):
        """Batch of 3 specs should all execute."""
        specs = [
            ExperimentSpec(
                circumstances=(Circumstance.FATHER_EDUCATION.value,),
                income_variable=IncomeVariable.HH_PC_IMPUTED.value,
                inequality_measure=m,
                method=EstimationMethod.OLS.value,
                decomposition_type=DecompositionType.LOWER_BOUND.value,
                bootstrap_n=5,
            )
            for m in [InequalityMeasure.GINI.value, InequalityMeasure.MLD.value, InequalityMeasure.THEIL_T.value]
        ]
        registry = DataRegistry()
        results = run_batch(specs, registry)

        assert len(results) == 3
        assert all(r.status == ExperimentStatus.SUCCESS.value for r in results)

        # Verify log file has 3 entries
        assert JSONL_PATH.exists()
        with open(JSONL_PATH) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 3

    def test_reproducibility(self, synthetic_data):
        """Same spec + seed should give identical results."""
        spec = ExperimentSpec(
            circumstances=(Circumstance.FATHER_EDUCATION.value,),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.OLS.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
            bootstrap_n=20,
            seed=123,
        )
        registry = DataRegistry()
        r1 = run_single_experiment(spec, registry)

        # Clean logs between runs
        for p in [JSONL_PATH, TSV_PATH]:
            if p.exists():
                p.unlink()

        r2 = run_single_experiment(spec, registry)

        assert r1.iop_share == pytest.approx(r2.iop_share, abs=1e-10)
        assert r1.ci_lower == pytest.approx(r2.ci_lower, abs=1e-10)

    def test_sample_filter(self, synthetic_data):
        """Sample filters should reduce n_obs."""
        spec_all = ExperimentSpec(
            circumstances=(Circumstance.FATHER_EDUCATION.value,),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.OLS.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
            sample_filter=SampleFilter.ALL.value,
            bootstrap_n=5,
        )
        spec_male = ExperimentSpec(
            circumstances=(Circumstance.FATHER_EDUCATION.value,),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.OLS.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
            sample_filter=SampleFilter.MALE.value,
            bootstrap_n=5,
        )
        registry = DataRegistry()
        r_all = run_single_experiment(spec_all, registry)

        for p in [JSONL_PATH, TSV_PATH]:
            if p.exists():
                p.unlink()

        r_male = run_single_experiment(spec_male, registry)

        assert r_all.n_obs > r_male.n_obs
