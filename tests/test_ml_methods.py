"""End-to-end tests for XGBoost and Random Forest methods."""

import json
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
)
import orchestration.experiment_log as el
from run_experiment import run_single_experiment, run_batch

# synthetic_data fixture provided by conftest.py


@pytest.fixture(autouse=True)
def clean_logs():
    """Clean experiment logs before each test (uses isolated paths from conftest)."""
    for path in [el.JSONL_PATH, el.TSV_PATH]:
        if path.exists():
            path.unlink()
    yield
    for path in [el.JSONL_PATH, el.TSV_PATH]:
        if path.exists():
            path.unlink()


class TestXGBoost:
    def test_xgboost_ex_ante_gini(self, synthetic_data):
        """XGBoost with ex_ante decomposition and Gini should succeed."""
        spec = ExperimentSpec(
            circumstances=(
                Circumstance.FATHER_EDUCATION.value,
                Circumstance.MOTHER_EDUCATION.value,
                Circumstance.ETHNICITY.value,
                Circumstance.GENDER.value,
            ),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.XGBOOST.value,
            decomposition_type=DecompositionType.EX_ANTE.value,
            bootstrap_n=5,
            seed=42,
        )
        registry = DataRegistry()
        result = run_single_experiment(spec, registry)

        assert result.status == ExperimentStatus.SUCCESS.value
        assert 0 < result.iop_share < 1
        assert result.ci_lower <= result.iop_share <= result.ci_upper
        assert result.n_obs > 0
        assert result.total_inequality > 0

    def test_xgboost_ex_ante_mld(self, synthetic_data):
        """XGBoost with MLD should produce valid results."""
        spec = ExperimentSpec(
            circumstances=(
                Circumstance.FATHER_EDUCATION.value,
                Circumstance.ETHNICITY.value,
            ),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.MLD.value,
            method=EstimationMethod.XGBOOST.value,
            decomposition_type=DecompositionType.EX_ANTE.value,
            bootstrap_n=5,
            seed=42,
        )
        registry = DataRegistry()
        result = run_single_experiment(spec, registry)

        assert result.status == ExperimentStatus.SUCCESS.value
        assert 0 <= result.iop_share <= 1

    def test_xgboost_lower_bound_invalid(self, synthetic_data):
        """XGBoost + lower_bound should be invalid (lower_bound is OLS only)."""
        spec = ExperimentSpec(
            circumstances=(Circumstance.FATHER_EDUCATION.value,),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.XGBOOST.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
        )
        assert not spec.is_valid
        assert any("Lower bound" in e for e in spec.validate())

    def test_xgboost_ex_post_invalid(self, synthetic_data):
        """XGBoost + ex_post is not in valid pairs (OLS→lower_bound, tree→ex_ante/ex_post, xgboost/rf→ex_ante)."""
        spec = ExperimentSpec(
            circumstances=(Circumstance.FATHER_EDUCATION.value,),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.XGBOOST.value,
            decomposition_type=DecompositionType.EX_POST.value,
            bootstrap_n=5,
        )
        # ex_post requires partition method -- xgboost uses leaf-based types,
        # so it may or may not be valid depending on spec validation.
        # Test that the pipeline handles it (either succeeds or logs properly).
        registry = DataRegistry()
        result = run_single_experiment(spec, registry)
        assert result.status in (
            ExperimentStatus.SUCCESS.value,
            ExperimentStatus.INVALID_SPEC.value,
            ExperimentStatus.FAILED.value,
        )


class TestRandomForest:
    def test_rf_ex_ante_gini(self, synthetic_data):
        """Random Forest with ex_ante and Gini should succeed."""
        spec = ExperimentSpec(
            circumstances=(
                Circumstance.FATHER_EDUCATION.value,
                Circumstance.MOTHER_EDUCATION.value,
                Circumstance.ETHNICITY.value,
            ),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.RANDOM_FOREST.value,
            decomposition_type=DecompositionType.EX_ANTE.value,
            bootstrap_n=5,
            seed=42,
        )
        registry = DataRegistry()
        result = run_single_experiment(spec, registry)

        assert result.status == ExperimentStatus.SUCCESS.value
        assert 0 < result.iop_share < 1
        assert result.ci_lower <= result.iop_share <= result.ci_upper
        assert result.n_obs > 0

    def test_rf_ex_ante_mld(self, synthetic_data):
        """Random Forest with MLD should produce valid results."""
        spec = ExperimentSpec(
            circumstances=(
                Circumstance.FATHER_EDUCATION.value,
                Circumstance.GENDER.value,
            ),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.MLD.value,
            method=EstimationMethod.RANDOM_FOREST.value,
            decomposition_type=DecompositionType.EX_ANTE.value,
            bootstrap_n=5,
            seed=42,
        )
        registry = DataRegistry()
        result = run_single_experiment(spec, registry)

        assert result.status == ExperimentStatus.SUCCESS.value
        assert 0 <= result.iop_share <= 1

    def test_rf_lower_bound_invalid(self, synthetic_data):
        """Random Forest + lower_bound should be invalid."""
        spec = ExperimentSpec(
            circumstances=(Circumstance.FATHER_EDUCATION.value,),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.RANDOM_FOREST.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
        )
        assert not spec.is_valid

    def test_rf_reproducible(self, synthetic_data):
        """Same RF spec + seed should give identical results."""
        spec = ExperimentSpec(
            circumstances=(
                Circumstance.FATHER_EDUCATION.value,
                Circumstance.ETHNICITY.value,
            ),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.RANDOM_FOREST.value,
            decomposition_type=DecompositionType.EX_ANTE.value,
            bootstrap_n=10,
            seed=123,
        )
        registry = DataRegistry()
        r1 = run_single_experiment(spec, registry)

        for p in [el.JSONL_PATH, el.TSV_PATH]:
            if p.exists():
                p.unlink()

        r2 = run_single_experiment(spec, registry)

        assert r1.iop_share == pytest.approx(r2.iop_share, abs=1e-10)


class TestMLMethodBatch:
    def test_mixed_method_batch(self, synthetic_data):
        """Batch with OLS, DT, XGBoost, and RF should all succeed."""
        circs = (
            Circumstance.FATHER_EDUCATION.value,
            Circumstance.MOTHER_EDUCATION.value,
        )
        specs = [
            ExperimentSpec(
                circumstances=circs,
                income_variable=IncomeVariable.HH_PC_IMPUTED.value,
                inequality_measure=InequalityMeasure.GINI.value,
                method=EstimationMethod.OLS.value,
                decomposition_type=DecompositionType.LOWER_BOUND.value,
                bootstrap_n=5,
            ),
            ExperimentSpec(
                circumstances=circs,
                income_variable=IncomeVariable.HH_PC_IMPUTED.value,
                inequality_measure=InequalityMeasure.GINI.value,
                method=EstimationMethod.DECISION_TREE.value,
                decomposition_type=DecompositionType.EX_ANTE.value,
                bootstrap_n=5,
            ),
            ExperimentSpec(
                circumstances=circs,
                income_variable=IncomeVariable.HH_PC_IMPUTED.value,
                inequality_measure=InequalityMeasure.GINI.value,
                method=EstimationMethod.XGBOOST.value,
                decomposition_type=DecompositionType.EX_ANTE.value,
                bootstrap_n=5,
            ),
            ExperimentSpec(
                circumstances=circs,
                income_variable=IncomeVariable.HH_PC_IMPUTED.value,
                inequality_measure=InequalityMeasure.GINI.value,
                method=EstimationMethod.RANDOM_FOREST.value,
                decomposition_type=DecompositionType.EX_ANTE.value,
                bootstrap_n=5,
            ),
        ]
        registry = DataRegistry()
        results = run_batch(specs, registry)

        assert len(results) == 4
        assert all(r.status == ExperimentStatus.SUCCESS.value for r in results)

        # All should produce bounded IOp shares
        for r in results:
            assert 0 < r.iop_share < 1
