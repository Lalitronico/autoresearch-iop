"""Integration tests for IOp decomposition pipeline."""

import numpy as np
import pandas as pd
import pytest

from core.decomposition import decompose_iop, IOpResult
from core.inequality_measures import compute_inequality
from core.specification import ExperimentSpec
from core.types import (
    Circumstance,
    DecompositionType,
    EstimationMethod,
    InequalityMeasure,
    IncomeVariable,
    SampleFilter,
)


class TestDecomposition:
    def _make_simple_data(self, n=500, seed=42):
        """Create simple test data with known structure."""
        rng = np.random.default_rng(seed)
        # Two types: high-income and low-income
        types = rng.choice([0, 1], size=n, p=[0.5, 0.5])
        y = np.where(types == 0, rng.lognormal(8, 0.5, n), rng.lognormal(9, 0.5, n))
        y_pred = np.where(types == 0, np.exp(8), np.exp(9))
        return y, y_pred.astype(float), types

    def test_lower_bound_bounded(self):
        """Lower bound IOp should be in [0, 1]."""
        y, y_pred, _ = self._make_simple_data()
        result = decompose_iop(y, y_pred, None, "gini", "lower_bound")
        assert 0 <= result.iop_share <= 1

    def test_ex_ante_bounded(self):
        y, y_pred, types = self._make_simple_data()
        result = decompose_iop(y, y_pred, types, "mld", "ex_ante")
        assert 0 <= result.iop_share <= 1

    def test_ex_post_bounded(self):
        y, y_pred, types = self._make_simple_data()
        result = decompose_iop(y, y_pred, types, "mld", "ex_post")
        assert 0 <= result.iop_share <= 1

    def test_mld_additively_decomposable(self):
        """For MLD: total = between + within (ex-ante + ex-post should ~ 1)."""
        y, y_pred, types = self._make_simple_data(n=2000)
        # Ex-ante IOp (between-group)
        ex_ante = decompose_iop(y, y_pred, types, "mld", "ex_ante")
        # Ex-post (within from between perspective)
        ex_post = decompose_iop(y, y_pred, types, "mld", "ex_post")

        # For MLD: between/total + within/total = 1
        # ex_ante.iop_share = between/total
        # ex_post gives: 1 - within/total = between/total
        # So both should give similar values
        assert ex_ante.iop_share == pytest.approx(ex_post.iop_share, abs=0.05)

    def test_perfect_equality_zero_iop(self):
        """If all types have same mean, IOp should be ~0."""
        rng = np.random.default_rng(42)
        y = rng.lognormal(8, 0.5, 500)
        types = rng.choice([0, 1, 2], size=500)
        # Shuffle y relative to types so types don't predict income
        y_pred = np.full_like(y, y.mean())

        result = decompose_iop(y, y_pred, types, "mld", "ex_ante")
        assert result.iop_share < 0.01  # Close to zero

    def test_total_inequality_matches(self):
        """Total inequality in result should match direct calculation."""
        y, y_pred, types = self._make_simple_data()
        result = decompose_iop(y, y_pred, types, "gini", "ex_ante")
        direct = compute_inequality(y, "gini")
        assert result.total_inequality == pytest.approx(direct, abs=1e-10)

    def test_more_circumstances_higher_iop(self):
        """More predictive information should generally give higher IOp."""
        rng = np.random.default_rng(42)
        n = 1000
        # Generate data where both type1 and type2 predict income
        type1 = rng.choice([0, 1], size=n)
        type2 = rng.choice([0, 1], size=n)
        y = np.exp(8 + 0.5 * type1 + 0.3 * type2 + rng.normal(0, 0.5, n))

        # Only type1
        y_pred_1 = np.where(type1 == 0, y[type1 == 0].mean(), y[type1 == 1].mean())
        r1 = decompose_iop(y, y_pred_1, type1, "mld", "ex_ante")

        # Both types (finer partition)
        combined = type1 * 2 + type2
        means = {t: y[combined == t].mean() for t in np.unique(combined)}
        y_pred_both = np.array([means[t] for t in combined])
        r2 = decompose_iop(y, y_pred_both, combined, "mld", "ex_ante")

        assert r2.iop_share >= r1.iop_share - 0.01  # Allow small numerical tolerance


class TestExperimentSpec:
    def test_valid_spec(self):
        spec = ExperimentSpec(
            circumstances=(Circumstance.FATHER_EDUCATION.value,),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.OLS.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
        )
        assert spec.is_valid

    def test_no_circumstances_invalid(self):
        spec = ExperimentSpec(
            circumstances=(),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.OLS.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
        )
        assert not spec.is_valid
        assert any("circumstance" in e.lower() for e in spec.validate())

    def test_double_log_invalid(self):
        spec = ExperimentSpec(
            circumstances=(Circumstance.FATHER_EDUCATION.value,),
            income_variable=IncomeVariable.LOG_HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.VAR_LOGS.value,
            method=EstimationMethod.OLS.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
        )
        assert not spec.is_valid
        assert any("double-log" in e for e in spec.validate())

    def test_ols_expost_invalid(self):
        spec = ExperimentSpec(
            circumstances=(Circumstance.FATHER_EDUCATION.value,),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.OLS.value,
            decomposition_type=DecompositionType.EX_POST.value,
        )
        assert not spec.is_valid

    def test_spec_id_deterministic(self):
        spec1 = ExperimentSpec(
            circumstances=(Circumstance.FATHER_EDUCATION.value, Circumstance.ETHNICITY.value),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.OLS.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
        )
        spec2 = ExperimentSpec(
            circumstances=(Circumstance.ETHNICITY.value, Circumstance.FATHER_EDUCATION.value),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.OLS.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
        )
        # Same circumstances (sorted), same spec_id
        assert spec1.spec_id == spec2.spec_id

    def test_roundtrip_serialization(self):
        spec = ExperimentSpec(
            circumstances=(Circumstance.FATHER_EDUCATION.value,),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.OLS.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
            method_params=(("max_depth", 4),),
            rationale="test",
        )
        d = spec.to_dict()
        spec2 = ExperimentSpec.from_dict(d)
        assert spec.spec_id == spec2.spec_id
        assert spec.circumstances == spec2.circumstances

    def test_lower_bound_only_ols(self):
        spec = ExperimentSpec(
            circumstances=(Circumstance.FATHER_EDUCATION.value,),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.DECISION_TREE.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
        )
        assert not spec.is_valid
        assert any("Lower bound" in e for e in spec.validate())
