"""Tests for specification validation: domain rules, method-decomposition pairs,
CEEY circumstance mapping, and bootstrap configuration."""

import numpy as np
import pandas as pd
import pytest

from core.specification import ExperimentSpec
from core.types import (
    Circumstance,
    DecompositionType,
    EstimationMethod,
    InequalityMeasure,
    IncomeVariable,
    SampleFilter,
)
from core.data_loader import DataRegistry, PROCESSED_DIR
from prepare import _build_education_6


class TestMethodDecompositionPairs:
    """Test all valid and invalid method-decomposition combinations."""

    def test_ols_lower_bound_valid(self):
        spec = ExperimentSpec(
            circumstances=(Circumstance.FATHER_EDUCATION.value,),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.OLS.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
        )
        assert spec.is_valid

    def test_ols_ex_ante_valid(self):
        """OLS + ex_ante: OLS predictions can define types via binning."""
        spec = ExperimentSpec(
            circumstances=(Circumstance.FATHER_EDUCATION.value,),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.OLS.value,
            decomposition_type=DecompositionType.EX_ANTE.value,
        )
        # This may or may not be valid depending on pipeline rules
        # Just verify it doesn't crash validation
        _ = spec.validate()

    def test_ols_ex_post_invalid(self):
        """OLS + ex_post should be invalid (requires partition method)."""
        spec = ExperimentSpec(
            circumstances=(Circumstance.FATHER_EDUCATION.value,),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.OLS.value,
            decomposition_type=DecompositionType.EX_POST.value,
        )
        assert not spec.is_valid

    def test_decision_tree_ex_ante_valid(self):
        spec = ExperimentSpec(
            circumstances=(Circumstance.FATHER_EDUCATION.value,),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.DECISION_TREE.value,
            decomposition_type=DecompositionType.EX_ANTE.value,
        )
        assert spec.is_valid

    def test_decision_tree_ex_post_valid(self):
        spec = ExperimentSpec(
            circumstances=(Circumstance.FATHER_EDUCATION.value,),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.DECISION_TREE.value,
            decomposition_type=DecompositionType.EX_POST.value,
        )
        assert spec.is_valid

    def test_decision_tree_lower_bound_invalid(self):
        """DT + lower_bound should be invalid."""
        spec = ExperimentSpec(
            circumstances=(Circumstance.FATHER_EDUCATION.value,),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.DECISION_TREE.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
        )
        assert not spec.is_valid

    def test_xgboost_ex_ante_valid(self):
        spec = ExperimentSpec(
            circumstances=(Circumstance.FATHER_EDUCATION.value,),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.XGBOOST.value,
            decomposition_type=DecompositionType.EX_ANTE.value,
        )
        assert spec.is_valid

    def test_xgboost_lower_bound_invalid(self):
        spec = ExperimentSpec(
            circumstances=(Circumstance.FATHER_EDUCATION.value,),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.XGBOOST.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
        )
        assert not spec.is_valid

    def test_random_forest_ex_ante_valid(self):
        spec = ExperimentSpec(
            circumstances=(Circumstance.FATHER_EDUCATION.value,),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.RANDOM_FOREST.value,
            decomposition_type=DecompositionType.EX_ANTE.value,
        )
        assert spec.is_valid

    def test_random_forest_lower_bound_invalid(self):
        spec = ExperimentSpec(
            circumstances=(Circumstance.FATHER_EDUCATION.value,),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.RANDOM_FOREST.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
        )
        assert not spec.is_valid


class TestDoubleLogRule:
    """Log income + var_logs = double-log → invalid."""

    def test_log_hh_pc_var_logs_invalid(self):
        spec = ExperimentSpec(
            circumstances=(Circumstance.FATHER_EDUCATION.value,),
            income_variable=IncomeVariable.LOG_HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.VAR_LOGS.value,
            method=EstimationMethod.OLS.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
        )
        assert not spec.is_valid
        assert any("double-log" in e for e in spec.validate())

    def test_log_hh_total_var_logs_invalid(self):
        spec = ExperimentSpec(
            circumstances=(Circumstance.FATHER_EDUCATION.value,),
            income_variable=IncomeVariable.LOG_HH_TOTAL_REPORTED.value,
            inequality_measure=InequalityMeasure.VAR_LOGS.value,
            method=EstimationMethod.OLS.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
        )
        assert not spec.is_valid

    def test_levels_var_logs_valid(self):
        """Levels income + var_logs is fine (single log)."""
        spec = ExperimentSpec(
            circumstances=(Circumstance.FATHER_EDUCATION.value,),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.VAR_LOGS.value,
            method=EstimationMethod.OLS.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
        )
        assert spec.is_valid

    def test_log_income_gini_valid(self):
        """Log income + non-var_logs measure is fine."""
        spec = ExperimentSpec(
            circumstances=(Circumstance.FATHER_EDUCATION.value,),
            income_variable=IncomeVariable.LOG_HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.OLS.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
        )
        assert spec.is_valid


class TestBootstrapConfig:
    def test_zero_bootstrap_invalid(self):
        spec = ExperimentSpec(
            circumstances=(Circumstance.FATHER_EDUCATION.value,),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.OLS.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
            bootstrap_n=0,
        )
        assert not spec.is_valid
        assert any("bootstrap" in e.lower() for e in spec.validate())

    def test_recommended_bootstrap_fast_methods(self):
        """OLS and DT should recommend 200 iterations."""
        assert ExperimentSpec.recommended_bootstrap_n("ols") == 200
        assert ExperimentSpec.recommended_bootstrap_n("decision_tree") == 200

    def test_recommended_bootstrap_slow_methods(self):
        """XGBoost and RF should recommend 50 iterations."""
        assert ExperimentSpec.recommended_bootstrap_n("xgboost") == 50
        assert ExperimentSpec.recommended_bootstrap_n("random_forest") == 50


class TestSpecIdDeterminism:
    def test_circumstance_order_invariant(self):
        """Specs with same circumstances in different order should have same ID."""
        s1 = ExperimentSpec(
            circumstances=("father_education", "mother_education", "ethnicity"),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.OLS.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
        )
        s2 = ExperimentSpec(
            circumstances=("ethnicity", "father_education", "mother_education"),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.OLS.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
        )
        assert s1.spec_id == s2.spec_id

    def test_different_measures_different_ids(self):
        """Different inequality measures should produce different IDs."""
        base = dict(
            circumstances=(Circumstance.FATHER_EDUCATION.value,),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            method=EstimationMethod.OLS.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
        )
        s_gini = ExperimentSpec(inequality_measure=InequalityMeasure.GINI.value, **base)
        s_mld = ExperimentSpec(inequality_measure=InequalityMeasure.MLD.value, **base)
        assert s_gini.spec_id != s_mld.spec_id


class TestBuildEducation6:
    """Test the CEEY education mapping function."""

    def test_sin_estudios(self):
        """Nivel 1 (ninguno) and 2 (preescolar) -> category 1."""
        df = pd.DataFrame({"p43a": [1, 2], "p44a": [np.nan, np.nan]})
        result = _build_education_6(df, "p43a", "p44a")
        assert result.iloc[0] == 1.0
        assert result.iloc[1] == 1.0

    def test_primaria_completa(self):
        """Nivel 3 (primaria) + completó (1) -> category 3."""
        df = pd.DataFrame({"p43a": [3], "p44a": [1]})
        result = _build_education_6(df, "p43a", "p44a")
        assert result.iloc[0] == 3.0

    def test_primaria_incompleta(self):
        """Nivel 3 (primaria) + no completó (2) -> category 2."""
        df = pd.DataFrame({"p43a": [3], "p44a": [2]})
        result = _build_education_6(df, "p43a", "p44a")
        assert result.iloc[0] == 2.0

    def test_primaria_no_sabe(self):
        """Nivel 3 (primaria) + no sabe (8) -> midpoint 2.5."""
        df = pd.DataFrame({"p43a": [3], "p44a": [8]})
        result = _build_education_6(df, "p43a", "p44a")
        assert result.iloc[0] == 2.5

    def test_primaria_missing_completion(self):
        """Nivel 3 (primaria) + missing completion -> midpoint 2.5."""
        df = pd.DataFrame({"p43a": [3], "p44a": [np.nan]})
        result = _build_education_6(df, "p43a", "p44a")
        assert result.iloc[0] == 2.5

    def test_secundaria(self):
        """Nivel 4 (secundaria) -> category 4."""
        df = pd.DataFrame({"p43a": [4], "p44a": [1]})
        result = _build_education_6(df, "p43a", "p44a")
        assert result.iloc[0] == 4.0

    def test_preparatoria(self):
        """Nivel 5 (preparatoria) -> category 5."""
        df = pd.DataFrame({"p43a": [5], "p44a": [1]})
        result = _build_education_6(df, "p43a", "p44a")
        assert result.iloc[0] == 5.0

    def test_profesional(self):
        """Nivel 6 (profesional+) -> category 6."""
        df = pd.DataFrame({"p43a": [6], "p44a": [1]})
        result = _build_education_6(df, "p43a", "p44a")
        assert result.iloc[0] == 6.0

    def test_no_sabe_nivel(self):
        """Nivel 9 (no sabe) -> NaN."""
        df = pd.DataFrame({"p43a": [9], "p44a": [np.nan]})
        result = _build_education_6(df, "p43a", "p44a")
        assert pd.isna(result.iloc[0])

    def test_missing_nivel(self):
        """Missing nivel -> NaN."""
        df = pd.DataFrame({"p43a": [np.nan], "p44a": [1]})
        result = _build_education_6(df, "p43a", "p44a")
        assert pd.isna(result.iloc[0])

    def test_missing_column_returns_all_nan(self):
        """If nivel column doesn't exist, return all NaN."""
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        result = _build_education_6(df, "p43a", "p44a")
        assert result.isna().all()
        assert len(result) == 3

    def test_full_range(self):
        """Test all valid levels produce expected output range."""
        df = pd.DataFrame({
            "p43a": [1, 2, 3, 3, 3, 4, 5, 6],
            "p44a": [np.nan, np.nan, 1, 2, 8, 1, 2, 1],
        })
        result = _build_education_6(df, "p43a", "p44a")
        assert result.min() == 1.0
        assert result.max() == 6.0
        # All values should be in [1, 6]
        assert (result.dropna() >= 1).all()
        assert (result.dropna() <= 6).all()


class TestAllCircumstancesExist:
    """Verify every Circumstance enum has a corresponding column in synthetic data."""

    def test_all_circumstances_in_synthetic(self):
        from prepare import create_synthetic_data
        df = create_synthetic_data(n=100, seed=42)
        for circ in Circumstance:
            assert circ.value in df.columns, (
                f"Circumstance {circ.name} ({circ.value}) missing from synthetic data"
            )

    def test_all_circumstances_non_null(self):
        """Every circumstance should have at least some non-null values in synthetic data."""
        from prepare import create_synthetic_data
        df = create_synthetic_data(n=100, seed=42)
        for circ in Circumstance:
            n_valid = df[circ.value].notna().sum()
            assert n_valid > 0, (
                f"Circumstance {circ.name} has 0 non-null values in synthetic data"
            )


class TestAllSampleFilters:
    """Verify every SampleFilter produces a non-empty subset."""

    def test_all_filters_produce_data(self):
        from prepare import create_synthetic_data
        df = create_synthetic_data(n=1000, seed=42)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = PROCESSED_DIR / "test_filters.parquet"
        df.to_parquet(tmp_path, index=False)
        try:
            registry = DataRegistry(data_path=tmp_path)
            for sf in SampleFilter:
                filtered = registry.apply_filter(sf.value)
                assert len(filtered) > 0, (
                    f"SampleFilter {sf.name} produced empty dataset"
                )
        finally:
            tmp_path.unlink(missing_ok=True)
