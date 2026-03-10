"""Tests for DataRegistry: validation, filtering, and data access."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import core.data_loader as dl
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

# synthetic_data fixture provided by conftest.py


class TestValidateSpec:
    def test_valid_spec_passes(self, synthetic_data):
        """A spec with valid columns should pass validation."""
        spec = ExperimentSpec(
            circumstances=(Circumstance.FATHER_EDUCATION.value,),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.OLS.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
        )
        registry = DataRegistry()
        errors = registry.validate_spec(spec)
        assert errors == []

    def test_missing_income_variable(self, synthetic_data):
        """Spec with non-existent income variable should fail validation."""
        # Create a spec with an income variable that maps to a base column
        # not in the dataset. We'll test by creating a custom registry with
        # a subset of columns.
        df = synthetic_data.drop(columns=["hh_pc_imputed"])
        tmp_path = dl.PROCESSED_DIR / "test_missing_income.parquet"
        df.to_parquet(tmp_path, index=False)
        try:
            registry = DataRegistry(data_path=tmp_path)
            spec = ExperimentSpec(
                circumstances=(Circumstance.FATHER_EDUCATION.value,),
                income_variable=IncomeVariable.HH_PC_IMPUTED.value,
                inequality_measure=InequalityMeasure.GINI.value,
                method=EstimationMethod.OLS.value,
                decomposition_type=DecompositionType.LOWER_BOUND.value,
            )
            errors = registry.validate_spec(spec)
            assert len(errors) > 0
            assert any("hh_pc_imputed" in e for e in errors)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_missing_circumstance(self, synthetic_data):
        """Spec with non-existent circumstance should fail validation."""
        df = synthetic_data.drop(columns=["ethnicity"])
        tmp_path = dl.PROCESSED_DIR / "test_missing_circ.parquet"
        df.to_parquet(tmp_path, index=False)
        try:
            registry = DataRegistry(data_path=tmp_path)
            spec = ExperimentSpec(
                circumstances=(
                    Circumstance.FATHER_EDUCATION.value,
                    Circumstance.ETHNICITY.value,
                ),
                income_variable=IncomeVariable.HH_PC_IMPUTED.value,
                inequality_measure=InequalityMeasure.GINI.value,
                method=EstimationMethod.OLS.value,
                decomposition_type=DecompositionType.LOWER_BOUND.value,
            )
            errors = registry.validate_spec(spec)
            assert len(errors) > 0
            assert any("ethnicity" in e for e in errors)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_all_nan_circumstance_fails(self, synthetic_data):
        """Circumstance with all NaN should fail the min_valid check."""
        df = synthetic_data.copy()
        df["ethnicity"] = np.nan
        tmp_path = dl.PROCESSED_DIR / "test_nan_circ.parquet"
        df.to_parquet(tmp_path, index=False)
        try:
            registry = DataRegistry(data_path=tmp_path)
            spec = ExperimentSpec(
                circumstances=(Circumstance.ETHNICITY.value,),
                income_variable=IncomeVariable.HH_PC_IMPUTED.value,
                inequality_measure=InequalityMeasure.GINI.value,
                method=EstimationMethod.OLS.value,
                decomposition_type=DecompositionType.LOWER_BOUND.value,
            )
            errors = registry.validate_spec(spec)
            assert len(errors) > 0
            assert any("non-null" in e.lower() or "non-null" in e for e in errors)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_few_valid_values_fails(self, synthetic_data):
        """Circumstance with < 100 non-null values should fail."""
        df = synthetic_data.copy()
        # Set all but 50 values to NaN
        mask = np.ones(len(df), dtype=bool)
        mask[:50] = False
        df.loc[mask, "ethnicity"] = np.nan
        tmp_path = dl.PROCESSED_DIR / "test_few_valid.parquet"
        df.to_parquet(tmp_path, index=False)
        try:
            registry = DataRegistry(data_path=tmp_path)
            spec = ExperimentSpec(
                circumstances=(Circumstance.ETHNICITY.value,),
                income_variable=IncomeVariable.HH_PC_IMPUTED.value,
                inequality_measure=InequalityMeasure.GINI.value,
                method=EstimationMethod.OLS.value,
                decomposition_type=DecompositionType.LOWER_BOUND.value,
            )
            errors = registry.validate_spec(spec)
            assert len(errors) > 0
            assert any("50" in e or "non-null" in e.lower() for e in errors)
        finally:
            tmp_path.unlink(missing_ok=True)


class TestSampleFilters:
    def test_all_filter_returns_full(self, synthetic_data):
        """ALL filter should return the complete dataset."""
        registry = DataRegistry()
        filtered = registry.apply_filter(SampleFilter.ALL.value)
        assert len(filtered) == len(registry.df)

    def test_gender_filters_partition(self, synthetic_data):
        """Male + Female should cover all observations."""
        registry = DataRegistry()
        males = registry.apply_filter(SampleFilter.MALE.value)
        females = registry.apply_filter(SampleFilter.FEMALE.value)
        assert len(males) + len(females) == len(registry.df)
        assert len(males) > 0
        assert len(females) > 0

    def test_urban_rural_partition(self, synthetic_data):
        """Urban + Rural should cover all observations."""
        registry = DataRegistry()
        urban = registry.apply_filter(SampleFilter.URBAN.value)
        rural = registry.apply_filter(SampleFilter.RURAL.value)
        assert len(urban) + len(rural) == len(registry.df)

    def test_age_filters_reduce_sample(self, synthetic_data):
        """Age filters should produce subsets."""
        registry = DataRegistry()
        full = len(registry.df)
        young = registry.apply_filter(SampleFilter.AGE_25_44.value)
        old = registry.apply_filter(SampleFilter.AGE_45_64.value)
        assert len(young) < full
        assert len(old) < full
        assert len(young) > 0
        assert len(old) > 0

    def test_cohort_filters_partition_by_age(self, synthetic_data):
        """Cohort filters should partition the 25-64 range."""
        registry = DataRegistry()
        c1 = registry.apply_filter(SampleFilter.COHORT_1.value)
        c2 = registry.apply_filter(SampleFilter.COHORT_2.value)
        c3 = registry.apply_filter(SampleFilter.COHORT_3.value)
        c4 = registry.apply_filter(SampleFilter.COHORT_4.value)
        total_cohorts = len(c1) + len(c2) + len(c3) + len(c4)
        # Should cover most of the data (all ages 25-64)
        assert total_cohorts == len(registry.df)

    def test_intersection_filter(self, synthetic_data):
        """Urban male should be subset of both urban and male."""
        registry = DataRegistry()
        urban = registry.apply_filter(SampleFilter.URBAN.value)
        male = registry.apply_filter(SampleFilter.MALE.value)
        urban_male = registry.apply_filter(SampleFilter.URBAN_MALE.value)
        assert len(urban_male) <= len(urban)
        assert len(urban_male) <= len(male)
        assert len(urban_male) > 0


class TestGetSampleForSpec:
    def test_returns_clean_data(self, synthetic_data):
        """get_sample_for_spec should return data with no NaN."""
        registry = DataRegistry()
        spec = ExperimentSpec(
            circumstances=(
                Circumstance.FATHER_EDUCATION.value,
                Circumstance.MOTHER_EDUCATION.value,
            ),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.OLS.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
        )
        y, X, idx = registry.get_sample_for_spec(spec)
        assert y.notna().all()
        assert X.notna().all().all()
        assert len(y) == len(X)
        assert len(y) > 0

    def test_log_income_transform(self, synthetic_data):
        """Log income variables should be properly transformed."""
        registry = DataRegistry()
        spec = ExperimentSpec(
            circumstances=(Circumstance.FATHER_EDUCATION.value,),
            income_variable=IncomeVariable.LOG_HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.OLS.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
        )
        y, X, idx = registry.get_sample_for_spec(spec)
        # Log-transformed income should be much smaller than raw
        raw_spec = ExperimentSpec(
            circumstances=(Circumstance.FATHER_EDUCATION.value,),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.OLS.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
        )
        y_raw, _, _ = registry.get_sample_for_spec(raw_spec)
        assert y.mean() < y_raw.mean()  # log values << raw values

    def test_filter_reduces_sample(self, synthetic_data):
        """Filtered spec should have fewer obs than unfiltered."""
        registry = DataRegistry()
        spec_all = ExperimentSpec(
            circumstances=(Circumstance.FATHER_EDUCATION.value,),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.OLS.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
            sample_filter=SampleFilter.ALL.value,
        )
        spec_male = ExperimentSpec(
            circumstances=(Circumstance.FATHER_EDUCATION.value,),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.OLS.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
            sample_filter=SampleFilter.MALE.value,
        )
        y_all, _, _ = registry.get_sample_for_spec(spec_all)
        y_male, _, _ = registry.get_sample_for_spec(spec_male)
        assert len(y_all) > len(y_male)


class TestIncomeAccess:
    def test_get_income_basic(self, synthetic_data):
        """get_income should return the correct series."""
        registry = DataRegistry()
        y = registry.get_income(IncomeVariable.HH_PC_IMPUTED.value)
        assert len(y) == len(registry.df)
        assert y.notna().sum() > 0

    def test_get_income_log(self, synthetic_data):
        """Log income should be log-transformed."""
        registry = DataRegistry()
        y_raw = registry.get_income(IncomeVariable.HH_PC_IMPUTED.value)
        y_log = registry.get_income(IncomeVariable.LOG_HH_PC_IMPUTED.value)
        # For positive values, log(y) should be close
        mask = y_raw > 0
        np.testing.assert_allclose(
            y_log[mask].values, np.log(y_raw[mask].values), rtol=1e-10
        )

    def test_get_income_missing_raises(self, synthetic_data):
        """Accessing a non-existent income column should raise KeyError."""
        df = synthetic_data.drop(columns=["hh_total_reported"])
        tmp_path = dl.PROCESSED_DIR / "test_no_reported.parquet"
        df.to_parquet(tmp_path, index=False)
        try:
            registry = DataRegistry(data_path=tmp_path)
            with pytest.raises(KeyError):
                registry.get_income(IncomeVariable.HH_TOTAL_REPORTED.value)
        finally:
            tmp_path.unlink(missing_ok=True)
