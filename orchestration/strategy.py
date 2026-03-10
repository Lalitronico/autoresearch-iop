"""Exploration strategy for the specification space.

Generates batches of ExperimentSpecs based on coverage gaps
and strategic priorities.
"""

from __future__ import annotations

from typing import Any

from core.specification import ExperimentSpec
from core.types import (
    Circumstance,
    DecompositionType,
    EstimationMethod,
    InequalityMeasure,
    IncomeVariable,
    SampleFilter,
)
from orchestration.experiment_log import get_completed_spec_ids


# Core circumstance sets for systematic coverage.
# Organized from minimal to maximal, reflecting IOp literature conventions.
# IOp is monotonically non-decreasing in the number of circumstances,
# so each level adds information about how much the lower bound grows.
CORE_CIRC_SETS: list[tuple[str, ...]] = [
    # 1. Minimal: single variable (baseline benchmark)
    (Circumstance.FATHER_EDUCATION.value,),

    # 2. Minimal pair: education + ethnicity
    (Circumstance.FATHER_EDUCATION.value, Circumstance.ETHNICITY.value),

    # 3. Classic IOp (Ferreira-Gignoux style): parental SES + ethnicity
    (
        Circumstance.FATHER_EDUCATION.value,
        Circumstance.MOTHER_EDUCATION.value,
        Circumstance.FATHER_OCCUPATION.value,
        Circumstance.ETHNICITY.value,
    ),

    # 4. Extended classic + phenotype + geography
    (
        Circumstance.FATHER_EDUCATION.value,
        Circumstance.MOTHER_EDUCATION.value,
        Circumstance.FATHER_OCCUPATION.value,
        Circumstance.ETHNICITY.value,
        Circumstance.SKIN_TONE.value,
        Circumstance.REGION_14.value,
    ),

    # 5. Full parental + identity + geography
    (
        Circumstance.FATHER_EDUCATION.value,
        Circumstance.MOTHER_EDUCATION.value,
        Circumstance.FATHER_OCCUPATION.value,
        Circumstance.MOTHER_OCCUPATION.value,
        Circumstance.ETHNICITY.value,
        Circumstance.INDIGENOUS_LANGUAGE.value,
        Circumstance.SKIN_TONE.value,
        Circumstance.REGION_14.value,
        Circumstance.RURAL_14.value,
    ),

    # 6. Add family structure
    (
        Circumstance.FATHER_EDUCATION.value,
        Circumstance.MOTHER_EDUCATION.value,
        Circumstance.FATHER_OCCUPATION.value,
        Circumstance.MOTHER_OCCUPATION.value,
        Circumstance.ETHNICITY.value,
        Circumstance.INDIGENOUS_LANGUAGE.value,
        Circumstance.SKIN_TONE.value,
        Circumstance.REGION_14.value,
        Circumstance.RURAL_14.value,
        Circumstance.HH_SIZE_14.value,
        Circumstance.N_SIBLINGS.value,
        Circumstance.BIRTH_ORDER.value,
    ),

    # 7. Add material conditions (wealth indices)
    (
        Circumstance.FATHER_EDUCATION.value,
        Circumstance.MOTHER_EDUCATION.value,
        Circumstance.FATHER_OCCUPATION.value,
        Circumstance.MOTHER_OCCUPATION.value,
        Circumstance.ETHNICITY.value,
        Circumstance.INDIGENOUS_LANGUAGE.value,
        Circumstance.SKIN_TONE.value,
        Circumstance.REGION_14.value,
        Circumstance.RURAL_14.value,
        Circumstance.HH_ASSETS_14.value,
        Circumstance.FINANCIAL_ASSETS_14.value,
        Circumstance.DWELLING_AMENITIES_14.value,
        Circumstance.NEIGHBORHOOD_QUALITY_14.value,
    ),

    # 8. Full set (all except gender)
    tuple(c.value for c in Circumstance if c != Circumstance.GENDER),

    # 9. Full set including gender
    tuple(c.value for c in Circumstance),

    # 10. Material conditions only (test: how much does childhood wealth
    #     explain vs parental human capital?)
    (
        Circumstance.HH_ASSETS_14.value,
        Circumstance.FINANCIAL_ASSETS_14.value,
        Circumstance.DWELLING_AMENITIES_14.value,
        Circumstance.DWELLING_FEATURES_14.value,
        Circumstance.DWELLING_ROOMS_14.value,
        Circumstance.NEIGHBORHOOD_QUALITY_14.value,
        Circumstance.N_AUTOMOBILES_14.value,
    ),

    # 11. CEEY-comparable: uses exact CEEY variables (IREH-O + max_parent_edu + region)
    # Allows direct comparison with Informe de Movilidad Social results
    (
        Circumstance.MAX_PARENT_EDUCATION.value,
        Circumstance.WEALTH_INDEX_ORIGIN.value,
        Circumstance.REGION_14.value,
        Circumstance.ETHNICITY.value,
        Circumstance.GENDER.value,
    ),

    # 12. IREH-O only (test: how much does the MCA-weighted index explain
    #     vs our crude count indices?)
    (
        Circumstance.WEALTH_INDEX_ORIGIN.value,
    ),

    # 13. 6-category education + IREH-O (richer human capital + wealth)
    (
        Circumstance.FATHER_EDUCATION_6.value,
        Circumstance.MOTHER_EDUCATION_6.value,
        Circumstance.FATHER_OCCUPATION.value,
        Circumstance.MOTHER_OCCUPATION.value,
        Circumstance.WEALTH_INDEX_ORIGIN.value,
        Circumstance.ETHNICITY.value,
        Circumstance.REGION_14.value,
    ),
]

# Priority income variables
CORE_INCOMES = [
    IncomeVariable.HH_PC_IMPUTED.value,
    IncomeVariable.HH_TOTAL_REPORTED.value,
    IncomeVariable.LOG_HH_PC_IMPUTED.value,
]

# Priority measures
CORE_MEASURES = [
    InequalityMeasure.GINI.value,
    InequalityMeasure.MLD.value,
    InequalityMeasure.THEIL_T.value,
]

# Valid method-decomposition combinations
VALID_METHOD_DECOMP = [
    (EstimationMethod.OLS.value, DecompositionType.LOWER_BOUND.value),
    (EstimationMethod.DECISION_TREE.value, DecompositionType.EX_ANTE.value),
    (EstimationMethod.DECISION_TREE.value, DecompositionType.EX_POST.value),
    (EstimationMethod.XGBOOST.value, DecompositionType.EX_ANTE.value),
    (EstimationMethod.RANDOM_FOREST.value, DecompositionType.EX_ANTE.value),
]


def generate_systematic_batch(
    batch_size: int = 10,
    sample_filter: str = SampleFilter.ALL.value,
) -> list[ExperimentSpec]:
    """Generate a batch of specs for systematic coverage.

    Generates all valid core combinations, filters already-completed ones,
    and returns up to batch_size specs. Iteration order prioritizes diversity:
    circ_sets (outer) → method-decomp → measures → incomes (inner),
    so early batches touch many circumstance sets and methods.
    """
    completed = get_completed_spec_ids()
    specs: list[ExperimentSpec] = []

    for circs in CORE_CIRC_SETS:
        for method, decomp in VALID_METHOD_DECOMP:
            for measure in CORE_MEASURES:
                for income in CORE_INCOMES:
                    spec = ExperimentSpec(
                        circumstances=circs,
                        income_variable=income,
                        inequality_measure=measure,
                        method=method,
                        decomposition_type=decomp,
                        sample_filter=sample_filter,
                        rationale=f"Systematic: {method}/{decomp}, {len(circs)} circs",
                    )
                    if spec.is_valid and spec.spec_id not in completed:
                        specs.append(spec)
                        completed.add(spec.spec_id)
                        if len(specs) >= batch_size:
                            return specs

    return specs


def generate_robustness_batch(
    base_spec: ExperimentSpec,
    batch_size: int = 5,
) -> list[ExperimentSpec]:
    """Generate robustness checks by varying one dimension at a time from a base spec."""
    completed = get_completed_spec_ids()
    specs: list[ExperimentSpec] = []

    # Vary income variable
    for income in IncomeVariable:
        spec = ExperimentSpec(
            circumstances=base_spec.circumstances,
            income_variable=income.value,
            inequality_measure=base_spec.inequality_measure,
            method=base_spec.method,
            decomposition_type=base_spec.decomposition_type,
            sample_filter=base_spec.sample_filter,
            rationale=f"Robustness: vary income to {income.value}",
        )
        if spec.is_valid and spec.spec_id not in completed:
            specs.append(spec)
            if len(specs) >= batch_size:
                return specs

    # Vary inequality measure
    for measure in InequalityMeasure:
        spec = ExperimentSpec(
            circumstances=base_spec.circumstances,
            income_variable=base_spec.income_variable,
            inequality_measure=measure.value,
            method=base_spec.method,
            decomposition_type=base_spec.decomposition_type,
            sample_filter=base_spec.sample_filter,
            rationale=f"Robustness: vary measure to {measure.value}",
        )
        if spec.is_valid and spec.spec_id not in completed:
            specs.append(spec)
            if len(specs) >= batch_size:
                return specs

    # Vary sample
    for sf in SampleFilter:
        spec = ExperimentSpec(
            circumstances=base_spec.circumstances,
            income_variable=base_spec.income_variable,
            inequality_measure=base_spec.inequality_measure,
            method=base_spec.method,
            decomposition_type=base_spec.decomposition_type,
            sample_filter=sf.value,
            rationale=f"Robustness: vary sample to {sf.value}",
        )
        if spec.is_valid and spec.spec_id not in completed:
            specs.append(spec)
            if len(specs) >= batch_size:
                return specs

    return specs


def generate_hypothesis_batch(
    hypothesis: str,
    specs_config: list[dict[str, Any]],
) -> list[ExperimentSpec]:
    """Generate specs from a specific research hypothesis.

    The agent provides a hypothesis and list of spec configurations.
    """
    specs = []
    for config in specs_config:
        config["rationale"] = f"Hypothesis: {hypothesis}"
        if isinstance(config.get("method_params"), dict):
            config["method_params"] = tuple(config["method_params"].items())
        if isinstance(config.get("circumstances"), list):
            config["circumstances"] = tuple(config["circumstances"])
        spec = ExperimentSpec(**config)
        if spec.is_valid:
            specs.append(spec)
    return specs
