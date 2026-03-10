"""Exploration strategy for the specification space.

Generates batches of ExperimentSpecs based on coverage gaps
and strategic priorities.
"""

from __future__ import annotations

import itertools
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
from orchestration.coverage_tracker import compute_coverage
from orchestration.experiment_log import get_completed_spec_ids


# Core circumstance sets for systematic coverage
CORE_CIRC_SETS: list[tuple[str, ...]] = [
    # Minimal sets
    (Circumstance.FATHER_EDUCATION.value,),
    (Circumstance.FATHER_EDUCATION.value, Circumstance.ETHNICITY.value),
    # Standard set (common in IOp literature)
    (
        Circumstance.FATHER_EDUCATION.value,
        Circumstance.MOTHER_EDUCATION.value,
        Circumstance.FATHER_OCCUPATION.value,
        Circumstance.ETHNICITY.value,
    ),
    # Extended set
    (
        Circumstance.FATHER_EDUCATION.value,
        Circumstance.MOTHER_EDUCATION.value,
        Circumstance.FATHER_OCCUPATION.value,
        Circumstance.ETHNICITY.value,
        Circumstance.SKIN_TONE.value,
        Circumstance.REGION_14.value,
    ),
    # Extended + indigenous language
    (
        Circumstance.FATHER_EDUCATION.value,
        Circumstance.MOTHER_EDUCATION.value,
        Circumstance.FATHER_OCCUPATION.value,
        Circumstance.ETHNICITY.value,
        Circumstance.INDIGENOUS_LANGUAGE.value,
        Circumstance.SKIN_TONE.value,
        Circumstance.REGION_14.value,
        Circumstance.RURAL_14.value,
    ),
    # Full set (all except gender, which is not a "circumstance" in all frameworks)
    tuple(c.value for c in Circumstance if c != Circumstance.GENDER),
    # Full set including gender
    tuple(c.value for c in Circumstance),
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

    Prioritizes filling gaps in the specification space.
    """
    completed = get_completed_spec_ids()
    coverage = compute_coverage()
    specs: list[ExperimentSpec] = []

    # Priority 1: Fill missing method-decomposition combinations
    for method, decomp in VALID_METHOD_DECOMP:
        if len(specs) >= batch_size:
            break
        for circs in CORE_CIRC_SETS[:3]:  # Start with smaller circ sets
            for measure in CORE_MEASURES:
                for income in CORE_INCOMES[:2]:
                    spec = ExperimentSpec(
                        circumstances=circs,
                        income_variable=income,
                        inequality_measure=measure,
                        method=method,
                        decomposition_type=decomp,
                        sample_filter=sample_filter,
                        rationale=f"Systematic coverage: {method}/{decomp}",
                    )
                    if spec.is_valid and spec.spec_id not in completed:
                        specs.append(spec)
                        completed.add(spec.spec_id)
                        if len(specs) >= batch_size:
                            break
                if len(specs) >= batch_size:
                    break
            if len(specs) >= batch_size:
                break

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
