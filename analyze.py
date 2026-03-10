"""Editable specification file. THE AGENT MODIFIES THIS FILE.

Each iteration, the agent defines a batch of ExperimentSpecs to run.
The specs are then executed by run_experiment.py.
"""

from core.specification import ExperimentSpec
from core.types import (
    Circumstance,
    DecompositionType,
    EstimationMethod,
    InequalityMeasure,
    IncomeVariable,
    SampleFilter,
)


def get_specs() -> list[ExperimentSpec]:
    """Return specs to run in this iteration.

    The LLM agent edits this function to define experiments.
    """
    specs = [
        # Baseline: OLS parametric with standard circumstances
        ExperimentSpec(
            circumstances=(
                Circumstance.FATHER_EDUCATION.value,
                Circumstance.MOTHER_EDUCATION.value,
                Circumstance.FATHER_OCCUPATION.value,
                Circumstance.ETHNICITY.value,
            ),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.GINI.value,
            method=EstimationMethod.OLS.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
            sample_filter=SampleFilter.ALL.value,
            rationale="Baseline parametric IOp with standard circumstances and Gini",
        ),
        # Same but with MLD (exactly decomposable)
        ExperimentSpec(
            circumstances=(
                Circumstance.FATHER_EDUCATION.value,
                Circumstance.MOTHER_EDUCATION.value,
                Circumstance.FATHER_OCCUPATION.value,
                Circumstance.ETHNICITY.value,
            ),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.MLD.value,
            method=EstimationMethod.OLS.value,
            decomposition_type=DecompositionType.LOWER_BOUND.value,
            sample_filter=SampleFilter.ALL.value,
            rationale="Parametric IOp with MLD -- exactly decomposable measure",
        ),
        # Non-parametric with decision tree
        ExperimentSpec(
            circumstances=(
                Circumstance.FATHER_EDUCATION.value,
                Circumstance.MOTHER_EDUCATION.value,
                Circumstance.FATHER_OCCUPATION.value,
                Circumstance.ETHNICITY.value,
            ),
            income_variable=IncomeVariable.HH_PC_IMPUTED.value,
            inequality_measure=InequalityMeasure.MLD.value,
            method=EstimationMethod.DECISION_TREE.value,
            decomposition_type=DecompositionType.EX_ANTE.value,
            sample_filter=SampleFilter.ALL.value,
            rationale="Non-parametric ex-ante IOp with decision tree types",
        ),
    ]
    return specs
