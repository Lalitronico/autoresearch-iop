"""Editable specification file. THE AGENT MODIFIES THIS FILE.

This is the only file the agent edits to define experiment batches.
Each iteration should have a clear hypothesis and rationale.

Example: Define 3 specs to test IOp with the classic Ferreira-Gignoux
circumstance set (parental education + occupation + ethnicity) across
three income measures using OLS lower bound.
"""

from core.specification import ExperimentSpec
from core.types import (
    DecompositionType,
    EstimationMethod,
    InequalityMeasure,
    SampleFilter,
)
from orchestration.experiment_log import get_completed_spec_ids
from orchestration.strategy import (
    CORE_CIRC_SETS,
    CORE_INCOMES,
)


# Classic IOp set: parental education + occupation + ethnicity (4 variables)
CIRCS = CORE_CIRC_SETS[2]

# Method and decomposition
METHOD = EstimationMethod.OLS.value
DECOMP = DecompositionType.LOWER_BOUND.value


def get_specs() -> list[ExperimentSpec]:
    """Define the specs for the current iteration.

    Edit this function to define your experiment batch.
    Each spec needs: circumstances, income_variable, inequality_measure,
    method, decomposition_type, sample_filter, and a rationale.

    Set use_mi=True for Multiple Imputation (requires prepare.py --impute).
    """
    completed = get_completed_spec_ids()
    specs = []

    for income in CORE_INCOMES:
        spec = ExperimentSpec(
            circumstances=CIRCS,
            income_variable=income,
            inequality_measure=InequalityMeasure.GINI.value,
            method=METHOD,
            decomposition_type=DECOMP,
            sample_filter=SampleFilter.ALL.value,
            rationale="Baseline: OLS lower bound on classic IOp set",
        )
        if spec.is_valid and spec.spec_id not in completed:
            specs.append(spec)

    return specs
