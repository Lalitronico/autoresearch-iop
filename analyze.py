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
from orchestration.experiment_log import get_completed_spec_ids
from orchestration.strategy import (
    CORE_CIRC_SETS,
    CORE_INCOMES,
    CORE_MEASURES,
    VALID_METHOD_DECOMP,
)


# --- Full 585 core specs + subgroup analysis ---
# 13 circ sets × 3 incomes × 3 measures × 5 method-decomp = 585
# Plus subgroup analysis on classic4 and ceey5 circ sets.

# Bootstrap iterations: fast methods get full 200, ML gets 20
BOOTSTRAP_FAST = 200   # OLS, DT: ~0.5-2.5s per spec
BOOTSTRAP_ML = 20      # XGB: ~25s, RF: ~60-354s per spec

# Subgroup analysis on two key circ sets
CIRCS_CLASSIC = CORE_CIRC_SETS[2]   # father_edu, mother_edu, father_occ, ethnicity
CIRCS_CEEY = CORE_CIRC_SETS[10]     # max_parent_edu, wealth_index, region, ethnicity, gender

SUBGROUP_FILTERS = [
    SampleFilter.MALE.value,
    SampleFilter.FEMALE.value,
    SampleFilter.URBAN.value,
    SampleFilter.RURAL.value,
    SampleFilter.AGE_25_44.value,
    SampleFilter.AGE_45_64.value,
    SampleFilter.COHORT_1.value,
    SampleFilter.COHORT_2.value,
    SampleFilter.COHORT_3.value,
    SampleFilter.COHORT_4.value,
]


def get_specs() -> list[ExperimentSpec]:
    """Return all 585 core specs + subgroup analysis.

    Core specs: 13 circ sets × 3 incomes × 3 measures × 5 method-decomp = 585
    Subgroup specs: 2 circ sets × 10 filters × 3 measures × OLS + 2 measures × DT = ~100
    Total: ~685 specs. Already-completed specs are skipped.
    """
    completed = get_completed_spec_ids()
    specs = []

    # --- Block 1: All 585 core specs ---
    # Fast methods first (OLS + DT), then ML (XGB + RF)
    fast_methods = [
        (EstimationMethod.OLS.value, DecompositionType.LOWER_BOUND.value),
        (EstimationMethod.DECISION_TREE.value, DecompositionType.EX_ANTE.value),
        (EstimationMethod.DECISION_TREE.value, DecompositionType.EX_POST.value),
    ]
    # XGBoost on ALL circ sets (manageable: ~25s per spec)
    xgb_method = [(EstimationMethod.XGBOOST.value, DecompositionType.EX_ANTE.value)]

    # RF only on 4 key circ sets to keep runtime practical (~6min per spec)
    # indices: 0=minimal(1), 2=classic(4), 5=extended(12), 10=CEEY(5)
    RF_CIRC_INDICES = [0, 2, 5, 10]
    rf_method = [(EstimationMethod.RANDOM_FOREST.value, DecompositionType.EX_ANTE.value)]

    # Fast methods: all 351 combinations
    for circs in CORE_CIRC_SETS:
        for method, decomp in fast_methods:
            for measure in CORE_MEASURES:
                for income in CORE_INCOMES:
                    spec = ExperimentSpec(
                        circumstances=circs,
                        income_variable=income,
                        inequality_measure=measure,
                        method=method,
                        decomposition_type=decomp,
                        sample_filter=SampleFilter.ALL.value,
                        bootstrap_n=BOOTSTRAP_FAST,
                        rationale=f"Core systematic: {method}/{decomp}, {len(circs)} circs",
                    )
                    if spec.is_valid and spec.spec_id not in completed:
                        specs.append(spec)

    # XGBoost: all 13 circ sets × 3 measures × 3 incomes = 117 specs
    for circs in CORE_CIRC_SETS:
        for method, decomp in xgb_method:
            for measure in CORE_MEASURES:
                for income in CORE_INCOMES:
                    spec = ExperimentSpec(
                        circumstances=circs,
                        income_variable=income,
                        inequality_measure=measure,
                        method=method,
                        decomposition_type=decomp,
                        sample_filter=SampleFilter.ALL.value,
                        bootstrap_n=BOOTSTRAP_ML,
                        rationale=f"Core ML: {method}/{decomp}, {len(circs)} circs",
                    )
                    if spec.is_valid and spec.spec_id not in completed:
                        specs.append(spec)

    # Random Forest: 4 key circ sets × 3 measures × 3 incomes = 36 specs
    for idx in RF_CIRC_INDICES:
        circs = CORE_CIRC_SETS[idx]
        for method, decomp in rf_method:
            for measure in CORE_MEASURES:
                for income in CORE_INCOMES:
                    spec = ExperimentSpec(
                        circumstances=circs,
                        income_variable=income,
                        inequality_measure=measure,
                        method=method,
                        decomposition_type=decomp,
                        sample_filter=SampleFilter.ALL.value,
                        bootstrap_n=BOOTSTRAP_ML,
                        rationale=f"Core ML: {method}/{decomp}, {len(circs)} circs",
                    )
                    if spec.is_valid and spec.spec_id not in completed:
                        specs.append(spec)

    # --- Block 2: Subgroup analysis ---
    for circs, label in [(CIRCS_CLASSIC, "classic4"), (CIRCS_CEEY, "ceey5")]:
        for sf in SUBGROUP_FILTERS:
            for measure in CORE_MEASURES:
                # OLS lower bound
                spec = ExperimentSpec(
                    circumstances=circs,
                    income_variable=IncomeVariable.HH_PC_IMPUTED.value,
                    inequality_measure=measure,
                    method=EstimationMethod.OLS.value,
                    decomposition_type=DecompositionType.LOWER_BOUND.value,
                    sample_filter=sf,
                    bootstrap_n=BOOTSTRAP_FAST,
                    rationale=f"Subgroup: {label}, {sf}",
                )
                if spec.is_valid and spec.spec_id not in completed:
                    specs.append(spec)

            # DT ex_ante (gini + mld only for subgroups)
            for measure in [InequalityMeasure.GINI.value, InequalityMeasure.MLD.value]:
                spec = ExperimentSpec(
                    circumstances=circs,
                    income_variable=IncomeVariable.HH_PC_IMPUTED.value,
                    inequality_measure=measure,
                    method=EstimationMethod.DECISION_TREE.value,
                    decomposition_type=DecompositionType.EX_ANTE.value,
                    sample_filter=sf,
                    bootstrap_n=BOOTSTRAP_FAST,
                    rationale=f"Subgroup DT: {label}, {sf}",
                )
                if spec.is_valid and spec.spec_id not in completed:
                    specs.append(spec)

    return specs
