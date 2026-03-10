"""Fixed experiment harness. NEVER modified by the agent.

Receives ExperimentSpec -> validates -> loads data -> runs method ->
computes IOp with bootstrap CI -> runs diagnostics -> logs result.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

from core.data_loader import DataRegistry
from core.specification import ExperimentSpec
from core.types import EstimationMethod, ExperimentStatus, IncomeVariable
from evaluation.diagnostics import run_diagnostics, DiagnosticResult
from evaluation.metrics import compute_iop_with_ci, IOpEstimate
from methods.parametric import estimate_parametric
from methods.nonparametric import estimate_nonparametric
from methods.ml_methods import estimate_xgboost, estimate_random_forest
from orchestration.experiment_log import log_experiment, get_completed_spec_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Complete result from a single experiment run."""
    spec_id: str
    status: str
    iop_share: float | None = None
    iop_absolute: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    total_inequality: float | None = None
    r_squared: float | None = None
    cv_r_squared: float | None = None
    n_obs: int | None = None
    n_types: int | None = None
    variable_importance: dict[str, float] = field(default_factory=dict)
    shap_importance: dict[str, float] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    error_message: str = ""


# Method dispatcher
_METHOD_MAP = {
    EstimationMethod.OLS.value: estimate_parametric,
    EstimationMethod.DECISION_TREE.value: estimate_nonparametric,
    EstimationMethod.XGBOOST.value: estimate_xgboost,
    EstimationMethod.RANDOM_FOREST.value: estimate_random_forest,
}


def run_single_experiment(
    spec: ExperimentSpec,
    data_registry: DataRegistry | None = None,
) -> ExperimentResult:
    """Execute a single IOp experiment from specification to logged result.

    Parameters
    ----------
    spec : ExperimentSpec
        The specification to run.
    data_registry : DataRegistry
        Data access object. Created fresh if None.

    Returns
    -------
    ExperimentResult
    """
    start = time.perf_counter()

    # 1. Validate spec
    validation_errors = spec.validate()
    if validation_errors:
        result = ExperimentResult(
            spec_id=spec.spec_id,
            status=ExperimentStatus.INVALID_SPEC.value,
            error_message="; ".join(validation_errors),
        )
        _log_result(spec, result)
        return result

    # 2. Load data
    try:
        if data_registry is None:
            data_registry = DataRegistry()

        data_errors = data_registry.validate_spec(spec)
        if data_errors:
            result = ExperimentResult(
                spec_id=spec.spec_id,
                status=ExperimentStatus.FAILED.value,
                error_message=f"Data validation: {'; '.join(data_errors)}",
            )
            _log_result(spec, result)
            return result

        y, X, valid_idx = data_registry.get_sample_for_spec(spec)
    except Exception as e:
        result = ExperimentResult(
            spec_id=spec.spec_id,
            status=ExperimentStatus.FAILED.value,
            error_message=f"Data loading error: {e}",
        )
        _log_result(spec, result)
        return result

    # 3. Handle income transformations for inequality calculation
    #    If income is log-transformed, we need levels for most inequality measures
    iv = IncomeVariable(spec.income_variable)
    if iv.is_log:
        # y is already log(income). For inequality measures needing positive values,
        # we compute IOp on the log scale directly.
        # This is valid for var_logs and captures relative inequality.
        y_for_ineq = y
    else:
        # Filter out non-positive values
        mask = y > 0
        if mask.sum() < 100:
            result = ExperimentResult(
                spec_id=spec.spec_id,
                status=ExperimentStatus.FAILED.value,
                error_message=f"Only {mask.sum()} positive income observations after filtering",
                n_obs=int(mask.sum()),
            )
            _log_result(spec, result)
            return result
        y = y[mask]
        X = X.loc[mask]
        y_for_ineq = y

    # 4. Get estimation function
    estimation_fn = _METHOD_MAP.get(spec.method)
    if estimation_fn is None:
        # Conditional forest: use nonparametric with different params
        if spec.method == EstimationMethod.CONDITIONAL_FOREST.value:
            estimation_fn = estimate_nonparametric
        else:
            result = ExperimentResult(
                spec_id=spec.spec_id,
                status=ExperimentStatus.FAILED.value,
                error_message=f"Unknown method: {spec.method}",
            )
            _log_result(spec, result)
            return result

    # 5. Run estimation with bootstrap CI
    try:
        iop_estimate = compute_iop_with_ci(
            y=y_for_ineq,
            X=X,
            estimation_fn=estimation_fn,
            measure=spec.inequality_measure,
            decomposition_type=spec.decomposition_type,
            method_params=spec.method_params_dict,
            bootstrap_n=spec.bootstrap_n,
            seed=spec.seed,
        )
    except Exception as e:
        result = ExperimentResult(
            spec_id=spec.spec_id,
            status=ExperimentStatus.FAILED.value,
            error_message=f"Estimation error: {e}",
            n_obs=len(y),
            runtime_seconds=time.perf_counter() - start,
        )
        _log_result(spec, result)
        return result

    # 6. Run diagnostics
    diag = run_diagnostics(
        iop_share=iop_estimate.iop_share,
        ci_lower=iop_estimate.iop_share_ci_lower,
        ci_upper=iop_estimate.iop_share_ci_upper,
        n_obs=iop_estimate.n_obs,
        n_types=iop_estimate.n_types,
        r_squared=iop_estimate.r_squared,
        cv_r_squared=iop_estimate.cv_r_squared,
        total_inequality=iop_estimate.total_inequality,
    )

    runtime = time.perf_counter() - start

    # 7. Build result
    result = ExperimentResult(
        spec_id=spec.spec_id,
        status=ExperimentStatus.SUCCESS.value,
        iop_share=iop_estimate.iop_share,
        iop_absolute=iop_estimate.iop_absolute,
        ci_lower=iop_estimate.iop_share_ci_lower,
        ci_upper=iop_estimate.iop_share_ci_upper,
        total_inequality=iop_estimate.total_inequality,
        r_squared=iop_estimate.r_squared,
        cv_r_squared=iop_estimate.cv_r_squared,
        n_obs=iop_estimate.n_obs,
        n_types=iop_estimate.n_types,
        variable_importance=iop_estimate.variable_importance,
        shap_importance=iop_estimate.shap_importance,
        diagnostics={
            "flags": diag.flags,
            "warnings": diag.warnings,
            "metrics": diag.metrics,
        },
        runtime_seconds=runtime,
    )

    # 8. Log
    _log_result(spec, result)

    # Print summary
    logger.info(
        f"Spec {spec.spec_id}: IOp={iop_estimate.iop_share:.4f} "
        f"[{iop_estimate.iop_share_ci_lower:.4f}, {iop_estimate.iop_share_ci_upper:.4f}] "
        f"n={iop_estimate.n_obs} | {runtime:.1f}s"
    )
    if diag.flags:
        logger.warning(f"  FLAGS: {diag.flags}")
    if diag.warnings:
        logger.info(f"  Warnings: {diag.warnings}")

    return result


def _log_result(spec: ExperimentSpec, result: ExperimentResult) -> None:
    """Merge spec and result into a single log record."""
    record = spec.to_dict()
    record.update(asdict(result))
    # Flatten diagnostics flags for TSV
    diag = record.get("diagnostics", {})
    record["flags"] = diag.get("flags", [])
    log_experiment(record)


def run_batch(
    specs: list[ExperimentSpec],
    data_registry: DataRegistry | None = None,
) -> list[ExperimentResult]:
    """Run a batch of experiments sequentially, skipping already-completed specs."""
    if data_registry is None:
        data_registry = DataRegistry()

    # Skip specs already completed successfully
    completed = get_completed_spec_ids()
    pending = [s for s in specs if s.spec_id not in completed]
    skipped = len(specs) - len(pending)
    if skipped:
        logger.info(f"Skipping {skipped} already-completed specs")

    results = []
    for i, spec in enumerate(pending, 1):
        logger.info(f"--- Experiment {i}/{len(pending)}: {spec} ---")
        result = run_single_experiment(spec, data_registry)
        results.append(result)

    # Summary
    success = sum(1 for r in results if r.status == ExperimentStatus.SUCCESS.value)
    failed = sum(1 for r in results if r.status == ExperimentStatus.FAILED.value)
    invalid = sum(1 for r in results if r.status == ExperimentStatus.INVALID_SPEC.value)
    logger.info(f"=== Batch complete: {success} success, {failed} failed, {invalid} invalid ===")

    return results


if __name__ == "__main__":
    # When called directly, run specs defined in analyze.py
    from analyze import get_specs

    specs = get_specs()
    if not specs:
        logger.info("No specs to run.")
        sys.exit(0)

    results = run_batch(specs)

    # Print summary table
    print("\n=== Results Summary ===")
    print(f"{'Spec ID':<14} {'Method':<12} {'Measure':<10} {'IOp Share':<10} {'CI':<20} {'Status':<10}")
    print("-" * 80)
    for r in results:
        if r.status == ExperimentStatus.SUCCESS.value:
            ci = f"[{r.ci_lower:.4f}, {r.ci_upper:.4f}]"
            print(f"{r.spec_id:<14} {'':<12} {'':<10} {r.iop_share:<10.4f} {ci:<20} {r.status:<10}")
        else:
            print(f"{r.spec_id:<14} {'':<12} {'':<10} {'N/A':<10} {'N/A':<20} {r.status:<10}")
