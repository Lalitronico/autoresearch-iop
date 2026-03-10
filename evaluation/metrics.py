"""IOp share calculation with bootstrap confidence intervals."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from core.decomposition import decompose_iop, IOpResult
from core.types import EstimationMethod

logger = logging.getLogger(__name__)


@dataclass
class IOpEstimate:
    """Full IOp estimate with confidence intervals."""
    iop_share: float
    iop_share_ci_lower: float
    iop_share_ci_upper: float
    iop_absolute: float
    total_inequality: float
    r_squared: float | None
    cv_r_squared: float | None
    n_obs: int
    n_types: int | None
    variable_importance: dict[str, float]
    shap_importance: dict[str, float] | None
    bootstrap_distribution: list[float]
    n_bootstrap_success: int


def compute_iop_with_ci(
    y: pd.Series,
    X: pd.DataFrame,
    estimation_fn: Callable,
    measure: str,
    decomposition_type: str,
    method_params: dict[str, Any],
    bootstrap_n: int = 200,
    seed: int = 42,
    confidence: float = 0.95,
) -> IOpEstimate:
    """Estimate IOp with bootstrap confidence intervals.

    Parameters
    ----------
    y : Series
        Income variable.
    X : DataFrame
        Circumstance variables.
    estimation_fn : Callable
        Function(y, X, params) -> result with y_predicted, type_labels.
    measure : str
        Inequality measure name.
    decomposition_type : str
        DecompositionType value.
    method_params : dict
        Parameters for the estimation method.
    bootstrap_n : int
        Number of bootstrap iterations.
    seed : int
        Random seed.
    confidence : float
        Confidence level (default 0.95 for 95% CI).

    Returns
    -------
    IOpEstimate
    """
    # Point estimate
    result = estimation_fn(y, X, method_params)
    iop_result = decompose_iop(
        y=y.values,
        y_predicted=result.y_predicted,
        type_labels=getattr(result, "type_labels", None),
        measure=measure,
        decomposition_type=decomposition_type,
    )

    # Bootstrap
    rng = np.random.default_rng(seed)
    n = len(y)
    boot_shares: list[float] = []
    n_boot_failed = 0

    for _ in range(bootstrap_n):
        idx = rng.choice(n, size=n, replace=True)
        y_boot = y.iloc[idx].reset_index(drop=True)
        X_boot = X.iloc[idx].reset_index(drop=True)

        try:
            boot_result = estimation_fn(y_boot, X_boot, method_params)
            boot_iop = decompose_iop(
                y=y_boot.values,
                y_predicted=boot_result.y_predicted,
                type_labels=getattr(boot_result, "type_labels", None),
                measure=measure,
                decomposition_type=decomposition_type,
            )
            boot_shares.append(boot_iop.iop_share)
        except Exception:
            n_boot_failed += 1
            continue  # Skip failed bootstrap iterations

    n_boot_success = len(boot_shares)
    if bootstrap_n > 0 and n_boot_failed / bootstrap_n > 0.10:
        logger.warning(
            f"Bootstrap: {n_boot_failed}/{bootstrap_n} iterations failed "
            f"({n_boot_failed / bootstrap_n:.1%}). "
            f"CI based on {n_boot_success} successful resamples."
        )

    # CI from bootstrap percentiles
    alpha = 1 - confidence
    if boot_shares:
        ci_lower = float(np.percentile(boot_shares, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_shares, 100 * (1 - alpha / 2)))
    else:
        ci_lower = ci_upper = iop_result.iop_share

    # Extract method-specific info
    r_squared = getattr(result, "r_squared", None)
    cv_r_squared = getattr(result, "cv_r_squared", None)
    n_types = getattr(result, "n_types", None)
    if n_types is None:
        n_types = iop_result.method_details.get("n_types")
    var_importance = getattr(result, "feature_importance", {})
    if isinstance(var_importance, (dict,)):
        pass
    elif hasattr(result, "coefficients"):
        var_importance = result.coefficients
    else:
        var_importance = {}
    shap_imp = getattr(result, "shap_importance", None)

    return IOpEstimate(
        iop_share=iop_result.iop_share,
        iop_share_ci_lower=ci_lower,
        iop_share_ci_upper=ci_upper,
        iop_absolute=iop_result.iop_absolute,
        total_inequality=iop_result.total_inequality,
        r_squared=r_squared,
        cv_r_squared=cv_r_squared,
        n_obs=len(y),
        n_types=n_types,
        variable_importance=var_importance,
        shap_importance=shap_imp,
        bootstrap_distribution=boot_shares,
        n_bootstrap_success=n_boot_success,
    )
