"""IOp share calculation with bootstrap confidence intervals.

Includes Rubin's rules for pooling across multiply-imputed datasets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats as scipy_stats

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


# --- Multiple Imputation: Rubin's Rules ---


@dataclass
class MIPooledEstimate:
    """IOp estimate pooled across multiply-imputed datasets via Rubin's rules."""
    iop_share: float              # Q_bar (pooled point estimate)
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
    # MI diagnostics
    within_variance: float        # U_bar
    between_variance: float       # B
    total_variance: float         # T = U_bar + (1+1/M)*B
    fraction_missing_info: float  # lambda = (B + B/M) / T
    n_imputations: int
    per_imputation_estimates: list[float]  # Q_m for each m
    n_bootstrap_success: int


def pool_rubin(
    estimates: list[float],
    variances: list[float],
    confidence: float = 0.95,
) -> dict[str, float]:
    """Pool estimates across M imputations using Rubin's rules.

    Parameters
    ----------
    estimates : list[float]
        Point estimates Q_m from each imputed dataset.
    variances : list[float]
        Within-imputation variance U_m from each dataset
        (e.g., bootstrap variance of Q_m).
    confidence : float
        Confidence level for CI.

    Returns
    -------
    dict with keys: q_bar, u_bar, b, t, df, ci_lower, ci_upper,
                    fraction_missing_info
    """
    m = len(estimates)
    if m < 2:
        q = estimates[0] if estimates else 0.0
        u = variances[0] if variances else 0.0
        return {
            "q_bar": q,
            "u_bar": u,
            "b": 0.0,
            "t": u,
            "df": float("inf"),
            "ci_lower": q - 1.96 * np.sqrt(u),
            "ci_upper": q + 1.96 * np.sqrt(u),
            "fraction_missing_info": 0.0,
        }

    Q = np.array(estimates)
    U = np.array(variances)

    # Pooled point estimate
    q_bar = float(Q.mean())

    # Within-imputation variance (average of within-variances)
    u_bar = float(U.mean())

    # Between-imputation variance
    b = float(Q.var(ddof=1))

    # Total variance
    t = u_bar + (1 + 1 / m) * b

    # Barnard-Rubin degrees of freedom
    if b > 0 and t > 0:
        r = (1 + 1 / m) * b / u_bar if u_bar > 0 else float("inf")
        df_old = (m - 1) * (1 + 1 / r) ** 2 if r > 0 else float("inf")
        # Barnard-Rubin adjustment (small-sample)
        df = df_old
    else:
        df = float("inf")

    # Fraction of missing information
    if t > 0:
        fmi = ((1 + 1 / m) * b) / t
    else:
        fmi = 0.0

    # Confidence interval
    alpha = 1 - confidence
    if np.isfinite(df) and df > 0:
        t_crit = scipy_stats.t.ppf(1 - alpha / 2, df)
    else:
        t_crit = scipy_stats.norm.ppf(1 - alpha / 2)

    se = np.sqrt(max(t, 0))
    ci_lower = q_bar - t_crit * se
    ci_upper = q_bar + t_crit * se

    return {
        "q_bar": q_bar,
        "u_bar": u_bar,
        "b": b,
        "t": t,
        "df": df,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "fraction_missing_info": fmi,
    }


def compute_iop_with_ci_mi(
    data_registry,
    spec,
    estimation_fn: Callable,
    measure: str,
    decomposition_type: str,
    method_params: dict[str, Any],
    m_total: int = 20,
    bootstrap_n: int = 100,
    seed: int = 42,
    confidence: float = 0.95,
) -> MIPooledEstimate:
    """Compute IOp with MI pooling via Rubin's rules.

    For each imputation m:
      1. Get (y_m, X_m) from imputed dataset m
      2. Point estimate Q_m via estimation_fn + decompose_iop
      3. Bootstrap B times -> within-variance U_m
    Then pool across M with pool_rubin().
    """
    rng = np.random.default_rng(seed)
    per_imp_estimates: list[float] = []
    per_imp_variances: list[float] = []
    per_imp_absolutes: list[float] = []
    per_imp_totals: list[float] = []
    per_imp_r2: list[float | None] = []
    per_imp_cv_r2: list[float | None] = []
    per_imp_n_types: list[int | None] = []
    per_imp_var_imp: list[dict] = []
    per_imp_shap: list[dict | None] = []
    total_boot_success = 0

    n_obs_first = None

    for m in range(m_total):
        y_m, X_m, idx_m = data_registry.get_sample_for_spec_mi(spec, m)

        if n_obs_first is None:
            n_obs_first = len(y_m)

        # Point estimate
        result_m = estimation_fn(y_m, X_m, method_params)
        iop_m = decompose_iop(
            y=y_m.values,
            y_predicted=result_m.y_predicted,
            type_labels=getattr(result_m, "type_labels", None),
            measure=measure,
            decomposition_type=decomposition_type,
        )

        per_imp_estimates.append(iop_m.iop_share)
        per_imp_absolutes.append(iop_m.iop_absolute)
        per_imp_totals.append(iop_m.total_inequality)
        per_imp_r2.append(getattr(result_m, "r_squared", None))
        per_imp_cv_r2.append(getattr(result_m, "cv_r_squared", None))
        per_imp_n_types.append(
            getattr(result_m, "n_types", None) or iop_m.method_details.get("n_types")
        )

        var_importance = getattr(result_m, "feature_importance", {})
        if not isinstance(var_importance, dict):
            var_importance = getattr(result_m, "coefficients", {})
        per_imp_var_imp.append(var_importance if isinstance(var_importance, dict) else {})
        per_imp_shap.append(getattr(result_m, "shap_importance", None))

        # Bootstrap within this imputation
        n = len(y_m)
        boot_shares: list[float] = []
        imp_seed = rng.integers(0, 2**31)
        boot_rng = np.random.default_rng(imp_seed)

        for _ in range(bootstrap_n):
            idx = boot_rng.choice(n, size=n, replace=True)
            y_boot = y_m.iloc[idx].reset_index(drop=True)
            X_boot = X_m.iloc[idx].reset_index(drop=True)
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
                continue

        total_boot_success += len(boot_shares)
        # Within-imputation variance = var of bootstrap distribution
        u_m = float(np.var(boot_shares, ddof=1)) if len(boot_shares) > 1 else 0.0
        per_imp_variances.append(u_m)

        logger.info(
            f"  MI dataset {m}/{m_total}: IOp={iop_m.iop_share:.4f}, "
            f"boot_var={u_m:.6f}, n={len(y_m)}"
        )

    # Pool using Rubin's rules
    pooled = pool_rubin(per_imp_estimates, per_imp_variances, confidence)

    # Average auxiliary quantities across imputations
    avg_absolute = float(np.mean(per_imp_absolutes))
    avg_total = float(np.mean(per_imp_totals))

    # Average R-squared (ignoring None)
    r2_vals = [v for v in per_imp_r2 if v is not None]
    avg_r2 = float(np.mean(r2_vals)) if r2_vals else None
    cv_r2_vals = [v for v in per_imp_cv_r2 if v is not None]
    avg_cv_r2 = float(np.mean(cv_r2_vals)) if cv_r2_vals else None

    # Average n_types
    nt_vals = [v for v in per_imp_n_types if v is not None]
    avg_n_types = int(np.mean(nt_vals)) if nt_vals else None

    # Average variable importance
    avg_var_imp: dict[str, float] = {}
    if per_imp_var_imp:
        all_keys = set()
        for d in per_imp_var_imp:
            all_keys.update(d.keys())
        for k in all_keys:
            vals = [d.get(k, 0.0) for d in per_imp_var_imp]
            avg_var_imp[k] = float(np.mean(vals))

    # Average SHAP importance
    avg_shap: dict[str, float] | None = None
    shap_dicts = [s for s in per_imp_shap if s is not None]
    if shap_dicts:
        avg_shap = {}
        all_keys = set()
        for d in shap_dicts:
            all_keys.update(d.keys())
        for k in all_keys:
            vals = [d.get(k, 0.0) for d in shap_dicts]
            avg_shap[k] = float(np.mean(vals))

    return MIPooledEstimate(
        iop_share=pooled["q_bar"],
        iop_share_ci_lower=pooled["ci_lower"],
        iop_share_ci_upper=pooled["ci_upper"],
        iop_absolute=avg_absolute,
        total_inequality=avg_total,
        r_squared=avg_r2,
        cv_r_squared=avg_cv_r2,
        n_obs=n_obs_first or 0,
        n_types=avg_n_types,
        variable_importance=avg_var_imp,
        shap_importance=avg_shap,
        within_variance=pooled["u_bar"],
        between_variance=pooled["b"],
        total_variance=pooled["t"],
        fraction_missing_info=pooled["fraction_missing_info"],
        n_imputations=m_total,
        per_imputation_estimates=per_imp_estimates,
        n_bootstrap_success=total_boot_success,
    )
