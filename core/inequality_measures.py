"""Inequality measures implemented from scratch.

All functions accept a 1-D array of incomes and optional sample weights.
Incomes must be strictly positive for most measures (enforced at call sites).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _validate_income(
    y: NDArray[np.floating], weights: NDArray[np.floating] | None, require_positive: bool = True
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Common input validation."""
    y = np.asarray(y, dtype=np.float64).ravel()
    if y.size < 2:
        raise ValueError("Need at least 2 observations")
    if weights is None:
        w = np.ones_like(y)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.shape != y.shape:
            raise ValueError("weights must have same shape as y")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
    if require_positive and np.any(y <= 0):
        raise ValueError("Income must be strictly positive for this measure")
    return y, w


def gini(y: NDArray, weights: NDArray | None = None) -> float:
    """Gini coefficient (0 = perfect equality, 1 = perfect inequality).

    Weighted version using the covariance formula:
        G = 2 * cov(y, F(y)) / mean(y)
    where F(y) is the cumulative distribution.
    """
    y, w = _validate_income(y, weights, require_positive=False)
    # Sort by income
    order = np.argsort(y)
    y_sorted = y[order]
    w_sorted = w[order]
    # Cumulative weight (for CDF)
    cum_w = np.cumsum(w_sorted)
    total_w = cum_w[-1]
    # Rank as midpoint of cumulative weight (for ties handling)
    F = (cum_w - w_sorted / 2) / total_w
    # Weighted mean
    mu = np.average(y_sorted, weights=w_sorted)
    if mu == 0:
        return 0.0
    # Covariance formula
    cov_yF = np.average((y_sorted - mu) * (F - 0.5), weights=w_sorted)
    return float(2 * cov_yF / mu)


def mld(y: NDArray, weights: NDArray | None = None) -> float:
    """Mean Log Deviation (GE(0) / Theil-L).

    MLD = (1/N) * sum( log(mu/y_i) ) = log(mu) - mean(log(y))
    """
    y, w = _validate_income(y, weights, require_positive=True)
    mu = np.average(y, weights=w)
    return float(np.log(mu) - np.average(np.log(y), weights=w))


def theil_t(y: NDArray, weights: NDArray | None = None) -> float:
    """Theil-T index (GE(1)).

    T = (1/N) * sum( (y_i/mu) * log(y_i/mu) )
    """
    y, w = _validate_income(y, weights, require_positive=True)
    mu = np.average(y, weights=w)
    ratios = y / mu
    return float(np.average(ratios * np.log(ratios), weights=w))


def var_logs(y: NDArray, weights: NDArray | None = None) -> float:
    """Variance of logarithms (half-squaredCV of logs).

    V = Var(log(y))
    """
    y, w = _validate_income(y, weights, require_positive=True)
    log_y = np.log(y)
    mu_log = np.average(log_y, weights=w)
    return float(np.average((log_y - mu_log) ** 2, weights=w))


def atkinson(y: NDArray, epsilon: float = 1.0, weights: NDArray | None = None) -> float:
    """Atkinson index with inequality aversion parameter epsilon.

    For epsilon == 1:
        A = 1 - exp(mean(log(y))) / mean(y)  (geometric mean / arithmetic mean)
    For epsilon != 1:
        A = 1 - ( mean(y^(1-eps)) )^(1/(1-eps)) / mean(y)
    """
    if epsilon < 0:
        raise ValueError("epsilon must be >= 0")
    y, w = _validate_income(y, weights, require_positive=True)
    mu = np.average(y, weights=w)
    if mu == 0:
        return 0.0
    if abs(epsilon - 1.0) < 1e-10:
        # Geometric mean / arithmetic mean
        log_mean = np.average(np.log(y), weights=w)
        ede = np.exp(log_mean)
    else:
        exponent = 1.0 - epsilon
        power_mean = np.average(y**exponent, weights=w)
        ede = power_mean ** (1.0 / exponent)
    return float(1.0 - ede / mu)


# --- Dispatcher ---

_MEASURE_REGISTRY: dict[str, callable] = {
    "gini": gini,
    "mld": mld,
    "theil_t": theil_t,
    "var_logs": var_logs,
    "atkinson_0.5": lambda y, w=None: atkinson(y, epsilon=0.5, weights=w),
    "atkinson_1": lambda y, w=None: atkinson(y, epsilon=1.0, weights=w),
    "atkinson_2": lambda y, w=None: atkinson(y, epsilon=2.0, weights=w),
}


def compute_inequality(
    y: NDArray, measure: str, weights: NDArray | None = None
) -> float:
    """Dispatch to the appropriate inequality measure by name."""
    if measure not in _MEASURE_REGISTRY:
        raise ValueError(f"Unknown measure: {measure}. Available: {list(_MEASURE_REGISTRY)}")
    return _MEASURE_REGISTRY[measure](y, weights)
