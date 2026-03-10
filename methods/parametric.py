"""Ferreira-Gignoux parametric IOp estimation via OLS.

Regresses income on circumstances, uses R^2 as lower bound IOp share,
and I(y_hat)/I(y) for measure-specific lower bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.typing import NDArray


@dataclass
class ParametricResult:
    """Output from parametric IOp estimation."""
    y_predicted: NDArray[np.floating]
    r_squared: float
    coefficients: dict[str, float]
    n_obs: int
    type_labels: None = None  # Parametric doesn't produce types


def estimate_parametric(
    y: pd.Series,
    X: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> ParametricResult:
    """OLS regression of income on circumstances.

    Parameters
    ----------
    y : Series
        Income (or log-income).
    X : DataFrame
        Circumstance variables (will be dummy-encoded if categorical).
    params : dict
        Optional parameters (currently unused, for future extensions).

    Returns
    -------
    ParametricResult with predictions, R^2, and coefficients.
    """
    # Prepare design matrix with dummies for categoricals
    X_encoded = pd.get_dummies(X, drop_first=True, dtype=float)
    X_const = sm.add_constant(X_encoded, has_constant="add")

    model = sm.OLS(y.values, X_const.values).fit()

    y_hat = model.fittedvalues

    # For measure-specific IOp, we need predicted values in levels
    # If y was log, y_hat is log(predicted). We exponentiate for inequality calc
    # But this is handled in run_experiment.py based on income_variable

    coefs = {}
    for name, val in zip(X_const.columns if hasattr(X_const, 'columns') else range(len(model.params)), model.params):
        coefs[str(name)] = float(val)

    return ParametricResult(
        y_predicted=y_hat,
        r_squared=float(model.rsquared),
        coefficients=coefs,
        n_obs=int(model.nobs),
    )
