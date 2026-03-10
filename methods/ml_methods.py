"""ML-based IOp estimation: XGBoost, Random Forest, with SHAP importance.

These methods capture non-linear relationships between circumstances
and income, potentially providing tighter IOp bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import OrdinalEncoder

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


@dataclass
class MLResult:
    """Output from ML-based IOp estimation."""
    y_predicted: NDArray[np.floating]
    type_labels: NDArray[np.integer] | None
    feature_importance: dict[str, float]
    shap_importance: dict[str, float] | None
    cv_r_squared: float
    n_obs: int
    method_name: str


def estimate_xgboost(
    y: pd.Series,
    X: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> MLResult:
    """XGBoost prediction of income from circumstances."""
    if not HAS_XGBOOST:
        raise ImportError("xgboost is required for this method")

    params = params or {}
    xgb_params = {
        "n_estimators": params.get("n_estimators", 100),
        "max_depth": params.get("max_depth", 4),
        "learning_rate": params.get("learning_rate", 0.1),
        "subsample": params.get("subsample", 0.8),
        "colsample_bytree": params.get("colsample_bytree", 0.8),
        "random_state": params.get("seed", 42),
        "n_jobs": -1,
    }

    X_encoded, feature_names = _encode_features(X)

    model = xgb.XGBRegressor(**xgb_params)

    # Cross-validated predictions to avoid overfitting bias
    y_pred_cv = cross_val_predict(model, X_encoded, y.values, cv=5)

    # Fit full model for importance
    model.fit(X_encoded, y.values)
    y_pred_full = model.predict(X_encoded)

    # Feature importance
    importance = {
        name: float(imp)
        for name, imp in zip(feature_names, model.feature_importances_)
    }

    # SHAP importance (if available)
    shap_imp = _compute_shap(model, X_encoded, feature_names)

    # CV R^2
    ss_res = np.sum((y.values - y_pred_cv) ** 2)
    ss_tot = np.sum((y.values - y.values.mean()) ** 2)
    cv_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Use leaf assignments as type labels
    type_labels = model.apply(X_encoded)
    # XGBoost apply returns matrix (n_obs, n_trees) -- use first tree for simplicity
    if type_labels.ndim > 1:
        type_labels = type_labels[:, 0]

    return MLResult(
        y_predicted=y_pred_cv,  # Use CV predictions for IOp to avoid overfit
        type_labels=type_labels.astype(np.int64),
        feature_importance=importance,
        shap_importance=shap_imp,
        cv_r_squared=float(cv_r2),
        n_obs=len(y),
        method_name="xgboost",
    )


def estimate_random_forest(
    y: pd.Series,
    X: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> MLResult:
    """Random Forest prediction of income from circumstances."""
    params = params or {}
    rf_params = {
        "n_estimators": params.get("n_estimators", 200),
        "max_depth": params.get("max_depth", 6),
        "min_samples_leaf": params.get("min_samples_leaf", 20),
        "random_state": params.get("seed", 42),
        "n_jobs": -1,
    }

    X_encoded, feature_names = _encode_features(X)

    model = RandomForestRegressor(**rf_params)

    # Cross-validated predictions
    y_pred_cv = cross_val_predict(model, X_encoded, y.values, cv=5)

    # Fit full model for importance
    model.fit(X_encoded, y.values)

    importance = {
        name: float(imp)
        for name, imp in zip(feature_names, model.feature_importances_)
    }

    shap_imp = _compute_shap(model, X_encoded, feature_names)

    ss_res = np.sum((y.values - y_pred_cv) ** 2)
    ss_tot = np.sum((y.values - y.values.mean()) ** 2)
    cv_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Leaf assignments from first tree
    type_labels = model.estimators_[0].apply(X_encoded).astype(np.int64)

    return MLResult(
        y_predicted=y_pred_cv,
        type_labels=type_labels,
        feature_importance=importance,
        shap_importance=shap_imp,
        cv_r_squared=float(cv_r2),
        n_obs=len(y),
        method_name="random_forest",
    )


def _encode_features(X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Ordinal-encode categorical features."""
    X_out = X.copy()
    for col in X_out.columns:
        if X_out[col].dtype == "object" or X_out[col].dtype.name == "category":
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_out[col] = enc.fit_transform(X_out[[col]]).ravel()
    return X_out.values.astype(np.float64), list(X_out.columns)


def _compute_shap(
    model, X: np.ndarray, feature_names: list[str]
) -> dict[str, float] | None:
    """Compute mean absolute SHAP values for feature importance."""
    if not HAS_SHAP:
        return None
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        mean_abs = np.abs(shap_values).mean(axis=0)
        return {
            name: float(val) for name, val in zip(feature_names, mean_abs)
        }
    except Exception:
        return None
