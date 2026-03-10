"""Non-parametric IOp estimation using decision tree partitioning.

Uses sklearn DecisionTreeRegressor as a proxy for conditional inference
trees (ctree). Creates "types" by partitioning individuals into groups
with identical circumstance profiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OrdinalEncoder


@dataclass
class NonParametricResult:
    """Output from non-parametric IOp estimation."""
    y_predicted: NDArray[np.floating]  # Type means assigned to each obs
    type_labels: NDArray[np.integer]   # Group assignment
    n_types: int
    n_obs: int
    tree_depth: int
    feature_importance: dict[str, float]


def estimate_nonparametric(
    y: pd.Series,
    X: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> NonParametricResult:
    """Decision tree partitioning of income by circumstances.

    Parameters
    ----------
    y : Series
        Income variable.
    X : DataFrame
        Circumstance variables.
    params : dict
        Tree hyperparameters:
        - max_depth: int (default 4)
        - min_samples_leaf: int (default 30)
        - ccp_alpha: float for cost-complexity pruning (default 0.01)

    Returns
    -------
    NonParametricResult
    """
    params = params or {}
    max_depth = params.get("max_depth", 4)
    min_samples_leaf = params.get("min_samples_leaf", 30)
    ccp_alpha = params.get("ccp_alpha", 0.01)

    # Encode categoricals
    X_encoded, feature_names = _encode_features(X)

    tree = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        ccp_alpha=ccp_alpha,
        random_state=42,
    )
    tree.fit(X_encoded, y.values)

    # Leaf node IDs = type labels
    type_labels = tree.apply(X_encoded)

    # Predicted values = leaf means (type means)
    y_predicted = tree.predict(X_encoded)

    # Feature importance
    importance = {}
    for name, imp in zip(feature_names, tree.feature_importances_):
        importance[name] = float(imp)

    # Map type labels to contiguous integers
    unique_types = np.unique(type_labels)
    label_map = {old: new for new, old in enumerate(unique_types)}
    type_labels_clean = np.array([label_map[t] for t in type_labels])

    return NonParametricResult(
        y_predicted=y_predicted,
        type_labels=type_labels_clean,
        n_types=len(unique_types),
        n_obs=len(y),
        tree_depth=tree.get_depth(),
        feature_importance=importance,
    )


def _encode_features(X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Encode categorical features for tree fitting.

    Trees handle ordinal encoding well (splits are binary anyway).
    """
    X_out = X.copy()
    for col in X_out.columns:
        if X_out[col].dtype == "object" or X_out[col].dtype.name == "category":
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_out[col] = enc.fit_transform(X_out[[col]]).ravel()
    return X_out.values.astype(np.float64), list(X_out.columns)
