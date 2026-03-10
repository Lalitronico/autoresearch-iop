"""IOp decomposition logic.

Computes Inequality of Opportunity shares by comparing
total inequality with between-group (type) inequality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from core.inequality_measures import compute_inequality
from core.types import DecompositionType


@dataclass
class IOpResult:
    """Result of an IOp decomposition."""
    iop_share: float           # IOp / I(y), bounded [0, 1]
    iop_absolute: float        # Absolute IOp value
    total_inequality: float    # I(y)
    method_details: dict[str, Any]  # Method-specific info (R^2, n_types, etc.)


def decompose_iop(
    y: NDArray[np.floating],
    y_predicted: NDArray[np.floating],
    type_labels: NDArray[np.integer] | None,
    measure: str,
    decomposition_type: str,
    weights: NDArray[np.floating] | None = None,
) -> IOpResult:
    """Compute IOp share given actual incomes, predictions, and type labels.

    Parameters
    ----------
    y : array
        Actual incomes (strictly positive for most measures).
    y_predicted : array
        Predicted incomes from circumstances (smoothed means).
    type_labels : array or None
        Group assignments for non-parametric methods. None for parametric.
    measure : str
        Inequality measure name (from InequalityMeasure enum values).
    decomposition_type : str
        DecompositionType enum value.
    weights : array or None
        Sample weights.

    Returns
    -------
    IOpResult
    """
    dt = DecompositionType(decomposition_type)
    total_ineq = compute_inequality(y, measure, weights)

    if total_ineq <= 0:
        return IOpResult(
            iop_share=0.0,
            iop_absolute=0.0,
            total_inequality=total_ineq,
            method_details={"note": "Total inequality is zero or negative"},
        )

    if dt == DecompositionType.LOWER_BOUND:
        return _lower_bound(y, y_predicted, measure, total_ineq, weights)
    elif dt == DecompositionType.EX_ANTE:
        return _ex_ante(y, y_predicted, type_labels, measure, total_ineq, weights)
    elif dt == DecompositionType.EX_POST:
        return _ex_post(y, type_labels, measure, total_ineq, weights)
    elif dt == DecompositionType.UPPER_BOUND:
        return _upper_bound(y, type_labels, measure, total_ineq, weights)
    else:
        raise ValueError(f"Unsupported decomposition type: {dt}")


def _lower_bound(
    y: NDArray, y_hat: NDArray, measure: str, total_ineq: float, weights: NDArray | None
) -> IOpResult:
    """Parametric lower bound (Ferreira-Gignoux):
    IOp = I(y_hat) / I(y), where y_hat = predicted from OLS.
    """
    iop_abs = compute_inequality(y_hat, measure, weights)
    share = min(iop_abs / total_ineq, 1.0)  # Cap at 1
    return IOpResult(
        iop_share=share,
        iop_absolute=iop_abs,
        total_inequality=total_ineq,
        method_details={"type": "lower_bound"},
    )


def _ex_ante(
    y: NDArray, y_predicted: NDArray, type_labels: NDArray | None,
    measure: str, total_ineq: float, weights: NDArray | None,
) -> IOpResult:
    """Ex-ante IOp: I(predicted_means) / I(y).

    If type_labels are provided, uses group means. Otherwise uses y_predicted directly.
    """
    if type_labels is not None:
        # Compute type means
        unique_types = np.unique(type_labels)
        type_means = np.empty_like(y)
        for t in unique_types:
            mask = type_labels == t
            if weights is not None:
                type_means[mask] = np.average(y[mask], weights=weights[mask])
            else:
                type_means[mask] = y[mask].mean()
        smoothed = type_means
        n_types = len(unique_types)
    else:
        smoothed = y_predicted
        n_types = None

    iop_abs = compute_inequality(smoothed, measure, weights)
    share = min(max(iop_abs / total_ineq, 0.0), 1.0)

    details = {"type": "ex_ante"}
    if n_types is not None:
        details["n_types"] = n_types

    return IOpResult(
        iop_share=share,
        iop_absolute=iop_abs,
        total_inequality=total_ineq,
        method_details=details,
    )


def _ex_post(
    y: NDArray, type_labels: NDArray, measure: str, total_ineq: float,
    weights: NDArray | None,
) -> IOpResult:
    """Ex-post IOp: weighted sum of within-type inequality.

    IOp_within = sum_k (n_k/N) * I(y_k)
    IOp_share = 1 - IOp_within / I(y)   [for path-independent measures like MLD]

    Note: This interpretation only cleanly decomposes for GE measures (MLD, Theil-T).
    """
    if type_labels is None:
        raise ValueError("Ex-post decomposition requires type_labels")

    unique_types = np.unique(type_labels)
    within_sum = 0.0
    total_w = weights.sum() if weights is not None else len(y)

    for t in unique_types:
        mask = type_labels == t
        y_k = y[mask]
        if len(y_k) < 2:
            continue
        w_k = weights[mask] if weights is not None else None
        group_w = w_k.sum() if w_k is not None else len(y_k)
        ineq_k = compute_inequality(y_k, measure, w_k)
        within_sum += (group_w / total_w) * ineq_k

    # Between = Total - Within (exact for GE measures)
    between = total_ineq - within_sum
    share = min(max(between / total_ineq, 0.0), 1.0)

    return IOpResult(
        iop_share=share,
        iop_absolute=between,
        total_inequality=total_ineq,
        method_details={
            "type": "ex_post",
            "within_inequality": within_sum,
            "n_types": len(unique_types),
        },
    )


def _upper_bound(
    y: NDArray, type_labels: NDArray, measure: str, total_ineq: float,
    weights: NDArray | None,
) -> IOpResult:
    """Upper bound: Total - within-group inequality.

    Same as ex-post for additive measures. Explicitly labeled for clarity.
    """
    return _ex_post(y, type_labels, measure, total_ineq, weights)
