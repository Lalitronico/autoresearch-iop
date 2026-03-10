"""Diagnostics and validation for IOp experiments.

Checks sample size, CI width, overfitting, and other quality flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DiagnosticResult:
    """Collection of diagnostic flags for an experiment."""
    flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def has_critical_flags(self) -> bool:
        return len(self.flags) > 0

    def add_flag(self, msg: str) -> None:
        self.flags.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


def run_diagnostics(
    iop_share: float,
    ci_lower: float,
    ci_upper: float,
    n_obs: int,
    n_types: int | None,
    r_squared: float | None,
    cv_r_squared: float | None,
    total_inequality: float,
) -> DiagnosticResult:
    """Run all diagnostic checks on an IOp estimate.

    Returns DiagnosticResult with flags (critical) and warnings (informational).
    """
    diag = DiagnosticResult()

    # --- Sample size guards ---
    if n_obs < 100:
        diag.add_flag(f"Sample size too small: {n_obs} < 100")
    elif n_obs < 500:
        diag.add_warning(f"Small sample size: {n_obs} < 500")

    # --- CI width ---
    ci_width = ci_upper - ci_lower
    diag.metrics["ci_width"] = ci_width
    if ci_width > 0.3:
        diag.add_flag(f"CI too wide: {ci_width:.3f} > 0.3")
    elif ci_width > 0.15:
        diag.add_warning(f"Wide CI: {ci_width:.3f} > 0.15")

    # --- IOp share bounds ---
    if iop_share < 0 or iop_share > 1:
        diag.add_flag(f"IOp share out of [0,1]: {iop_share:.4f}")
    if iop_share > 0.8:
        diag.add_warning(f"Very high IOp share: {iop_share:.4f}")

    # --- Types guard ---
    if n_types is not None:
        diag.metrics["n_types"] = n_types
        if n_types < 2:
            diag.add_flag(f"Only {n_types} type(s) -- no meaningful partition")
        if n_types > 0 and n_obs / n_types < 5:
            diag.add_warning(
                f"Sparse types: {n_obs} obs / {n_types} types = "
                f"{n_obs / n_types:.1f} obs/type"
            )

    # --- Overfitting detection ---
    if r_squared is not None and cv_r_squared is not None:
        overfit_gap = r_squared - cv_r_squared
        diag.metrics["overfit_gap"] = overfit_gap
        if overfit_gap > 0.10:
            diag.add_flag(f"Overfitting detected: R^2={r_squared:.3f} vs CV R^2={cv_r_squared:.3f}")
        elif overfit_gap > 0.05:
            diag.add_warning(f"Possible overfitting: gap={overfit_gap:.3f}")

    # --- Total inequality near zero ---
    if total_inequality < 0.01:
        diag.add_flag(f"Total inequality near zero: {total_inequality:.4f}")

    return diag
