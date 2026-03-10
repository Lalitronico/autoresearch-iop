"""ExperimentSpec dataclass: defines a single point in the specification space."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any

from core.types import (
    Circumstance,
    DecompositionType,
    EstimationMethod,
    InequalityMeasure,
    IncomeVariable,
    SampleFilter,
)


@dataclass(frozen=True)
class ExperimentSpec:
    """A single specification in the IOp multiverse analysis.

    Each instance represents one combination of methodological choices
    that produces an IOp estimate.
    """

    circumstances: tuple[str, ...]
    income_variable: str
    inequality_measure: str
    method: str
    decomposition_type: str
    sample_filter: str = SampleFilter.ALL.value
    method_params: tuple[tuple[str, Any], ...] = ()
    seed: int = 42
    bootstrap_n: int = 200
    rationale: str = ""

    def __post_init__(self) -> None:
        # Normalize circumstances to sorted tuple for consistency
        if isinstance(self.circumstances, list):
            object.__setattr__(self, "circumstances", tuple(sorted(self.circumstances)))
        elif isinstance(self.circumstances, tuple):
            object.__setattr__(self, "circumstances", tuple(sorted(self.circumstances)))

    @property
    def spec_id(self) -> str:
        """Deterministic hash of the spec (excluding rationale and bootstrap_n)."""
        key_fields = {
            "circumstances": list(self.circumstances),
            "income_variable": self.income_variable,
            "inequality_measure": self.inequality_measure,
            "method": self.method,
            "decomposition_type": self.decomposition_type,
            "sample_filter": self.sample_filter,
            "method_params": dict(self.method_params),
            "seed": self.seed,
        }
        raw = json.dumps(key_fields, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    @property
    def method_params_dict(self) -> dict[str, Any]:
        return dict(self.method_params)

    @staticmethod
    def recommended_bootstrap_n(method: str) -> int:
        """Recommended bootstrap iterations by method runtime.

        OLS/DT are fast (~1-2s/spec), so 200 iterations is fine.
        XGBoost/RF are expensive (~3-10min/spec), so fewer iterations.
        """
        fast_methods = {EstimationMethod.OLS.value, EstimationMethod.DECISION_TREE.value}
        if method in fast_methods:
            return 200
        return 50

    def validate(self) -> list[str]:
        """Validate domain rules. Returns list of error messages (empty = valid)."""
        errors: list[str] = []

        # --- Circumstances ---
        valid_circs = {c.value for c in Circumstance}
        for c in self.circumstances:
            if c not in valid_circs:
                errors.append(f"Unknown circumstance: {c}")
        if len(self.circumstances) == 0:
            errors.append("At least one circumstance is required")

        # --- Income variable ---
        try:
            iv = IncomeVariable(self.income_variable)
        except ValueError:
            errors.append(f"Unknown income variable: {self.income_variable}")
            iv = None

        # --- Inequality measure ---
        try:
            im = InequalityMeasure(self.inequality_measure)
        except ValueError:
            errors.append(f"Unknown inequality measure: {self.inequality_measure}")
            im = None

        # --- Method ---
        try:
            EstimationMethod(self.method)
        except ValueError:
            errors.append(f"Unknown estimation method: {self.method}")

        # --- Decomposition ---
        try:
            dt = DecompositionType(self.decomposition_type)
        except ValueError:
            errors.append(f"Unknown decomposition type: {self.decomposition_type}")
            dt = None

        # --- Sample filter ---
        try:
            SampleFilter(self.sample_filter)
        except ValueError:
            errors.append(f"Unknown sample filter: {self.sample_filter}")

        # --- Domain rules ---
        # No double-log: if income is already log, var_logs is redundant
        if iv and im:
            if iv.is_log and im == InequalityMeasure.VAR_LOGS:
                errors.append(
                    "var_logs on log-income is double-log transformation; use levels instead"
                )

        # Ex-post requires partition-based method (not OLS)
        if dt == DecompositionType.EX_POST:
            if self.method == EstimationMethod.OLS.value:
                errors.append("Ex-post decomposition requires a partition method, not OLS")

        # Lower bound is specific to parametric
        if dt == DecompositionType.LOWER_BOUND:
            if self.method != EstimationMethod.OLS.value:
                errors.append("Lower bound decomposition is only defined for OLS (parametric)")

        # MLD and Theil require strictly positive incomes
        if im in (InequalityMeasure.MLD, InequalityMeasure.THEIL_T):
            if iv and iv.is_log:
                pass  # log incomes can be negative, but exp(log_inc) is positive -- handled at runtime

        # Bootstrap must be positive
        if self.bootstrap_n < 1:
            errors.append("bootstrap_n must be >= 1")

        return errors

    @property
    def is_valid(self) -> bool:
        return len(self.validate()) == 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["spec_id"] = self.spec_id
        d["method_params"] = dict(self.method_params)
        d["circumstances"] = list(self.circumstances)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentSpec:
        d = d.copy()
        d.pop("spec_id", None)
        if isinstance(d.get("method_params"), dict):
            d["method_params"] = tuple(d["method_params"].items())
        if isinstance(d.get("circumstances"), list):
            d["circumstances"] = tuple(d["circumstances"])
        return cls(**d)

    def __str__(self) -> str:
        circs = ", ".join(self.circumstances)
        return (
            f"Spec[{self.spec_id}]: {self.method}/{self.decomposition_type} | "
            f"income={self.income_variable} | measure={self.inequality_measure} | "
            f"circs=[{circs}] | sample={self.sample_filter}"
        )
