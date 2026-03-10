"""Track coverage of the specification space.

Identifies which regions of the multiverse have been explored
and which remain unexplored.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from core.types import (
    Circumstance,
    DecompositionType,
    EstimationMethod,
    InequalityMeasure,
    IncomeVariable,
    SampleFilter,
)
from orchestration.experiment_log import load_experiment_log


@dataclass
class CoverageReport:
    """Summary of specification space coverage."""
    total_experiments: int
    successful_experiments: int
    by_method: dict[str, int]
    by_measure: dict[str, int]
    by_income: dict[str, int]
    by_decomposition: dict[str, int]
    by_sample: dict[str, int]
    by_circumstance: dict[str, int]
    missing_methods: list[str]
    missing_measures: list[str]
    missing_incomes: list[str]
    missing_decompositions: list[str]
    coverage_pct: float  # Overall coverage estimate

    def summary(self) -> str:
        lines = [
            f"=== Coverage Report ===",
            f"Total experiments: {self.total_experiments}",
            f"Successful: {self.successful_experiments}",
            f"Overall coverage: {self.coverage_pct:.1f}%",
            f"",
            f"By method: {self.by_method}",
            f"By measure: {self.by_measure}",
            f"By income: {self.by_income}",
            f"By decomposition: {self.by_decomposition}",
            f"By sample: {self.by_sample}",
            f"",
            f"Missing methods: {self.missing_methods}",
            f"Missing measures: {self.missing_measures}",
            f"Missing incomes: {self.missing_incomes}",
            f"Missing decompositions: {self.missing_decompositions}",
        ]
        return "\n".join(lines)


def compute_coverage() -> CoverageReport:
    """Analyze experiment log and compute coverage metrics."""
    records = load_experiment_log()
    successful = [r for r in records if r.get("status") == "success"]

    by_method: Counter[str] = Counter()
    by_measure: Counter[str] = Counter()
    by_income: Counter[str] = Counter()
    by_decomposition: Counter[str] = Counter()
    by_sample: Counter[str] = Counter()
    by_circumstance: Counter[str] = Counter()

    for r in successful:
        by_method[r.get("method", "")] += 1
        by_measure[r.get("inequality_measure", "")] += 1
        by_income[r.get("income_variable", "")] += 1
        by_decomposition[r.get("decomposition_type", "")] += 1
        by_sample[r.get("sample_filter", "")] += 1
        for c in r.get("circumstances", []):
            by_circumstance[c] += 1

    # Missing dimensions
    all_methods = {m.value for m in EstimationMethod}
    all_measures = {m.value for m in InequalityMeasure}
    all_incomes = {m.value for m in IncomeVariable}
    all_decomps = {m.value for m in DecompositionType}

    missing_methods = sorted(all_methods - set(by_method.keys()))
    missing_measures = sorted(all_measures - set(by_measure.keys()))
    missing_incomes = sorted(all_incomes - set(by_income.keys()))
    missing_decomps = sorted(all_decomps - set(by_decomposition.keys()))

    # Coverage percentage: fraction of dimension values touched
    n_dims = len(all_methods) + len(all_measures) + len(all_incomes) + len(all_decomps)
    n_covered = (
        len(all_methods - set(missing_methods))
        + len(all_measures - set(missing_measures))
        + len(all_incomes - set(missing_incomes))
        + len(all_decomps - set(missing_decomps))
    )
    coverage_pct = (n_covered / n_dims * 100) if n_dims > 0 else 0.0

    return CoverageReport(
        total_experiments=len(records),
        successful_experiments=len(successful),
        by_method=dict(by_method),
        by_measure=dict(by_measure),
        by_income=dict(by_income),
        by_decomposition=dict(by_decomposition),
        by_sample=dict(by_sample),
        by_circumstance=dict(by_circumstance),
        missing_methods=missing_methods,
        missing_measures=missing_measures,
        missing_incomes=missing_incomes,
        missing_decompositions=missing_decomps,
        coverage_pct=coverage_pct,
    )
