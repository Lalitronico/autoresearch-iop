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
    unique_specs_completed: int  # Distinct successful spec slots
    core_coverage_completed: int  # Core slots completed (out of core_coverage_total)
    core_coverage_total: int  # Theoretical core space size (585)
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
    coverage_pct: float  # Core combinatorial coverage (core_completed / core_total * 100)

    def summary(self) -> str:
        lines = [
            f"=== Coverage Report ===",
            f"Total experiments: {self.total_experiments}",
            f"Successful: {self.successful_experiments}",
            f"Unique specs completed: {self.unique_specs_completed}",
            f"Core coverage: {self.core_coverage_completed}/{self.core_coverage_total}"
            f" ({self.coverage_pct:.1f}%)",
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
    """Analyze experiment log and compute coverage metrics.

    Coverage is measured combinatorially: what fraction of the 585 core
    specification slots (5 method-decomp × 3 incomes × 3 measures × 13 circ sets)
    have been successfully completed.
    """
    # Lazy import to avoid circular dependency (strategy imports coverage_tracker)
    from orchestration.strategy import (
        CORE_CIRC_SETS,
        CORE_INCOMES,
        CORE_MEASURES,
        VALID_METHOD_DECOMP,
    )

    records = load_experiment_log()
    successful = [r for r in records if r.get("status") == "success"]

    by_method: Counter[str] = Counter()
    by_measure: Counter[str] = Counter()
    by_income: Counter[str] = Counter()
    by_decomposition: Counter[str] = Counter()
    by_sample: Counter[str] = Counter()
    by_circumstance: Counter[str] = Counter()

    # Build set of completed spec slots: (method, decomp, income, measure, frozenset(circs))
    completed_slots: set[tuple] = set()
    for r in successful:
        by_method[r.get("method", "")] += 1
        by_measure[r.get("inequality_measure", "")] += 1
        by_income[r.get("income_variable", "")] += 1
        by_decomposition[r.get("decomposition_type", "")] += 1
        by_sample[r.get("sample_filter", "")] += 1
        circs = r.get("circumstances", [])
        for c in circs:
            by_circumstance[c] += 1
        completed_slots.add((
            r.get("method", ""),
            r.get("decomposition_type", ""),
            r.get("income_variable", ""),
            r.get("inequality_measure", ""),
            frozenset(circs),
        ))

    # Build theoretical core space
    core_slots: set[tuple] = set()
    for method, decomp in VALID_METHOD_DECOMP:
        for circs in CORE_CIRC_SETS:
            for measure in CORE_MEASURES:
                for income in CORE_INCOMES:
                    core_slots.add((method, decomp, income, measure, frozenset(circs)))

    core_completed = core_slots & completed_slots
    core_coverage_pct = (
        len(core_completed) / len(core_slots) * 100 if core_slots else 0.0
    )

    # Missing dimensions (still useful for gap identification)
    all_methods = {m.value for m in EstimationMethod}
    all_measures = {m.value for m in InequalityMeasure}
    all_incomes = {m.value for m in IncomeVariable}
    all_decomps = {m.value for m in DecompositionType}

    missing_methods = sorted(all_methods - set(by_method.keys()))
    missing_measures = sorted(all_measures - set(by_measure.keys()))
    missing_incomes = sorted(all_incomes - set(by_income.keys()))
    missing_decomps = sorted(all_decomps - set(by_decomposition.keys()))

    return CoverageReport(
        total_experiments=len(records),
        successful_experiments=len(successful),
        unique_specs_completed=len(completed_slots),
        core_coverage_completed=len(core_completed),
        core_coverage_total=len(core_slots),
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
        coverage_pct=core_coverage_pct,
    )
