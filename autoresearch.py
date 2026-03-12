"""autoresearch.py -- Agent toolkit for IOp specification curve exploration.

The AI agent (Claude Code) is the reasoning engine that drives the research.
This script provides diagnostic commands the agent uses to make decisions
about what specifications to explore next.

Agent-in-the-loop workflow:
    1. Agent runs `python autoresearch.py status` to see coverage and gaps
    2. Agent runs `python autoresearch.py findings` to detect patterns
    3. Agent reasons about what to explore (scientific value, not just coverage)
    4. Agent edits `analyze.py` with justified specs
    5. Agent runs `python run_experiment.py` to execute specs
    6. Agent analyzes results and decides next steps
    7. Agent runs `python autoresearch.py synthesize` when appropriate

Usage:
    python autoresearch.py status                  # coverage + experiment summary
    python autoresearch.py findings                # detect patterns in results
    python autoresearch.py gaps                    # show specific missing slots
    python autoresearch.py gaps --mi               # show MI-specific gaps
    python autoresearch.py recent                  # last 10 experiment results
    python autoresearch.py recent 20               # last 20 results
    python autoresearch.py synthesize              # generate spec curve + tables
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from core.specification import ExperimentSpec
from core.types import (
    DecompositionType,
    EstimationMethod,
    InequalityMeasure,
    IncomeVariable,
    SampleFilter,
)
from orchestration.coverage_tracker import compute_coverage, CoverageReport
from orchestration.experiment_log import (
    load_experiment_log,
    get_completed_spec_ids,
    get_experiment_count,
)
from orchestration.strategy import (
    CORE_CIRC_SETS,
    CORE_INCOMES,
    CORE_MEASURES,
    VALID_METHOD_DECOMP,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


# ---------------------------------------------------------------------------
# Command: status
# ---------------------------------------------------------------------------

def cmd_status() -> None:
    """Show coverage, experiment count, and breakdown."""
    coverage = compute_coverage()
    records = load_experiment_log()
    successful = [r for r in records if r.get("status") == "success"]

    # Split listwise vs MI
    mi_success = [r for r in successful if r.get("use_mi") is True]
    listwise_success = [r for r in successful if r.get("use_mi") is not True]

    print(f"\n{'='*60}")
    print(f"  AUTORESEARCH STATUS")
    print(f"{'='*60}")
    print(f"  Total experiments:  {len(records)}")
    print(f"  Successful:         {len(successful)}")
    print(f"    Listwise:         {len(listwise_success)}")
    print(f"    MI:               {len(mi_success)}")
    print(f"  Failed/error:       {len(records) - len(successful)}")
    print(f"  Core coverage:      {coverage.core_coverage_completed}/{coverage.core_coverage_total}"
          f" ({coverage.coverage_pct:.1f}%)")
    print()

    # Method breakdown
    print("  By method:")
    for method, count in sorted(coverage.by_method.items()):
        print(f"    {method:20s} {count:4d}")
    print()

    # Measure breakdown
    print("  By measure:")
    for measure, count in sorted(coverage.by_measure.items()):
        print(f"    {measure:20s} {count:4d}")
    print()

    # Income breakdown
    print("  By income:")
    for income, count in sorted(coverage.by_income.items()):
        print(f"    {income:30s} {count:4d}")
    print()

    # MI breakdown by method
    if mi_success:
        mi_by_method: dict[str, int] = {}
        for r in mi_success:
            m = r.get("method", "unknown")
            mi_by_method[m] = mi_by_method.get(m, 0) + 1
        print("  MI specs by method:")
        for method, count in sorted(mi_by_method.items()):
            print(f"    {method:20s} {count:4d}")
        print()

    # IOp summary stats
    if successful:
        iop_values = [r["iop_share"] for r in successful if "iop_share" in r]
        if iop_values:
            iop_series = pd.Series(iop_values)
            print("  IOp share distribution:")
            print(f"    Mean:    {iop_series.mean():.3f}")
            print(f"    Median:  {iop_series.median():.3f}")
            print(f"    Std:     {iop_series.std():.3f}")
            print(f"    Min:     {iop_series.min():.3f}")
            print(f"    Max:     {iop_series.max():.3f}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Command: findings -- detect patterns for the agent to reason about
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    """A single interesting pattern detected in the results."""
    category: str
    description: str
    evidence: dict[str, Any]
    suggested_followup: str


@dataclass
class FindingsReport:
    """Collection of findings from analyzing the experiment log."""
    n_experiments: int
    coverage_pct: float
    findings: list[Finding] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"=== Findings Report ({self.n_experiments} experiments, "
            f"{self.coverage_pct:.1f}% coverage) ===",
            "",
        ]
        if not self.findings:
            lines.append("No notable findings yet. Run more experiments.")
        for i, f in enumerate(self.findings, 1):
            lines.append(f"{i}. [{f.category}] {f.description}")
            lines.append(f"   -> Suggested followup: {f.suggested_followup}")
            lines.append("")
        return "\n".join(lines)


def _circ_key(val: Any) -> str:
    if isinstance(val, list):
        return "|".join(sorted(val))
    return str(val)


def _detect_method_divergence(df: pd.DataFrame, report: FindingsReport) -> None:
    """Flag when different methods give very different IOp for same circumstances."""
    if "method" not in df.columns or df["method"].nunique() < 2:
        return
    df = df.copy()
    df["_circ_key"] = df["circumstances"].apply(_circ_key)
    group_cols = ["_circ_key", "income_variable", "inequality_measure"]
    for _, group in df.groupby(group_cols):
        if group["method"].nunique() < 2:
            continue
        iop_range = group["iop_share"].max() - group["iop_share"].min()
        if iop_range > 0.10:
            methods = dict(zip(group["method"], group["iop_share"].round(4)))
            report.findings.append(Finding(
                category="method_divergence",
                description=f"IOp diverges by {iop_range:.3f} across methods: {methods}",
                evidence={"methods": methods, "range": round(iop_range, 4)},
                suggested_followup=(
                    f"The OLS-XGBoost gap ({iop_range:.3f}) captures interaction "
                    f"effects. Report this delta in the paper."
                ),
            ))


def _detect_circumstance_sensitivity(df: pd.DataFrame, report: FindingsReport) -> None:
    """Flag when IOp varies heavily across circumstance sets (same method)."""
    group_cols = ["method", "income_variable", "inequality_measure", "decomposition_type"]
    for _, group in df.groupby(group_cols):
        if len(group) < 2:
            continue
        iop_range = group["iop_share"].max() - group["iop_share"].min()
        if iop_range > 0.15:
            report.findings.append(Finding(
                category="circumstance_sensitivity",
                description=(
                    f"IOp varies by {iop_range:.3f} across circumstance sets "
                    f"for {group.iloc[0]['method']}/{group.iloc[0]['inequality_measure']}"
                ),
                evidence={
                    "iop_range": round(iop_range, 4),
                    "n_specs": len(group),
                    "min_iop": round(group["iop_share"].min(), 4),
                    "max_iop": round(group["iop_share"].max(), 4),
                },
                suggested_followup=(
                    "Run hypothesis specs isolating which circumstance drives "
                    "the change. Add/remove one variable at a time."
                ),
            ))


def _detect_measure_sensitivity(df: pd.DataFrame, report: FindingsReport) -> None:
    """Flag when different inequality measures give very different IOp shares."""
    if df["inequality_measure"].nunique() < 2:
        return
    df = df.copy()
    df["_circ_key"] = df["circumstances"].apply(_circ_key)
    group_cols = ["_circ_key", "method", "income_variable", "decomposition_type"]
    for _, group in df.groupby(group_cols):
        if group["inequality_measure"].nunique() < 2:
            continue
        iop_range = group["iop_share"].max() - group["iop_share"].min()
        if iop_range > 0.10:
            measures = dict(zip(group["inequality_measure"], group["iop_share"].round(4)))
            report.findings.append(Finding(
                category="measure_sensitivity",
                description=f"IOp varies by {iop_range:.3f} across measures: {measures}",
                evidence={"measures": measures, "range": round(iop_range, 4)},
                suggested_followup=(
                    "Expected -- report the full range in the spec curve. "
                    "MLD is exactly decomposable; Gini is not."
                ),
            ))


def _detect_wide_cis(df: pd.DataFrame, report: FindingsReport) -> None:
    """Flag specs with unusually wide confidence intervals."""
    if "ci_lower" not in df.columns or "ci_upper" not in df.columns:
        return
    ci_width = df["ci_upper"] - df["ci_lower"]
    wide = df[ci_width > 0.15]
    if len(wide) > 0:
        report.findings.append(Finding(
            category="wide_ci",
            description=f"{len(wide)} specs have CI width > 0.15 (max: {ci_width.max():.3f})",
            evidence={
                "n_wide": len(wide),
                "max_width": round(ci_width.max(), 4),
                "spec_ids": wide["spec_id"].tolist()[:5],
            },
            suggested_followup=(
                "Wide CIs suggest small effective sample or high variance. "
                "Check sample sizes and consider income outliers."
            ),
        ))


def _detect_outliers(df: pd.DataFrame, report: FindingsReport) -> None:
    """Flag IOp shares that are unusually high or low (IQR method)."""
    if len(df) < 5:
        return
    q1 = df["iop_share"].quantile(0.25)
    q3 = df["iop_share"].quantile(0.75)
    iqr = q3 - q1
    if iqr < 0.01:
        return
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = df[(df["iop_share"] < lower) | (df["iop_share"] > upper)]
    if len(outliers) > 0:
        report.findings.append(Finding(
            category="outlier_iop",
            description=(
                f"{len(outliers)} specs have outlier IOp values "
                f"(outside [{lower:.3f}, {upper:.3f}])"
            ),
            evidence={
                "n_outliers": len(outliers),
                "bounds": [round(lower, 4), round(upper, 4)],
                "outlier_values": outliers["iop_share"].round(4).tolist(),
            },
            suggested_followup=(
                "Investigate what drives outlier specs. "
                "Are they methodologically valid or flagged by diagnostics?"
            ),
        ))


def _detect_mi_vs_listwise(df: pd.DataFrame, report: FindingsReport) -> None:
    """Compare MI vs listwise results on overlapping specs."""
    if "use_mi" not in df.columns:
        return

    mi_df = df[df["use_mi"] == True].copy()
    lw_df = df[df["use_mi"] != True].copy()

    if len(mi_df) == 0 or len(lw_df) == 0:
        return

    # Match on method, decomp, measure, income, circumstances
    mi_df["_circ_key"] = mi_df["circumstances"].apply(_circ_key)
    lw_df["_circ_key"] = lw_df["circumstances"].apply(_circ_key)

    merge_cols = ["method", "decomposition_type", "inequality_measure",
                  "income_variable", "_circ_key"]
    merged = mi_df.merge(lw_df, on=merge_cols, suffixes=("_mi", "_lw"))

    if len(merged) == 0:
        return

    merged["iop_diff"] = merged["iop_share_mi"] - merged["iop_share_lw"]
    mean_diff = merged["iop_diff"].mean()
    max_diff = merged["iop_diff"].abs().max()

    report.findings.append(Finding(
        category="mi_vs_listwise",
        description=(
            f"MI vs listwise comparison on {len(merged)} matched specs: "
            f"mean diff = {mean_diff:+.4f}, max |diff| = {max_diff:.4f}"
        ),
        evidence={
            "n_matched": len(merged),
            "mean_diff": round(mean_diff, 4),
            "max_abs_diff": round(max_diff, 4),
            "median_diff": round(merged["iop_diff"].median(), 4),
        },
        suggested_followup=(
            "If mean diff is near zero, listwise deletion is not biasing results. "
            "If large, MI results should be preferred in the paper."
        ),
    ))

    # Flag high FMI specs
    if "mi_fraction_missing_info_mi" in merged.columns:
        high_fmi = merged[merged["mi_fraction_missing_info_mi"] > 0.5]
        if len(high_fmi) > 0:
            report.findings.append(Finding(
                category="high_fmi",
                description=f"{len(high_fmi)} MI specs have FMI > 0.5",
                evidence={"n_high_fmi": len(high_fmi)},
                suggested_followup=(
                    "FMI > 0.5 means results are heavily influenced by imputation model. "
                    "Consider increasing M or reviewing imputation model."
                ),
            ))


def detect_findings() -> FindingsReport:
    """Analyze experiment log for interesting patterns.

    The agent reads these findings to reason about what to explore next.
    """
    records = load_experiment_log()
    successful = [r for r in records if r.get("status") == "success"]
    coverage = compute_coverage()

    report = FindingsReport(
        n_experiments=len(records),
        coverage_pct=coverage.coverage_pct,
    )

    if len(successful) < 3:
        return report

    df = pd.DataFrame(successful)

    _detect_method_divergence(df, report)
    _detect_circumstance_sensitivity(df, report)
    _detect_measure_sensitivity(df, report)
    _detect_wide_cis(df, report)
    _detect_outliers(df, report)
    _detect_mi_vs_listwise(df, report)

    return report


def cmd_findings() -> None:
    """Detect and print findings from current results."""
    report = detect_findings()
    print(report.summary())


# ---------------------------------------------------------------------------
# Command: gaps -- show specific missing specification slots
# ---------------------------------------------------------------------------

def cmd_gaps(mi_only: bool = False) -> None:
    """Show specific gaps in the specification space."""
    completed = get_completed_spec_ids()
    records = load_experiment_log()
    successful = [r for r in records if r.get("status") == "success"]

    # Build completed slots keyed by (method, decomp, income, measure, circ_key, use_mi)
    completed_slots: set[tuple] = set()
    for r in successful:
        circs = tuple(sorted(r.get("circumstances", [])))
        use_mi = r.get("use_mi", False) or False
        completed_slots.add((
            r.get("method"), r.get("decomposition_type"),
            r.get("income_variable"), r.get("inequality_measure"),
            circs, use_mi,
        ))

    # Build expected slots
    missing_listwise = []
    missing_mi = []

    for i, circs in enumerate(CORE_CIRC_SETS):
        circs_key = tuple(sorted(circs))
        for method, decomp in VALID_METHOD_DECOMP:
            for measure in CORE_MEASURES:
                for income in CORE_INCOMES:
                    slot_lw = (method, decomp, income, measure, circs_key, False)
                    slot_mi = (method, decomp, income, measure, circs_key, True)
                    if slot_lw not in completed_slots:
                        missing_listwise.append({
                            "circ_set": i, "n_circs": len(circs),
                            "method": method, "decomp": decomp,
                            "measure": measure, "income": income,
                        })
                    if slot_mi not in completed_slots:
                        missing_mi.append({
                            "circ_set": i, "n_circs": len(circs),
                            "method": method, "decomp": decomp,
                            "measure": measure, "income": income,
                        })

    if mi_only:
        _print_gaps("MI", missing_mi)
    else:
        _print_gaps("Listwise", missing_listwise)
        print()
        _print_gaps("MI", missing_mi)


def _print_gaps(label: str, gaps: list[dict]) -> None:
    """Print gap summary grouped by method."""
    print(f"\n--- {label} gaps: {len(gaps)} missing slots ---")
    if not gaps:
        print("  None! Full coverage.")
        return

    # Group by method
    by_method: dict[str, list] = {}
    for g in gaps:
        by_method.setdefault(g["method"], []).append(g)

    for method, method_gaps in sorted(by_method.items()):
        # Group by circ set
        by_circ: dict[int, int] = {}
        for g in method_gaps:
            by_circ[g["circ_set"]] = by_circ.get(g["circ_set"], 0) + 1
        circ_summary = ", ".join(f"set{k}({v})" for k, v in sorted(by_circ.items()))
        print(f"  {method}: {len(method_gaps)} gaps [{circ_summary}]")


# ---------------------------------------------------------------------------
# Command: recent -- show last N experiment results
# ---------------------------------------------------------------------------

def cmd_recent(n: int = 10) -> None:
    """Show the last N experiment results."""
    records = load_experiment_log()
    recent = records[-n:]

    print(f"\n--- Last {len(recent)} experiments ---")
    print(f"{'Status':8s} {'Method':15s} {'Decomp':12s} {'Measure':8s} "
          f"{'IOp':7s} {'CI':15s} {'MI':3s} {'Circs':5s} {'Time':6s}")
    print("-" * 90)

    for r in recent:
        status = r.get("status", "?")
        method = r.get("method", "?")[:15]
        decomp = r.get("decomposition_type", "?")[:12]
        measure = r.get("inequality_measure", "?")[:8]
        iop = r.get("iop_share")
        iop_str = f"{iop:.4f}" if iop is not None else "  --  "
        ci_lo = r.get("ci_lower")
        ci_hi = r.get("ci_upper")
        ci_str = f"[{ci_lo:.3f},{ci_hi:.3f}]" if ci_lo is not None else "      --      "
        mi = "Y" if r.get("use_mi") else "N"
        n_circs = len(r.get("circumstances", []))
        duration = r.get("duration_seconds")
        dur_str = f"{duration:.0f}s" if duration is not None else "  -- "
        print(f"{status:8s} {method:15s} {decomp:12s} {measure:8s} "
              f"{iop_str:7s} {ci_str:15s} {mi:3s} {n_circs:5d} {dur_str:6s}")
    print()


# ---------------------------------------------------------------------------
# Command: synthesize -- generate spec curve and summary tables
# ---------------------------------------------------------------------------

def cmd_synthesize() -> None:
    """Generate specification curve plot and summary tables."""
    print("Generating specification curve...")
    try:
        from synthesis.spec_curve import plot_specification_curve
        plot_specification_curve()
        print("  -> Spec curve saved to results/figures/")
    except Exception as e:
        print(f"  -> Spec curve failed: {e}")

    print("Generating summary tables...")
    try:
        from synthesis.summary_tables import generate_all_tables
        generate_all_tables()
        print("  -> Summary tables saved to results/tables/")
    except Exception as e:
        print(f"  -> Summary tables failed: {e}")

    print("Generating additional figures...")
    try:
        from synthesis.figures import generate_all_figures
        generate_all_figures()
        print("  -> Figures saved to results/figures/")
    except Exception as e:
        print(f"  -> Figures failed: {e}")

    print("\nSynthesis complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agent toolkit for IOp specification curve exploration",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # status
    subparsers.add_parser("status", help="Show coverage and experiment summary")

    # findings
    subparsers.add_parser("findings", help="Detect patterns in results")

    # gaps
    gaps_parser = subparsers.add_parser("gaps", help="Show missing specification slots")
    gaps_parser.add_argument("--mi", action="store_true", help="Show MI gaps only")

    # recent
    recent_parser = subparsers.add_parser("recent", help="Show last N experiment results")
    recent_parser.add_argument("n", nargs="?", type=int, default=10, help="Number of results (default: 10)")

    # synthesize
    subparsers.add_parser("synthesize", help="Generate spec curve and summary tables")

    args = parser.parse_args()

    if args.command == "status":
        cmd_status()
    elif args.command == "findings":
        cmd_findings()
    elif args.command == "gaps":
        cmd_gaps(mi_only=args.mi)
    elif args.command == "recent":
        cmd_recent(n=args.n)
    elif args.command == "synthesize":
        cmd_synthesize()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
