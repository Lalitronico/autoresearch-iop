"""autoresearch.py -- Autonomous IOp specification curve exploration.

Separates two fundamentally different exploration modes:

SYSTEMATIC (60%, deterministic):
    Fills coverage gaps across method x measure x income x decomposition.
    No LLM needed. The spec space is finite and enumerable.

HYPOTHESIS (30%, LLM-driven):
    Detects interesting patterns in results and surfaces them as findings.
    The LLM agent reads findings and proposes targeted specs via analyze.py.

ROBUSTNESS (10%, semi-automatic):
    Takes key baseline specs and varies one dimension at a time.

Usage:
    python autoresearch.py                       # systematic until 80% coverage
    python autoresearch.py --target 90           # systematic until 90%
    python autoresearch.py --findings            # show findings from current results
    python autoresearch.py --full --max-iter 10  # full loop with hypothesis phases
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from core.specification import ExperimentSpec
from orchestration.coverage_tracker import compute_coverage, CoverageReport
from orchestration.experiment_log import (
    load_experiment_log,
    get_completed_spec_ids,
    get_experiment_count,
)
from orchestration.strategy import (
    generate_systematic_batch,
    generate_robustness_batch,
)
from run_experiment import run_batch
from synthesis.spec_curve import plot_specification_curve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
FINDINGS_PATH = RESULTS_DIR / "latest_findings.txt"


# ---------------------------------------------------------------------------
# Findings detection -- surfaces patterns for the LLM to act on
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    """A single interesting pattern detected in the results."""

    category: str  # method_divergence, circumstance_sensitivity, ...
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

    def save(self, path: Path = FINDINGS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.summary(), encoding="utf-8")
        logger.info(f"Findings saved to {path}")


def detect_findings() -> FindingsReport:
    """Analyze experiment log for interesting patterns.

    This is the bridge between the deterministic pipeline and the LLM.
    The LLM reads these findings to decide what hypothesis specs to propose.
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

    return report


def _circ_key(val: Any) -> str:
    """Normalize circumstances to a sortable string key."""
    if isinstance(val, list):
        return "|".join(sorted(val))
    return str(val)


def _detect_method_divergence(df: pd.DataFrame, report: FindingsReport) -> None:
    """Flag when different methods give very different IOp for same circumstances.

    This is the 'free finding': the OLS-vs-XGBoost gap captures interaction
    effects without any extra code.
    """
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
                description=(
                    f"IOp diverges by {iop_range:.3f} across methods: {methods}"
                ),
                evidence={"methods": methods, "range": round(iop_range, 4)},
                suggested_followup=(
                    f"The OLS-XGBoost gap ({iop_range:.3f}) captures interaction "
                    f"effects. Report this delta in the paper."
                ),
            ))


def _detect_circumstance_sensitivity(
    df: pd.DataFrame, report: FindingsReport
) -> None:
    """Flag when IOp varies heavily across circumstance sets (same method)."""
    group_cols = [
        "method", "income_variable", "inequality_measure", "decomposition_type",
    ]
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


def _detect_measure_sensitivity(
    df: pd.DataFrame, report: FindingsReport
) -> None:
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
            measures = dict(
                zip(group["inequality_measure"], group["iop_share"].round(4))
            )
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
            description=(
                f"{len(wide)} specs have CI width > 0.15 "
                f"(max: {ci_width.max():.3f})"
            ),
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


# ---------------------------------------------------------------------------
# Phase 1: Systematic (deterministic, no LLM)
# ---------------------------------------------------------------------------

def run_systematic_phase(
    batch_size: int = 10,
    target_coverage: float = 80.0,
    max_experiments: int = 200,
) -> CoverageReport:
    """Fill coverage gaps across the specification space.

    Fully deterministic. Iterates until target coverage is reached,
    the spec space is exhausted, or the experiment budget runs out.
    """
    total_run = 0
    iteration = 0

    while total_run < max_experiments:
        iteration += 1
        coverage = compute_coverage()

        logger.info(
            f"\n{'='*60}\n"
            f"SYSTEMATIC -- Iteration {iteration}\n"
            f"Coverage: {coverage.coverage_pct:.1f}% "
            f"(target: {target_coverage}%) | "
            f"Total experiments: {get_experiment_count()}\n"
            f"{'='*60}"
        )

        if coverage.coverage_pct >= target_coverage:
            logger.info(f"Target coverage reached: {coverage.coverage_pct:.1f}%")
            break

        specs = generate_systematic_batch(batch_size=batch_size)
        if not specs:
            logger.info(
                "No more systematic specs to generate. "
                "Coverage may be at maximum for defined core sets."
            )
            break

        logger.info(f"Generated {len(specs)} systematic specs")
        results = run_batch(specs)
        total_run += len(specs)

        success = sum(1 for r in results if r.status == "success")
        logger.info(f"Batch result: {success}/{len(results)} successful")

    final = compute_coverage()
    logger.info(f"\nSystematic phase complete. Coverage: {final.coverage_pct:.1f}%")
    return final


# ---------------------------------------------------------------------------
# Phase 2: Hypothesis (LLM-driven)
# ---------------------------------------------------------------------------

def run_hypothesis_phase() -> int:
    """Execute hypothesis specs defined by the LLM in analyze.py.

    Returns number of new specs executed.
    """
    try:
        from analyze import get_specs

        all_specs = get_specs()
    except Exception as e:
        logger.warning(f"Could not load hypothesis specs from analyze.py: {e}")
        return 0

    completed = get_completed_spec_ids()
    new_specs = [s for s in all_specs if s.spec_id not in completed]

    if not new_specs:
        logger.info("No new hypothesis specs in analyze.py")
        return 0

    logger.info(f"Running {len(new_specs)} hypothesis specs from analyze.py")
    run_batch(new_specs)
    return len(new_specs)


# ---------------------------------------------------------------------------
# Phase 3: Robustness (semi-automatic)
# ---------------------------------------------------------------------------

def run_robustness_phase(
    n_baselines: int = 3,
    batch_size: int = 5,
) -> int:
    """Vary one dimension at a time from key baseline specs.

    Returns number of robustness specs executed.
    """
    records = load_experiment_log()
    successful = [r for r in records if r.get("status") == "success"]

    if not successful:
        logger.info("No successful experiments for robustness checks.")
        return 0

    total = 0
    for base_record in successful[:n_baselines]:
        base_spec = ExperimentSpec.from_dict(base_record)
        specs = generate_robustness_batch(base_spec, batch_size=batch_size)
        if specs:
            logger.info(
                f"Robustness for {base_spec.spec_id}: {len(specs)} variants"
            )
            run_batch(specs)
            total += len(specs)

    return total


# ---------------------------------------------------------------------------
# Full loop: systematic + findings + hypothesis + robustness
# ---------------------------------------------------------------------------

def run_full_loop(
    target_coverage: float = 80.0,
    max_iterations: int = 10,
    systematic_batch_size: int = 10,
    max_experiments: int = 500,
    synthesis_every: int = 25,
) -> None:
    """Full autoresearch loop.

    Each iteration:
    1. Systematic: deterministic gap-filling (no LLM)
    2. Findings: detect patterns, save report (no LLM)
    3. Hypothesis: run LLM-proposed specs from analyze.py (LLM edits between iterations)
    4. Robustness: vary dimensions from baselines (every other iteration)
    5. Synthesis: generate spec curve at checkpoints
    """
    for iteration in range(1, max_iterations + 1):
        n_before = get_experiment_count()
        coverage = compute_coverage()

        logger.info(
            f"\n{'#'*60}\n"
            f"AUTORESEARCH -- Iteration {iteration}/{max_iterations}\n"
            f"Experiments: {n_before} | Coverage: {coverage.coverage_pct:.1f}%\n"
            f"{'#'*60}"
        )

        if coverage.coverage_pct >= target_coverage:
            logger.info("Target coverage reached.")
            break

        if n_before >= max_experiments:
            logger.info("Experiment budget exhausted.")
            break

        # -- Phase 1: Systematic (deterministic) --
        logger.info("\n-- Phase 1: SYSTEMATIC (deterministic, no LLM) --")
        specs = generate_systematic_batch(batch_size=systematic_batch_size)
        if specs:
            run_batch(specs)
        else:
            logger.info("Systematic: no gaps remain in core space.")

        # -- Phase 2: Findings (bridge to LLM) --
        logger.info("\n-- Phase 2: FINDINGS (for LLM hypothesis generation) --")
        findings = detect_findings()
        print(findings.summary())
        findings.save()

        # -- Phase 3: Hypothesis (LLM-driven) --
        logger.info("\n-- Phase 3: HYPOTHESIS (LLM-proposed specs) --")
        run_hypothesis_phase()

        # -- Phase 4: Robustness (every other iteration) --
        if iteration % 2 == 0:
            logger.info("\n-- Phase 4: ROBUSTNESS --")
            run_robustness_phase()

        # -- Synthesis checkpoint --
        n_after = get_experiment_count()
        if (
            n_after >= synthesis_every
            and n_after // synthesis_every > n_before // synthesis_every
        ):
            logger.info("\n-- Generating specification curve --")
            try:
                plot_specification_curve()
            except Exception as e:
                logger.warning(f"Spec curve generation failed: {e}")

    # Final synthesis
    logger.info("\n-- Final synthesis --")
    try:
        plot_specification_curve()
    except Exception as e:
        logger.warning(f"Spec curve generation failed: {e}")

    final = compute_coverage()
    print(
        f"\n{'='*60}\n"
        f"AUTORESEARCH COMPLETE\n"
        f"Total experiments: {get_experiment_count()}\n"
        f"Coverage: {final.coverage_pct:.1f}%\n"
        f"{'='*60}"
    )
    print(final.summary())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous IOp specification curve exploration",
    )
    parser.add_argument(
        "--systematic", action="store_true",
        help="Run systematic phase only (deterministic, no LLM)",
    )
    parser.add_argument(
        "--findings", action="store_true",
        help="Detect and print findings from current results",
    )
    parser.add_argument(
        "--robustness", action="store_true",
        help="Run robustness checks on baseline specs",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Full loop: systematic + findings + hypothesis + robustness",
    )
    parser.add_argument(
        "--target", type=float, default=80.0,
        help="Target coverage percentage (default: 80)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10,
        help="Specs per systematic batch (default: 10)",
    )
    parser.add_argument(
        "--max-experiments", type=int, default=500,
        help="Max total experiments (default: 500)",
    )
    parser.add_argument(
        "--max-iter", type=int, default=10,
        help="Max loop iterations for --full (default: 10)",
    )

    args = parser.parse_args()

    if args.findings:
        report = detect_findings()
        print(report.summary())
    elif args.robustness:
        run_robustness_phase()
    elif args.full:
        run_full_loop(
            target_coverage=args.target,
            max_iterations=args.max_iter,
            systematic_batch_size=args.batch_size,
            max_experiments=args.max_experiments,
        )
    else:
        # Default: systematic only
        run_systematic_phase(
            batch_size=args.batch_size,
            target_coverage=args.target,
            max_experiments=args.max_experiments,
        )


if __name__ == "__main__":
    main()
