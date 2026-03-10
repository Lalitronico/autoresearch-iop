"""Publication-quality figures for IOp analysis.

Beyond the spec curve, generates:
- IOp distribution histograms
- Variable importance plots
- Method comparison forest plots
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from orchestration.experiment_log import load_experiment_log

logger = logging.getLogger(__name__)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})


def _load_successful_df() -> pd.DataFrame:
    records = load_experiment_log()
    successful = [r for r in records if r.get("status") == "success"]
    return pd.DataFrame(successful) if successful else pd.DataFrame()


def plot_iop_distribution(output_path: Path | None = None) -> plt.Figure:
    """Histogram of IOp shares across all specifications."""
    df = _load_successful_df()
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["iop_share"], bins=30, edgecolor="white", color="steelblue", alpha=0.8)
    ax.axvline(df["iop_share"].median(), color="red", linestyle="--", label=f'Median: {df["iop_share"].median():.3f}')
    ax.axvline(df["iop_share"].mean(), color="orange", linestyle=":", label=f'Mean: {df["iop_share"].mean():.3f}')
    ax.set_xlabel("IOp Share")
    ax.set_ylabel("Number of Specifications")
    ax.set_title("Distribution of IOp Estimates Across Specifications")
    ax.legend()

    output_path = output_path or FIGURES_DIR / "iop_distribution.png"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved IOp distribution to {output_path}")
    return fig


def plot_method_comparison(output_path: Path | None = None) -> plt.Figure:
    """Forest plot comparing IOp shares by estimation method."""
    df = _load_successful_df()
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    summary = df.groupby("method").agg(
        mean=("iop_share", "mean"),
        ci_low=("iop_share", lambda x: np.percentile(x, 2.5)),
        ci_high=("iop_share", lambda x: np.percentile(x, 97.5)),
        n=("iop_share", "count"),
    ).sort_values("mean")

    fig, ax = plt.subplots(figsize=(8, max(4, len(summary) * 0.8)))
    y_pos = range(len(summary))

    ax.errorbar(
        summary["mean"], y_pos,
        xerr=[summary["mean"] - summary["ci_low"], summary["ci_high"] - summary["mean"]],
        fmt="o", color="steelblue", capsize=4, markersize=6,
    )

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([f'{m} (n={n})' for m, n in zip(summary.index, summary["n"])])
    ax.set_xlabel("IOp Share")
    ax.set_title("IOp Estimates by Estimation Method")
    ax.axvline(df["iop_share"].median(), color="gray", linestyle="--", alpha=0.5)

    output_path = output_path or FIGURES_DIR / "method_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved method comparison to {output_path}")
    return fig


def plot_measure_comparison(output_path: Path | None = None) -> plt.Figure:
    """Box plot comparing IOp shares by inequality measure."""
    df = _load_successful_df()
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=(10, 5))
    order = df.groupby("inequality_measure")["iop_share"].median().sort_values().index
    sns.boxplot(data=df, x="inequality_measure", y="iop_share", order=order, ax=ax,
                color="steelblue", width=0.6)
    ax.set_xlabel("Inequality Measure")
    ax.set_ylabel("IOp Share")
    ax.set_title("IOp Estimates by Inequality Measure")
    plt.xticks(rotation=30)

    output_path = output_path or FIGURES_DIR / "measure_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved measure comparison to {output_path}")
    return fig


def generate_all_figures() -> None:
    """Generate all publication figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_iop_distribution()
    plot_method_comparison()
    plot_measure_comparison()
    # Spec curve is in its own module
    from synthesis.spec_curve import plot_specification_curve
    plot_specification_curve()
    logger.info("All figures generated")


if __name__ == "__main__":
    generate_all_figures()
