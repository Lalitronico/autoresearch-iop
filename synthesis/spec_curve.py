"""Specification curve plot for IOp multiverse analysis.

Creates a two-panel figure:
- Top: IOp shares sorted by magnitude with confidence intervals
- Bottom: Grid showing which methodological choices are active for each spec
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from orchestration.experiment_log import load_experiment_log

logger = logging.getLogger(__name__)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"


def build_spec_curve_data() -> pd.DataFrame:
    """Load experiment log and build DataFrame for spec curve plot."""
    records = load_experiment_log()
    successful = [r for r in records if r.get("status") == "success"]

    if not successful:
        logger.warning("No successful experiments to plot")
        return pd.DataFrame()

    rows = []
    for r in successful:
        rows.append({
            "spec_id": r["spec_id"],
            "iop_share": r["iop_share"],
            "ci_lower": r.get("ci_lower", r["iop_share"]),
            "ci_upper": r.get("ci_upper", r["iop_share"]),
            "method": r["method"],
            "decomposition_type": r["decomposition_type"],
            "income_variable": r["income_variable"],
            "inequality_measure": r["inequality_measure"],
            "sample_filter": r.get("sample_filter", "all"),
            "n_circumstances": len(r.get("circumstances", [])),
            "circumstances": "|".join(r.get("circumstances", [])),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("iop_share").reset_index(drop=True)
    return df


def plot_specification_curve(
    df: pd.DataFrame | None = None,
    output_path: Path | None = None,
    figsize: tuple[float, float] = (14, 10),
    title: str = "Specification Curve: Inequality of Opportunity",
) -> plt.Figure:
    """Create the specification curve plot.

    Parameters
    ----------
    df : DataFrame
        Spec curve data (from build_spec_curve_data). Auto-loaded if None.
    output_path : Path
        Where to save the figure. Defaults to results/figures/spec_curve.png.
    figsize : tuple
        Figure size.
    title : str
        Main title.

    Returns
    -------
    matplotlib Figure
    """
    if df is None:
        df = build_spec_curve_data()

    if df.empty:
        logger.warning("No data to plot")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No successful experiments", ha="center", va="center")
        return fig

    n_specs = len(df)

    # --- Dimension categories for bottom panel ---
    dimensions = {
        "Method": df["method"].unique(),
        "Decomposition": df["decomposition_type"].unique(),
        "Income": df["income_variable"].unique(),
        "Measure": df["inequality_measure"].unique(),
        "Sample": df["sample_filter"].unique(),
    }

    # Count total rows in bottom panel
    total_rows = sum(len(vals) for vals in dimensions.values())

    # --- Create figure ---
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        2, 1, height_ratios=[2, total_rows / 3], hspace=0.05, figure=fig
    )

    # --- Top panel: IOp shares with CIs ---
    ax_top = fig.add_subplot(gs[0])
    x = np.arange(n_specs)

    # CI bars
    ax_top.fill_between(
        x, df["ci_lower"], df["ci_upper"],
        alpha=0.2, color="steelblue", label="95% CI"
    )
    # Point estimates
    ax_top.scatter(x, df["iop_share"], s=8, color="steelblue", zorder=3)

    # Median line
    median_iop = df["iop_share"].median()
    ax_top.axhline(median_iop, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_top.text(n_specs * 0.02, median_iop + 0.01, f"Median: {median_iop:.3f}",
                fontsize=8, color="gray")

    ax_top.set_ylabel("IOp Share")
    ax_top.set_title(title, fontsize=13, fontweight="bold")
    ax_top.set_xlim(-0.5, n_specs - 0.5)
    ax_top.set_ylim(0, min(df["ci_upper"].max() * 1.1, 1.0))
    ax_top.tick_params(labelbottom=False)
    ax_top.legend(loc="upper left", fontsize=8)

    # --- Bottom panel: specification indicators ---
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    row_idx = 0
    y_ticks = []
    y_labels = []

    for dim_name, values in dimensions.items():
        for val in sorted(values):
            col_name = {
                "Method": "method",
                "Decomposition": "decomposition_type",
                "Income": "income_variable",
                "Measure": "inequality_measure",
                "Sample": "sample_filter",
            }[dim_name]

            active = df[col_name] == val
            ax_bot.scatter(
                x[active], [row_idx] * active.sum(),
                s=6, color="steelblue", marker="s"
            )

            y_ticks.append(row_idx)
            y_labels.append(f"{val}")
            row_idx += 1

        # Separator between dimensions
        if row_idx < total_rows:
            ax_bot.axhline(row_idx - 0.5, color="lightgray", linewidth=0.5)

    ax_bot.set_yticks(y_ticks)
    ax_bot.set_yticklabels(y_labels, fontsize=7)
    ax_bot.set_ylim(-0.5, total_rows - 0.5)
    ax_bot.invert_yaxis()
    ax_bot.set_xlabel("Specification (sorted by IOp share)")
    ax_bot.tick_params(labelbottom=False)

    # Add dimension labels on right
    dim_start = 0
    for dim_name, values in dimensions.items():
        mid = dim_start + len(values) / 2 - 0.5
        ax_bot.text(
            n_specs + 0.5, mid, dim_name,
            fontsize=8, fontweight="bold", va="center",
            transform=ax_bot.get_yaxis_transform(),
        )
        dim_start += len(values)

    plt.tight_layout()

    # Save
    if output_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FIGURES_DIR / "spec_curve.png"

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved specification curve to {output_path}")

    return fig


if __name__ == "__main__":
    plot_specification_curve()
