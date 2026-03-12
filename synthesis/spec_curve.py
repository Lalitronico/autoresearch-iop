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
            "use_mi": r.get("use_mi", False),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("iop_share").reset_index(drop=True)
    return df


def plot_specification_curve(
    df: pd.DataFrame | None = None,
    output_path: Path | None = None,
    title: str = "Specification Curve: Inequality of Opportunity",
) -> plt.Figure:
    """Create the specification curve plot (Simonsohn et al., 2020 style).

    Two-panel figure: top panel shows IOp shares sorted by magnitude with CIs,
    bottom panel shows indicator grid for which choices are active per spec.
    Proportions: ~40% top, ~60% bottom (adapts to number of indicator rows).

    Parameters
    ----------
    df : DataFrame
        Spec curve data (from build_spec_curve_data). Auto-loaded if None.
    output_path : Path
        Where to save the figure. Defaults to results/figures/spec_curve.png.
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

    total_rows = sum(len(vals) for vals in dimensions.values())

    # --- Adaptive figure sizing ---
    # Top panel: fixed 3.5 inches. Bottom panel: ~0.3 inches per row.
    top_height = 3.5
    bot_height = max(total_rows * 0.3, 3.0)
    fig_height = top_height + bot_height + 0.8  # margins
    fig_width = min(max(n_specs * 0.025, 10), 16)  # scale with specs, cap at 16

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(
        2, 1, height_ratios=[top_height, bot_height], hspace=0.08, figure=fig
    )

    # --- Top panel: IOp shares with CIs ---
    ax_top = fig.add_subplot(gs[0])
    x = np.arange(n_specs)

    ax_top.fill_between(
        x, df["ci_lower"], df["ci_upper"],
        alpha=0.2, color="steelblue", label="95% CI"
    )
    ax_top.scatter(x, df["iop_share"], s=6, color="steelblue", zorder=3)

    median_iop = df["iop_share"].median()
    ax_top.axhline(median_iop, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_top.text(n_specs * 0.02, median_iop + 0.01, f"Median: {median_iop:.3f}",
                fontsize=9, color="gray")

    ax_top.set_ylabel("IOp Share", fontsize=10)
    ax_top.set_title(title, fontsize=12, fontweight="bold")
    ax_top.set_xlim(-0.5, n_specs - 0.5)
    ax_top.set_ylim(0, min(df["ci_upper"].max() * 1.1, 1.0))
    ax_top.tick_params(labelbottom=False)
    ax_top.legend(loc="upper left", fontsize=8)

    # --- Bottom panel: specification indicators ---
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    row_idx = 0
    y_ticks = []
    y_labels = []
    dim_col_map = {
        "Method": "method",
        "Decomposition": "decomposition_type",
        "Income": "income_variable",
        "Measure": "inequality_measure",
        "Sample": "sample_filter",
    }

    for dim_name, values in dimensions.items():
        for val in sorted(values):
            active = df[dim_col_map[dim_name]] == val
            ax_bot.scatter(
                x[active], [row_idx] * active.sum(),
                s=4, color="steelblue", marker="s", linewidths=0,
            )
            y_ticks.append(row_idx)
            y_labels.append(val)
            row_idx += 1

        if row_idx < total_rows:
            ax_bot.axhline(row_idx - 0.5, color="lightgray", linewidth=0.5)

    ax_bot.set_yticks(y_ticks)
    ax_bot.set_yticklabels(y_labels, fontsize=7)
    ax_bot.set_ylim(-0.5, total_rows - 0.5)
    ax_bot.invert_yaxis()
    ax_bot.set_xlabel(f"Specification (sorted by IOp share, n={n_specs})", fontsize=10)
    ax_bot.tick_params(labelbottom=False)

    # Dimension group labels on the right margin
    dim_start = 0
    for dim_name, values in dimensions.items():
        mid = dim_start + len(values) / 2 - 0.5
        ax_bot.text(
            1.01, mid, dim_name,
            fontsize=8, fontweight="bold", va="center",
            transform=ax_bot.get_yaxis_transform(),
        )
        dim_start += len(values)

    # Save
    if output_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FIGURES_DIR / "spec_curve.png"

    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.3)
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.3)
    logger.info(f"Saved specification curve to {output_path} and {pdf_path}")
    plt.close(fig)

    return fig


if __name__ == "__main__":
    plot_specification_curve()
