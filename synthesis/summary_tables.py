"""Summary tables for IOp multiverse analysis.

Generates tables in LaTeX and Markdown format for publication.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from orchestration.experiment_log import load_experiment_log

logger = logging.getLogger(__name__)

TABLES_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"


def _load_successful() -> pd.DataFrame:
    """Load successful experiments as DataFrame."""
    records = load_experiment_log()
    successful = [r for r in records if r.get("status") == "success"]
    if not successful:
        return pd.DataFrame()
    df = pd.DataFrame(successful)
    return df


def iop_by_method_measure() -> pd.DataFrame:
    """Cross-table: IOp share by method x inequality measure."""
    df = _load_successful()
    if df.empty:
        return df

    pivot = df.pivot_table(
        values="iop_share",
        index="method",
        columns="inequality_measure",
        aggfunc=["mean", "std", "count"],
    )
    return pivot


def iop_by_circumstances() -> pd.DataFrame:
    """IOp share by number of circumstances included."""
    df = _load_successful()
    if df.empty:
        return df

    df["n_circs"] = df["circumstances"].apply(
        lambda x: len(x) if isinstance(x, (list, tuple)) else 0
    )
    summary = df.groupby("n_circs").agg(
        mean_iop=("iop_share", "mean"),
        std_iop=("iop_share", "std"),
        min_iop=("iop_share", "min"),
        max_iop=("iop_share", "max"),
        n_specs=("iop_share", "count"),
    ).round(4)
    return summary


def iop_by_sample() -> pd.DataFrame:
    """IOp share by sample filter."""
    df = _load_successful()
    if df.empty:
        return df

    summary = df.groupby("sample_filter").agg(
        mean_iop=("iop_share", "mean"),
        std_iop=("iop_share", "std"),
        n_specs=("iop_share", "count"),
        median_iop=("iop_share", "median"),
    ).round(4)
    return summary


def generate_latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    output_path: Path | None = None,
) -> str:
    """Convert DataFrame to LaTeX table string."""
    latex = df.to_latex(
        caption=caption,
        label=label,
        float_format="%.4f",
        na_rep="--",
    )

    if output_path:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(latex)
        logger.info(f"Saved LaTeX table to {output_path}")

    return latex


def generate_markdown_table(df: pd.DataFrame) -> str:
    """Convert DataFrame to Markdown table string."""
    return df.to_markdown(floatfmt=".4f")


def generate_all_tables() -> None:
    """Generate all summary tables and save to results/tables/."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Method x Measure
    mm = iop_by_method_measure()
    if not mm.empty:
        generate_latex_table(
            mm, "IOp share by estimation method and inequality measure",
            "tab:method_measure", TABLES_DIR / "method_measure.tex"
        )

    # By circumstances
    bc = iop_by_circumstances()
    if not bc.empty:
        generate_latex_table(
            bc, "IOp share by number of circumstances",
            "tab:by_circumstances", TABLES_DIR / "by_circumstances.tex"
        )
        md = generate_markdown_table(bc)
        (TABLES_DIR / "by_circumstances.md").write_text(md, encoding="utf-8")

    # By sample
    bs = iop_by_sample()
    if not bs.empty:
        generate_latex_table(
            bs, "IOp share by sample restriction",
            "tab:by_sample", TABLES_DIR / "by_sample.tex"
        )

    logger.info("All summary tables generated")


if __name__ == "__main__":
    generate_all_tables()
