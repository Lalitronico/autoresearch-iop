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
    return df.fillna("--").to_markdown(floatfmt=".4f")


def iop_by_circ_set() -> pd.DataFrame:
    """IOp share by specific circumstance set (frozen set), with readable labels."""
    df = _load_successful()
    if df.empty:
        return df

    # Create a readable label from circumstances list
    def circ_label(circs):
        if not isinstance(circs, (list, tuple)):
            return "unknown"
        n = len(circs)
        if n <= 3:
            return " + ".join(c.replace("_", " ") for c in sorted(circs))
        return f"{n} circs: " + ", ".join(sorted(circs)[:3]) + "..."

    df["circ_set"] = df["circumstances"].apply(
        lambda x: tuple(sorted(x)) if isinstance(x, (list, tuple)) else ()
    )
    df["circ_label"] = df["circumstances"].apply(circ_label)
    df["n_circs"] = df["circ_set"].apply(len)

    summary = df.groupby(["n_circs", "circ_label"]).agg(
        mean_iop=("iop_share", "mean"),
        std_iop=("iop_share", "std"),
        min_iop=("iop_share", "min"),
        max_iop=("iop_share", "max"),
        n_specs=("iop_share", "count"),
    ).round(4).sort_index()

    return summary


def iop_method_sensitivity() -> pd.DataFrame:
    """Compare IOp across methods for the same circ set (method sensitivity table).

    Shows how much IOp estimates vary by estimation method, holding
    everything else constant. Key robustness check.
    """
    df = _load_successful()
    if df.empty:
        return df

    # Filter to primary income + gini for clean comparison
    mask = (df["income_variable"] == "hh_pc_imputed") & (df["inequality_measure"] == "gini")
    sub = df[mask].copy()
    if sub.empty:
        sub = df.copy()

    sub["n_circs"] = sub["circumstances"].apply(
        lambda x: len(x) if isinstance(x, (list, tuple)) else 0
    )
    pivot = sub.pivot_table(
        values="iop_share",
        index="n_circs",
        columns="method",
        aggfunc="mean",
    ).round(4)

    return pivot


def iop_listwise_vs_mi() -> pd.DataFrame:
    """Compare IOp estimates: listwise deletion vs multiple imputation.

    Groups by method and number of circumstances, showing mean IOp share
    for listwise and MI side by side. Key sensitivity analysis table.
    """
    df = _load_successful()
    if df.empty:
        return df

    df["use_mi"] = df.get("use_mi", False).fillna(False).astype(bool)
    df["mi_label"] = df["use_mi"].map({True: "MI", False: "Listwise"})
    df["n_circs"] = df["circumstances"].apply(
        lambda x: len(x) if isinstance(x, (list, tuple)) else 0
    )

    # Only include specs that have both listwise and MI variants
    if df["use_mi"].nunique() < 2:
        logger.info("Both listwise and MI results needed for comparison table")
        return pd.DataFrame()

    pivot = df.pivot_table(
        values="iop_share",
        index=["method", "n_circs"],
        columns="mi_label",
        aggfunc=["mean", "count"],
    ).round(4)

    return pivot


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
        md = generate_markdown_table(bs)
        (TABLES_DIR / "by_sample.md").write_text(md, encoding="utf-8")

    # By specific circumstance set
    cs = iop_by_circ_set()
    if not cs.empty:
        generate_latex_table(
            cs, "IOp share by circumstance set",
            "tab:by_circ_set", TABLES_DIR / "by_circ_set.tex"
        )
        md = generate_markdown_table(cs)
        (TABLES_DIR / "by_circ_set.md").write_text(md, encoding="utf-8")

    # Method sensitivity
    ms = iop_method_sensitivity()
    if not ms.empty:
        generate_latex_table(
            ms, "IOp share by method and number of circumstances (Gini, primary income)",
            "tab:method_sensitivity", TABLES_DIR / "method_sensitivity.tex"
        )
        md = generate_markdown_table(ms)
        (TABLES_DIR / "method_sensitivity.md").write_text(md, encoding="utf-8")

    # Listwise vs MI comparison
    lm = iop_listwise_vs_mi()
    if not lm.empty:
        generate_latex_table(
            lm, "IOp share: listwise deletion vs multiple imputation",
            "tab:listwise_vs_mi", TABLES_DIR / "listwise_vs_mi.tex"
        )
        md = generate_markdown_table(lm)
        (TABLES_DIR / "listwise_vs_mi.md").write_text(md, encoding="utf-8")

    logger.info("All summary tables generated")


if __name__ == "__main__":
    generate_all_tables()
