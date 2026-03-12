"""Multiple imputation via MICE (miceforest) for circumstance variables.

Imputes only circumstance columns. Income and demographic variables are
used as predictors but never imputed themselves.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_imputed_datasets(
    df: pd.DataFrame,
    m: int = 20,
    circ_cols: list[str] | None = None,
    income_cols: list[str] | None = None,
    predictor_cols: list[str] | None = None,
    seed: int = 42,
    iterations: int = 10,
    mean_match_candidates: int = 5,
) -> list[pd.DataFrame]:
    """Create M multiply-imputed datasets using miceforest MICE.

    Parameters
    ----------
    df : DataFrame
        Analytical dataset (e.g., emovi_analytical.parquet).
    m : int
        Number of imputed datasets to create.
    circ_cols : list[str] or None
        Circumstance columns to impute. If None, auto-detected from core.types.
    income_cols : list[str] or None
        Income columns used as predictors only (never imputed).
    predictor_cols : list[str] or None
        Additional predictor-only columns (age, gender, urban).
    seed : int
        Random seed for reproducibility.
    iterations : int
        Number of MICE iterations (convergence standard: 10).
    mean_match_candidates : int
        Number of candidates for predictive mean matching.

    Returns
    -------
    list[pd.DataFrame]
        M DataFrames, each with complete circumstance columns.
    """
    import miceforest as mf

    if circ_cols is None:
        from core.types import Circumstance
        circ_cols = [c.value for c in Circumstance]

    if income_cols is None:
        income_cols = ["hh_pc_imputed", "hh_total_reported"]

    if predictor_cols is None:
        predictor_cols = ["age", "gender", "urban"]

    # Identify which circ cols actually need imputation
    available_circs = [c for c in circ_cols if c in df.columns]
    circs_with_missing = [c for c in available_circs if df[c].isna().any()]
    circs_no_missing = [c for c in available_circs if not df[c].isna().any()]

    logger.info(f"Circumstance columns: {len(available_circs)} available, "
                f"{len(circs_with_missing)} need imputation, "
                f"{len(circs_no_missing)} complete")

    if not circs_with_missing:
        logger.info("No missing circumstances -- returning M copies of original")
        return [df.copy() for _ in range(m)]

    # Build the set of columns to include in the imputation model
    # Predictors: income + demographics (never imputed)
    available_predictors = [c for c in income_cols + predictor_cols if c in df.columns]

    # All columns for the imputation model
    model_cols = circs_with_missing + circs_no_missing + available_predictors
    model_cols = [c for c in model_cols if c in df.columns]
    # Remove duplicates preserving order
    seen = set()
    model_cols_unique = []
    for c in model_cols:
        if c not in seen:
            seen.add(c)
            model_cols_unique.append(c)
    model_cols = model_cols_unique

    df_model = df[model_cols].copy()

    # Ensure categorical columns have appropriate dtype for miceforest
    # Track which imputed cols are categorical for mean_match_strategy
    categorical_imputed = []
    for col in circs_with_missing + circs_no_missing:
        if col not in df_model.columns:
            continue
        n_unique = df_model[col].nunique()
        if n_unique <= 20:
            df_model[col] = df_model[col].astype("category")
            if col in circs_with_missing:
                categorical_imputed.append(col)

    # Define variable schema: only impute circumstance columns with missing
    variable_schema = {col: model_cols for col in circs_with_missing}

    # Use "fast" mean matching for categorical columns to avoid logodds/KDTree
    # crash when lightgbm predicts 0.0 or 1.0 probability for rare categories.
    # "fast" uses weighted random sampling from predicted class probabilities.
    mean_match_strategy = {}
    for col in circs_with_missing:
        if col in categorical_imputed:
            mean_match_strategy[col] = "fast"
        else:
            mean_match_strategy[col] = "normal"

    logger.info(f"Running MICE: m={m}, iterations={iterations}, "
                f"imputing {len(circs_with_missing)} columns using "
                f"{len(model_cols)} total predictors "
                f"({len(categorical_imputed)} categorical → fast matching)")

    # Create kernel and run
    kernel = mf.ImputationKernel(
        data=df_model,
        num_datasets=m,
        save_all_iterations_data=False,
        random_state=seed,
        mean_match_candidates=mean_match_candidates,
        mean_match_strategy=mean_match_strategy,
    )

    # min_data_in_leaf=20 stabilizes predictions for rare categories
    kernel.mice(iterations=iterations, verbose=False, min_data_in_leaf=20)

    # Extract imputed datasets
    imputed_dfs = []
    for i in range(m):
        imputed_model = kernel.complete_data(dataset=i)

        # Merge imputed circs back into full DataFrame
        result = df.copy()
        for col in circs_with_missing:
            result[col] = imputed_model[col].values

        imputed_dfs.append(result)
        n_remaining = result[available_circs].isna().sum().sum()
        if n_remaining > 0:
            logger.warning(f"  Dataset {i}: {n_remaining} NaN remaining in circs")

    logger.info(f"Created {m} imputed datasets")
    return imputed_dfs


def save_imputed_datasets(
    dfs: list[pd.DataFrame],
    output_dir: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save imputed datasets as individual parquet files with metadata.

    Parameters
    ----------
    dfs : list[DataFrame]
        M imputed datasets.
    output_dir : Path
        Directory to save to (e.g., data/processed/imputed/).
    metadata : dict or None
        Additional metadata to save.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, df in enumerate(dfs):
        path = output_dir / f"m_{i:02d}.parquet"
        df.to_parquet(path, index=False)

    # Save metadata
    meta = metadata or {}
    meta.update({
        "m": len(dfs),
        "n_obs": len(dfs[0]) if dfs else 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info(f"Saved {len(dfs)} imputed datasets to {output_dir}")


def load_imputed_dataset(output_dir: Path, m: int) -> pd.DataFrame:
    """Load a single imputed dataset by index."""
    path = Path(output_dir) / f"m_{m:02d}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Imputed dataset {m} not found at {path}")
    return pd.read_parquet(path)


def load_imputation_metadata(output_dir: Path) -> dict[str, Any]:
    """Load imputation metadata."""
    meta_path = Path(output_dir) / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Imputation metadata not found at {meta_path}")
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def validate_imputation(
    original_df: pd.DataFrame,
    imputed_dfs: list[pd.DataFrame],
    circ_cols: list[str] | None = None,
    income_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Validate imputed datasets against the original.

    Checks:
    - No NaN in circumstance columns
    - Income columns unchanged
    - Row counts match
    - Distribution checks (mean/std within reasonable range)

    Returns
    -------
    dict with validation results
    """
    if circ_cols is None:
        from core.types import Circumstance
        circ_cols = [c.value for c in Circumstance]

    if income_cols is None:
        income_cols = ["hh_pc_imputed", "hh_total_reported"]

    available_circs = [c for c in circ_cols if c in original_df.columns]
    available_income = [c for c in income_cols if c in original_df.columns]

    results: dict[str, Any] = {
        "n_datasets": len(imputed_dfs),
        "n_obs_original": len(original_df),
        "checks": {},
    }

    all_ok = True

    for i, imp_df in enumerate(imputed_dfs):
        dataset_checks: dict[str, Any] = {}

        # Row count
        dataset_checks["n_obs_match"] = len(imp_df) == len(original_df)
        if not dataset_checks["n_obs_match"]:
            all_ok = False

        # No NaN in circs
        circ_nan = {c: int(imp_df[c].isna().sum()) for c in available_circs
                    if c in imp_df.columns}
        dataset_checks["circ_nan_counts"] = circ_nan
        dataset_checks["circs_complete"] = all(v == 0 for v in circ_nan.values())
        if not dataset_checks["circs_complete"]:
            all_ok = False

        # Income unchanged
        income_match = {}
        for col in available_income:
            if col in imp_df.columns and col in original_df.columns:
                match = imp_df[col].equals(original_df[col])
                if not match:
                    # Check with NaN-aware comparison
                    match = (imp_df[col].fillna(-999) == original_df[col].fillna(-999)).all()
                income_match[col] = bool(match)
        dataset_checks["income_unchanged"] = income_match
        if not all(income_match.values()):
            all_ok = False

        results["checks"][f"dataset_{i}"] = dataset_checks

    # Distribution checks on imputed columns (across all datasets)
    dist_checks = {}
    for col in available_circs:
        if col not in original_df.columns:
            continue
        # Skip categorical columns (mean/std not meaningful)
        if hasattr(original_df[col], "cat") or hasattr(imputed_dfs[0][col], "cat"):
            dist_checks[col] = {"type": "categorical", "skipped": True}
            continue
        try:
            orig_mean = original_df[col].mean()
            orig_std = original_df[col].std()
            imp_means = [df[col].mean() for df in imputed_dfs]
            imp_stds = [df[col].std() for df in imputed_dfs]
            dist_checks[col] = {
                "original_mean": float(orig_mean) if pd.notna(orig_mean) else None,
                "original_std": float(orig_std) if pd.notna(orig_std) else None,
                "imputed_mean_range": [float(min(imp_means)), float(max(imp_means))],
                "imputed_std_range": [float(min(imp_stds)), float(max(imp_stds))],
            }
        except TypeError:
            dist_checks[col] = {"type": "categorical", "skipped": True}
    results["distribution_checks"] = dist_checks
    results["all_passed"] = all_ok

    return results
