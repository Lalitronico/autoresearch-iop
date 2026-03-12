"""Shared test configuration.

Redirects data and log paths to temporary directories so tests
never overwrite production data or experiment logs.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prepare import create_synthetic_data, generate_codebook


@pytest.fixture(autouse=True, scope="session")
def isolate_test_paths(tmp_path_factory):
    """Redirect all production paths to temp directories for the test session.

    This prevents tests from:
    - Overwriting emovi_analytical.parquet with synthetic data
    - Deleting/truncating experiment_log.jsonl
    """
    import core.data_loader as dl
    import orchestration.experiment_log as el

    tmp = tmp_path_factory.mktemp("autoresearch_test")
    test_processed = tmp / "data" / "processed"
    test_processed.mkdir(parents=True)
    test_results = tmp / "results"
    test_results.mkdir(parents=True)

    # Save original paths
    orig_processed_dir = dl.PROCESSED_DIR
    orig_analytical = dl.ANALYTICAL_FILE
    orig_codebook = dl.CODEBOOK_PATH
    orig_imputed_dir = dl.IMPUTED_DIR
    orig_results_dir = el.RESULTS_DIR
    orig_jsonl = el.JSONL_PATH
    orig_tsv = el.TSV_PATH

    # Redirect data paths
    dl.PROCESSED_DIR = test_processed
    dl.ANALYTICAL_FILE = test_processed / "emovi_analytical.parquet"
    dl.CODEBOOK_PATH = tmp / "data" / "codebook.json"
    dl.IMPUTED_DIR = test_processed / "imputed"

    # Redirect log paths
    el.RESULTS_DIR = test_results
    el.JSONL_PATH = test_results / "experiment_log.jsonl"
    el.TSV_PATH = test_results / "experiment_log.tsv"

    # Create synthetic data in the test directory
    df = create_synthetic_data(n=500, seed=42)
    df.to_parquet(dl.ANALYTICAL_FILE, index=False)
    codebook = generate_codebook(df)
    with open(dl.CODEBOOK_PATH, "w") as f:
        json.dump(codebook, f, indent=2, default=str)

    # Generate small imputed datasets (M=3) for tests
    _create_test_imputed_data(df, dl.IMPUTED_DIR)

    yield tmp

    # Restore original paths
    dl.PROCESSED_DIR = orig_processed_dir
    dl.ANALYTICAL_FILE = orig_analytical
    dl.CODEBOOK_PATH = orig_codebook
    dl.IMPUTED_DIR = orig_imputed_dir
    el.RESULTS_DIR = orig_results_dir
    el.JSONL_PATH = orig_jsonl
    el.TSV_PATH = orig_tsv


def _create_test_imputed_data(df: pd.DataFrame, imputed_dir: Path):
    """Create M=3 imputed datasets for test use.

    Introduces synthetic missingness in a few circ columns then fills with
    simple random imputation (not MICE — avoids miceforest dependency in tests).
    """
    import numpy as np

    rng = np.random.default_rng(42)
    m = 3

    # Introduce ~10% missing in some circumstance columns
    cols_to_make_missing = ["father_education", "mother_education", "ethnicity"]
    df_with_missing = df.copy()
    for col in cols_to_make_missing:
        if col in df_with_missing.columns:
            mask = rng.random(len(df_with_missing)) < 0.10
            df_with_missing.loc[mask, col] = np.nan

    imputed_dir.mkdir(parents=True, exist_ok=True)
    for i in range(m):
        imp_df = df_with_missing.copy()
        for col in cols_to_make_missing:
            if col in imp_df.columns:
                missing_mask = imp_df[col].isna()
                n_missing = missing_mask.sum()
                if n_missing > 0:
                    valid_vals = imp_df.loc[~missing_mask, col].values
                    imp_df.loc[missing_mask, col] = rng.choice(valid_vals, size=n_missing)
        imp_df.to_parquet(imputed_dir / f"m_{i:02d}.parquet", index=False)

    # Save metadata
    meta = {
        "m": m,
        "n_obs": len(df),
        "seed": 42,
        "vars_imputed": cols_to_make_missing,
        "missingness_rates": {c: 10.0 for c in cols_to_make_missing},
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
    with open(imputed_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


@pytest.fixture(scope="session")
def synthetic_data(isolate_test_paths):
    """Provide the synthetic DataFrame for tests. Data is already saved by isolate_test_paths."""
    import core.data_loader as dl
    return pd.read_parquet(dl.ANALYTICAL_FILE)
