"""Shared test configuration.

Redirects data and log paths to temporary directories so tests
never overwrite production data or experiment logs.
"""

import json
from pathlib import Path

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
    orig_results_dir = el.RESULTS_DIR
    orig_jsonl = el.JSONL_PATH
    orig_tsv = el.TSV_PATH

    # Redirect data paths
    dl.PROCESSED_DIR = test_processed
    dl.ANALYTICAL_FILE = test_processed / "emovi_analytical.parquet"
    dl.CODEBOOK_PATH = tmp / "data" / "codebook.json"

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

    yield tmp

    # Restore original paths
    dl.PROCESSED_DIR = orig_processed_dir
    dl.ANALYTICAL_FILE = orig_analytical
    dl.CODEBOOK_PATH = orig_codebook
    el.RESULTS_DIR = orig_results_dir
    el.JSONL_PATH = orig_jsonl
    el.TSV_PATH = orig_tsv


@pytest.fixture(scope="session")
def synthetic_data(isolate_test_paths):
    """Provide the synthetic DataFrame for tests. Data is already saved by isolate_test_paths."""
    import core.data_loader as dl
    return pd.read_parquet(dl.ANALYTICAL_FILE)
