"""DataRegistry: loads ESRU-EMOVI data and provides validated access.

Handles:
- Loading raw data (Stata .dta or CSV)
- Constructing analytical variables
- Applying sample filters
- Validating that requested variables exist
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.types import Circumstance, IncomeVariable, SampleFilter

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CODEBOOK_PATH = DATA_DIR / "codebook.json"

ANALYTICAL_FILE = PROCESSED_DIR / "emovi_analytical.parquet"


class DataRegistry:
    """Central access point for ESRU-EMOVI data."""

    def __init__(self, data_path: Path | None = None):
        self._data_path = data_path or ANALYTICAL_FILE
        self._df: pd.DataFrame | None = None
        self._codebook: dict[str, Any] | None = None

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self.load()
        return self._df

    @property
    def codebook(self) -> dict[str, Any]:
        if self._codebook is None:
            self._load_codebook()
        return self._codebook

    def load(self) -> pd.DataFrame:
        """Load the analytical dataset."""
        if not self._data_path.exists():
            raise FileNotFoundError(
                f"Analytical dataset not found at {self._data_path}. "
                f"Run prepare.py first to create it."
            )
        logger.info(f"Loading data from {self._data_path}")
        self._df = pd.read_parquet(self._data_path)
        logger.info(f"Loaded {len(self._df)} observations, {len(self._df.columns)} columns")
        return self._df

    def _load_codebook(self) -> None:
        if CODEBOOK_PATH.exists():
            with open(CODEBOOK_PATH) as f:
                self._codebook = json.load(f)
        else:
            self._codebook = {}
            logger.warning(f"Codebook not found at {CODEBOOK_PATH}")

    def get_income(self, income_var: str) -> pd.Series:
        """Get income variable, applying log transform if needed."""
        iv = IncomeVariable(income_var)
        base_col = iv.base_variable

        if base_col not in self.df.columns:
            raise KeyError(
                f"Income variable '{base_col}' not in dataset. "
                f"Available: {[c for c in self.df.columns if 'income' in c.lower() or 'ingreso' in c.lower()]}"
            )

        y = self.df[base_col].copy()
        if iv.is_log:
            # Replace non-positive with NaN before log
            y = y.where(y > 0, np.nan)
            y = np.log(y)
        return y

    def get_circumstances(self, circ_names: list[str]) -> pd.DataFrame:
        """Get circumstance variables as a DataFrame."""
        missing = [c for c in circ_names if c not in self.df.columns]
        if missing:
            raise KeyError(
                f"Circumstance variables not in dataset: {missing}. "
                f"Available: {list(self.df.columns)}"
            )
        return self.df[circ_names].copy()

    def apply_filter(self, sample_filter: str) -> pd.DataFrame:
        """Apply sample restriction and return filtered DataFrame."""
        sf = SampleFilter(sample_filter)
        df = self.df.copy()

        filters = {
            SampleFilter.ALL: lambda d: d,
            SampleFilter.MALE: lambda d: d[d["gender"] == 1],
            SampleFilter.FEMALE: lambda d: d[d["gender"] == 0],
            SampleFilter.AGE_25_44: lambda d: d[d["age"].between(25, 44)],
            SampleFilter.AGE_45_64: lambda d: d[d["age"].between(45, 64)],
            SampleFilter.AGE_25_55: lambda d: d[d["age"].between(25, 55)],
            SampleFilter.URBAN: lambda d: d[d["urban"] == 1],
            SampleFilter.RURAL: lambda d: d[d["urban"] == 0],
            SampleFilter.URBAN_MALE: lambda d: d[(d["urban"] == 1) & (d["gender"] == 1)],
            SampleFilter.URBAN_FEMALE: lambda d: d[(d["urban"] == 1) & (d["gender"] == 0)],
            SampleFilter.RURAL_MALE: lambda d: d[(d["urban"] == 0) & (d["gender"] == 1)],
            SampleFilter.RURAL_FEMALE: lambda d: d[(d["urban"] == 0) & (d["gender"] == 0)],
            SampleFilter.COHORT_1: lambda d: d[d["age"].between(25, 34)],
            SampleFilter.COHORT_2: lambda d: d[d["age"].between(35, 44)],
            SampleFilter.COHORT_3: lambda d: d[d["age"].between(45, 54)],
            SampleFilter.COHORT_4: lambda d: d[d["age"].between(55, 64)],
        }

        filter_fn = filters.get(sf)
        if filter_fn is None:
            raise ValueError(f"No filter implementation for {sf}")
        return filter_fn(df)

    def validate_spec(self, spec) -> list[str]:
        """Validate that a spec's variables exist in the dataset."""
        errors: list[str] = []
        iv = IncomeVariable(spec.income_variable)
        base = iv.base_variable
        if base not in self.df.columns:
            errors.append(f"Income variable '{base}' not in dataset")
        for c in spec.circumstances:
            if c not in self.df.columns:
                errors.append(f"Circumstance '{c}' not in dataset")
        return errors

    def get_sample_for_spec(self, spec) -> tuple[pd.Series, pd.DataFrame, pd.Index]:
        """Get filtered income and circumstances for a spec.

        Returns (y, X_circs, valid_index) with NaN rows dropped.
        """
        filtered_df = self.apply_filter(spec.sample_filter)
        # Temporarily set the filtered df
        original_df = self._df
        self._df = filtered_df

        try:
            y = self.get_income(spec.income_variable)
            X = self.get_circumstances(list(spec.circumstances))

            # Combine and drop NaN
            combined = pd.concat([y.rename("__income__"), X], axis=1).dropna()
            y_clean = combined["__income__"]
            X_clean = combined.drop(columns=["__income__"])

            return y_clean, X_clean, combined.index
        finally:
            self._df = original_df
