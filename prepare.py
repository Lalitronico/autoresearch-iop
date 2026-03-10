"""FIXED: Data preparation for ESRU-EMOVI 2023.

Loads raw data, constructs analytical variables, generates codebook,
and saves the analytical dataset as parquet.

Usage:
    python prepare.py [--input path/to/raw/data] [--output path/to/output]
    python prepare.py --synthetic  # Generate test data without real ESRU-EMOVI
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CODEBOOK_PATH = DATA_DIR / "codebook.json"
OUTPUT_PATH = PROCESSED_DIR / "emovi_analytical.parquet"


def load_raw_data(input_path: Path | None = None) -> pd.DataFrame:
    """Load raw ESRU-EMOVI data from .dta, .csv, or .parquet.

    Searches RAW_DIR for common file patterns if no path specified.
    """
    if input_path is not None:
        return _read_file(input_path)

    # Auto-detect raw data files
    for ext in ["*.dta", "*.csv", "*.parquet", "*.xlsx"]:
        files = list(RAW_DIR.glob(ext))
        if files:
            logger.info(f"Found raw data: {files[0]}")
            return _read_file(files[0])

    raise FileNotFoundError(
        f"No raw data found in {RAW_DIR}. "
        f"Place ESRU-EMOVI data (.dta, .csv, .parquet, or .xlsx) in {RAW_DIR}"
    )


def _read_file(path: Path) -> pd.DataFrame:
    """Read a data file based on extension."""
    ext = path.suffix.lower()
    if ext == ".dta":
        try:
            return pd.read_stata(path)
        except ValueError:
            # Some .dta files have duplicate value labels (e.g. municipality names)
            return pd.read_stata(path, convert_categoricals=False)
    elif ext == ".csv":
        return pd.read_csv(path)
    elif ext == ".parquet":
        return pd.read_parquet(path)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def _skin_tone_letter_to_num(val) -> float | None:
    """Convert skin tone letter code (A-K) to numeric 1-11."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().upper()
    if len(s) == 1 and "A" <= s <= "K":
        return float(ord(s) - ord("A") + 1)
    # Try if already numeric
    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan


def construct_analytical_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Construct standardized analytical variables from raw ESRU-EMOVI 2023 columns.

    Maps real ESRU-EMOVI variable names to standardized pipeline names.
    Based on actual entrevistado_2023.dta structure.
    """
    out = pd.DataFrame(index=df.index)

    # === Income variables ===
    # ingc_pc: imputed per capita household income (no missing)
    if "ingc_pc" in df.columns:
        out["hh_pc_imputed"] = pd.to_numeric(df["ingc_pc"], errors="coerce")
        logger.info(f"  Mapped: ingc_pc -> hh_pc_imputed ({out['hh_pc_imputed'].notna().sum()} valid)")
    else:
        logger.warning("  ingc_pc not found in raw data")

    # p101: self-reported total monthly household income (25% missing)
    # Codes 999998, 999999 = "don't know" / "refused" -> NaN
    if "p101" in df.columns:
        hh_total = pd.to_numeric(df["p101"], errors="coerce")
        hh_total = hh_total.replace({999998: np.nan, 999999: np.nan})
        hh_total = hh_total.where(hh_total > 0, np.nan)
        out["hh_total_reported"] = hh_total
        n_valid = hh_total.notna().sum()
        pct_miss = hh_total.isna().mean() * 100
        logger.info(f"  Mapped: p101 -> hh_total_reported ({n_valid} valid, {pct_miss:.1f}% missing)")
    else:
        logger.warning("  p101 not found in raw data")

    # === Circumstance variables ===
    # educp: Father's education (categorical)
    if "educp" in df.columns:
        out["father_education"] = df["educp"].copy()
        logger.info(f"  Mapped: educp -> father_education ({df['educp'].nunique()} categories)")
    else:
        logger.warning("  educp not found in raw data")

    # educm: Mother's education (categorical)
    if "educm" in df.columns:
        out["mother_education"] = df["educm"].copy()
        logger.info(f"  Mapped: educm -> mother_education ({df['educm'].nunique()} categories)")
    else:
        logger.warning("  educm not found in raw data")

    # clasep: Father's occupational class (categorical)
    if "clasep" in df.columns:
        out["father_occupation"] = df["clasep"].copy()
        logger.info(f"  Mapped: clasep -> father_occupation ({df['clasep'].nunique()} categories)")
    else:
        logger.warning("  clasep not found in raw data")

    # p110: Ethnic self-identification (categorical, 5 categories)
    if "p110" in df.columns:
        out["ethnicity"] = df["p110"].copy()
        logger.info(f"  Mapped: p110 -> ethnicity ({df['p110'].nunique()} categories)")
    else:
        logger.warning("  p110 not found in raw data")

    # p111: Indigenous language speaker (binary)
    if "p111" in df.columns:
        out["indigenous_language"] = pd.to_numeric(df["p111"], errors="coerce")
        logger.info(f"  Mapped: p111 -> indigenous_language")
    else:
        logger.warning("  p111 not found in raw data")

    # p112: Skin tone self-report (letters A-K -> numeric 1-11)
    if "p112" in df.columns:
        out["skin_tone"] = df["p112"].apply(_skin_tone_letter_to_num)
        n_valid = out["skin_tone"].notna().sum()
        logger.info(f"  Mapped: p112 -> skin_tone (letters->numeric, {n_valid} valid)")
    else:
        logger.warning("  p112 not found in raw data")

    # p113dL: Skin tone CIELab L* from colorimeter (continuous)
    if "p113dL" in df.columns:
        out["skin_tone_cielab"] = pd.to_numeric(df["p113dL"], errors="coerce")
        logger.info(f"  Mapped: p113dL -> skin_tone_cielab")
    elif "p113dl" in df.columns:
        out["skin_tone_cielab"] = pd.to_numeric(df["p113dl"], errors="coerce")
        logger.info(f"  Mapped: p113dl -> skin_tone_cielab")
    else:
        logger.warning("  p113dL not found in raw data")

    # region_14: Region where lived at age 14 (categorical, 5 regions)
    if "region_14" in df.columns:
        out["region_14"] = df["region_14"].copy()
        logger.info(f"  Mapped: region_14 -> region_14 ({df['region_14'].nunique()} regions)")
    else:
        logger.warning("  region_14 not found in raw data")

    # p21: Urbanity at age 14 -> binarize to rural_14
    # Original has 5 levels; binarize: rural/semi-rural vs urban
    if "p21" in df.columns:
        p21 = pd.to_numeric(df["p21"], errors="coerce")
        # Typical coding: 1=rural, 2=semi-rural, 3=semi-urban, 4=urban, 5=very urban
        # Binarize: 1-2 = rural (1), 3-5 = urban (0)
        out["rural_14"] = (p21 <= 2).astype(float)
        out.loc[p21.isna(), "rural_14"] = np.nan
        logger.info(f"  Mapped: p21 -> rural_14 (binarized)")
    else:
        logger.warning("  p21 not found in raw data")

    # sexo: Gender (1=Male, 2=Female -> 1=Male, 0=Female)
    if "sexo" in df.columns:
        sexo = pd.to_numeric(df["sexo"], errors="coerce")
        out["gender"] = (sexo == 1).astype(float)
        out.loc[sexo.isna(), "gender"] = np.nan
        logger.info(f"  Mapped: sexo -> gender (1=male, 0=female)")
    else:
        logger.warning("  sexo not found in raw data")

    # === Demographics for filtering ===
    # edad: Age
    if "edad" in df.columns:
        out["age"] = pd.to_numeric(df["edad"], errors="coerce")
        logger.info(f"  Mapped: edad -> age (range {out['age'].min():.0f}-{out['age'].max():.0f})")
    else:
        logger.warning("  edad not found in raw data")

    # rururb: Urban/rural at time of survey (1=urbano, 2=rural -> 1/0)
    if "rururb" in df.columns:
        rururb = pd.to_numeric(df["rururb"], errors="coerce")
        out["urban"] = (rururb == 1).astype(float)
        out.loc[rururb.isna(), "urban"] = np.nan
        logger.info(f"  Mapped: rururb -> urban (1=urban, 0=rural)")
    else:
        logger.warning("  rururb not found in raw data")

    # factor: Survey sampling weight
    if "factor" in df.columns:
        out["weight"] = pd.to_numeric(df["factor"], errors="coerce")
        logger.info(f"  Mapped: factor -> weight")
    else:
        logger.warning("  factor not found in raw data -- unweighted analysis")

    # === Convert categorical columns to numeric codes for ML methods ===
    for col in ["father_education", "mother_education", "father_occupation",
                "ethnicity", "region_14"]:
        if col in out.columns and out[col].dtype == "category":
            out[col] = out[col].cat.codes.replace(-1, np.nan).astype(float)
        elif col in out.columns and out[col].dtype == object:
            out[col] = pd.Categorical(out[col]).codes.replace(-1, np.nan).astype(float)

    logger.info(f"Analytical dataset: {len(out)} obs, {len(out.columns)} variables")
    logger.info(f"Columns: {list(out.columns)}")
    return out


def generate_codebook(df: pd.DataFrame) -> dict:
    """Generate codebook metadata for the analytical dataset."""
    codebook = {
        "n_observations": len(df),
        "n_variables": len(df.columns),
        "variables": {},
    }

    circumstance_vars = {
        "father_education", "mother_education", "father_occupation",
        "ethnicity", "indigenous_language", "skin_tone", "skin_tone_cielab",
        "region_14", "rural_14", "gender",
    }
    income_vars = {"hh_pc_imputed", "hh_total_reported"}

    for col in df.columns:
        info: dict = {
            "dtype": str(df[col].dtype),
            "n_missing": int(df[col].isna().sum()),
            "pct_missing": float(df[col].isna().mean() * 100),
        }
        if col in circumstance_vars:
            info["role"] = "circumstance"
        elif col in income_vars:
            info["role"] = "income"
        elif col == "weight":
            info["role"] = "weight"
        else:
            info["role"] = "demographic"

        if df[col].dtype in ("object", "category") or df[col].nunique() < 20:
            info["unique_values"] = int(df[col].nunique())
            vc = df[col].value_counts(dropna=False).head(15)
            info["value_counts"] = {str(k): int(v) for k, v in vc.items()}
        else:
            if df[col].dtype.kind in "iufb":
                info["mean"] = float(df[col].mean())
                info["std"] = float(df[col].std())
                info["min"] = float(df[col].min())
                info["max"] = float(df[col].max())
                info["median"] = float(df[col].median())
                info["p25"] = float(df[col].quantile(0.25))
                info["p75"] = float(df[col].quantile(0.75))

        codebook["variables"][col] = info

    return codebook


def create_synthetic_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Create synthetic ESRU-EMOVI-like data for testing.

    Matches the analytical variable structure produced by construct_analytical_variables().
    """
    rng = np.random.default_rng(seed)

    df = pd.DataFrame()

    # Circumstances (matching real ESRU-EMOVI categories)
    # Father's education: 4 categories (0=primaria o menos, 1=secundaria, 2=media superior, 3=profesional)
    df["father_education"] = rng.choice([0, 1, 2, 3], size=n, p=[0.35, 0.25, 0.22, 0.18]).astype(float)
    # Mother's education: same 4 categories
    df["mother_education"] = rng.choice([0, 1, 2, 3], size=n, p=[0.40, 0.25, 0.20, 0.15]).astype(float)
    # Father's occupation: 6 categories
    df["father_occupation"] = rng.choice([0, 1, 2, 3, 4, 5], size=n,
                                          p=[0.20, 0.20, 0.18, 0.17, 0.15, 0.10]).astype(float)
    # Ethnicity: 5 categories
    df["ethnicity"] = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.40, 0.25, 0.15, 0.12, 0.08]).astype(float)
    # Indigenous language: binary
    df["indigenous_language"] = rng.choice([0.0, 1.0], size=n, p=[0.90, 0.10])
    # Skin tone: 1-11 scale (from letters A-K)
    df["skin_tone"] = rng.choice(np.arange(1, 12, dtype=float), size=n)
    # Skin tone CIELab: continuous ~40-80
    df["skin_tone_cielab"] = rng.normal(55, 10, size=n).clip(30, 85)
    # Region at 14: 5 regions
    df["region_14"] = rng.choice([0, 1, 2, 3, 4], size=n).astype(float)
    # Rural at 14: binary
    df["rural_14"] = rng.choice([0.0, 1.0], size=n, p=[0.55, 0.45])
    # Gender: binary (1=male, 0=female)
    df["gender"] = rng.choice([0.0, 1.0], size=n, p=[0.50, 0.50])

    # Demographics
    df["age"] = rng.integers(25, 65, size=n).astype(float)
    df["urban"] = rng.choice([0.0, 1.0], size=n, p=[0.30, 0.70])

    # Income: influenced by circumstances (what IOp measures)
    log_income = (
        8.5  # base log income
        + 0.20 * df["father_education"]
        + 0.15 * df["mother_education"]
        + 0.10 * df["father_occupation"]
        - 0.08 * df["ethnicity"]
        - 0.15 * df["indigenous_language"]
        - 0.02 * df["skin_tone"]
        + 0.05 * df["gender"]
        + 0.10 * df["urban"]
        - 0.08 * df["rural_14"]
        + 0.01 * df["age"]
        + rng.normal(0, 0.6, size=n)  # effort + luck
    )

    # hh_pc_imputed: primary income variable (no missing, all positive)
    df["hh_pc_imputed"] = np.exp(log_income) * rng.uniform(0.8, 1.2, size=n)

    # hh_total_reported: ~25% missing
    df["hh_total_reported"] = df["hh_pc_imputed"] * rng.uniform(2.0, 5.0, size=n)
    missing_mask = rng.random(size=n) < 0.25
    df.loc[missing_mask, "hh_total_reported"] = np.nan

    # Survey weight
    df["weight"] = rng.uniform(0.5, 3.0, size=n)

    return df


def main(input_path: Path | None = None, output_path: Path | None = None, synthetic: bool = False):
    """Main preparation pipeline."""
    output_path = output_path or OUTPUT_PATH

    if synthetic:
        logger.info("Generating synthetic ESRU-EMOVI data for testing...")
        df_raw = create_synthetic_data()
        df_analytical = df_raw  # Already in analytical format
    else:
        logger.info("Loading raw ESRU-EMOVI data...")
        df_raw = load_raw_data(input_path)
        logger.info(f"Raw data: {len(df_raw)} observations, {len(df_raw.columns)} variables")

        logger.info("Constructing analytical variables...")
        df_analytical = construct_analytical_variables(df_raw)

    # Generate codebook
    logger.info("Generating codebook...")
    codebook = generate_codebook(df_analytical)

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_analytical.to_parquet(output_path, index=False)
    logger.info(f"Saved analytical dataset to {output_path}")

    with open(CODEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(codebook, f, indent=2, default=str)
    logger.info(f"Saved codebook to {CODEBOOK_PATH}")

    # EDA summary
    logger.info("\n=== Dataset Summary ===")
    logger.info(f"Observations: {len(df_analytical)}")
    logger.info(f"Variables: {list(df_analytical.columns)}")
    for col in df_analytical.columns:
        n_miss = df_analytical[col].isna().sum()
        logger.info(f"  {col}: {df_analytical[col].dtype}, {n_miss} missing ({n_miss/len(df_analytical)*100:.1f}%)")

    return df_analytical, codebook


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ESRU-EMOVI data for IOp analysis")
    parser.add_argument("--input", type=Path, help="Path to raw data file")
    parser.add_argument("--output", type=Path, help="Output path for analytical dataset")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic test data")
    args = parser.parse_args()

    main(input_path=args.input, output_path=args.output, synthetic=args.synthetic)
