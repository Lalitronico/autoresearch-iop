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


def _map_categorical(df: pd.DataFrame, out: pd.DataFrame,
                     src: str, dst: str) -> None:
    """Map a categorical raw column to the analytical dataset."""
    if src in df.columns:
        out[dst] = df[src].copy()
        logger.info(f"  Mapped: {src} -> {dst} ({df[src].nunique()} categories)")
    else:
        logger.warning(f"  {src} not found")


def _map_binary(df: pd.DataFrame, out: pd.DataFrame,
                src: str, dst: str) -> None:
    """Map a binary raw column to the analytical dataset."""
    if src in df.columns:
        out[dst] = pd.to_numeric(df[src], errors="coerce")
        logger.info(f"  Mapped: {src} -> {dst} (binary)")
    else:
        logger.warning(f"  {src} not found")


def _map_numeric(df: pd.DataFrame, out: pd.DataFrame,
                 src: str, dst: str) -> None:
    """Map a numeric raw column to the analytical dataset."""
    if src in df.columns:
        out[dst] = pd.to_numeric(df[src], errors="coerce")
        n_valid = out[dst].notna().sum()
        logger.info(f"  Mapped: {src} -> {dst} ({n_valid} valid)")
    else:
        logger.warning(f"  {src} not found")


def _build_count_index(df: pd.DataFrame, out: pd.DataFrame,
                       dst: str, src_cols: list[str]) -> None:
    """Build a count index from binary columns (sum of 1s).

    Treats values >= 1 as present (1), 0 or NaN as absent (0).
    Result is NaN only if ALL source columns are missing.
    """
    found = [c for c in src_cols if c in df.columns]
    if not found:
        logger.warning(f"  No columns found for {dst} (tried {src_cols[:3]}...)")
        return
    binary = pd.DataFrame({
        c: pd.to_numeric(df[c], errors="coerce").ge(1).astype(float)
        for c in found
    })
    # NaN in original -> NaN in binary indicator
    for c in found:
        binary.loc[pd.to_numeric(df[c], errors="coerce").isna(), c] = np.nan
    out[dst] = binary.sum(axis=1, min_count=1)  # NaN if all missing
    logger.info(f"  Built index: {dst} from {len(found)}/{len(src_cols)} cols "
                f"(range {out[dst].min():.0f}-{out[dst].max():.0f})")


# CEEY state-to-region mapping (from Informe de Movilidad Social 2025 do-file)
# R1=Norte, R2=Noroccidente, R3=Centro-norte, R4=Centro, R5=Sur
CEEY_STATE_TO_REGION: dict[int, int] = {
    # R1 Norte
    2: 1, 5: 1, 8: 1, 19: 1, 26: 1, 28: 1,
    # R2 Noroccidente
    3: 2, 10: 2, 18: 2, 25: 2, 32: 2,
    # R3 Centro-norte
    1: 3, 6: 3, 14: 3, 16: 3, 24: 3,
    # R4 Centro
    9: 4, 11: 4, 13: 4, 15: 4, 17: 4, 21: 4, 22: 4, 29: 4,
    # R5 Sur
    4: 5, 7: 5, 12: 5, 20: 5, 23: 5, 27: 5, 30: 5, 31: 5,
}


def _build_education_6(df: pd.DataFrame, nivel_col: str, completado_col: str) -> pd.Series:
    """Build 6-category education scale from EMOVI nivel + completion columns.

    CEEY methodology from Informe de Movilidad Social 2025:
    1 = Sin estudios
    2 = Primaria incompleta
    3 = Primaria completa
    4 = Secundaria (completa o incompleta)
    5 = Preparatoria (completa o incompleta)
    6 = Profesional (completa o incompleta)

    ESRU-EMOVI 2023 coding for nivel (p43a/p43b):
    1 = Ninguno, 2 = Preescolar, 3 = Primaria,
    4 = Secundaria, 5 = Preparatoria, 6 = Profesional+, 9 = No sabe

    Completion (p44a/p44b): 1 = Completó, 2 = No completó, 8 = No sabe
    """
    if nivel_col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)

    nivel = pd.to_numeric(df[nivel_col], errors="coerce")
    compl = pd.to_numeric(df.get(completado_col, pd.Series(np.nan, index=df.index)),
                          errors="coerce")

    result = pd.Series(np.nan, index=df.index, dtype=float)

    # Sin estudios: ninguno (1) or preescolar (2)
    result.loc[nivel.isin([1, 2])] = 1.0

    # Primaria (nivel == 3): split by completion
    mask_primaria = nivel == 3
    result.loc[mask_primaria & (compl == 2)] = 2.0   # primaria incompleta
    result.loc[mask_primaria & (compl == 1)] = 3.0   # primaria completa
    result.loc[mask_primaria & (compl == 8)] = 2.5   # no sabe -> midpoint
    result.loc[mask_primaria & compl.isna()] = 2.5   # missing -> midpoint

    # Secundaria: nivel == 4
    result.loc[nivel == 4] = 4.0

    # Preparatoria / bachillerato: nivel == 5
    result.loc[nivel == 5] = 5.0

    # Profesional / universidad+: nivel == 6
    result.loc[nivel == 6] = 6.0

    # No sabe (9) -> NaN (already default)

    return result


def _compute_ireh_o(df: pd.DataFrame, out: pd.DataFrame) -> None:
    """Compute IREH-O (wealth index) using MCA a la CEEY.

    Replicates the CEEY's Informe de Movilidad Social methodology:
    - 19 binary asset indicators + hacinamiento binary
    - MCA with Burt method, per cohort
    - First dimension score, sign-flipped so higher = wealthier

    Falls back to simple count index if MCA dependencies unavailable
    or sample too small per cohort.
    """
    # Define the 19 IREH-O items (exact CEEY specification)
    ireh_items = {
        "ac_or1": "p31a",    # estufa
        "ac_or2": "p32a",    # otra vivienda
        "ac_or3": "p31b",    # lavadora
        "ac_or4": "p31h",    # TV cable
        "ac_or5": "p31c",    # refrigerador
        "ac_or6": "p26a",    # agua
        "ac_or7": "p31e",    # TV
        "ac_or8": "p31d",    # telefono
        "ac_or9": "p31k",    # computadora
        "ac_or10": "p26b",   # electricidad
        "ac_or11": "p31m",   # VHS/DVD
        "ac_or12": "p31i",   # microondas
        "ac_or13": "p32b",   # local comercial
        "ac_or14": "p31g",   # aspiradora
        "ac_or15": "p30",    # auto (binarized: >=1)
        "ac_or16": "p26d",   # boiler
        "ac_or17": ["p32e", "p32k", "p32l"],  # compound: bank account
        "ac_or18": ["p32f", "p32g"],           # compound: credit card
        "ac_or19": "p26e",   # servicio domestico
    }

    # Build binary indicator matrix
    indicators = pd.DataFrame(index=df.index)
    for name, src in ireh_items.items():
        if isinstance(src, list):
            # Compound: 1 if any component >= 1
            cols_found = [c for c in src if c in df.columns]
            if cols_found:
                vals = pd.DataFrame({
                    c: pd.to_numeric(df[c], errors="coerce").ge(1).astype(float)
                    for c in cols_found
                })
                indicators[name] = vals.max(axis=1)
                # NaN only if ALL components are NaN
                all_na = pd.DataFrame({
                    c: pd.to_numeric(df[c], errors="coerce").isna()
                    for c in cols_found
                }).all(axis=1)
                indicators.loc[all_na, name] = np.nan
            else:
                indicators[name] = np.nan
        else:
            if src in df.columns:
                val = pd.to_numeric(df[src], errors="coerce")
                indicators[name] = val.ge(1).astype(float)
                indicators.loc[val.isna(), name] = np.nan
            else:
                indicators[name] = np.nan

    # Add hacinamiento binary: 1 if p22/p23 <= 2.5
    if "p22" in df.columns and "p23" in df.columns:
        hh_size = pd.to_numeric(df["p22"], errors="coerce")
        rooms = pd.to_numeric(df["p23"], errors="coerce")
        hacina = hh_size / rooms.replace(0, np.nan)
        indicators["hac_or"] = (hacina <= 2.5).astype(float)
        indicators.loc[hacina.isna(), "hac_or"] = np.nan
    else:
        indicators["hac_or"] = np.nan

    # Drop columns that are entirely NaN
    valid_cols = [c for c in indicators.columns if indicators[c].notna().any()]
    if len(valid_cols) < 3:
        logger.warning("  IREH-O: too few valid indicators, falling back to count")
        out["wealth_index_origin"] = indicators[valid_cols].sum(axis=1, min_count=1)
        return

    indicators = indicators[valid_cols]
    n_items = len(valid_cols)

    # Try MCA per cohort
    try:
        from prince import MCA
        has_mca = True
    except ImportError:
        has_mca = False
        logger.warning("  IREH-O: prince not installed, falling back to count index. "
                       "Install with: pip install prince")

    if not has_mca:
        out["wealth_index_origin"] = indicators.sum(axis=1, min_count=1)
        logger.info(f"  Built IREH-O (count fallback) from {n_items} items")
        return

    # Determine cohorts from age
    if "edad" in df.columns:
        age = pd.to_numeric(df["edad"], errors="coerce")
        cohort = pd.cut(age, bins=[24, 34, 44, 54, 64, 100],
                        labels=[1, 2, 3, 4, 5], right=True)
    elif "age" in out.columns:
        cohort = pd.cut(out["age"], bins=[24, 34, 44, 54, 64, 100],
                        labels=[1, 2, 3, 4, 5], right=True)
    else:
        # No cohort info -> single MCA
        cohort = pd.Series(1, index=df.index)

    # MCA per cohort
    scores = pd.Series(np.nan, index=df.index, dtype=float)
    for c_val in cohort.dropna().unique():
        mask = cohort == c_val
        sub = indicators.loc[mask].dropna()
        if len(sub) < 30:
            # Too few obs for MCA, use count
            scores.loc[sub.index] = sub.sum(axis=1)
            continue

        # MCA requires categorical input
        sub_cat = sub.astype(int).astype(str)
        try:
            mca = MCA(n_components=1)
            mca.fit(sub_cat)
            dim1 = mca.row_coordinates(sub_cat).iloc[:, 0]

            # Sign flip: higher count of assets should = higher score
            asset_count = sub.sum(axis=1)
            if dim1.corr(asset_count) < 0:
                dim1 = -dim1

            scores.loc[sub.index] = dim1.values
        except Exception as e:
            logger.warning(f"  IREH-O MCA failed for cohort {c_val}: {e}, using count")
            scores.loc[sub.index] = sub.sum(axis=1)

    out["wealth_index_origin"] = scores
    logger.info(f"  Built IREH-O (MCA) from {n_items} items across "
                f"{cohort.nunique()} cohorts")


def _build_index(df: pd.DataFrame, out: pd.DataFrame,
                 dst: str, src_cols: list[str]) -> None:
    """Build a simple additive index from ordinal columns.

    Each column is standardized to 0-1 range before summing.
    """
    found = [c for c in src_cols if c in df.columns]
    if not found:
        logger.warning(f"  No columns found for {dst}")
        return
    parts = []
    for c in found:
        vals = pd.to_numeric(df[c], errors="coerce")
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            parts.append((vals - vmin) / (vmax - vmin))
        else:
            parts.append(vals * 0.0)
    combined = pd.concat(parts, axis=1)
    out[dst] = combined.sum(axis=1, min_count=1)
    logger.info(f"  Built index: {dst} from {len(found)} cols")


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

    # -- Parental education (4-category) --
    _map_categorical(df, out, "educp", "father_education")
    _map_categorical(df, out, "educm", "mother_education")
    _map_binary(df, out, "p44a", "father_literacy")
    _map_binary(df, out, "p44b", "mother_literacy")

    # -- Parental education (6-category CEEY scale) --
    out["father_education_6"] = _build_education_6(df, "p43a", "p44a")
    if out["father_education_6"].notna().any():
        logger.info(f"  Built: father_education_6 (6 cat, "
                    f"{out['father_education_6'].notna().sum()} valid)")
    else:
        logger.warning("  father_education_6: p43a not found")

    out["mother_education_6"] = _build_education_6(df, "p43b", "p44b")
    if out["mother_education_6"].notna().any():
        logger.info(f"  Built: mother_education_6 (6 cat, "
                    f"{out['mother_education_6'].notna().sum()} valid)")
    else:
        logger.warning("  mother_education_6: p43b not found")

    # max_parent_education: max of father and mother (CEEY primary variable)
    out["max_parent_education"] = pd.DataFrame({
        "f": out.get("father_education_6", pd.Series(dtype=float)),
        "m": out.get("mother_education_6", pd.Series(dtype=float)),
    }).max(axis=1)
    if out["max_parent_education"].notna().any():
        logger.info(f"  Built: max_parent_education (max of 6-cat parents)")

    # -- Parental occupation --
    _map_categorical(df, out, "clasep", "father_occupation")
    _map_categorical(df, out, "clasem", "mother_occupation")

    # -- Ethnicity & phenotype --
    _map_categorical(df, out, "p110", "ethnicity")
    _map_binary(df, out, "p111", "indigenous_language")

    if "p112" in df.columns:
        out["skin_tone"] = df["p112"].apply(_skin_tone_letter_to_num)
        logger.info(f"  Mapped: p112 -> skin_tone ({out['skin_tone'].notna().sum()} valid)")
    else:
        logger.warning("  p112 not found")

    for col_name in ("p113dL", "p113dl"):
        if col_name in df.columns:
            out["skin_tone_cielab"] = pd.to_numeric(df[col_name], errors="coerce")
            logger.info(f"  Mapped: {col_name} -> skin_tone_cielab")
            break
    else:
        logger.warning("  p113dL not found")

    # -- Geography at 14 --
    # If region_14 exists in raw data, use it; otherwise derive from state (p19)
    if "region_14" in df.columns:
        _map_categorical(df, out, "region_14", "region_14")
    elif "p19" in df.columns:
        state = pd.to_numeric(df["p19"], errors="coerce")
        out["region_14"] = state.map(CEEY_STATE_TO_REGION).astype(float)
        logger.info(f"  Derived: region_14 from p19 using CEEY mapping "
                    f"({out['region_14'].notna().sum()} valid)")
    else:
        logger.warning("  region_14 and p19 both not found")

    _map_categorical(df, out, "p19", "state_14")

    if "p21" in df.columns:
        p21 = pd.to_numeric(df["p21"], errors="coerce")
        out["rural_14"] = (p21 <= 2).astype(float)
        out.loc[p21.isna(), "rural_14"] = np.nan
        logger.info("  Mapped: p21 -> rural_14 (binarized)")
    else:
        logger.warning("  p21 not found")

    # -- Gender --
    if "sexo" in df.columns:
        sexo = pd.to_numeric(df["sexo"], errors="coerce")
        out["gender"] = (sexo == 1).astype(float)
        out.loc[sexo.isna(), "gender"] = np.nan
        logger.info("  Mapped: sexo -> gender (1=male, 0=female)")
    else:
        logger.warning("  sexo not found")

    # -- Family structure at 14 --
    _map_numeric(df, out, "p22", "hh_size_14")
    _map_numeric(df, out, "p57", "n_siblings")
    _map_numeric(df, out, "p58", "birth_order")
    _map_categorical(df, out, "p39", "lived_with_14")
    _map_categorical(df, out, "p40", "breadwinner_14")

    # -- Material conditions at 14 (composite indices) --
    _map_numeric(df, out, "p23", "dwelling_rooms_14")

    # Dwelling quality index: floor material (p25) + dwelling type (p28)
    # Higher = better quality. Simple additive index.
    _build_index(df, out, "dwelling_quality_14", ["p25", "p28"])

    # Dwelling amenities: p26a-e (water, electricity, bathroom, boiler, domestic service)
    _build_count_index(df, out, "dwelling_amenities_14",
                       ["p26a", "p26b", "p26c", "p26d", "p26e"])

    # Dwelling features: p29a-g (living room, garden, patio, laundry, TV room, garage, kitchen)
    _build_count_index(df, out, "dwelling_features_14",
                       ["p29a", "p29b", "p29c", "p29d", "p29e", "p29f", "p29g"])

    _map_numeric(df, out, "p30", "n_automobiles_14")

    # Household assets: p31a-o (15 items: stove, washer, fridge, phone, TV, ...)
    _build_count_index(df, out, "hh_assets_14",
                       [f"p31{c}" for c in "abcdefghijklmno"])

    # Financial assets: p32a-o (15 items: other dwelling, land, savings, ...)
    _build_count_index(df, out, "financial_assets_14",
                       [f"p32{c}" for c in "abcdefghijklmno"])

    # Neighborhood quality: p33a-i (9 items: lighting, schools, clinics, ...)
    _build_count_index(df, out, "neighborhood_quality_14",
                       [f"p33{c}" for c in "abcdefghi"])

    # Floor material at 14: p25 (CEEY: tierra=0, otro=1)
    if "p25" in df.columns:
        p25 = pd.to_numeric(df["p25"], errors="coerce")
        # CEEY coding: recode (2 3=1)(1=0) -> 1=tierra -> 0, 2/3=otro -> 1
        out["floor_material_14"] = p25.map({1: 0.0, 2: 1.0, 3: 1.0})
        out.loc[p25.isna(), "floor_material_14"] = np.nan
        logger.info(f"  Mapped: p25 -> floor_material_14 (binary: tierra=0, otro=1)")
    else:
        logger.warning("  p25 not found for floor_material_14")

    # Overcrowding ratio at 14: p22/p23 (hacinamiento, continuous)
    if "p22" in df.columns and "p23" in df.columns:
        hh_size = pd.to_numeric(df["p22"], errors="coerce")
        rooms = pd.to_numeric(df["p23"], errors="coerce")
        out["overcrowding_14"] = hh_size / rooms.replace(0, np.nan)
        logger.info(f"  Built: overcrowding_14 = p22/p23 "
                    f"(mean={out['overcrowding_14'].mean():.2f})")
    else:
        logger.warning("  p22/p23 not found for overcrowding_14")

    # IREH-O: CEEY official wealth index (MCA-based)
    _compute_ireh_o(df, out)

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
    cat_cols = [
        "father_education", "mother_education", "father_occupation",
        "mother_occupation", "ethnicity", "region_14", "state_14",
        "lived_with_14", "breadwinner_14",
        "father_education_6", "mother_education_6", "max_parent_education",
    ]
    for col in cat_cols:
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

    from core.types import Circumstance
    circumstance_vars = {c.value for c in Circumstance}
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

    # === Parental education (4-cat) ===
    df["father_education"] = rng.choice([0, 1, 2, 3], size=n, p=[0.35, 0.25, 0.22, 0.18]).astype(float)
    df["mother_education"] = rng.choice([0, 1, 2, 3], size=n, p=[0.40, 0.25, 0.20, 0.15]).astype(float)
    # Literacy correlated with education (higher edu -> always literate)
    df["father_literacy"] = (rng.random(n) < (0.7 + 0.1 * df["father_education"])).astype(float)
    df["mother_literacy"] = (rng.random(n) < (0.65 + 0.1 * df["mother_education"])).astype(float)

    # === Parental education (6-cat CEEY scale) ===
    # Map 4-cat to 6-cat with some noise to simulate finer granularity
    _edu_4_to_6 = {0: 1.0, 1: 2.5, 2: 4.0, 3: 5.5}
    df["father_education_6"] = df["father_education"].map(_edu_4_to_6)
    df["father_education_6"] = (df["father_education_6"] + rng.choice([-0.5, 0, 0.5], size=n)).clip(1, 6).round()
    df["mother_education_6"] = df["mother_education"].map(_edu_4_to_6)
    df["mother_education_6"] = (df["mother_education_6"] + rng.choice([-0.5, 0, 0.5], size=n)).clip(1, 6).round()
    df["max_parent_education"] = np.maximum(df["father_education_6"], df["mother_education_6"])

    # === Parental occupation ===
    df["father_occupation"] = rng.choice([0, 1, 2, 3, 4, 5], size=n,
                                          p=[0.20, 0.20, 0.18, 0.17, 0.15, 0.10]).astype(float)
    df["mother_occupation"] = rng.choice([0, 1, 2, 3, 4, 5], size=n,
                                          p=[0.30, 0.20, 0.18, 0.15, 0.10, 0.07]).astype(float)

    # === Ethnicity & phenotype ===
    df["ethnicity"] = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.40, 0.25, 0.15, 0.12, 0.08]).astype(float)
    df["indigenous_language"] = rng.choice([0.0, 1.0], size=n, p=[0.90, 0.10])
    df["skin_tone"] = rng.choice(np.arange(1, 12, dtype=float), size=n)
    df["skin_tone_cielab"] = rng.normal(55, 10, size=n).clip(30, 85)

    # === Geography at 14 ===
    df["region_14"] = rng.choice([0, 1, 2, 3, 4], size=n).astype(float)
    df["state_14"] = rng.choice(np.arange(32, dtype=float), size=n)
    df["rural_14"] = rng.choice([0.0, 1.0], size=n, p=[0.55, 0.45])

    # === Gender ===
    df["gender"] = rng.choice([0.0, 1.0], size=n, p=[0.50, 0.50])

    # === Family structure at 14 ===
    df["hh_size_14"] = rng.choice(np.arange(1, 13, dtype=float), size=n,
                                   p=[0.02, 0.05, 0.10, 0.20, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01])
    df["n_siblings"] = rng.poisson(3, size=n).clip(0, 15).astype(float)
    df["birth_order"] = rng.choice(np.arange(1, 8, dtype=float), size=n,
                                    p=[0.25, 0.25, 0.20, 0.13, 0.08, 0.05, 0.04])
    df["lived_with_14"] = rng.choice([0, 1, 2, 3], size=n, p=[0.70, 0.15, 0.10, 0.05]).astype(float)
    df["breadwinner_14"] = rng.choice([0, 1, 2, 3], size=n, p=[0.65, 0.20, 0.10, 0.05]).astype(float)

    # === Material conditions at 14 (composite indices) ===
    # Wealth factor: correlated with parental education/occupation
    wealth = (df["father_education"] + df["mother_education"] +
              df["father_occupation"] * 0.5 + rng.normal(0, 2, size=n))
    wealth_norm = (wealth - wealth.min()) / (wealth.max() - wealth.min())

    df["dwelling_rooms_14"] = rng.poisson(3 + 2 * wealth_norm).clip(1, 15).astype(float)
    df["dwelling_quality_14"] = (wealth_norm * 1.5 + rng.normal(0, 0.3, size=n)).clip(0, 2).round(2)
    df["dwelling_amenities_14"] = rng.binomial(5, 0.3 + 0.5 * wealth_norm).astype(float)
    df["dwelling_features_14"] = rng.binomial(7, 0.2 + 0.5 * wealth_norm).astype(float)
    df["n_automobiles_14"] = rng.poisson(0.3 + 0.7 * wealth_norm).clip(0, 5).astype(float)
    df["hh_assets_14"] = rng.binomial(15, 0.2 + 0.5 * wealth_norm).astype(float)
    df["financial_assets_14"] = rng.binomial(15, 0.05 + 0.3 * wealth_norm).astype(float)
    df["neighborhood_quality_14"] = rng.binomial(9, 0.3 + 0.4 * wealth_norm).astype(float)
    df["floor_material_14"] = (rng.random(n) < (0.6 + 0.35 * wealth_norm)).astype(float)
    df["overcrowding_14"] = (df["hh_size_14"] / df["dwelling_rooms_14"].replace(0, 1)).round(2)

    # IREH-O: MCA score approximation (in synthetic, use weighted sum)
    ireh_items = np.column_stack([
        df["hh_assets_14"] / 15,
        df["financial_assets_14"] / 15,
        df["dwelling_amenities_14"] / 5,
        df["dwelling_features_14"] / 7,
        df["floor_material_14"],
        (df["overcrowding_14"] <= 2.5).astype(float),
        (df["n_automobiles_14"] >= 1).astype(float),
    ])
    df["wealth_index_origin"] = ireh_items.mean(axis=1) * 10 + rng.normal(0, 0.5, size=n)

    # === Demographics ===
    df["age"] = rng.integers(25, 65, size=n).astype(float)
    df["urban"] = rng.choice([0.0, 1.0], size=n, p=[0.30, 0.70])

    # === Income: influenced by circumstances (what IOp measures) ===
    log_income = (
        8.5
        + 0.15 * df["father_education"]
        + 0.12 * df["mother_education"]
        + 0.08 * df["father_occupation"]
        + 0.05 * df["mother_occupation"]
        - 0.06 * df["ethnicity"]
        - 0.12 * df["indigenous_language"]
        - 0.015 * df["skin_tone"]
        + 0.04 * df["gender"]
        + 0.08 * df["urban"]
        - 0.06 * df["rural_14"]
        + 0.01 * df["age"]
        # New variables: material conditions have additional explanatory power
        + 0.02 * df["hh_assets_14"]
        + 0.03 * df["financial_assets_14"]
        + 0.01 * df["dwelling_amenities_14"]
        + 0.01 * df["neighborhood_quality_14"]
        - 0.02 * df["n_siblings"]
        + 0.04 * df["max_parent_education"]
        + 0.03 * df["floor_material_14"]
        - 0.03 * df["overcrowding_14"].clip(0, 10)
        + 0.05 * df["wealth_index_origin"] / 10
        # Interaction: father_education * wealth amplifies advantage
        + 0.01 * df["father_education"] * df["hh_assets_14"]
        + rng.normal(0, 0.55, size=n)  # effort + luck
    )

    df["hh_pc_imputed"] = np.exp(log_income) * rng.uniform(0.8, 1.2, size=n)

    df["hh_total_reported"] = df["hh_pc_imputed"] * rng.uniform(2.0, 5.0, size=n)
    missing_mask = rng.random(size=n) < 0.25
    df.loc[missing_mask, "hh_total_reported"] = np.nan

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
