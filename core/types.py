"""Shared enums and type definitions for the IOp specification curve pipeline.

Aligned with ESRU-EMOVI 2023 actual variable structure.
"""

from enum import Enum


class IncomeVariable(str, Enum):
    """Income variable options for IOp estimation.

    ESRU-EMOVI 2023 has:
    - ingc_pc: imputed per capita household income (no missing)
    - p101: self-reported total monthly household income (25% missing)
    """
    HH_PC_IMPUTED = "hh_pc_imputed"         # ingc_pc (primary, no missing)
    HH_TOTAL_REPORTED = "hh_total_reported"  # p101 (25% missing)
    LOG_HH_PC_IMPUTED = "log_hh_pc_imputed"
    LOG_HH_TOTAL_REPORTED = "log_hh_total_reported"

    @property
    def is_log(self) -> bool:
        return self.value.startswith("log_")

    @property
    def base_variable(self) -> str:
        return self.value.removeprefix("log_")


class InequalityMeasure(str, Enum):
    """Inequality measures for IOp decomposition."""
    GINI = "gini"
    MLD = "mld"  # Mean Log Deviation (GE(0))
    THEIL_T = "theil_t"  # Theil index (GE(1))
    VAR_LOGS = "var_logs"  # Variance of logarithms
    ATKINSON_05 = "atkinson_0.5"
    ATKINSON_1 = "atkinson_1"
    ATKINSON_2 = "atkinson_2"

    @property
    def atkinson_epsilon(self) -> float | None:
        if not self.value.startswith("atkinson_"):
            return None
        return float(self.value.split("_")[1])


class EstimationMethod(str, Enum):
    """Estimation method for predicting income from circumstances."""
    OLS = "ols"
    DECISION_TREE = "decision_tree"
    CONDITIONAL_FOREST = "cforest"
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"


class DecompositionType(str, Enum):
    """Type of IOp decomposition."""
    EX_ANTE = "ex_ante"       # I(mu_types) / I(y) -- uses predicted type means
    EX_POST = "ex_post"       # sum over types: v_k * I(y_k) -- within-type
    LOWER_BOUND = "lower_bound"  # Parametric R^2-based (Ferreira-Gignoux)
    UPPER_BOUND = "upper_bound"  # Total - IOp_within


class Circumstance(str, Enum):
    """Circumstance variables available in ESRU-EMOVI 2023.

    All measured at or before age 14 (predetermined under Roemer framework).
    Organized by domain. Source columns in parentheses.

    Individual variables are kept for distinct concepts.
    Composite indices aggregate groups of related binary items
    (e.g., 15 household assets -> 1 count) to control dimensionality.
    """

    # === Parental education (4-category: sin estudios/basica/media/superior) ===
    FATHER_EDUCATION = "father_education"      # educp (4 cat)
    MOTHER_EDUCATION = "mother_education"      # educm (4 cat)
    FATHER_LITERACY = "father_literacy"        # p44a (binary)
    MOTHER_LITERACY = "mother_literacy"        # p44b (binary)

    # === Parental education (6-category CEEY scale from p43+p44) ===
    # sin estudios / primaria incompleta / primaria completa /
    # secundaria / preparatoria / profesional
    FATHER_EDUCATION_6 = "father_education_6"  # from p43+p44 (6 cat)
    MOTHER_EDUCATION_6 = "mother_education_6"  # from p43m+p44m (6 cat)
    MAX_PARENT_EDUCATION = "max_parent_education"  # max(father, mother) -- CEEY primary

    # === Parental occupation ===
    FATHER_OCCUPATION = "father_occupation"    # clasep (6 cat)
    MOTHER_OCCUPATION = "mother_occupation"    # clasem (6 cat)

    # === Ethnicity & phenotype ===
    ETHNICITY = "ethnicity"                    # p110 (5 cat)
    INDIGENOUS_LANGUAGE = "indigenous_language"  # p111 (binary)
    SKIN_TONE = "skin_tone"                    # p112 letters A-K -> 1-11
    SKIN_TONE_CIELAB = "skin_tone_cielab"      # p113dL (CIELab L*, continuous)

    # === Geography at 14 ===
    REGION_14 = "region_14"                    # region_14 (5 regions)
    STATE_14 = "state_14"                      # p19 (32 states)
    RURAL_14 = "rural_14"                      # p21 (binarized)

    # === Gender ===
    GENDER = "gender"                          # sexo (binary)

    # === Family structure at 14 ===
    HH_SIZE_14 = "hh_size_14"                  # p22 (count)
    N_SIBLINGS = "n_siblings"                  # p57 (count)
    BIRTH_ORDER = "birth_order"                # p58 (ordinal)
    LIVED_WITH_14 = "lived_with_14"            # p39 (categorical)
    BREADWINNER_14 = "breadwinner_14"          # p40 (categorical)

    # === Material conditions at 14 (composite indices) ===
    DWELLING_ROOMS_14 = "dwelling_rooms_14"    # p23 total rooms (count)
    DWELLING_QUALITY_14 = "dwelling_quality_14"  # p25 floor + p28 type -> index
    DWELLING_AMENITIES_14 = "dwelling_amenities_14"  # p26a-e -> count 0-5
    DWELLING_FEATURES_14 = "dwelling_features_14"    # p29a-g -> count 0-7
    N_AUTOMOBILES_14 = "n_automobiles_14"      # p30 (count)
    HH_ASSETS_14 = "hh_assets_14"              # p31a-o -> count 0-15
    FINANCIAL_ASSETS_14 = "financial_assets_14"  # p32a-o -> count 0-15
    NEIGHBORHOOD_QUALITY_14 = "neighborhood_quality_14"  # p33a-i -> count 0-9
    FLOOR_MATERIAL_14 = "floor_material_14"    # p25: tierra=0, otro=1 (binary)
    OVERCROWDING_14 = "overcrowding_14"        # p22/p23: hacinamiento ratio (continuous)

    # === CEEY official wealth index (MCA-based) ===
    # IREH-O: Indice de Recursos Economicos del Hogar de Origen
    # 19 binary assets + hacinamiento, estimated via MCA (Burt) per cohort
    WEALTH_INDEX_ORIGIN = "wealth_index_origin"  # IREH-O (continuous, MCA score)


class SampleFilter(str, Enum):
    """Sample restriction dimensions."""
    ALL = "all"
    MALE = "male"
    FEMALE = "female"
    AGE_25_44 = "age_25_44"
    AGE_45_64 = "age_45_64"
    AGE_25_55 = "age_25_55"
    URBAN = "urban"
    RURAL = "rural"
    URBAN_MALE = "urban_male"
    URBAN_FEMALE = "urban_female"
    RURAL_MALE = "rural_male"
    RURAL_FEMALE = "rural_female"
    COHORT_1 = "cohort_25_34"
    COHORT_2 = "cohort_35_44"
    COHORT_3 = "cohort_45_54"
    COHORT_4 = "cohort_55_64"


class ExperimentStatus(str, Enum):
    """Status of an experiment run."""
    SUCCESS = "success"
    FAILED = "failed"
    INVALID_SPEC = "invalid_spec"
    SKIPPED = "skipped"
