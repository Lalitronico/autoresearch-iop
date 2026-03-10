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

    Source columns in parentheses:
    - educp: Father's education (4 categories: primaria o menos, secundaria, media superior, profesional)
    - educm: Mother's education (4 categories, same as above)
    - clasep: Father's occupational class (6 categories)
    - p110: Ethnic self-identification (5 categories)
    - p111: Indigenous language speaker (binary)
    - skin_tone_num: Derived from p112 (letters A-K -> 1-11)
    - skin_tone_cielab: From p113dL (CIELab L* continuous, colorimeter)
    - region_14: Region where lived at age 14 (5 regions)
    - rural_14: Derived from p21 (urbanity at age 14, binarized)
    - gender: From sexo (recoded to 0=female, 1=male)
    """
    FATHER_EDUCATION = "father_education"
    MOTHER_EDUCATION = "mother_education"
    FATHER_OCCUPATION = "father_occupation"
    ETHNICITY = "ethnicity"
    INDIGENOUS_LANGUAGE = "indigenous_language"
    SKIN_TONE = "skin_tone"
    SKIN_TONE_CIELAB = "skin_tone_cielab"
    REGION_14 = "region_14"
    RURAL_14 = "rural_14"
    GENDER = "gender"


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
