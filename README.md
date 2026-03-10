# autoresearch-iop

**Specification Curve Analysis for Inequality of Opportunity**

An adaptation of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) for social science. Instead of optimizing a model, this pipeline systematically maps the entire multiverse of IOp estimates across methodological choices.

## What is this?

Inequality of Opportunity (IOp) estimates are notoriously sensitive to methodological decisions: which circumstances to include, which income measure to use, which inequality index, which estimation method. A single paper can report IOp shares from **10% to 50%** depending on these choices.

This pipeline automates the exploration of that specification space, producing a **specification curve** that transparently shows how IOp estimates vary across all valid methodological combinations. Instead of cherry-picking one specification, we map the entire **multiverse of estimates**.

### Theoretical Framework

**Inequality of Opportunity** (Roemer, 1998) decomposes total income inequality into:

- **Circumstances** (exogenous): factors beyond individual control -- parental education, ethnicity, geography at birth
- **Effort** (endogenous): individual choices and actions

The IOp share measures what fraction of total inequality is attributable to circumstances. This pipeline estimates that share using multiple decomposition approaches, inequality measures, and estimation methods.

## Architecture

```
prepare.py (FIXED)           →  emovi_analytical.parquet (17,843 obs × 38 vars)
                                          ↓
analyze.py (AGENT EDITS)     →  ExperimentSpec batch
                                          ↓
run_experiment.py (FIXED)    →  validate → estimate → bootstrap CI → log
                                          ↓
autoresearch.py (FIXED)      →  systematic → hypothesis → robustness (autonomous loop)
                                          ↓
results/experiment_log.jsonl    (APPEND-ONLY)
                                          ↓
synthesis/                   →  spec curve plot, tables, figures
```

The LLM agent only edits `analyze.py` to define experiment batches. Everything else is fixed infrastructure. The autonomous loop (`autoresearch.py`) can run without human intervention.

## Specification Space

| Dimension | Options | Details |
|-----------|---------|---------|
| Circumstances | 33 variables | Parental education (4-cat & 6-cat CEEY), occupation, ethnicity, phenotype, geography, family structure, material conditions, IREH-O wealth index |
| Income | 4 variants | `hh_pc_imputed`, `hh_total_reported`, + log transforms |
| Inequality measure | 7 | Gini, MLD, Theil-T, Var(logs), Atkinson(0.5, 1, 2) |
| Estimation method | 5 | OLS, Decision Tree, Conditional Forest, XGBoost, Random Forest |
| Decomposition | 4 | ex_ante, ex_post, lower_bound, upper_bound |
| Sample filter | 16 | all, male, female, 4 age cohorts, urban/rural, urban/rural × gender |

The full combinatorial space is astronomically large. The pipeline strategically explores ~200-500 specifications using a 60/30/10 rule.

### Circumstance Variables (33)

All measured at or before age 14 (predetermined under the Roemer framework):

| Domain | Variables | Source |
|--------|-----------|--------|
| Parental education (4-cat) | `father_education`, `mother_education`, `father_literacy`, `mother_literacy` | educp, educm, p44a, p44b |
| Parental education (6-cat CEEY) | `father_education_6`, `mother_education_6`, `max_parent_education` | p43+p44, CEEY scale |
| Parental occupation | `father_occupation`, `mother_occupation` | clasep, clasem |
| Ethnicity & phenotype | `ethnicity`, `indigenous_language`, `skin_tone`, `skin_tone_cielab` | p110, p111, p112, p113dL |
| Geography at 14 | `region_14`, `state_14`, `rural_14` | CEEY 5-region mapping, p19, p21 |
| Gender | `gender` | sexo |
| Family structure | `hh_size_14`, `n_siblings`, `birth_order`, `lived_with_14`, `breadwinner_14` | p22, p57, p58, p39, p40 |
| Material conditions | `dwelling_rooms_14`, `dwelling_quality_14`, `dwelling_amenities_14`, `dwelling_features_14`, `n_automobiles_14`, `hh_assets_14`, `financial_assets_14`, `neighborhood_quality_14`, `floor_material_14`, `overcrowding_14` | p23-p33 composite indices |
| CEEY wealth index | `wealth_index_origin` | IREH-O (MCA, 19 assets + hacinamiento) |

## CEEY Alignment

Variable definitions are aligned with the CEEY's official methodology (*Informe de Movilidad Social en México 2025*):

- **6-category education scale**: sin estudios / primaria incompleta / primaria completa / secundaria / preparatoria / profesional (from p43 + p44)
- **MAX_PARENT_EDUCATION**: max(father, mother) education -- CEEY's primary human capital variable
- **IREH-O**: *Índice de Recursos Económicos del Hogar de Origen*, a wealth index computed via Multiple Correspondence Analysis (MCA, Burt method) on 19 binary assets + hacinamiento, estimated per cohort
- **CEEY region mapping**: 5 regions (Norte, Noroccidente, Centro-norte, Centro, Sur) derived from the 32 Mexican states using the exact CEEY classification
- **Hacinamiento**: overcrowding ratio (household size / rooms)

## Karpathy vs. This Pipeline

| Karpathy autoresearch | autoresearch-iop |
|----------------------|-----------------|
| `train.py` (agent edits model code) | `analyze.py` (agent defines specs) |
| BPB (single metric, lower = better) | IOp share + CI (no "better", map all) |
| Keep/revert (optimize) | Log everything (multiverse) |
| 5 min/experiment (GPU) | Seconds/experiment (tabular, CPU) |
| Output: best model weights | Output: specification curve + tables |

## Quick Start

```bash
# Clone and install
git clone https://github.com/Lalitronico/autoresearch-iop.git
cd autoresearch-iop
pip install -e ".[dev]"

# Prepare data (synthetic for testing, or place real .dta in data/raw/)
python prepare.py --synthetic

# Run baseline experiments (3 specs defined in analyze.py)
python run_experiment.py

# Run autonomous exploration loop
python autoresearch.py --full          # Full loop: systematic + hypothesis + robustness
python autoresearch.py --systematic    # Systematic coverage only
python autoresearch.py --findings      # Show detected patterns from completed experiments
python autoresearch.py --robustness    # Robustness checks on existing results

# Run the test suite (104 tests)
python -m pytest tests/ -v
```

## Data

Uses **ESRU-EMOVI 2023** from Mexico (Centro de Estudios Espinosa Yglesias). 17,843 respondents aged 25-64.

### Obtaining the data

1. Visit the [CEEY data portal](https://ceey.org.mx/contenido/que-hacemos/emovi/) or contact CEEY directly
2. Download the respondent-level file (`entrevistado_2023.dta`)
3. Place it in `data/raw/entrevistado_2023.dta`
4. Run `python prepare.py` to generate the analytical dataset

The pipeline produces `data/processed/emovi_analytical.parquet` (17,843 obs x 38 variables) with all 33 circumstance variables pre-coded.

### Synthetic data for testing

```bash
python prepare.py --synthetic  # Generates 500 observations with realistic distributions
```

This creates synthetic data mimicking the real distributions, sufficient for testing the pipeline end-to-end. All tests use synthetic data via test fixtures.

### CI badge

[![CI](https://github.com/Lalitronico/autoresearch-iop/actions/workflows/ci.yml/badge.svg)](https://github.com/Lalitronico/autoresearch-iop/actions/workflows/ci.yml)

## Autonomous Exploration Loop

`autoresearch.py` implements a three-phase autonomous exploration:

| Phase | Share | Strategy |
|-------|-------|----------|
| Systematic | 60% | Deterministic gap-filling across 13 circumstance sets × 5 method-decomposition combos × 3 measures × 2 incomes |
| Hypothesis | 30% | Auto-detect findings (method divergence, circumstance sensitivity, wide CIs, outliers) and generate targeted follow-up specs |
| Robustness | 10% | Vary one dimension at a time from top specs to test sensitivity |

### Findings Detection

The system automatically detects patterns between iterations:

- **Method divergence**: OLS vs XGBoost give different IOp estimates (captures non-linearities)
- **Circumstance sensitivity**: Which circumstance subsets drive IOp variation
- **Measure sensitivity**: How Gini vs MLD vs Theil disagree
- **Wide CIs**: Specs with unreliable estimates needing larger samples
- **Outliers**: IOp values outside expected range

## Estimation Methods

| Method | Decompositions | Description |
|--------|---------------|-------------|
| OLS | lower_bound | Ferreira-Gignoux parametric. R² provides lower bound. |
| Decision Tree | ex_ante, ex_post | sklearn tree as proxy for ctree. Partitions into "types". |
| XGBoost | ex_ante | Gradient boosting with 5-fold CV predictions. SHAP for importance. |
| Random Forest | ex_ante | Ensemble of 200 trees with CV predictions. |

## Safeguards

- **Append-only log**: All results recorded, never deleted
- **Coverage-based exploration**: Agent explores by gaps, not by outcome (prevents p-hacking)
- **Spec validation**: Domain rules prevent invalid combinations (no double-log, valid method-decomposition pairs)
- **Diagnostics**: Sample size guards (min 100), CI width checks, overfitting detection
- **Reproducibility**: Deterministic seeds, spec hashing (SHA-256, order-independent)
- **Deterministic spec IDs**: Each specification gets a unique 12-character hash; duplicate specs are detected automatically

## Project Structure

```
autoresearch-iop/
├── core/                         # Types, spec validation, measures, decomposition
│   ├── types.py                  # 33 Circumstance enums + income, measure, method types
│   ├── specification.py          # ExperimentSpec dataclass + validation + hashing
│   ├── inequality_measures.py    # Gini, MLD, Theil-T, Var(logs), Atkinson
│   ├── decomposition.py          # IOp decomposition: ex_ante, ex_post, lower/upper bound
│   └── data_loader.py            # DataRegistry: load EMOVI, filter samples, validate specs
├── methods/                      # Estimation methods
│   ├── parametric.py             # Ferreira-Gignoux OLS
│   ├── nonparametric.py          # Decision tree type partitioning
│   └── ml_methods.py             # XGBoost, Random Forest, SHAP
├── evaluation/                   # Bootstrap CI + diagnostics
│   ├── metrics.py                # compute_iop_with_ci (bootstrap)
│   └── diagnostics.py            # Flags, warnings, sample guards
├── orchestration/                # Logging, coverage, strategy
│   ├── experiment_log.py         # Append-only JSONL + TSV logging
│   ├── coverage_tracker.py       # Track explored spec regions
│   └── strategy.py               # 13 circumstance sets, 60/30/10 exploration strategy
├── synthesis/                    # Publication-quality outputs
│   ├── spec_curve.py             # Specification curve plot
│   ├── summary_tables.py         # LaTeX / markdown tables
│   └── figures.py                # Publication-quality figures
├── tests/                        # 104 tests (unit + integration + e2e)
├── data/
│   ├── raw/                      # ESRU-EMOVI .dta files (gitignored)
│   ├── processed/                # .parquet analytical dataset (gitignored)
│   └── codebook.json             # Variable metadata (auto-generated)
├── results/                      # Experiment logs + figures + tables
├── docs/
│   └── pipeline.html             # Interactive pipeline documentation
├── analyze.py                    # Agent-editable spec definitions
├── autoresearch.py               # Autonomous exploration loop (systematic → hypothesis → robustness)
├── prepare.py                    # Data preparation + CEEY alignment (FIXED)
├── run_experiment.py             # Experiment harness (FIXED)
├── program.md                    # Agent instructions
└── CLAUDE.md                     # Claude Code project config
```

## Requirements

Python 3.10+. No GPU needed -- everything is tabular and runs on any laptop.

Core: pandas, numpy, scipy, scikit-learn, statsmodels, matplotlib, seaborn, pyarrow
ML: xgboost, shap
Optional: prince (for MCA-based IREH-O; falls back to count index if not installed)

## License

MIT

## References

- Roemer, J. E. (1998). *Equality of Opportunity*. Harvard University Press.
- Ferreira, F. H. G., & Gignoux, J. (2011). The measurement of inequality of opportunity: Theory and an application to Latin America. *Review of Income and Wealth*, 57(4), 622-657.
- Centro de Estudios Espinosa Yglesias (2023). ESRU-EMOVI 2023.
- Simonsohn, U., Simmons, J. P., & Nelson, L. D. (2020). Specification curve analysis. *Nature Human Behaviour*, 4(11), 1208-1214.
- Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)
