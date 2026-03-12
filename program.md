# Program: IOp Specification Curve Analysis

## Objective
Systematically explore the space of methodological specifications for
Inequality of Opportunity (IOp) estimation using ESRU-EMOVI 2023 data from Mexico.
Produce a multiverse analysis aligned with CEEY official methodology.

## Agent-in-the-Loop Design

This project follows Karpathy's "vibe coding" principle adapted for academic research:
**the AI agent (Claude Code) IS the researcher**, not a Python script pretending to be one.

The pipeline infrastructure (data loading, estimation methods, bootstrap, MI pooling)
is fixed and tested. The agent's job is to **reason** about what to explore, **interpret**
results in the context of IOp theory, and **decide** next steps based on scientific value.

### What the agent does (Claude Code)
- Reads results and coverage → **reasons** about gaps and patterns
- Formulates hypotheses based on IOp literature and observed data
- Edits `analyze.py` with justified specs → runs `run_experiment.py`
- Interprets results → decides whether to explore deeper or move on
- Generates synthesis when a meaningful batch is complete

### What the infrastructure does (Python)
- `run_experiment.py`: Executes specs, logs results immutably
- `core/`, `methods/`, `evaluation/`: Fixed estimation pipeline
- `orchestration/`: Coverage tracking, strategy constants
- `synthesis/`: Spec curve plots, summary tables, figures
- `autoresearch.py`: **Diagnostic toolkit** for the agent (status, findings, gaps)

## Agent Toolkit

The agent uses `autoresearch.py` as a diagnostic tool, not as an autonomous loop:

```bash
python autoresearch.py status          # Coverage, experiment count, IOp distribution
python autoresearch.py findings        # Pattern detection (method divergence, outliers, etc.)
python autoresearch.py gaps            # Missing specification slots (listwise + MI)
python autoresearch.py gaps --mi       # MI-specific gaps only
python autoresearch.py recent          # Last 10 experiment results
python autoresearch.py recent 20       # Last 20 results
python autoresearch.py synthesize      # Generate spec curve + tables + figures
```

## Workflow per iteration

1. **Diagnose**: `python autoresearch.py status` + `findings` + `gaps`
2. **Reason**: What has scientific value? What gap matters most? What hypothesis to test?
3. **Design**: Edit `analyze.py` with 5-15 specs, each with a clear `rationale`
4. **Execute**: `python run_experiment.py`
5. **Interpret**: Read results, compare with expectations from IOp theory
6. **Synthesize**: When a meaningful batch completes → `python autoresearch.py synthesize`
7. **Report**: Summarize findings to the user, propose next steps

## Rules

### What the agent CAN do
- Edit `analyze.py` to define new batches of specs
- Read experiment results from `results/experiment_log.jsonl`
- Run diagnostic commands via `autoresearch.py`
- Generate synthesis outputs (spec curves, tables, figures)
- Formulate and test hypotheses based on observed patterns
- Reason about results in the context of IOp literature

### What the agent CANNOT do
- Modify `run_experiment.py`, `core/`, `methods/`, `evaluation/` (FIXED)
- Delete or modify entries in the experiment log
- Cherry-pick results (all specs get logged, successful or not)
- Make causal claims from IOp estimates
- Choose specs based on their results (choose based on coverage gaps + scientific value)

### Exploration principles
- **Coverage first**: Fill systematic gaps before hypothesis testing
- **Scientific value**: Not all gaps are equal — prioritize specs that test meaningful hypotheses
- **No p-hacking**: Select specs based on what's MISSING, not what gives "better" results
- **Monotonicity check**: IOp should be non-decreasing in circumstances (violation = investigate)
- **Method triangulation**: OLS lower bound vs ML methods reveals interaction effects
- **MI robustness**: Compare listwise vs MI to assess missing data sensitivity

## Circumstance variables (33, from ESRU-EMOVI 2023)

All measured at or before age 14 (predetermined under Roemer framework).

### Parental education (4-category)
- `father_education`: Father's education (educp, 4 cat: primaria o menos, secundaria, media superior, profesional)
- `mother_education`: Mother's education (educm, 4 cat)
- `father_literacy`: Father literate (p44a, binary)
- `mother_literacy`: Mother literate (p44b, binary)

### Parental education (6-category CEEY scale)
- `father_education_6`: Father's education (p43+p44, 6 cat: sin estudios / primaria incompleta / primaria completa / secundaria / preparatoria / profesional)
- `mother_education_6`: Mother's education (p43m+p44m, 6 cat)
- `max_parent_education`: max(father, mother) education -- CEEY's primary human capital variable

### Parental occupation
- `father_occupation`: Father's occupational class (clasep, 6 cat)
- `mother_occupation`: Mother's occupational class (clasem, 6 cat)

### Ethnicity & phenotype
- `ethnicity`: Ethnic self-identification (p110, 5 cat)
- `indigenous_language`: Indigenous language speaker (p111, binary)
- `skin_tone`: Skin tone self-report (p112, A-K -> 1-11)
- `skin_tone_cielab`: Skin tone colorimeter CIELab L* (p113dL, continuous)

### Geography at 14
- `region_14`: Region at age 14 (5 CEEY regions: Norte, Noroccidente, Centro-norte, Centro, Sur)
- `state_14`: State at age 14 (p19, 32 states)
- `rural_14`: Rural at age 14 (p21, binarized)

### Gender
- `gender`: Gender (sexo, 1=male, 0=female)

### Family structure at 14
- `hh_size_14`: Household size (p22, count)
- `n_siblings`: Number of siblings (p57, count)
- `birth_order`: Birth order (p58, ordinal)
- `lived_with_14`: Lived with whom at 14 (p39, categorical)
- `breadwinner_14`: Main breadwinner at 14 (p40, categorical)

### Material conditions at 14 (composite indices)
- `dwelling_rooms_14`: Total rooms (p23, count)
- `dwelling_quality_14`: Floor + dwelling type index (p25+p28)
- `dwelling_amenities_14`: Amenities count 0-5 (p26a-e)
- `dwelling_features_14`: Features count 0-7 (p29a-g)
- `n_automobiles_14`: Automobiles (p30, count)
- `hh_assets_14`: Household assets count 0-15 (p31a-o)
- `financial_assets_14`: Financial assets count 0-15 (p32a-o)
- `neighborhood_quality_14`: Neighborhood quality count 0-9 (p33a-i)
- `floor_material_14`: Dirt floor indicator (p25: tierra=0, otro=1)
- `overcrowding_14`: Hacinamiento ratio (p22/p23, continuous)

### CEEY wealth index
- `wealth_index_origin`: IREH-O via MCA (Burt method, per cohort, 19 binary assets + hacinamiento)

## Income variables
- `hh_pc_imputed` (ingc_pc): Imputed per-capita HH income (primary, no missing by construction)
- `hh_total_reported` (p101): Self-reported total HH income (has nonresponse)
- Log variants: `log_hh_pc_imputed`, `log_hh_total_reported`

## Valid method-decomposition combinations
| Method | Valid decompositions |
|--------|---------------------|
| OLS | lower_bound |
| decision_tree | ex_ante, ex_post |
| cforest | ex_ante, ex_post |
| xgboost | ex_ante |
| random_forest | ex_ante |

## Multiple Imputation

### Setup
1. Run `python prepare.py --impute --impute-m 20` to create 20 MICE-imputed datasets
2. Imputed datasets saved in `data/processed/imputed/m_00.parquet` ... `m_19.parquet`
3. Only circumstance columns with missing data are imputed; income is predictor only

### Specs
- Set `use_mi=True` in ExperimentSpec to use MI instead of listwise deletion
- MI specs get different `spec_id`s from their listwise counterparts
- Bootstrap is tiered per imputation: OLS/DT=100, XGBoost=10, RF=5
- Pooling uses Rubin's rules (Barnard-Rubin df, fraction of missing info)

### MI diagnostics
- `mi_within_variance` (U_bar), `mi_between_variance` (B)
- `mi_fraction_missing_info` (FMI = (1+1/M)*B / T)
- `mi_n_imputations` (M)
- FMI > 0.5 suggests results are heavily influenced by missing data handling

### Sensitivity analysis
Compare `use_mi=False` vs `use_mi=True` on same specs to assess impact of listwise deletion.

## Quality checks
Before submitting specs, verify:
- [ ] No double-log (log income + var_logs)
- [ ] Valid method-decomposition combination
- [ ] At least 1 circumstance
- [ ] Clear rationale for each spec
- [ ] Not duplicating an already-run spec_id
- [ ] MI specs have imputed data available (`prepare.py --impute`)
