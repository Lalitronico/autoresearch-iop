# Program: IOp Specification Curve Analysis

## Objective
Systematically explore the space of methodological specifications for
Inequality of Opportunity (IOp) estimation using ESRU-EMOVI data from Mexico.

## Your role
You are a research agent that designs and executes IOp experiments. You modify
`analyze.py` to define batches of `ExperimentSpec`s, then the fixed harness
(`run_experiment.py`) executes them. Alternatively, `autoresearch.py` runs
the full autonomous loop (systematic → hypothesis → robustness).

## Rules

### What you CAN do
- Edit `analyze.py` to define new batches of specs
- Read experiment results from `results/experiment_log.jsonl`
- Read coverage reports via `orchestration/coverage_tracker.py`
- Generate synthesis outputs (spec curves, tables, figures)
- Suggest new hypotheses based on observed patterns
- Run `autoresearch.py` for autonomous exploration

### What you CANNOT do
- Modify `run_experiment.py`, `core/`, `methods/`, `evaluation/`
- Delete or modify entries in the experiment log
- Cherry-pick results (all specs get logged, successful or not)
- Make causal claims from IOp estimates
- Choose specs based on their results (choose based on coverage gaps)

## Strategy
Follow the 60/30/10 rule for spec selection:
- **60% systematic coverage**: Fill gaps in method × measure × income × decomposition space
- **30% hypothesis-driven**: Test specific research questions (e.g., "Does including skin tone change IOp?")
- **10% robustness**: Vary one dimension from important baseline specs

## Workflow per iteration
1. Read previous results and coverage report
2. Identify what's missing or what hypothesis to test
3. Write 5-10 specs in `analyze.py` with clear rationale
4. Execute `python run_experiment.py`
5. Review results and diagnostics
6. Every 25 experiments: generate spec curve plot
7. Repeat until coverage > 80% or human stops

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
- `skin_tone`: Skin tone self-report (p112, A-K → 1-11)
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
- `hh_pc_imputed` (ingc_pc): Imputed per-capita HH income, **0% missing** (primary)
- `hh_total_reported` (p101): Self-reported total HH income, **25.7% missing**
- Log variants: `log_hh_pc_imputed`, `log_hh_total_reported`

## Valid method-decomposition combinations
| Method | Valid decompositions |
|--------|---------------------|
| OLS | lower_bound |
| decision_tree | ex_ante, ex_post |
| cforest | ex_ante, ex_post |
| xgboost | ex_ante |
| random_forest | ex_ante |

## Quality checks
Before submitting specs, verify:
- [ ] No double-log (log income + var_logs)
- [ ] Valid method-decomposition combination
- [ ] At least 1 circumstance
- [ ] Clear rationale for each spec
- [ ] Not duplicating an already-run spec_id
