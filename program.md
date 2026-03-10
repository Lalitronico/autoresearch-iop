# Program: IOp Specification Curve Analysis

## Objective
Systematically explore the space of methodological specifications for
Inequality of Opportunity (IOp) estimation using ESRU-EMOVI data from Mexico.

## Your role
You are a research agent that designs and executes IOp experiments. You modify
`analyze.py` to define batches of `ExperimentSpec`s, then the fixed harness
(`run_experiment.py`) executes them.

## Rules

### What you CAN do
- Edit `analyze.py` to define new batches of specs
- Read experiment results from `results/experiment_log.jsonl`
- Read coverage reports via `orchestration/coverage_tracker.py`
- Generate synthesis outputs (spec curves, tables, figures)
- Suggest new hypotheses based on observed patterns

### What you CANNOT do
- Modify `run_experiment.py`, `core/`, `methods/`, `evaluation/`
- Delete or modify entries in the experiment log
- Cherry-pick results (all specs get logged, successful or not)
- Make causal claims from IOp estimates
- Choose specs based on their results (choose based on coverage gaps)

## Strategy
Follow the 60/30/10 rule for spec selection:
- **60% systematic coverage**: Fill gaps in method x measure x income x decomposition space
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

## Circumstance variables (from ESRU-EMOVI 2023)
- `father_education`: Father's education (educp, 4 categories: primaria o menos, secundaria, media superior, profesional)
- `mother_education`: Mother's education (educm, 4 categories, same)
- `father_occupation`: Father's occupational class (clasep, 6 categories)
- `ethnicity`: Ethnic self-identification (p110, 5 categories)
- `indigenous_language`: Indigenous language speaker (p111, binary 0/1)
- `skin_tone`: Skin tone self-report (p112, letters A-K -> numeric 1-11)
- `skin_tone_cielab`: Skin tone colorimeter CIELab L* (p113dL, continuous)
- `region_14`: Region at age 14 (5 regions)
- `rural_14`: Rural at age 14 (p21, binarized from 5 levels)
- `gender`: Gender (sexo, recoded: 1=male, 0=female)

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
