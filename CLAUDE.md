# Autoresearch IOp - Project Configuration

## What is this
Specification Curve Analysis pipeline for Inequality of Opportunity (IOp) estimation.
Systematically explores methodological choices to produce a multiverse analysis.

## Architecture
- `core/`: Types, specification, inequality measures, decomposition, data loader (FIXED)
- `methods/`: OLS, decision tree, XGBoost, Random Forest (FIXED)
- `evaluation/`: Metrics with bootstrap CI, diagnostics (FIXED)
- `orchestration/`: Experiment log, coverage tracker, strategy (FIXED)
- `synthesis/`: Spec curve plot, summary tables, figures (FIXED)
- `run_experiment.py`: Fixed harness -- NEVER modify (FIXED)
- `prepare.py`: Data preparation -- NEVER modify (FIXED)
- `analyze.py`: **AGENT EDITS THIS** to define experiment batches
- `program.md`: Instructions for the agent

## Key rules
1. The agent ONLY edits `analyze.py` to define specs
2. All results are logged immutably in `results/experiment_log.jsonl`
3. Explore by coverage gaps, NOT by result values (avoid p-hacking)
4. Every spec needs a `rationale`
5. Valid method-decomposition pairs: OLS->lower_bound, tree->ex_ante/ex_post, xgboost/rf->ex_ante

## Running
```bash
# Generate synthetic data for testing
python prepare.py --synthetic

# Run experiments defined in analyze.py
python run_experiment.py

# Generate spec curve and figures
python -m synthesis.spec_curve
python -m synthesis.figures

# Run tests
python -m pytest tests/ -v
```

## Data
Real data: place ESRU-EMOVI files in `data/raw/`
Synthetic: `python prepare.py --synthetic`
