# Autoresearch IOp - Project Configuration

## What is this
Specification Curve Analysis pipeline for Inequality of Opportunity (IOp) estimation.
Systematically explores methodological choices to produce a multiverse analysis.
Aligned with CEEY official methodology for ESRU-EMOVI data.

## Agent-in-the-Loop
The LLM agent (Claude Code) IS the research agent. The agent reasons about what to explore,
interprets results in the context of IOp theory, and decides next steps.
`autoresearch.py` is a diagnostic toolkit the agent uses -- NOT an autonomous loop.

## Architecture
- `core/`: Types (33 circumstances), specification (`use_mi` field), inequality measures, decomposition, data loader (MI-aware) (FIXED)
- `methods/`: OLS, decision tree, XGBoost, Random Forest (FIXED)
- `evaluation/`: Metrics with bootstrap CI + Rubin's rules MI pooling, diagnostics (FIXED)
- `orchestration/`: Experiment log (MI columns), coverage tracker, strategy (13 circ sets) (FIXED)
- `synthesis/`: Spec curve plot (`use_mi` column), summary tables (listwise vs MI), figures (FIXED)
- `imputation/`: MICE via miceforest -- `mice_imputer.py` creates/saves/loads M imputed datasets (FIXED)
- `run_experiment.py`: Fixed harness with MI branch -- NEVER modify (FIXED)
- `prepare.py`: Data preparation + CEEY alignment + `--impute` flag -- NEVER modify (FIXED)
- `autoresearch.py`: **Agent diagnostic toolkit** (status, findings, gaps, recent, synthesize) (FIXED)
- `analyze.py`: **AGENT EDITS THIS** to define experiment batches
- `program.md`: Agent-in-the-loop instructions and reference

## Key rules
1. The agent ONLY edits `analyze.py` to define specs
2. All results are logged immutably in `results/experiment_log.jsonl`
3. Explore by coverage gaps + scientific value, NOT by result values (avoid p-hacking)
4. Every spec needs a `rationale`
5. Valid method-decomposition pairs: OLS->lower_bound, tree->ex_ante/ex_post, xgboost/rf->ex_ante

## Agent workflow
```bash
# 1. Diagnose current state
python autoresearch.py status          # Coverage + experiment summary
python autoresearch.py findings        # Pattern detection
python autoresearch.py gaps            # Missing specification slots
python autoresearch.py gaps --mi       # MI-specific gaps
python autoresearch.py recent          # Last 10 results
python autoresearch.py recent 20       # Last 20 results

# 2. Agent edits analyze.py with justified specs

# 3. Execute experiments
python run_experiment.py

# 4. Generate synthesis
python autoresearch.py synthesize      # Spec curve + tables + figures
```

## Getting started
```bash
# Generate synthetic data for testing
python prepare.py --synthetic

# Generate imputed datasets (M=20, requires miceforest)
python prepare.py --impute --impute-m 20

# Run experiments defined in analyze.py
python run_experiment.py

# Run tests
python -m pytest tests/ -v
```

## Multiple Imputation
- `prepare.py --impute`: creates M imputed datasets via MICE (miceforest), saves to `data/processed/imputed/`
- Specs with `use_mi=True` use Rubin's rules to pool across M imputations
- Tiered bootstrap per imputation: OLS/DT=100, XGBoost=10, RF=5
- MI preserves full sample size (no listwise deletion on circumstances)

## Data
Real data: place ESRU-EMOVI files in `data/raw/`
Synthetic: `python prepare.py --synthetic`
