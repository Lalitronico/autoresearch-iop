# autoresearch-iop

**Specification Curve Analysis for Inequality of Opportunity**

An adaptation of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) for social science. Instead of optimizing a model, this pipeline systematically maps the entire "multiverse" of IOp estimates across methodological choices.

## What is this?

Inequality of Opportunity (IOp) estimates are notoriously sensitive to methodological decisions: which circumstances to include, which income measure to use, which inequality index, which estimation method. A single paper can report IOp shares from 10% to 50% depending on these choices.

This pipeline automates the exploration of that specification space, producing a **specification curve** that transparently shows how IOp estimates vary across all valid methodological combinations.

## Architecture

```
prepare.py (FIXED)          →  data/processed/emovi_analytical.parquet
                                        ↓
analyze.py (AGENT EDITS)    →  ExperimentSpec batch
                                        ↓
run_experiment.py (FIXED)   →  validate → estimate → bootstrap CI → log
                                        ↓
results/experiment_log.jsonl   (APPEND-ONLY)
                                        ↓
synthesis/                  →  spec curve plot, tables, figures
```

The LLM agent only edits `analyze.py` to define experiment batches. Everything else is fixed infrastructure.

## Specification Space

| Dimension | Options | Examples |
|-----------|---------|---------|
| Circumstances | 10 variables | father_education, ethnicity, skin_tone, ... |
| Income | 4 variants | hh_pc_imputed, hh_total_reported, + log |
| Inequality measure | 7 | Gini, MLD, Theil-T, var_logs, Atkinson(ε) |
| Estimation method | 5 | OLS, decision tree, cforest, XGBoost, RF |
| Decomposition | 4 | ex_ante, ex_post, lower_bound, upper_bound |
| Sample filter | 16 | all, male, female, cohorts, urban/rural |

## Karpathy vs. This Pipeline

| Karpathy autoresearch | autoresearch-iop |
|----------------------|-----------------|
| `train.py` (agent edits code) | `analyze.py` (agent defines specs) |
| BPB (single metric, lower = better) | IOp share + CI (no "better", map all) |
| Keep/revert (optimize) | Log everything (multiverse) |
| 5 min/experiment (GPU) | Seconds/experiment (tabular, CPU) |

## Data

Uses ESRU-EMOVI 2023 from Mexico (17,843 respondents, ages 25-64). Place raw `.dta` file in `data/raw/` and run:

```bash
python prepare.py
```

For testing without real data:

```bash
python prepare.py --synthetic
```

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Prepare data (real or synthetic)
python prepare.py --synthetic

# Run experiments
python run_experiment.py

# Run tests
python -m pytest tests/ -v
```

## Safeguards

- **Append-only log**: All results recorded, never deleted
- **Coverage-based exploration**: Agent explores by gaps, not by outcome
- **Spec validation**: Domain rules prevent invalid combinations
- **Diagnostics**: Sample size guards, CI width checks, overfitting detection
- **Reproducibility**: Deterministic seeds, spec hashing

## Project Structure

```
├── core/                     # Types, spec validation, measures, decomposition
├── methods/                  # OLS, decision tree, XGBoost, Random Forest
├── evaluation/               # Bootstrap CI, diagnostics
├── orchestration/            # Logging, coverage tracking, strategy
├── synthesis/                # Spec curve plot, tables, figures
├── tests/                    # 44 tests (unit + integration + e2e)
├── prepare.py                # Data preparation (FIXED)
├── run_experiment.py         # Experiment harness (FIXED)
├── analyze.py                # Agent-editable spec definitions
└── program.md                # Agent instructions
```

## License

MIT

## Acknowledgments

- Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)
- IOp framework: Roemer (1998), Ferreira & Gignoux (2011)
- Data: ESRU-EMOVI 2023, Centro de Estudios Espinosa Yglesias
