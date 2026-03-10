"""Append-only experiment log in JSONL and TSV formats.

Every experiment result is logged immutably. No deletion, no modification.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
JSONL_PATH = RESULTS_DIR / "experiment_log.jsonl"
TSV_PATH = RESULTS_DIR / "experiment_log.tsv"

TSV_COLUMNS = [
    "timestamp",
    "spec_id",
    "method",
    "decomposition_type",
    "income_variable",
    "inequality_measure",
    "circumstances",
    "sample_filter",
    "iop_share",
    "ci_lower",
    "ci_upper",
    "total_inequality",
    "n_obs",
    "n_types",
    "r_squared",
    "status",
    "runtime_seconds",
    "flags",
]


def log_experiment(record: dict[str, Any]) -> None:
    """Append a single experiment record to JSONL and TSV logs.

    Parameters
    ----------
    record : dict
        Must contain at minimum: spec_id, status.
        Full record includes all spec fields + results + diagnostics.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Add timestamp
    record["timestamp"] = datetime.now(timezone.utc).isoformat()

    # --- JSONL (full record) ---
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")

    # --- TSV (summary columns) ---
    tsv_exists = TSV_PATH.exists() and TSV_PATH.stat().st_size > 0
    with open(TSV_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TSV_COLUMNS, delimiter="\t", extrasaction="ignore")
        if not tsv_exists:
            writer.writeheader()

        # Flatten for TSV
        row = {col: record.get(col, "") for col in TSV_COLUMNS}
        if isinstance(row.get("circumstances"), (list, tuple)):
            row["circumstances"] = "|".join(row["circumstances"])
        if isinstance(row.get("flags"), (list, tuple)):
            row["flags"] = "|".join(row["flags"])
        writer.writerow(row)

    logger.info(f"Logged experiment {record.get('spec_id', 'unknown')} ({record.get('status', '?')})")


def load_experiment_log() -> list[dict[str, Any]]:
    """Load all experiments from JSONL log."""
    if not JSONL_PATH.exists():
        return []

    records = []
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_completed_spec_ids() -> set[str]:
    """Get set of spec_ids that have already been run successfully."""
    records = load_experiment_log()
    return {r["spec_id"] for r in records if r.get("status") == "success"}


def get_experiment_count() -> int:
    """Count total experiments logged."""
    if not JSONL_PATH.exists():
        return 0
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())
