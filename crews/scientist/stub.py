"""Crew 2 (scientist) STUB.

Reads ONLY the cleaned file and the contract handed over by Crew 1, and writes a
placeholder model report. No training, no CrewAI Agents/Tasks, no LLM calls yet.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPORT_FILENAME = "model_report.json"


def run(data_path: Path, contract_path: Path, out_dir: Path) -> Path:
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    df = pd.read_csv(Path(data_path))
    target = contract["target_column"]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / REPORT_FILENAME
    report = {
        "status": "placeholder",
        "model": None,
        "trained_on": Path(data_path).name,
        "contract_version": contract["contract_version"],
        "target_column": target,
        "rows_seen": len(df),
        "target_distribution": df[target].value_counts().sort_index().to_dict(),
        "metrics": {},
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path
