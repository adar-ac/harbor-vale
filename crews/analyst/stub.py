"""Crew 1 (analyst) STUB.

Writes a hardcoded, already-"clean" Telco sample plus the dataset_contract.json
that describes it. No CrewAI Agents/Tasks and no LLM calls yet; the real crew
will read data/raw/telco.csv and replace this.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CLEAN_FILENAME = "clean.csv"
CONTRACT_FILENAME = "dataset_contract.json"

# A few real-looking Telco rows, subset of columns. Placeholder only.
_ROWS = [
    ("7590-VHVEG", "Female", 0, 1, "Month-to-month", "Electronic check", 29.85, 29.85, "No"),
    ("5575-GNVDE", "Male", 0, 34, "One year", "Mailed check", 56.95, 1889.50, "No"),
    ("3668-QPYBK", "Male", 0, 2, "Month-to-month", "Mailed check", 53.85, 108.15, "Yes"),
    ("7795-CFOCW", "Male", 0, 45, "One year", "Bank transfer (automatic)", 42.30, 1840.75, "No"),
    ("9237-HQITU", "Female", 0, 2, "Month-to-month", "Electronic check", 70.70, 151.65, "Yes"),
    ("9305-CDSKC", "Female", 1, 8, "Month-to-month", "Electronic check", 99.65, 820.50, "Yes"),
]
_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "tenure", "Contract",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]


def build_contract(row_count: int) -> dict:
    """The contract for the stub data. Ranges mirror the real cleaned Telco data."""
    return {
        "contract_version": "1.0",
        "dataset": "telco_customer_churn",
        "produced_by": "crew1_analyst_stub",
        "data_file": CLEAN_FILENAME,
        "target_column": "Churn",
        # Real cleaned Telco is 7032 rows; the stub ships len(_ROWS) with zero tolerance.
        "row_count": {"expected": row_count, "tolerance_pct": 0.0},
        "columns": {
            "customerID": {"dtype": "string", "kind": "identifier", "nullable": False},
            "gender": {"dtype": "string", "kind": "categorical", "allowed_values": ["Female", "Male"]},
            "SeniorCitizen": {"dtype": "int", "kind": "categorical", "allowed_values": [0, 1]},
            "tenure": {"dtype": "int", "kind": "numeric", "min": 1, "max": 72, "unit": "months"},
            "Contract": {
                "dtype": "string",
                "kind": "categorical",
                "allowed_values": ["Month-to-month", "One year", "Two year"],
            },
            "PaymentMethod": {
                "dtype": "string",
                "kind": "categorical",
                "allowed_values": [
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                    "Electronic check",
                    "Mailed check",
                ],
            },
            "MonthlyCharges": {
                "dtype": "float", "kind": "numeric", "monetary": True, "unit": "USD",
                "min": 18.25, "max": 118.75,
            },
            "TotalCharges": {
                "dtype": "float", "kind": "numeric", "monetary": True, "unit": "USD",
                "min": 18.8, "max": 8684.8,
            },
            "Churn": {"dtype": "string", "kind": "categorical", "allowed_values": ["No", "Yes"]},
        },
    }


def run(out_dir: Path) -> dict[str, Path]:
    """Write clean.csv + dataset_contract.json into out_dir; return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / CLEAN_FILENAME
    contract_path = out_dir / CONTRACT_FILENAME

    pd.DataFrame(_ROWS, columns=_COLUMNS).to_csv(data_path, index=False)
    contract_path.write_text(json.dumps(build_contract(len(_ROWS)), indent=2), encoding="utf-8")
    return {"data": data_path, "contract": contract_path}
