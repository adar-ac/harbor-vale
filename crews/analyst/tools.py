"""Deterministic data functions for Crew 1 (analyst).

Every number that ends up in clean.csv or dataset_contract.json is computed here by
pandas, never by an LLM. crew.py wraps these as CrewAI tools; tests call them directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pandas.api import types as ptypes

CLEAN_FILENAME = "clean.csv"
CONTRACT_FILENAME = "dataset_contract.json"
TARGET = "Churn"
ID_COLUMN = "customerID"

# Declared metadata (not statistics): units and money flags per column.
_UNITS = {"tenure": "months", "MonthlyCharges": "USD", "TotalCharges": "USD"}
_MONETARY = {"MonthlyCharges", "TotalCharges"}


def _logical_dtype(s: pd.Series) -> str:
    if ptypes.is_bool_dtype(s):
        return "bool"
    if ptypes.is_integer_dtype(s):
        return "int"
    if ptypes.is_float_dtype(s):
        return "float"
    return "string"


def _py(v):
    """numpy scalar -> plain Python for JSON."""
    return v.item() if hasattr(v, "item") else v


def profile(raw_path: Path) -> dict:
    """Structure, dtypes, missing values and cardinality of the raw CSV."""
    df = pd.read_csv(raw_path)
    columns = {}
    for name in df.columns:
        s = df[name]
        info = {
            "pandas_dtype": str(s.dtype),
            "nulls": int(s.isna().sum()),
            "n_unique": int(s.nunique()),
        }
        if not ptypes.is_numeric_dtype(s):
            info["blank_strings"] = int((s.astype(str).str.strip() == "").sum())
            if s.nunique() <= 10:
                info["values"] = sorted(s.dropna().unique().tolist())
            # Flag text columns that are numeric apart from a few unparseable cells.
            coerced = pd.to_numeric(s, errors="coerce")
            if coerced.notna().mean() > 0.95:
                info["numeric_stored_as_text"] = True
                info["unparseable_as_number"] = int(coerced.isna().sum())
        else:
            info["min"], info["max"] = _py(s.min()), _py(s.max())
        columns[name] = info
    return {"rows": len(df), "n_columns": df.shape[1], "columns": columns}


def clean(raw_path: Path, out_path: Path) -> dict:
    """Apply the fixed cleaning decisions and write clean.csv.

    - TotalCharges -> numeric (errors="coerce"); blanks become NaN
    - drop rows where TotalCharges is NaN (tenure=0 customers, never billed)
    - keep all columns
    """
    df = pd.read_csv(raw_path)
    rows_in = len(df)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    dropped = df[df["TotalCharges"].isna()]
    if not (dropped["tenure"] == 0).all():
        raise ValueError("unexpected: some rows with blank TotalCharges have tenure != 0")
    df = df.dropna(subset=["TotalCharges"])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return {
        "output": str(out_path),
        "rows_in": rows_in,
        "rows_dropped": len(dropped),
        "rows_out": len(df),
        "columns_out": df.shape[1],
        "dropped_customer_ids": dropped[ID_COLUMN].tolist(),
        "dropped_rows_tenure_values": sorted(dropped["tenure"].unique().tolist()),
        "TotalCharges_dtype_after": str(df["TotalCharges"].dtype),
    }


def build_contract(clean_path: Path, produced_by: str = "crew1_analyst") -> dict:
    """Derive dataset_contract.json entirely from the cleaned file."""
    df = pd.read_csv(clean_path)
    columns: dict[str, dict] = {}
    for name in df.columns:
        s = df[name]
        spec: dict = {"dtype": _logical_dtype(s)}
        if name == ID_COLUMN:
            spec["kind"] = "identifier"
        elif ptypes.is_numeric_dtype(s) and s.nunique() > 2:
            spec["kind"] = "numeric"
            spec["min"], spec["max"] = _py(s.min()), _py(s.max())
        else:  # text columns and 0/1 flags like SeniorCitizen
            spec["kind"] = "categorical"
            spec["allowed_values"] = sorted(_py(v) for v in s.dropna().unique())
        spec["nullable"] = bool(s.isna().any())
        if name in _MONETARY:
            spec["monetary"] = True
        if name in _UNITS:
            spec["unit"] = _UNITS[name]
        columns[name] = spec

    return {
        "contract_version": "1.0",
        "dataset": "telco_customer_churn",
        "produced_by": produced_by,
        "data_file": Path(clean_path).name,
        "target_column": TARGET,
        "row_count": {"expected": len(df), "tolerance_pct": 0.0},
        "columns": columns,
    }


def write_contract(clean_path: Path, contract_path: Path) -> dict:
    contract = build_contract(clean_path)
    Path(contract_path).write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return contract


def eda_summary(clean_path: Path) -> dict:
    """Churn rates and distributions on the cleaned data, for the narrative reports."""
    df = pd.read_csv(clean_path)
    churn = df[TARGET].eq("Yes")
    out: dict = {
        "rows": len(df),
        "churn_count": int(churn.sum()),
        "churn_rate_pct": round(churn.mean() * 100, 2),
        "numeric": {},
        "churn_rate_by_category_pct": {},
    }
    for name in df.columns:
        s = df[name]
        if name in (ID_COLUMN, TARGET):
            continue
        if ptypes.is_numeric_dtype(s) and s.nunique() > 2:
            out["numeric"][name] = {
                "mean": round(float(s.mean()), 2),
                "median": round(float(s.median()), 2),
                "min": _py(s.min()),
                "max": _py(s.max()),
                "mean_if_churned": round(float(s[churn].mean()), 2),
                "mean_if_retained": round(float(s[~churn].mean()), 2),
            }
        else:
            g = churn.groupby(s)
            out["churn_rate_by_category_pct"][name] = {
                str(k): {"customers": int(n), "churn_rate_pct": round(r * 100, 2)}
                for k, n, r in zip(g.size().index, g.size(), g.mean())
            }
    return out
