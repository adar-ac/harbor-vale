"""Deterministic contract check between Crew 1 (analyst) and Crew 2 (scientist).

Pure pandas + stdlib. No LLM calls, no network. See README.md for the
dataset_contract.json schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pandas.api import types as ptypes

from validation.exceptions import ContractViolation, Violation

# Logical dtypes used in the contract -> pandas predicate. Logical names keep the
# contract stable across pandas versions (pandas 3 reads text as "str", not "object").
_DTYPE_CHECKS = {
    "int": ptypes.is_integer_dtype,
    "float": ptypes.is_float_dtype,
    "bool": ptypes.is_bool_dtype,
    "string": lambda s: ptypes.is_string_dtype(s) or ptypes.is_object_dtype(s),
}


def load_contract(contract_path: Path) -> dict:
    return json.loads(Path(contract_path).read_text(encoding="utf-8"))


def validate(data_path: Path, contract_path: Path) -> list[Violation]:
    """Check the CSV at `data_path` against the contract. Returns ALL violations
    found (empty list means the data passes). Never raises on a data problem."""
    contract = load_contract(contract_path)
    df = pd.read_csv(Path(data_path))
    columns: dict[str, dict] = contract["columns"]
    violations: list[Violation] = []

    def add(check: str, column: str | None, message: str) -> None:
        violations.append(Violation(check, column, message))

    # --- contract self-consistency ---------------------------------------
    target = contract["target_column"]
    if target not in columns:
        add("contract", target, f"{target}: target_column is not declared in contract columns")
    for name, spec in columns.items():
        if spec.get("monetary") and not spec.get("unit"):
            add("contract", name, f"{name}: monetary column has no 'unit' in the contract")

    # --- column set ----------------------------------------------------------
    if target not in df.columns:
        add("target_missing", target, f"{target}: target column is missing from the data")
    for name in columns.keys() - set(df.columns):
        if name != target:  # already reported above
            add("missing_column", name, f"{name}: declared in contract but missing from data")
    for name in set(df.columns) - columns.keys():
        add("extra_column", name, f"{name}: present in data but not declared in contract")

    # --- per-column checks ----------------------------------------------------
    for name, spec in columns.items():
        if name not in df.columns:
            continue
        s = df[name]
        dtype = spec["dtype"]

        dtype_ok = _DTYPE_CHECKS[dtype](s)
        if not dtype_ok:
            add("dtype", name, f"{name}: expected dtype {dtype!r}, got {str(s.dtype)!r}")

        nulls = int(s.isna().sum())
        if nulls and not spec.get("nullable", False):
            add("nulls", name, f"{name}: {nulls} null value(s) in a non-nullable column")

        values = s.dropna()

        if ("min" in spec or "max" in spec) and dtype_ok and ptypes.is_numeric_dtype(s):
            lo, hi = spec.get("min", float("-inf")), spec.get("max", float("inf"))
            bad = values[(values < lo) | (values > hi)]
            if len(bad):
                unit = spec.get("unit")
                unit_txt = f" {unit}" if unit else ""
                msg = (
                    f"{name}: {len(bad)} value(s) outside declared range "
                    f"[{lo}, {hi}]{unit_txt} (observed {bad.min()}..{bad.max()})"
                )
                if unit:
                    msg += f"; contract unit is {unit!r} - check for a unit mismatch (e.g. cents vs USD)"
                add("range", name, msg)

        if "allowed_values" in spec:
            unexpected = set(values.unique().tolist()) - set(spec["allowed_values"])
            if unexpected:
                shown = sorted(map(str, unexpected))[:10]
                add(
                    "categorical",
                    name,
                    f"{name}: {len(unexpected)} unexpected value(s) {shown}; "
                    f"allowed {spec['allowed_values']}",
                )

    # --- row count drift ------------------------------------------------------
    rc = contract["row_count"]
    expected, tol_pct = rc["expected"], rc.get("tolerance_pct", 0.0)
    actual = len(df)
    drift_pct = abs(actual - expected) / expected * 100 if expected else float(actual > 0) * 100
    if drift_pct > tol_pct:
        add(
            "row_count",
            None,
            f"row count {actual} differs from expected {expected} by {drift_pct:.2f}% "
            f"(tolerance {tol_pct}%)",
        )

    return violations


def gate(data_path: Path, contract_path: Path) -> None:
    """Raise ContractViolation (with every violation) if the data fails the contract."""
    violations = validate(data_path, contract_path)
    if violations:
        raise ContractViolation(violations)
