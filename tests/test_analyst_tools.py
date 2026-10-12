"""Deterministic Crew 1 tools on the real raw data. No LLM calls."""

from pathlib import Path

import pytest

from crews.analyst import tools as T
from validation.contract_check import gate

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco.csv"
pytestmark = pytest.mark.skipif(not RAW.exists(), reason="data/raw/telco.csv not present")


def test_clean_and_contract_pass_gate(tmp_path):
    clean_path = tmp_path / T.CLEAN_FILENAME
    contract_path = tmp_path / T.CONTRACT_FILENAME

    summary = T.clean(RAW, clean_path)
    assert (summary["rows_in"], summary["rows_dropped"], summary["rows_out"]) == (7043, 11, 7032)
    assert summary["columns_out"] == 21

    contract = T.write_contract(clean_path, contract_path)
    gate(clean_path, contract_path)  # must not raise

    assert contract["produced_by"] == "crew1_analyst"
    assert contract["target_column"] == "Churn"
    assert contract["row_count"] == {"expected": 7032, "tolerance_pct": 0.0}
    cols = contract["columns"]
    assert len(cols) == 21
    assert {s["dtype"] for s in cols.values()} <= {"int", "float", "bool", "string"}
    assert cols["customerID"] == {"dtype": "string", "kind": "identifier", "nullable": False}
    for money in ("MonthlyCharges", "TotalCharges"):
        assert cols[money]["monetary"] is True and cols[money]["unit"] == "USD"
    assert cols["TotalCharges"]["dtype"] == "float"
