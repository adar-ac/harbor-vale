from pathlib import Path

import pandas as pd
import pytest

from crews.analyst import stub as analyst
from validation.contract_check import gate
from validation.exceptions import ContractViolation


@pytest.fixture
def crew1_artifacts(tmp_path: Path) -> dict[str, Path]:
    return analyst.run(tmp_path)


def test_clean_data_passes_gate(crew1_artifacts):
    gate(crew1_artifacts["data"], crew1_artifacts["contract"])  # must not raise


def test_money_column_in_cents_is_blocked(crew1_artifacts, tmp_path):
    df = pd.read_csv(crew1_artifacts["data"])
    df["MonthlyCharges"] = df["MonthlyCharges"] * 100  # USD -> cents
    bad_path = tmp_path / "clean_cents.csv"
    df.to_csv(bad_path, index=False)

    with pytest.raises(ContractViolation) as excinfo:
        gate(bad_path, crew1_artifacts["contract"])

    message = str(excinfo.value)
    assert "MonthlyCharges" in message
    assert "USD" in message
    assert {v.column for v in excinfo.value.violations} == {"MonthlyCharges"}
