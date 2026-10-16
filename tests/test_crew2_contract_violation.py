"""Crew 2 ContractViolation test: the flow stops at the gate and Crew 2 never runs
when clean.csv is missing a column the contract requires. No LLM calls."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from crews.analyst import stub as analyst
from flows import main_flow
from validation.exceptions import ContractViolation

DROPPED = "tenure"  # declared in the stub contract, not the target


@pytest.fixture
def flow_with_missing_column(tmp_path: Path, monkeypatch) -> dict:
    """Wire the real flow to a Crew 1 that drops a contracted column and a spy Crew 2."""

    def broken_analyst_run(out_dir: Path) -> dict[str, Path]:
        artifacts = analyst.run(out_dir)
        df = pd.read_csv(artifacts["data"])
        df.drop(columns=DROPPED).to_csv(artifacts["data"], index=False)
        return artifacts

    scientist_calls = []

    def spy_scientist_run(*args) -> dict[str, Path]:
        scientist_calls.append(args)
        return {}

    monkeypatch.setattr(main_flow, "analyst", SimpleNamespace(__name__="broken_analyst", run=broken_analyst_run))
    monkeypatch.setattr(main_flow, "scientist", SimpleNamespace(__name__="spy_scientist", run=spy_scientist_run))
    monkeypatch.setattr(main_flow, "CREW1_DIR", tmp_path / "crew1")
    monkeypatch.setattr(main_flow, "CREW2_DIR", tmp_path / "crew2")
    return {"scientist_calls": scientist_calls, "crew2_dir": tmp_path / "crew2"}


def test_missing_column_stops_flow_before_crew2(flow_with_missing_column):
    with pytest.raises(ContractViolation) as excinfo:
        main_flow.ChurnPipelineFlow().kickoff()

    message = str(excinfo.value)
    assert DROPPED in message
    assert "missing from data" in message
    assert [(v.check, v.column) for v in excinfo.value.violations] == [("missing_column", DROPPED)]

    assert flow_with_missing_column["scientist_calls"] == []  # Crew 2 never ran
    assert not flow_with_missing_column["crew2_dir"].exists()  # and wrote nothing
