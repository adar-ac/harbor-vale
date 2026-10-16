"""Deterministic Crew 2 tools on Crew 1's real handoff. No LLM calls."""

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from crews.analyst import tools as A
from crews.scientist import dryrun, tools as T

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco.csv"
pytestmark = pytest.mark.skipif(not RAW.exists(), reason="data/raw/telco.csv not present")


@pytest.fixture(scope="module")
def handoff(tmp_path_factory) -> dict[str, Path]:
    """Crew 1's two output files, produced by its deterministic tools."""
    d = tmp_path_factory.mktemp("crew1")
    data, contract = d / A.CLEAN_FILENAME, d / A.CONTRACT_FILENAME
    A.clean(RAW, data)
    A.write_contract(data, contract)
    return {"data": data, "contract": contract}


@pytest.fixture(scope="module")
def trained(handoff, tmp_path_factory) -> dict:
    d = tmp_path_factory.mktemp("crew2")
    paths = {"features": d / T.FEATURES_FILENAME, "model": d / T.MODEL_FILENAME,
             "metrics": d / T.METRICS_FILENAME}
    summary = T.engineer_features(handoff["data"], handoff["contract"], paths["features"])
    metrics = T.train_and_compare(paths["features"], paths["model"], paths["metrics"])
    return {"paths": paths, "summary": summary, "metrics": metrics}


def test_features_follow_contract(trained, handoff):
    features = pd.read_csv(trained["paths"]["features"])
    contract = json.loads(handoff["contract"].read_text(encoding="utf-8"))

    assert len(features) == contract["row_count"]["expected"]
    assert "customerID" not in features.columns
    assert trained["summary"]["dropped_columns"] == ["customerID"]
    assert features.columns[-1] == "Churn"
    assert set(features["Churn"].unique()) == {0, 1}
    assert all(pd.api.types.is_numeric_dtype(features[c]) for c in features.columns)
    assert not features.isna().any().any()
    # every non-identifier, non-target contract column feeds at least one feature
    for name, spec in contract["columns"].items():
        if spec["kind"] != "identifier" and name != "Churn":
            assert any(c == name or c.startswith(name + "_") for c in features.columns), name


def test_model_and_metrics_are_valid(trained):
    paths, metrics = trained["paths"], trained["metrics"]
    assert set(metrics["models"]) == {"logistic_regression", "random_forest"}

    for result in metrics["models"].values():
        test = result["test"]
        for key in ("precision", "recall", "f1", "roc_auc", "accuracy"):
            assert 0.0 <= test[key] <= 1.0, key
        assert test["roc_auc"] > 0.75  # far better than chance on Telco churn
        assert sum(test["confusion_matrix"].values()) == metrics["data"]["rows_test"]

    best = max(metrics["models"], key=lambda n: metrics["models"][n]["cv_roc_auc_mean"])
    assert metrics["selected_model"] == best
    assert json.loads(paths["metrics"].read_text(encoding="utf-8")) == metrics

    model = joblib.load(paths["model"])
    X = pd.read_csv(paths["features"]).drop(columns="Churn")
    proba = model.predict_proba(X.head(50))[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    T.verify_saved_model(paths["features"], paths["model"], paths["metrics"])  # must not raise


def test_training_is_reproducible(trained, tmp_path):
    again = T.train_and_compare(trained["paths"]["features"], tmp_path / "m.pkl", tmp_path / "m.json")
    assert again["models"] == trained["metrics"]["models"]


def test_dryrun_writes_all_artifacts(handoff, tmp_path):
    paths = dryrun.run(handoff["data"], handoff["contract"], tmp_path)
    assert all(p.is_file() for p in paths.values())
    for key in ("evaluation_report", "model_card"):
        text = paths[key].read_text(encoding="utf-8")
        assert T.METRICS_TABLE_PLACEHOLDER not in text
        assert "| random_forest |" in text and "| logistic_regression |" in text
    assert "## Ethical considerations" in paths["model_card"].read_text(encoding="utf-8")
