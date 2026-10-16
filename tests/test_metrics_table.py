"""The metrics table in crew2 documents is filled by code, whatever token the LLM echoes."""

import pytest

from crews.scientist import tools as T

_TEST = {"precision": 0.5, "recall": 0.8, "f1": 0.6, "roc_auc": 0.84, "accuracy": 0.75,
         "confusion_matrix": {"tn": 1, "fp": 2, "fn": 3, "tp": 4}}
METRICS = {
    "models": {
        "logistic_regression": {"test": _TEST, "cv_roc_auc_mean": 0.84},
        "random_forest": {"test": _TEST, "cv_roc_auc_mean": 0.85},
    },
    "selected_model": "random_forest",
}


@pytest.mark.parametrize("token", [
    "{{METRICS_TABLE}}",
    "{METRICS_TABLE}",
    "{{ METRICS_TABLE }}",
    "`{{METRICS_TABLE}}`",
    "{{{METRICS_TABLE}}}",
])
def test_placeholder_variants_are_replaced_in_place(token):
    doc = f"# Card\n\n## Metrics\n\n{token}\n\n## Limitations\n- x\n"
    out = T.fill_metrics_table(doc, METRICS)
    assert "METRICS_TABLE" not in out
    table = T.metrics_table_md(METRICS)
    assert f"## Metrics\n\n{table}\n\n## Limitations" in out  # in place, no stray `{}


def test_missing_placeholder_appends_table():
    out = T.fill_metrics_table("# Card\n\nNo table here.\n", METRICS)
    assert out.endswith(T.metrics_table_md(METRICS) + "\n")


def test_unfillable_mention_raises():
    with pytest.raises(RuntimeError, match="placeholder"):
        T.fill_metrics_table("see METRICS_TABLE above\n", METRICS)
