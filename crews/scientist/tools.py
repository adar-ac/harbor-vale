"""Deterministic modelling functions for Crew 2 (scientist).

Inputs are ONLY Crew 1's handoff: clean.csv + dataset_contract.json. The feature list,
encodings and target come from the contract. Every metric in metrics.json, the
evaluation report and the model card is computed here by sklearn, never by an LLM.
crew.py wraps these as CrewAI tools; dryrun.py and tests call them directly.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_FILENAME = "features.csv"
MODEL_FILENAME = "model.pkl"
METRICS_FILENAME = "metrics.json"

# Declared modelling choices (not statistics).
POSITIVE_LABEL = "Yes"  # churned
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
THRESHOLD = 0.5
SELECTION_METRIC = "roc_auc"  # chosen on cross-validation of the training split only
METRICS_TABLE_PLACEHOLDER = "{{METRICS_TABLE}}"
# LLMs don't always echo the token verbatim ({METRICS_TABLE}, {{ METRICS_TABLE }},
# `{{METRICS_TABLE}}`), so match any brace/space/backtick wrapping of the name.
_PLACEHOLDER_RE = re.compile(r"`*\{+\s*METRICS_TABLE\s*\}+`*")


def _candidates() -> dict:
    # class_weight="balanced": churn is ~27% positive, so unweighted models favour "No".
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def load_contract(contract_path: Path) -> dict:
    return json.loads(Path(contract_path).read_text(encoding="utf-8"))


def feature_plan(contract: dict) -> dict:
    """Split the contract's columns into identifier / target / numeric / categorical."""
    target = contract["target_column"]
    plan = {"target": target, "identifiers": [], "numeric": [], "categorical": {}}
    for name, spec in contract["columns"].items():
        if name == target:
            continue
        if spec["kind"] == "identifier":
            plan["identifiers"].append(name)
        elif spec["kind"] == "numeric":
            plan["numeric"].append(name)
        elif spec["kind"] == "categorical":
            plan["categorical"][name] = spec["allowed_values"]
        else:
            raise ValueError(f"{name}: unknown kind {spec['kind']!r} in contract")
    if POSITIVE_LABEL not in contract["columns"][target].get("allowed_values", []):
        raise ValueError(f"positive label {POSITIVE_LABEL!r} not among {target}'s allowed_values")
    return plan


def engineer_features(data_path: Path, contract_path: Path, out_path: Path) -> dict:
    """Encode clean.csv into an all-numeric features.csv.

    - identifier columns (customerID) are dropped
    - numeric columns pass through unchanged
    - categoricals are one-hot encoded over the contract's allowed_values, dropping the
      first level, so the column set depends on the contract, not on the rows present
    - the target becomes 0/1 (1 = churned)
    """
    contract = load_contract(contract_path)
    plan = feature_plan(contract)
    df = pd.read_csv(data_path)
    target = plan["target"]

    parts = [df[plan["numeric"]]]
    for name, levels in plan["categorical"].items():
        cat = pd.Categorical(df[name], categories=levels)
        dummies = pd.get_dummies(cat, prefix=name, drop_first=True, dtype=int)
        dummies.index = df.index
        parts.append(dummies)
    parts.append(df[target].eq(POSITIVE_LABEL).astype(int).rename(target))
    features = pd.concat(parts, axis=1)

    if features.isna().any().any():
        raise ValueError("features contain NaN; data does not match the contract")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path, index=False)
    return {
        "output": str(out_path),
        "rows": len(features),
        "dropped_columns": plan["identifiers"],
        "numeric_features": plan["numeric"],
        "categorical_features_encoded": list(plan["categorical"]),
        "n_model_features": features.shape[1] - 1,
        "model_features": [c for c in features.columns if c != target],
        "target": target,
        "target_encoding": {
            v: int(v == POSITIVE_LABEL) for v in contract["columns"][target]["allowed_values"]
        },
    }


def _split(features_path: Path):
    df = pd.read_csv(features_path)
    target = df.columns[-1]
    X, y = df.drop(columns=target), df[target]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def _r(v: float) -> float:
    return round(float(v), 4)


def evaluate(model, X, y) -> dict:
    """Hold-out metrics at THRESHOLD. Positive class = churned (1)."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "precision": _r(precision_score(y, pred, zero_division=0)),
        "recall": _r(recall_score(y, pred)),
        "f1": _r(f1_score(y, pred)),
        "roc_auc": _r(roc_auc_score(y, proba)),
        "accuracy": _r(accuracy_score(y, pred)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _top_features(model, columns, n: int = 10) -> list[dict]:
    if hasattr(model, "feature_importances_"):
        weights, kind = model.feature_importances_, "importance"
    else:  # pipeline ending in LogisticRegression, on standardised inputs
        weights, kind = model[-1].coef_[0], "standardised_coefficient"
    ranked = sorted(zip(columns, weights), key=lambda cw: abs(cw[1]), reverse=True)[:n]
    return [{"feature": c, kind: _r(w)} for c, w in ranked]


def _describe(model) -> dict:
    est = model[-1] if hasattr(model, "steps") else model
    keys = ("class_weight", "max_iter", "n_estimators", "min_samples_leaf", "random_state")
    p = est.get_params()
    out = {k: p[k] for k in keys if k in p}
    out["estimator"] = type(est).__name__
    if hasattr(model, "steps"):
        out["preprocessing"] = "StandardScaler"
    return out


def train_and_compare(features_path: Path, model_path: Path, metrics_path: Path) -> dict:
    """Train logistic regression and random forest on the same stratified split,
    pick the one with the higher cross-validated ROC-AUC on the training split, save it
    to model.pkl, and write every metric to metrics.json."""
    X_train, X_test, y_train, y_test = _split(features_path)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    models, results = {}, {}
    for name, model in _candidates().items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=SELECTION_METRIC)
        model.fit(X_train, y_train)
        models[name] = model
        results[name] = {
            "params": _describe(model),
            f"cv_{SELECTION_METRIC}_mean": _r(cv_scores.mean()),
            f"cv_{SELECTION_METRIC}_std": _r(cv_scores.std()),
            "test": evaluate(model, X_test, y_test),
            "top_features": _top_features(model, list(X_train.columns)),
        }

    selected = max(results, key=lambda n: results[n][f"cv_{SELECTION_METRIC}_mean"])
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models[selected], model_path)

    metrics = {
        "data": {
            "features_file": Path(features_path).name,
            "rows_train": len(X_train),
            "rows_test": len(X_test),
            "n_features": X_train.shape[1],
            "churn_rate_train": _r(y_train.mean()),
            "churn_rate_test": _r(y_test.mean()),
        },
        "setup": {
            "split": f"stratified hold-out, test_size={TEST_SIZE}",
            "random_state": RANDOM_STATE,
            "cv_folds": CV_FOLDS,
            "selection_metric": f"cv_{SELECTION_METRIC}_mean",
            "decision_threshold": THRESHOLD,
            "positive_class": f"{POSITIVE_LABEL} (churned) = 1",
        },
        "models": results,
        "selected_model": selected,
        "model_file": model_path.name,
    }
    Path(metrics_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def verify_saved_model(features_path: Path, model_path: Path, metrics_path: Path) -> None:
    """Reload model.pkl, re-score the hold-out split and require it to match metrics.json."""
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    _, X_test, _, y_test = _split(features_path)
    model = joblib.load(model_path)
    recomputed = evaluate(model, X_test, y_test)
    if recomputed != metrics["models"][metrics["selected_model"]]["test"]:
        raise RuntimeError("model.pkl does not reproduce the metrics recorded in metrics.json")


def training_data_summary(data_path: Path, contract_path: Path, features_path: Path) -> dict:
    """What the model was trained on, for the model card."""
    contract = load_contract(contract_path)
    plan = feature_plan(contract)
    df = pd.read_csv(data_path)
    target = plan["target"]
    return {
        "source_files": [Path(data_path).name, Path(contract_path).name],
        "dataset": contract["dataset"],
        "contract_version": contract["contract_version"],
        "produced_by": contract["produced_by"],
        "rows": len(df),
        "target": target,
        "class_counts": {str(k): int(v) for k, v in df[target].value_counts().sort_index().items()},
        "churn_rate_pct": round(df[target].eq(POSITIVE_LABEL).mean() * 100, 2),
        "dropped_identifier_columns": plan["identifiers"],
        "numeric_features": {
            n: {k: contract["columns"][n].get(k) for k in ("min", "max", "unit") if k in contract["columns"][n]}
            for n in plan["numeric"]
        },
        "categorical_features": list(plan["categorical"]),
        "n_model_features_after_encoding": pd.read_csv(features_path, nrows=0).shape[1] - 1,
    }


def metrics_table_md(metrics: dict) -> str:
    """Markdown comparison table rendered from metrics.json (hold-out set)."""
    cols = ["precision", "recall", "f1", "roc_auc", "accuracy"]
    cv_key = f"cv_{SELECTION_METRIC}_mean"
    lines = [
        "| Model | " + " | ".join(cols) + f" | {cv_key} | selected |",
        "|---|" + "---|" * (len(cols) + 2),
    ]
    for name, r in metrics["models"].items():
        vals = " | ".join(f"{r['test'][c]:.4f}" for c in cols)
        sel = "yes" if name == metrics["selected_model"] else ""
        lines.append(f"| {name} | {vals} | {r[cv_key]:.4f} | {sel} |")
    for name, r in metrics["models"].items():
        cm = r["test"]["confusion_matrix"]
        lines.append("")
        lines.append(f"Confusion matrix, {name} (hold-out, threshold {THRESHOLD}): "
                     f"TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")
    return "\n".join(lines)


def fill_metrics_table(text: str, metrics: dict) -> str:
    """Insert the generated table at the placeholder, or append it if the agent dropped it.
    Raises if any trace of the placeholder survives."""
    table = metrics_table_md(metrics)
    filled, n = _PLACEHOLDER_RE.subn(lambda _: table, text)
    if n == 0:
        filled = text.rstrip() + "\n\n## Metrics (generated from metrics.json)\n\n" + table + "\n"
    if "METRICS_TABLE" in filled:
        raise RuntimeError("metrics table placeholder left unfilled in a crew2 document")
    return filled
