"""Crew 2 (scientist) dry-run: deterministic tools only, no LLM / CrewAI.

Runs the same sklearn pipeline the agents invoke via tools, writes all crew2
artifacts, and renders evaluation_report.md and model_card.md from tool output.
"""

from __future__ import annotations

from pathlib import Path

from crews.scientist import tools as T

EVAL_FILENAME = "evaluation_report.md"
CARD_FILENAME = "model_card.md"


def _selection_note(metrics: dict) -> str:
    sel, key = metrics["selection"], metrics["setup"]["selection_metric"]
    best, other = metrics["selected_model"], sel["runner_up"]
    note = (f"{key}: `{best}` {metrics['models'][best][key]} vs `{other}` "
            f"{metrics['models'][other][key]} (margin {sel['margin']})")
    if sel["near_tie"]:
        note += (f"; a near tie (under {sel['near_tie_threshold']}), so the choice could "
                 "reasonably be made on other grounds such as interpretability")
    return note


def _render_evaluation(metrics: dict) -> str:
    setup, data = metrics["setup"], metrics["data"]
    selected = metrics["selected_model"]
    lines = [
        "# Model evaluation: logistic regression vs random forest (dry-run)",
        "",
        "Generated without an LLM; numbers come from `crews/scientist/tools.py` (metrics.json).",
        "",
        "## Setup",
        f"- Split: {setup['split']}, random_state={setup['random_state']}",
        f"- Train rows: {data['rows_train']}, test rows: {data['rows_test']}, features: {data['n_features']}",
        f"- Churn rate: train {data['churn_rate_train']}, test {data['churn_rate_test']}",
        f"- Selection: highest {setup['selection_metric']} ({setup['cv_folds']}-fold CV on the training split)",
        f"- Decision threshold: {setup['decision_threshold']}; positive class: {setup['positive_class']}",
        "",
        "## Hold-out metrics",
        "",
        T.METRICS_TABLE_PLACEHOLDER,
        "",
        "## Selected model",
        f"- `{selected}`, saved as `{metrics['model_file']}`",
        f"- {_selection_note(metrics)}",
        "",
    ]
    return T.fill_metrics_table("\n".join(lines), metrics)


def _render_card(metrics: dict, summary: dict) -> str:
    selected = metrics["selected_model"]
    sel = metrics["models"][selected]
    top = ", ".join(f"`{f['feature']}`" for f in sel["top_features"][:5])
    lines = [
        "# Model card: Telco churn classifier (dry-run)",
        "",
        "Generated without an LLM; numbers come from `crews/scientist/tools.py`.",
        "",
        "## Purpose",
        "- Estimate the probability that a Telco customer churns, to prioritise retention outreach.",
        "",
        "## Model",
        f"- Selected: `{selected}` ({sel['params']['estimator']}), file `{metrics['model_file']}`",
        f"- Selection: {_selection_note(metrics)}",
        f"- Most influential features: {top}",
        "",
        "## Training data",
        f"- Source: {', '.join(summary['source_files'])} (contract v{summary['contract_version']}, "
        f"produced by {summary['produced_by']})",
        f"- Rows: {summary['rows']}; class counts: {summary['class_counts']}; "
        f"churn rate {summary['churn_rate_pct']}%",
        f"- Dropped identifier columns: {summary['dropped_identifier_columns']}",
        f"- Features after encoding: {summary['n_model_features_after_encoding']}",
        "",
        "## Metrics (hold-out)",
        "",
        T.METRICS_TABLE_PLACEHOLDER,
        "",
        "## Limitations",
        "- Single static snapshot; no time-based validation, so drift is untested.",
        "- Threshold fixed at 0.5 with balanced class weights; precision/recall trade-off not tuned.",
        "- Customers with tenure 0 were removed in cleaning, so new customers are out of scope.",
        "",
        "## Ethical considerations",
        "- Inputs include gender and SeniorCitizen; check outcomes per group before acting on scores.",
        "- Use scores to offer help, not to penalise customers or withhold service.",
        "",
    ]
    return T.fill_metrics_table("\n".join(lines), metrics)


def run(data_path: Path, contract_path: Path, out_dir: Path) -> dict[str, Path]:
    """Same return shape as crew.run()."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "features": out_dir / T.FEATURES_FILENAME,
        "model": out_dir / T.MODEL_FILENAME,
        "metrics": out_dir / T.METRICS_FILENAME,
        "evaluation_report": out_dir / EVAL_FILENAME,
        "model_card": out_dir / CARD_FILENAME,
    }
    for p in paths.values():
        p.unlink(missing_ok=True)

    T.engineer_features(data_path, contract_path, paths["features"])
    metrics = T.train_and_compare(paths["features"], paths["model"], paths["metrics"])
    T.verify_saved_model(paths["features"], paths["model"], paths["metrics"])
    summary = T.training_data_summary(data_path, contract_path, paths["features"])

    paths["evaluation_report"].write_text(_render_evaluation(metrics), encoding="utf-8")
    paths["model_card"].write_text(_render_card(metrics, summary), encoding="utf-8")
    return paths
