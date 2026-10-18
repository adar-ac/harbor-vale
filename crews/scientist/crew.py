"""Crew 2 (scientist): a real CrewAI crew over Crew 1's handoff.

Reads ONLY artifacts/crew1/clean.csv and dataset_contract.json (never data/raw/).
Three agents (Feature Engineer, Model Trainer, Model Card Author). Encoding, training,
model selection and every metric run inside deterministic tools
(crews/scientist/tools.py); the agents call those tools and write only the narrative
outputs (evaluation_report.md, model_card.md). The metrics table in both documents is
rendered by code, not typed by an agent. The tools are bound to fixed paths and take no
arguments, so an agent can't point them elsewhere.

dryrun.py and stub.py are offline fallbacks (see flows/main_flow.py).
"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # OPENAI_API_KEY, CREWAI_TRACING_ENABLED, ... before crewai reads them

from crewai import LLM, Agent, Crew, Process, Task  # noqa: E402
from crewai.llms.retry import run_with_rate_limit_retry  # noqa: E402
from crewai.tools import tool  # noqa: E402

from crews.analyst.crew import MODEL, _json, _strip_fences  # noqa: E402
from crews.scientist import tools as T  # noqa: E402

EVAL_FILENAME = "evaluation_report.md"
CARD_FILENAME = "model_card.md"


def build_crew(data_path: Path, contract_path: Path, out_dir: Path) -> tuple[Crew, dict[str, Task]]:
    features_path = out_dir / T.FEATURES_FILENAME
    model_path = out_dir / T.MODEL_FILENAME
    metrics_path = out_dir / T.METRICS_FILENAME

    @tool("engineer_features")
    def engineer_features() -> str:
        """Derive the feature list from dataset_contract.json, drop identifier columns,
        one-hot encode categoricals, encode the target as 0/1 and write features.csv.
        Returns a JSON summary. Takes no arguments."""
        return _json(T.engineer_features(data_path, contract_path, features_path))

    @tool("train_and_compare_models")
    def train_and_compare_models() -> str:
        """Train logistic regression and random forest on features.csv (stratified split,
        fixed random_state), compare them, save the better one as model.pkl and write
        metrics.json. Returns all metrics. Takes no arguments. Run engineer_features first."""
        return _json(T.train_and_compare(features_path, model_path, metrics_path))

    @tool("get_metrics")
    def get_metrics() -> str:
        """Return metrics.json: setup, per-model hold-out precision/recall/F1/ROC-AUC/
        accuracy/confusion matrix, CV ROC-AUC, top features, selected model. No arguments."""
        return metrics_path.read_text(encoding="utf-8")

    @tool("training_data_summary")
    def training_data_summary() -> str:
        """Summarise the training data: source files, contract version, rows, class
        counts, churn rate, dropped columns, features. Takes no arguments."""
        return _json(T.training_data_summary(data_path, contract_path, features_path))

    llm = LLM(model=MODEL, temperature=0.2)
    grounding = (
        "Every number you state must be copied from a tool result. Never estimate, "
        "compute or invent metrics or statistics yourself."
    )
    table_rule = (
        f"Do not type a metrics table. Put the line {T.METRICS_TABLE_PLACEHOLDER} on its "
        "own where the metrics table belongs; code replaces it with the table from metrics.json."
    )
    selection_rule = (
        "When you state why the model was selected, quote both models' mean CV ROC-AUC and "
        "the margin from metrics.json selection.margin (do not compute it yourself). If "
        "selection.near_tie is true, call it a near tie: say the two models are effectively "
        "equal on the selection metric and that the choice could reasonably be made on other "
        "grounds, such as interpretability (logistic regression coefficients are easier to "
        "explain than a random forest). Do not describe a near tie as a clear win."
    )

    engineer = Agent(
        role="Feature Engineer",
        goal="Turn the contracted clean dataset into a model-ready features.csv.",
        backstory=(
            "You build features strictly from the dataset contract published by the analyst "
            "crew, using the approved encoding tool. You never read the raw data. " + grounding
        ),
        tools=[engineer_features],
        llm=llm,
        verbose=True,
    )
    trainer = Agent(
        role="Model Trainer",
        goal="Train and compare two churn models and report how they differ on imbalanced data.",
        backstory=(
            "An ML engineer who knows churn is imbalanced, so accuracy alone misleads. You "
            "judge models on precision, recall, F1 and ROC-AUC, as measured by the training "
            "tool. " + grounding
        ),
        tools=[train_and_compare_models, get_metrics],
        llm=llm,
        verbose=True,
    )
    author = Agent(
        role="Model Card Author",
        goal="Document the selected model honestly in a model card.",
        backstory=(
            "You write model cards for stakeholders: purpose, data, metrics, limitations and "
            "ethical considerations, with no overstated claims. " + grounding
        ),
        tools=[get_metrics, training_data_summary],
        llm=llm,
        verbose=True,
    )

    features_task = Task(
        description=(
            "Call engineer_features exactly once. Report the rows written, the dropped "
            "identifier columns, the numeric and categorical features used, the number of "
            "model features after encoding, and the target encoding."
        ),
        expected_output="A short feature-engineering log quoting the tool's result.",
        agent=engineer,
    )
    train_task = Task(
        description=(
            "Call train_and_compare_models exactly once. Report which model was selected, "
            "the selection rule, and each model's hold-out precision, recall, F1 and ROC-AUC."
        ),
        expected_output="A short training log quoting the tool's result.",
        agent=trainer,
        context=[features_task],
    )
    eval_task = Task(
        description=(
            "Call get_metrics. Write evaluation_report.md comparing logistic regression and "
            "random forest: the setup (split, random_state, CV, threshold, class weighting), "
            "the metrics table, a discussion of precision vs recall and what the confusion "
            "matrices mean for churn outreach, why accuracy alone is misleading at this churn "
            "rate, how large the gap between the models is, the most important features of "
            "each, and which model was selected and why. " + selection_rule + " " + table_rule
        ),
        expected_output="A Markdown document with headings. Output only the Markdown.",
        agent=trainer,
        context=[features_task, train_task],
    )
    card_task = Task(
        description=(
            "Call training_data_summary and get_metrics. Write model_card.md for the selected "
            "model with these sections: Purpose, Model details, Training data summary, "
            "Metrics, Limitations, Ethical considerations. Limitations should cover the "
            "static snapshot, the fixed 0.5 threshold, the tenure-0 customers removed during "
            "cleaning, and correlated features. Ethical considerations should cover the "
            "sensitive attributes among the features (e.g. gender, SeniorCitizen) and how "
            "scores should and should not be used. Model details must say why this model was "
            "selected. " + selection_rule + " " + table_rule
        ),
        expected_output="A Markdown document with headings. Output only the Markdown.",
        agent=author,
        context=[features_task, train_task],
    )

    tasks = {"features": features_task, "train": train_task, "eval": eval_task, "card": card_task}
    crew = Crew(
        agents=[engineer, trainer, author],
        tasks=list(tasks.values()),
        process=Process.sequential,
        verbose=True,
    )
    return crew, tasks


def run(data_path: Path, contract_path: Path, out_dir: Path) -> dict[str, Path]:
    """Run the scientist crew; write features.csv, model.pkl, metrics.json,
    evaluation_report.md and model_card.md into out_dir. Same return shape as dryrun.run()."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "features": out_dir / T.FEATURES_FILENAME,
        "model": out_dir / T.MODEL_FILENAME,
        "metrics": out_dir / T.METRICS_FILENAME,
        "evaluation_report": out_dir / EVAL_FILENAME,
        "model_card": out_dir / CARD_FILENAME,
    }
    # Remove stale outputs so a run where an agent skipped its tool can't pass on old files.
    for p in paths.values():
        p.unlink(missing_ok=True)

    crew, tasks = build_crew(Path(data_path), Path(contract_path), out_dir)
    run_with_rate_limit_retry(crew.kickoff)

    for key in ("features", "model", "metrics"):
        if not paths[key].exists():
            raise RuntimeError(f"scientist crew finished without writing {paths[key].name}; "
                               "an agent did not call its tool")
    # Belt and braces: the saved model must reproduce the recorded metrics.
    T.verify_saved_model(paths["features"], paths["model"], paths["metrics"])

    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    for key, task in (("evaluation_report", "eval"), ("model_card", "card")):
        text = T.fill_metrics_table(_strip_fences(tasks[task].output.raw), metrics)
        paths[key].write_text(text, encoding="utf-8")
    return paths
