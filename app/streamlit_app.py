"""Read-only dashboard over the pipeline's artifacts.

Never runs the crews or retrains anything. The one live computation is the
contract gate: validate() against whatever clean.csv and contract are on disk.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # streamlit puts app/ on the path, not the repo root

from validation.contract_check import validate  # noqa: E402

CREW1_DIR = ROOT / "artifacts" / "crew1"
CREW2_DIR = ROOT / "artifacts" / "crew2"
CONTRACT = CREW1_DIR / "dataset_contract.json"
CLEAN = CREW1_DIR / "clean.csv"
INSIGHTS = CREW1_DIR / "insights.md"
METRICS = CREW2_DIR / "metrics.json"
EVAL_REPORT = CREW2_DIR / "evaluation_report.md"
MODEL_CARD = CREW2_DIR / "model_card.md"
MODEL = CREW2_DIR / "model.pkl"

ARTIFACTS = {
    "Crew 1 · dataset contract": CONTRACT,
    "Crew 1 · clean data": CLEAN,
    "Crew 1 · insights": INSIGHTS,
    "Crew 2 · metrics": METRICS,
    "Crew 2 · model": MODEL,
    "Crew 2 · evaluation report": EVAL_REPORT,
    "Crew 2 · model card": MODEL_CARD,
}

RUN_HINT = "Run the pipeline first: `python -m flows.main_flow` (or with `$env:ANALYST_DRYRUN = \"1\"` for an LLM-free run)."

MODEL_LABELS = {"logistic_regression": "Logistic regression", "random_forest": "Random forest"}


def missing(path: Path) -> bool:
    if path.exists():
        return False
    st.warning(f"`{path.relative_to(ROOT)}` not found. {RUN_HINT}")
    return True


def mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime)


@st.cache_data
def load_json(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data
def load_csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def read_json(path: Path) -> dict | None:
    try:
        return load_json(str(path), path.stat().st_mtime)
    except (OSError, ValueError) as e:
        st.error(f"Could not read `{path.relative_to(ROOT)}`: {e}")
        return None


def read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return load_csv(str(path), path.stat().st_mtime)
    except (OSError, ValueError) as e:
        st.error(f"Could not read `{path.relative_to(ROOT)}`: {e}")
        return None


st.set_page_config(page_title="Harbor Vale · Churn pipeline", layout="wide")
st.title("Harbor Vale · Churn pipeline")
st.caption(
    "Crew 1 (analyst) cleans the data and writes a contract. A deterministic gate checks the data "
    "against that contract before Crew 2 (scientist) is allowed to train. This page only reads the results."
)

# --- 1. Pipeline status -------------------------------------------------------
st.header("1 · Pipeline status")
present = {name: p for name, p in ARTIFACTS.items() if p.exists()}
if len(present) == len(ARTIFACTS):
    last = max(mtime(p) for p in present.values())
    st.success(f"Full run on disk: all {len(ARTIFACTS)} artifacts present. Last written {last:%Y-%m-%d %H:%M}.")
elif present:
    st.warning(f"Partial run: {len(present)} of {len(ARTIFACTS)} artifacts present. {RUN_HINT}")
else:
    st.error(f"No run yet: none of the artifacts exist. {RUN_HINT}")

st.dataframe(
    pd.DataFrame(
        [
            {
                "Artifact": name,
                "Present": "✅" if p.exists() else "❌",
                "File": str(p.relative_to(ROOT)),
                "Last modified": f"{mtime(p):%Y-%m-%d %H:%M:%S}" if p.exists() else "—",
            }
            for name, p in ARTIFACTS.items()
        ]
    ),
    hide_index=True,
    width="stretch",
)

# --- 2. The contract ------------------------------------------------------------
st.header("2 · The dataset contract")
st.markdown(
    "The contract is Crew 1's promise about `clean.csv`: every column, its type, and what values are legal. "
    "Crew 2 trusts nothing that isn't written here, and the gate (section 6) enforces it."
)
contract = None if missing(CONTRACT) else read_json(CONTRACT)
if contract:
    target = contract.get("target_column")
    rc = contract.get("row_count", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset", contract.get("dataset", "—"))
    c2.metric("Columns", len(contract.get("columns", {})))
    c3.metric("Expected rows", f"{rc.get('expected', 0):,}", help=f"Tolerance {rc.get('tolerance_pct', 0)}%")
    c4.metric("Target", target or "—")

    def constraint(spec: dict) -> str:
        if "allowed_values" in spec:
            return ", ".join(map(str, spec["allowed_values"]))
        if "min" in spec or "max" in spec:
            return f"{spec.get('min', '−∞')} – {spec.get('max', '∞')}"
        return "—"

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Column": name + ("  🎯 target" if name == target else ""),
                    "Dtype": spec.get("dtype", "—"),
                    "Kind": spec.get("kind", "—"),
                    "Range / allowed values": constraint(spec),
                    "Unit": spec.get("unit", "—"),
                    "Nullable": "yes" if spec.get("nullable") else "no",
                }
                for name, spec in contract.get("columns", {}).items()
            ]
        ),
        hide_index=True,
        width="stretch",
        height=35 * (len(contract.get("columns", {})) + 1) + 3,
    )
    st.caption(
        f"Contract v{contract.get('contract_version', '?')} · produced by {contract.get('produced_by', '?')} · "
        f"describes `{contract.get('data_file', '?')}`"
    )
    with st.expander("Raw dataset_contract.json"):
        st.json(contract)

# --- 3. Data --------------------------------------------------------------------
st.header("3 · Data")
df = None if missing(CLEAN) else read_csv(CLEAN)
if df is not None:
    needed = {"Churn", "Contract", "tenure", "MonthlyCharges"}
    if not needed <= set(df.columns):
        st.error(f"`clean.csv` is missing columns needed for the charts: {sorted(needed - set(df.columns))}. See section 6.")
    else:
        churned = df["Churn"].eq("Yes")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Churned", f"{churned.sum():,}", f"{churned.mean():.1%} of customers", delta_color="off")
        c3.metric("Retained", f"{(~churned).sum():,}", f"{(~churned).mean():.1%} of customers", delta_color="off")

        def churn_rate(by: pd.Series) -> pd.DataFrame:
            return churned.groupby(by, observed=True).mean().mul(100).rename("Churn rate (%)").to_frame()

        tenure_bins = pd.cut(df["tenure"], [0, 12, 24, 48, 72], labels=["0–12 mo", "13–24 mo", "25–48 mo", "49–72 mo"])
        charge_bins = pd.cut(df["MonthlyCharges"], [0, 35, 70, 90, 200], labels=["< $35", "$35–70", "$70–90", "> $90"])

        st.markdown("**Churn rate (%)** by segment")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("By contract type")
            st.bar_chart(churn_rate(df["Contract"]), y_label="Churn rate (%)")
        with c2:
            st.caption("By tenure")
            st.bar_chart(churn_rate(tenure_bins), y_label="Churn rate (%)")
        with c3:
            st.caption("By monthly charges")
            st.bar_chart(churn_rate(charge_bins), y_label="Churn rate (%)")

        with st.expander("First 50 rows of clean.csv"):
            st.dataframe(df.head(50), width="stretch")

# --- 4. Models --------------------------------------------------------------------
st.header("4 · Models")
metrics = None if missing(METRICS) else read_json(METRICS)
if metrics:
    models: dict = metrics.get("models", {})
    selected = metrics.get("selected_model")
    sel = metrics.get("selection", {})
    setup = metrics.get("setup", {})

    if selected:
        st.success(f"Selected model: **{MODEL_LABELS.get(selected, selected)}** (by `{setup.get('selection_metric', '?')}`)")
    if sel.get("near_tie"):
        st.warning(
            f"**Near tie.** {MODEL_LABELS.get(selected, selected)} beat "
            f"{MODEL_LABELS.get(sel.get('runner_up'), sel.get('runner_up'))} by only "
            f"{sel.get('margin')} CV ROC-AUC (near-tie threshold {sel.get('near_tie_threshold')}). "
            "The choice is within noise; either model is a defensible pick."
        )

    rows = {
        "CV ROC-AUC (mean ± std)": lambda m: f"{m['cv_roc_auc_mean']:.4f} ± {m['cv_roc_auc_std']:.4f}",
        "Test ROC-AUC": lambda m: f"{m['test']['roc_auc']:.4f}",
        "Test precision": lambda m: f"{m['test']['precision']:.4f}",
        "Test recall": lambda m: f"{m['test']['recall']:.4f}",
        "Test F1": lambda m: f"{m['test']['f1']:.4f}",
        "Test accuracy": lambda m: f"{m['test']['accuracy']:.4f}",
    }
    try:
        table = pd.DataFrame(
            {
                MODEL_LABELS.get(k, k) + (" ✓" if k == selected else ""): {label: f(m) for label, f in rows.items()}
                for k, m in models.items()
            }
        )
        st.dataframe(table, width="stretch")
    except KeyError as e:
        st.error(f"`metrics.json` is missing an expected field: {e}")

    data = metrics.get("data", {})
    st.caption(
        f"{setup.get('split', '?')} · {data.get('rows_train', '?')} train / {data.get('rows_test', '?')} test rows · "
        f"{data.get('n_features', '?')} features · {setup.get('cv_folds', '?')}-fold CV · "
        f"threshold {setup.get('decision_threshold', '?')} · positive class: {setup.get('positive_class', '?')}"
    )

    st.subheader("Confusion matrices (hold-out set)")
    for col, (k, m) in zip(st.columns(len(models) or 1), models.items()):
        cm = m.get("test", {}).get("confusion_matrix")
        with col:
            st.markdown(f"**{MODEL_LABELS.get(k, k)}**" + (" ✓ selected" if k == selected else ""))
            if cm:
                st.table(
                    pd.DataFrame(
                        [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
                        index=["Actual: stayed", "Actual: churned"],
                        columns=["Predicted: stay", "Predicted: churn"],
                    )
                )
            else:
                st.info("No confusion matrix recorded.")

    with st.expander("Top features per model"):
        for col, (k, m) in zip(st.columns(len(models) or 1), models.items()):
            with col:
                st.markdown(f"**{MODEL_LABELS.get(k, k)}**")
                st.dataframe(pd.DataFrame(m.get("top_features", [])), hide_index=True, width="stretch")

# --- 5. Documents -------------------------------------------------------------------
st.header("5 · Documents")
for tab, path in zip(st.tabs(["Insights (Crew 1)", "Evaluation report (Crew 2)", "Model card (Crew 2)"]), [INSIGHTS, EVAL_REPORT, MODEL_CARD]):
    with tab:
        if not missing(path):
            st.caption(f"`{path.relative_to(ROOT)}` · {mtime(path):%Y-%m-%d %H:%M}")
            st.markdown(path.read_text(encoding="utf-8"))

# --- 6. Gate status -----------------------------------------------------------------
st.header("6 · Contract gate (live)")
st.markdown(
    "`validate()` runs **now**, against the `clean.csv` and `dataset_contract.json` currently on disk. "
    "If this fails, the pipeline would stop here and Crew 2 would never train."
)
if not (missing(CLEAN) or missing(CONTRACT)):
    if st.button("Re-run gate"):
        pass  # any interaction reruns the script, which re-validates below
    try:
        violations = validate(CLEAN, CONTRACT)
    except Exception as e:  # malformed contract or unreadable CSV: still a gate failure
        st.error(f"## ⛔ GATE ERROR\n\nThe validator could not run: `{type(e).__name__}: {e}`")
    else:
        if not violations:
            st.success(f"## ✅ GATE PASSED\n\n`clean.csv` satisfies every rule in the contract. Checked at {datetime.now():%H:%M:%S}.")
        else:
            st.error(
                f"## ⛔ GATE FAILED: {len(violations)} violation(s)\n\n"
                "The data does not match its contract. The pipeline would halt before Crew 2 trains."
            )
            st.dataframe(
                pd.DataFrame([{"Check": v.check, "Column": v.column or "(table)", "Problem": v.message} for v in violations]),
                hide_index=True,
                width="stretch",
            )
