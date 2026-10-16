# Telco churn: two agent crews with a contract gate

Two CrewAI crews predict customer churn on the IBM Telco dataset. A data contract between them is checked by code before the second crew runs.

```
data/raw/telco.csv
      │
      ▼
 Crew 1: analyst ──► artifacts/crew1/clean.csv + dataset_contract.json (+ EDA reports)
      │
      ▼
 Contract gate (pandas, no LLM) ── fails ──► ContractViolation; Crew 2 never runs
      │ passes
      ▼
 Crew 2: scientist (reads ONLY clean.csv + contract) ──► artifacts/crew2/ (features, model, reports)
```

## What this is and why the contract matters

- **Crew 1 (analyst)** profiles and cleans the raw CSV. It also publishes `dataset_contract.json`, which states exactly what `clean.csv` contains: the columns, their types and ranges, the allowed category values, and the units (e.g. `MonthlyCharges` is USD).
- **The gate** (`validation/contract_check.py`) is plain pandas code. It checks `clean.csv` against the contract. If anything doesn't match, it raises `ContractViolation` and lists every problem, and the flow stops.
- **Crew 2 (scientist)** reads only `clean.csv` and the contract, never the raw data. It builds its feature list from the contract, trains logistic regression and random forest, keeps the better one, and writes an evaluation report and a model card.

The handoff failure this prevents is a silent schema or unit change. A dropped column, a renamed category, or charges switched from dollars to cents would all still load as a valid CSV, and Crew 2 would train a wrong model without complaint. With the gate in place, the pipeline refuses to train and says which column broke and how.

LLM agents only orchestrate and write the narrative. Cleaning, the contract, features, training and every metric are computed by deterministic functions in `crews/*/tools.py`, and the agents call those functions as tools.

## Setup (Windows / PowerShell, Python 3.12)

crewai 1.15.23 needs Python `>=3.10,<3.14`. It will not install on 3.14.

```powershell
git clone <repo-url> harbor-vale
cd harbor-vale
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY` (get one at https://platform.openai.com/api-keys). Both crews default to `openai/gpt-4o-mini`. To use a different model, set `MODEL`, e.g. `MODEL=gemini/gemini-3.6-flash` with `GEMINI_API_KEY`. The dry-run and stub modes below need no key.

**Get the data.** Download *Telco Customer Churn* from Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn (a free Kaggle account is required). Unzip it, then save `WA_Fn-UseC_-Telco-Customer-Churn.csv` as:

```
data/raw/telco.csv
```

The file should have 7043 rows and 21 columns. `data/raw/` is gitignored.

On macOS/Linux, use `python3.12 -m venv .venv`, `source .venv/bin/activate` and `cp .env.example .env`, and set env vars as `ANALYST_DRYRUN=1 python -m flows.main_flow`.

## How to run

Run everything from the project root, and always use `python -m ...`. The `-m` form puts the root on `sys.path`, which the imports need.

| Command | What runs | Needs |
|---|---|---|
| `python -m flows.main_flow` | Full pipeline: both real crews, with LLM calls | `OPENAI_API_KEY`, `telco.csv` |
| `$env:ANALYST_DRYRUN = "1"; python -m flows.main_flow` | Same tools and the same gate, no LLM. The reports are plain templates | `telco.csv` |
| `$env:ANALYST_STUB = "1"; python -m flows.main_flow` | 6 hardcoded rows → gate → placeholder `model_report.json`. Quick wiring check | nothing |
| `python -m pytest -v` | All tests | Tests that use real data skip if `telco.csv` is missing |

The env vars last for the PowerShell session. Clear them with `Remove-Item Env:ANALYST_DRYRUN`, and likewise for the others.

Every mode writes into `artifacts/`. `ANALYST_STUB=1` replaces `artifacts/crew1/clean.csv` with the 6-row sample. A dry run replaces the LLM-written `insights.md` and `eda_report.html` with templates. Run the full flow again to get the real outputs back.

CrewAI prints emoji. On a legacy Windows code page you'll see harmless `'charmap' codec can't encode` lines. Run `$env:PYTHONUTF8 = "1"` first to avoid them.

Training takes about 15 s; the rest of a full run is LLM calls.

## `dataset_contract.json`

Crew 1 generates it from `clean.csv`; nobody writes it by hand. It is the only thing Crew 2 may assume about the data.

| Field | Meaning |
|---|---|
| `contract_version` | Schema version. Bump it when field meanings change |
| `dataset` | Dataset name |
| `produced_by` | Which crew/mode wrote it (`crew1_analyst`, `crew1_analyst_stub`) |
| `data_file` | Data file the contract describes, relative to the contract's folder |
| `target_column` | Label column. Must be declared in `columns` and present in the data |
| `row_count.expected` / `row_count.tolerance_pct` | Expected rows, and the allowed drift in % |
| `columns` | The exact column set. Missing **or** extra columns both fail |

Per column:

| Field | Meaning |
|---|---|
| `dtype` | Logical type: `int`, `float`, `string`, `bool` (not a pandas dtype string, so pandas 2 vs 3 doesn't matter) |
| `kind` | `numeric`, `categorical` or `identifier`. Crew 2 uses this to build features: identifiers are dropped, categoricals one-hot encoded |
| `nullable` | Whether nulls are allowed (default `false`) |
| `min` / `max` | Inclusive numeric range |
| `allowed_values` | Closed set of values for a categorical |
| `monetary` | Money column. If `true`, `unit` is required |
| `unit` | e.g. `USD`, `months`. Named in range errors so unit mix-ups are obvious |

Example column:

```json
"MonthlyCharges": {"dtype": "float", "kind": "numeric", "min": 18.25, "max": 118.75,
                   "nullable": false, "monetary": true, "unit": "USD"}
```

The gate runs every check and reports all violations together:
1. The contract is consistent with itself: the target is declared, and monetary columns have a unit.
2. The target column is present.
3. No columns are missing and none are extra.
4. dtypes match.
5. Non-nullable columns have no nulls.
6. Values are within `[min, max]`.
7. Categoricals use only `allowed_values`.
8. Row-count drift is within tolerance.

## Break-it demo: a unit change stops the pipeline

Normally the flow regenerates `clean.csv` from `telco.csv`, which would overwrite a hand edit. `REUSE_CREW1=1` makes the flow skip Crew 1 and send the files already in `artifacts/crew1/` through the gate.

```powershell
# 1. Produce clean artifacts once (any non-stub mode works)
$env:ANALYST_DRYRUN = "1"; python -m flows.main_flow

# 2. Back up clean.csv, then convert MonthlyCharges from USD to cents
Copy-Item artifacts\crew1\clean.csv artifacts\crew1\clean.csv.bak
python -c "import pandas as pd; p = 'artifacts/crew1/clean.csv'; df = pd.read_csv(p); df['MonthlyCharges'] *= 100; df.to_csv(p, index=False)"

# 3. Rerun from the gate
$env:REUSE_CREW1 = "1"; python -m flows.main_flow
```

The flow stops at step 2, and Crew 2 never starts (there is no `step 3/3` line):

```
INFO pipeline: step 1/3: REUSE_CREW1=1, skipping analyst; using existing files in ...\artifacts\crew1
INFO pipeline: step 2/3: validating clean.csv against dataset_contract.json
ERROR pipeline: contract violation: MonthlyCharges: 7032 value(s) outside declared range [18.25, 118.75] USD (observed 1825.0..11875.0); contract unit is 'USD' - check for a unit mismatch (e.g. cents vs USD)
...
validation.exceptions.ContractViolation: 1 contract violation(s):
  - [range] MonthlyCharges: 7032 value(s) outside declared range [18.25, 118.75] USD (observed 1825.0..11875.0); contract unit is 'USD' - check for a unit mismatch (e.g. cents vs USD)
```

Restore:

```powershell
Move-Item -Force artifacts\crew1\clean.csv.bak artifacts\crew1\clean.csv
Remove-Item Env:REUSE_CREW1
```

You can try other breaks the same way: drop a column (`df = df.drop(columns='tenure')` gives `[missing_column]`) or rename a category (`[categorical]`). `tests/test_gate.py` and `tests/test_crew2_contract_violation.py` test the same behaviour automatically.

## Artifacts

`*.csv` and `*.pkl` under `artifacts/` are gitignored; generate them by running the flow.

| File | Written by | Contents |
|---|---|---|
| `crew1/clean.csv` | Crew 1 tool | Raw data with `TotalCharges` made numeric, and the 11 tenure-0 rows with blank `TotalCharges` dropped. 7032 rows, 21 columns |
| `crew1/dataset_contract.json` | Crew 1 tool | The contract above, measured from `clean.csv` |
| `crew1/eda_report.html` | Crew 1 agent | Self-contained EDA report: churn rate by category, numeric summaries |
| `crew1/insights.md` | Crew 1 agent | Churn drivers and caveats, written for the modelling crew |
| `crew2/features.csv` | Crew 2 tool | Numeric feature matrix: `customerID` dropped, categoricals one-hot encoded, `Churn` as 0/1 (last column) |
| `crew2/model.pkl` | Crew 2 tool | Selected sklearn model (`joblib.load`). Its feature names are in `feature_names_in_` |
| `crew2/metrics.json` | Crew 2 tool | Setup, plus for each model: hold-out precision/recall/F1/ROC-AUC/accuracy, confusion matrix, 5-fold CV ROC-AUC, top features. Also the selected model |
| `crew2/evaluation_report.md` | Crew 2 agent | Logistic regression vs random forest comparison |
| `crew2/model_card.md` | Crew 2 agent | Purpose, training data, metrics, limitations, ethical considerations |
| `crew2/model_report.json` | Crew 2 stub | Only written in `ANALYST_STUB=1` mode: placeholder, no model |

Training settings: stratified 80/20 split, `random_state=42`, `class_weight="balanced"` (churn is about 27%). The model with the higher cross-validated ROC-AUC on the training split is selected. The metrics tables in both Crew 2 reports are inserted by code from `metrics.json`; agents don't type them. After each real run, `model.pkl` is reloaded and must reproduce `metrics.json` exactly, or the run fails.

## Project layout

```
flows/main_flow.py             CrewAI Flow: analyst → contract gate → scientist; mode switches via env vars
crews/analyst/
  crew.py                      Crew 1: Profiler, Cleaner, Contract Author agents
  tools.py                     Profiling, cleaning, contract generation, EDA stats (pandas)
  dryrun.py                    Crew 1 without LLM
  stub.py                      Crew 1 with 6 hardcoded rows
crews/scientist/
  crew.py                      Crew 2: Feature Engineer, Model Trainer, Model Card Author agents
  tools.py                     Contract-driven features, training, metrics (pandas/sklearn)
  dryrun.py                    Crew 2 without LLM
  stub.py                      Placeholder Crew 2 for ANALYST_STUB=1
validation/
  contract_check.py            validate() returns all violations; gate() raises
  exceptions.py                Violation, ContractViolation
tests/
  test_gate.py                 Clean data passes; MonthlyCharges in cents is blocked
  test_crew2_contract_violation.py  Flow stops at the gate on a missing column; Crew 2 never runs
  test_analyst_tools.py        Real cleaning + contract on telco.csv passes the gate
  test_scientist_tools.py      Features follow the contract; model and metrics valid and reproducible
  test_metrics_table.py        Metrics table placeholder is always filled
data/raw/telco.csv             Kaggle input (gitignored, you provide it)
artifacts/crew1/, crew2/       Pipeline outputs
scratch_*.py, scratch_facts.json  One-off exploration used to design the cleaning; not used by the pipeline
.env.example                   Template for .env
```
