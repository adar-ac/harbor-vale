# Telco churn: two crews with a contract gate

```
data/raw/telco.csv
      │
      ▼
 Crew 1: analyst ──► artifacts/crew1/clean.csv
                     artifacts/crew1/dataset_contract.json
      │
      ▼
 Contract gate (pandas, deterministic; no LLM)  ── fails ──► ContractViolation, Crew 2 never runs
      │ passes
      ▼
 Crew 2: scientist (reads ONLY clean.csv + contract) ──► artifacts/crew2/model_report.json
```

**Status: skeleton only.** Both crews are stub functions that write hardcoded
placeholder artifacts. There are no Agents, Tasks, or LLM calls anywhere.

## Layout

| Path | Purpose |
|---|---|
| `flows/main_flow.py` | CrewAI Flow: `@start` analyst → `@listen` gate → `@listen` scientist |
| `crews/analyst/stub.py` | Crew 1 placeholder: writes `clean.csv` + `dataset_contract.json` |
| `crews/scientist/stub.py` | Crew 2 placeholder: reads the handoff, writes `model_report.json` |
| `validation/contract_check.py` | `validate(data_path, contract_path)` returns all violations; `gate()` raises |
| `validation/exceptions.py` | `Violation`, `ContractViolation` |
| `tests/test_gate.py` | Clean data passes; `MonthlyCharges` × 100 (cents) is blocked |

## Setup (Windows, Python 3.12)

crewai 1.15.23 requires Python `>=3.10,<3.14`. It will not install on 3.14.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install "crewai==1.15.23" pandas pytest
copy .env.example .env   # optional for now; the stubs don't call any LLM
```

## Verify

Run from the project root:

```powershell
python -m pytest -v           # 2 tests: gate passes / gate blocks cents
python -m flows.main_flow     # end-to-end: writes artifacts/crew1/* and artifacts/crew2/model_report.json
```

Use `python -m ...`, not `pytest` or `python flows\main_flow.py`. The `-m` form
puts the project root on `sys.path`, which the imports rely on.

CrewAI prints emoji to the console. On a legacy Windows code page you'll see harmless
`[CrewAIEventsBus] ... 'charmap' codec can't encode` lines. Fix it with
`$env:PYTHONUTF8 = "1"` before running.

## `dataset_contract.json` schema

Crew 1 publishes this next to the cleaned CSV. It is the only thing Crew 2 may
assume about the data.

```jsonc
{
  "contract_version": "1.0",          // bump when the schema's meaning changes
  "dataset": "telco_customer_churn",
  "produced_by": "crew1_analyst_stub",
  "data_file": "clean.csv",           // relative to the contract's folder
  "target_column": "Churn",           // must be a key in "columns" and present in the data
  "row_count": {
    "expected": 7032,                 // rows after cleaning
    "tolerance_pct": 1.0              // allowed |actual - expected| / expected * 100
  },
  "columns": {                        // exact column set: missing OR extra columns fail
    "<name>": {
      "dtype": "int|float|string|bool",       // logical type (required)
      "kind": "numeric|categorical|identifier", // informational
      "nullable": false,                      // default false
      "min": 0, "max": 100,                   // numeric range, inclusive (optional)
      "allowed_values": ["A", "B"],           // closed set for categoricals (optional)
      "monetary": true,                       // money column: "unit" becomes required
      "unit": "USD"                           // e.g. "USD" vs "cents", "months"
    }
  }
}
```

Example (monetary column):

```json
"MonthlyCharges": {"dtype": "float", "kind": "numeric", "monetary": true,
                   "unit": "USD", "min": 18.25, "max": 118.75}
```

Notes:
- `dtype` is a logical type rather than a pandas dtype string. pandas 3 reads text as
  `str` and pandas 2 reads it as `object`, and the contract shouldn't depend on which one is installed.
- Every column with `"monetary": true` must declare a `unit`. The gate reports a
  contract error otherwise. When a monetary column goes out of range, the violation
  message names the column and its declared unit, because a value that is 100× too big is
  usually a USD/cents mix-up.
- The stub ships 6 rows with `tolerance_pct: 0`. The real Crew 1 should emit the full
  21-column schema and `expected: 7032`.

## Checks performed by `validate()`

All checks run, and every violation is returned together:

1. Contract self-consistency: the target is declared, and monetary columns have a `unit`.
2. Target column present in the data.
3. Missing columns and extra columns.
4. dtype mismatch.
5. Nulls in non-nullable columns.
6. Numeric values outside `[min, max]`, with the unit in the message.
7. Categorical values not in `allowed_values`.
8. Row count drift beyond `tolerance_pct`.
