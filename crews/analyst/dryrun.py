"""Crew 1 (analyst) dry-run: deterministic tools only, no LLM / CrewAI.

Runs the same pandas pipeline the agents invoke via tools, writes all crew1
artifacts, and emits minimal insights.md and eda_report.html from tool output.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

from crews.analyst import tools as T

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "telco.csv"
INSIGHTS_FILENAME = "insights.md"
EDA_FILENAME = "eda_report.html"


def _render_insights(clean_summary: dict, contract: dict, eda: dict) -> str:
    lines = [
        "# Telco churn insights (dry-run)",
        "",
        "Generated without an LLM; numbers come from `crews/analyst/tools.py`.",
        "",
        "## Cleaning",
        f"- Rows in: {clean_summary['rows_in']}",
        f"- Rows dropped (blank TotalCharges): {clean_summary['rows_dropped']}",
        f"- Rows out: {clean_summary['rows_out']}",
        "",
        "## Contract",
        f"- Target: `{contract['target_column']}`",
        f"- Expected row count: {contract['row_count']['expected']}",
        "",
        "## EDA",
        f"- Overall churn rate: {eda['churn_rate_pct']}%",
        f"- Churned customers: {eda['churn_count']} / {eda['rows']}",
        "",
        "## Caveats",
        "- Dry-run narrative only; use the real crew for full EDA write-ups.",
        "",
    ]
    return "\n".join(lines)


def _render_eda_html(clean_summary: dict, eda: dict) -> str:
    cat_rows = []
    for col, by_val in eda["churn_rate_by_category_pct"].items():
        for val, stats in by_val.items():
            cat_rows.append(
                f"<tr><td>{escape(col)}</td><td>{escape(val)}</td>"
                f"<td>{stats['customers']}</td><td>{stats['churn_rate_pct']}%</td></tr>"
            )
    num_rows = []
    for col, stats in eda["numeric"].items():
        num_rows.append(
            f"<tr><td>{escape(col)}</td><td>{stats['mean']}</td>"
            f"<td>{stats['mean_if_churned']}</td><td>{stats['mean_if_retained']}</td></tr>"
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Telco EDA (dry-run)</title>
<style>
body {{ font-family: sans-serif; margin: 1.5rem; }}
table {{ border-collapse: collapse; margin: 1rem 0; }}
th, td {{ border: 1px solid #ccc; padding: 0.35rem 0.6rem; }}
th {{ background: #f4f4f4; }}
</style>
</head>
<body>
<h1>Telco churn EDA (dry-run)</h1>
<p>Rows after cleaning: {eda['rows']}. Churn rate: {eda['churn_rate_pct']}%.</p>
<p>Cleaning: {clean_summary['rows_dropped']} rows dropped (blank TotalCharges).</p>
<h2>Churn rate by category</h2>
<table>
<thead><tr><th>Column</th><th>Value</th><th>Customers</th><th>Churn rate</th></tr></thead>
<tbody>
{"".join(cat_rows)}
</tbody>
</table>
<h2>Numeric summaries</h2>
<table>
<thead><tr><th>Column</th><th>Mean</th><th>Mean (churned)</th><th>Mean (retained)</th></tr></thead>
<tbody>
{"".join(num_rows)}
</tbody>
</table>
</body>
</html>
"""


def run(out_dir: Path) -> dict[str, Path]:
    """Same return shape as crew.run() and stub.run() (plus narrative paths)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "data": out_dir / T.CLEAN_FILENAME,
        "contract": out_dir / T.CONTRACT_FILENAME,
        "insights": out_dir / INSIGHTS_FILENAME,
        "eda_report": out_dir / EDA_FILENAME,
    }
    for p in paths.values():
        p.unlink(missing_ok=True)

    T.profile(RAW_PATH)
    clean_summary = T.clean(RAW_PATH, paths["data"])
    contract = T.write_contract(paths["data"], paths["contract"])
    eda = T.eda_summary(paths["data"])

    paths["insights"].write_text(
        _render_insights(clean_summary, contract, eda), encoding="utf-8"
    )
    paths["eda_report"].write_text(
        _render_eda_html(clean_summary, eda), encoding="utf-8"
    )

    on_disk = json.loads(paths["contract"].read_text(encoding="utf-8"))
    if T.build_contract(paths["data"]) != on_disk:
        raise RuntimeError("dataset_contract.json differs from the contract measured from clean.csv")

    return paths
