"""Analyst -> contract gate -> scientist, as a CrewAI Flow.

Written against crewai==1.15.23 (verified in that release's source):
  - import path:  from crewai.flow.flow import Flow, listen, start
  - @start() takes parentheses; @listen(<method or method name>)
  - a listener with a parameter receives the triggering method's return value
  - an exception raised inside a step propagates out of kickoff() unchanged

Run from the project root:  python -m flows.main_flow
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from crewai.flow.flow import Flow, listen, start  # noqa: E402

# ANALYST_DRYRUN=1: both crews' real tools + artifacts, no LLM.
# ANALYST_STUB=1: tiny hardcoded sample and a placeholder model report.
if os.getenv("ANALYST_DRYRUN") == "1":
    from crews.analyst import dryrun as analyst
    from crews.scientist import dryrun as scientist
elif os.getenv("ANALYST_STUB") == "1":
    from crews.analyst import stub as analyst
    from crews.scientist import stub as scientist
else:
    from crews.analyst import crew as analyst
    from crews.scientist import crew as scientist
from validation.contract_check import validate
from validation.exceptions import ContractViolation

ROOT = Path(__file__).resolve().parents[1]
CREW1_DIR = ROOT / "artifacts" / "crew1"
CREW2_DIR = ROOT / "artifacts" / "crew2"

log = logging.getLogger("pipeline")


class ChurnPipelineFlow(Flow):
    @start()
    def run_analyst(self) -> dict[str, Path]:
        if os.getenv("REUSE_CREW1") == "1":
            # Skip Crew 1 and gate the files already on disk (e.g. after a hand edit).
            artifacts = {"data": CREW1_DIR / "clean.csv", "contract": CREW1_DIR / "dataset_contract.json"}
            missing = [p.name for p in artifacts.values() if not p.exists()]
            if missing:
                raise FileNotFoundError(f"REUSE_CREW1=1 but {missing} not in {CREW1_DIR}; run the flow once first")
            log.info("step 1/3: REUSE_CREW1=1, skipping analyst; using existing files in %s", CREW1_DIR)
            return artifacts
        log.info("step 1/3: analyst (%s) -> %s", analyst.__name__, CREW1_DIR)
        artifacts = analyst.run(CREW1_DIR)
        log.info("analyst wrote %s and %s", artifacts["data"].name, artifacts["contract"].name)
        return artifacts

    @listen(run_analyst)
    def contract_gate(self, artifacts: dict[str, Path]) -> dict[str, Path]:
        log.info("step 2/3: validating %s against %s", artifacts["data"].name, artifacts["contract"].name)
        violations = validate(artifacts["data"], artifacts["contract"])
        if violations:
            for v in violations:
                log.error("contract violation: %s", v)
            raise ContractViolation(violations)
        log.info("gate passed: data satisfies the contract")
        return artifacts

    @listen(contract_gate)
    def run_scientist(self, artifacts: dict[str, Path]) -> dict[str, Path]:
        log.info("step 3/3: scientist (%s) -> %s", scientist.__name__, CREW2_DIR)
        # Crew 2 gets only the handoff: clean.csv + dataset_contract.json.
        outputs = scientist.run(artifacts["data"], artifacts["contract"], CREW2_DIR)
        log.info("scientist wrote %s", ", ".join(p.name for p in outputs.values()))
        return outputs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    result = ChurnPipelineFlow().kickoff()
    log.info("pipeline finished: %s", result)


if __name__ == "__main__":
    main()
