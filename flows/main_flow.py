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
from pathlib import Path

from crewai.flow.flow import Flow, listen, start

from crews.analyst import stub as analyst
from crews.scientist import stub as scientist
from validation.contract_check import validate
from validation.exceptions import ContractViolation

ROOT = Path(__file__).resolve().parents[1]
CREW1_DIR = ROOT / "artifacts" / "crew1"
CREW2_DIR = ROOT / "artifacts" / "crew2"

log = logging.getLogger("pipeline")


class ChurnPipelineFlow(Flow):
    @start()
    def run_analyst(self) -> dict[str, Path]:
        log.info("step 1/3: analyst stub -> %s", CREW1_DIR)
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
    def run_scientist(self, artifacts: dict[str, Path]) -> Path:
        log.info("step 3/3: scientist stub -> %s", CREW2_DIR)
        report = scientist.run(artifacts["data"], artifacts["contract"], CREW2_DIR)
        log.info("scientist wrote %s", report.name)
        return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    result = ChurnPipelineFlow().kickoff()
    log.info("pipeline finished: %s", result)


if __name__ == "__main__":
    main()
