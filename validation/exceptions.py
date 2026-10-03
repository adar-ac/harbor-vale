"""Exceptions raised by the contract gate."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Violation:
    """One broken rule. `column` is None for table-level checks (e.g. row count)."""

    check: str
    column: str | None
    message: str

    def __str__(self) -> str:
        return self.message


class ContractViolation(Exception):
    """Raised by the gate when the cleaned data does not satisfy the contract.

    Carries every violation found, not just the first.
    """

    def __init__(self, violations: list[Violation]) -> None:
        self.violations = list(violations)
        lines = "\n".join(f"  - [{v.check}] {v.message}" for v in self.violations)
        super().__init__(f"{len(self.violations)} contract violation(s):\n{lines}")
