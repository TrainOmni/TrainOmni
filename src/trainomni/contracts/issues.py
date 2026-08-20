"""Framework-wide structured validation issues."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class IssueSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    message: str
    severity: IssueSeverity = IssueSeverity.ERROR
    path: str | None = None
    source: str | None = None
    hint: str | None = None

    def to_dict(self) -> dict[str, str]:
        value = {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
        }
        for name in ("path", "source", "hint"):
            item = getattr(self, name)
            if item is not None:
                value[name] = item
        return value


@dataclass(frozen=True, slots=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def valid(self) -> bool:
        return not any(issue.severity is IssueSeverity.ERROR for issue in self.issues)

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(
            issue for issue in self.issues if issue.severity is IssueSeverity.ERROR
        )

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(
            issue for issue in self.issues if issue.severity is IssueSeverity.WARNING
        )

    @classmethod
    def combine(cls, reports: Iterable[ValidationReport]) -> ValidationReport:
        return cls(tuple(issue for report in reports for issue in report.issues))

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "issues": [issue.to_dict() for issue in self.issues],
        }
