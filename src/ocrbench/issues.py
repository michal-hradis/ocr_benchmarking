"""Validation findings shared by the dataset manifest and PAGE XML reader."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Severity(StrEnum):
    ERROR = "error"  # the sample cannot be used
    WARNING = "warning"  # usable, but the reference may not be what the annotator intended
    INFO = "info"  # noteworthy, never a problem on its own


@dataclass(frozen=True)
class Issue:
    severity: Severity
    code: str
    message: str
    sample_id: str | None = None
    location: str | None = None  # file, and XML element id or line where known

    def format(self) -> str:
        where = " ".join(part for part in (self.sample_id, self.location) if part)
        prefix = f"{self.severity.value}: [{self.code}]"
        return f"{prefix} {where}: {self.message}" if where else f"{prefix} {self.message}"
