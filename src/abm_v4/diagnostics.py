from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiagnosticMessage:
    """Structured diagnostic message for inspectable ABM v4 workflows."""

    step: str
    level: str
    message: str
