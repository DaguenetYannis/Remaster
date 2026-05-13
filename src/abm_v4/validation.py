from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationMessage:
    """Structured validation message that does not hide uncertainty."""

    check_name: str
    passed: bool
    message: str
