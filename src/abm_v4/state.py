from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.abm_v4.paths import ABMV4Paths


@dataclass(frozen=True)
class StateSourceDiagnostic:
    """Diagnostic for ABM v4 state-source discovery."""

    selected_source: Path | None
    checked_sources: tuple[Path, ...]
    message: str

    @property
    def has_source(self) -> bool:
        return self.selected_source is not None


def discover_state_source(
    paths: ABMV4Paths,
    start_year: int,
    end_year: int,
) -> StateSourceDiagnostic:
    """Find the first available state source using the ABM v4 priority order."""
    checked_sources = paths.state_source_candidates(start_year, end_year)
    for source_path in checked_sources:
        if source_path.exists():
            return StateSourceDiagnostic(
                selected_source=source_path,
                checked_sources=checked_sources,
                message=f"Selected state source: {source_path}",
            )

    return StateSourceDiagnostic(
        selected_source=None,
        checked_sources=checked_sources,
        message="No valid ABM v4 state source was found.",
    )
