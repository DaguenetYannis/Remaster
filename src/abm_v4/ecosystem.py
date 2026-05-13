from __future__ import annotations

from dataclasses import dataclass

from src.abm_v4.schemas import VALID_ECOSYSTEM_SOURCES


@dataclass(frozen=True)
class EcosystemAssignment:
    """One explicit sector-to-ecosystem assignment."""

    sector: str
    ecosystem_id: str
    ecosystem_label: str
    ecosystem_source: str

    def is_known_source(self) -> bool:
        """Return whether the source follows the ABM v4 source vocabulary."""
        return self.ecosystem_source in VALID_ECOSYSTEM_SOURCES
