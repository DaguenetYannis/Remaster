from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.abm_v3.data_contracts import DataContractValidator, ValidationResult


@dataclass(frozen=True)
class ABMStateMetadata:
    """Metadata describing one ABM v3 state snapshot."""

    year: int
    scenario: str = "baseline_continuation"
    step: int = 0
    notes: dict[str, object] = field(default_factory=dict)


@dataclass
class ABMState:
    """Container for node and optional edge state.

    Behavioural update logic deliberately lives in ``dynamics/`` so state stays
    inspectable and easy to serialize.
    """

    nodes: pd.DataFrame
    metadata: ABMStateMetadata
    edges: pd.DataFrame | None = None

    def copy(self) -> "ABMState":
        return ABMState(
            nodes=self.nodes.copy(deep=True),
            metadata=ABMStateMetadata(
                year=self.metadata.year,
                scenario=self.metadata.scenario,
                step=self.metadata.step,
                notes=dict(self.metadata.notes),
            ),
            edges=None if self.edges is None else self.edges.copy(deep=True),
        )

    def validate_basic(self) -> list[ValidationResult]:
        validator = DataContractValidator()
        return [
            validator.validate_country_sector_key(self.nodes),
            validator.validate_no_duplicate_nodes(self.nodes),
        ]

    def for_year(self, year: int) -> "ABMState":
        if "Year" not in self.nodes.columns:
            return self.copy()
        year_nodes = self.nodes[self.nodes["Year"] == year].copy()
        return ABMState(
            nodes=year_nodes,
            metadata=ABMStateMetadata(
                year=year,
                scenario=self.metadata.scenario,
                step=self.metadata.step,
                notes=dict(self.metadata.notes),
            ),
            edges=None if self.edges is None else self.edges.copy(deep=True),
        )
