from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from src.abm_v3.leontief.coefficients import LeontiefYearData


@dataclass(frozen=True)
class BehaviouralScenarioContext:
    """Metadata for a single-year behavioural Leontief perturbation experiment."""

    year: int
    scenario_name: str
    mode: str
    input_panel_orientation: str
    shock_size: float
    selector_name: str | None
    notes: str


class BehaviouralScenarioShock(Protocol):
    """Protocol for exogenous ABM v3 behavioural Leontief scenario shocks."""

    name: str
    description: str
    selector_name: str
    shock_size: float
    shock_mode: str

    def apply(
        self,
        year_data: LeontiefYearData,
        capacity: pd.Series,
        input_panel: pd.DataFrame,
        context: BehaviouralScenarioContext,
    ) -> tuple[LeontiefYearData, pd.Series, pd.DataFrame]:
        """Return modified year data, modified capacity, and selected-node diagnostics."""
