from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.abm_v3.dynamics.demand import project_demand

LOGGER = logging.getLogger(__name__)


@dataclass
class DemandProvider:
    """Provide demand for ABM v3 historical and projected steps.

    During 1995-2016, demand is historical and exogenous: it must be read from
    a separate historical panel keyed by ``country_sector`` and ``Year``. For
    post-2016 simulations, demand is scenario-based and grows from the current
    node state through the existing projection rule.
    """

    historical_panel: pd.DataFrame | None = None
    node_col: str = "country_sector"
    year_col: str = "Year"
    demand_col: str = "D"

    def historical_demand(self, current_nodes: pd.DataFrame, year: int) -> pd.Series:
        """Return exogenous historical demand aligned to ``current_nodes``.

        Missing node-year matches are returned as NaN and logged explicitly.
        They are not interpreted as zero because missing historical demand is
        data uncertainty, not evidence of absent demand.
        """

        if self.historical_panel is None:
            raise ValueError("Historical demand requested but no historical_panel was provided.")
        required_current = [self.node_col]
        missing_current = [column for column in required_current if column not in current_nodes.columns]
        if missing_current:
            raise ValueError(f"Current nodes missing demand key columns: {missing_current}")
        required_panel = [self.node_col, self.year_col, self.demand_col]
        missing_panel = [column for column in required_panel if column not in self.historical_panel.columns]
        if missing_panel:
            raise ValueError(f"Historical panel missing demand columns: {missing_panel}")

        historical_year = self.historical_panel[self.historical_panel[self.year_col] == year]
        demand_map = historical_year.set_index(self.node_col)[self.demand_col]
        demand = current_nodes[self.node_col].map(demand_map)
        demand = pd.Series(demand.to_numpy(dtype=float), index=current_nodes.index, name=self.demand_col)

        missing_count = int(demand.isna().sum())
        if missing_count:
            LOGGER.warning(
                "Historical demand missing for %s current nodes in year %s; returning NaN for those nodes.",
                missing_count,
                year,
            )
        return demand

    def projected_demand(self, current_nodes: pd.DataFrame, scenario: object) -> pd.Series:
        """Return scenario-based projected demand for post-2016 simulation."""

        return project_demand(current_nodes, scenario)
