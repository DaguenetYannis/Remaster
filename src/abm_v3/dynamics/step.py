from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.abm_v3.dynamics.demand import get_historical_demand, project_demand
from src.abm_v3.dynamics.emissions import compute_emissions, update_emissions_intensity
from src.abm_v3.dynamics.inputs import compute_input_availability
from src.abm_v3.dynamics.production import plan_production, realize_production
from src.abm_v3.state import ABMState, ABMStateMetadata


@dataclass
class ABMV3StepEngine:
    """Coordinate one quantity-based ABM v3 step."""

    production_model: object | None = None
    emissions_model: object | None = None

    def step(self, current_state: ABMState, next_year: int, scenario: object | None = None) -> tuple[ABMState, dict[str, object]]:
        nodes = current_state.nodes.copy()
        demand = (
            get_historical_demand(nodes, next_year)
            if scenario is None or next_year <= 2016
            else project_demand(current_state, scenario)
        )
        planned = plan_production(nodes, self.production_model)
        input_availability = compute_input_availability(nodes)
        realized = realize_production(planned, demand.reindex(nodes.index), input_availability)
        next_ei = update_emissions_intensity(nodes, self.emissions_model)
        emissions = compute_emissions(realized, next_ei)
        next_nodes = nodes.copy()
        next_nodes["Year"] = next_year
        next_nodes["D"] = demand.reindex(nodes.index).to_numpy(dtype=float)
        next_nodes["planned_output"] = planned
        next_nodes["realized_output"] = realized
        next_nodes["X"] = realized
        next_nodes["EI"] = next_ei
        next_nodes["emissions"] = emissions
        next_state = ABMState(
            nodes=next_nodes,
            metadata=ABMStateMetadata(
                year=next_year,
                scenario=getattr(scenario, "name", current_state.metadata.scenario),
                step=current_state.metadata.step + 1,
            ),
            edges=current_state.edges,
        )
        diagnostics = {
            "year": next_year,
            "total_planned_output": float(planned.sum()),
            "total_realized_output": float(realized.sum()),
            "total_emissions": float(emissions.sum()),
        }
        return next_state, diagnostics
