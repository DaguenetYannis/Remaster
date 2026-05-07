from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.abm_v3.dynamics.demand_provider import DemandProvider
from src.abm_v3.dynamics.emissions import compute_emissions, update_emissions_intensity
from src.abm_v3.dynamics.inputs import compute_input_availability
from src.abm_v3.dynamics.production import plan_production, realize_production
from src.abm_v3.dynamics.substitution import compute_substitution_adjusted_input_availability
from src.abm_v3.state import ABMState, ABMStateMetadata


@dataclass
class ABMV3StepEngine:
    """Coordinate one quantity-based ABM v3 step.

    Historical demand is exogenous through 2016 and must come from a separate
    demand provider. After 2016, demand is scenario-based. Supplier
    substitution is applied before production realization as a simplified
    node-level placeholder; the base step introduces no prices and no
    anti-collapse guardrail.
    """

    production_model: object | None = None
    emissions_model: object | None = None
    demand_provider: DemandProvider | None = None
    sigma: float = 0.25
    historical_end_year: int = 2016

    def step(self, current_state: ABMState, next_year: int, scenario: object | None = None) -> tuple[ABMState, dict[str, object]]:
        nodes = current_state.nodes.copy()
        demand_provider = self.demand_provider or DemandProvider()
        if next_year <= self.historical_end_year:
            demand = demand_provider.historical_demand(nodes, next_year)
        else:
            demand = demand_provider.projected_demand(nodes, scenario)
        planned = plan_production(nodes, self.production_model)
        input_availability = compute_input_availability(nodes)
        adjusted_input_availability = compute_substitution_adjusted_input_availability(
            nodes=nodes,
            input_availability=input_availability,
            demand=demand,
            planned_output=planned,
            sigma=self.sigma,
        )
        substitution_gain = adjusted_input_availability - input_availability.reindex(nodes.index)
        realized = realize_production(planned, demand.reindex(nodes.index), adjusted_input_availability)
        next_ei = update_emissions_intensity(nodes, self.emissions_model)
        emissions = compute_emissions(realized, next_ei)
        next_nodes = nodes.copy()
        next_nodes["Year"] = next_year
        next_nodes["D"] = demand.reindex(nodes.index).to_numpy(dtype=float)
        next_nodes["planned_output"] = planned
        next_nodes["input_availability"] = input_availability
        next_nodes["adjusted_input_availability"] = adjusted_input_availability
        next_nodes["substitution_gain"] = substitution_gain
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
            "total_initial_input_availability": float(input_availability.sum()),
            "total_adjusted_input_availability": float(adjusted_input_availability.sum()),
            "total_substitution_gain": float(substitution_gain.sum()),
        }
        return next_state, diagnostics
