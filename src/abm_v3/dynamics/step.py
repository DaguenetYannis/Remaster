from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.abm_v3.dynamics.demand_provider import DemandProvider
from src.abm_v3.dynamics.emissions import compute_emissions, update_emissions_intensity
from src.abm_v3.dynamics.inputs import compute_input_availability
from src.abm_v3.dynamics.production import plan_production, realize_production_with_soft_input_constraint
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
    input_rigidity: float = 0.5

    def step(self, current_state: ABMState, next_year: int, scenario: object | None = None) -> tuple[ABMState, dict[str, object]]:
        nodes = current_state.nodes.copy()
        demand_provider = self.demand_provider or DemandProvider()
        if next_year <= self.historical_end_year:
            demand = demand_provider.historical_demand(nodes, next_year)
        else:
            demand = demand_provider.projected_demand(nodes, scenario)
        planned = plan_production(nodes, self.production_model)
        capacity = pd.Series(nodes["K"].to_numpy(dtype=float), index=nodes.index, name="K") if "K" in nodes.columns else None
        desired_pre_substitution = pd.concat(
            [
                planned,
                demand.reindex(nodes.index),
                pd.Series(float("inf"), index=nodes.index) if capacity is None else capacity,
            ],
            axis=1,
        ).min(axis=1, skipna=False)
        input_availability = compute_input_availability(nodes)
        adjusted_input_availability = compute_substitution_adjusted_input_availability(
            nodes=nodes,
            input_availability=input_availability,
            demand=desired_pre_substitution,
            planned_output=desired_pre_substitution,
            sigma=self.sigma,
        )
        substitution_gain = adjusted_input_availability - input_availability.reindex(nodes.index)
        if "effective_input_intensity" in nodes.columns:
            effective_input_intensity = pd.Series(
                nodes["effective_input_intensity"].to_numpy(dtype=float),
                index=nodes.index,
                name="effective_input_intensity",
            )
        else:
            effective_input_intensity = pd.Series(float("nan"), index=nodes.index, name="effective_input_intensity")
        production_result = realize_production_with_soft_input_constraint(
            planned_output=planned,
            demand=demand.reindex(nodes.index),
            capacity=capacity,
            adjusted_input_availability=adjusted_input_availability,
            effective_input_intensity=effective_input_intensity,
            input_rigidity=self.input_rigidity,
        )
        realized = production_result["realized_output"]
        next_ei = update_emissions_intensity(nodes, self.emissions_model)
        emissions = compute_emissions(realized, next_ei)
        next_nodes = nodes.copy()
        next_nodes["Year"] = next_year
        next_nodes["D"] = demand.reindex(nodes.index).to_numpy(dtype=float)
        next_nodes["planned_output"] = planned
        next_nodes["input_availability"] = input_availability
        next_nodes["adjusted_input_availability"] = adjusted_input_availability
        next_nodes["substitution_gain"] = substitution_gain
        next_nodes["desired_output"] = production_result["desired_output"]
        next_nodes["effective_input_intensity"] = effective_input_intensity
        if "input_intensity_source" not in next_nodes.columns:
            next_nodes["input_intensity_source"] = "missing"
        next_nodes["input_feasible_output"] = production_result["input_feasible_output"]
        next_nodes["input_stress_ratio"] = production_result["input_stress_ratio"]
        next_nodes["input_stress_factor"] = production_result["input_stress_factor"]
        next_nodes["input_constraint_penalty"] = production_result["input_constraint_penalty"]
        next_nodes["input_constraint_binding"] = production_result["input_constraint_binding"]
        next_nodes["input_feasibility_missing"] = production_result["input_feasibility_missing"]
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
            "total_desired_output": float(production_result["desired_output"].sum()),
            "total_realized_output": float(realized.sum()),
            "total_emissions": float(emissions.sum()),
            "total_initial_input_availability": float(input_availability.sum()),
            "total_adjusted_input_availability": float(adjusted_input_availability.sum()),
            "total_substitution_gain": float(substitution_gain.sum()),
            "mean_input_stress_factor": float(production_result["input_stress_factor"].mean()),
            "share_input_constrained": float(production_result["input_constraint_binding"].mean()),
            "share_input_feasibility_missing": float(production_result["input_feasibility_missing"].mean()),
            "total_input_constraint_penalty": float(production_result["input_constraint_penalty"].sum()),
        }
        return next_state, diagnostics
