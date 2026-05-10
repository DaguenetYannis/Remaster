from __future__ import annotations

from src.abm_v3.leontief.scenarios.base import BehaviouralScenarioShock
from src.abm_v3.leontief.scenarios.capacity_shocks import CapacityShock
from src.abm_v3.leontief.scenarios.demand_shocks import FinalDemandShock


def _build_registry() -> dict[str, BehaviouralScenarioShock]:
    return {
        "low_ei_node_demand_expansion_10": FinalDemandShock(
            name="low_ei_node_demand_expansion_10",
            selector_name="low_EI",
            shock_size=0.10,
            description="Increase final demand by 10 percent for low-EI country-sector nodes.",
        ),
        "green_capability_node_demand_expansion_10": FinalDemandShock(
            name="green_capability_node_demand_expansion_10",
            selector_name="high_green_capability_export_share",
            shock_size=0.10,
            description="Increase final demand by 10 percent for high green productive-capability nodes.",
        ),
        "clean_and_capable_node_demand_expansion_10": FinalDemandShock(
            name="clean_and_capable_node_demand_expansion_10",
            selector_name="clean_and_capable",
            shock_size=0.10,
            description="Increase final demand by 10 percent for nodes that are both low-EI and high-capability.",
        ),
        "transition_pivot_node_demand_expansion_10": FinalDemandShock(
            name="transition_pivot_node_demand_expansion_10",
            selector_name="transition_pivot",
            shock_size=0.10,
            description="Increase final demand by 10 percent for high-EI, high-capability transition-pivot nodes.",
        ),
        "high_ei_node_capacity_bottleneck_10": CapacityShock(
            name="high_ei_node_capacity_bottleneck_10",
            selector_name="high_EI",
            shock_size=-0.10,
            description="Reduce capacity by 10 percent for high-EI nodes as an exogenous bottleneck stress test.",
        ),
    }


SCENARIO_REGISTRY = _build_registry()


def get_behavioural_scenario(name: str) -> BehaviouralScenarioShock:
    if name not in SCENARIO_REGISTRY:
        raise KeyError(f"Unknown behavioural scenario '{name}'. Available: {list_behavioural_scenarios()}")
    return SCENARIO_REGISTRY[name]


def list_behavioural_scenarios() -> list[str]:
    return sorted(SCENARIO_REGISTRY)
