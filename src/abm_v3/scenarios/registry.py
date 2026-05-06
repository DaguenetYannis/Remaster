from __future__ import annotations

from src.abm_v3.scenarios.base import BaseScenario
from src.abm_v3.scenarios.definitions import SCENARIO_DEFINITIONS


def build_registry() -> dict[str, BaseScenario]:
    registry: dict[str, BaseScenario] = {}
    for name, definition in SCENARIO_DEFINITIONS.items():
        registry[name] = BaseScenario(
            name=name,
            description=str(definition["description"]),
            parameters=dict(definition.get("parameters", {})),
            demand_mode=str(definition.get("demand_mode", "historical_then_projected")),
            network_greenness_affects_demand=bool(
                definition.get("network_greenness_affects_demand", False)
            ),
            network_greenness_affects_supplier_choice=bool(
                definition.get("network_greenness_affects_supplier_choice", False)
            ),
            green_supplier_preference=bool(definition.get("green_supplier_preference", False)),
        )
    return registry


SCENARIO_REGISTRY = build_registry()


def get_scenario(name: str) -> BaseScenario:
    if name not in SCENARIO_REGISTRY:
        available = sorted(SCENARIO_REGISTRY)
        raise KeyError(f"Unknown scenario {name!r}. Available scenarios: {available}")
    return SCENARIO_REGISTRY[name]


def list_scenarios() -> list[str]:
    return sorted(SCENARIO_REGISTRY)
