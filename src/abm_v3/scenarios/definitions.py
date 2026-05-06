from __future__ import annotations

SCENARIO_DEFINITIONS: dict[str, dict[str, object]] = {
    "baseline_continuation": {
        "description": "Historical mechanisms continue with neutral demand growth.",
        "parameters": {"demand_growth_rate": 0.0, "substitution_friction": 0.25},
    },
    "green_demand_shift": {
        "description": "Placeholder for future green-biased demand scenario.",
        "parameters": {"demand_growth_rate": 0.01},
        "network_greenness_affects_demand": True,
    },
    "strong_network_diffusion": {
        "description": "Placeholder for stronger network effects on EI reduction.",
        "parameters": {"network_diffusion_multiplier": 1.5},
    },
    "green_supplier_preference": {
        "description": "Scenario extension allowing explicit green supplier preference.",
        "parameters": {"green_supplier_weight": 0.5},
        "network_greenness_affects_supplier_choice": True,
        "green_supplier_preference": True,
    },
    "low_substitution_friction": {
        "description": "Scenario with easier supplier substitution.",
        "parameters": {"substitution_friction": 0.1},
    },
    "high_substitution_friction": {
        "description": "Scenario with harder supplier substitution.",
        "parameters": {"substitution_friction": 0.75},
    },
}
