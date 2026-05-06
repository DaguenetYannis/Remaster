"""
Scenario definitions for the ABM.

This module contains parameter presets only.
No model logic, no plotting, no file loading.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str

    n_steps: int = 30

    # Production constraints
    kappa: float = 1.10
    inventory_days: float = 30.0
    inventory_replenishment_rate: float = 0.20

    # Green transition dynamics
    alpha: float = 0.02
    beta: float = 0.03

    # Capability readiness
    lambda_local: float = 0.50
    lambda_network: float = 0.50
    lambda_ecosystem: float = 0.00

    # Network evolution
    eta: float = 0.01
    sigma: float = 0.00

    # Capability-constrained structural change
    q: float = 0.80
    rho: float = 0.00

    # Numerical stability
    epsilon: float = 1e-12


BASELINE = Scenario(
    name="baseline",
    description="Baseline constraint-based transition with moderate capacity slack and network diffusion.",
)

NO_CAPACITY_SLACK = replace(
    BASELINE,
    name="no_capacity_slack",
    description="Production capacity is fixed at observed baseline output.",
    kappa=1.00,
)

HIGH_CAPACITY_SLACK = replace(
    BASELINE,
    name="high_capacity_slack",
    description="Higher productive capacity margin.",
    kappa=1.25,
)

LOW_INVENTORY = replace(
    BASELINE,
    name="low_inventory",
    description="Low inventory buffers increase vulnerability to input constraints.",
    inventory_days=7.0,
)

HIGH_INVENTORY = replace(
    BASELINE,
    name="high_inventory",
    description="High inventory buffers soften input constraints.",
    inventory_days=60.0,
)

FAST_INTERNAL_GREENING = replace(
    BASELINE,
    name="fast_internal_greening",
    description="Faster within-node emissions intensity reduction.",
    alpha=0.06,
)

HIGH_NETWORK_DIFFUSION = replace(
    BASELINE,
    name="high_network_diffusion",
    description="Stronger influence of network green exposure on emissions intensity.",
    beta=0.08,
)

LOW_NETWORK_DIFFUSION = replace(
    BASELINE,
    name="low_network_diffusion",
    description="Weaker influence of network green exposure.",
    beta=0.005,
)

FAST_NETWORK_ADJUSTMENT = replace(
    BASELINE,
    name="fast_network_adjustment",
    description="Production links move faster toward feasible links.",
    eta=0.05,
)

CAPABILITY_LED_TRANSITION = replace(
    BASELINE,
    name="capability_led_transition",
    description="Capability readiness is driven more strongly by local green capability.",
    lambda_local=0.75,
    lambda_network=0.20,
    lambda_ecosystem=0.05,
)

NETWORK_LED_TRANSITION = replace(
    BASELINE,
    name="network_led_transition",
    description="Capability readiness is driven more strongly by network exposure.",
    lambda_local=0.25,
    lambda_network=0.70,
    lambda_ecosystem=0.05,
)

SCENARIOS: dict[str, Scenario] = {
    scenario.name: scenario
    for scenario in [
        BASELINE,
        NO_CAPACITY_SLACK,
        HIGH_CAPACITY_SLACK,
        LOW_INVENTORY,
        HIGH_INVENTORY,
        FAST_INTERNAL_GREENING,
        HIGH_NETWORK_DIFFUSION,
        LOW_NETWORK_DIFFUSION,
        FAST_NETWORK_ADJUSTMENT,
        CAPABILITY_LED_TRANSITION,
        NETWORK_LED_TRANSITION,
    ]
}


def get_scenario(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError as exc:
        available = ", ".join(sorted(SCENARIOS))
        raise ValueError(
            f"Unknown scenario: {name}. Available scenarios: {available}"
        ) from exc


def list_scenarios() -> list[str]:
    return sorted(SCENARIOS)


def override_scenario(base: Scenario, **kwargs: float | int | str) -> Scenario:
    """
    Creates a modified scenario without mutating the original one.
    """
    return replace(base, **kwargs)