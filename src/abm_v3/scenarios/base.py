from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BaseScenario:
    """Scenario boundary for ABM v3 base and extension mechanisms."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    demand_mode: str = "historical_then_projected"
    network_greenness_affects_demand: bool = False
    network_greenness_affects_supplier_choice: bool = False
    green_supplier_preference: bool = False
