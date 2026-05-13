from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ABMV4Scenario:
    """Scenario metadata for future ABM v4 policy extensions."""

    name: str
    family: str
    description: str
    demand_policy_enabled: bool = False
    expands_supplier_opportunities: bool = False
