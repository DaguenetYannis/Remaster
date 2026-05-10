"""ABM v3 behavioural Leontief scenario-readiness layer.

These scenarios are production-network perturbation experiments, not forecasts.
They are single-year comparative statics around a historical year: the baseline
and scenario use the same network structure, A matrix, corrected input panel,
and capacity vector unless capacity is explicitly shocked.

Green-ness is intentionally decomposed. Low EI and high green capability export
share are retained separately for interpretation. High green capability export
share is a productive capability proxy, not direct emissions performance. EI
transition dynamics do not drive ABM v3 scenarios. Capacity shocks are exogenous
stress tests, not adaptive capacity, investment, depreciation, or optimization.
"""

from src.abm_v3.leontief.scenarios.base import BehaviouralScenarioContext, BehaviouralScenarioShock
from src.abm_v3.leontief.scenarios.capacity_shocks import CapacityShock
from src.abm_v3.leontief.scenarios.demand_shocks import FinalDemandShock
from src.abm_v3.leontief.scenarios.registry import get_behavioural_scenario, list_behavioural_scenarios
from src.abm_v3.leontief.scenarios.runner import BehaviouralLeontiefScenarioRunner
from src.abm_v3.leontief.scenarios.selectors import GreenNodeSelector

__all__ = [
    "BehaviouralLeontiefScenarioRunner",
    "BehaviouralScenarioContext",
    "BehaviouralScenarioShock",
    "CapacityShock",
    "FinalDemandShock",
    "GreenNodeSelector",
    "get_behavioural_scenario",
    "list_behavioural_scenarios",
]
