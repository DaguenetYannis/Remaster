from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from src.abm_v3.calibration.emissions_model import EmissionsIntensityModel
from src.abm_v3.calibration.production_model import ProductionPlanningModel
from src.abm_v3.config import ABMV3Config
from src.abm_v3.data_loader import ABMV3DataLoader
from src.abm_v3.dynamics.step import ABMV3StepEngine
from src.abm_v3.paths import ABMV3Paths
from src.abm_v3.scenarios.base import BaseScenario
from src.abm_v3.scenarios.registry import get_scenario
from src.abm_v3.state import ABMState

LOGGER = logging.getLogger(__name__)


@dataclass
class ABMV3Model:
    """Orchestrate ABM v3 data, calibration, validation, and simulation."""

    config: ABMV3Config = field(default_factory=ABMV3Config)
    paths: ABMV3Paths = field(default_factory=ABMV3Paths)
    data_loader: ABMV3DataLoader | None = None
    production_model: ProductionPlanningModel | None = None
    emissions_model: EmissionsIntensityModel | None = None
    scenario: BaseScenario | None = None

    def __post_init__(self) -> None:
        if self.data_loader is None:
            self.data_loader = ABMV3DataLoader(self.paths)
        if self.scenario is None:
            self.scenario = get_scenario("baseline_continuation")

    def fit_historical(self, start_year: int | None = None, end_year: int | None = None) -> dict[str, object]:
        start = self.config.calibration.start_year if start_year is None else start_year
        end = self.config.calibration.end_year if end_year is None else end_year
        LOGGER.info("Scaffold historical fit requested for %s-%s.", start, end)
        return {"status": "scaffold", "start_year": start, "end_year": end}

    def validate_historical(self, split_year: int | None = None) -> dict[str, object]:
        split = self.config.calibration.validation_split_year if split_year is None else split_year
        LOGGER.info("Scaffold historical validation requested with split year %s.", split)
        return {"status": "scaffold", "split_year": split}

    def simulate(
        self,
        start_year: int,
        end_year: int,
        scenario: BaseScenario | str | None = None,
        initial_state: ABMState | None = None,
    ) -> pd.DataFrame:
        scenario_obj = get_scenario(scenario) if isinstance(scenario, str) else scenario or self.scenario
        if initial_state is None:
            LOGGER.info("Scaffold simulation requested without initial_state; returning empty frame.")
            return pd.DataFrame(
                columns=["country_sector", "Year", "X", "D", "EI", "emissions", "scenario"]
            )
        engine = ABMV3StepEngine(
            production_model=self.production_model,
            emissions_model=self.emissions_model,
        )
        states = [initial_state.nodes.assign(scenario=scenario_obj.name)]
        state = initial_state
        for year in range(start_year + 1, end_year + 1):
            state, _diagnostics = engine.step(state, next_year=year, scenario=scenario_obj)
            states.append(state.nodes.assign(scenario=scenario_obj.name))
        return pd.concat(states, ignore_index=True)
