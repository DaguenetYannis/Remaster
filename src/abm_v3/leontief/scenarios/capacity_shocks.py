from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.abm_v3.leontief.coefficients import LeontiefYearData
from src.abm_v3.leontief.scenarios.base import BehaviouralScenarioContext
from src.abm_v3.leontief.scenarios.selectors import GreenNodeSelector


@dataclass
class CapacityShock:
    """Exogenous capacity stress test for selected country-sector nodes."""

    name: str
    selector_name: str
    shock_size: float
    shock_mode: str = "multiplicative"
    description: str = "Exogenous capacity shock; not adaptive capacity or investment."
    low_ei_quantile: float = 0.25
    high_ei_quantile: float = 0.75
    high_capability_quantile: float = 0.75
    countries: list[str] | None = None
    sectors: list[str] | None = None
    country_sectors: list[str] | None = None

    def apply(
        self,
        year_data: LeontiefYearData,
        capacity: pd.Series,
        input_panel: pd.DataFrame,
        context: BehaviouralScenarioContext,
    ) -> tuple[LeontiefYearData, pd.Series, pd.DataFrame]:
        selected = GreenNodeSelector(input_panel, context.year).select(
            self.selector_name,
            low_ei_quantile=self.low_ei_quantile,
            high_ei_quantile=self.high_ei_quantile,
            high_capability_quantile=self.high_capability_quantile,
            countries=self.countries,
            sectors=self.sectors,
            country_sectors=self.country_sectors,
        )
        selected_labels = selected["country_sector"].astype(str).tolist()
        self._validate_alignment(capacity, selected_labels)
        scenario_capacity = capacity.copy()
        k_baseline = scenario_capacity.reindex(selected_labels).astype(float)
        k_scenario = self._apply_capacity_shock(k_baseline)
        scenario_capacity.loc[selected_labels] = k_scenario
        selected = selected.copy()
        selected["scenario_name"] = context.scenario_name
        selected["selector_name"] = self.selector_name
        selected["shock_type"] = "capacity"
        selected["shock_mode"] = self.shock_mode
        selected["shock_size"] = self.shock_size
        selected["K_baseline"] = k_baseline.to_numpy(dtype=float)
        selected["K_scenario"] = k_scenario.to_numpy(dtype=float)
        selected["delta_K"] = selected["K_scenario"] - selected["K_baseline"]
        selected["pct_delta_K"] = self._safe_ratio(selected["delta_K"], selected["K_baseline"])
        return year_data, scenario_capacity, selected

    def _apply_capacity_shock(self, selected_capacity: pd.Series) -> pd.Series:
        if self.shock_mode != "multiplicative":
            raise ValueError("CapacityShock currently supports only multiplicative shock_mode.")
        shocked = selected_capacity * (1.0 + float(self.shock_size))
        return shocked.clip(lower=0.0)

    def _validate_alignment(self, capacity: pd.Series, selected_labels: list[str]) -> None:
        missing = sorted(set(selected_labels).difference(set(capacity.index.astype(str))))
        if missing:
            raise ValueError(f"Selected nodes are missing from capacity: {missing[:5]}")

    def _safe_ratio(self, numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        ratio = pd.to_numeric(numerator, errors="coerce") / pd.to_numeric(denominator, errors="coerce").where(
            pd.to_numeric(denominator, errors="coerce") != 0.0
        )
        return ratio.replace([np.inf, -np.inf], np.nan)
