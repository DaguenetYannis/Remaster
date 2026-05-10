from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from src.abm_v3.leontief.coefficients import LeontiefYearData
from src.abm_v3.leontief.scenarios.base import BehaviouralScenarioContext
from src.abm_v3.leontief.scenarios.selectors import GreenNodeSelector


@dataclass
class FinalDemandShock:
    """Exogenous final-demand shock for selected country-sector nodes."""

    name: str
    selector_name: str
    shock_size: float
    shock_mode: str = "multiplicative"
    description: str = "Exogenous final-demand perturbation for selected nodes."
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
        self._validate_alignment(year_data.Y_final_demand, selected_labels, "Y_final_demand")
        y_scenario = year_data.Y_final_demand.copy()
        y_baseline_selected = y_scenario.reindex(selected_labels).astype(float)
        y_scenario_selected = self._apply_y_shock(y_scenario, selected_labels)
        y_scenario.loc[selected_labels] = y_scenario_selected
        selected = selected.copy()
        selected["scenario_name"] = context.scenario_name
        selected["selector_name"] = self.selector_name
        selected["shock_type"] = "final_demand"
        selected["shock_mode"] = self.shock_mode
        selected["shock_size"] = self.shock_size
        selected["Y_baseline"] = y_baseline_selected.to_numpy(dtype=float)
        selected["Y_scenario"] = y_scenario_selected.to_numpy(dtype=float)
        selected["delta_Y"] = selected["Y_scenario"] - selected["Y_baseline"]
        selected["pct_delta_Y"] = self._safe_ratio(selected["delta_Y"], selected["Y_baseline"])
        return replace(year_data, Y_final_demand=y_scenario), capacity.copy(), selected

    def _apply_y_shock(self, y: pd.Series, selected_labels: list[str]) -> pd.Series:
        selected_y = y.reindex(selected_labels).astype(float)
        if self.shock_mode == "multiplicative":
            return selected_y * (1.0 + float(self.shock_size))
        if self.shock_mode == "additive_share_of_total_final_demand":
            total_y = float(pd.to_numeric(y, errors="coerce").sum(skipna=True))
            addition = float(self.shock_size) * total_y / len(selected_labels)
            return selected_y + addition
        if self.shock_mode == "additive_share_of_selected_final_demand":
            selected_total = float(selected_y.sum(skipna=True))
            if not np.isfinite(selected_total) or selected_total <= 0.0:
                raise ValueError("additive_share_of_selected_final_demand requires positive selected final demand.")
            weights = selected_y / selected_total
            return selected_y + float(self.shock_size) * selected_total * weights
        raise ValueError(
            "Unknown final-demand shock_mode "
            f"'{self.shock_mode}'. Allowed: multiplicative, additive_share_of_total_final_demand, "
            "additive_share_of_selected_final_demand."
        )

    def _validate_alignment(self, series: pd.Series, selected_labels: list[str], label: str) -> None:
        missing = sorted(set(selected_labels).difference(set(series.index.astype(str))))
        if missing:
            raise ValueError(f"Selected nodes are missing from {label}: {missing[:5]}")

    def _safe_ratio(self, numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        ratio = pd.to_numeric(numerator, errors="coerce") / pd.to_numeric(denominator, errors="coerce").where(
            pd.to_numeric(denominator, errors="coerce") != 0.0
        )
        return ratio.replace([np.inf, -np.inf], np.nan)
