from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


CAPABILITY_FALLBACK_COLUMNS = [
    "green_capability_export_share",
    "green_capability",
    "green_capability_share",
]

CAPABILITY_BASED_SELECTORS = {
    "high_green_capability_export_share",
    "clean_and_capable",
    "transition_pivot",
}

ALLOWED_SELECTORS = {
    "all_nodes",
    "low_EI",
    "high_green_capability_export_share",
    "clean_and_capable",
    "transition_pivot",
    "high_EI",
    "manual_country",
    "manual_sector",
    "manual_country_sector",
}


@dataclass
class GreenNodeSelector:
    """Build node-selection diagnostics without collapsing green-ness to one label."""

    input_panel: pd.DataFrame
    year: int
    ei_column: str = "EI"
    green_capability_column: str = "green_capability_export_share"

    def build_diagnostics(
        self,
        low_ei_quantile: float = 0.25,
        high_ei_quantile: float = 0.75,
        high_capability_quantile: float = 0.75,
    ) -> pd.DataFrame:
        year_panel = self.input_panel.loc[pd.to_numeric(self.input_panel["Year"], errors="coerce").eq(self.year)].copy()
        if year_panel.empty:
            raise ValueError(f"No input-panel rows found for scenario year {self.year}.")
        required = ["Year", "country_sector", "Country", "Country_detail", "Category", "Sector"]
        missing = [column for column in required if column not in year_panel.columns]
        if missing:
            raise ValueError(f"Input panel is missing selector label columns: {missing}")

        diagnostics = year_panel.reindex(columns=self._diagnostic_source_columns()).copy()
        diagnostics["EI"] = pd.to_numeric(year_panel.get(self.ei_column), errors="coerce")
        metric_column = self._capability_metric_column(year_panel)
        diagnostics["green_capability_metric_used"] = metric_column or ""
        if metric_column is None:
            diagnostics["green_capability_metric_value"] = np.nan
        else:
            diagnostics["green_capability_metric_value"] = pd.to_numeric(year_panel[metric_column], errors="coerce")

        valid_ei = np.isfinite(diagnostics["EI"]) & (diagnostics["EI"] > 0.0)
        diagnostics["EI_rank_pct"] = np.nan
        diagnostics.loc[valid_ei, "EI_rank_pct"] = diagnostics.loc[valid_ei, "EI"].rank(pct=True, ascending=True)
        valid_capability = np.isfinite(diagnostics["green_capability_metric_value"])
        diagnostics["green_capability_rank_pct"] = np.nan
        diagnostics.loc[valid_capability, "green_capability_rank_pct"] = diagnostics.loc[
            valid_capability,
            "green_capability_metric_value",
        ].rank(pct=True, ascending=True)
        diagnostics["is_low_EI"] = valid_ei & (diagnostics["EI_rank_pct"] <= low_ei_quantile)
        diagnostics["is_high_EI"] = valid_ei & (diagnostics["EI_rank_pct"] >= high_ei_quantile)
        diagnostics["is_high_green_capability_export_share"] = valid_capability & (
            diagnostics["green_capability_rank_pct"] >= high_capability_quantile
        )
        diagnostics["is_clean_and_capable"] = diagnostics["is_low_EI"] & diagnostics["is_high_green_capability_export_share"]
        diagnostics["is_transition_pivot"] = diagnostics["is_high_EI"] & diagnostics["is_high_green_capability_export_share"]
        return diagnostics

    def select(
        self,
        selector_name: str,
        low_ei_quantile: float = 0.25,
        high_ei_quantile: float = 0.75,
        high_capability_quantile: float = 0.75,
        countries: list[str] | None = None,
        sectors: list[str] | None = None,
        country_sectors: list[str] | None = None,
    ) -> pd.DataFrame:
        if selector_name not in ALLOWED_SELECTORS:
            raise ValueError(f"Unknown selector '{selector_name}'. Allowed selectors: {sorted(ALLOWED_SELECTORS)}")
        diagnostics = self.build_diagnostics(low_ei_quantile, high_ei_quantile, high_capability_quantile)
        if selector_name in CAPABILITY_BASED_SELECTORS and not diagnostics["green_capability_metric_used"].astype(bool).any():
            raise ValueError(
                "No green capability metric found. Expected one of: "
                f"{CAPABILITY_FALLBACK_COLUMNS}"
            )

        selected = self._filter_diagnostics(
            diagnostics,
            selector_name,
            countries=countries,
            sectors=sectors,
            country_sectors=country_sectors,
        )
        if selected.empty:
            raise ValueError(f"Selector '{selector_name}' returned no nodes for year {self.year}.")
        return selected.copy()

    def _filter_diagnostics(
        self,
        diagnostics: pd.DataFrame,
        selector_name: str,
        countries: list[str] | None,
        sectors: list[str] | None,
        country_sectors: list[str] | None,
    ) -> pd.DataFrame:
        if selector_name == "all_nodes":
            return diagnostics
        if selector_name == "low_EI":
            return diagnostics.loc[diagnostics["is_low_EI"]]
        if selector_name == "high_EI":
            return diagnostics.loc[diagnostics["is_high_EI"]]
        if selector_name == "high_green_capability_export_share":
            return diagnostics.loc[diagnostics["is_high_green_capability_export_share"]]
        if selector_name == "clean_and_capable":
            return diagnostics.loc[diagnostics["is_clean_and_capable"]]
        if selector_name == "transition_pivot":
            return diagnostics.loc[diagnostics["is_transition_pivot"]]
        if selector_name == "manual_country":
            if not countries:
                raise ValueError("manual_country selector requires countries.")
            return diagnostics.loc[diagnostics["Country"].astype(str).isin(countries)]
        if selector_name == "manual_sector":
            if not sectors:
                raise ValueError("manual_sector selector requires sectors.")
            return diagnostics.loc[diagnostics["Sector"].astype(str).isin(sectors)]
        if selector_name == "manual_country_sector":
            if not country_sectors:
                raise ValueError("manual_country_sector selector requires country_sectors.")
            return diagnostics.loc[diagnostics["country_sector"].astype(str).isin(country_sectors)]
        raise ValueError(f"Unhandled selector '{selector_name}'.")

    def _capability_metric_column(self, df: pd.DataFrame) -> str | None:
        candidates = [self.green_capability_column] + [
            column for column in CAPABILITY_FALLBACK_COLUMNS if column != self.green_capability_column
        ]
        for column in candidates:
            if column in df.columns:
                return column
        return None

    def _diagnostic_source_columns(self) -> list[str]:
        return [
            "Year",
            "country_sector",
            "Country",
            "Country_detail",
            "Category",
            "Sector",
            "EI",
            "green_capability_export_share",
            "green_capability",
            "green_capability_share",
        ]
