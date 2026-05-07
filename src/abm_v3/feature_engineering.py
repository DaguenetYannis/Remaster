from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two aligned series and replace non-finite results with NaN."""

    result = numerator.astype(float) / denominator.astype(float)
    return result.replace([np.inf, -np.inf], np.nan)


def safe_log(series: pd.Series) -> pd.Series:
    """Log positive values and return NaN for zero, negative, or missing values."""

    numeric = pd.to_numeric(series, errors="coerce")
    return np.log(numeric.where(numeric > 0))


@dataclass
class FeatureEngineer:
    """Create transparent model-ready features from country-sector panels."""

    node_col: str = "country_sector"
    year_col: str = "Year"

    def sort_panel(self, df: pd.DataFrame) -> pd.DataFrame:
        required = [self.node_col, self.year_col]
        missing = [column for column in required if column not in df.columns]
        if missing:
            LOGGER.warning("Cannot sort panel; missing columns: %s", missing)
            return df.copy()
        return df.sort_values(required).reset_index(drop=True)

    def create_lags(
        self,
        df: pd.DataFrame,
        columns: list[str],
        periods: int = 1,
    ) -> pd.DataFrame:
        result = self.sort_panel(df)
        for column in columns:
            if column not in result.columns:
                LOGGER.warning("Skipping lag for missing column: %s", column)
                continue
            result[f"{column}_lag{periods}"] = result.groupby(self.node_col)[column].shift(periods)
        return result

    def create_growth_rates(
        self,
        df: pd.DataFrame,
        columns: list[str],
    ) -> pd.DataFrame:
        result = self.sort_panel(df)
        for column in columns:
            if column not in result.columns:
                LOGGER.warning("Skipping growth for missing column: %s", column)
                continue
            lag_col = f"{column}_lag1"
            if lag_col not in result.columns:
                result[lag_col] = result.groupby(self.node_col)[column].shift(1)
            result[f"{column}_growth"] = safe_divide(
                result[column] - result[lag_col],
                result[lag_col],
            )
        return result

    def create_demand_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        if {"D", "X"}.issubset(result.columns):
            result["demand_gap"] = safe_divide(result["D"] - result["X"], result["X"])
        else:
            LOGGER.warning("Cannot create demand_gap; D or X is missing.")
        return result

    def create_sector_level_growth(self, df: pd.DataFrame, value_col: str = "X") -> pd.DataFrame:
        result = df.copy()
        if not {"Sector", self.year_col, value_col}.issubset(result.columns):
            LOGGER.warning("Cannot create sector growth; required columns are missing.")
            return result
        sector_year = (
            result.groupby(["Sector", self.year_col], as_index=False)[value_col]
            .sum()
            .sort_values(["Sector", self.year_col])
        )
        sector_year["sector_X_growth"] = sector_year.groupby("Sector")[value_col].pct_change()
        return result.merge(
            sector_year[["Sector", self.year_col, "sector_X_growth"]],
            on=["Sector", self.year_col],
            how="left",
        )

    def create_country_level_growth(self, df: pd.DataFrame, value_col: str = "X") -> pd.DataFrame:
        result = df.copy()
        if not {"Country", self.year_col, value_col}.issubset(result.columns):
            LOGGER.warning("Cannot create country growth; required columns are missing.")
            return result
        country_year = (
            result.groupby(["Country", self.year_col], as_index=False)[value_col]
            .sum()
            .sort_values(["Country", self.year_col])
        )
        country_year["country_X_growth"] = country_year.groupby("Country")[value_col].pct_change()
        return result.merge(
            country_year[["Country", self.year_col, "country_X_growth"]],
            on=["Country", self.year_col],
            how="left",
        )

    def create_group_level_growth(
        self,
        df: pd.DataFrame,
        group_col: str,
        value_col: str,
        output_col: str,
    ) -> pd.DataFrame:
        """Create transparent group-year growth for controls such as sector EI."""

        result = df.copy()
        if not {group_col, self.year_col, value_col}.issubset(result.columns):
            LOGGER.warning(
                "Cannot create %s; required columns are missing.",
                output_col,
            )
            return result
        group_year = (
            result.groupby([group_col, self.year_col], as_index=False)[value_col]
            .mean()
            .sort_values([group_col, self.year_col])
        )
        group_year[output_col] = group_year.groupby(group_col)[value_col].pct_change()
        return result.merge(
            group_year[[group_col, self.year_col, output_col]],
            on=[group_col, self.year_col],
            how="left",
        )

    def create_emissions(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        if {"X", "EI"}.issubset(result.columns):
            result["emissions"] = result["X"].astype(float) * result["EI"].astype(float)
        else:
            LOGGER.warning("Cannot create emissions; X or EI is missing.")
        return result

    def create_next_period_targets(
        self,
        df: pd.DataFrame,
        x_col: str = "X",
        ei_col: str = "EI",
    ) -> pd.DataFrame:
        """Create one-step-ahead calibration targets for production and EI.

        Targets remain NaN when current or next-period values are unavailable,
        zero, or negative. NaN targets are preserved so calibration models can
        explicitly drop incomplete rows instead of treating missing dynamics as
        zero change.
        """

        result = self.sort_panel(df)
        if x_col in result.columns:
            result["X_next"] = result.groupby(self.node_col)[x_col].shift(-1)
            result["delta_log_X_next"] = safe_log(result["X_next"]) - safe_log(result[x_col])
        else:
            LOGGER.warning("Cannot create production target; missing column: %s", x_col)
            result["X_next"] = np.nan
            result["delta_log_X_next"] = np.nan

        if ei_col in result.columns:
            result["EI_next"] = result.groupby(self.node_col)[ei_col].shift(-1)
            result["delta_log_EI_next"] = safe_log(result["EI_next"]) - safe_log(result[ei_col])
        else:
            LOGGER.warning("Cannot create EI target; missing column: %s", ei_col)
            result["EI_next"] = np.nan
            result["delta_log_EI_next"] = np.nan
        return result

    def create_production_planning_scaffolds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create optional production-planning signals when source data exists."""

        result = df.copy()
        if {"available_inputs", "X"}.issubset(result.columns):
            result["input_availability_ratio"] = safe_divide(result["available_inputs"], result["X"])
        else:
            LOGGER.warning("Cannot create input_availability_ratio; available_inputs or X is missing.")
            result["input_availability_ratio"] = np.nan

        inventory_col = "I" if "I" in result.columns else "inventory" if "inventory" in result.columns else None
        if inventory_col is not None and "D" in result.columns:
            result["inventory_stress"] = 1.0 - safe_divide(result[inventory_col], result["D"])
        else:
            LOGGER.warning("Cannot create inventory_stress; inventory/I or D is missing.")
            result["inventory_stress"] = np.nan

        capacity_col = "K" if "K" in result.columns else "capacity" if "capacity" in result.columns else None
        if capacity_col is not None and "X" in result.columns:
            result["capacity_utilization"] = safe_divide(result["X"], result[capacity_col])
        else:
            LOGGER.warning("Cannot create capacity_utilization; capacity/K or X is missing.")
            result["capacity_utilization"] = np.nan

        return result

    def create_model_ready_panel(self, df: pd.DataFrame) -> pd.DataFrame:
        result = self.sort_panel(df)
        result = self.create_lags(result, ["X", "EI", "D"])
        result = self.create_growth_rates(result, ["X", "EI", "D"])
        result = self.create_demand_gaps(result)
        result = self.create_sector_level_growth(result)
        result = self.create_country_level_growth(result)
        result = self.create_group_level_growth(result, "Sector", "EI", "sector_EI_growth")
        result = self.create_group_level_growth(result, "Country", "EI", "country_EI_growth")
        result = self.create_production_planning_scaffolds(result)
        result = self.create_emissions(result)
        result = self.create_next_period_targets(result)
        if "X_lag1" in result.columns:
            result["log_X_lag1"] = safe_log(result["X_lag1"])
        if "EI_lag1" in result.columns:
            result["log_EI_lag1"] = safe_log(result["EI_lag1"])
        if {"g_in", "g_out"}.issubset(result.columns):
            result["g_network"] = 0.5 * (result["g_in"].astype(float) + result["g_out"].astype(float))
        return result
