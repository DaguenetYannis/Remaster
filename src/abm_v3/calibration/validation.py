from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.abm_v3.calibration.loss_functions import emissions_loss, ei_loss, output_loss


@dataclass
class HistoricalValidator:
    """Validate simulated historical dynamics against observed panels."""

    split_year: int = 2008

    def split_panel(self, df: pd.DataFrame, split_year: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        split = self.split_year if split_year is None else split_year
        calibration = df[df["Year"] <= split].copy()
        validation = df[df["Year"] > split].copy()
        return calibration, validation

    def validate_simulation(self, simulated: pd.DataFrame, observed: pd.DataFrame) -> dict[str, float]:
        merged = simulated.merge(
            observed,
            on=["country_sector", "Year"],
            suffixes=("_simulated", "_observed"),
            how="inner",
        )
        return {
            "output_loss": output_loss(merged["X_simulated"], merged["X_observed"]),
            "emissions_loss": emissions_loss(merged["emissions_simulated"], merged["emissions_observed"]),
            "ei_loss": ei_loss(merged["EI_simulated"], merged["EI_observed"]),
            "matched_rows": float(len(merged)),
        }

    def summarize_validation(self, metrics: dict[str, float]) -> pd.DataFrame:
        return pd.DataFrame([{"metric": key, "value": value} for key, value in metrics.items()])
