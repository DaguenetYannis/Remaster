from __future__ import annotations

import pandas as pd


def emissions_decomposition(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    key_col: str = "country_sector",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Decompose emissions change into output and EI components."""

    merged = previous[[key_col, "X", "EI"]].merge(
        current[[key_col, "X", "EI"]],
        on=key_col,
        suffixes=("_t", "_next"),
        how="inner",
    )
    merged["delta_X"] = merged["X_next"] - merged["X_t"]
    merged["delta_EI"] = merged["EI_next"] - merged["EI_t"]
    merged["ei_effect"] = merged["X_t"] * merged["delta_EI"]
    merged["output_effect"] = merged["EI_t"] * merged["delta_X"]
    merged["interaction_effect"] = merged["delta_X"] * merged["delta_EI"]
    merged["delta_emissions_approx"] = (
        merged["ei_effect"] + merged["output_effect"] + merged["interaction_effect"]
    )
    summary = {
        "ei_effect": float(merged["ei_effect"].sum()),
        "output_effect": float(merged["output_effect"].sum()),
        "interaction_effect": float(merged["interaction_effect"].sum()),
        "delta_emissions_approx": float(merged["delta_emissions_approx"].sum()),
    }
    return merged, summary
