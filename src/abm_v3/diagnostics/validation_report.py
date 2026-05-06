from __future__ import annotations

import pandas as pd


def compare_simulated_observed(
    simulated: pd.DataFrame,
    observed: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compare simulated and observed historical node-level values."""

    columns = columns or ["X", "EI", "emissions"]
    merged = simulated.merge(
        observed,
        on=["country_sector", "Year"],
        suffixes=("_simulated", "_observed"),
        how="inner",
    )
    records = []
    for column in columns:
        sim_col = f"{column}_simulated"
        obs_col = f"{column}_observed"
        if sim_col not in merged.columns or obs_col not in merged.columns:
            continue
        error = merged[sim_col] - merged[obs_col]
        records.append(
            {
                "variable": column,
                "matched_rows": len(merged),
                "mean_error": float(error.mean()),
                "mean_absolute_error": float(error.abs().mean()),
            }
        )
    return pd.DataFrame(records)
