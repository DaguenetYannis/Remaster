from __future__ import annotations

import pandas as pd


def rank_scenarios(
    scenario_summary: pd.DataFrame,
    emissions_reduction_col: str = "emissions_reduction",
    output_loss_col: str = "output_loss",
    output_loss_penalty: float = 2.0,
) -> pd.DataFrame:
    """Rank scenarios while penalizing output loss."""

    required = ["scenario", emissions_reduction_col, output_loss_col]
    missing = [column for column in required if column not in scenario_summary.columns]
    if missing:
        raise ValueError(f"Missing scenario ranking columns: {missing}")
    result = scenario_summary.copy()
    result["score"] = (
        result[emissions_reduction_col].astype(float)
        - output_loss_penalty * result[output_loss_col].astype(float)
    )
    return result.sort_values("score", ascending=False).reset_index(drop=True)
