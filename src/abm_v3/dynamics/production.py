from __future__ import annotations

import numpy as np
import pandas as pd


def plan_production(
    current_nodes: pd.DataFrame,
    planning_model: object | None = None,
    current_output_col: str = "X",
) -> pd.Series:
    """Plan production from a calibrated model or hold current output constant."""

    if planning_model is None:
        return pd.Series(current_nodes[current_output_col].to_numpy(dtype=float), index=current_nodes.index, name="planned_output")
    delta_log = planning_model.predict(current_nodes)
    planned = current_nodes[current_output_col].astype(float) * np.exp(delta_log)
    return pd.Series(planned, index=current_nodes.index, name="planned_output")


def realize_production(
    planned_output: pd.Series,
    demand: pd.Series,
    input_availability: pd.Series,
) -> pd.Series:
    """Realize output after substitution subject to quantity constraints.

    This is the final feasibility check. It is not an anti-collapse guardrail:
    missing or binding constraints remain visible in the output instead of
    being reset upward.
    """

    combined = pd.concat([planned_output, demand, input_availability], axis=1).astype(float)
    realized = combined.min(axis=1, skipna=False)
    return pd.Series(realized, index=planned_output.index, name="realized_output")
