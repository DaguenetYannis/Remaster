from __future__ import annotations

import pandas as pd


def compute_input_needs(
    planned_output: pd.Series,
    technical_coefficients: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute toy-compatible input needs from historical coefficients."""

    if technical_coefficients is None:
        return pd.DataFrame({"country_sector": planned_output.index, "input_need": planned_output.to_numpy(dtype=float)}, index=planned_output.index)
    return technical_coefficients.mul(planned_output, axis=1)


def compute_input_availability(
    nodes: pd.DataFrame,
    supply_col: str = "available_inputs",
    fallback_col: str = "X",
) -> pd.Series:
    """Return available input constraint without recalculating Eora coefficients."""

    if supply_col in nodes.columns:
        return pd.Series(nodes[supply_col].to_numpy(dtype=float), index=nodes.index, name="input_availability")
    if fallback_col in nodes.columns:
        return pd.Series(nodes[fallback_col].to_numpy(dtype=float), index=nodes.index, name="input_availability")
    raise ValueError("Cannot compute input availability; no supply or fallback column found.")
