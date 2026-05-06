from __future__ import annotations

import numpy as np
import pandas as pd


def update_emissions_intensity(
    nodes: pd.DataFrame,
    emissions_model: object | None = None,
    current_ei_col: str = "EI",
) -> pd.Series:
    """Update EI with a log-scale model so simulated EI remains non-negative."""

    if emissions_model is None:
        return pd.Series(nodes[current_ei_col].to_numpy(dtype=float), index=nodes.index, name="EI")
    next_ei = emissions_model.predict_next_ei(nodes, current_ei_col=current_ei_col)
    return pd.Series(next_ei.replace([np.inf, -np.inf], np.nan), index=nodes.index, name="EI")


def compute_emissions(output: pd.Series, emissions_intensity: pd.Series) -> pd.Series:
    """Apply the emissions identity E = X * EI."""

    emissions = output.astype(float) * emissions_intensity.astype(float)
    return pd.Series(emissions.to_numpy(dtype=float), index=output.index, name="emissions")
