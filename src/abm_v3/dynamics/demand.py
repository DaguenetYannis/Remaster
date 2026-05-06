from __future__ import annotations

import pandas as pd


def get_historical_demand(state_or_panel: object, year: int) -> pd.Series:
    """Return exogenous historical demand for a year."""

    panel = getattr(state_or_panel, "nodes", state_or_panel)
    year_panel = panel[panel["Year"] == year] if "Year" in panel.columns else panel
    if "D" not in year_panel.columns:
        raise ValueError("Historical demand requires column D.")
    return pd.Series(year_panel["D"].to_numpy(dtype=float), index=year_panel.index, name="D")


def project_demand(previous_state: object, scenario: object) -> pd.Series:
    """Project demand from scenario parameters for post-2016 simulations."""

    nodes = getattr(previous_state, "nodes", previous_state)
    if "D" not in nodes.columns:
        raise ValueError("Projection demand requires previous D column.")
    parameters = getattr(scenario, "parameters", {}) or {}
    growth_rate = float(parameters.get("demand_growth_rate", 0.0))
    return pd.Series(nodes["D"].astype(float) * (1.0 + growth_rate), index=nodes.index, name="D")
