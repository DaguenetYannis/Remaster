from __future__ import annotations

import pandas as pd
import pytest

from src.abm_v3.dynamics.demand_provider import DemandProvider


def test_historical_demand_provider_uses_future_year_panel() -> None:
    current_nodes = pd.DataFrame(
        {
            "country_sector": ["A", "B"],
            "Year": [1995, 1995],
            "D": [1.0, 2.0],
        },
        index=[10, 20],
    )
    historical_panel = pd.DataFrame(
        {
            "country_sector": ["A", "B"],
            "Year": [1996, 1996],
            "D": [100.0, 200.0],
        }
    )

    demand = DemandProvider(historical_panel=historical_panel).historical_demand(current_nodes, 1996)

    assert demand.tolist() == [100.0, 200.0]
    assert demand.index.tolist() == [10, 20]


def test_historical_demand_provider_raises_without_panel() -> None:
    current_nodes = pd.DataFrame({"country_sector": ["A"], "Year": [1995], "D": [1.0]})

    with pytest.raises(ValueError, match="no historical_panel"):
        DemandProvider().historical_demand(current_nodes, 1996)
