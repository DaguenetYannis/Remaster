from __future__ import annotations

import pandas as pd

from src.abm_v3.dynamics.demand_provider import DemandProvider
from src.abm_v3.dynamics.step import ABMV3StepEngine
from src.abm_v3.state import ABMState, ABMStateMetadata


def current_state_for_substitution() -> ABMState:
    nodes = pd.DataFrame(
        {
            "country_sector": ["A", "B"],
            "Country": ["AAA", "BBB"],
            "Sector": ["Manufacturing", "Manufacturing"],
            "Year": [1995, 1995],
            "X": [100.0, 80.0],
            "D": [10.0, 20.0],
            "EI": [2.0, 1.0],
            "available_inputs": [50.0, 200.0],
        }
    )
    return ABMState(nodes=nodes, metadata=ABMStateMetadata(year=1995))


def historical_panel_1996() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "country_sector": ["A", "B"],
            "Year": [1996, 1996],
            "D": [100.0, 80.0],
        }
    )


def test_step_engine_uses_historical_next_year_demand() -> None:
    engine = ABMV3StepEngine(
        demand_provider=DemandProvider(historical_panel=historical_panel_1996()),
        sigma=0.0,
    )

    next_state, _diagnostics = engine.step(current_state_for_substitution(), next_year=1996)

    assert next_state.nodes["D"].tolist() == [100.0, 80.0]


def test_step_engine_integrates_supplier_substitution() -> None:
    engine = ABMV3StepEngine(
        demand_provider=DemandProvider(historical_panel=historical_panel_1996()),
        sigma=1.0,
    )

    next_state, diagnostics = engine.step(current_state_for_substitution(), next_year=1996)
    node_a = next_state.nodes[next_state.nodes["country_sector"] == "A"].iloc[0]

    assert node_a["adjusted_input_availability"] > node_a["input_availability"]
    assert node_a["realized_output"] > node_a["input_availability"]
    assert diagnostics["total_substitution_gain"] > 0


def test_sigma_zero_no_substitution_in_step() -> None:
    engine = ABMV3StepEngine(
        demand_provider=DemandProvider(historical_panel=historical_panel_1996()),
        sigma=0.0,
    )

    next_state, diagnostics = engine.step(current_state_for_substitution(), next_year=1996)

    assert (
        next_state.nodes["adjusted_input_availability"]
        == next_state.nodes["input_availability"]
    ).all()
    assert (next_state.nodes["substitution_gain"] == 0.0).all()
    assert diagnostics["total_substitution_gain"] == 0.0


def test_green_supplier_preference_not_used_in_base_step() -> None:
    nodes = pd.DataFrame(
        {
            "country_sector": ["A", "B", "C"],
            "Sector": ["Manufacturing", "Manufacturing", "Services"],
            "Year": [1995, 1995, 1995],
            "X": [100.0, 80.0, 80.0],
            "D": [10.0, 20.0, 20.0],
            "EI": [2.0, 1.0, 1.0],
            "g_local": [0.1, 0.0, 1.0],
            "available_inputs": [50.0, 200.0, 200.0],
        }
    )
    historical_panel = pd.DataFrame(
        {
            "country_sector": ["A", "B", "C"],
            "Year": [1996, 1996, 1996],
            "D": [100.0, 80.0, 80.0],
        }
    )
    engine = ABMV3StepEngine(
        demand_provider=DemandProvider(historical_panel=historical_panel),
        sigma=1.0,
    )

    next_state, diagnostics = engine.step(
        ABMState(nodes=nodes, metadata=ABMStateMetadata(year=1995)),
        next_year=1996,
    )
    node_a = next_state.nodes[next_state.nodes["country_sector"] == "A"].iloc[0]

    assert node_a["substitution_gain"] > 0
    assert diagnostics["total_substitution_gain"] > 0
