from __future__ import annotations

import pandas as pd

from src.abm_v3.dynamics.production import plan_production, realize_production


def test_planned_production_returns_expected_shape() -> None:
    nodes = pd.DataFrame({"X": [10.0, 20.0]})
    planned = plan_production(nodes)
    assert len(planned) == len(nodes)
    assert planned.name == "planned_output"


def test_realized_production_respects_simple_constraints() -> None:
    planned = pd.Series([10.0, 20.0])
    demand = pd.Series([8.0, 30.0])
    input_availability = pd.Series([9.0, 15.0])
    realized = realize_production(planned, demand, input_availability)
    assert realized.tolist() == [8.0, 15.0]
