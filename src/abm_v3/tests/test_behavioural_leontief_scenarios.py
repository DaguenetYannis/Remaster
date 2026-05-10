from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy import sparse

from src.abm_v3.config import ABMV3Config
from src.abm_v3.leontief.coefficients import LeontiefYearData
from src.abm_v3.leontief.scenarios.base import BehaviouralScenarioContext
from src.abm_v3.leontief.scenarios.capacity_shocks import CapacityShock
from src.abm_v3.leontief.scenarios.demand_shocks import FinalDemandShock
from src.abm_v3.leontief.scenarios.registry import get_behavioural_scenario, list_behavioural_scenarios
from src.abm_v3.leontief.scenarios.runner import BehaviouralLeontiefScenarioRunner
from src.abm_v3.leontief.scenarios.selectors import GreenNodeSelector
from src.abm_v3.paths import ABMV3Paths
from src.abm_v3.runner import build_parser


def labels() -> list[str]:
    return [
        "AAA | AAA | Industries | Agriculture",
        "BBB | BBB | Industries | Manufacturing",
        "CCC | CCC | Industries | Energy",
        "DDD | DDD | Industries | Services",
    ]


def input_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Year": [1995, 1995, 1995, 1995],
            "country_sector": labels(),
            "Country": ["AAA", "BBB", "CCC", "DDD"],
            "Country_detail": ["AAA", "BBB", "CCC", "DDD"],
            "Category": ["Industries", "Industries", "Industries", "Industries"],
            "Sector": ["Agriculture", "Manufacturing", "Energy", "Services"],
            "EI": [1.0, 2.0, 3.0, 4.0],
            "green_capability_export_share": [0.9, 0.1, 0.8, 0.2],
            "X_observed": [100.0, 100.0, 100.0, 100.0],
            "K": [110.0, 110.0, 110.0, 110.0],
        }
    )


def year_data() -> LeontiefYearData:
    node_labels = labels()
    label_frame = input_panel()[["country_sector", "Country", "Country_detail", "Category", "Sector"]].copy()
    x = pd.Series([100.0, 100.0, 100.0, 100.0], index=node_labels, name="X_observed")
    y = pd.Series([10.0, 20.0, 30.0, 40.0], index=node_labels, name="Y_final_demand")
    return LeontiefYearData(
        year=1995,
        labels=label_frame,
        X_observed=x,
        Y_final_demand=y,
        A=sparse.csr_matrix(np.eye(4) * 0.1),
        mode="transpose_row_output_fd_without_inventory",
        input_panel_orientation="transpose_row_fd_without_inventory",
    )


def context(name: str = "test_scenario") -> BehaviouralScenarioContext:
    return BehaviouralScenarioContext(
        year=1995,
        scenario_name=name,
        mode="transpose_row_output_fd_without_inventory",
        input_panel_orientation="transpose_row_fd_without_inventory",
        shock_size=0.10,
        selector_name="low_EI",
        notes="test",
    )


def capacity() -> pd.Series:
    return pd.Series([110.0, 110.0, 110.0, 110.0], index=labels(), name="K")


def test_green_node_selector_distinguishes_green_dimensions() -> None:
    diagnostics = GreenNodeSelector(input_panel(), 1995).build_diagnostics()

    by_country = diagnostics.set_index("Country")
    assert bool(by_country.loc["AAA", "is_low_EI"])
    assert bool(by_country.loc["AAA", "is_high_green_capability_export_share"])
    assert bool(by_country.loc["AAA", "is_clean_and_capable"])
    assert bool(by_country.loc["CCC", "is_high_EI"])
    assert bool(by_country.loc["CCC", "is_high_green_capability_export_share"])
    assert bool(by_country.loc["CCC", "is_transition_pivot"])


def test_final_demand_shock_modifies_only_selected_nodes() -> None:
    shock = FinalDemandShock(name="test", selector_name="low_EI", shock_size=0.10)

    shocked_year_data, unchanged_capacity, selected = shock.apply(year_data(), capacity(), input_panel(), context())

    assert unchanged_capacity.equals(capacity())
    assert selected["country_sector"].tolist() == [labels()[0]]
    assert shocked_year_data.Y_final_demand.loc[labels()[0]] == 11.0
    assert shocked_year_data.Y_final_demand.loc[labels()[1]] == 20.0


def test_capacity_shock_modifies_only_selected_nodes_and_clips_at_zero() -> None:
    shock = CapacityShock(name="test", selector_name="high_EI", shock_size=-2.0)
    original_year_data = year_data()

    unchanged_year_data, shocked_capacity, selected = shock.apply(original_year_data, capacity(), input_panel(), context())

    assert unchanged_year_data is original_year_data
    assert set(selected["country_sector"]) == {labels()[2], labels()[3]}
    assert shocked_capacity.loc[labels()[2]] == 0.0
    assert shocked_capacity.loc[labels()[3]] == 0.0
    assert shocked_capacity.loc[labels()[0]] == 110.0


def test_scenario_comparison_computes_deltas_and_percent_deltas() -> None:
    runner = BehaviouralLeontiefScenarioRunner(
        paths=ABMV3Paths(project_root=Path("tmp") / "scenario_test" / uuid4().hex[:8]),
        config=ABMV3Config(),
    )
    baseline = input_panel()[["Year", "country_sector", "Country", "Country_detail", "Category", "Sector"]].copy()
    baseline["X_realized"] = [100.0, 100.0, 100.0, 100.0]
    baseline["X_desired"] = [120.0, 120.0, 120.0, 120.0]
    baseline["output_ratio"] = [1.0, 1.0, 1.0, 1.0]
    baseline["capacity_binding_rounds"] = [0, 1, 0, 1]
    baseline["capacity_missing_rounds"] = [0, 0, 0, 0]
    scenario = baseline.copy()
    scenario["X_realized"] = [110.0, 100.0, 90.0, 100.0]
    scenario["X_desired"] = [132.0, 120.0, 108.0, 120.0]
    scenario["capacity_binding_rounds"] = [1, 1, 2, 1]
    selected = pd.DataFrame({"country_sector": [labels()[0]]})

    comparison = runner.build_node_comparison(baseline, scenario, selected, "test", "low_EI")

    first = comparison.loc[comparison["country_sector"].eq(labels()[0])].iloc[0]
    assert first["delta_X_realized"] == 10.0
    assert first["pct_delta_X_realized"] == 0.10
    assert first["delta_X_desired"] == 12.0
    assert first["delta_capacity_binding_rounds"] == 1


def test_registry_returns_initial_behavioural_scenarios() -> None:
    expected = {
        "low_ei_node_demand_expansion_10",
        "green_capability_node_demand_expansion_10",
        "clean_and_capable_node_demand_expansion_10",
        "transition_pivot_node_demand_expansion_10",
        "high_ei_node_capacity_bottleneck_10",
    }

    assert set(list_behavioural_scenarios()) == expected
    assert get_behavioural_scenario("low_ei_node_demand_expansion_10").selector_name == "low_EI"


def test_behavioural_scenario_cli_commands_are_registered() -> None:
    parser = build_parser()

    scenario_args = parser.parse_args(
        ["behavioural-scenario", "--year", "2016", "--scenario", "green_capability_node_demand_expansion_10"]
    )
    range_args = parser.parse_args(
        [
            "behavioural-scenario-range",
            "--start-year",
            "1995",
            "--end-year",
            "2016",
            "--scenario",
            "low_ei_node_demand_expansion_10",
        ]
    )
    list_args = parser.parse_args(["list-behavioural-scenarios"])
    assert scenario_args.command == "behavioural-scenario"
    assert range_args.command == "behavioural-scenario-range"
    assert list_args.command == "list-behavioural-scenarios"
