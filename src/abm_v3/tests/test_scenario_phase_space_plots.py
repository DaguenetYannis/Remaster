from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.abm_v3.leontief.scenarios.phase_space_plots import (
    ScenarioPhaseSpacePlotBuilder,
    build_delta_summary,
    build_endpoint_summary,
    build_scenario_time_series,
    build_scenario_node_panel,
    discover_scenario_phase_space_sources,
    resolve_reference_scenario,
)
from src.abm_v3.leontief.scenarios.plots import clean_display_label
from src.abm_v3.paths import ABMV3Paths
from src.abm_v3.runner import build_parser


def toy_paths() -> ABMV3Paths:
    root = Path("tmp") / "scenario_phase_space_tests" / uuid4().hex[:8]
    root.mkdir(parents=True, exist_ok=True)
    return ABMV3Paths(project_root=root)


def write_synthetic_sources(paths: ABMV3Paths, start_year: int = 1995, end_year: int = 1996) -> Path:
    paths.behavioural_leontief_scenario_analysis_tables_dir.mkdir(parents=True, exist_ok=True)
    paths.behavioural_leontief_scenario_diagnostics_dir.mkdir(parents=True, exist_ok=True)
    state_path = paths.abm_v3_output_root / "phase_space" / f"abm_v3_phase_space_state_panel_{start_year}_{end_year}.parquet"
    state_path.parent.mkdir(parents=True, exist_ok=True)

    scenarios = ["green_capability_node_demand_expansion_10", "high_ei_node_capacity_bottleneck_10"]
    summary = pd.DataFrame(
        {
            "scenario_name": scenarios,
            "scenario_type": ["demand_expansion", "capacity_bottleneck"],
            "mean_pct_delta_realized_output_total": [0.02, 0.0],
        }
    )
    summary.to_csv(paths.behavioural_scenario_analysis_summary_path(start_year, end_year), index=False)
    by_year_rows = []
    state_rows = []
    node_rows = []
    for year in range(start_year, end_year + 1):
        for node_index in range(4):
            state_rows.append(
                {
                    "country_sector": f"C{node_index} | Sector {node_index}",
                    "Year": year,
                    "Country": f"C{node_index}",
                    "Sector": "Finacial Services" if node_index == 0 else f"Sector {node_index}",
                    "green_capability_export_share": 0.2 + 0.1 * node_index + 0.01 * (year - start_year),
                    "g_local": 0.4 + 0.05 * node_index + 0.01 * (year - start_year),
                    "g_in_network": 0.3 + 0.02 * node_index,
                    "g_out_network": 0.35 + 0.03 * node_index,
                    "log_X_observed": 4.0 + node_index,
                    "X_observed": 100.0 + node_index * 10,
                    "emissions_observed": 40.0 + node_index * 5,
                    "EI": 0.3 + node_index * 0.02,
                    "is_top25_by_output_over_period": True,
                    "is_top25_by_emissions_over_period": True,
                }
            )
        for scenario in scenarios:
            by_year_rows.append(
                {
                    "Year": year,
                    "scenario_name": scenario,
                    "scenario_realized_output_total": 500.0,
                    "delta_realized_output_total": 10.0 if "green" in scenario else 0.0,
                    "pct_delta_realized_output_total": 0.02 if "green" in scenario else 0.0,
                }
            )
            node_frame = pd.DataFrame(
                {
                    "Year": [year] * 4,
                    "scenario_name": [scenario] * 4,
                    "selector_name": ["selector"] * 4,
                    "country_sector": [f"C{i} | Sector {i}" for i in range(4)],
                    "Country": [f"C{i}" for i in range(4)],
                    "Sector": ["Finacial Services", "Sector 1", "Sector 2", "Sector 3"],
                    "X_realized_baseline": [100.0, 110.0, 120.0, 130.0],
                    "X_realized_scenario": [110.0, 121.0, 132.0, 143.0] if "green" in scenario else [100.0, 110.0, 120.0, 130.0],
                    "delta_X_realized": [10.0, 11.0, 12.0, 13.0] if "green" in scenario else [0.0, 0.0, 0.0, 0.0],
                    "pct_delta_X_realized": [0.1, 0.1, 0.1, 0.1] if "green" in scenario else [0.0, 0.0, 0.0, 0.0],
                    "is_selected_node": [True, False, False, False],
                }
            )
            node_path = paths.behavioural_leontief_scenario_diagnostics_dir / f"node_comparison_{scenario}_{year}_toy.csv"
            node_frame.to_csv(node_path, index=False)
            node_rows.append(node_frame)
    pd.DataFrame(by_year_rows).to_csv(paths.behavioural_scenario_analysis_by_year_path(start_year, end_year), index=False)
    pd.DataFrame({"scenario_name": scenarios, "Sector": ["Sector 1", "Sector 2"], "total_delta_X_realized_sum": [10.0, 0.0]}).to_csv(
        paths.behavioural_scenario_analysis_sector_effects_path(start_year, end_year),
        index=False,
    )
    pd.DataFrame({"scenario_name": scenarios, "Country": ["C1", "C2"], "total_delta_X_realized_sum": [10.0, 0.0]}).to_csv(
        paths.behavioural_scenario_analysis_country_effects_path(start_year, end_year),
        index=False,
    )
    pd.DataFrame(state_rows).to_parquet(state_path, index=False)
    return state_path


def test_scenario_source_discovery_and_reference_fallback() -> None:
    paths = toy_paths()
    state_path = write_synthetic_sources(paths)

    sources = discover_scenario_phase_space_sources(paths, 1995, 1996, state_path)

    assert sources.scenario_summary.exists()
    assert sources.scenario_by_year.exists()
    assert sources.historical_state_panel == state_path
    assert resolve_reference_scenario(["scenario_a"], "historical_or_baseline") == "historical_endpoint_reference"
    assert resolve_reference_scenario(["baseline_scenario"], "historical_or_baseline") == "baseline_scenario"


def test_endpoint_and_delta_summary_construction() -> None:
    paths = toy_paths()
    state_path = write_synthetic_sources(paths)
    diagnostics: list[dict[str, object]] = []
    state = pd.read_parquet(state_path)
    nodes = build_scenario_node_panel(paths.behavioural_leontief_scenario_diagnostics_dir, state, [], 1995, 1996, diagnostics)
    by_year = pd.read_csv(paths.behavioural_scenario_analysis_by_year_path(1995, 1996))

    time_series = build_scenario_time_series(nodes, by_year)
    endpoint = build_endpoint_summary(time_series, 1996, "historical_endpoint_reference")
    delta = build_delta_summary(endpoint, "historical_endpoint_reference")

    assert {"scenario_name", "g_local", "g_in_network", "total_realized_output"}.issubset(endpoint.columns)
    assert {"scenario_name", "variable", "delta", "reference_scenario"}.issubset(delta.columns)
    assert delta["reference_scenario"].eq("historical_endpoint_reference").all()


def test_builder_no_plots_writes_required_tables_and_empty_manifest() -> None:
    paths = toy_paths()
    state_path = write_synthetic_sources(paths)

    written = ScenarioPhaseSpacePlotBuilder(paths=paths, state_panel=state_path, make_plots=False).build(1995, 1996)

    assert written["endpoint_summary"].exists()
    assert written["delta_summary"].exists()
    assert written["time_series"].exists()
    assert written["sector_summary"].exists()
    assert written["node_summary"].exists()
    assert written["diagnostics"].exists()
    assert written["manifest"].exists()
    manifest = pd.read_csv(written["manifest"])
    assert {
        "figure_path",
        "figure_name",
        "figure_family",
        "recommendation_status",
        "title",
        "caption",
        "axis_x",
        "axis_y",
        "reference_scenario",
    }.issubset(manifest.columns)


def test_builder_with_plots_writes_manifest_and_recommendations() -> None:
    paths = toy_paths()
    state_path = write_synthetic_sources(paths)

    written = ScenarioPhaseSpacePlotBuilder(paths=paths, state_panel=state_path, make_plots=True).build(1995, 1996)
    manifest = pd.read_csv(written["manifest"])
    recommendations = pd.read_csv(written["figure_recommendations"])

    assert not manifest.empty
    assert manifest["figure_family"].str.contains("global_overlay").any()
    assert {"plot_file", "figure_tier", "recommendation_status", "title"}.issubset(recommendations.columns)
    assert recommendations["figure_tier"].eq("thesis-core").any()


def test_missing_variable_diagnostics_are_written() -> None:
    paths = toy_paths()
    state_path = write_synthetic_sources(paths)
    state = pd.read_parquet(state_path).drop(columns=["g_local"])
    state.to_parquet(state_path, index=False)

    written = ScenarioPhaseSpacePlotBuilder(paths=paths, state_panel=state_path, make_plots=False).build(1995, 1996)
    diagnostics = pd.read_csv(written["diagnostics"])

    assert diagnostics["diagnostic_type"].str.contains("join_warning|missing", case=False, na=False).any()


def test_display_cleaning_cli_registration_and_defaults() -> None:
    assert clean_display_label("Finacial Services and Hotels and Restraurants") == "Financial Services and Hotels and Restaurants"
    parser = build_parser()

    args = parser.parse_args(["scenario-phase-space-plots", "--start-year", "1995", "--end-year", "2016", "--no-plots"])

    assert args.command == "scenario-phase-space-plots"
    assert args.reference_scenario == "historical_or_baseline"
    assert args.top_sector_n == 8
    assert args.top_node_n == 10
    assert args.research_top_node_n == 25
    assert args.no_plots
