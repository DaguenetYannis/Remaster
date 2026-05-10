from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.abm_v3.leontief.scenarios.analysis_report import BehaviouralScenarioAnalysisReportBuilder
from src.abm_v3.leontief.scenarios.plots import (
    clean_display_label,
    display_label_cleaner,
    plot_scenario_output_effect,
    plot_selector_overlap,
    plot_top_sector_effects,
)
from src.abm_v3.paths import ABMV3Paths
from src.abm_v3.runner import build_parser


SCENARIOS = [
    "low_ei_node_demand_expansion_10",
    "green_capability_node_demand_expansion_10",
    "clean_and_capable_node_demand_expansion_10",
    "transition_pivot_node_demand_expansion_10",
    "high_ei_node_capacity_bottleneck_10",
]


def toy_paths() -> ABMV3Paths:
    root = Path("tmp") / "scenario_analysis_report_tests" / uuid4().hex[:8]
    root.mkdir(parents=True, exist_ok=True)
    return ABMV3Paths(project_root=root)


def write_synthetic_outputs(paths: ABMV3Paths, start_year: int = 1995, end_year: int = 1996, skip: tuple[str, int] | None = None) -> None:
    paths.behavioural_leontief_scenario_diagnostics_dir.mkdir(parents=True, exist_ok=True)
    paths.behavioural_leontief_scenario_selected_nodes_dir.mkdir(parents=True, exist_ok=True)
    for scenario_index, scenario in enumerate(SCENARIOS):
        for year in range(start_year, end_year + 1):
            if skip == (scenario, year):
                continue
            effect = 0.01 * (scenario_index + 1)
            if "capacity_bottleneck" in scenario:
                effect = 0.0
            summary = pd.DataFrame(
                [
                    {
                        "Year": year,
                        "scenario_name": scenario,
                        "selected_node_count": 2 + scenario_index,
                        "total_node_count": 10,
                        "pct_delta_realized_output_total": effect,
                        "delta_realized_output_total": effect * 1000,
                        "baseline_realized_output_total": 1000.0,
                        "scenario_realized_output_total": 1000.0 + effect * 1000,
                        "pct_delta_desired_output_total": effect * 1.1,
                        "baseline_converged": True,
                        "scenario_converged": True,
                        "baseline_final_residual_share": 0.0,
                        "scenario_final_residual_share": 0.0,
                    }
                ]
            )
            summary.to_csv(paths.behavioural_leontief_scenario_summary_path(year, scenario, "transpose_row_output_fd_without_inventory", "transpose_row_fd_without_inventory"), index=False)
            aggregate_rows = []
            for level, keys in {"Sector": ["Agriculture", "Manufacturing"], "Country": ["AAA", "BBB"]}.items():
                for key_index, key in enumerate(keys):
                    aggregate_rows.append(
                        {
                            "Year": year,
                            "scenario_name": scenario,
                            "aggregation_level": level,
                            "aggregation_key": key,
                            "X_realized_baseline_sum": 100.0,
                            "X_realized_scenario_sum": 100.0 + effect * 100 * (key_index + 1),
                            "delta_X_realized_sum": effect * 100 * (key_index + 1),
                            "pct_delta_X_realized_sum": effect * (key_index + 1),
                            "X_desired_baseline_sum": 120.0,
                            "X_desired_scenario_sum": 120.0 + effect * 120,
                            "delta_X_desired_sum": effect * 120,
                            "selected_node_count": key_index + 1,
                            "total_node_count": 5,
                        }
                    )
            pd.DataFrame(aggregate_rows).to_csv(paths.behavioural_leontief_scenario_aggregate_path(year, scenario, "transpose_row_output_fd_without_inventory", "transpose_row_fd_without_inventory"), index=False)
            selected = pd.DataFrame(
                [
                    {
                        "Year": year,
                        "scenario_name": scenario,
                        "country_sector": f"AAA | AAA | Industries | Agriculture {year}",
                        "is_low_EI": True,
                        "is_high_EI": False,
                        "is_high_green_capability_export_share": True,
                        "is_clean_and_capable": True,
                        "is_transition_pivot": False,
                        "EI": 1.0,
                        "green_capability_metric_value": 0.9,
                        "green_capability_metric_used": "green_capability_export_share",
                    },
                    {
                        "Year": year,
                        "scenario_name": scenario,
                        "country_sector": f"BBB | BBB | Industries | Manufacturing {year}",
                        "is_low_EI": False,
                        "is_high_EI": True,
                        "is_high_green_capability_export_share": True,
                        "is_clean_and_capable": False,
                        "is_transition_pivot": True,
                        "EI": 4.0,
                        "green_capability_metric_value": 0.8,
                        "green_capability_metric_used": "green_capability_export_share",
                    },
                ]
            )
            selected.to_csv(paths.behavioural_leontief_scenario_selected_nodes_path(year, scenario), index=False)


def test_builder_reads_yearly_files_and_writes_outputs() -> None:
    paths = toy_paths()
    write_synthetic_outputs(paths)

    written = BehaviouralScenarioAnalysisReportBuilder(paths, make_plots=False).build(1995, 1996)

    assert {"summary", "by_year", "selector_overlap", "sector_effects", "country_effects", "flags", "markdown"}.issubset(written)
    assert all(path.exists() for path in written.values())


def test_missing_scenario_year_creates_completeness_flag() -> None:
    paths = toy_paths()
    write_synthetic_outputs(paths, skip=("low_ei_node_demand_expansion_10", 1996))

    written = BehaviouralScenarioAnalysisReportBuilder(paths, make_plots=False).build(1995, 1996)
    flags = pd.read_csv(written["flags"])

    assert flags["flag"].str.contains("Missing scenario-year").any()


def test_selector_overlap_shares_are_computed() -> None:
    paths = toy_paths()
    write_synthetic_outputs(paths)

    written = BehaviouralScenarioAnalysisReportBuilder(paths, make_plots=False).build(1995, 1996)
    overlap = pd.read_csv(written["selector_overlap"])

    first = overlap.iloc[0]
    assert first["low_EI_share"] == 0.5
    assert first["high_EI_share"] == 0.5
    assert first["high_green_capability_share"] == 1.0
    assert first["clean_and_capable_share"] == 0.5
    assert first["transition_pivot_share"] == 0.5


def test_sector_and_country_rankings_are_written() -> None:
    paths = toy_paths()
    write_synthetic_outputs(paths)

    written = BehaviouralScenarioAnalysisReportBuilder(paths, make_plots=False).build(1995, 1996)
    sectors = pd.read_csv(written["sector_effects"])
    countries = pd.read_csv(written["country_effects"])

    assert "rank_within_scenario_by_abs_effect" in sectors.columns
    assert "rank_within_scenario_by_abs_effect" in countries.columns


def test_capacity_bottleneck_near_zero_flag_and_demand_classification() -> None:
    paths = toy_paths()
    write_synthetic_outputs(paths)

    written = BehaviouralScenarioAnalysisReportBuilder(paths, make_plots=False).build(1995, 1996)
    summary = pd.read_csv(written["summary"])
    flags = pd.read_csv(written["flags"])

    assert summary.loc[summary["scenario_name"].eq("low_ei_node_demand_expansion_10"), "scenario_type"].iloc[0] == "demand_expansion"
    assert flags["flag"].str.contains("near-zero effect").any()


def test_plot_functions_save_validate_and_do_not_mutate() -> None:
    paths = toy_paths()
    write_synthetic_outputs(paths)
    written = BehaviouralScenarioAnalysisReportBuilder(paths, make_plots=False).build(1995, 1996)
    summary = pd.read_csv(written["summary"])
    overlap = pd.read_csv(written["selector_overlap"])
    summary_before = summary.copy(deep=True)
    output_path = paths.behavioural_leontief_scenario_plot_dir / "test_plot.png"

    fig = plot_scenario_output_effect(summary, output_path=output_path)
    assert fig is not None
    assert output_path.exists()
    pd.testing.assert_frame_equal(summary, summary_before)
    try:
        plot_selector_overlap(overlap, audience="invalid")
        raise AssertionError("Invalid audience should fail.")
    except ValueError:
        pass
    try:
        plot_selector_overlap(overlap, color_mode="invalid")
        raise AssertionError("Invalid color mode should fail.")
    except ValueError:
        pass


def test_display_label_cleaning_is_display_only() -> None:
    assert clean_display_label("Finacial Intermediation and Business Activities") == "Financial Intermediation and Business Activities"
    assert clean_display_label("Hotels and Restraurants") == "Hotels and Restaurants"
    assert clean_display_label("green capabiliity metric") == "green capability metric"
    assert clean_display_label("high_ei_node_capacity_bottleneck_10") == "High EI capacity bottleneck"
    assert clean_display_label("capacity bottelneck stress") == "capacity bottleneck stress"
    assert display_label_cleaner("capacity botleneck stress") == "capacity bottleneck stress"


def test_top_effect_plot_uses_billions_without_mutating() -> None:
    sector = pd.DataFrame(
        {
            "scenario_name": ["green_capability_node_demand_expansion_10"],
            "Sector": ["Finacial Intermediation and Business Activities"],
            "total_delta_X_realized_sum": [2_000_000_000.0],
        }
    )
    before = sector.copy(deep=True)
    output_path = toy_paths().behavioural_leontief_scenario_plot_dir / "sector_billions.png"

    fig = plot_top_sector_effects(
        sector,
        "green_capability_node_demand_expansion_10",
        output_path=output_path,
    )

    assert fig is not None
    assert output_path.exists()
    assert fig.axes[0].get_xlabel() == "Total change in realized output, billions"
    tick_labels = [tick.get_text() for tick in fig.axes[0].get_yticklabels()]
    assert "Financial Intermediation and Business Activities" in tick_labels
    pd.testing.assert_frame_equal(sector, before)


def test_report_writes_portfolio_heatmap_and_capacity_diagnostic() -> None:
    paths = toy_paths()
    write_synthetic_outputs(paths)

    written = BehaviouralScenarioAnalysisReportBuilder(paths, make_plots=True).build(1995, 1996)

    assert "plot_selector_overlap_heatmap_portfolio" in written
    assert "plot_selector_overlap_portfolio" not in written
    assert "plot_capacity_bottleneck_diagnostic_research" in written
    assert written["plot_selector_overlap_heatmap_portfolio"].exists()
    assert written["plot_capacity_bottleneck_diagnostic_research"].exists()


def test_scenario_plot_uses_clean_human_scenario_labels() -> None:
    summary = pd.DataFrame(
        {
            "scenario_name": ["green_capability_node_demand_expansion_10", "high_ei_node_capacity_bottleneck_10"],
            "mean_pct_delta_realized_output_total": [0.02, 0.0],
        }
    )

    fig = plot_scenario_output_effect(summary)

    labels = [tick.get_text() for tick in fig.axes[0].get_yticklabels()]
    assert "Green capability demand" in labels
    assert "High EI capacity bottleneck" in labels


def test_cli_parser_includes_behavioural_scenario_report() -> None:
    parser = build_parser()
    args = parser.parse_args(["behavioural-scenario-report", "--start-year", "1995", "--end-year", "2016"])
    assert args.command == "behavioural-scenario-report"
