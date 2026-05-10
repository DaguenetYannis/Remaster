from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.abm_v3.paths import ABMV3Paths
from src.abm_v3.runner import build_parser
from src.abm_v3.validation_report import ABMV3ValidationReportBuilder


def toy_paths() -> ABMV3Paths:
    root = Path("tmp") / "abm_v3_validation_report_tests" / uuid4().hex[:8]
    root.mkdir(parents=True, exist_ok=True)
    return ABMV3Paths(project_root=root)


def write_required_diagnostics(paths: ABMV3Paths, ratio: float = 0.95) -> None:
    start_year = 1995
    end_year = 2016
    paths.abm_v3_diagnostics_dir.mkdir(parents=True, exist_ok=True)
    paths.leontief_behavioural_summary_diagnostics_dir.mkdir(parents=True, exist_ok=True)
    paths.leontief_behavioural_node_comparison_diagnostics_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "Year": 1995,
                "row_count": 2,
                "missing_EI_count": 0,
                "negative_EI_count": 0,
                "unmatched_merged_labels_count": 0,
                "inventory_excluded_column_count": 1,
                "negative_Y_no_inventory_count": 0,
                "negative_FD_no_inventory_entries": 0,
            },
            {
                "Year": 1996,
                "row_count": 2,
                "missing_EI_count": 0,
                "negative_EI_count": 0,
                "unmatched_merged_labels_count": 0,
                "inventory_excluded_column_count": 1,
                "negative_Y_no_inventory_count": 0,
                "negative_FD_no_inventory_entries": 0,
            },
        ]
    ).to_csv(paths.corrected_input_panel_build_report_path(start_year, end_year), index=False)

    pd.DataFrame(
        {
            "check": ["abm_ready_input_panel_selected", "required_columns_present"],
            "passed": [True, True],
        }
    ).to_csv(paths.corrected_real_data_smoke_test_path(start_year, end_year), index=False)

    pd.DataFrame(
        [
            {
                "Year": 1995,
                "correlation_old_X_corrected_X": 0.8,
                "mean_absolute_percentage_difference_X": 0.2,
                "median_absolute_percentage_difference_X": 0.1,
            }
        ]
    ).to_csv(paths.input_panel_orientation_comparison_path(start_year, end_year), index=False)

    pd.DataFrame(columns=["Year", "country_sector", "Country", "Sector", "X_observed", "EI", "emissions_observed"]).to_csv(
        paths.corrected_negative_ei_rows_path(start_year, end_year),
        index=False,
    )

    observed_total = 100.0
    realized_total = observed_total * ratio
    pd.DataFrame(
        [
            {
                "Year": 1995,
                "converged": True,
                "observed_output_total": observed_total,
                "realized_output_total": realized_total,
                "relative_error_total": abs(1.0 - ratio),
                "correlation_realized_vs_observed": 0.99,
                "median_absolute_percentage_error": abs(1.0 - ratio),
                "final_residual_share": 0.0,
                "mean_capacity_stress_over_rounds": 1.0,
                "mean_share_capacity_binding": 0.0,
                "mean_share_capacity_missing": 0.0,
            },
            {
                "Year": 1996,
                "converged": True,
                "observed_output_total": observed_total,
                "realized_output_total": realized_total,
                "relative_error_total": abs(1.0 - ratio),
                "correlation_realized_vs_observed": 0.98,
                "median_absolute_percentage_error": abs(1.0 - ratio),
                "final_residual_share": 0.0,
                "mean_capacity_stress_over_rounds": 1.0,
                "mean_share_capacity_binding": 0.0,
                "mean_share_capacity_missing": 0.0,
            },
        ]
    ).to_csv(paths.behavioural_leontief_summary_range_path(start_year, end_year), index=False)

    pd.DataFrame(
        [
            {
                "Year": 1995,
                "country_sector": "AAA | AAA | Industries | Agriculture",
                "Country": "AAA",
                "Country_detail": "AAA",
                "Category": "Industries",
                "Sector": "Agriculture",
                "X_observed": 70.0,
                "X_realized": 60.0,
                "X_desired": 75.0,
                "output_gap": -10.0,
                "output_ratio": 60.0 / 70.0,
                "absolute_error": 10.0,
                "absolute_percentage_error": 10.0 / 70.0,
            },
            {
                "Year": 1995,
                "country_sector": "BBB | BBB | Industries | Manufacturing",
                "Country": "BBB",
                "Country_detail": "BBB",
                "Category": "Industries",
                "Sector": "Manufacturing",
                "X_observed": 30.0,
                "X_realized": 10.0,
                "X_desired": 35.0,
                "output_gap": -20.0,
                "output_ratio": 10.0 / 30.0,
                "absolute_error": 20.0,
                "absolute_percentage_error": 20.0 / 30.0,
            },
        ]
    ).to_csv(paths.behavioural_leontief_node_comparison_range_path(start_year, end_year), index=False)


def write_ei_diagnostics(paths: ABMV3Paths) -> None:
    paths.ei_transition_diagnostics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "total_rows": 100,
                "included_rows": 80,
                "excluded_rows": 20,
                "included_share": 0.8,
            }
        ]
    ).to_csv(paths.ei_transition_sample_report_path(1995, 2016), index=False)
    pd.DataFrame(
        [
            {"model_name": "economic_only", "rmse": 1.0, "mae": 0.8, "r2": 0.2, "correlation_predicted_observed": 0.5},
            {"model_name": "green_transition", "rmse": 0.9, "mae": 0.7, "r2": 0.3, "correlation_predicted_observed": 0.6},
            {"model_name": "network_robustness", "rmse": 1.1, "mae": 0.9, "r2": 0.1, "correlation_predicted_observed": 0.4},
        ]
    ).to_csv(paths.ei_transition_model_scores_path(1995, 2016), index=False)


def write_weak_ei_diagnostics_without_included_share(paths: ABMV3Paths) -> None:
    paths.ei_transition_diagnostics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "total_rows": 100,
                "included_rows": 40,
                "excluded_rows": 60,
            }
        ]
    ).to_csv(paths.ei_transition_sample_report_path(1995, 2016), index=False)
    pd.DataFrame(
        [
            {"model_name": "economic_only", "rmse": 1.0, "mae": 0.8, "r2": 0.3, "correlation_predicted_observed": 0.5},
            {"model_name": "green_transition", "rmse": 1.1, "mae": 0.9, "r2": 0.2, "correlation_predicted_observed": 0.4},
        ]
    ).to_csv(paths.ei_transition_model_scores_path(1995, 2016), index=False)


def test_builder_writes_all_five_outputs() -> None:
    paths = toy_paths()
    write_required_diagnostics(paths)
    write_ei_diagnostics(paths)

    written_paths = ABMV3ValidationReportBuilder(paths).build(1995, 2016)

    assert set(written_paths) == {"summary", "by_year", "flags", "top_output_errors", "markdown"}
    assert all(path.exists() for path in written_paths.values())


def test_missing_optional_ei_files_do_not_crash_and_create_warning_flags() -> None:
    paths = toy_paths()
    write_required_diagnostics(paths)

    written_paths = ABMV3ValidationReportBuilder(paths).build(1995, 2016)

    flags = pd.read_csv(written_paths["flags"])
    assert "warning" in flags.loc[flags["area"].eq("ei_transition"), "severity"].tolist()


def test_systematic_underproduction_is_detected() -> None:
    paths = toy_paths()
    write_required_diagnostics(paths, ratio=0.75)
    write_ei_diagnostics(paths)

    written_paths = ABMV3ValidationReportBuilder(paths).build(1995, 2016)

    flags = pd.read_csv(written_paths["flags"])
    assert flags["flag"].str.contains("Systematic underproduction", case=False).any()


def test_top_output_errors_are_sorted_by_absolute_error() -> None:
    paths = toy_paths()
    write_required_diagnostics(paths)
    write_ei_diagnostics(paths)

    written_paths = ABMV3ValidationReportBuilder(paths).build(1995, 2016)

    top_errors = pd.read_csv(written_paths["top_output_errors"])
    assert top_errors["absolute_error"].tolist() == sorted(top_errors["absolute_error"], reverse=True)
    assert top_errors["country_sector"].iloc[0].startswith("BBB")


def test_validation_report_cli_command_is_registered() -> None:
    parser = build_parser()

    args = parser.parse_args(["validation-report", "--start-year", "1995", "--end-year", "2016"])

    assert args.command == "validation-report"
    assert args.start_year == 1995
    assert args.end_year == 2016


def test_markdown_computes_missing_included_share_and_keeps_weak_ei_diagnostic_only() -> None:
    paths = toy_paths()
    write_required_diagnostics(paths)
    write_weak_ei_diagnostics_without_included_share(paths)

    written_paths = ABMV3ValidationReportBuilder(paths).build(1995, 2016)

    markdown = written_paths["markdown"].read_text(encoding="utf-8")
    summary = pd.read_csv(written_paths["summary"])
    assert "included_share=0.4" in markdown
    assert summary.loc[0, "ei_transition_status"] == "diagnostic_only"
    assert summary.loc[0, "overall_status"] == "production_ready_ei_pending"
