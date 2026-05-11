from __future__ import annotations

import argparse
import logging
from dataclasses import replace

import numpy as np
import pandas as pd

from src.abm_v3.config import ABMV3Config
from src.abm_v3.data_inventory import build_data_inventory
from src.abm_v3.data_loader import ABMV3DataLoader
from src.abm_v3.diagnostics.hypothesis_reports import HypothesisReportGenerator
from src.abm_v3.input_panel_builder import ABMV3InputPanelBuilder, CorrectedOrientationInputPanelBuilder
from src.abm_v3.ei_transition.models import EITransitionModelSuite
from src.abm_v3.ei_transition.outputs import EITransitionOutputWriter
from src.abm_v3.ei_transition.panel import EITransitionPanelBuilder
from src.abm_v3.ei_transition.validation import validate_transition_split
from src.abm_v3.leontief.behavioural import (
    BehaviouralLeontiefEngine,
    BehaviouralLeontiefOutputWriter,
    BehaviouralLeontiefValidator,
)
from src.abm_v3.leontief.coefficients import LeontiefCoefficientBuilder, LeontiefYearData
from src.abm_v3.leontief.comparison import LeontiefModeComparator
from src.abm_v3.leontief.orientation import LeontiefOrientationAuditor
from src.abm_v3.leontief.outputs import LeontiefOutputWriter
from src.abm_v3.leontief.propagation import LeontiefPropagationEngine
from src.abm_v3.leontief.scenarios.analysis_report import BehaviouralScenarioAnalysisReportBuilder
from src.abm_v3.leontief.scenarios.phase_space_plots import ScenarioPhaseSpacePlotBuilder
from src.abm_v3.leontief.scenarios.registry import get_behavioural_scenario, list_behavioural_scenarios
from src.abm_v3.leontief.scenarios.runner import BehaviouralLeontiefScenarioRunner
from src.abm_v3.leontief.validation import LeontiefPropagationValidator
from src.abm_v3.leontief.viability import LeontiefViabilityAnalyzer
from src.abm_v3.model import ABMV3Model
from src.abm_v3.paths import ABMV3Paths
from src.abm_v3.phase_space.plots import PhaseSpacePlotBuilder
from src.abm_v3.phase_space.state_panel import ABMV3PhaseSpaceStatePanelBuilder
from src.abm_v3.real_data_smoke_test import RealDataSmokeTester
from src.abm_v3.scenarios.registry import list_scenarios
from src.abm_v3.validation_report import ABMV3ValidationReportBuilder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ABM v3 scaffold CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--start-year", type=int, default=1995)
    calibrate.add_argument("--end-year", type=int, default=2016)
    calibrate.add_argument("--ei-mode", default="green_transition")
    calibrate.add_argument("--validation-mode", default="rolling")

    validate = subparsers.add_parser("validate")
    validate.add_argument("--split-year", type=int, default=2008)

    simulate = subparsers.add_parser("simulate")
    simulate.add_argument("--scenario", default="baseline_continuation")
    simulate.add_argument("--start-year", type=int, default=2017)
    simulate.add_argument("--end-year", type=int, default=2027)

    subparsers.add_parser("compare-scenarios")

    smoke_test = subparsers.add_parser("smoke-test")
    smoke_test.add_argument("--no-write", action="store_true")
    smoke_group = smoke_test.add_mutually_exclusive_group()
    smoke_group.add_argument("--input-panel", action="store_true")
    smoke_group.add_argument("--merged-panel", action="store_true")

    fit_historical = subparsers.add_parser("fit-historical")
    fit_historical.add_argument("--start-year", type=int, default=1995)
    fit_historical.add_argument("--end-year", type=int, default=2016)
    fit_historical.add_argument("--ei-mode", default="green_transition")
    fit_historical.add_argument("--validation-mode", default="rolling")

    hypothesis_report = subparsers.add_parser("hypothesis-report")
    hypothesis_report.add_argument("--ei-mode", default="green_transition")

    build_input_panel = subparsers.add_parser("build-input-panel")
    build_input_panel.add_argument("--start-year", type=int, default=1995)
    build_input_panel.add_argument("--end-year", type=int, default=2016)
    build_input_panel.add_argument("--overwrite", action="store_true")

    build_corrected_input_panel = subparsers.add_parser("build-corrected-input-panel")
    build_corrected_input_panel.add_argument("--start-year", type=int, default=1995)
    build_corrected_input_panel.add_argument("--end-year", type=int, default=2016)
    build_corrected_input_panel.add_argument("--overwrite", action="store_true")
    build_corrected_input_panel.add_argument("--orientation", default="transpose_row_fd_without_inventory")
    build_corrected_input_panel.add_argument("--capacity-margin", type=float, default=None)
    build_corrected_input_panel.add_argument("--inventory-days", type=int, default=None)

    build_ei_transition_panel = subparsers.add_parser("build-ei-transition-panel")
    build_ei_transition_panel.add_argument("--start-year", type=int, default=1995)
    build_ei_transition_panel.add_argument("--end-year", type=int, default=2016)
    build_ei_transition_panel.add_argument("--overwrite", action="store_true")

    fit_ei_transition = subparsers.add_parser("fit-ei-transition")
    fit_ei_transition.add_argument("--start-year", type=int, default=1995)
    fit_ei_transition.add_argument("--end-year", type=int, default=2016)
    fit_ei_transition.add_argument("--train-end-year", type=int, default=2012)
    fit_ei_transition.add_argument("--validation-start-year", type=int, default=2013)
    fit_ei_transition.add_argument("--validation-end-year", type=int, default=2015)
    fit_ei_transition.add_argument("--overwrite-panel", action="store_true")

    smoke_corrected_input_panel = subparsers.add_parser("smoke-test-corrected-input-panel")
    smoke_corrected_input_panel.add_argument("--start-year", type=int, default=1995)
    smoke_corrected_input_panel.add_argument("--end-year", type=int, default=2016)
    smoke_corrected_input_panel.add_argument("--orientation", default="transpose_row_fd_without_inventory")

    leontief_propagate = subparsers.add_parser("leontief-propagate")
    leontief_propagate.add_argument("--year", type=int, required=True)
    leontief_propagate.add_argument("--mode", default="raw")
    leontief_propagate.add_argument("--tolerance", type=float, default=None)
    leontief_propagate.add_argument("--max-rounds", type=int, default=None)
    leontief_propagate.add_argument("--column-sum-cap", type=float, default=None)
    leontief_propagate.add_argument(
        "--input-panel-orientation",
        choices=["current_column", "transpose_row_fd_without_inventory"],
        default=None,
    )

    leontief_range = subparsers.add_parser("leontief-propagate-range")
    leontief_range.add_argument("--start-year", type=int, required=True)
    leontief_range.add_argument("--end-year", type=int, required=True)
    leontief_range.add_argument("--mode", default="raw")
    leontief_range.add_argument("--tolerance", type=float, default=None)
    leontief_range.add_argument("--max-rounds", type=int, default=None)
    leontief_range.add_argument("--column-sum-cap", type=float, default=None)
    leontief_range.add_argument(
        "--input-panel-orientation",
        choices=["current_column", "transpose_row_fd_without_inventory"],
        default=None,
    )

    leontief_diagnose = subparsers.add_parser("leontief-diagnose")
    leontief_diagnose.add_argument("--year", type=int, required=True)
    leontief_diagnose.add_argument("--mode", default="raw")
    leontief_diagnose.add_argument("--column-sum-cap", type=float, default=None)

    leontief_diagnose_range = subparsers.add_parser("leontief-diagnose-range")
    leontief_diagnose_range.add_argument("--start-year", type=int, required=True)
    leontief_diagnose_range.add_argument("--end-year", type=int, required=True)
    leontief_diagnose_range.add_argument("--mode", default="raw")
    leontief_diagnose_range.add_argument("--column-sum-cap", type=float, default=None)

    leontief_compare_modes = subparsers.add_parser("leontief-compare-modes")
    leontief_compare_modes.add_argument("--year", type=int, required=True)
    leontief_compare_modes.add_argument("--modes", nargs="+", default=None)
    leontief_compare_modes.add_argument("--tolerance", type=float, default=None)
    leontief_compare_modes.add_argument("--max-rounds", type=int, default=None)
    leontief_compare_modes.add_argument("--column-sum-cap", type=float, default=None)

    leontief_compare_modes_range = subparsers.add_parser("leontief-compare-modes-range")
    leontief_compare_modes_range.add_argument("--start-year", type=int, required=True)
    leontief_compare_modes_range.add_argument("--end-year", type=int, required=True)
    leontief_compare_modes_range.add_argument("--modes", nargs="+", default=None)
    leontief_compare_modes_range.add_argument("--tolerance", type=float, default=None)
    leontief_compare_modes_range.add_argument("--max-rounds", type=int, default=None)
    leontief_compare_modes_range.add_argument("--column-sum-cap", type=float, default=None)

    leontief_audit_orientation = subparsers.add_parser("leontief-audit-orientation")
    leontief_audit_orientation.add_argument("--year", type=int, required=True)
    leontief_audit_orientation.add_argument("--max-rounds", type=int, default=400)
    leontief_audit_orientation.add_argument("--tolerance", type=float, default=1e-8)
    leontief_audit_orientation.add_argument("--spectral-max-iter", type=int, default=None)
    leontief_audit_orientation.add_argument(
        "--include-fd-without-inventory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    leontief_audit_orientation.add_argument("--reference", choices=["abm_ready", "current"], default="abm_ready")

    leontief_audit_orientation_range = subparsers.add_parser("leontief-audit-orientation-range")
    leontief_audit_orientation_range.add_argument("--start-year", type=int, required=True)
    leontief_audit_orientation_range.add_argument("--end-year", type=int, required=True)
    leontief_audit_orientation_range.add_argument("--max-rounds", type=int, default=400)
    leontief_audit_orientation_range.add_argument("--tolerance", type=float, default=1e-8)
    leontief_audit_orientation_range.add_argument("--spectral-max-iter", type=int, default=None)
    leontief_audit_orientation_range.add_argument(
        "--include-fd-without-inventory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    leontief_audit_orientation_range.add_argument("--reference", choices=["abm_ready", "current"], default="abm_ready")

    behavioural_leontief = subparsers.add_parser("behavioural-leontief")
    behavioural_leontief.add_argument("--year", type=int, required=True)
    behavioural_leontief.add_argument("--mode", default=None)
    behavioural_leontief.add_argument("--eta-capacity", type=float, default=None)
    behavioural_leontief.add_argument("--tolerance", type=float, default=None)
    behavioural_leontief.add_argument("--max-rounds", type=int, default=None)
    behavioural_leontief.add_argument("--no-node-rounds", action="store_true")
    behavioural_leontief.add_argument(
        "--input-panel-orientation",
        choices=["current_column", "transpose_row_fd_without_inventory"],
        default=None,
    )

    behavioural_leontief_range = subparsers.add_parser("behavioural-leontief-range")
    behavioural_leontief_range.add_argument("--start-year", type=int, required=True)
    behavioural_leontief_range.add_argument("--end-year", type=int, required=True)
    behavioural_leontief_range.add_argument("--mode", default=None)
    behavioural_leontief_range.add_argument("--eta-capacity", type=float, default=None)
    behavioural_leontief_range.add_argument("--tolerance", type=float, default=None)
    behavioural_leontief_range.add_argument("--max-rounds", type=int, default=None)
    behavioural_leontief_range.add_argument("--no-node-rounds", action="store_true")
    behavioural_leontief_range.add_argument(
        "--input-panel-orientation",
        choices=["current_column", "transpose_row_fd_without_inventory"],
        default=None,
    )

    validation_report = subparsers.add_parser("validation-report")
    validation_report.add_argument("--start-year", type=int, default=1995)
    validation_report.add_argument("--end-year", type=int, default=2016)

    behavioural_scenario = subparsers.add_parser("behavioural-scenario")
    behavioural_scenario.add_argument("--year", type=int, required=True)
    behavioural_scenario.add_argument("--scenario", required=True)
    behavioural_scenario.add_argument("--mode", default="transpose_row_output_fd_without_inventory")
    behavioural_scenario.add_argument("--input-panel-orientation", default="transpose_row_fd_without_inventory")
    behavioural_scenario.add_argument("--shock-size", type=float, default=None)
    behavioural_scenario.add_argument("--selector", default=None)
    behavioural_scenario.add_argument("--low-ei-quantile", type=float, default=0.25)
    behavioural_scenario.add_argument("--high-ei-quantile", type=float, default=0.75)
    behavioural_scenario.add_argument("--high-capability-quantile", type=float, default=0.75)
    add_behavioural_scenario_runtime_arguments(behavioural_scenario)

    behavioural_scenario_range = subparsers.add_parser("behavioural-scenario-range")
    behavioural_scenario_range.add_argument("--start-year", type=int, required=True)
    behavioural_scenario_range.add_argument("--end-year", type=int, required=True)
    behavioural_scenario_range.add_argument("--scenario", required=True)
    behavioural_scenario_range.add_argument("--mode", default="transpose_row_output_fd_without_inventory")
    behavioural_scenario_range.add_argument("--input-panel-orientation", default="transpose_row_fd_without_inventory")
    behavioural_scenario_range.add_argument("--shock-size", type=float, default=None)
    behavioural_scenario_range.add_argument("--selector", default=None)
    behavioural_scenario_range.add_argument("--low-ei-quantile", type=float, default=0.25)
    behavioural_scenario_range.add_argument("--high-ei-quantile", type=float, default=0.75)
    behavioural_scenario_range.add_argument("--high-capability-quantile", type=float, default=0.75)
    add_behavioural_scenario_runtime_arguments(behavioural_scenario_range)

    subparsers.add_parser("list-behavioural-scenarios")

    behavioural_scenario_report = subparsers.add_parser("behavioural-scenario-report")
    behavioural_scenario_report.add_argument("--start-year", type=int, default=1995)
    behavioural_scenario_report.add_argument("--end-year", type=int, default=2016)
    behavioural_scenario_report.add_argument("--mode", default="transpose_row_output_fd_without_inventory")
    behavioural_scenario_report.add_argument("--input-panel-orientation", default="transpose_row_fd_without_inventory")
    behavioural_scenario_report.add_argument("--audience", choices=["portfolio", "research", "both"], default="both")
    behavioural_scenario_report.add_argument("--color-mode", choices=["default", "colorblind"], default="default")
    behavioural_scenario_report.add_argument("--no-plots", action="store_true")

    data_inventory = subparsers.add_parser("data-inventory")
    data_inventory.add_argument("--root", default="data")
    data_inventory.add_argument("--focus", choices=["all", "abm_v3"], default="abm_v3")
    data_inventory.add_argument("--sample-rows", type=int, default=5)
    data_inventory.add_argument("--max-files", type=int, default=None)
    data_inventory.add_argument("--include-raw", action="store_true")
    data_inventory.add_argument("--output-dir", default="data/abm_v3/data_inventory")

    phase_space_state_panel = subparsers.add_parser("phase-space-state-panel")
    phase_space_state_panel.add_argument("--start-year", type=int, default=1995)
    phase_space_state_panel.add_argument("--end-year", type=int, default=2016)
    phase_space_state_panel.add_argument("--output-dir", default="data/abm_v3/phase_space")
    phase_space_state_panel.add_argument(
        "--base-panel",
        default="data/abm_v3/inputs/abm_v3_historical_panel_1995_2016_transpose_row_fd_without_inventory.parquet",
    )
    phase_space_state_panel.add_argument(
        "--include-ei-transition",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    phase_space_state_panel.add_argument("--include-scenario-overlays", action="store_true", default=False)
    phase_space_state_panel.add_argument("--strict", action="store_true", default=False)

    phase_space_plots = subparsers.add_parser("phase-space-plots")
    phase_space_plots.add_argument("--start-year", type=int, default=1995)
    phase_space_plots.add_argument("--end-year", type=int, default=2016)
    phase_space_plots.add_argument(
        "--state-panel",
        default="data/abm_v3/phase_space/abm_v3_phase_space_state_panel_1995_2016.parquet",
    )
    phase_space_plots.add_argument("--output-dir", default="outputs/plots/abm_v3/phase_space")
    phase_space_plots.add_argument(
        "--audience",
        choices=["portfolio", "research", "diagnostic", "both"],
        default="both",
    )
    phase_space_plots.add_argument("--color-mode", choices=["default", "colorblind"], default="default")
    phase_space_plots.add_argument("--plot-3d", action=argparse.BooleanOptionalAction, default=True)
    phase_space_plots.add_argument("--plot-2d", action=argparse.BooleanOptionalAction, default=True)
    phase_space_plots.add_argument("--plot-vector-fields", action=argparse.BooleanOptionalAction, default=True)
    phase_space_plots.add_argument("--top-n", type=int, default=25)
    phase_space_plots.add_argument("--title-mode", choices=["theory", "technical"], default="theory")
    phase_space_plots.add_argument("--top-sector-n", type=int, default=8)
    phase_space_plots.add_argument("--top-node-n", type=int, default=10)
    phase_space_plots.add_argument("--research-top-node-n", type=int, default=25)
    phase_space_plots.add_argument("--mark-years", default="1995,2000,2008,2016")
    phase_space_plots.add_argument("--validate-vector-fields", action=argparse.BooleanOptionalAction, default=True)
    phase_space_plots.add_argument("--write-movement-diagnostics", action=argparse.BooleanOptionalAction, default=True)
    phase_space_plots.add_argument("--no-global", action="store_true")
    phase_space_plots.add_argument("--no-sector", action="store_true")
    phase_space_plots.add_argument("--no-node", action="store_true")
    phase_space_plots.add_argument("--strict", action="store_true", default=False)

    scenario_phase_space_plots = subparsers.add_parser("scenario-phase-space-plots")
    scenario_phase_space_plots.add_argument("--start-year", type=int, default=1995)
    scenario_phase_space_plots.add_argument("--end-year", type=int, default=2016)
    scenario_phase_space_plots.add_argument("--scenario-names", nargs="+", default=None)
    scenario_phase_space_plots.add_argument("--reference-scenario", default="historical_or_baseline")
    scenario_phase_space_plots.add_argument("--title-mode", choices=["interpretive", "technical"], default="interpretive")
    scenario_phase_space_plots.add_argument("--top-sector-n", type=int, default=8)
    scenario_phase_space_plots.add_argument("--top-node-n", type=int, default=10)
    scenario_phase_space_plots.add_argument("--research-top-node-n", type=int, default=25)
    scenario_phase_space_plots.add_argument("--mark-years", nargs="+", type=int, default=[1995, 2000, 2008, 2016])
    scenario_phase_space_plots.add_argument("--write-diagnostics", action=argparse.BooleanOptionalAction, default=True)
    scenario_phase_space_plots.add_argument("--no-plots", action="store_true")
    scenario_phase_space_plots.add_argument("--color-mode", choices=["default", "colorblind"], default="default")
    return parser


def add_behavioural_scenario_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    """Add behavioural propagation runtime overrides to a scenario parser."""
    parser.add_argument("--tolerance", type=float, default=None)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--eta-capacity", type=float, default=None)
    parser.add_argument("--no-node-rounds", action="store_true")


def run_leontief_year(
    year: int,
    paths: ABMV3Paths | None = None,
    config: ABMV3Config | None = None,
) -> dict[str, object]:
    """Build, propagate, validate, and write one Leontief baseline year."""
    active_paths = paths or ABMV3Paths()
    active_config = config or ABMV3Config()
    coefficient_builder = LeontiefCoefficientBuilder(active_paths, active_config.leontief)
    year_data = coefficient_builder.load_year(year)
    year_data = apply_input_panel_validation_target(year_data, active_paths, active_config, year)
    viability = LeontiefViabilityAnalyzer(active_config.leontief).analyze(year_data)
    LeontiefOutputWriter(active_paths).write_viability(viability)
    engine = LeontiefPropagationEngine(
        tolerance=active_config.leontief.tolerance,
        max_rounds=active_config.leontief.max_rounds,
    )
    result = engine.propagate(year_data)
    validator = LeontiefPropagationValidator()
    node_comparison = validator.build_node_comparison(year_data, result)
    summary = validator.build_summary(year_data, result, node_comparison)
    written_paths = LeontiefOutputWriter(active_paths).write_all(year_data, result, node_comparison, summary)
    relative_error = summary["relative_error_total"].iloc[0]
    print(
        f"[ABM v3 Leontief] Finished year {year}: "
        f"converged={result.converged}, rounds_used={result.rounds_used}"
    )
    print(
        "[ABM v3 Leontief] "
        f"observed_total={summary['observed_output_total'].iloc[0]:.12g}, "
        f"iterative_total={summary['accumulated_output_total'].iloc[0]:.12g}, "
        f"relative_error={relative_error:.12g}"
    )
    print(f"[ABM v3 Leontief] Wrote diagnostics to {active_paths.leontief_diagnostics_dir}")
    return {
        "year_data": year_data,
        "viability": viability,
        "result": result,
        "node_comparison": node_comparison,
        "summary": summary,
        "written_paths": written_paths,
    }


def run_leontief_diagnostics(
    year: int,
    paths: ABMV3Paths | None = None,
    config: ABMV3Config | None = None,
) -> dict[str, object]:
    """Build and write coefficient viability diagnostics for one year."""
    active_paths = paths or ABMV3Paths()
    active_config = config or ABMV3Config()
    print(f"[ABM v3 Leontief] Diagnosing coefficient viability for year {year}...")
    year_data = LeontiefCoefficientBuilder(active_paths, active_config.leontief).load_year(year)
    diagnostics = LeontiefViabilityAnalyzer(active_config.leontief).analyze(year_data)
    written_paths = LeontiefOutputWriter(active_paths).write_viability(diagnostics)
    summary = diagnostics.summary.iloc[0]
    spectral_a = diagnostics.spectral.loc[diagnostics.spectral["matrix"] == "A"].iloc[0]
    spectral_abs_a = diagnostics.spectral.loc[diagnostics.spectral["matrix"] == "abs_A"].iloc[0]
    print(f"[ABM v3 Leontief] mode={year_data.mode}")
    print(f"[ABM v3 Leontief] suspicious_columns={summary['suspicious_column_count']}")
    print(f"[ABM v3 Leontief] near_zero_positive_output={summary['near_zero_positive_output_count']}")
    print(f"[ABM v3 Leontief] negative_final_demand={summary['negative_final_demand_count']}")
    print(f"[ABM v3 Leontief] max_abs_column_sum_A={summary['max_abs_column_sum_A']:.12g}")
    print(
        "[ABM v3 Leontief] "
        f"approximate_spectral_radius_A={spectral_a['approximate_spectral_radius']:.12g}"
    )
    print(
        "[ABM v3 Leontief] "
        f"approximate_spectral_radius_abs_A={spectral_abs_a['approximate_spectral_radius']:.12g}"
    )
    print(f"[ABM v3 Leontief] Diagnostics written to {active_paths.leontief_diagnostics_dir}")
    return {"year_data": year_data, "diagnostics": diagnostics, "written_paths": written_paths}


def run_leontief_mode_comparison(
    year: int,
    modes: list[str] | None = None,
    paths: ABMV3Paths | None = None,
    config: ABMV3Config | None = None,
) -> pd.DataFrame:
    """Run and write the Leontief mode comparison for one year."""
    active_paths = paths or ABMV3Paths()
    active_config = config or ABMV3Config()
    print(f"[ABM v3 Leontief] Comparing coefficient modes for {year}")
    comparison = LeontiefModeComparator(active_paths, active_config).compare_year(year, modes=modes)
    for row in comparison.to_dict("records"):
        print(
            "[ABM v3 Leontief] "
            f"mode={row['mode']}, "
            f"spectral_radius={row['approximate_spectral_radius_A']:.12g}, "
            f"converged={row['converged']}, "
            f"relative_error={row['relative_error_total']:.12g}"
        )
    print(f"[ABM v3 Leontief] Wrote mode comparison to {active_paths.leontief_mode_comparison_path(year)}")
    return comparison


def run_leontief_orientation_audit(
    year: int,
    paths: ABMV3Paths | None = None,
    config: ABMV3Config | None = None,
    max_rounds: int = 400,
    tolerance: float = 1e-8,
    include_fd_without_inventory: bool = True,
    reference: str = "abm_ready",
    spectral_max_iter: int | None = None,
) -> dict[str, object]:
    """Run and write the Eora T orientation audit for one year."""
    active_paths = paths or ABMV3Paths()
    active_config = config or ABMV3Config()
    print(f"[ABM v3 Leontief] Auditing T orientation for {year}")
    audit = LeontiefOrientationAuditor(active_paths, active_config).audit_year(
        year=year,
        max_rounds=max_rounds,
        tolerance=tolerance,
        include_fd_without_inventory=include_fd_without_inventory,
        reference=reference,
        spectral_max_iter=spectral_max_iter,
    )
    written_paths = LeontiefOutputWriter(active_paths).write_orientation_audit(year, audit)
    for row in audit.summary.to_dict("records"):
        print(
            "[ABM v3 Leontief] "
            f"mode={row['orientation_mode']}, "
            f"rho={row['spectral_radius_A']:.12g}, "
            f"rel_error={row['relative_error_total']:.12g}, "
            f"converged={row['converged']}"
        )
    print(f"[ABM v3 Leontief] Wrote orientation audit to {written_paths['summary']}")
    return {"audit": audit, "written_paths": written_paths}


def build_leontief_config(args: argparse.Namespace) -> ABMV3Config:
    """Apply optional CLI Leontief overrides without mutating defaults."""
    config = ABMV3Config()
    leontief_config = config.leontief
    if hasattr(args, "tolerance") and args.tolerance is not None:
        leontief_config = replace(leontief_config, tolerance=args.tolerance)
    if hasattr(args, "max_rounds") and args.max_rounds is not None:
        leontief_config = replace(leontief_config, max_rounds=args.max_rounds)
    if hasattr(args, "mode") and args.mode is not None:
        leontief_config = replace(leontief_config, leontief_mode=args.mode)
    if hasattr(args, "column_sum_cap") and args.column_sum_cap is not None:
        leontief_config = replace(leontief_config, leontief_column_sum_cap=args.column_sum_cap)
    if hasattr(args, "input_panel_orientation") and args.input_panel_orientation is not None:
        leontief_config = replace(leontief_config, input_panel_orientation=args.input_panel_orientation)
    if hasattr(args, "eta_capacity") and args.eta_capacity is not None:
        leontief_config = replace(leontief_config, behavioural_capacity_eta=args.eta_capacity)
    if hasattr(args, "no_node_rounds") and args.no_node_rounds:
        leontief_config = replace(leontief_config, write_behavioural_node_rounds=False)
    return replace(config, leontief=leontief_config)


def build_behavioural_leontief_config(args: argparse.Namespace) -> ABMV3Config:
    config = build_leontief_config(args)
    leontief_config = config.leontief
    mode = args.mode if getattr(args, "mode", None) is not None else leontief_config.behavioural_default_mode
    leontief_config = replace(leontief_config, leontief_mode=mode)
    if getattr(args, "tolerance", None) is not None:
        leontief_config = replace(leontief_config, behavioural_tolerance=args.tolerance)
    if getattr(args, "max_rounds", None) is not None:
        leontief_config = replace(leontief_config, behavioural_max_rounds=args.max_rounds)
    return replace(config, leontief=leontief_config)


def build_behavioural_scenario_config(args: argparse.Namespace) -> ABMV3Config:
    """Build the corrected behavioural Leontief scenario config from CLI args."""
    config = ABMV3Config()
    leontief_config = replace(
        config.leontief,
        leontief_mode=args.mode,
        input_panel_orientation=args.input_panel_orientation,
    )
    if getattr(args, "tolerance", None) is not None:
        leontief_config = replace(leontief_config, behavioural_tolerance=args.tolerance)
    if getattr(args, "max_rounds", None) is not None:
        leontief_config = replace(leontief_config, behavioural_max_rounds=args.max_rounds)
    if getattr(args, "eta_capacity", None) is not None:
        leontief_config = replace(leontief_config, behavioural_capacity_eta=args.eta_capacity)
    if getattr(args, "no_node_rounds", False):
        leontief_config = replace(leontief_config, write_behavioural_node_rounds=False)
    return replace(config, leontief=leontief_config)


def build_behavioural_scenario_shock(args: argparse.Namespace):
    """Load a registered behavioural scenario and apply CLI overrides."""
    shock = get_behavioural_scenario(args.scenario)
    updates: dict[str, object] = {
        "low_ei_quantile": args.low_ei_quantile,
        "high_ei_quantile": args.high_ei_quantile,
        "high_capability_quantile": args.high_capability_quantile,
    }
    if args.shock_size is not None:
        updates["shock_size"] = args.shock_size
    if args.selector is not None:
        updates["selector_name"] = args.selector
    return replace(shock, **updates)


def build_corrected_input_panel_config(args: argparse.Namespace) -> ABMV3Config:
    """Apply corrected input-panel CLI overrides without changing defaults."""
    config = ABMV3Config()
    calibration_config = config.calibration
    if getattr(args, "capacity_margin", None) is not None:
        calibration_config = replace(calibration_config, capacity_margin=args.capacity_margin)
    if getattr(args, "inventory_days", None) is not None:
        calibration_config = replace(calibration_config, inventory_days=args.inventory_days)
    return replace(config, calibration=calibration_config)


def run_corrected_input_panel_build(
    start_year: int,
    end_year: int,
    overwrite: bool = False,
    orientation: str = "transpose_row_fd_without_inventory",
    paths: ABMV3Paths | None = None,
    config: ABMV3Config | None = None,
) -> pd.DataFrame:
    """Build the experimental corrected-orientation input panel."""
    active_paths = paths or ABMV3Paths()
    active_config = config or ABMV3Config()
    print(f"[ABM v3] Building corrected input panel: orientation={orientation}")
    builder = CorrectedOrientationInputPanelBuilder(active_paths, active_config, orientation=orientation)
    panel = builder.build(start_year=start_year, end_year=end_year, overwrite=overwrite)
    for year, group in panel.groupby("Year", dropna=False):
        excluded_path = active_paths.abm_v3_output_root / "diagnostics" / (
            f"abm_v3_excluded_inventory_fd_columns_{start_year}_{end_year}.csv"
        )
        excluded_columns = 0
        if excluded_path.exists():
            excluded = pd.read_csv(excluded_path)
            excluded_columns = int((excluded["Year"].astype(int) == int(year)).sum()) if "Year" in excluded.columns else 0
        print(
            "[ABM v3] "
            f"Year {int(year)}: "
            f"X_corrected_total={group['X_corrected'].sum():.12g}, "
            f"inventory_excluded_columns={excluded_columns}"
        )
    output_path = builder.output_path(start_year, end_year)
    diagnostics_dir = active_paths.abm_v3_output_root / "diagnostics"
    print(f"[ABM v3] Wrote corrected panel to {output_path}")
    print(f"[ABM v3] Wrote orientation comparison diagnostics to {diagnostics_dir}")
    return panel


def run_corrected_input_panel_smoke_test(
    start_year: int,
    end_year: int,
    orientation: str = "transpose_row_fd_without_inventory",
    paths: ABMV3Paths | None = None,
) -> pd.DataFrame:
    """Smoke test the corrected panel explicitly."""
    active_paths = paths or ABMV3Paths()
    panel_path = active_paths.abm_v3_corrected_historical_panel_file(start_year, end_year, orientation)
    if not panel_path.exists():
        raise FileNotFoundError(f"Corrected input panel not found: {panel_path}")
    panel = pd.read_parquet(panel_path)
    report = RealDataSmokeTester(active_paths, start_year=start_year, end_year=end_year).run(
        df=panel,
        write_report=False,
    )
    output_path = active_paths.abm_v3_output_root / "diagnostics" / (
        f"real_data_smoke_test_{orientation}_{start_year}_{end_year}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    passed = int(report["passed"].sum()) if "passed" in report.columns else 0
    print(f"[ABM v3] Corrected input panel smoke test: rows={len(report)}, passed={passed}")
    print(f"[ABM v3] Wrote corrected smoke test to {output_path}")
    return report


def run_ei_transition_panel_build(
    start_year: int = 1995,
    end_year: int = 2016,
    overwrite: bool = False,
    paths: ABMV3Paths | None = None,
) -> dict[str, object]:
    """Build and write the historical EI transition panel."""
    active_paths = paths or ABMV3Paths()
    panel_path = active_paths.ei_transition_panel_path(start_year, end_year)
    if panel_path.exists() and not overwrite:
        print(f"[ABM v3 EI Transition] Panel already exists: {panel_path}")
        panel = pd.read_parquet(panel_path)
        return {"panel": panel, "written_paths": {"panel": panel_path}}
    builder = EITransitionPanelBuilder(active_paths)
    result = builder.build(start_year=start_year, end_year=end_year)
    written_paths = EITransitionOutputWriter(active_paths).write_panel(start_year, end_year, result)
    report = result.sample_report.iloc[0]
    print(
        "[ABM v3 EI Transition] "
        f"Built panel rows={report['total_rows']}, included={report['included_rows']}, "
        f"excluded={report['excluded_rows']}"
    )
    print(f"[ABM v3 EI Transition] Wrote transition panel to {written_paths['panel']}")
    return {"panel": result.panel, "sample_report": result.sample_report, "written_paths": written_paths}


def run_ei_transition_fit(
    start_year: int = 1995,
    end_year: int = 2016,
    train_end_year: int = 2012,
    validation_start_year: int = 2013,
    validation_end_year: int = 2015,
    overwrite_panel: bool = False,
    paths: ABMV3Paths | None = None,
) -> dict[str, object]:
    """Build/load the EI transition panel, fit models, and write diagnostics."""
    active_paths = paths or ABMV3Paths()
    panel_path = active_paths.ei_transition_panel_path(start_year, end_year)
    if overwrite_panel or not panel_path.exists():
        panel_output = run_ei_transition_panel_build(
            start_year=start_year,
            end_year=end_year,
            overwrite=True,
            paths=active_paths,
        )
        panel = panel_output["panel"]
    else:
        print(f"[ABM v3 EI Transition] Loading transition panel from {panel_path}")
        panel = pd.read_parquet(panel_path)
    validate_transition_split(panel, train_end_year, validation_start_year, validation_end_year)
    fit_output = EITransitionModelSuite(alpha=1.0).fit_all(
        panel,
        train_end_year=train_end_year,
        validation_start_year=validation_start_year,
        validation_end_year=validation_end_year,
    )
    written_paths = EITransitionOutputWriter(active_paths).write_fit_outputs(
        start_year,
        end_year,
        fit_output["scores"],
        fit_output["coefficients"],
        fit_output["expected_signs"],
        fit_output["predictions"],
    )
    print(
        "[ABM v3 EI Transition] "
        f"Fitted {len(fit_output['scores'])} models with validation years "
        f"{validation_start_year}-{validation_end_year}"
    )
    for row in fit_output["scores"].to_dict("records"):
        print(
            "[ABM v3 EI Transition] "
            f"model={row['model_name']}, rmse={row['rmse']:.12g}, "
            f"mae={row['mae']:.12g}, r2={row['r2']:.12g}, "
            f"corr={row['correlation_predicted_observed']:.12g}"
        )
    print(f"[ABM v3 EI Transition] Wrote diagnostics to {active_paths.ei_transition_dir}")
    return {
        "panel": panel,
        "scores": fit_output["scores"],
        "coefficients": fit_output["coefficients"],
        "expected_signs": fit_output["expected_signs"],
        "predictions": fit_output["predictions"],
        "written_paths": written_paths,
    }


def apply_input_panel_validation_target(
    year_data: LeontiefYearData,
    paths: ABMV3Paths,
    config: ABMV3Config,
    year: int,
) -> LeontiefYearData:
    """Override validation X from the explicitly selected input panel."""
    orientation = config.leontief.input_panel_orientation
    if orientation is None:
        return year_data
    panel = ABMV3DataLoader(paths).load_input_panel_for_orientation(
        config.calibration.start_year,
        config.calibration.end_year,
        orientation,
        config,
    )
    required_columns = {"Year", "country_sector", "X_observed"}
    missing = required_columns.difference(panel.columns)
    if missing:
        raise ValueError(
            f"Input panel orientation '{orientation}' is missing validation columns: {sorted(missing)}"
        )
    labels = year_data.labels["country_sector"].astype(str).tolist()
    year_panel = panel.loc[panel["Year"].astype(int) == int(year), ["country_sector", "X_observed"]].copy()
    if year_panel.empty:
        raise ValueError(f"Input panel orientation '{orientation}' has no validation rows for year {year}")
    x_observed = pd.Series(
        pd.to_numeric(year_panel["X_observed"], errors="coerce").to_numpy(dtype=float),
        index=year_panel["country_sector"].astype(str),
        name="X_observed",
    ).reindex(labels)
    if x_observed.isna().all():
        raise ValueError(
            f"Input panel orientation '{orientation}' has no country_sector labels matching Leontief year {year}"
        )
    invalid_output_columns = build_invalid_output_columns_for_validation_target(year_data, x_observed)
    panel_path = ABMV3DataLoader(paths).input_panel_path_for_orientation(
        config.calibration.start_year,
        config.calibration.end_year,
        orientation,
    )
    print(
        "[ABM v3 Leontief] "
        f"Validation target: input_panel_orientation={orientation}, X_observed={panel_path}"
    )
    return replace(
        year_data,
        X_observed=x_observed,
        input_panel_orientation=orientation,
        validation_reference=f"input_panel:{orientation}:X_observed",
        invalid_output_columns=invalid_output_columns,
    )


def build_invalid_output_columns_for_validation_target(year_data: LeontiefYearData, x_observed: pd.Series) -> pd.DataFrame:
    """Report invalid output rows after applying an explicit validation target."""
    labels_frame = year_data.labels.copy()
    x_values = x_observed.to_numpy(dtype=float)
    invalid_mask = (~np.isfinite(x_values)) | (x_values <= 0.0)
    invalid = labels_frame.loc[invalid_mask].copy()
    invalid.insert(0, "Year", year_data.year)
    invalid["X_observed"] = x_values[invalid_mask]
    invalid["reason"] = ["missing_or_non_finite_output" if pd.isna(value) else "non_positive_output" for value in x_values[invalid_mask]]
    return invalid


def load_behavioural_capacity(paths: ABMV3Paths, config: ABMV3Config, year: int) -> pd.Series:
    """Load same-year K from the selected ABM-ready input panel."""
    orientation = config.leontief.input_panel_orientation
    selected_orientation = orientation or "current_column"
    print(
        "[ABM v3 Behavioural Leontief] "
        f"Loading capacity from input_panel_orientation={selected_orientation}..."
    )
    panel = ABMV3DataLoader(paths).load_input_panel_for_orientation(
        config.calibration.start_year,
        config.calibration.end_year,
        orientation,
        config,
    )
    required_columns = {"Year", "country_sector", "K"}
    missing = required_columns.difference(panel.columns)
    if missing:
        raise ValueError(f"ABM-ready panel is missing required capacity columns: {sorted(missing)}")
    year_panel = panel.loc[panel["Year"].astype(int) == int(year), ["country_sector", "K"]].copy()
    if year_panel.empty:
        raise ValueError(f"No ABM-ready capacity rows found for year {year}")
    return pd.Series(
        pd.to_numeric(year_panel["K"], errors="coerce").to_numpy(dtype=float),
        index=year_panel["country_sector"].astype(str),
        name="K",
    )


def run_behavioural_leontief_year(
    year: int,
    paths: ABMV3Paths | None = None,
    config: ABMV3Config | None = None,
) -> dict[str, object]:
    active_paths = paths or ABMV3Paths()
    active_config = config or ABMV3Config()
    print(
        "[ABM v3 Behavioural Leontief] "
        f"Loading year {year} with mode={active_config.leontief.leontief_mode}..."
    )
    year_data = LeontiefCoefficientBuilder(active_paths, active_config.leontief).load_year(year)
    year_data = apply_input_panel_validation_target(year_data, active_paths, active_config, year)
    capacity = load_behavioural_capacity(active_paths, active_config, year)
    orientation = active_config.leontief.input_panel_orientation or "current_column"
    capacity_path = ABMV3DataLoader(active_paths).input_panel_path_for_orientation(
        active_config.calibration.start_year,
        active_config.calibration.end_year,
        active_config.leontief.input_panel_orientation,
    )
    year_data = replace(
        year_data,
        input_panel_orientation=active_config.leontief.input_panel_orientation,
        capacity_source=f"input_panel:{orientation}:K:{capacity_path}",
    )
    result = BehaviouralLeontiefEngine(active_config.leontief).propagate(year_data, capacity)
    validator = BehaviouralLeontiefValidator()
    node_comparison = validator.build_node_comparison(year_data, result)
    summary = validator.build_summary(year_data, result, node_comparison)
    written_paths = BehaviouralLeontiefOutputWriter(active_paths).write_all(
        year_data,
        result,
        node_comparison,
        summary,
    )
    relative_error = summary["relative_error_total"].iloc[0]
    print(
        "[ABM v3 Behavioural Leontief] "
        f"Finished: converged={result.converged}, rounds_used={result.rounds_used}"
    )
    print(
        "[ABM v3 Behavioural Leontief] "
        f"observed_total={summary['observed_output_total'].iloc[0]:.12g}, "
        f"realized_total={summary['realized_output_total'].iloc[0]:.12g}, "
        f"relative_error={relative_error:.12g}"
    )
    print(
        "[ABM v3 Behavioural Leontief] "
        f"Wrote diagnostics to {active_paths.behavioural_leontief_diagnostics_dir}"
    )
    return {
        "year_data": year_data,
        "capacity": capacity,
        "result": result,
        "node_comparison": node_comparison,
        "summary": summary,
        "written_paths": written_paths,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = build_parser().parse_args()
    model = ABMV3Model(config=ABMV3Config(), paths=ABMV3Paths())

    if args.command == "calibrate":
        result = model.fit_historical(args.start_year, args.end_year, args.ei_mode, args.validation_mode)
        print(f"Calibration result: {result}")
    elif args.command == "validate":
        result = model.validate_historical(args.split_year)
        print(f"Validation result: {result}")
    elif args.command == "simulate":
        result = model.simulate(args.start_year, args.end_year, scenario=args.scenario)
        print(
            "Simulation scaffold: "
            f"scenario={args.scenario}, years={args.start_year}-{args.end_year}, rows={len(result)}"
        )
    elif args.command == "compare-scenarios":
        print(f"Scenario comparison scaffold. Registered scenarios: {list_scenarios()}")
    elif args.command == "smoke-test":
        panel_kind = "input_panel" if args.input_panel else "merged_panel" if args.merged_panel else "auto"
        report = RealDataSmokeTester(ABMV3Paths()).run(write_report=not args.no_write, panel_kind=panel_kind)
        print(f"Smoke test complete: rows={len(report)}, passed={int(report['passed'].sum()) if 'passed' in report.columns else 0}")
    elif args.command == "fit-historical":
        result = model.fit_historical(
            args.start_year,
            args.end_year,
            ei_mode=args.ei_mode,
            validation_mode=args.validation_mode,
        )
        print(f"Historical fit result: {result}")
    elif args.command == "hypothesis-report":
        panel = model.prepare_model_ready_panel(
            model.data_loader.load_abm_ready_historical_panel(
                model.config.calibration.start_year,
                model.config.calibration.end_year,
                model.config,
            )
        )
        reports = HypothesisReportGenerator(model.paths).write_all(panel)
        print(f"Hypothesis reports written: {list(reports)}")
    elif args.command == "build-input-panel":
        builder = ABMV3InputPanelBuilder(ABMV3Paths(), ABMV3Config())
        path = builder.output_path(args.start_year, args.end_year)
        if path.exists() and not args.overwrite:
            print(f"ABM-ready input panel already exists: {path}")
        else:
            panel = builder.build(args.start_year, args.end_year, overwrite=args.overwrite)
            print(f"ABM-ready input panel written: {path} rows={len(panel)}")
    elif args.command == "build-corrected-input-panel":
        run_corrected_input_panel_build(
            args.start_year,
            args.end_year,
            overwrite=args.overwrite,
            orientation=args.orientation,
            paths=ABMV3Paths(),
            config=build_corrected_input_panel_config(args),
        )
    elif args.command == "build-ei-transition-panel":
        run_ei_transition_panel_build(
            start_year=args.start_year,
            end_year=args.end_year,
            overwrite=args.overwrite,
            paths=ABMV3Paths(),
        )
    elif args.command == "fit-ei-transition":
        run_ei_transition_fit(
            start_year=args.start_year,
            end_year=args.end_year,
            train_end_year=args.train_end_year,
            validation_start_year=args.validation_start_year,
            validation_end_year=args.validation_end_year,
            overwrite_panel=args.overwrite_panel,
            paths=ABMV3Paths(),
        )
    elif args.command == "smoke-test-corrected-input-panel":
        run_corrected_input_panel_smoke_test(
            args.start_year,
            args.end_year,
            orientation=args.orientation,
            paths=ABMV3Paths(),
        )
    elif args.command == "leontief-propagate":
        run_leontief_year(args.year, paths=ABMV3Paths(), config=build_leontief_config(args))
    elif args.command == "leontief-propagate-range":
        config = build_leontief_config(args)
        for year in range(args.start_year, args.end_year + 1):
            print(f"[ABM v3 Leontief] Range progress: year={year}")
            run_leontief_year(year, paths=ABMV3Paths(), config=config)
    elif args.command == "leontief-diagnose":
        run_leontief_diagnostics(args.year, paths=ABMV3Paths(), config=build_leontief_config(args))
    elif args.command == "leontief-diagnose-range":
        config = build_leontief_config(args)
        for year in range(args.start_year, args.end_year + 1):
            print(f"[ABM v3 Leontief] Diagnostic range progress: year={year}")
            run_leontief_diagnostics(year, paths=ABMV3Paths(), config=config)
    elif args.command == "leontief-compare-modes":
        run_leontief_mode_comparison(
            args.year,
            modes=args.modes,
            paths=ABMV3Paths(),
            config=build_leontief_config(args),
        )
    elif args.command == "leontief-compare-modes-range":
        config = build_leontief_config(args)
        comparisons = []
        for year in range(args.start_year, args.end_year + 1):
            comparisons.append(
                run_leontief_mode_comparison(
                    year,
                    modes=args.modes,
                    paths=ABMV3Paths(),
                    config=config,
                )
            )
        combined = pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame()
        output_path = ABMV3Paths().leontief_mode_comparison_range_path(args.start_year, args.end_year)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)
        print(f"[ABM v3 Leontief] Wrote range mode comparison to {output_path}")
    elif args.command == "leontief-audit-orientation":
        run_leontief_orientation_audit(
            args.year,
            paths=ABMV3Paths(),
            config=ABMV3Config(),
            max_rounds=args.max_rounds,
            tolerance=args.tolerance,
            include_fd_without_inventory=args.include_fd_without_inventory,
            reference=args.reference,
            spectral_max_iter=args.spectral_max_iter,
        )
    elif args.command == "leontief-audit-orientation-range":
        summaries = []
        for year in range(args.start_year, args.end_year + 1):
            print(f"[ABM v3 Leontief] Orientation audit range progress: year={year}")
            output = run_leontief_orientation_audit(
                year,
                paths=ABMV3Paths(),
                config=ABMV3Config(),
                max_rounds=args.max_rounds,
                tolerance=args.tolerance,
                include_fd_without_inventory=args.include_fd_without_inventory,
                reference=args.reference,
                spectral_max_iter=args.spectral_max_iter,
            )
            summaries.append(output["audit"].summary)
        combined = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
        output_path = ABMV3Paths().leontief_orientation_summary_range_path(args.start_year, args.end_year)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)
        print(f"[ABM v3 Leontief] Wrote range orientation audit to {output_path}")
    elif args.command == "behavioural-leontief":
        run_behavioural_leontief_year(
            args.year,
            paths=ABMV3Paths(),
            config=build_behavioural_leontief_config(args),
        )
    elif args.command == "behavioural-leontief-range":
        config = build_behavioural_leontief_config(args)
        for year in range(args.start_year, args.end_year + 1):
            print(f"[ABM v3 Behavioural Leontief] Range progress: year={year}")
            run_behavioural_leontief_year(year, paths=ABMV3Paths(), config=config)
    elif args.command == "validation-report":
        written_paths = ABMV3ValidationReportBuilder(ABMV3Paths()).build(
            start_year=args.start_year,
            end_year=args.end_year,
        )
        for name, path in written_paths.items():
            print(f"[ABM v3 Validation Report] {name}: {path}")
    elif args.command == "list-behavioural-scenarios":
        for scenario_name in list_behavioural_scenarios():
            scenario = get_behavioural_scenario(scenario_name)
            print(f"{scenario_name}: {scenario.description}")
    elif args.command == "behavioural-scenario":
        shock = build_behavioural_scenario_shock(args)
        runner = BehaviouralLeontiefScenarioRunner(
            paths=ABMV3Paths(),
            config=build_behavioural_scenario_config(args),
        )
        output = runner.run_year(args.year, args.scenario, shock)
        for name, path in output["written_paths"].items():
            print(f"[ABM v3 Behavioural Scenario] {name}: {path}")
    elif args.command == "behavioural-scenario-range":
        shock = build_behavioural_scenario_shock(args)
        runner = BehaviouralLeontiefScenarioRunner(
            paths=ABMV3Paths(),
            config=build_behavioural_scenario_config(args),
        )
        for year in range(args.start_year, args.end_year + 1):
            print(f"[ABM v3 Behavioural Scenario] Range progress: year={year}")
            output = runner.run_year(year, args.scenario, shock)
            for name, path in output["written_paths"].items():
                print(f"[ABM v3 Behavioural Scenario] {year} {name}: {path}")
    elif args.command == "behavioural-scenario-report":
        written_paths = BehaviouralScenarioAnalysisReportBuilder(
            paths=ABMV3Paths(),
            mode=args.mode,
            input_panel_orientation=args.input_panel_orientation,
            audience=args.audience,
            color_mode=args.color_mode,
            make_plots=not args.no_plots,
        ).build(start_year=args.start_year, end_year=args.end_year)
        for name, path in written_paths.items():
            print(f"[ABM v3 Behavioural Scenario Report] {name}: {path}")
    elif args.command == "data-inventory":
        written_paths = build_data_inventory(
            root=args.root,
            focus=args.focus,
            sample_rows=args.sample_rows,
            max_files=args.max_files,
            include_raw=args.include_raw,
            output_dir=args.output_dir,
        )
        for name, path in written_paths.items():
            print(f"[ABM v3 Data Inventory] {name}: {path}")
    elif args.command == "phase-space-state-panel":
        written_paths = ABMV3PhaseSpaceStatePanelBuilder(
            base_panel=args.base_panel,
            output_dir=args.output_dir,
            include_ei_transition=args.include_ei_transition,
            include_scenario_overlays=args.include_scenario_overlays,
            strict=args.strict,
        ).build(start_year=args.start_year, end_year=args.end_year)
        for name, path in written_paths.items():
            print(f"[ABM v3 Phase Space] {name}: {path}")
    elif args.command == "phase-space-plots":
        written_paths = PhaseSpacePlotBuilder(
            state_panel=args.state_panel,
            output_dir=args.output_dir,
            audience=args.audience,
            color_mode=args.color_mode,
            plot_3d=args.plot_3d,
            plot_2d=args.plot_2d,
            plot_vector_fields=args.plot_vector_fields,
            top_n=args.top_n,
            top_sector_n=args.top_sector_n,
            top_node_n=args.top_node_n,
            research_top_node_n=args.research_top_node_n,
            title_mode=args.title_mode,
            mark_years=tuple(int(token.strip()) for token in str(args.mark_years).split(",") if token.strip()),
            validate_vector_fields=args.validate_vector_fields,
            write_movement_diagnostics=args.write_movement_diagnostics,
            include_global=not args.no_global,
            include_sector=not args.no_sector,
            include_node=not args.no_node,
            strict=args.strict,
        ).build(start_year=args.start_year, end_year=args.end_year)
        for name, path in written_paths.items():
            print(f"[ABM v3 Phase Space Plots] {name}: {path}")
    elif args.command == "scenario-phase-space-plots":
        written_paths = ScenarioPhaseSpacePlotBuilder(
            paths=ABMV3Paths(),
            scenario_names=args.scenario_names,
            reference_scenario=args.reference_scenario,
            title_mode=args.title_mode,
            top_sector_n=args.top_sector_n,
            top_node_n=args.top_node_n,
            research_top_node_n=args.research_top_node_n,
            mark_years=tuple(args.mark_years),
            write_diagnostics=args.write_diagnostics,
            make_plots=not args.no_plots,
            color_mode=args.color_mode,
        ).build(start_year=args.start_year, end_year=args.end_year)
        for name, path in written_paths.items():
            print(f"[ABM v3 Scenario Phase Space] {name}: {path}")


if __name__ == "__main__":
    main()
