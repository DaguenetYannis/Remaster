from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.abm_v2.calibrate import (
    apply_calibrated_parameters,
    calibrate_scenario,
    load_calibrated_parameters,
)
from src.abm_v2.metrics import build_abm_metrics_panel
from src.abm_v2.model import GreenTransitionABM
from src.abm_v2.plots import ABMPlotter
from src.abm_v2.scenarii import Scenario, get_scenario, list_scenarios


def load_or_build_metrics_panel(
    years: list[int],
    metrics_panel_path: Path,
    metrics_dir: Path,
    rebuild_metrics: bool,
) -> pd.DataFrame:
    if metrics_panel_path.exists() and not rebuild_metrics:
        return pd.read_parquet(metrics_panel_path)

    panel = build_abm_metrics_panel(
        years=years,
        metrics_dir=metrics_dir,
        output_dir=metrics_panel_path.parent,
    )

    return panel


def run_single_scenario(
    metrics_panel: pd.DataFrame,
    scenario: Scenario,
    start_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = GreenTransitionABM(
        metrics_panel=metrics_panel,
        scenario=scenario,
        start_year=start_year,
    )

    return model.run()


def build_scenarios_for_run(
    metrics_panel: pd.DataFrame,
    scenario_names: list[str],
    start_year: int,
    calibrate: bool,
    calibration_end_year: int | None,
    calibration_output_dir: Path,
    use_calibrated_parameters: bool,
    calibrated_parameters_path: Path,
) -> list[Scenario]:
    scenarios: list[Scenario] = []

    for scenario_name in scenario_names:
        scenario = get_scenario(scenario_name)

        if calibrate:
            if calibration_end_year is None:
                raise ValueError(
                    "--calibration-end-year is required when using --calibrate."
                )

            scenario, _ = calibrate_scenario(
                metrics_panel=metrics_panel,
                scenario_name=scenario_name,
                start_year=start_year,
                end_year=calibration_end_year,
                output_dir=calibration_output_dir / scenario_name,
            )

        elif use_calibrated_parameters:
            parameters = load_calibrated_parameters(calibrated_parameters_path)
            scenario = apply_calibrated_parameters(
                scenario=scenario,
                parameters=parameters,
            )

        scenarios.append(scenario)

    return scenarios


def run_many_scenarios(
    metrics_panel: pd.DataFrame,
    scenarios: list[Scenario],
    start_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_node_results = []
    all_aggregate_results = []

    for scenario in scenarios:
        node_results, aggregate_results = run_single_scenario(
            metrics_panel=metrics_panel,
            scenario=scenario,
            start_year=start_year,
        )

        all_node_results.append(node_results)
        all_aggregate_results.append(aggregate_results)

    return (
        pd.concat(all_node_results, ignore_index=True),
        pd.concat(all_aggregate_results, ignore_index=True),
    )


def save_results(
    node_results: pd.DataFrame,
    aggregate_results: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    node_results.to_parquet(output_dir / "node_results.parquet", index=False)
    aggregate_results.to_parquet(output_dir / "aggregate_results.parquet", index=False)

    node_results.to_csv(output_dir / "node_results.csv", index=False)
    aggregate_results.to_csv(output_dir / "aggregate_results.csv", index=False)


def parse_years(raw_years: str) -> list[int]:
    if "-" in raw_years:
        start, end = raw_years.split("-")
        return list(range(int(start), int(end) + 1))

    return [int(year.strip()) for year in raw_years.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the green transition ABM."
    )

    parser.add_argument(
        "--years",
        type=str,
        default="1990-2016",
        help="Years used to build the metrics panel.",
    )

    parser.add_argument(
        "--start-year",
        type=int,
        required=True,
        help="Base year used to initialize the ABM.",
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default="baseline",
        help="Scenario name.",
    )

    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Run all registered scenarios.",
    )

    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="Print available scenarios and exit.",
    )

    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("data/metrics"),
        help="Directory containing yearly metric outputs.",
    )

    parser.add_argument(
        "--metrics-panel-path",
        type=Path,
        default=Path("data/abm/metrics/abm_metrics_panel.parquet"),
        help="Path to the ABM-ready metrics panel.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/abm/runs"),
        help="Directory where simulation outputs are saved.",
    )

    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("outputs/abm/plots"),
        help="Directory where plots are saved.",
    )

    parser.add_argument(
        "--rebuild-metrics",
        action="store_true",
        help="Force rebuilding the ABM metrics panel.",
    )

    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation.",
    )

    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Calibrate scenario parameters before running the ABM.",
    )

    parser.add_argument(
        "--calibration-end-year",
        type=int,
        default=None,
        help="Final historical year used for calibration.",
    )

    parser.add_argument(
        "--calibration-output-dir",
        type=Path,
        default=Path("outputs/abm/calibration"),
        help="Directory where calibration outputs are saved.",
    )

    parser.add_argument(
        "--use-calibrated-parameters",
        action="store_true",
        help="Load calibrated parameters from JSON before running.",
    )

    parser.add_argument(
        "--calibrated-parameters-path",
        type=Path,
        default=Path("outputs/abm/calibration/baseline/best_parameters.json"),
        help="Path to saved calibrated parameters JSON.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_scenarios:
        print("Available scenarios:")
        for scenario_name in list_scenarios():
            print(f"- {scenario_name}")
        return

    if args.calibrate and args.use_calibrated_parameters:
        raise ValueError(
            "Use either --calibrate or --use-calibrated-parameters, not both."
        )

    years = parse_years(args.years)

    metrics_panel = load_or_build_metrics_panel(
        years=years,
        metrics_panel_path=args.metrics_panel_path,
        metrics_dir=args.metrics_dir,
        rebuild_metrics=args.rebuild_metrics,
    )

    scenario_names = list_scenarios() if args.all_scenarios else [args.scenario]

    scenarios = build_scenarios_for_run(
        metrics_panel=metrics_panel,
        scenario_names=scenario_names,
        start_year=args.start_year,
        calibrate=args.calibrate,
        calibration_end_year=args.calibration_end_year,
        calibration_output_dir=args.calibration_output_dir,
        use_calibrated_parameters=args.use_calibrated_parameters,
        calibrated_parameters_path=args.calibrated_parameters_path,
    )

    node_results, aggregate_results = run_many_scenarios(
        metrics_panel=metrics_panel,
        scenarios=scenarios,
        start_year=args.start_year,
    )

    run_label = (
        f"all_scenarios_{args.start_year}"
        if args.all_scenarios
        else f"{args.scenario}_{args.start_year}"
    )

    if args.calibrate:
        run_label = f"{run_label}_calibrated"

    if args.use_calibrated_parameters:
        run_label = f"{run_label}_using_calibrated_parameters"

    run_output_dir = args.output_dir / run_label

    save_results(
        node_results=node_results,
        aggregate_results=aggregate_results,
        output_dir=run_output_dir,
    )

    if not args.skip_plots:
        plotter = ABMPlotter(output_dir=args.plots_dir / run_label)
        plotter.save_all(
            aggregate_results=aggregate_results,
            node_results=node_results,
        )

    print(f"[OK] ABM run complete: {run_label}")
    print(f"[OK] Results saved to: {run_output_dir}")

    if args.calibrate:
        print(f"[OK] Calibration saved to: {args.calibration_output_dir}")

    if not args.skip_plots:
        print(f"[OK] Plots saved to: {args.plots_dir / run_label}")


if __name__ == "__main__":
    main()