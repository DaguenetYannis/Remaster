from __future__ import annotations

import argparse
import logging
from dataclasses import replace

from src.abm_v3.config import ABMV3Config
from src.abm_v3.diagnostics.hypothesis_reports import HypothesisReportGenerator
from src.abm_v3.input_panel_builder import ABMV3InputPanelBuilder
from src.abm_v3.leontief.coefficients import LeontiefCoefficientBuilder
from src.abm_v3.leontief.outputs import LeontiefOutputWriter
from src.abm_v3.leontief.propagation import LeontiefPropagationEngine
from src.abm_v3.leontief.validation import LeontiefPropagationValidator
from src.abm_v3.model import ABMV3Model
from src.abm_v3.paths import ABMV3Paths
from src.abm_v3.real_data_smoke_test import RealDataSmokeTester
from src.abm_v3.scenarios.registry import list_scenarios


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

    leontief_propagate = subparsers.add_parser("leontief-propagate")
    leontief_propagate.add_argument("--year", type=int, required=True)
    leontief_propagate.add_argument("--tolerance", type=float, default=None)
    leontief_propagate.add_argument("--max-rounds", type=int, default=None)

    leontief_range = subparsers.add_parser("leontief-propagate-range")
    leontief_range.add_argument("--start-year", type=int, required=True)
    leontief_range.add_argument("--end-year", type=int, required=True)
    leontief_range.add_argument("--tolerance", type=float, default=None)
    leontief_range.add_argument("--max-rounds", type=int, default=None)
    return parser


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
        "result": result,
        "node_comparison": node_comparison,
        "summary": summary,
        "written_paths": written_paths,
    }


def build_leontief_config(args: argparse.Namespace) -> ABMV3Config:
    """Apply optional CLI Leontief overrides without mutating defaults."""
    config = ABMV3Config()
    leontief_config = config.leontief
    if args.tolerance is not None:
        leontief_config = replace(leontief_config, tolerance=args.tolerance)
    if args.max_rounds is not None:
        leontief_config = replace(leontief_config, max_rounds=args.max_rounds)
    return replace(config, leontief=leontief_config)


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
    elif args.command == "leontief-propagate":
        run_leontief_year(args.year, paths=ABMV3Paths(), config=build_leontief_config(args))
    elif args.command == "leontief-propagate-range":
        config = build_leontief_config(args)
        for year in range(args.start_year, args.end_year + 1):
            print(f"[ABM v3 Leontief] Range progress: year={year}")
            run_leontief_year(year, paths=ABMV3Paths(), config=config)


if __name__ == "__main__":
    main()
