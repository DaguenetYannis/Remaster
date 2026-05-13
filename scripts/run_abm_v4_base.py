from __future__ import annotations

import argparse

from src.abm_v4.config import ABMV4Config
from src.abm_v4.diagnostics import build_path_audit_rows, format_path_audit_table
from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.simulation import inspect_base_model_readiness
from src.abm_v4.state import build_state_panel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ABM v4 base readiness and dry-run audit entry point."
    )
    parser.add_argument("--start-year", type=int, default=1995)
    parser.add_argument("--end-year", type=int, default=2016)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit source paths without creating ABM v4 output folders.",
    )
    parser.add_argument(
        "--create-output-dirs",
        action="store_true",
        help="Create data/abm_v4 output directories for a future run.",
    )
    parser.add_argument(
        "--build-state",
        action="store_true",
        help="Build the ABM v4 state panel. Requires --create-output-dirs to write files.",
    )
    return parser


def main() -> None:
    """Inspect ABM v4 base-model readiness without generating simulation data."""
    args = build_parser().parse_args()
    config = ABMV4Config(start_year=args.start_year, end_year=args.end_year)
    paths = ABMV4Paths()

    if args.dry_run:
        rows = build_path_audit_rows(paths, config.start_year, config.end_year)
        print(format_path_audit_table(rows))
        return

    if args.build_state:
        if not args.create_output_dirs:
            raise SystemExit("--build-state requires --create-output-dirs to write outputs.")
        result = build_state_panel(
            paths=paths,
            start_year=config.start_year,
            end_year=config.end_year,
            write_outputs=True,
            epsilon=config.epsilon,
        )
        print(f"Selected state source: {result.selected_source}")
        print(f"State panel rows: {result.state_panel.height}")
        print(f"State panel path: {result.output_path}")
        return

    if args.create_output_dirs:
        paths.ensure_output_directories()

    report = inspect_base_model_readiness(paths=paths, config=config)

    print(report.state_source.message)
    print(f"Can run base model: {report.can_run_base_model}")


if __name__ == "__main__":
    main()
