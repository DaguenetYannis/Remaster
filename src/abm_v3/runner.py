from __future__ import annotations

import argparse
import logging

from src.abm_v3.config import ABMV3Config
from src.abm_v3.model import ABMV3Model
from src.abm_v3.paths import ABMV3Paths
from src.abm_v3.scenarios.registry import list_scenarios


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ABM v3 scaffold CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--start-year", type=int, default=1995)
    calibrate.add_argument("--end-year", type=int, default=2016)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--split-year", type=int, default=2008)

    simulate = subparsers.add_parser("simulate")
    simulate.add_argument("--scenario", default="baseline_continuation")
    simulate.add_argument("--start-year", type=int, default=2017)
    simulate.add_argument("--end-year", type=int, default=2027)

    subparsers.add_parser("compare-scenarios")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = build_parser().parse_args()
    model = ABMV3Model(config=ABMV3Config(), paths=ABMV3Paths())

    if args.command == "calibrate":
        result = model.fit_historical(args.start_year, args.end_year)
        print(f"Calibration scaffold: {result}")
    elif args.command == "validate":
        result = model.validate_historical(args.split_year)
        print(f"Validation scaffold: {result}")
    elif args.command == "simulate":
        result = model.simulate(args.start_year, args.end_year, scenario=args.scenario)
        print(
            "Simulation scaffold: "
            f"scenario={args.scenario}, years={args.start_year}-{args.end_year}, rows={len(result)}"
        )
    elif args.command == "compare-scenarios":
        print(f"Scenario comparison scaffold. Registered scenarios: {list_scenarios()}")


if __name__ == "__main__":
    main()
