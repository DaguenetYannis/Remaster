from __future__ import annotations

from pathlib import Path

from src.abm_v2.transition_diagnostics import TransitionDiagnostics
from src.abm_v2.transition_plots import TransitionPlotter


def main() -> None:
    scenario_directory = Path("data/abm/scenarios")
    output_directory = Path("outputs/abm_diagnostics")

    scenario_files = sorted(
        scenario_directory.glob("*_simulation_panel.parquet")
    )

    if not scenario_files:
        raise FileNotFoundError(
            f"No scenario files found in {scenario_directory}"
        )

    for scenario_file in scenario_files:
        scenario_name = scenario_file.name.replace(
            "_simulation_panel.parquet",
            "",
        )

        print(f"[INFO] Running transition diagnostics for {scenario_name}")

        diagnostics = TransitionDiagnostics(
            scenario_name=scenario_name,
            scenario_path=scenario_file,
            output_dir=output_directory,
        )

        tables = diagnostics.run()

        plotter = TransitionPlotter(
            scenario_name=scenario_name,
            tables=tables,
            output_dir=output_directory,
        )

        plotter.run()

        print(f"[OK] Finished {scenario_name}")

    print(f"[DONE] Outputs written to {output_directory}")


if __name__ == "__main__":
    main()