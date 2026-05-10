from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.abm_v3.leontief.scenarios.base import BehaviouralScenarioContext
from src.abm_v3.paths import ABMV3Paths


class BehaviouralScenarioOutputWriter:
    """Write behavioural Leontief scenario outputs and diagnostics."""

    def __init__(self, paths: ABMV3Paths) -> None:
        self.paths = paths

    def write_all(
        self,
        year: int,
        scenario_name: str,
        context: BehaviouralScenarioContext,
        selected_nodes: pd.DataFrame,
        node_comparison: pd.DataFrame,
        aggregate: pd.DataFrame,
        summary: pd.DataFrame,
        scenario_output: pd.DataFrame,
    ) -> dict[str, Path]:
        self.paths.behavioural_leontief_scenario_outputs_dir.mkdir(parents=True, exist_ok=True)
        self.paths.behavioural_leontief_scenario_diagnostics_dir.mkdir(parents=True, exist_ok=True)
        self.paths.behavioural_leontief_scenario_selected_nodes_dir.mkdir(parents=True, exist_ok=True)

        output_path = self.paths.behavioural_leontief_scenario_output_path(
            year,
            scenario_name,
            context.mode,
            context.input_panel_orientation,
        )
        summary_path = self.paths.behavioural_leontief_scenario_summary_path(
            year,
            scenario_name,
            context.mode,
            context.input_panel_orientation,
        )
        node_comparison_path = self.paths.behavioural_leontief_scenario_node_comparison_path(
            year,
            scenario_name,
            context.mode,
            context.input_panel_orientation,
        )
        selected_nodes_path = self.paths.behavioural_leontief_scenario_selected_nodes_path(year, scenario_name)
        aggregate_path = self.paths.behavioural_leontief_scenario_aggregate_path(
            year,
            scenario_name,
            context.mode,
            context.input_panel_orientation,
        )
        metadata_path = self.paths.behavioural_leontief_scenario_metadata_path(year, scenario_name)

        scenario_output.to_parquet(output_path, index=False)
        summary.to_csv(summary_path, index=False)
        node_comparison.to_csv(node_comparison_path, index=False)
        selected_nodes.to_csv(selected_nodes_path, index=False)
        aggregate.to_csv(aggregate_path, index=False)
        pd.DataFrame([asdict(context)]).to_csv(metadata_path, index=False)
        return {
            "output": output_path,
            "summary": summary_path,
            "node_comparison": node_comparison_path,
            "selected_nodes": selected_nodes_path,
            "aggregate": aggregate_path,
            "metadata": metadata_path,
        }
