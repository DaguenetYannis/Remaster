from __future__ import annotations

import pandas as pd

from src.abm_v3.leontief.coefficients import LeontiefYearData
from src.abm_v3.leontief.propagation import LeontiefPropagationResult
from src.abm_v3.leontief.viability import LeontiefViabilityDiagnostics
from src.abm_v3.paths import ABMV3Paths


class LeontiefOutputWriter:
    """Write Leontief propagation outputs and diagnostics."""

    def __init__(self, paths: ABMV3Paths) -> None:
        self.paths = paths

    def write_all(
        self,
        year_data: LeontiefYearData,
        result: LeontiefPropagationResult,
        node_comparison: pd.DataFrame,
        summary: pd.DataFrame,
    ) -> dict[str, object]:
        """Write the full diagnostic bundle for one year."""
        self.paths.leontief_outputs_dir.mkdir(parents=True, exist_ok=True)
        self.paths.leontief_diagnostics_dir.mkdir(parents=True, exist_ok=True)

        iterative_output_path = self.paths.leontief_iterative_output_path(year_data.year, year_data.mode)
        summary_path = self.paths.leontief_summary_path(year_data.year, year_data.mode)
        node_comparison_path = self.paths.leontief_node_comparison_path(year_data.year, year_data.mode)
        rounds_path = self.paths.leontief_rounds_path(year_data.year, year_data.mode)
        invalid_path = self.paths.leontief_invalid_output_columns_path(year_data.year, year_data.mode)
        mode_diagnostics_path = self.paths.leontief_mode_diagnostics_path(year_data.year, year_data.mode)
        excluded_fd_columns_path = self.paths.leontief_excluded_fd_columns_path(year_data.year, year_data.mode)
        rescaled_columns_path = self.paths.leontief_rescaled_columns_path(year_data.year, year_data.mode)

        node_comparison.to_parquet(iterative_output_path, index=False)
        summary.to_csv(summary_path, index=False)
        node_comparison.to_csv(node_comparison_path, index=False)
        result.round_summaries.to_csv(rounds_path, index=False)
        self._write_invalid_output_columns(year_data.invalid_output_columns, invalid_path)
        self._write_optional_frame(year_data.mode_diagnostics, mode_diagnostics_path)
        self._write_optional_frame(year_data.excluded_fd_columns, excluded_fd_columns_path)
        self._write_optional_frame(year_data.rescaled_columns, rescaled_columns_path)

        return {
            "iterative_output": iterative_output_path,
            "summary": summary_path,
            "node_comparison": node_comparison_path,
            "rounds": rounds_path,
            "invalid_output_columns": invalid_path,
            "mode_diagnostics": mode_diagnostics_path,
            "excluded_fd_columns": excluded_fd_columns_path,
            "rescaled_columns": rescaled_columns_path,
        }

    def _write_invalid_output_columns(self, invalid_output_columns: pd.DataFrame | None, path: object) -> None:
        columns = [
            "Year",
            "country_sector",
            "Country",
            "Country_detail",
            "Category",
            "Sector",
            "X_observed",
            "reason",
        ]
        if invalid_output_columns is None or invalid_output_columns.empty:
            pd.DataFrame(columns=columns).to_csv(path, index=False)
            return
        invalid_output_columns.reindex(columns=columns).to_csv(path, index=False)

    def write_viability(self, diagnostics: LeontiefViabilityDiagnostics) -> dict[str, object]:
        """Write coefficient viability diagnostics for one year."""
        self.paths.leontief_diagnostics_dir.mkdir(parents=True, exist_ok=True)
        mode = self._diagnostic_mode(diagnostics)
        summary_path = self.paths.leontief_viability_summary_path(diagnostics.year, mode)
        columns_path = self.paths.leontief_viability_columns_path(diagnostics.year, mode)
        negative_flows_path = self.paths.leontief_negative_flows_path(diagnostics.year, mode)
        spectral_path = self.paths.leontief_spectral_diagnostics_path(diagnostics.year, mode)
        top_nodes_path = self.paths.leontief_top_unstable_nodes_path(diagnostics.year, mode)

        diagnostics.summary.to_csv(summary_path, index=False)
        diagnostics.columns.to_csv(columns_path, index=False)
        diagnostics.negative_flows.to_csv(negative_flows_path, index=False)
        diagnostics.spectral.to_csv(spectral_path, index=False)
        if diagnostics.top_unstable_nodes is None:
            diagnostics.columns.head(0).to_csv(top_nodes_path, index=False)
        else:
            diagnostics.top_unstable_nodes.to_csv(top_nodes_path, index=False)

        return {
            "viability_summary": summary_path,
            "viability_columns": columns_path,
            "negative_flows": negative_flows_path,
            "spectral": spectral_path,
            "top_unstable_nodes": top_nodes_path,
        }

    def _write_optional_frame(self, frame: pd.DataFrame | None, path: object) -> None:
        if frame is None:
            pd.DataFrame().to_csv(path, index=False)
            return
        frame.to_csv(path, index=False)

    def _diagnostic_mode(self, diagnostics: LeontiefViabilityDiagnostics) -> str:
        if "mode" in diagnostics.summary.columns and len(diagnostics.summary):
            return str(diagnostics.summary["mode"].iloc[0])
        if "mode" in diagnostics.columns.columns and len(diagnostics.columns):
            return str(diagnostics.columns["mode"].iloc[0])
        return "raw"
