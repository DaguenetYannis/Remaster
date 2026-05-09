from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ABMV3Paths:
    """Centralized paths for ABM v3 data, metrics, references, and outputs."""

    project_root: Path = Path(".")

    @property
    def data_root(self) -> Path:
        return self.project_root / "data"

    @property
    def parquet_root(self) -> Path:
        return self.data_root / "parquet"

    @property
    def raw_root(self) -> Path:
        return self.data_root / "raw"

    @property
    def metrics_root(self) -> Path:
        return self.data_root / "metrics"

    @property
    def final_root(self) -> Path:
        return self.data_root / "final"

    @property
    def atlas_processed_root(self) -> Path:
        return self.data_root / "atlas" / "processed"

    @property
    def eora_atlas_merged_file(self) -> Path:
        return self.final_root / "eora_atlas_merged.parquet"

    @property
    def eora_metrics_panel_file(self) -> Path:
        return self.final_root / "eora_metrics_panel.parquet"

    @property
    def atlas_capability_file(self) -> Path:
        return (
            self.atlas_processed_root
            / "atlas_eora26_sector_capabilities_1995_2016.parquet"
        )

    @property
    def abm_v3_output_root(self) -> Path:
        return self.data_root / "abm_v3"

    @property
    def abm_v3_input_root(self) -> Path:
        return self.abm_v3_output_root / "inputs"

    @property
    def ei_transition_dir(self) -> Path:
        return self.abm_v3_output_root / "ei_transition"

    @property
    def ei_transition_inputs_dir(self) -> Path:
        return self.ei_transition_dir / "inputs"

    @property
    def ei_transition_diagnostics_dir(self) -> Path:
        return self.ei_transition_dir / "diagnostics"

    @property
    def ei_transition_models_dir(self) -> Path:
        return self.ei_transition_dir / "models"

    @property
    def ei_transition_predictions_dir(self) -> Path:
        return self.ei_transition_dir / "predictions"

    def ei_transition_panel_path(self, start_year: int, end_year: int) -> Path:
        return self.ei_transition_inputs_dir / f"ei_transition_panel_{start_year}_{end_year}.parquet"

    def ei_transition_sample_report_path(self, start_year: int, end_year: int) -> Path:
        return self.ei_transition_diagnostics_dir / f"ei_transition_sample_report_{start_year}_{end_year}.csv"

    def ei_transition_sample_report_by_year_path(self, start_year: int, end_year: int) -> Path:
        return self.ei_transition_diagnostics_dir / f"ei_transition_sample_report_by_year_{start_year}_{end_year}.csv"

    def ei_transition_model_scores_path(self, start_year: int, end_year: int) -> Path:
        return self.ei_transition_diagnostics_dir / f"ei_transition_model_scores_{start_year}_{end_year}.csv"

    def ei_transition_coefficients_path(self, start_year: int, end_year: int) -> Path:
        return self.ei_transition_models_dir / f"ei_transition_coefficients_{start_year}_{end_year}.csv"

    def ei_transition_expected_signs_path(self, start_year: int, end_year: int) -> Path:
        return self.ei_transition_diagnostics_dir / f"ei_transition_expected_signs_{start_year}_{end_year}.csv"

    def ei_transition_predictions_path(self, start_year: int, end_year: int) -> Path:
        return self.ei_transition_predictions_dir / f"ei_transition_predictions_{start_year}_{end_year}.parquet"

    def abm_v3_historical_panel_file(self, start_year: int, end_year: int) -> Path:
        return (
            self.abm_v3_input_root
            / f"abm_v3_historical_panel_{start_year}_{end_year}.parquet"
        )

    def abm_v3_corrected_historical_panel_file(
        self,
        start_year: int,
        end_year: int,
        orientation: str = "transpose_row_fd_without_inventory",
    ) -> Path:
        return (
            self.abm_v3_input_root
            / f"abm_v3_historical_panel_{start_year}_{end_year}_{orientation}.parquet"
        )

    def abm_v3_historical_panel_file_for_orientation(
        self,
        start_year: int,
        end_year: int,
        input_panel_orientation: str | None = None,
    ) -> Path:
        """Return the historical input panel path for an explicit orientation."""
        if input_panel_orientation in {None, "current_column"}:
            return self.abm_v3_historical_panel_file(start_year, end_year)
        if input_panel_orientation == "transpose_row_fd_without_inventory":
            return self.abm_v3_corrected_historical_panel_file(
                start_year,
                end_year,
                input_panel_orientation,
            )
        raise ValueError(f"Unknown input panel orientation: {input_panel_orientation}")

    def format_leontief_suffix(
        self,
        year: int,
        mode: str,
        input_panel_orientation: str | None = None,
    ) -> str:
        """Build a readable single-year suffix for Leontief artifacts."""
        if input_panel_orientation is None:
            return f"{year}_{mode}"
        return f"{year}_{mode}__{input_panel_orientation}"

    def format_leontief_range_suffix(
        self,
        start_year: int,
        end_year: int,
        mode: str | None = None,
        input_panel_orientation: str | None = None,
    ) -> str:
        """Build a readable range suffix for Leontief summary artifacts."""
        year_part = f"{start_year}_{end_year}"
        if mode is None:
            return year_part
        if input_panel_orientation is None:
            return f"{year_part}_{mode}"
        return f"{year_part}_{mode}__{input_panel_orientation}"

    @property
    def leontief_dir(self) -> Path:
        return self.abm_v3_output_root / "leontief"

    @property
    def leontief_outputs_dir(self) -> Path:
        return self.leontief_pure_outputs_dir

    @property
    def leontief_diagnostics_dir(self) -> Path:
        return self.leontief_pure_propagation_diagnostics_dir

    @property
    def leontief_pure_dir(self) -> Path:
        return self.leontief_dir / "pure"

    @property
    def leontief_pure_outputs_dir(self) -> Path:
        return self.leontief_pure_dir / "outputs"

    @property
    def leontief_pure_diagnostics_dir(self) -> Path:
        return self.leontief_pure_dir / "diagnostics"

    @property
    def leontief_pure_propagation_diagnostics_dir(self) -> Path:
        return self.leontief_pure_diagnostics_dir / "propagation"

    @property
    def leontief_pure_viability_diagnostics_dir(self) -> Path:
        return self.leontief_pure_diagnostics_dir / "viability"

    @property
    def leontief_pure_mode_comparison_diagnostics_dir(self) -> Path:
        return self.leontief_pure_diagnostics_dir / "mode_comparison"

    @property
    def leontief_pure_orientation_audit_diagnostics_dir(self) -> Path:
        return self.leontief_pure_diagnostics_dir / "orientation_audit"

    def leontief_iterative_output_path(
        self,
        year: int,
        mode: str = "raw",
        input_panel_orientation: str | None = None,
    ) -> Path:
        suffix = self.format_leontief_suffix(year, mode, input_panel_orientation)
        return self.leontief_pure_outputs_dir / f"iterative_output_{suffix}.parquet"

    def leontief_summary_path(self, year: int, mode: str = "raw", input_panel_orientation: str | None = None) -> Path:
        suffix = self.format_leontief_suffix(year, mode, input_panel_orientation)
        return self.leontief_pure_propagation_diagnostics_dir / f"summary_{suffix}.csv"

    def leontief_node_comparison_path(self, year: int, mode: str = "raw", input_panel_orientation: str | None = None) -> Path:
        suffix = self.format_leontief_suffix(year, mode, input_panel_orientation)
        return self.leontief_pure_propagation_diagnostics_dir / f"node_comparison_{suffix}.csv"

    def leontief_rounds_path(self, year: int, mode: str = "raw", input_panel_orientation: str | None = None) -> Path:
        suffix = self.format_leontief_suffix(year, mode, input_panel_orientation)
        return self.leontief_pure_propagation_diagnostics_dir / f"rounds_{suffix}.csv"

    def leontief_invalid_output_columns_path(self, year: int, mode: str = "raw", input_panel_orientation: str | None = None) -> Path:
        suffix = self.format_leontief_suffix(year, mode, input_panel_orientation)
        return self.leontief_pure_viability_diagnostics_dir / f"invalid_output_columns_{suffix}.csv"

    def leontief_viability_summary_path(self, year: int, mode: str = "raw", input_panel_orientation: str | None = None) -> Path:
        suffix = self.format_leontief_suffix(year, mode, input_panel_orientation)
        return self.leontief_pure_viability_diagnostics_dir / f"summary_{suffix}.csv"

    def leontief_viability_columns_path(self, year: int, mode: str = "raw", input_panel_orientation: str | None = None) -> Path:
        suffix = self.format_leontief_suffix(year, mode, input_panel_orientation)
        return self.leontief_pure_viability_diagnostics_dir / f"columns_{suffix}.csv"

    def leontief_negative_flows_path(self, year: int, mode: str = "raw", input_panel_orientation: str | None = None) -> Path:
        suffix = self.format_leontief_suffix(year, mode, input_panel_orientation)
        return self.leontief_pure_viability_diagnostics_dir / f"negative_flows_{suffix}.csv"

    def leontief_spectral_diagnostics_path(self, year: int, mode: str = "raw", input_panel_orientation: str | None = None) -> Path:
        suffix = self.format_leontief_suffix(year, mode, input_panel_orientation)
        return self.leontief_pure_viability_diagnostics_dir / f"spectral_diagnostics_{suffix}.csv"

    def leontief_top_unstable_nodes_path(self, year: int, mode: str = "raw", input_panel_orientation: str | None = None) -> Path:
        suffix = self.format_leontief_suffix(year, mode, input_panel_orientation)
        return self.leontief_pure_viability_diagnostics_dir / f"top_unstable_nodes_{suffix}.csv"

    def leontief_mode_diagnostics_path(self, year: int, mode: str = "raw", input_panel_orientation: str | None = None) -> Path:
        suffix = self.format_leontief_suffix(year, mode, input_panel_orientation)
        return self.leontief_pure_mode_comparison_diagnostics_dir / f"mode_diagnostics_{suffix}.csv"

    def leontief_excluded_fd_columns_path(self, year: int, mode: str = "raw", input_panel_orientation: str | None = None) -> Path:
        suffix = self.format_leontief_suffix(year, mode, input_panel_orientation)
        return self.leontief_pure_mode_comparison_diagnostics_dir / f"excluded_fd_columns_{suffix}.csv"

    def leontief_rescaled_columns_path(self, year: int, mode: str = "raw", input_panel_orientation: str | None = None) -> Path:
        suffix = self.format_leontief_suffix(year, mode, input_panel_orientation)
        return self.leontief_pure_mode_comparison_diagnostics_dir / f"rescaled_columns_{suffix}.csv"

    def leontief_mode_comparison_path(self, year: int) -> Path:
        return self.leontief_pure_mode_comparison_diagnostics_dir / f"summary_{year}.csv"

    def leontief_mode_comparison_range_path(self, start_year: int, end_year: int) -> Path:
        suffix = self.format_leontief_range_suffix(start_year, end_year)
        return self.leontief_pure_mode_comparison_diagnostics_dir / f"summary_{suffix}.csv"

    def leontief_orientation_summary_path(self, year: int) -> Path:
        return self.leontief_pure_orientation_audit_diagnostics_dir / f"summary_{year}.csv"

    def leontief_orientation_node_comparison_path(self, year: int) -> Path:
        return self.leontief_pure_orientation_audit_diagnostics_dir / f"node_comparison_{year}.csv"

    def leontief_orientation_suspicious_nodes_path(self, year: int) -> Path:
        return self.leontief_pure_orientation_audit_diagnostics_dir / f"suspicious_nodes_{year}.csv"

    def leontief_orientation_summary_range_path(self, start_year: int, end_year: int) -> Path:
        suffix = self.format_leontief_range_suffix(start_year, end_year)
        return self.leontief_pure_orientation_audit_diagnostics_dir / f"summary_{suffix}.csv"

    @property
    def behavioural_leontief_dir(self) -> Path:
        return self.leontief_dir / "behavioural"

    @property
    def leontief_behavioural_outputs_dir(self) -> Path:
        return self.behavioural_leontief_dir / "outputs"

    @property
    def leontief_behavioural_diagnostics_dir(self) -> Path:
        return self.behavioural_leontief_dir / "diagnostics"

    @property
    def leontief_behavioural_summary_diagnostics_dir(self) -> Path:
        return self.leontief_behavioural_diagnostics_dir / "summary"

    @property
    def leontief_behavioural_rounds_diagnostics_dir(self) -> Path:
        return self.leontief_behavioural_diagnostics_dir / "rounds"

    @property
    def leontief_behavioural_node_comparison_diagnostics_dir(self) -> Path:
        return self.leontief_behavioural_diagnostics_dir / "node_comparison"

    @property
    def leontief_behavioural_node_rounds_diagnostics_dir(self) -> Path:
        return self.leontief_behavioural_diagnostics_dir / "node_rounds"

    @property
    def behavioural_leontief_outputs_dir(self) -> Path:
        return self.leontief_behavioural_outputs_dir

    @property
    def behavioural_leontief_diagnostics_dir(self) -> Path:
        return self.leontief_behavioural_diagnostics_dir

    @property
    def leontief_comparisons_dir(self) -> Path:
        return self.leontief_dir / "comparisons"

    @property
    def leontief_pure_vs_behavioural_comparisons_dir(self) -> Path:
        return self.leontief_comparisons_dir / "pure_vs_behavioural"

    @property
    def leontief_current_vs_corrected_orientation_comparisons_dir(self) -> Path:
        return self.leontief_comparisons_dir / "current_vs_corrected_orientation"

    @property
    def leontief_yearly_summaries_comparisons_dir(self) -> Path:
        return self.leontief_comparisons_dir / "yearly_summaries"

    def behavioural_leontief_output_path(self, year: int, mode: str = "fd_without_inventory", input_panel_orientation: str | None = None) -> Path:
        suffix = self.format_leontief_suffix(year, mode, input_panel_orientation)
        return self.leontief_behavioural_outputs_dir / f"output_{suffix}.parquet"

    def behavioural_leontief_summary_path(self, year: int, mode: str = "fd_without_inventory", input_panel_orientation: str | None = None) -> Path:
        suffix = self.format_leontief_suffix(year, mode, input_panel_orientation)
        return self.leontief_behavioural_summary_diagnostics_dir / f"summary_{suffix}.csv"

    def behavioural_leontief_node_comparison_path(self, year: int, mode: str = "fd_without_inventory", input_panel_orientation: str | None = None) -> Path:
        suffix = self.format_leontief_suffix(year, mode, input_panel_orientation)
        return self.leontief_behavioural_node_comparison_diagnostics_dir / f"node_comparison_{suffix}.csv"

    def behavioural_leontief_rounds_path(self, year: int, mode: str = "fd_without_inventory", input_panel_orientation: str | None = None) -> Path:
        suffix = self.format_leontief_suffix(year, mode, input_panel_orientation)
        return self.leontief_behavioural_rounds_diagnostics_dir / f"rounds_{suffix}.csv"

    def behavioural_leontief_node_rounds_path(self, year: int, mode: str = "fd_without_inventory", input_panel_orientation: str | None = None) -> Path:
        suffix = self.format_leontief_suffix(year, mode, input_panel_orientation)
        return self.leontief_behavioural_node_rounds_diagnostics_dir / f"node_rounds_{suffix}.csv"

    def metric_file(self, year: int, metric_name: str) -> Path:
        return self.metrics_root / str(year) / f"{metric_name}_{year}.parquet"

    def et_file(self, year: int) -> Path:
        return self.metric_file(year, "et")

    def ei_file(self, year: int) -> Path:
        return self.metric_file(year, "ei")

    def greenness_file(self, year: int) -> Path:
        return self.metric_file(year, "greenness")

    def centrality_file(self, year: int) -> Path:
        return self.metric_file(year, "centrality")

    def efficiency_file(self, year: int) -> Path:
        return self.metric_file(year, "efficiency")

    def eora_matrix_file(self, year: int, matrix_name: str) -> Path:
        return self.parquet_root / str(year) / f"{matrix_name}.parquet"

    def label_file(self, year: int, label_name: str) -> Path:
        return self.raw_root / str(year) / f"{label_name}.txt"
