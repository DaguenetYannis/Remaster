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

    def abm_v3_historical_panel_file(self, start_year: int, end_year: int) -> Path:
        return (
            self.abm_v3_input_root
            / f"abm_v3_historical_panel_{start_year}_{end_year}.parquet"
        )

    @property
    def leontief_dir(self) -> Path:
        return self.abm_v3_output_root / "leontief"

    @property
    def leontief_outputs_dir(self) -> Path:
        return self.leontief_dir / "outputs"

    @property
    def leontief_diagnostics_dir(self) -> Path:
        return self.leontief_dir / "diagnostics"

    def leontief_iterative_output_path(self, year: int, mode: str = "raw") -> Path:
        return self.leontief_outputs_dir / f"leontief_iterative_output_{year}_{mode}.parquet"

    def leontief_summary_path(self, year: int, mode: str = "raw") -> Path:
        return self.leontief_diagnostics_dir / f"leontief_propagation_summary_{year}_{mode}.csv"

    def leontief_node_comparison_path(self, year: int, mode: str = "raw") -> Path:
        return self.leontief_diagnostics_dir / f"leontief_node_comparison_{year}_{mode}.csv"

    def leontief_rounds_path(self, year: int, mode: str = "raw") -> Path:
        return self.leontief_diagnostics_dir / f"leontief_rounds_{year}_{mode}.csv"

    def leontief_invalid_output_columns_path(self, year: int, mode: str = "raw") -> Path:
        return self.leontief_diagnostics_dir / f"leontief_invalid_output_columns_{year}_{mode}.csv"

    def leontief_viability_summary_path(self, year: int, mode: str = "raw") -> Path:
        return self.leontief_diagnostics_dir / f"leontief_coefficient_viability_summary_{year}_{mode}.csv"

    def leontief_viability_columns_path(self, year: int, mode: str = "raw") -> Path:
        return self.leontief_diagnostics_dir / f"leontief_coefficient_viability_columns_{year}_{mode}.csv"

    def leontief_negative_flows_path(self, year: int, mode: str = "raw") -> Path:
        return self.leontief_diagnostics_dir / f"leontief_negative_flows_{year}_{mode}.csv"

    def leontief_spectral_diagnostics_path(self, year: int, mode: str = "raw") -> Path:
        return self.leontief_diagnostics_dir / f"leontief_spectral_diagnostics_{year}_{mode}.csv"

    def leontief_top_unstable_nodes_path(self, year: int, mode: str = "raw") -> Path:
        return self.leontief_diagnostics_dir / f"leontief_top_unstable_nodes_{year}_{mode}.csv"

    def leontief_mode_diagnostics_path(self, year: int, mode: str = "raw") -> Path:
        return self.leontief_diagnostics_dir / f"leontief_mode_diagnostics_{year}_{mode}.csv"

    def leontief_excluded_fd_columns_path(self, year: int, mode: str = "raw") -> Path:
        return self.leontief_diagnostics_dir / f"leontief_excluded_fd_columns_{year}_{mode}.csv"

    def leontief_rescaled_columns_path(self, year: int, mode: str = "raw") -> Path:
        return self.leontief_diagnostics_dir / f"leontief_rescaled_columns_{year}_{mode}.csv"

    def leontief_mode_comparison_path(self, year: int) -> Path:
        return self.leontief_diagnostics_dir / f"leontief_mode_comparison_{year}.csv"

    def leontief_mode_comparison_range_path(self, start_year: int, end_year: int) -> Path:
        return self.leontief_diagnostics_dir / f"leontief_mode_comparison_{start_year}_{end_year}.csv"

    def leontief_orientation_summary_path(self, year: int) -> Path:
        return self.leontief_diagnostics_dir / f"leontief_orientation_summary_{year}.csv"

    def leontief_orientation_node_comparison_path(self, year: int) -> Path:
        return self.leontief_diagnostics_dir / f"leontief_orientation_node_comparison_{year}.csv"

    def leontief_orientation_suspicious_nodes_path(self, year: int) -> Path:
        return self.leontief_diagnostics_dir / f"leontief_orientation_suspicious_nodes_{year}.csv"

    def leontief_orientation_summary_range_path(self, start_year: int, end_year: int) -> Path:
        return self.leontief_diagnostics_dir / f"leontief_orientation_summary_{start_year}_{end_year}.csv"

    @property
    def behavioural_leontief_dir(self) -> Path:
        return self.leontief_dir / "behavioural"

    @property
    def behavioural_leontief_outputs_dir(self) -> Path:
        return self.behavioural_leontief_dir / "outputs"

    @property
    def behavioural_leontief_diagnostics_dir(self) -> Path:
        return self.behavioural_leontief_dir / "diagnostics"

    def behavioural_leontief_output_path(self, year: int, mode: str = "fd_without_inventory") -> Path:
        return self.behavioural_leontief_outputs_dir / f"behavioural_leontief_output_{year}_{mode}.parquet"

    def behavioural_leontief_summary_path(self, year: int, mode: str = "fd_without_inventory") -> Path:
        return self.behavioural_leontief_diagnostics_dir / f"behavioural_leontief_summary_{year}_{mode}.csv"

    def behavioural_leontief_node_comparison_path(self, year: int, mode: str = "fd_without_inventory") -> Path:
        return self.behavioural_leontief_diagnostics_dir / f"behavioural_leontief_node_comparison_{year}_{mode}.csv"

    def behavioural_leontief_rounds_path(self, year: int, mode: str = "fd_without_inventory") -> Path:
        return self.behavioural_leontief_diagnostics_dir / f"behavioural_leontief_rounds_{year}_{mode}.csv"

    def behavioural_leontief_node_rounds_path(self, year: int, mode: str = "fd_without_inventory") -> Path:
        return self.behavioural_leontief_diagnostics_dir / f"behavioural_leontief_node_rounds_{year}_{mode}.csv"

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
