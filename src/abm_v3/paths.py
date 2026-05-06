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
