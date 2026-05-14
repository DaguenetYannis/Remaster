from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.paths import PROJECT_ROOT


@dataclass(frozen=True)
class ABMV4Paths:
    """Centralized ABM v4 paths without implicit directory creation."""

    project_root: Path = PROJECT_ROOT

    @property
    def data_root(self) -> Path:
        return self.project_root / "data"

    @property
    def data_abm_v4(self) -> Path:
        return self.data_root / "abm_v4"

    @property
    def inputs(self) -> Path:
        return self.data_abm_v4 / "inputs"

    @property
    def interim(self) -> Path:
        return self.data_abm_v4 / "interim"

    @property
    def diagnostics(self) -> Path:
        return self.data_abm_v4 / "diagnostics"

    @property
    def simulations(self) -> Path:
        return self.data_abm_v4 / "simulations"

    @property
    def scenarios(self) -> Path:
        return self.data_abm_v4 / "scenarios"

    @property
    def validation(self) -> Path:
        return self.data_abm_v4 / "validation"

    @property
    def data_abm_v3(self) -> Path:
        return self.data_root / "abm_v3"

    @property
    def data_final(self) -> Path:
        return self.data_root / "final"

    @property
    def data_metrics(self) -> Path:
        return self.data_root / "metrics"

    @property
    def data_atlas(self) -> Path:
        return self.data_root / "atlas"

    @property
    def data_abm_legacy(self) -> Path:
        return self.data_root / "abm"

    def output_directories(self) -> tuple[Path, ...]:
        """Return ABM v4 output directories without creating them."""
        return (
            self.inputs,
            self.interim,
            self.diagnostics,
            self.simulations,
            self.scenarios,
            self.validation,
        )

    def ensure_output_directories(self) -> None:
        """Create only ABM v4 output directories."""
        for output_directory in self.output_directories():
            output_directory.mkdir(parents=True, exist_ok=True)

    def state_panel_path(self, start_year: int, end_year: int) -> Path:
        """Return the ABM v4 state panel output path."""
        return self.inputs / f"abm_v4_state_panel_{start_year}_{end_year}.parquet"

    @property
    def ecosystem_mapping_path(self) -> Path:
        return self.inputs / "ecosystem_mapping.csv"

    @property
    def ecosystem_adjacency_path(self) -> Path:
        return self.inputs / "ecosystem_adjacency.csv"

    @property
    def ecosystem_assignment_report_path(self) -> Path:
        return self.diagnostics / "ecosystem_assignment_report.csv"

    @property
    def ecosystem_sector_coverage_path(self) -> Path:
        return self.diagnostics / "ecosystem_sector_coverage.csv"

    @property
    def historical_supplier_edges_path(self) -> Path:
        return self.interim / "historical_supplier_edges.parquet"

    @property
    def raw_t_supplier_edges_path(self) -> Path:
        return self.interim / "historical_supplier_edges_raw_T.parquet"

    @property
    def supplier_candidates_historical_top_path(self) -> Path:
        return self.interim / "supplier_candidates_historical_top.parquet"

    @property
    def supplier_pool_same_sector_path(self) -> Path:
        return self.interim / "supplier_pool_same_sector.parquet"

    @property
    def supplier_pool_ecosystem_path(self) -> Path:
        return self.interim / "supplier_pool_ecosystem.parquet"

    @property
    def supplier_opportunity_sets_path(self) -> Path:
        return self.interim / "supplier_opportunity_sets.parquet"

    @property
    def supplier_edge_report_path(self) -> Path:
        return self.diagnostics / "supplier_edge_report.csv"

    @property
    def raw_t_supplier_edge_report_path(self) -> Path:
        return self.diagnostics / "supplier_edge_raw_T_report.csv"

    @property
    def supplier_edge_source_comparison_path(self) -> Path:
        return self.diagnostics / "supplier_edge_source_comparison.csv"

    @property
    def supplier_candidate_base_report_path(self) -> Path:
        return self.diagnostics / "supplier_candidate_base_report.csv"

    @property
    def supplier_opportunity_set_report_path(self) -> Path:
        return self.diagnostics / "supplier_opportunity_set_report.csv"

    @property
    def supplier_edge_schema_report_path(self) -> Path:
        return self.diagnostics / "supplier_edge_schema_report.csv"

    @property
    def emissions_decomposition_base_path(self) -> Path:
        return self.diagnostics / "emissions_decomposition_base.csv"

    def abm_v3_state_candidates(self, start_year: int, end_year: int) -> tuple[Path, ...]:
        """Return likely current ABM v3 state/input panels in priority order."""
        return (
            self.data_abm_v3 / "inputs" / f"abm_v3_historical_panel_{start_year}_{end_year}_transpose_row_fd_without_inventory.parquet",
            self.data_abm_v3 / "inputs" / f"abm_v3_historical_panel_{start_year}_{end_year}.parquet",
            self.data_abm_v3 / "phase_space" / f"abm_v3_phase_space_state_panel_{start_year}_{end_year}.parquet",
        )

    @property
    def final_state_candidates(self) -> tuple[Path, ...]:
        """Return final-panel fallback candidates in priority order."""
        return (
            self.data_final / "eora_atlas_dynamic_panel.parquet",
            self.data_final / "eora_atlas_merged.parquet",
        )

    @property
    def legacy_state_candidates(self) -> tuple[Path, ...]:
        """Return earlier ABM fallback candidates."""
        return (
            self.data_abm_legacy / "agents_panel.parquet",
        )

    @property
    def edge_candidates(self) -> tuple[Path, ...]:
        """Return likely supplier edge sources in priority order."""
        return (
            self.data_abm_v3 / "leontief" / "behavioural" / "scenarios" / "diagnostics",
            self.data_abm_v3 / "leontief",
            self.data_abm_legacy / "edges_panel.parquet",
        )

    def state_source_candidates(self, start_year: int, end_year: int) -> tuple[Path, ...]:
        """Return all state source candidates in the ABM v4 priority order."""
        return (
            *self.abm_v3_state_candidates(start_year, end_year),
            *self.final_state_candidates,
            *self.legacy_state_candidates,
        )

    def existing_state_sources(self, start_year: int, end_year: int) -> tuple[Path, ...]:
        """Return state source candidates that exist locally."""
        return tuple(
            candidate_path
            for candidate_path in self.state_source_candidates(start_year, end_year)
            if candidate_path.exists()
        )
