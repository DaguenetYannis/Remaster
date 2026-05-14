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
    def supplier_initial_weights_path(self) -> Path:
        return self.interim / "supplier_initial_weights.parquet"

    @property
    def supplier_rewiring_flags_path(self) -> Path:
        return self.interim / "supplier_rewiring_flags.parquet"

    @property
    def supplier_updated_weights_path(self) -> Path:
        return self.interim / "supplier_updated_weights.parquet"

    @property
    def capability_exposure_panel_path(self) -> Path:
        return self.interim / "capability_exposure_panel.parquet"

    @property
    def capability_update_panel_path(self) -> Path:
        return self.interim / "capability_update_panel.parquet"

    @property
    def production_feasibility_panel_path(self) -> Path:
        return self.interim / "production_feasibility_panel.parquet"

    @property
    def emissions_update_panel_path(self) -> Path:
        return self.interim / "emissions_update_panel.parquet"

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
    def supplier_rewiring_report_path(self) -> Path:
        return self.diagnostics / "supplier_rewiring_report.csv"

    @property
    def capability_update_report_path(self) -> Path:
        return self.diagnostics / "capability_update_report.csv"

    @property
    def capability_join_report_path(self) -> Path:
        return self.diagnostics / "capability_join_report.csv"

    @property
    def capability_coverage_by_year_path(self) -> Path:
        return self.diagnostics / "capability_coverage_by_year.csv"

    @property
    def capability_coverage_by_sector_path(self) -> Path:
        return self.diagnostics / "capability_coverage_by_sector.csv"

    @property
    def io_capability_lambda_calibration_path(self) -> Path:
        return self.diagnostics / "io_capability_lambda_calibration.csv"

    @property
    def io_capability_model_report_path(self) -> Path:
        return self.diagnostics / "io_capability_model_report.csv"

    @property
    def io_capability_coverage_by_sector_path(self) -> Path:
        return self.diagnostics / "io_capability_coverage_by_sector.csv"

    @property
    def io_capability_coverage_by_source_path(self) -> Path:
        return self.diagnostics / "io_capability_coverage_by_source.csv"

    @property
    def io_capability_robustness_path(self) -> Path:
        return self.diagnostics / "io_capability_robustness.csv"

    @property
    def io_capability_threshold_sensitivity_path(self) -> Path:
        return self.diagnostics / "io_capability_threshold_sensitivity.csv"

    @property
    def io_downstream_exposure_audit_path(self) -> Path:
        return self.diagnostics / "io_downstream_exposure_audit.csv"

    @property
    def production_feasibility_report_path(self) -> Path:
        return self.diagnostics / "production_feasibility_report.csv"

    @property
    def emissions_update_report_path(self) -> Path:
        return self.diagnostics / "emissions_update_report.csv"

    @property
    def emissions_historical_rEI_summary_path(self) -> Path:
        return self.diagnostics / "emissions_historical_rEI_summary.csv"

    @property
    def emissions_sector_background_trend_path(self) -> Path:
        return self.diagnostics / "emissions_sector_background_trend.csv"

    @property
    def emissions_frontier_gap_report_path(self) -> Path:
        return self.diagnostics / "emissions_frontier_gap_report.csv"

    @property
    def emissions_transition_comparison_path(self) -> Path:
        return self.diagnostics / "emissions_transition_comparison.csv"

    @property
    def one_step_base_validation_report_csv_path(self) -> Path:
        return self.validation / "one_step_base_validation_report.csv"

    @property
    def one_step_base_validation_report_md_path(self) -> Path:
        return self.validation / "one_step_base_validation_report.md"

    @property
    def one_step_base_status_json_path(self) -> Path:
        return self.validation / "one_step_base_status.json"

    @property
    def multiyear_error_panel_path(self) -> Path:
        return self.validation / "multiyear_error_panel.parquet"

    @property
    def multiyear_error_summary_path(self) -> Path:
        return self.validation / "multiyear_error_summary.csv"

    @property
    def multiyear_error_by_sector_path(self) -> Path:
        return self.validation / "multiyear_error_by_sector.csv"

    @property
    def multiyear_error_by_country_path(self) -> Path:
        return self.validation / "multiyear_error_by_country.csv"

    @property
    def multiyear_error_by_ecosystem_path(self) -> Path:
        return self.validation / "multiyear_error_by_ecosystem.csv"

    @property
    def multiyear_error_by_capability_source_path(self) -> Path:
        return self.validation / "multiyear_error_by_capability_source.csv"

    @property
    def multiyear_calibration_targets_path(self) -> Path:
        return self.validation / "multiyear_calibration_targets.csv"

    @property
    def multiyear_validation_report_md_path(self) -> Path:
        return self.validation / "multiyear_validation_report.md"

    @property
    def emissions_calibration_dataset_path(self) -> Path:
        return self.validation / "emissions_calibration_dataset.parquet"

    @property
    def emissions_parameter_search_results_path(self) -> Path:
        return self.validation / "emissions_parameter_search_results.csv"

    @property
    def emissions_best_parameters_path(self) -> Path:
        return self.validation / "emissions_best_parameters.json"

    @property
    def emissions_calibration_validation_summary_path(self) -> Path:
        return self.validation / "emissions_calibration_validation_summary.csv"

    @property
    def emissions_calibration_by_sector_path(self) -> Path:
        return self.validation / "emissions_calibration_by_sector.csv"

    @property
    def emissions_calibration_by_capability_source_path(self) -> Path:
        return self.validation / "emissions_calibration_by_capability_source.csv"

    @property
    def emissions_model_comparison_path(self) -> Path:
        return self.validation / "emissions_model_comparison.csv"

    @property
    def emissions_parameter_plausibility_path(self) -> Path:
        return self.validation / "emissions_parameter_plausibility.csv"

    @property
    def emissions_calibration_report_path(self) -> Path:
        return self.validation / "emissions_calibration_report.md"

    @property
    def emissions_hypothesis_diagnosis_path(self) -> Path:
        return self.validation / "emissions_hypothesis_diagnosis.csv"

    @property
    def emissions_target_horizon_panel_path(self) -> Path:
        return self.validation / "emissions_target_horizon_panel.parquet"

    @property
    def emissions_target_horizon_summary_path(self) -> Path:
        return self.validation / "emissions_target_horizon_summary.csv"

    @property
    def emissions_predictor_screening_path(self) -> Path:
        return self.validation / "emissions_predictor_screening.csv"

    @property
    def emissions_sector_dominance_diagnostics_path(self) -> Path:
        return self.validation / "emissions_sector_dominance_diagnostics.csv"

    @property
    def emissions_capability_source_diagnostics_path(self) -> Path:
        return self.validation / "emissions_capability_source_diagnostics.csv"

    @property
    def emissions_readiness_threshold_diagnostics_path(self) -> Path:
        return self.validation / "emissions_readiness_threshold_diagnostics.csv"

    @property
    def emissions_frontier_specification_diagnostics_path(self) -> Path:
        return self.validation / "emissions_frontier_specification_diagnostics.csv"

    @property
    def emissions_macro_shock_diagnostics_path(self) -> Path:
        return self.validation / "emissions_macro_shock_diagnostics.csv"

    @property
    def emissions_hypothesis_diagnostic_report_path(self) -> Path:
        return self.validation / "emissions_hypothesis_diagnostic_report.md"

    @property
    def emissions_transition_variant_results_path(self) -> Path:
        return self.validation / "emissions_transition_variant_results.csv"

    @property
    def emissions_transition_variant_by_sector_family_path(self) -> Path:
        return self.validation / "emissions_transition_variant_by_sector_family.csv"

    @property
    def emissions_transition_variant_by_capability_source_path(self) -> Path:
        return self.validation / "emissions_transition_variant_by_capability_source.csv"

    @property
    def emissions_transition_variant_best_parameters_path(self) -> Path:
        return self.validation / "emissions_transition_variant_best_parameters.json"

    @property
    def emissions_transition_variant_recommendation_path(self) -> Path:
        return self.validation / "emissions_transition_variant_recommendation.csv"

    @property
    def emissions_transition_variant_report_path(self) -> Path:
        return self.validation / "emissions_transition_variant_report.md"

    @property
    def base_multiyear_state_panel_path(self) -> Path:
        return self.simulations / "base_multiyear_state_panel.parquet"

    @property
    def base_multiyear_state_panel_historical_frontier_gap_path(self) -> Path:
        return self.simulations / "base_multiyear_state_panel_historical_frontier_gap.parquet"

    @property
    def base_multiyear_summary_panel_path(self) -> Path:
        return self.simulations / "base_multiyear_summary_panel.csv"

    @property
    def base_multiyear_summary_panel_historical_frontier_gap_path(self) -> Path:
        return self.simulations / "base_multiyear_summary_panel_historical_frontier_gap.csv"

    @property
    def base_multiyear_validation_report_path(self) -> Path:
        return self.diagnostics / "base_multiyear_validation_report.csv"

    @property
    def base_multiyear_validation_report_historical_frontier_gap_csv_path(self) -> Path:
        return self.validation / "base_multiyear_validation_report_historical_frontier_gap.csv"

    @property
    def base_multiyear_validation_report_historical_frontier_gap_md_path(self) -> Path:
        return self.validation / "base_multiyear_validation_report_historical_frontier_gap.md"

    @property
    def base_multiyear_yearly_diagnostics_path(self) -> Path:
        return self.diagnostics / "base_multiyear_yearly_diagnostics.csv"

    @property
    def base_multiyear_yearly_diagnostics_historical_frontier_gap_path(self) -> Path:
        return self.diagnostics / "base_multiyear_yearly_diagnostics_historical_frontier_gap.csv"

    @property
    def multiyear_base_model_comparison_csv_path(self) -> Path:
        return self.validation / "multiyear_base_model_comparison.csv"

    @property
    def multiyear_base_model_comparison_md_path(self) -> Path:
        return self.validation / "multiyear_base_model_comparison.md"

    @property
    def transition_rule_error_decomposition_path(self) -> Path:
        return self.validation / "transition_rule_error_decomposition.csv"

    @property
    def transition_rule_sign_failure_panel_path(self) -> Path:
        return self.validation / "transition_rule_sign_failure_panel.parquet"

    @property
    def transition_rule_sign_failure_by_year_path(self) -> Path:
        return self.validation / "transition_rule_sign_failure_by_year.csv"

    @property
    def transition_rule_sign_failure_by_sector_path(self) -> Path:
        return self.validation / "transition_rule_sign_failure_by_sector.csv"

    @property
    def transition_rule_sign_failure_by_country_path(self) -> Path:
        return self.validation / "transition_rule_sign_failure_by_country.csv"

    @property
    def transition_rule_sign_failure_by_ecosystem_path(self) -> Path:
        return self.validation / "transition_rule_sign_failure_by_ecosystem.csv"

    @property
    def transition_rule_sign_failure_by_capability_source_path(self) -> Path:
        return self.validation / "transition_rule_sign_failure_by_capability_source.csv"

    @property
    def transition_rule_sign_failure_by_decile_path(self) -> Path:
        return self.validation / "transition_rule_sign_failure_by_decile.csv"

    @property
    def transition_rule_aggregate_contribution_path(self) -> Path:
        return self.validation / "transition_rule_aggregate_contribution.csv"

    @property
    def transition_rule_hypothesis_tests_path(self) -> Path:
        return self.validation / "transition_rule_hypothesis_tests.csv"

    @property
    def transition_rule_error_tradeoff_report_path(self) -> Path:
        return self.validation / "transition_rule_error_tradeoff_report.md"

    @property
    def high_emissions_concentration_diagnostic_path(self) -> Path:
        return self.validation / "high_emissions_concentration_diagnostic.csv"

    @property
    def electricity_sector_dampening_diagnostic_path(self) -> Path:
        return self.validation / "electricity_sector_dampening_diagnostic.csv"

    @property
    def china_electricity_transition_diagnostic_path(self) -> Path:
        return self.validation / "china_electricity_transition_diagnostic.csv"

    @property
    def readiness_dampening_diagnostic_path(self) -> Path:
        return self.validation / "readiness_dampening_diagnostic.csv"

    @property
    def simplified_model_selection_tradeoff_path(self) -> Path:
        return self.validation / "simplified_model_selection_tradeoff.csv"

    @property
    def phase17_recommendation_path(self) -> Path:
        return self.validation / "phase17_recommendation.csv"

    @property
    def phase17_high_emissions_dampening_report_path(self) -> Path:
        return self.validation / "phase17_high_emissions_dampening_report.md"

    @property
    def electricity_node_inventory_path(self) -> Path:
        return self.validation / "electricity_node_inventory.csv"

    @property
    def china_electricity_observed_series_audit_path(self) -> Path:
        return self.validation / "china_electricity_observed_series_audit.csv"

    @property
    def china_electricity_model_series_audit_path(self) -> Path:
        return self.validation / "china_electricity_model_series_audit.csv"

    @property
    def electricity_anomaly_flags_path(self) -> Path:
        return self.validation / "electricity_anomaly_flags.csv"

    @property
    def electricity_cross_country_comparison_path(self) -> Path:
        return self.validation / "electricity_cross_country_comparison.csv"

    @property
    def electricity_data_audit_recommendation_path(self) -> Path:
        return self.validation / "electricity_data_audit_recommendation.csv"

    @property
    def electricity_data_audit_report_path(self) -> Path:
        return self.validation / "electricity_data_audit_report.md"

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
