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

    @property
    def final(self) -> Path:
        return self.data_abm_v4 / "final"

    @property
    def final_tables(self) -> Path:
        return self.final / "tables"

    @property
    def final_plots(self) -> Path:
        return self.final / "plots"

    @property
    def final_tables_narrative(self) -> Path:
        return self.final / "tables_narrative"

    @property
    def final_plots_narrative(self) -> Path:
        return self.final / "plots_narrative"

    @property
    def final_tables_polished(self) -> Path:
        return self.final / "tables_polished"

    @property
    def final_plots_polished(self) -> Path:
        return self.final / "plots_polished"

    @property
    def outputs_root(self) -> Path:
        return self.project_root / "outputs"

    @property
    def outputs_plots_abm_v4_final(self) -> Path:
        return self.outputs_root / "plots" / "abm_v4_final"

    @property
    def outputs_plots_abm_v4_final_narrative(self) -> Path:
        return self.outputs_root / "plots" / "abm_v4_final_narrative"

    @property
    def outputs_plots_abm_v4_final_polished(self) -> Path:
        return self.outputs_root / "plots" / "abm_v4_final_polished"

    def final_artifact_directories(self) -> tuple[Path, ...]:
        """Return final ABM v4 artifact directories without creating them."""
        return (
            self.final,
            self.final_tables,
            self.final_plots,
            self.outputs_plots_abm_v4_final,
        )

    def ensure_final_artifact_directories(self) -> None:
        """Create only final ABM v4 artifact directories."""
        for output_directory in self.final_artifact_directories():
            output_directory.mkdir(parents=True, exist_ok=True)

    def final_narrative_artifact_directories(self) -> tuple[Path, ...]:
        """Return narrative final artifact directories without creating them."""
        return (
            self.final,
            self.final_tables_narrative,
            self.final_plots_narrative,
            self.outputs_plots_abm_v4_final_narrative,
        )

    def ensure_final_narrative_artifact_directories(self) -> None:
        """Create only final ABM v4 narrative artifact directories."""
        for output_directory in self.final_narrative_artifact_directories():
            output_directory.mkdir(parents=True, exist_ok=True)

    def final_polished_artifact_directories(self) -> tuple[Path, ...]:
        """Return polished final artifact directories without creating them."""
        return (
            self.final,
            self.final_tables_polished,
            self.final_plots_polished,
            self.outputs_plots_abm_v4_final_polished,
        )

    def ensure_final_polished_artifact_directories(self) -> None:
        """Create only final ABM v4 polished artifact directories."""
        for output_directory in self.final_polished_artifact_directories():
            output_directory.mkdir(parents=True, exist_ok=True)

    @property
    def final_artifact_index_path(self) -> Path:
        return self.final / "abm_v4_final_artifact_index.csv"

    @property
    def final_narrative_plot_index_path(self) -> Path:
        return self.final / "abm_v4_narrative_plot_index.csv"

    @property
    def final_polished_plot_index_path(self) -> Path:
        return self.final / "abm_v4_polished_plot_index.csv"

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
    def base_multiyear_state_panel_EID_diagnostic_path(self) -> Path:
        return self.simulations / "base_multiyear_state_panel_EID_diagnostic.parquet"

    @property
    def base_multiyear_summary_panel_path(self) -> Path:
        return self.simulations / "base_multiyear_summary_panel.csv"

    @property
    def base_multiyear_summary_panel_historical_frontier_gap_path(self) -> Path:
        return self.simulations / "base_multiyear_summary_panel_historical_frontier_gap.csv"

    @property
    def base_multiyear_summary_panel_EID_diagnostic_path(self) -> Path:
        return self.simulations / "base_multiyear_summary_panel_EID_diagnostic.csv"

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
    def base_multiyear_EID_diagnostic_validation_report_path(self) -> Path:
        return self.diagnostics / "base_multiyear_EID_diagnostic_validation_report.csv"

    @property
    def base_multiyear_yearly_diagnostics_path(self) -> Path:
        return self.diagnostics / "base_multiyear_yearly_diagnostics.csv"

    @property
    def base_multiyear_yearly_diagnostics_historical_frontier_gap_path(self) -> Path:
        return self.diagnostics / "base_multiyear_yearly_diagnostics_historical_frontier_gap.csv"

    @property
    def base_multiyear_EID_diagnostic_yearly_diagnostics_path(self) -> Path:
        return self.diagnostics / "base_multiyear_EID_diagnostic_yearly_diagnostics.csv"

    @property
    def multiyear_EID_diagnostic_error_panel_path(self) -> Path:
        return self.validation / "multiyear_EID_diagnostic_error_panel.parquet"

    @property
    def multiyear_EID_diagnostic_error_summary_path(self) -> Path:
        return self.validation / "multiyear_EID_diagnostic_error_summary.csv"

    @property
    def multiyear_EID_diagnostic_by_sector_path(self) -> Path:
        return self.validation / "multiyear_EID_diagnostic_by_sector.csv"

    @property
    def multiyear_EID_diagnostic_by_country_path(self) -> Path:
        return self.validation / "multiyear_EID_diagnostic_by_country.csv"

    @property
    def multiyear_EID_diagnostic_by_electricity_path(self) -> Path:
        return self.validation / "multiyear_EID_diagnostic_by_electricity.csv"

    @property
    def multiyear_EID_diagnostic_china_electricity_path(self) -> Path:
        return self.validation / "multiyear_EID_diagnostic_china_electricity.csv"

    @property
    def multiyear_EID_diagnostic_by_EID_decile_path(self) -> Path:
        return self.validation / "multiyear_EID_diagnostic_by_EID_decile.csv"

    @property
    def multiyear_EID_diagnostic_by_subtype_path(self) -> Path:
        return self.validation / "multiyear_EID_diagnostic_by_subtype.csv"

    @property
    def multiyear_EID_diagnostic_pseudo_agent_sensitivity_path(self) -> Path:
        return self.validation / "multiyear_EID_diagnostic_pseudo_agent_sensitivity.csv"

    @property
    def multiyear_EID_diagnostic_comparison_path(self) -> Path:
        return self.validation / "multiyear_EID_diagnostic_comparison.csv"

    @property
    def multiyear_EID_diagnostic_recommendation_path(self) -> Path:
        return self.validation / "multiyear_EID_diagnostic_recommendation.csv"

    @property
    def multiyear_EID_diagnostic_report_path(self) -> Path:
        return self.validation / "multiyear_EID_diagnostic_report.md"

    @property
    def multiyear_EID_diagnostic_mechanism_audit_path(self) -> Path:
        return self.validation / "multiyear_EID_diagnostic_mechanism_audit.csv"

    @property
    def multiyear_EID_diagnostic_abm_v5_implications_path(self) -> Path:
        return self.validation / "multiyear_EID_diagnostic_abm_v5_implications.csv"

    @property
    def adaptive_EID_parameter_grid_path(self) -> Path:
        return self.validation / "adaptive_EID_parameter_grid.csv"

    @property
    def adaptive_EID_calibration_windows_path(self) -> Path:
        return self.validation / "adaptive_EID_calibration_windows.csv"

    @property
    def adaptive_EID_calibration_results_path(self) -> Path:
        return self.validation / "adaptive_EID_calibration_results.csv"

    @property
    def adaptive_EID_validation_panel_path(self) -> Path:
        return self.validation / "adaptive_EID_validation_panel.parquet"

    @property
    def adaptive_EID_model_comparison_path(self) -> Path:
        return self.validation / "adaptive_EID_model_comparison.csv"

    @property
    def adaptive_EID_parameter_stability_path(self) -> Path:
        return self.validation / "adaptive_EID_parameter_stability.csv"

    @property
    def adaptive_EID_by_subtype_path(self) -> Path:
        return self.validation / "adaptive_EID_by_subtype.csv"

    @property
    def adaptive_EID_pseudo_agent_sensitivity_path(self) -> Path:
        return self.validation / "adaptive_EID_pseudo_agent_sensitivity.csv"

    @property
    def adaptive_EID_hypothesis_tests_path(self) -> Path:
        return self.validation / "adaptive_EID_hypothesis_tests.csv"

    @property
    def adaptive_EID_recommendation_path(self) -> Path:
        return self.validation / "adaptive_EID_recommendation.csv"

    @property
    def adaptive_EID_report_path(self) -> Path:
        return self.validation / "adaptive_EID_report.md"

    @property
    def q_energy_source_inventory_path(self) -> Path:
        return self.validation / "q_energy_source_inventory.csv"

    @property
    def q_energy_row_mapping_path(self) -> Path:
        return self.validation / "q_energy_row_mapping.csv"

    @property
    def q_energy_mix_panel_path(self) -> Path:
        return self.validation / "q_energy_mix_panel.parquet"

    @property
    def q_energy_mix_quality_audit_path(self) -> Path:
        return self.validation / "q_energy_mix_quality_audit.csv"

    @property
    def q_energy_mix_quality_by_year_path(self) -> Path:
        return self.validation / "q_energy_mix_quality_by_year.csv"

    @property
    def q_energy_mix_quality_by_sector_path(self) -> Path:
        return self.validation / "q_energy_mix_quality_by_sector.csv"

    @property
    def q_energy_mix_quality_by_country_path(self) -> Path:
        return self.validation / "q_energy_mix_quality_by_country.csv"

    @property
    def q_energy_mix_aggregate_plausibility_path(self) -> Path:
        return self.validation / "q_energy_mix_aggregate_plausibility.csv"

    @property
    def q_energy_mix_china_electricity_audit_path(self) -> Path:
        return self.validation / "q_energy_mix_china_electricity_audit.csv"

    @property
    def q_energy_mix_transition_error_panel_path(self) -> Path:
        return self.validation / "q_energy_mix_transition_error_panel.parquet"

    @property
    def q_energy_mix_predictor_screening_path(self) -> Path:
        return self.validation / "q_energy_mix_predictor_screening.csv"

    @property
    def q_energy_mix_by_subtype_path(self) -> Path:
        return self.validation / "q_energy_mix_by_subtype.csv"

    @property
    def q_energy_mix_hypothesis_tests_path(self) -> Path:
        return self.validation / "q_energy_mix_hypothesis_tests.csv"

    @property
    def q_energy_mix_recommendation_path(self) -> Path:
        return self.validation / "q_energy_mix_recommendation.csv"

    @property
    def q_energy_mix_report_path(self) -> Path:
        return self.validation / "q_energy_mix_report.md"

    @property
    def final_abm_v4_input_availability_path(self) -> Path:
        return self.validation / "final_abm_v4_input_availability.csv"

    @property
    def final_surviving_rule_comparison_path(self) -> Path:
        return self.validation / "final_surviving_rule_comparison.csv"

    @property
    def final_validation_objective_matrix_path(self) -> Path:
        return self.validation / "final_validation_objective_matrix.csv"

    @property
    def final_rejected_mechanism_register_path(self) -> Path:
        return self.validation / "final_rejected_mechanism_register.csv"

    @property
    def final_model_boundary_statement_path(self) -> Path:
        return self.validation / "final_model_boundary_statement.md"

    @property
    def final_scenario_readiness_assessment_path(self) -> Path:
        return self.validation / "final_scenario_readiness_assessment.csv"

    @property
    def final_abm_v5_research_agenda_path(self) -> Path:
        return self.validation / "final_abm_v5_research_agenda.csv"

    @property
    def final_abm_v4_hypothesis_status_path(self) -> Path:
        return self.validation / "final_abm_v4_hypothesis_status.csv"

    @property
    def final_abm_v4_consolidation_report_path(self) -> Path:
        return self.validation / "final_abm_v4_consolidation_report.md"

    @property
    def final_abm_v4_portfolio_summary_path(self) -> Path:
        return self.validation / "final_abm_v4_portfolio_summary.md"

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
    def raw_eora_electricity_source_inventory_path(self) -> Path:
        return self.validation / "raw_eora_electricity_source_inventory.csv"

    @property
    def raw_eora_china_electricity_series_by_source_path(self) -> Path:
        return self.validation / "raw_eora_china_electricity_series_by_source.csv"

    @property
    def raw_eora_china_electricity_cross_source_comparison_path(self) -> Path:
        return self.validation / "raw_eora_china_electricity_cross_source_comparison.csv"

    @property
    def raw_eora_electricity_scaling_audit_path(self) -> Path:
        return self.validation / "raw_eora_electricity_scaling_audit.csv"

    @property
    def raw_eora_electricity_mapping_audit_path(self) -> Path:
        return self.validation / "raw_eora_electricity_mapping_audit.csv"

    @property
    def raw_eora_electricity_breakpoint_audit_path(self) -> Path:
        return self.validation / "raw_eora_electricity_breakpoint_audit.csv"

    @property
    def raw_eora_major_electricity_comparison_path(self) -> Path:
        return self.validation / "raw_eora_major_electricity_comparison.csv"

    @property
    def raw_eora_electricity_data_audit_recommendation_path(self) -> Path:
        return self.validation / "raw_eora_electricity_data_audit_recommendation.csv"

    @property
    def raw_eora_electricity_data_audit_report_path(self) -> Path:
        return self.validation / "raw_eora_electricity_data_audit_report.md"

    @property
    def electricity_transition_target_diagnostics_path(self) -> Path:
        return self.validation / "electricity_transition_target_diagnostics.csv"

    @property
    def electricity_transition_rule_comparison_path(self) -> Path:
        return self.validation / "electricity_transition_rule_comparison.csv"

    @property
    def electricity_transition_rule_by_country_path(self) -> Path:
        return self.validation / "electricity_transition_rule_by_country.csv"

    @property
    def electricity_transition_rule_by_year_path(self) -> Path:
        return self.validation / "electricity_transition_rule_by_year.csv"

    @property
    def electricity_transition_rule_by_decile_path(self) -> Path:
        return self.validation / "electricity_transition_rule_by_decile.csv"

    @property
    def electricity_transition_rule_by_jump_status_path(self) -> Path:
        return self.validation / "electricity_transition_rule_by_jump_status.csv"

    @property
    def china_electricity_rule_comparison_path(self) -> Path:
        return self.validation / "china_electricity_rule_comparison.csv"

    @property
    def electricity_transition_regime_recommendation_path(self) -> Path:
        return self.validation / "electricity_transition_regime_recommendation.csv"

    @property
    def electricity_transition_regime_report_path(self) -> Path:
        return self.validation / "electricity_transition_regime_report.md"

    @property
    def structural_signature_metric_inventory_path(self) -> Path:
        return self.validation / "structural_signature_metric_inventory.csv"

    @property
    def structural_signature_node_year_panel_path(self) -> Path:
        return self.validation / "structural_signature_node_year_panel.parquet"

    @property
    def structural_signature_node_panel_path(self) -> Path:
        return self.validation / "structural_signature_node_panel.parquet"

    @property
    def structural_signature_label_summary_path(self) -> Path:
        return self.validation / "structural_signature_label_summary.csv"

    @property
    def electricity_structural_signature_contrast_path(self) -> Path:
        return self.validation / "electricity_structural_signature_contrast.csv"

    @property
    def structural_signature_metric_screening_path(self) -> Path:
        return self.validation / "structural_signature_metric_screening.csv"

    @property
    def structural_signature_non_electricity_lookalikes_path(self) -> Path:
        return self.validation / "structural_signature_non_electricity_lookalikes.csv"

    @property
    def candidate_transition_inertia_proxies_path(self) -> Path:
        return self.validation / "candidate_transition_inertia_proxies.csv"

    @property
    def structural_signature_recommendation_path(self) -> Path:
        return self.validation / "structural_signature_recommendation.csv"

    @property
    def structural_signature_report_path(self) -> Path:
        return self.validation / "structural_signature_report.md"

    @property
    def essential_input_supplier_buyer_panel_path(self) -> Path:
        return self.validation / "essential_input_supplier_buyer_panel.parquet"

    @property
    def essential_input_node_metrics_path(self) -> Path:
        return self.validation / "essential_input_node_metrics.csv"

    @property
    def electricity_dependence_signature_contrast_path(self) -> Path:
        return self.validation / "electricity_dependence_signature_contrast.csv"

    @property
    def dependence_vs_symptom_metric_comparison_path(self) -> Path:
        return self.validation / "dependence_vs_symptom_metric_comparison.csv"

    @property
    def dependence_metric_screening_path(self) -> Path:
        return self.validation / "dependence_metric_screening.csv"

    @property
    def essential_input_non_electricity_lookalikes_path(self) -> Path:
        return self.validation / "essential_input_non_electricity_lookalikes.csv"

    @property
    def candidate_structural_dependence_proxies_path(self) -> Path:
        return self.validation / "candidate_structural_dependence_proxies.csv"

    @property
    def essential_input_dependence_recommendation_path(self) -> Path:
        return self.validation / "essential_input_dependence_recommendation.csv"

    @property
    def essential_input_dependence_report_path(self) -> Path:
        return self.validation / "essential_input_dependence_report.md"

    @property
    def essential_input_dampener_candidate_grid_path(self) -> Path:
        return self.validation / "essential_input_dampener_candidate_grid.csv"

    @property
    def essential_input_dampener_scores_path(self) -> Path:
        return self.validation / "essential_input_dampener_scores.csv"

    @property
    def essential_input_historical_residual_panel_path(self) -> Path:
        return self.validation / "essential_input_historical_residual_panel.parquet"

    @property
    def essential_input_historical_residual_summary_path(self) -> Path:
        return self.validation / "essential_input_historical_residual_summary.csv"

    @property
    def essential_input_dampener_validation_results_path(self) -> Path:
        return self.validation / "essential_input_dampener_validation_results.csv"

    @property
    def essential_input_dampener_by_sector_path(self) -> Path:
        return self.validation / "essential_input_dampener_by_sector.csv"

    @property
    def essential_input_dampener_by_country_path(self) -> Path:
        return self.validation / "essential_input_dampener_by_country.csv"

    @property
    def essential_input_dampener_by_electricity_path(self) -> Path:
        return self.validation / "essential_input_dampener_by_electricity.csv"

    @property
    def essential_input_dampener_china_electricity_path(self) -> Path:
        return self.validation / "essential_input_dampener_china_electricity.csv"

    @property
    def essential_input_dampener_by_EID_decile_path(self) -> Path:
        return self.validation / "essential_input_dampener_by_EID_decile.csv"

    @property
    def essential_input_dampener_mechanism_decomposition_path(self) -> Path:
        return self.validation / "essential_input_dampener_mechanism_decomposition.csv"

    @property
    def essential_input_dampener_abm_v5_implications_path(self) -> Path:
        return self.validation / "essential_input_dampener_abm_v5_implications.csv"

    @property
    def essential_input_dampener_recommendation_path(self) -> Path:
        return self.validation / "essential_input_dampener_recommendation.csv"

    @property
    def essential_input_dampener_report_path(self) -> Path:
        return self.validation / "essential_input_dampener_report.md"

    @property
    def eid_high_node_heterogeneity_panel_path(self) -> Path:
        return self.validation / "eid_high_node_heterogeneity_panel.csv"

    @property
    def eid_subtype_composition_path(self) -> Path:
        return self.validation / "eid_subtype_composition.csv"

    @property
    def eid_dampener_performance_by_subtype_path(self) -> Path:
        return self.validation / "eid_dampener_performance_by_subtype.csv"

    @property
    def eid_dampener_failure_modes_path(self) -> Path:
        return self.validation / "eid_dampener_failure_modes.csv"

    @property
    def eid_pseudo_agent_audit_path(self) -> Path:
        return self.validation / "eid_pseudo_agent_audit.csv"

    @property
    def eid_abm_v5_agent_type_candidates_path(self) -> Path:
        return self.validation / "eid_abm_v5_agent_type_candidates.csv"

    @property
    def eid_failure_mode_recommendation_path(self) -> Path:
        return self.validation / "eid_failure_mode_recommendation.csv"

    @property
    def eid_failure_mode_report_path(self) -> Path:
        return self.validation / "eid_failure_mode_report.md"

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
