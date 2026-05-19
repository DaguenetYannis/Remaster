from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import polars as pl
import pytest

from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.reporting import ABMV4NarrativePlotBuilder


def _toy_paths() -> ABMV4Paths:
    return ABMV4Paths(project_root=Path("tmp") / "abm_v4_narrative_tests" / uuid4().hex)


def _write_required_inputs(paths: ABMV4Paths) -> None:
    paths.validation.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "rule_name": ["frontier_gap_readiness", "historical_frontier_gap_only"],
            "retained_as": ["aggregate_safe_baseline", "transition_mechanism_benchmark"],
            "scenario_use_status": ["not_scenario_ready", "not_scenario_ready"],
        }
    ).write_csv(paths.final_surviving_rule_comparison_path)
    pl.DataFrame(
        {
            "objective": ["transition_mechanism_validity", "aggregate_emissions_validity"],
            "frontier_gap_readiness_assessment": ["supported", "supported"],
            "historical_frontier_gap_only_assessment": ["supported", "limited"],
            "EID_assessment": ["diagnostic only", "not supported"],
            "Q_energy_mix_assessment": ["aggregate-only evidence", "diagnostic only"],
        }
    ).write_csv(paths.final_validation_objective_matrix_path)
    pl.DataFrame(
        {
            "mechanism": [
                "legacy_raw_log emissions rule",
                "fixed EID dampener",
                "adaptive EID dampener",
                "Q energy mix country-sector transition rule",
                "historical residual as scenario-facing rule",
            ],
            "test_result": ["rejected", "rejected_for_abm_v4_rule", "rejected", "rejected_for_abm_v4_node_level_rule", "diagnostic_only"],
            "reason_rejected_or_limited": ["sign issue", "failed as transition rule", "overfitting risk", "node-level quality limits", "not scenario-facing"],
            "retained_value": ["foil", "ontology evidence", "calibration warning", "aggregate diagnostics and ABM v5 evidence", "diagnostic only"],
            "future_use": ["none", "ABM v5 ontology", "ABM v5 warning", "cleaner energy data", "feature discovery"],
            "scenario_status": ["not_scenario_ready"] * 5,
        }
    ).write_csv(paths.final_rejected_mechanism_register_path)
    pl.DataFrame(
        {
            "readiness_dimension": [
                "emissions_transition_rule",
                "production_dynamics",
                "supplier_substitution",
                "capability_dynamics",
                "electricity_energy_system",
                "policy_institutional_variables",
                "data_quality",
                "validation_metrics",
                "interpretation_risk",
                "overall_scenario_readiness",
            ],
            "status": ["blocked", "blocked", "limited", "limited", "blocked", "blocked", "blocked", "limited", "blocked", "not_scenario_ready"],
            "blocking_issue": ["blocker"] * 10,
            "required_future_work": ["future work"] * 10,
            "interpretation": ["historical diagnostic only"] * 10,
        }
    ).write_csv(paths.final_scenario_readiness_assessment_path)
    pl.DataFrame(
        {
            "research_priority": ["energy/fuel structure"],
            "motivation_from_abm_v4": ["motivation"],
            "required_data": ["data"],
            "candidate_mechanism": ["mechanism"],
            "candidate_agent_type": ["agent"],
            "expected_validation_test": ["test"],
            "priority_level": ["high"],
        }
    ).write_csv(paths.final_abm_v5_research_agenda_path)
    pl.DataFrame(
        {
            "hypothesis": [
                "frontier_gap_readiness_aggregate_safe",
                "historical_frontier_gap_transition_benchmark",
                "EID_as_transition_dampener",
                "EID_as_ontology_signal",
                "Q_energy_mix_as_country_sector_rule",
                "Q_energy_mix_as_aggregate_diagnostic",
                "ABM_v4_scenario_ready",
            ],
            "status": ["supported", "supported", "not_supported", "supported", "not_supported", "supported", "not_supported"],
            "evidence": ["evidence"] * 7,
            "implication": ["implication"] * 7,
        }
    ).write_csv(paths.final_abm_v4_hypothesis_status_path)
    paths.final_model_boundary_statement_path.write_text("ABM v4 is not scenario-ready.", encoding="utf-8")


def _write_metric_inputs(paths: ABMV4Paths) -> None:
    pl.DataFrame(
        {
            "model_variant": ["frontier_gap_readiness", "historical_frontier_gap_only", "fixed_EID_diagnostic"],
            "mean_yearly_aggregate_emissions_pct_error": [0.40, 0.57, 0.90],
            "rEI_MAE": [0.14, 0.13, 0.30],
        }
    ).write_csv(paths.multiyear_base_model_comparison_csv_path)


def test_architecture_layer_source_table_is_created() -> None:
    paths = _toy_paths()
    _write_required_inputs(paths)
    table = ABMV4NarrativePlotBuilder(paths).build_architecture_layers_source()
    assert table.height == 5
    assert "Historical data inputs" in set(table["layer_name"])


def test_emissions_decomposition_source_table_is_created() -> None:
    table = ABMV4NarrativePlotBuilder(_toy_paths()).build_emissions_decomposition_logic_source()
    assert "E = X * EI" in set(table["formula"])
    assert table.filter(pl.col("component") == "Emissions-intensity effect").height == 1


def test_two_rule_metric_tradeoff_uses_quantitative_metrics_when_available() -> None:
    paths = _toy_paths()
    _write_required_inputs(paths)
    _write_metric_inputs(paths)
    table = ABMV4NarrativePlotBuilder(paths).build_two_rule_metric_tradeoff_source()
    row = table.filter(pl.col("model_variant") == "frontier_gap_readiness").row(0, named=True)
    assert row["transition_error_metric"] == 0.14
    assert row["aggregate_error_metric"] == 0.40


def test_two_rule_metric_tradeoff_writes_fallback_score_table_when_metrics_are_unavailable() -> None:
    paths = _toy_paths()
    _write_required_inputs(paths)
    table = ABMV4NarrativePlotBuilder(paths).build_two_rule_metric_tradeoff_source()
    assert "diagnostic score, not direct error metric" in set(table["interpretation"])


def test_mechanism_decision_tree_marks_eid_as_rejected_rule_but_ontology_evidence() -> None:
    paths = _toy_paths()
    _write_required_inputs(paths)
    table = ABMV4NarrativePlotBuilder(paths).build_mechanism_decision_tree_source()
    row = table.filter(pl.col("mechanism") == "fixed EID dampener").row(0, named=True)
    assert row["status"] == "rejected as ABM v4 rule"
    assert "ontology evidence" in row["retained_value"]


def test_q_energy_boundary_marks_aggregate_retained_and_node_level_rejected() -> None:
    paths = _toy_paths()
    _write_required_inputs(paths)
    table = ABMV4NarrativePlotBuilder(paths).build_q_energy_mix_quality_boundary_source()
    aggregate = table.filter(pl.col("check") == "aggregate diagnostic value").row(0, named=True)
    node_level = table.filter(pl.col("check") == "valid node-level shares").row(0, named=True)
    assert aggregate["status"] == "pass"
    assert node_level["status"] == "fail"


def test_scenario_readiness_checklist_uses_statuses_and_blockers() -> None:
    paths = _toy_paths()
    _write_required_inputs(paths)
    table = ABMV4NarrativePlotBuilder(paths).build_scenario_readiness_checklist_source()
    assert {"status", "blocking_issue", "required_future_work"}.issubset(set(table.columns))
    assert "blocked" in set(table["status"])


def test_abm_v5_roadmap_includes_all_five_required_branches() -> None:
    table = ABMV4NarrativePlotBuilder(_toy_paths()).build_abm_v4_to_v5_roadmap_source()
    assert {
        "Q energy mix aggregate-only",
        "electricity boundary",
        "EID ontology evidence",
        "historical production forcing",
        "high-emissions infrastructure",
    }.issubset(set(table["abm_v4_finding"]))


def test_hypothesis_status_table_groups_hypotheses_by_status() -> None:
    paths = _toy_paths()
    _write_required_inputs(paths)
    table = ABMV4NarrativePlotBuilder(paths).build_hypothesis_status_table_source()
    assert {"supported", "not_supported"}.issubset(set(table["status"]))


def test_all_plot_functions_write_png_and_svg() -> None:
    paths = _toy_paths()
    _write_required_inputs(paths)
    result = ABMV4NarrativePlotBuilder(paths).run(write_outputs=True)
    assert len([path for path in result.plot_paths if path.suffix == ".png"]) == 10
    assert len([path for path in result.plot_paths if path.suffix == ".svg"]) == 10


def test_narrative_plot_index_includes_generated_plots_and_source_tables() -> None:
    paths = _toy_paths()
    _write_required_inputs(paths)
    result = ABMV4NarrativePlotBuilder(paths).run(write_outputs=True)
    index = pl.read_csv(result.plot_index_path)
    assert "narrative_plot" in set(index["artifact_type"])
    assert "narrative_source_table" in set(index["artifact_type"])


def test_dry_run_creates_no_outputs() -> None:
    paths = _toy_paths()
    _write_required_inputs(paths)
    ABMV4NarrativePlotBuilder(paths).run(write_outputs=False)
    assert not paths.final.exists()
    assert not paths.outputs_plots_abm_v4_final_narrative.exists()


def test_missing_required_phase28_files_fail_clearly() -> None:
    paths = _toy_paths()
    paths.validation.mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError, match="required Phase 28 outputs are missing"):
        ABMV4NarrativePlotBuilder(paths).validate_required_inputs()


def test_no_latex_report_is_created() -> None:
    paths = _toy_paths()
    _write_required_inputs(paths)
    ABMV4NarrativePlotBuilder(paths).run(write_outputs=True)
    assert not list(paths.project_root.rglob("*.tex"))


def test_no_webpage_file_is_created() -> None:
    paths = _toy_paths()
    _write_required_inputs(paths)
    ABMV4NarrativePlotBuilder(paths).run(write_outputs=True)
    assert not list(paths.project_root.rglob("*.html"))
