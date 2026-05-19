from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import polars as pl

from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.reporting import ABMV4PolishedPlotBuilder, POLISHED_PLOT_NAMES


def _toy_paths() -> ABMV4Paths:
    return ABMV4Paths(project_root=Path("tmp") / "abm_v4_polished_tests" / uuid4().hex)


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
                "electricity_energy_system",
                "policy_institutional_variables",
                "data_quality",
                "overall_scenario_readiness",
            ],
            "status": ["blocked", "blocked", "blocked", "blocked", "blocked", "not_scenario_ready"],
            "blocking_issue": ["blocker"] * 6,
            "required_future_work": ["future work"] * 6,
            "interpretation": ["historical diagnostic only"] * 6,
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
    pl.DataFrame(
        {
            "model_variant": ["frontier_gap_readiness", "historical_frontier_gap_only"],
            "mean_yearly_aggregate_emissions_pct_error": [0.40, 0.57],
            "rEI_MAE": [0.14, 0.13],
        }
    ).write_csv(paths.multiyear_base_model_comparison_csv_path)


def test_architecture_arrows_and_order_are_top_to_bottom() -> None:
    table = ABMV4PolishedPlotBuilder(_toy_paths()).build_architecture_layers_polished_source()
    assert table["layer_order"].to_list() == [1, 2, 3, 4, 5]
    assert set(table["flow_direction"]) == {"top_to_bottom"}


def test_two_rule_scorecard_is_created_without_ordinal_bars() -> None:
    paths = _toy_paths()
    _write_required_inputs(paths)
    table = ABMV4PolishedPlotBuilder(paths).build_two_rule_scorecard_source()
    assert "validation_dimension" in table.columns
    assert "bar" not in " ".join(table.columns).lower()


def test_scorecard_has_both_rules_and_blocks_scenario_readiness() -> None:
    paths = _toy_paths()
    _write_required_inputs(paths)
    table = ABMV4PolishedPlotBuilder(paths).build_two_rule_scorecard_source()
    row = table.filter(pl.col("validation_dimension") == "Scenario readiness").row(0, named=True)
    assert row["frontier_gap_readiness_assessment"] == "blocked"
    assert row["historical_frontier_gap_only_assessment"] == "blocked"


def test_mechanism_grid_marks_eid_as_rejected_rule_but_evidence() -> None:
    paths = _toy_paths()
    _write_required_inputs(paths)
    table = ABMV4PolishedPlotBuilder(paths).build_mechanism_status_grid_source()
    eid = table.filter(pl.col("mechanism") == "fixed EID dampener").row(0, named=True)
    assert eid["final_status"] == "rejected as ABM v4 rule, retained as evidence"
    assert "ontology evidence" in eid["retained_value"]


def test_mechanism_grid_marks_q_energy_as_aggregate_v5_evidence() -> None:
    paths = _toy_paths()
    _write_required_inputs(paths)
    table = ABMV4PolishedPlotBuilder(paths).build_mechanism_status_grid_source()
    q_energy = table.filter(pl.col("mechanism") == "Q energy mix country-sector rule").row(0, named=True)
    assert q_energy["final_status"] == "rejected as ABM v4 rule, retained as evidence"
    assert "ABM v5 evidence" in q_energy["retained_value"]


def test_china_source_does_not_label_observed_rei_as_model_error() -> None:
    paths = _toy_paths()
    _write_required_inputs(paths)
    table = ABMV4PolishedPlotBuilder(paths).build_china_electricity_boundary_case_polished_source()
    assert "model error" not in str(table["observed_series_label"][0]).lower()


def test_scenario_readiness_polished_source_has_no_more_than_seven_rows() -> None:
    paths = _toy_paths()
    _write_required_inputs(paths)
    table = ABMV4PolishedPlotBuilder(paths).build_scenario_readiness_checklist_polished_source()
    assert table.height <= 7


def test_q_energy_web_version_has_no_more_than_six_rows() -> None:
    paths = _toy_paths()
    _write_required_inputs(paths)
    builder = ABMV4PolishedPlotBuilder(paths)
    web = builder.build_q_energy_mix_quality_boundary_web_source(builder.build_q_energy_mix_quality_boundary_polished_source())
    assert web.height <= 6


def test_manifest_recommends_core_plots_for_report_and_webpage() -> None:
    manifest = ABMV4PolishedPlotBuilder(_toy_paths()).build_final_visual_selection_manifest()
    required = {
        "abm_v4_architecture_layers_polished",
        "abm_v4_emissions_decomposition_logic_polished",
        "abm_v4_two_rule_scorecard",
        "abm_v4_to_v5_roadmap_polished",
    }
    selected = manifest.filter(pl.col("use_in_latex_report") & pl.col("use_in_portfolio_webpage"))
    assert required.issubset(set(selected["plot_file"]))


def test_hypothesis_status_is_written_as_table_even_without_plot() -> None:
    paths = _toy_paths()
    _write_required_inputs(paths)
    result = ABMV4PolishedPlotBuilder(paths).run(write_outputs=True)
    assert paths.final_tables_polished.joinpath("hypothesis_status_report_table.csv") in result.source_table_paths
    assert not paths.final_plots_polished.joinpath("abm_v4_hypothesis_status_compact.png").exists()


def test_all_required_polished_plots_write_png_and_svg() -> None:
    paths = _toy_paths()
    _write_required_inputs(paths)
    result = ABMV4PolishedPlotBuilder(paths).run(write_outputs=True)
    assert len([path for path in result.plot_paths if path.suffix == ".png"]) == len(POLISHED_PLOT_NAMES)
    assert len([path for path in result.plot_paths if path.suffix == ".svg"]) == len(POLISHED_PLOT_NAMES)


def test_polished_dry_run_creates_no_outputs() -> None:
    paths = _toy_paths()
    _write_required_inputs(paths)
    ABMV4PolishedPlotBuilder(paths).run(write_outputs=False)
    assert not paths.final.exists()
    assert not paths.outputs_plots_abm_v4_final_polished.exists()


def test_no_latex_webpage_scenario_abm_v5_or_transition_rule_outputs() -> None:
    paths = _toy_paths()
    _write_required_inputs(paths)
    ABMV4PolishedPlotBuilder(paths).run(write_outputs=True)
    assert not list(paths.project_root.rglob("*.tex"))
    assert not list(paths.project_root.rglob("*.html"))
    assert not paths.data_abm_v4.joinpath("scenarios").exists()
    generated_abm_v5_paths = [
        path
        for path in paths.final.rglob("*abm_v5*")
        if path.name != "abm_v4_to_v5_roadmap_polished_source.csv"
    ]
    assert not generated_abm_v5_paths
    assert not list(paths.project_root.rglob("*transition_rule*.py"))
