from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import polars as pl
import pytest

from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.reporting import ABMV4FinalArtifactBuilder


def _toy_paths() -> ABMV4Paths:
    return ABMV4Paths(project_root=Path("tmp") / "abm_v4_final_artifact_tests" / uuid4().hex)


def _write_phase28_outputs(paths: ABMV4Paths) -> None:
    paths.validation.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "rule_name": ["frontier_gap_readiness", "historical_frontier_gap_only"],
            "theoretical_role": ["aggregate-safe baseline", "transition benchmark"],
            "strengths": ["aggregate emissions plausibility", "clean frontier-gap interpretation"],
            "weaknesses": ["less pure mechanism", "weaker aggregate safety"],
            "retained_as": ["aggregate_safe_baseline", "transition_mechanism_benchmark"],
            "scenario_use_status": ["not_scenario_ready", "not_scenario_ready"],
        }
    ).write_csv(paths.final_surviving_rule_comparison_path)
    pl.DataFrame(
        {
            "mechanism": [
                "legacy_raw_log emissions rule",
                "fixed EID dampener",
                "adaptive EID dampener",
                "EID diagnostic multi-year mode",
                "Q energy mix country-sector transition rule",
                "historical residual as scenario-facing rule",
                "electricity-specific transition patch",
            ],
            "test_result": [
                "rejected",
                "rejected_for_abm_v4_rule",
                "rejected_for_abm_v4_rule",
                "diagnostic_only",
                "rejected_for_abm_v4_node_level_rule",
                "rejected_for_scenarios",
                "diagnostic_only",
            ],
            "reason_rejected_or_limited": [
                "implicit rule",
                "failed validation as transition rule",
                "adaptive calibration did not rescue EID",
                "multi-year branch worsened aggregate fit",
                "node-level Q quality limits",
                "historical leakage",
                "not validated as general rule",
            ],
            "retained_value": [
                "baseline foil",
                "ontology evidence",
                "ontology evidence",
                "failure-mode evidence",
                "aggregate diagnostics and ABM v5 evidence",
                "diagnostic benchmark",
                "electricity mechanism evidence",
            ],
            "future_use": [
                "historical comparison",
                "ABM v5 agent typing",
                "ABM v5 design input",
                "ontology diagnostics",
                "cleaner energy-system data",
                "feature discovery",
                "ABM v5 energy module",
            ],
            "scenario_status": [
                "rejected_for_scenarios",
                "not_scenario_ready",
                "not_scenario_ready",
                "not_scenario_ready",
                "not_scenario_ready",
                "rejected_for_scenarios",
                "not_scenario_ready",
            ],
        }
    ).write_csv(paths.final_rejected_mechanism_register_path)
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
            "status": [
                "supported",
                "supported",
                "not_supported",
                "supported",
                "not_supported",
                "supported",
                "not_supported",
            ],
            "evidence": [
                "aggregate-safe",
                "benchmark",
                "EID failed as dampener",
                "EID classifies node roles",
                "Q node-level rule failed",
                "Q aggregate diagnostic supported",
                "historical forcing blocks scenarios",
            ],
            "interpretation": [""] * 7,
            "implication": [""] * 7,
        }
    ).write_csv(paths.final_abm_v4_hypothesis_status_path)
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
            "evidence": [
                "two rules survive",
                "historical production forcing remains central",
                "missing fuel mechanism",
                "policy variables absent",
                "invalid Q shares",
                "multiple blockers",
            ],
            "blocking_issue": [
                "no single scenario-facing rule",
                "no endogenous output dynamics",
                "node-level energy data quality insufficient",
                "policy variables are absent",
                "country-sector Q quality blocks rules",
                "historical diagnostic only",
            ],
            "required_future_work": ["future work"] * 6,
        }
    ).write_csv(paths.final_scenario_readiness_assessment_path)
    pl.DataFrame(
        {
            "research_priority": [
                "energy/fuel structure",
                "policy/institutional regime",
                "capital-stock inertia",
                "explicit agent ontology",
                "endogenous production dynamics",
            ],
            "motivation_from_abm_v4": ["motivation"] * 5,
            "required_data": ["data"] * 5,
            "candidate_mechanism": ["mechanism"] * 5,
            "candidate_agent_type": ["agent"] * 5,
            "expected_validation_test": ["test"] * 5,
            "priority_level": ["high"] * 5,
        }
    ).write_csv(paths.final_abm_v5_research_agenda_path)
    pl.DataFrame(
        {
            "objective": [
                "transition_mechanism_validity",
                "aggregate_emissions_validity",
                "electricity_high_emissions_validity",
                "capability_source_validity",
                "production_feasibility_validity",
                "data_quality_validity",
                "scenario_readiness",
            ],
            "frontier_gap_readiness_assessment": ["supported"] * 7,
            "historical_frontier_gap_only_assessment": ["supported"] * 7,
            "EID_assessment": ["diagnostic only"] * 7,
            "Q_energy_mix_assessment": ["aggregate-only evidence"] * 7,
            "evidence": ["evidence"] * 7,
            "conclusion": ["conclusion"] * 7,
        }
    ).write_csv(paths.final_validation_objective_matrix_path)
    paths.final_model_boundary_statement_path.write_text(
        "ABM v4 is a historical diagnostic framework. Historical production forcing remains central.",
        encoding="utf-8",
    )
    paths.final_abm_v4_consolidation_report_path.write_text("Final report source.", encoding="utf-8")
    paths.final_abm_v4_portfolio_summary_path.write_text("Portfolio source.", encoding="utf-8")


def test_final_table_builder_creates_two_rule_summary() -> None:
    paths = _toy_paths()
    _write_phase28_outputs(paths)

    table = ABMV4FinalArtifactBuilder(paths).build_two_rule_summary()

    assert table.height == 2
    assert "frontier_gap_readiness" in set(table["rule_name"])


def test_final_mechanism_table_marks_eid_as_ontology_evidence_not_active_rule() -> None:
    paths = _toy_paths()
    _write_phase28_outputs(paths)

    table = ABMV4FinalArtifactBuilder(paths).build_mechanism_status()
    row = table.filter(pl.col("mechanism") == "fixed_EID_dampener").row(0, named=True)

    assert "ontology evidence" in row["retained_value"]
    assert row["status"] == "rejected_for_abm_v4_rule"


def test_final_mechanism_table_marks_q_energy_as_aggregate_diagnostic_not_node_rule() -> None:
    paths = _toy_paths()
    _write_phase28_outputs(paths)

    table = ABMV4FinalArtifactBuilder(paths).build_mechanism_status()
    country_sector = table.filter(pl.col("mechanism") == "Q_energy_mix_country_sector_rule").row(0, named=True)
    aggregate = table.filter(pl.col("mechanism") == "Q_energy_mix_aggregate_diagnostic").row(0, named=True)

    assert "rejected as ABM v4 node-level rule" in country_sector["retained_value"]
    assert "aggregate diagnostic" in aggregate["retained_value"]


def test_final_scenario_blockers_include_historical_production_forcing() -> None:
    paths = _toy_paths()
    _write_phase28_outputs(paths)

    blockers = ABMV4FinalArtifactBuilder(paths).build_scenario_blockers()

    assert "historical production forcing" in set(blockers["blocker"])


def test_final_abm_v5_priorities_include_expected_priorities() -> None:
    paths = _toy_paths()
    _write_phase28_outputs(paths)

    priorities = ABMV4FinalArtifactBuilder(paths).build_abm_v5_priorities()

    assert {
        "energy/fuel structure",
        "policy/institutional regime",
        "capital-stock inertia",
        "explicit agent ontology",
        "endogenous production dynamics",
    }.issubset(set(priorities["priority"]))


def test_final_portfolio_metrics_include_fixed_final_values() -> None:
    metrics = ABMV4FinalArtifactBuilder(_toy_paths()).build_portfolio_metrics()
    values = set(metrics["value"])

    assert {"4,915", "1995-2016", "108,130", "355 passed", "2", "not scenario-ready"}.issubset(values)


def test_plot_builder_writes_png_and_svg() -> None:
    paths = _toy_paths()
    _write_phase28_outputs(paths)

    result = ABMV4FinalArtifactBuilder(paths).run(write_outputs=True)

    assert any(path.suffix == ".png" for path in result.plot_paths)
    assert any(path.suffix == ".svg" for path in result.plot_paths)


def test_validation_objective_plot_can_be_built_from_toy_matrix() -> None:
    paths = _toy_paths()
    _write_phase28_outputs(paths)
    matrix = pl.read_csv(paths.final_validation_objective_matrix_path)

    figure = ABMV4FinalArtifactBuilder(paths).plot_validation_objective_matrix(matrix)

    assert figure.axes


def test_mechanism_funnel_plot_can_be_built_from_toy_mechanism_table() -> None:
    paths = _toy_paths()
    _write_phase28_outputs(paths)
    builder = ABMV4FinalArtifactBuilder(paths)

    figure = builder.plot_mechanism_funnel(builder.build_mechanism_status())

    assert figure.axes


def test_scenario_readiness_plot_can_be_built_from_toy_blocker_table() -> None:
    paths = _toy_paths()
    _write_phase28_outputs(paths)
    builder = ABMV4FinalArtifactBuilder(paths)

    figure = builder.plot_scenario_readiness_blockers(builder.build_scenario_blockers())

    assert figure.axes


def test_artifact_index_includes_tables_and_plots() -> None:
    paths = _toy_paths()
    _write_phase28_outputs(paths)

    result = ABMV4FinalArtifactBuilder(paths).run(write_outputs=True)
    index = pl.read_csv(result.artifact_index_path)

    assert "final_table" in set(index["artifact_type"])
    assert "final_plot" in set(index["artifact_type"])


def test_dry_run_creates_no_outputs() -> None:
    paths = _toy_paths()
    _write_phase28_outputs(paths)

    ABMV4FinalArtifactBuilder(paths).run(write_outputs=False)

    assert not paths.final.exists()
    assert not paths.outputs_plots_abm_v4_final.exists()


def test_missing_required_phase28_outputs_fail_with_actionable_message() -> None:
    paths = _toy_paths()
    paths.validation.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="Run: python scripts/run_abm_v4_base.py --finalize-abm-v4"):
        ABMV4FinalArtifactBuilder(paths).validate_required_phase28_outputs()


def test_no_latex_report_is_created() -> None:
    paths = _toy_paths()
    _write_phase28_outputs(paths)

    ABMV4FinalArtifactBuilder(paths).run(write_outputs=True)

    assert not list(paths.project_root.rglob("*.tex"))


def test_no_webpage_file_is_created() -> None:
    paths = _toy_paths()
    _write_phase28_outputs(paths)

    ABMV4FinalArtifactBuilder(paths).run(write_outputs=True)

    assert not list(paths.project_root.rglob("*.html"))
