from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import polars as pl
import pytest

from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.validation import ABMV4FinalConsolidator


def _toy_paths() -> ABMV4Paths:
    return ABMV4Paths(project_root=Path("tmp") / "abm_v4_final_tests" / uuid4().hex)


def _write_required_inputs(paths: ABMV4Paths) -> None:
    paths.simulations.mkdir(parents=True, exist_ok=True)
    paths.validation.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "country_sector": ["CHN | Electricity", "USA | Manufacturing"],
            "year": [1995, 1995],
            "simulated_emissions": [10.0, 5.0],
        }
    ).write_parquet(paths.base_multiyear_state_panel_path)
    pl.DataFrame(
        {
            "country_sector": ["CHN | Electricity", "USA | Manufacturing"],
            "year": [1995, 1995],
            "simulated_emissions": [11.0, 4.0],
        }
    ).write_parquet(paths.base_multiyear_state_panel_historical_frontier_gap_path)
    pl.DataFrame({"year": [1995], "aggregate_emissions": [15.0]}).write_csv(
        paths.base_multiyear_summary_panel_path
    )
    pl.DataFrame({"year": [1995], "aggregate_emissions": [15.0]}).write_csv(
        paths.base_multiyear_summary_panel_historical_frontier_gap_path
    )
    pl.DataFrame({"metric": ["mae"], "value": [0.1]}).write_csv(paths.multiyear_error_summary_path)


def _write_optional_inputs(paths: ABMV4Paths) -> None:
    pl.DataFrame({"recommendation": ["reject_EID_for_v4"]}).write_csv(
        paths.essential_input_dampener_recommendation_path
    )
    pl.DataFrame({"recommendation": ["reject_EID_for_v4_confirmed"]}).write_csv(
        paths.adaptive_EID_recommendation_path
    )
    pl.DataFrame({"recommendation": ["aggregate_only_energy_mix_usable"]}).write_csv(
        paths.q_energy_mix_recommendation_path
    )


def test_input_availability_report_marks_required_and_optional_files() -> None:
    paths = _toy_paths()
    _write_required_inputs(paths)
    _write_optional_inputs(paths)

    report = ABMV4FinalConsolidator(paths).build_input_availability_report()

    required = report.filter(pl.col("required_or_optional") == "required")
    optional = report.filter(pl.col("required_or_optional") == "optional")
    assert required.filter(~pl.col("exists")).is_empty()
    assert optional.height > 0
    assert "available" in set(report["status"])
    assert "missing_optional" in set(report["status"])


def test_missing_required_surviving_rule_files_fail_clearly() -> None:
    paths = _toy_paths()
    paths.validation.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="missing required surviving-rule inputs"):
        ABMV4FinalConsolidator(paths).run()


def test_missing_optional_branch_files_do_not_fail_but_are_reported() -> None:
    paths = _toy_paths()
    _write_required_inputs(paths)

    result = ABMV4FinalConsolidator(paths).run()

    assert "missing_optional" in set(result.input_availability["status"])
    assert result.surviving_rule_comparison.height == 2


def test_surviving_rule_comparison_retains_two_final_rules() -> None:
    paths = _toy_paths()
    _write_required_inputs(paths)

    comparison = ABMV4FinalConsolidator(paths).build_surviving_rule_comparison()

    retained = dict(zip(comparison["rule_name"], comparison["retained_as"]))
    assert retained["frontier_gap_readiness"] == "aggregate_safe_baseline"
    assert retained["historical_frontier_gap_only"] == "transition_mechanism_benchmark"


def test_validation_objective_matrix_records_scenario_readiness_as_premature() -> None:
    matrix = ABMV4FinalConsolidator(_toy_paths()).build_validation_objective_matrix()
    row = matrix.filter(pl.col("objective") == "scenario_readiness").row(0, named=True)

    assert "premature" in row["conclusion"]
    assert row["frontier_gap_readiness_assessment"] == "not scenario-ready"


def test_rejected_register_retains_eid_and_q_energy_as_evidence() -> None:
    register = ABMV4FinalConsolidator(_toy_paths()).build_rejected_mechanism_register()
    eid = register.filter(pl.col("mechanism") == "fixed EID dampener").row(0, named=True)
    q_energy = register.filter(pl.col("mechanism") == "Q energy mix country-sector transition rule").row(0, named=True)

    assert eid["test_result"] == "rejected_for_abm_v4_rule"
    assert "ontology evidence" in eid["retained_value"]
    assert q_energy["test_result"] == "rejected_for_abm_v4_node_level_rule"
    assert "aggregate diagnostics" in q_energy["retained_value"]
    assert "ABM v5" in q_energy["retained_value"]


def test_model_boundary_statement_contains_core_claims() -> None:
    statement = ABMV4FinalConsolidator(_toy_paths()).build_model_boundary_statement()

    assert "historical diagnostic framework" in statement
    assert "not scenario-ready" in statement


def test_scenario_readiness_assessment_returns_overall_not_ready() -> None:
    readiness = ABMV4FinalConsolidator(_toy_paths()).build_scenario_readiness_assessment()
    row = readiness.filter(pl.col("readiness_dimension") == "overall_scenario_readiness").row(0, named=True)

    assert row["status"] == "not_scenario_ready"


def test_abm_v5_research_agenda_includes_expected_priorities() -> None:
    agenda = ABMV4FinalConsolidator(_toy_paths()).build_abm_v5_research_agenda()

    assert {
        "energy/fuel structure",
        "policy/institutional regime",
        "capital-stock inertia",
        "explicit agent ontology",
        "endogenous production dynamics",
    }.issubset(set(agenda["research_priority"]))


def test_final_hypothesis_table_marks_scenario_ready_as_not_supported() -> None:
    hypotheses = ABMV4FinalConsolidator(_toy_paths()).build_hypothesis_status_table()
    row = hypotheses.filter(pl.col("hypothesis") == "ABM_v4_scenario_ready").row(0, named=True)

    assert row["status"] == "not_supported"


def test_final_report_is_written_only_with_explicit_write_call() -> None:
    paths = _toy_paths()
    _write_required_inputs(paths)

    result = ABMV4FinalConsolidator(paths).run()

    assert not paths.final_abm_v4_consolidation_report_path.exists()
    ABMV4FinalConsolidator(paths).write_outputs(result)
    assert paths.final_abm_v4_consolidation_report_path.exists()


def test_dry_run_style_in_memory_finalization_creates_no_outputs() -> None:
    paths = _toy_paths()
    _write_required_inputs(paths)

    ABMV4FinalConsolidator(paths).run()

    assert not paths.final_abm_v4_input_availability_path.exists()
    assert not paths.final_surviving_rule_comparison_path.exists()
    assert not paths.final_abm_v4_portfolio_summary_path.exists()
