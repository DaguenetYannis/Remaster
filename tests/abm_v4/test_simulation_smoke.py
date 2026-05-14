from pathlib import Path
from uuid import uuid4

import polars as pl
import pytest

from src.abm_v4.config import ABMV4Config
from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.simulation import inspect_base_model_readiness, run_one_step_base_orchestration


def test_simulation_readiness_fails_clearly_without_sources() -> None:
    root = Path("tmp") / "abm_v4_tests" / uuid4().hex
    report = inspect_base_model_readiness(
        paths=ABMV4Paths(project_root=root),
        config=ABMV4Config(),
    )

    assert not report.can_run_base_model
    assert not report.state_source.has_source


def _toy_paths() -> ABMV4Paths:
    return ABMV4Paths(project_root=Path("tmp") / "abm_v4_one_step_tests" / uuid4().hex)


def _write_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([row]).write_csv(path)


def _write_required_one_step_outputs(paths: ABMV4Paths, config: ABMV4Config) -> None:
    state_path = paths.state_panel_path(config.start_year, config.end_year)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "country_sector": ["A", "A", "B", "B"],
            "Year": [2015, 2016, 2015, 2016],
            "Sector": ["S1", "S1", "S2", "S2"],
            "X_observed": [1.0, 1.1, 2.0, 2.1],
            "EI": [0.5, 0.4, 0.3, 0.25],
        }
    ).write_parquet(state_path)
    paths.raw_t_supplier_edges_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"year": [2016], "transaction_value": [1.0]}).write_parquet(
        paths.raw_t_supplier_edges_path
    )
    _write_csv(
        paths.diagnostics / "state_source_report.csv",
        {
            "selected_source": "toy",
            "row_count": 4,
            "year_min": 2015,
            "year_max": 2016,
            "country_sector_count": 2,
        },
    )
    _write_csv(
        paths.ecosystem_assignment_report_path,
        {
            "mapped_nodes": 2,
            "unmapped_nodes": 0,
            "number_of_ecosystems": 1,
            "ecosystem_source_counts": "toy=2",
        },
    )
    _write_csv(
        paths.raw_t_supplier_edge_report_path,
        {
            "selected_source": "raw_eora_T",
            "row_count": 1,
            "unique_supplier_buyer_pairs": 1,
        },
    )
    _write_csv(
        paths.supplier_candidate_base_report_path,
        {
            "historical_candidate_rows": 2,
            "same_sector_candidate_rows": 2,
            "ecosystem_candidate_rows": 2,
        },
    )
    _write_csv(
        paths.supplier_opportunity_set_report_path,
        {
            "opportunity_rows": 4,
            "median_candidates_per_buyer": 2,
        },
    )
    _write_csv(
        paths.supplier_rewiring_report_path,
        {
            "number_of_buyers": 2,
            "rewired_buyer_share": 0.0,
            "fallback_stress_buyers": 2,
            "max_initial_weight_sum_error": 0.0,
            "max_updated_weight_sum_error": 0.0,
        },
    )
    _write_csv(
        paths.capability_update_report_path,
        {
            "year": 2016,
            "mean_cap": 0.6,
            "mean_gcap": 0.2,
            "mean_exposure_cap": 0.5,
            "mean_exposure_gcap": 0.3,
            "mean_delta_cap": 0.01,
            "mean_delta_gcap": 0.02,
            "share_general_capability_filled": 0.3,
            "share_green_capability_filled": 0.1,
            "cap_clipped_count": 0,
            "gcap_clipped_count": 0,
        },
    )
    _write_csv(
        paths.production_feasibility_report_path,
        {
            "year": 2016,
            "aggregate_feasibility_ratio": 0.99,
            "mean_input_feasibility": 0.98,
            "share_nodes_with_input_feasibility_below_1": 0.9,
            "p95_supplier_pressure_max": 0.4,
            "share_nodes_with_supplier_pressure_above_1": 0.0,
        },
    )
    _write_csv(
        paths.emissions_update_report_path,
        {
            "year": 2016,
            "emissions_transition_mode": "frontier_gap_readiness",
            "node_count": 2,
            "valid_EI_nodes": 2,
            "invalid_EI_nodes": 0,
            "mean_rEI_used": 0.04,
            "median_rEI_used": 0.04,
            "aggregate_delta_emissions": -1.0,
            "decomposition_residual": 0.0,
            "bad_transition_flag": False,
        },
    )


def test_one_step_orchestrator_detects_existing_component_outputs() -> None:
    paths = _toy_paths()
    config = ABMV4Config(start_year=2015, end_year=2016)
    _write_required_one_step_outputs(paths, config)

    result = run_one_step_base_orchestration(paths, config, write_outputs=False)

    assert result.reused_existing_outputs
    assert result.raw_t_rebuild_skipped
    assert result.validation.report.height == 6


def test_one_step_orchestrator_refuses_missing_inputs_without_build_flags() -> None:
    paths = _toy_paths()
    config = ABMV4Config(start_year=2015, end_year=2016)

    with pytest.raises(FileNotFoundError, match="required component outputs are missing"):
        run_one_step_base_orchestration(paths, config, write_outputs=False)


def test_one_step_validation_aggregates_layers_and_pass_fail_rules() -> None:
    paths = _toy_paths()
    config = ABMV4Config(start_year=2015, end_year=2016)
    _write_required_one_step_outputs(paths, config)

    result = run_one_step_base_orchestration(paths, config, write_outputs=False)
    by_layer = {row["layer"]: row for row in result.validation.report.to_dicts()}

    assert by_layer["state"]["status"] == "pass"
    assert by_layer["ecosystem"]["status"] == "pass"
    assert by_layer["supplier"]["status"] == "warning"
    assert result.validation.status["overall_status"] == "warning"


def test_one_step_warnings_do_not_fail_run() -> None:
    paths = _toy_paths()
    config = ABMV4Config(start_year=2015, end_year=2016)
    _write_required_one_step_outputs(paths, config)

    result = run_one_step_base_orchestration(paths, config, write_outputs=False)

    assert result.validation.passed
    assert "supplier" in result.validation.status["warning_layers"]


def test_one_step_raw_t_rebuild_is_skipped_by_default() -> None:
    paths = _toy_paths()
    config = ABMV4Config(start_year=2015, end_year=2016)
    _write_required_one_step_outputs(paths, config)

    result = run_one_step_base_orchestration(paths, config, write_outputs=False)

    assert result.raw_t_rebuild_skipped


def test_one_step_dry_build_creates_no_validation_outputs() -> None:
    paths = _toy_paths()
    config = ABMV4Config(start_year=2015, end_year=2016)
    _write_required_one_step_outputs(paths, config)

    run_one_step_base_orchestration(paths, config, write_outputs=False)

    assert not paths.one_step_base_validation_report_csv_path.exists()
    assert not paths.one_step_base_validation_report_md_path.exists()


def test_one_step_markdown_report_written_only_when_enabled() -> None:
    paths = _toy_paths()
    config = ABMV4Config(start_year=2015, end_year=2016)
    _write_required_one_step_outputs(paths, config)

    run_one_step_base_orchestration(paths, config, write_outputs=True)

    assert paths.one_step_base_validation_report_csv_path.exists()
    assert paths.one_step_base_validation_report_md_path.exists()
    assert paths.one_step_base_status_json_path.exists()
