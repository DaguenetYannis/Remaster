from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import src.abm_v5 as abm_v5
from src.abm_v5.phase_space import (
    CAPABILITY_COLUMNS,
    NETWORK_COLUMNS,
    PhaseSpaceBuildResult,
    add_regime_placeholders,
    build_historical_phase_space_panel,
    build_phase_space_position,
    compute_emissions_intensity_gap,
    compute_phase_space_completeness,
    load_historical_phase_space_panel,
    summarize_phase_space_missingness,
    summarize_phase_space_variable_coverage,
    validate_phase_space_panel,
)


LABELS = (
    "USA | United States | Industry | Manufacturing",
    "CHN | China | Industry | Manufacturing",
)


def _base_panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "country_sector": [LABELS[0], LABELS[1], "USA | United States | Industry | Services"],
            "country": ["USA", "CHN", "USA"],
            "country_detail": ["United States", "China", "United States"],
            "category": ["Industry", "Industry", "Industry"],
            "sector": ["Manufacturing", "Manufacturing", "Services"],
            "year": [1995, 1995, 1995],
            "output": [100.0, 200.0, 50.0],
            "final_demand": [10.0, 20.0, 5.0],
            "emissions": [10.0, 30.0, 5.0],
            "emissions_intensity": [0.1, 0.3, None],
            "local_greenness": [0.8, 0.2, None],
            "accounting_output_positive_flag": [True, True, True],
            "accounting_emissions_nonnegative_flag": [True, True, True],
            "accounting_ei_valid_flag": [True, True, False],
            "general_capability": [1.0, None, 0.2],
            "green_capability": [0.4, 0.1, None],
            "capability_density": [None, None, None],
            "green_capability_density": [None, None, None],
            "ecosystem_proximity": [None, None, None],
            "directed_green_precedence": [None, None, None],
            "reachable_green_complexity": [None, None, None],
            "transition_sector_score": [None, None, None],
            "general_capability_available_flag": [True, False, True],
            "green_capability_available_flag": [True, True, False],
            "capability_density_available_flag": [False, False, False],
            "green_capability_density_available_flag": [False, False, False],
            "ecosystem_proximity_available_flag": [False, False, False],
            "directed_green_precedence_available_flag": [False, False, False],
            "reachable_green_complexity_available_flag": [False, False, False],
            "transition_sector_score_available_flag": [False, False, False],
            "supplier_count": [2, 1, 0],
            "buyer_count": [1, 2, 0],
            "total_inputs_from_suppliers": [100.0, 50.0, 0.0],
            "total_outputs_to_buyers": [80.0, 70.0, 0.0],
            "supplier_concentration_hhi": [0.5, 1.0, None],
            "buyer_concentration_hhi": [1.0, 0.5, None],
            "import_dependence_proxy": [0.2, 0.3, None],
            "export_dependence_proxy": [0.1, 0.4, None],
            "network_green_exposure": [0.6, 0.7, None],
            "incoming_network_green_exposure": [0.5, 0.8, None],
            "outgoing_network_green_exposure": [0.7, 0.6, None],
            "brown_centrality": [0.2, 0.8, None],
            "supplier_lock_in": [0.5, 1.0, None],
        }
    )


def _complete_phase_panel() -> pl.DataFrame:
    panel = compute_emissions_intensity_gap(_base_panel())
    panel = build_phase_space_position(panel)
    panel = compute_phase_space_completeness(panel)
    return add_regime_placeholders(panel)


def _write_pyproject(project_root: Path) -> None:
    project_root.joinpath("pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")


def _write_fake_phase_inputs(project_root: Path) -> None:
    _write_pyproject(project_root)
    accounting_dir = project_root / "data" / "abm_v5" / "accounting"
    capability_dir = project_root / "data" / "abm_v5" / "capabilities"
    network_dir = project_root / "data" / "abm_v5" / "supplier_network"
    accounting_dir.mkdir(parents=True, exist_ok=True)
    capability_dir.mkdir(parents=True, exist_ok=True)
    network_dir.mkdir(parents=True, exist_ok=True)
    panel = _base_panel()
    accounting_columns = [
        "country_sector",
        "country",
        "country_detail",
        "category",
        "sector",
        "year",
        "output",
        "final_demand",
        "emissions",
        "emissions_intensity",
        "local_greenness",
        "accounting_output_positive_flag",
        "accounting_emissions_nonnegative_flag",
        "accounting_ei_valid_flag",
    ]
    capability_columns = [
        "country_sector",
        "year",
        *CAPABILITY_COLUMNS,
        *(f"{column}_available_flag" for column in CAPABILITY_COLUMNS),
    ]
    network_columns = ["country_sector", "year", *NETWORK_COLUMNS]
    panel.select(accounting_columns).write_parquet(
        accounting_dir / "accounting_state_panel_1995_2016.parquet"
    )
    panel.select(capability_columns).write_parquet(
        capability_dir / "capability_state_panel_1995_2016.parquet"
    )
    panel.select(network_columns).write_parquet(network_dir / "network_state_panel_1995_2016.parquet")


def test_compute_emissions_intensity_gap_uses_sector_year_median() -> None:
    result = compute_emissions_intensity_gap(_base_panel())
    values = result.filter(pl.col("sector") == "Manufacturing").sort("emissions_intensity")[
        "emissions_intensity_gap"
    ].to_list()
    assert values == pytest.approx([-0.1, 0.1])


def test_compute_emissions_intensity_gap_preserves_invalid_ei_as_null() -> None:
    result = compute_emissions_intensity_gap(_base_panel())
    assert result.filter(pl.col("sector") == "Services")["emissions_intensity_gap"].item() is None


def test_build_phase_space_position_uses_available_empirical_variables() -> None:
    result = build_phase_space_position(compute_emissions_intensity_gap(_base_panel()))
    position = result["phase_space_position"].to_list()[0]
    assert "network_green_exposure" in position
    assert "local_greenness" not in position


def test_build_phase_space_position_marks_unavailable_design_targets() -> None:
    result = build_phase_space_position(compute_emissions_intensity_gap(_base_panel()))
    position = result["phase_space_position"].to_list()[0]
    assert "capability_density" in position["unavailable_design_targets"]


def test_compute_phase_space_completeness_does_not_require_design_targets() -> None:
    panel = compute_phase_space_completeness(build_phase_space_position(compute_emissions_intensity_gap(_base_panel())))
    ready_row = panel.row(0, named=True)
    assert ready_row["phase_space_design_target_completeness"] == 0.0
    assert ready_row["phase_space_ready_for_regime_discovery_flag"] is True


def test_compute_phase_space_completeness_ready_flag_requires_core_empirical_variables() -> None:
    panel = compute_phase_space_completeness(build_phase_space_position(compute_emissions_intensity_gap(_base_panel())))
    assert panel.row(2, named=True)["phase_space_ready_for_regime_discovery_flag"] is False


def test_add_regime_placeholders_creates_null_columns() -> None:
    panel = add_regime_placeholders(_base_panel())
    for column in (
        "regime_membership",
        "regime_probability",
        "regime_confidence",
        "previous_regime_membership",
        "regime_switch_flag",
        "threshold_rule_id",
    ):
        assert panel[column].null_count() == panel.height


def test_validate_phase_space_panel_requires_columns() -> None:
    results = validate_phase_space_panel(pl.DataFrame({"country_sector": [LABELS[0]]}))
    assert any(result.status.value == "failed" for result in results)


def test_validate_phase_space_panel_detects_duplicate_keys() -> None:
    panel = _complete_phase_panel()
    results = validate_phase_space_panel(pl.concat([panel.head(1), panel.head(1)], how="vertical"))
    assert any(result.check_name == "phase_space_unique_country_sector_year" and result.n_failed > 0 for result in results)


def test_validate_phase_space_panel_checks_bounded_columns() -> None:
    panel = _complete_phase_panel().with_columns(pl.lit(2.0).alias("network_green_exposure"))
    results = validate_phase_space_panel(panel)
    assert any(result.check_name == "phase_space_network_green_exposure_bounds" and result.n_failed > 0 for result in results)


def test_validate_phase_space_panel_requires_regime_placeholders_null() -> None:
    panel = _complete_phase_panel().with_columns(pl.lit("not_allowed").alias("regime_membership"))
    results = validate_phase_space_panel(panel)
    assert any(result.check_name == "phase_space_regime_placeholders_null" and result.n_failed > 0 for result in results)


def test_summarize_phase_space_missingness_returns_groups() -> None:
    summary = summarize_phase_space_missingness(_complete_phase_panel())
    assert {"accounting", "capability", "network", "phase_space", "regime_placeholder"}.issubset(
        set(summary["variable_group"].to_list())
    )


def test_summarize_phase_space_variable_coverage_returns_expected_keys() -> None:
    summary = summarize_phase_space_variable_coverage(_complete_phase_panel())
    assert set(summary) == {
        "n_rows",
        "n_agents",
        "start_year",
        "end_year",
        "mean_empirical_completeness",
        "mean_design_target_completeness",
        "share_ready_for_regime_discovery",
        "share_valid_emissions_intensity",
        "share_green_capability_available",
        "share_general_capability_available",
        "share_network_green_exposure_available",
        "share_brown_centrality_available",
        "share_supplier_lock_in_available",
        "regime_placeholders_all_null",
    }


def test_build_historical_phase_space_panel_requires_accounting_panel(tmp_path: Path) -> None:
    _write_pyproject(tmp_path)
    with pytest.raises(FileNotFoundError, match="Phase 2.3"):
        build_historical_phase_space_panel(tmp_path)


def test_build_historical_phase_space_panel_requires_capability_panel(tmp_path: Path) -> None:
    _write_pyproject(tmp_path)
    accounting_dir = tmp_path / "data" / "abm_v5" / "accounting"
    accounting_dir.mkdir(parents=True)
    _base_panel().write_parquet(accounting_dir / "accounting_state_panel_1995_2016.parquet")
    with pytest.raises(FileNotFoundError, match="Phase 2.4"):
        build_historical_phase_space_panel(tmp_path)


def test_build_historical_phase_space_panel_requires_network_panel(tmp_path: Path) -> None:
    _write_fake_phase_inputs(tmp_path)
    (tmp_path / "data" / "abm_v5" / "supplier_network" / "network_state_panel_1995_2016.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="Phase 2.5"):
        build_historical_phase_space_panel(tmp_path)


def test_build_historical_phase_space_panel_writes_outputs_with_fake_data(tmp_path: Path) -> None:
    _write_fake_phase_inputs(tmp_path)
    result = build_historical_phase_space_panel(tmp_path)
    assert isinstance(result, PhaseSpaceBuildResult)
    assert result.output_path.exists()
    assert result.validation_path.exists()
    assert result.missingness_summary_path.exists()
    assert result.coverage_summary_path.exists()
    loaded = load_historical_phase_space_panel(result.output_path)
    assert loaded.height == 3


def test_build_historical_phase_space_panel_preserves_accounting_backbone(tmp_path: Path) -> None:
    _write_fake_phase_inputs(tmp_path)
    result = build_historical_phase_space_panel(tmp_path)
    assert load_historical_phase_space_panel(result.output_path).height == _base_panel().height


def test_init_exports_phase_space_objects() -> None:
    assert abm_v5.PhaseSpaceBuildResult is PhaseSpaceBuildResult
    assert abm_v5.compute_emissions_intensity_gap is compute_emissions_intensity_gap
    assert abm_v5.build_phase_space_position is build_phase_space_position
    assert abm_v5.compute_phase_space_completeness is compute_phase_space_completeness
    assert abm_v5.add_regime_placeholders is add_regime_placeholders
    assert abm_v5.validate_phase_space_panel is validate_phase_space_panel
    assert abm_v5.summarize_phase_space_missingness is summarize_phase_space_missingness
    assert abm_v5.summarize_phase_space_variable_coverage is summarize_phase_space_variable_coverage
    assert abm_v5.build_historical_phase_space_panel is build_historical_phase_space_panel
    assert abm_v5.load_historical_phase_space_panel is load_historical_phase_space_panel
