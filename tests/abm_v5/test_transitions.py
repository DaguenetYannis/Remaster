from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import src.abm_v5 as abm_v5
from src.abm_v5.regimes import RegimeLabel
from src.abm_v5.transitions import (
    TransitionEncodingBuildResult,
    TransitionState,
    add_forward_transition_columns,
    assign_transition_states,
    build_historical_transition_encoding,
    build_regime_transition_matrix,
    build_transition_state_by_sector,
    build_transition_state_by_year,
    build_transition_state_summary,
    compute_transition_delta_thresholds,
    load_historical_transition_panel,
    validate_transition_encoding_outputs,
)


def _regime_panel() -> pl.DataFrame:
    rows = []
    specs = [
        ("A", RegimeLabel.GREEN_CAPABLE_EMBEDDED.value, [0.0, 0.0], [0.4, 0.7], [0.5, 0.5], [0.4, 0.4]),
        ("B", RegimeLabel.INSUFFICIENT_DATA.value, [0.0, 0.1], [0.4, 0.5], [0.5, 0.5], [0.4, 0.4]),
        ("C", RegimeLabel.LOW_SIGNAL_PERIPHERAL.value, [0.0, 0.1], [0.4, 0.5], [0.5, 0.5], [0.4, 0.4]),
        ("D", RegimeLabel.MIXED_INTERMEDIATE.value, [0.0, 0.0], [0.4, 0.4], [0.2, 0.8], [0.4, 0.4]),
        ("E", RegimeLabel.MIXED_INTERMEDIATE.value, [0.0, 0.0], [0.4, 0.4], [0.5, 0.5], [0.8, 0.2]),
        ("F", RegimeLabel.MIXED_INTERMEDIATE.value, [0.4, -0.2], [0.4, 0.4], [0.5, 0.5], [0.4, 0.4]),
        ("G", RegimeLabel.MIXED_INTERMEDIATE.value, [0.0, 0.0], [0.4, 0.4], [0.5, 0.5], [0.4, 0.4]),
    ]
    for node, regime, ei_values, network_values, capability_values, lock_values in specs:
        for offset, year in enumerate([1995, 1996]):
            next_regime = RegimeLabel.GREEN_CAPABLE_CONSTRAINED.value if node == "C" and year == 1996 else regime
            rows.append(
                {
                    "country_sector": f"{node} | Country {node} | Industry | Sector",
                    "country": node,
                    "sector": "Sector",
                    "year": year,
                    "regime_membership": next_regime,
                    "regime_confidence": 1.0,
                    "emissions_intensity_gap": ei_values[offset],
                    "network_green_exposure": network_values[offset],
                    "green_capability": capability_values[offset],
                    "general_capability": capability_values[offset],
                    "supplier_lock_in": lock_values[offset],
                    "brown_centrality": 0.5,
                    "local_greenness": 0.5,
                    "output": 100.0,
                    "emissions": 10.0,
                }
            )
    return pl.DataFrame(rows)


def _thresholds() -> dict:
    return {
        "method": "global_robust_delta_quantiles",
        "variables": [],
        "thresholds": {
            "delta_emissions_intensity_gap": {"meaningful_change_threshold": 0.2},
            "delta_network_green_exposure": {"meaningful_change_threshold": 0.2},
            "delta_green_capability": {"meaningful_change_threshold": 0.2},
            "delta_general_capability": {"meaningful_change_threshold": 0.2},
            "delta_supplier_lock_in": {"meaningful_change_threshold": 0.2},
            "delta_brown_centrality": {"meaningful_change_threshold": 0.2},
            "delta_local_greenness": {"meaningful_change_threshold": 0.2},
        },
    }


def _transition_panel() -> pl.DataFrame:
    return assign_transition_states(add_forward_transition_columns(_regime_panel()), _thresholds())


def test_add_forward_transition_columns_adds_next_regime_and_deltas() -> None:
    panel = add_forward_transition_columns(_regime_panel())
    row = panel.filter(pl.col("country_sector").str.starts_with("A") & (pl.col("year") == 1995)).row(0, named=True)
    assert row["next_year"] == 1996
    assert row["next_regime_membership"] == RegimeLabel.GREEN_CAPABLE_EMBEDDED.value
    assert row["delta_network_green_exposure"] == pytest.approx(0.3)


def test_add_forward_transition_columns_marks_last_year_no_next() -> None:
    panel = add_forward_transition_columns(_regime_panel())
    row = panel.filter(pl.col("country_sector").str.starts_with("A") & (pl.col("year") == 1996)).row(0, named=True)
    assert row["next_year"] is None
    assert row["regime_switch_flag"] is None


def test_compute_transition_delta_thresholds_returns_method_and_thresholds() -> None:
    thresholds = compute_transition_delta_thresholds(add_forward_transition_columns(_regime_panel()))
    assert thresholds["method"] == "global_robust_delta_quantiles"
    assert "delta_network_green_exposure" in thresholds["thresholds"]


def test_assign_transition_states_assigns_no_next_year() -> None:
    row = _transition_panel().filter(pl.col("country_sector").str.starts_with("A") & (pl.col("year") == 1996)).row(0, named=True)
    assert row["transition_state"] == TransitionState.NO_NEXT_YEAR.value


def test_assign_transition_states_assigns_insufficient_data_transition() -> None:
    row = _transition_panel().filter(pl.col("country_sector").str.starts_with("B") & (pl.col("year") == 1995)).row(0, named=True)
    assert row["transition_state"] == TransitionState.INSUFFICIENT_DATA_TRANSITION.value


def test_assign_transition_states_prioritizes_regime_switch() -> None:
    row = _transition_panel().filter(pl.col("country_sector").str.starts_with("C") & (pl.col("year") == 1995)).row(0, named=True)
    assert row["transition_state"] == TransitionState.REGIME_SWITCH.value


def test_assign_transition_states_assigns_green_embedding_improvement() -> None:
    row = _transition_panel().filter(pl.col("country_sector").str.starts_with("A") & (pl.col("year") == 1995)).row(0, named=True)
    assert row["transition_state"] == TransitionState.GREEN_EMBEDDING_IMPROVEMENT.value


def test_assign_transition_states_assigns_capability_gain() -> None:
    row = _transition_panel().filter(pl.col("country_sector").str.starts_with("D") & (pl.col("year") == 1995)).row(0, named=True)
    assert row["transition_state"] == TransitionState.CAPABILITY_GAIN.value


def test_assign_transition_states_assigns_supplier_lock_in_relief() -> None:
    row = _transition_panel().filter(pl.col("country_sector").str.starts_with("E") & (pl.col("year") == 1995)).row(0, named=True)
    assert row["transition_state"] == TransitionState.SUPPLIER_LOCK_IN_RELIEF.value


def test_assign_transition_states_assigns_dirty_gap_improvement() -> None:
    row = _transition_panel().filter(pl.col("country_sector").str.starts_with("F") & (pl.col("year") == 1995)).row(0, named=True)
    assert row["transition_state"] == TransitionState.DIRTY_GAP_IMPROVEMENT.value


def test_assign_transition_states_assigns_stable_same_regime() -> None:
    row = _transition_panel().filter(pl.col("country_sector").str.starts_with("G") & (pl.col("year") == 1995)).row(0, named=True)
    assert row["transition_state"] == TransitionState.STABLE_SAME_REGIME.value


def test_build_regime_transition_matrix_counts_transitions() -> None:
    matrix = build_regime_transition_matrix(_transition_panel())
    assert "regime_membership" in matrix.columns


def test_build_transition_state_summary_returns_expected_columns() -> None:
    summary = build_transition_state_summary(_transition_panel())
    assert {"transition_state", "n_rows", "share_rows", "n_agents", "mean_transition_confidence"}.issubset(summary.columns)


def test_build_transition_state_by_year_returns_expected_columns() -> None:
    summary = build_transition_state_by_year(_transition_panel())
    assert {"year", "transition_state", "n_rows", "share_year_rows", "mean_transition_confidence"}.issubset(summary.columns)


def test_build_transition_state_by_sector_returns_expected_columns() -> None:
    summary = build_transition_state_by_sector(_transition_panel())
    assert {"sector", "transition_state", "n_rows", "share_sector_rows", "mean_transition_confidence", "mean_output", "mean_emissions"}.issubset(summary.columns)


def test_validate_transition_encoding_outputs_rejects_regime_probability() -> None:
    panel = _transition_panel().with_columns(pl.lit(0.5).alias("regime_probability"))
    results = validate_transition_encoding_outputs(panel, _thresholds())
    assert any(result.check_name == "transition_no_probability_simulation_or_scenario_columns" and result.n_failed > 0 for result in results)


def test_validate_transition_encoding_outputs_passes_valid_output() -> None:
    results = validate_transition_encoding_outputs(_transition_panel(), _thresholds())
    assert all(result.status.value == "passed" for result in results)


def _write_pyproject(root: Path) -> None:
    root.joinpath("pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")


def test_build_historical_transition_encoding_requires_regime_panel(tmp_path: Path) -> None:
    _write_pyproject(tmp_path)
    with pytest.raises(FileNotFoundError, match="Phase 3.1-3.3"):
        build_historical_transition_encoding(tmp_path)


def test_build_historical_transition_encoding_writes_outputs_with_fake_data(tmp_path: Path) -> None:
    _write_pyproject(tmp_path)
    regime_dir = tmp_path / "data" / "abm_v5" / "regimes"
    regime_dir.mkdir(parents=True)
    _regime_panel().write_parquet(regime_dir / "historical_regime_panel_1995_2016.parquet")
    result = build_historical_transition_encoding(tmp_path)
    assert isinstance(result, TransitionEncodingBuildResult)
    assert result.transition_panel_path.exists()
    assert result.transition_matrix_path.exists()
    assert result.validation_path.exists()


def test_build_historical_transition_encoding_preserves_backbone(tmp_path: Path) -> None:
    _write_pyproject(tmp_path)
    regime_dir = tmp_path / "data" / "abm_v5" / "regimes"
    regime_dir.mkdir(parents=True)
    _regime_panel().write_parquet(regime_dir / "historical_regime_panel_1995_2016.parquet")
    result = build_historical_transition_encoding(tmp_path)
    assert load_historical_transition_panel(result.transition_panel_path).height == _regime_panel().height


def test_init_exports_transition_objects() -> None:
    assert abm_v5.TransitionEncodingBuildResult is TransitionEncodingBuildResult
    assert abm_v5.TransitionState is TransitionState
    assert abm_v5.add_forward_transition_columns is add_forward_transition_columns
    assert abm_v5.compute_transition_delta_thresholds is compute_transition_delta_thresholds
    assert abm_v5.assign_transition_states is assign_transition_states
    assert abm_v5.build_historical_transition_encoding is build_historical_transition_encoding
