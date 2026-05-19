from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import src.abm_v5 as abm_v5
from src.abm_v5.regime_handoff import (
    MechanismTargetFamily,
    MechanismTargetStatus,
    RegimeHandoffBuildResult,
    build_mechanism_learning_sample_summary,
    build_mechanism_target_candidates,
    build_regime_handoff,
    build_regime_handoff_panel,
    build_regime_stability_summary,
    build_transition_target_summary,
    load_regime_handoff_panel,
    validate_regime_handoff_outputs,
)
from src.abm_v5.regimes import RegimeLabel
from src.abm_v5.transitions import TransitionState


def _regime_panel() -> pl.DataFrame:
    rows = []
    states = [
        TransitionState.NO_NEXT_YEAR.value,
        TransitionState.INSUFFICIENT_DATA_TRANSITION.value,
        TransitionState.GREEN_EMBEDDING_IMPROVEMENT.value,
        TransitionState.CAPABILITY_GAIN.value,
        TransitionState.REGIME_SWITCH.value,
        TransitionState.MIXED_MOVEMENT.value,
    ]
    for index, state in enumerate(states):
        rows.append(
            {
                "country_sector": f"C{index} | Country {index} | Industry | Sector",
                "country": f"C{index}",
                "sector": "Sector",
                "year": 1995,
                "regime_membership": RegimeLabel.GREEN_CAPABLE_EMBEDDED.value
                if index != 1
                else RegimeLabel.INSUFFICIENT_DATA.value,
                "regime_confidence": 1.0,
                "output": 100.0,
                "emissions": 10.0,
                "emissions_intensity_gap": 0.1,
                "network_green_exposure": 0.5,
                "green_capability": 0.5,
                "general_capability": 0.5,
                "supplier_lock_in": 0.5,
                "brown_centrality": 0.5,
                "accounting_ei_valid_flag": index != 5,
            }
        )
    return pl.DataFrame(rows)


def _transition_panel() -> pl.DataFrame:
    rows = []
    states = [
        TransitionState.NO_NEXT_YEAR.value,
        TransitionState.INSUFFICIENT_DATA_TRANSITION.value,
        TransitionState.GREEN_EMBEDDING_IMPROVEMENT.value,
        TransitionState.CAPABILITY_GAIN.value,
        TransitionState.REGIME_SWITCH.value,
        TransitionState.MIXED_MOVEMENT.value,
    ]
    for index, state in enumerate(states):
        rows.append(
            {
                "country_sector": f"C{index} | Country {index} | Industry | Sector",
                "year": 1995,
                "next_year": None if state == TransitionState.NO_NEXT_YEAR.value else 1996,
                "next_regime_membership": None
                if state == TransitionState.NO_NEXT_YEAR.value
                else (
                    RegimeLabel.INSUFFICIENT_DATA.value
                    if state == TransitionState.INSUFFICIENT_DATA_TRANSITION.value
                    else RegimeLabel.MIXED_INTERMEDIATE.value
                ),
                "transition_state": state,
                "transition_confidence": 0.75 if state != TransitionState.INSUFFICIENT_DATA_TRANSITION.value else 0.25,
                "regime_switch_flag": None
                if state == TransitionState.NO_NEXT_YEAR.value
                else state == TransitionState.REGIME_SWITCH.value,
                "delta_emissions_intensity_gap": -0.1,
                "delta_network_green_exposure": 0.2,
                "delta_green_capability": 0.3,
                "delta_general_capability": 0.3,
                "delta_supplier_lock_in": -0.2,
                "delta_brown_centrality": -0.1,
            }
        )
    return pl.DataFrame(rows)


def _handoff() -> pl.DataFrame:
    return build_regime_handoff_panel(_regime_panel(), _transition_panel())


def test_build_regime_handoff_panel_preserves_rows() -> None:
    assert _handoff().height == _regime_panel().height


def test_build_regime_handoff_panel_excludes_no_next_year_from_learning() -> None:
    row = _handoff().filter(pl.col("transition_state") == TransitionState.NO_NEXT_YEAR.value).row(0, named=True)
    assert row["mechanism_learning_eligible_flag"] is False
    assert row["mechanism_learning_exclusion_reason"] == "no_next_year"


def test_build_regime_handoff_panel_excludes_insufficient_data_transition() -> None:
    row = _handoff().filter(pl.col("transition_state") == TransitionState.INSUFFICIENT_DATA_TRANSITION.value).row(0, named=True)
    assert row["mechanism_learning_eligible_flag"] is False
    assert row["mechanism_learning_exclusion_reason"] == "insufficient_data"


def test_build_regime_handoff_panel_sets_mechanism_specific_flags() -> None:
    row = _handoff().filter(pl.col("transition_state") == TransitionState.GREEN_EMBEDDING_IMPROVEMENT.value).row(0, named=True)
    assert row["usable_for_network_mechanism_flag"] is True
    assert row["usable_for_capability_mechanism_flag"] is True


def test_build_regime_stability_summary_returns_expected_columns() -> None:
    summary = build_regime_stability_summary(_handoff())
    assert {"regime_membership", "n_rows", "eligible_share", "stability_rate", "switch_rate"}.issubset(summary.columns)


def test_build_transition_target_summary_maps_states_to_families() -> None:
    summary = build_transition_target_summary(_handoff())
    row = summary.filter(pl.col("transition_state") == TransitionState.GREEN_EMBEDDING_IMPROVEMENT.value).row(0, named=True)
    assert row["mechanism_target_family"] == MechanismTargetFamily.NETWORK_EMBEDDING.value


def test_build_transition_target_summary_marks_data_limited_states() -> None:
    summary = build_transition_target_summary(_handoff())
    row = summary.filter(pl.col("transition_state") == TransitionState.NO_NEXT_YEAR.value).row(0, named=True)
    assert row["target_status"] in {
        MechanismTargetStatus.DATA_LIMITED.value,
        MechanismTargetStatus.EXCLUDE_FROM_MECHANISM_LEARNING.value,
    }


def test_build_mechanism_target_candidates_excludes_data_limited() -> None:
    summary = build_transition_target_summary(_handoff())
    candidates = build_mechanism_target_candidates(_handoff(), summary)
    assert not candidates["transition_state"].is_in([
        TransitionState.NO_NEXT_YEAR.value,
        TransitionState.INSUFFICIENT_DATA_TRANSITION.value,
    ]).any()


def test_build_mechanism_learning_sample_summary_returns_families() -> None:
    summary = build_mechanism_learning_sample_summary(_handoff())
    assert set(summary["mechanism_target_family"].to_list()) == {
        MechanismTargetFamily.EMISSIONS_INTENSITY.value,
        MechanismTargetFamily.NETWORK_EMBEDDING.value,
        MechanismTargetFamily.CAPABILITY.value,
        MechanismTargetFamily.SUPPLIER_LOCK_IN.value,
        MechanismTargetFamily.BROWN_CENTRALITY.value,
    }


def test_validate_regime_handoff_outputs_rejects_regime_probability() -> None:
    handoff = _handoff().with_columns(pl.lit(0.5).alias("regime_probability"))
    summary = build_transition_target_summary(handoff)
    candidates = build_mechanism_target_candidates(handoff, summary)
    results = validate_regime_handoff_outputs(handoff, candidates, summary)
    assert any(result.check_name == "handoff_no_probability_or_scenario_columns" and result.n_failed > 0 for result in results)


def test_validate_regime_handoff_outputs_passes_valid_outputs() -> None:
    handoff = _handoff()
    summary = build_transition_target_summary(handoff)
    candidates = build_mechanism_target_candidates(handoff, summary)
    results = validate_regime_handoff_outputs(handoff, candidates, summary)
    assert all(result.status.value == "passed" for result in results)


def _write_pyproject(root: Path) -> None:
    root.joinpath("pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")


def test_build_regime_handoff_requires_regime_panel(tmp_path: Path) -> None:
    _write_pyproject(tmp_path)
    with pytest.raises(FileNotFoundError, match="Phase 3.1-3.3"):
        build_regime_handoff(tmp_path)


def test_build_regime_handoff_requires_transition_panel(tmp_path: Path) -> None:
    _write_pyproject(tmp_path)
    regime_dir = tmp_path / "data" / "abm_v5" / "regimes"
    regime_dir.mkdir(parents=True)
    _regime_panel().write_parquet(regime_dir / "historical_regime_panel_1995_2016.parquet")
    with pytest.raises(FileNotFoundError, match="Phase 3.4"):
        build_regime_handoff(tmp_path)


def test_build_regime_handoff_writes_outputs_with_fake_data(tmp_path: Path) -> None:
    _write_pyproject(tmp_path)
    regime_dir = tmp_path / "data" / "abm_v5" / "regimes"
    regime_dir.mkdir(parents=True)
    _regime_panel().write_parquet(regime_dir / "historical_regime_panel_1995_2016.parquet")
    _transition_panel().write_parquet(regime_dir / "historical_transition_panel_1995_2016.parquet")
    result = build_regime_handoff(tmp_path)
    assert isinstance(result, RegimeHandoffBuildResult)
    assert result.handoff_panel_path.exists()
    assert result.mechanism_target_candidates_path.exists()
    assert result.validation_path.exists()


def test_build_regime_handoff_preserves_backbone(tmp_path: Path) -> None:
    _write_pyproject(tmp_path)
    regime_dir = tmp_path / "data" / "abm_v5" / "regimes"
    regime_dir.mkdir(parents=True)
    _regime_panel().write_parquet(regime_dir / "historical_regime_panel_1995_2016.parquet")
    _transition_panel().write_parquet(regime_dir / "historical_transition_panel_1995_2016.parquet")
    result = build_regime_handoff(tmp_path)
    assert load_regime_handoff_panel(result.handoff_panel_path).height == _regime_panel().height


def test_init_exports_regime_handoff_objects() -> None:
    assert abm_v5.RegimeHandoffBuildResult is RegimeHandoffBuildResult
    assert abm_v5.MechanismTargetFamily is MechanismTargetFamily
    assert abm_v5.MechanismTargetStatus is MechanismTargetStatus
    assert abm_v5.build_regime_handoff_panel is build_regime_handoff_panel
    assert abm_v5.build_regime_handoff is build_regime_handoff
