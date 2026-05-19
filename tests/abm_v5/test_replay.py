import json

import pytest

from src.abm_v5.replay import (
    DEFAULT_REPLAY_MODE,
    DEFAULT_REPLAY_RUN_ID,
    HISTORICAL_END_YEAR,
    HISTORICAL_START_YEAR,
    ReplayMetadata,
    build_default_replay_state_columns,
    build_initial_replay_state,
    build_replay_scaffold,
    run_identity_historical_replay,
    validate_replay_scaffold_output,
)
from src.abm_v5.validation import ValidationStatus


def _handoff_df():
    import polars as pl

    return pl.DataFrame(
        {
            "country_sector": ["A | Detail | Cat | S1", "A | Detail | Cat | S1", "B | Detail | Cat | S2"],
            "country": ["A", "A", "B"],
            "country_detail": ["Detail", "Detail", "Detail"],
            "category": ["Cat", "Cat", "Cat"],
            "sector": ["S1", "S1", "S2"],
            "year": [1995, 1996, 1995],
            "output": [100.0, 110.0, 80.0],
            "final_demand": [25.0, 30.0, 20.0],
            "emissions": [10.0, 9.0, 15.0],
            "emissions_intensity": [0.10, 0.0818, 0.1875],
            "local_greenness": [0.7, 0.8, 0.3],
            "emissions_intensity_gap": [0.0, -0.02, 0.08],
            "green_capability": [0.6, 0.65, 0.2],
            "general_capability": [0.7, 0.72, 0.4],
            "network_green_exposure": [0.55, 0.58, 0.25],
            "brown_centrality": [0.2, 0.18, 0.7],
            "supplier_lock_in": [0.4, 0.35, 0.8],
            "supplier_count": [2, 2, 1],
            "buyer_count": [1, 1, 2],
            "supplier_concentration_hhi": [0.5, 0.45, 1.0],
            "buyer_concentration_hhi": [1.0, 1.0, 0.5],
            "import_dependence_proxy": [0.1, 0.1, 0.3],
            "export_dependence_proxy": [0.2, 0.2, 0.4],
            "phase_space_empirical_completeness": [1.0, 1.0, 1.0],
            "phase_space_ready_for_regime_discovery_flag": [True, True, True],
            "regime_membership": ["mixed_intermediate", "green_capable_embedded", "brown_central_constrained"],
            "regime_confidence": [0.5, 1.0, 0.75],
            "transition_state": ["green_embedding_improvement", "no_next_year", "no_next_year"],
            "transition_confidence": [0.9, 1.0, 1.0],
            "mechanism_learning_eligible_flag": [True, False, False],
        }
    )


def _metadata() -> ReplayMetadata:
    return ReplayMetadata(
        run_id=DEFAULT_REPLAY_RUN_ID,
        start_year=HISTORICAL_START_YEAR,
        end_year=HISTORICAL_END_YEAR,
        mode=DEFAULT_REPLAY_MODE,
        active_mechanisms=("identity_replay",),
        scaffold_only=True,
        notes="No behavioural mechanisms are implemented.",
    )


def test_build_default_replay_state_columns_contains_core_variables():
    columns = build_default_replay_state_columns()

    assert "output" in columns.core_state_columns
    assert "network_green_exposure" in columns.core_state_columns
    assert "transition_state" in columns.regime_columns


def test_replay_metadata_to_dict_is_json_serializable():
    payload = _metadata().to_dict()

    assert json.loads(json.dumps(payload))["active_mechanisms"] == ["identity_replay"]


def test_build_initial_replay_state_adds_simulated_columns():
    replay = build_initial_replay_state(_handoff_df())

    assert "simulated_output" in replay.columns
    assert "simulated_network_green_exposure" in replay.columns
    assert "replay_scaffold_only_flag" in replay.columns


def test_build_initial_replay_state_copies_observed_values():
    replay = build_initial_replay_state(_handoff_df())

    assert replay["simulated_output"].to_list() == replay["output"].to_list()
    assert replay["simulated_green_capability"].to_list() == replay["green_capability"].to_list()


def test_run_identity_historical_replay_preserves_observed_state():
    replay = run_identity_historical_replay(_handoff_df(), _metadata())

    assert replay["simulated_emissions"].to_list() == replay["emissions"].to_list()
    assert replay["simulated_supplier_lock_in"].to_list() == replay["supplier_lock_in"].to_list()


def test_validate_replay_scaffold_output_passes_identity_replay():
    replay = run_identity_historical_replay(_handoff_df(), _metadata())

    results = validate_replay_scaffold_output(replay)

    assert all(result.status is ValidationStatus.PASSED for result in results)


def test_validate_replay_scaffold_output_rejects_scenario_columns():
    import polars as pl

    replay = run_identity_historical_replay(_handoff_df(), _metadata()).with_columns(pl.lit("scenario").alias("scenario_id"))

    results = validate_replay_scaffold_output(replay)

    assert any(result.check_name == "replay_no_scenario_or_policy_columns" and result.status is ValidationStatus.FAILED for result in results)


def test_build_replay_scaffold_requires_handoff_panel(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fake'\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Phase 3.5"):
        build_replay_scaffold(tmp_path)


def test_build_replay_scaffold_writes_outputs_with_fake_data(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fake'\n", encoding="utf-8")
    handoff_path = tmp_path / "data" / "abm_v5" / "regimes" / "regime_handoff_panel_1995_2016.parquet"
    handoff_path.parent.mkdir(parents=True)
    _handoff_df().write_parquet(handoff_path)

    result = build_replay_scaffold(tmp_path)

    assert result.replay_panel_path.exists()
    assert result.replay_metadata_path.exists()
    assert result.mechanism_registry_path.exists()
    assert result.ablation_config_path.exists()
    assert result.validation_path.exists()


def test_build_replay_scaffold_preserves_backbone(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fake'\n", encoding="utf-8")
    handoff_path = tmp_path / "data" / "abm_v5" / "regimes" / "regime_handoff_panel_1995_2016.parquet"
    handoff_path.parent.mkdir(parents=True)
    handoff = _handoff_df()
    handoff.write_parquet(handoff_path)

    result = build_replay_scaffold(tmp_path)

    assert result.n_rows == handoff.height
    assert result.n_agents == handoff["country_sector"].n_unique()


def test_init_exports_replay_objects():
    import src.abm_v5 as abm_v5

    assert hasattr(abm_v5, "ReplayMetadata")
    assert hasattr(abm_v5, "ReplayScaffoldBuildResult")
    assert hasattr(abm_v5, "build_replay_scaffold")
    assert hasattr(abm_v5, "build_default_mechanism_runtime_registry")
