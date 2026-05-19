import json

import pytest

from src.abm_v5.mechanisms import (
    AblationConfig,
    MechanismExecutionStatus,
    MechanismPhase,
    MechanismRuntimeSpec,
    build_default_mechanism_runtime_registry,
    mechanism_registry_to_dict,
    write_default_ablation_config,
    write_mechanism_registry_json,
)
from src.abm_v5.ontology import MechanismName, get_core_mechanism_specs


def test_build_default_mechanism_runtime_registry_covers_core_mechanisms():
    registry = build_default_mechanism_runtime_registry()
    runtime_names = {spec.mechanism_name for spec in registry}
    core_names = {spec.name for spec in get_core_mechanism_specs()}

    assert runtime_names == core_names
    assert all(spec.execution_status is MechanismExecutionStatus.NOT_IMPLEMENTED for spec in registry)


def test_mechanism_runtime_spec_validates():
    spec = MechanismRuntimeSpec(
        mechanism_name=MechanismName.PRODUCTION_FEASIBILITY,
        phase=MechanismPhase.PHASE_4_5_PRODUCTION_FEASIBILITY,
        execution_status=MechanismExecutionStatus.NOT_IMPLEMENTED,
        input_variables=("desired_output",),
        update_targets=("realized_output",),
        required_inputs_available=False,
        description="Runtime metadata only.",
    )

    spec.validate()


def test_mechanism_registry_to_dict_is_json_serializable():
    registry = build_default_mechanism_runtime_registry()

    encoded = json.dumps(mechanism_registry_to_dict(registry))

    assert "behavioural_mechanisms_implemented" in encoded


def test_ablation_config_phase_4_1_defaults_disable_actual_mechanisms():
    config = AblationConfig()

    config.validate()

    assert config.active_mechanism_families() == ("identity_replay",)
    assert not config.emissions_intensity_enabled
    assert not config.network_exposure_enabled
    assert not config.supplier_lock_in_enabled
    assert not config.production_feasibility_enabled
    assert not config.capability_accumulation_enabled
    assert not config.brown_centrality_enabled


def test_ablation_config_rejects_policy_regime_enabled():
    with pytest.raises(ValueError, match="policy_regime_enabled"):
        AblationConfig(policy_regime_enabled=True).validate()


def test_write_mechanism_registry_json_writes_file(tmp_path):
    path = tmp_path / "mechanism_registry.json"

    write_mechanism_registry_json(build_default_mechanism_runtime_registry(), path)

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["behavioural_mechanisms_implemented"] is False


def test_write_default_ablation_config_writes_file(tmp_path):
    path = tmp_path / "ablation_config.json"

    write_default_ablation_config(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["identity_replay_enabled"] is True
    assert payload["active_mechanism_families"] == ["identity_replay"]
