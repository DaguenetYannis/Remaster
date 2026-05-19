from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from src.abm_v5.ontology import MechanismName, get_core_mechanism_specs


class MechanismExecutionStatus(str, Enum):
    NOT_IMPLEMENTED = "not_implemented"
    NO_OP = "no_op"
    ACTIVE = "active"
    DISABLED = "disabled"


class MechanismPhase(str, Enum):
    PHASE_4_1_SCAFFOLD = "phase_4_1_scaffold"
    PHASE_4_2_EMISSIONS_INTENSITY = "phase_4_2_emissions_intensity"
    PHASE_4_3_NETWORK_EXPOSURE = "phase_4_3_network_exposure"
    PHASE_4_4_SUPPLIER_LOCK_IN = "phase_4_4_supplier_lock_in"
    PHASE_4_5_PRODUCTION_FEASIBILITY = "phase_4_5_production_feasibility"
    PHASE_4_6_CAPABILITY_ACCUMULATION = "phase_4_6_capability_accumulation"
    PHASE_4_7_HISTORICAL_REPLAY_VALIDATION = "phase_4_7_historical_replay_validation"
    PHASE_4_8_ABLATION_TESTS = "phase_4_8_ablation_tests"


@dataclass(frozen=True)
class MechanismRuntimeSpec:
    mechanism_name: MechanismName
    phase: MechanismPhase
    execution_status: MechanismExecutionStatus
    input_variables: tuple[str, ...]
    update_targets: tuple[str, ...]
    required_inputs_available: bool
    description: str
    ablation_enabled: bool = True

    def validate(self) -> None:
        if not isinstance(self.mechanism_name, MechanismName):
            raise ValueError("mechanism_name must be MechanismName.")
        if not isinstance(self.phase, MechanismPhase):
            raise ValueError("phase must be MechanismPhase.")
        if not isinstance(self.execution_status, MechanismExecutionStatus):
            raise ValueError("execution_status must be MechanismExecutionStatus.")
        if not self.input_variables:
            raise ValueError("input_variables must not be empty.")
        if not self.update_targets:
            raise ValueError("update_targets must not be empty.")
        if not self.description:
            raise ValueError("description must not be empty.")
        if not isinstance(self.required_inputs_available, bool):
            raise ValueError("required_inputs_available must be bool.")
        if not isinstance(self.ablation_enabled, bool):
            raise ValueError("ablation_enabled must be bool.")


PHASE_BY_MECHANISM = {
    MechanismName.EMISSIONS_INTENSITY_TRANSITION: MechanismPhase.PHASE_4_2_EMISSIONS_INTENSITY,
    MechanismName.NETWORK_GREEN_EXPOSURE_FEEDBACK: MechanismPhase.PHASE_4_3_NETWORK_EXPOSURE,
    MechanismName.SUPPLIER_SUBSTITUTION: MechanismPhase.PHASE_4_4_SUPPLIER_LOCK_IN,
    MechanismName.BOUNDED_SUPPLIER_SEARCH: MechanismPhase.PHASE_4_4_SUPPLIER_LOCK_IN,
    MechanismName.BROWN_LOCK_IN_FEEDBACK: MechanismPhase.PHASE_4_4_SUPPLIER_LOCK_IN,
    MechanismName.PRODUCTION_FEASIBILITY: MechanismPhase.PHASE_4_5_PRODUCTION_FEASIBILITY,
    MechanismName.ENERGY_DEPENDENCE_CONSTRAINT: MechanismPhase.PHASE_4_5_PRODUCTION_FEASIBILITY,
    MechanismName.CAPITAL_STOCK_INERTIA: MechanismPhase.PHASE_4_5_PRODUCTION_FEASIBILITY,
    MechanismName.CAPABILITY_ACCUMULATION: MechanismPhase.PHASE_4_6_CAPABILITY_ACCUMULATION,
    MechanismName.DIRECTED_ECOSYSTEM_MOVEMENT: MechanismPhase.PHASE_4_6_CAPABILITY_ACCUMULATION,
    MechanismName.POLICY_REGIME_EXPOSURE: MechanismPhase.PHASE_4_8_ABLATION_TESTS,
    MechanismName.PHASE_SPACE_REGIME_SWITCHING: MechanismPhase.PHASE_4_7_HISTORICAL_REPLAY_VALIDATION,
}


def build_default_mechanism_runtime_registry() -> tuple[MechanismRuntimeSpec, ...]:
    registry: list[MechanismRuntimeSpec] = []
    for spec in get_core_mechanism_specs():
        runtime = MechanismRuntimeSpec(
            mechanism_name=spec.name,
            phase=PHASE_BY_MECHANISM[spec.name],
            execution_status=MechanismExecutionStatus.NOT_IMPLEMENTED,
            input_variables=spec.input_variables,
            update_targets=spec.update_targets,
            required_inputs_available=False,
            description=(
                "Phase 4.1 registry entry only. Behavioural equations are not implemented; "
                f"ontology description: {spec.description}"
            ),
            ablation_enabled=True,
        )
        runtime.validate()
        registry.append(runtime)
    return tuple(registry)


def mechanism_registry_to_dict(registry: tuple[MechanismRuntimeSpec, ...]) -> dict[str, Any]:
    entries = []
    for spec in registry:
        spec.validate()
        entries.append(
            {
                "mechanism_name": spec.mechanism_name.value,
                "phase": spec.phase.value,
                "execution_status": spec.execution_status.value,
                "input_variables": list(spec.input_variables),
                "update_targets": list(spec.update_targets),
                "required_inputs_available": spec.required_inputs_available,
                "description": spec.description,
                "ablation_enabled": spec.ablation_enabled,
            }
        )
    return {
        "registry_scope": "abm_v5_phase_4_1_mechanism_runtime_registry",
        "behavioural_mechanisms_implemented": False,
        "mechanisms": entries,
    }


def write_mechanism_registry_json(registry: tuple[MechanismRuntimeSpec, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mechanism_registry_to_dict(registry), indent=2, sort_keys=True), encoding="utf-8")


@dataclass(frozen=True)
class AblationConfig:
    emissions_intensity_enabled: bool = False
    network_exposure_enabled: bool = False
    supplier_lock_in_enabled: bool = False
    production_feasibility_enabled: bool = False
    capability_accumulation_enabled: bool = False
    brown_centrality_enabled: bool = False
    policy_regime_enabled: bool = False
    identity_replay_enabled: bool = True

    def validate(self) -> None:
        if not self.identity_replay_enabled:
            raise ValueError("identity_replay_enabled must be True in Phase 4.1.")
        if self.policy_regime_enabled:
            raise ValueError("policy_regime_enabled is forbidden in Phase 4.1.")
        flags = (
            self.emissions_intensity_enabled,
            self.network_exposure_enabled,
            self.supplier_lock_in_enabled,
            self.production_feasibility_enabled,
            self.capability_accumulation_enabled,
            self.brown_centrality_enabled,
        )
        if any(flags):
            raise ValueError("Actual mechanism flags must be False in Phase 4.1.")

    def active_mechanism_families(self) -> tuple[str, ...]:
        self.validate()
        return ("identity_replay",)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "emissions_intensity_enabled": self.emissions_intensity_enabled,
            "network_exposure_enabled": self.network_exposure_enabled,
            "supplier_lock_in_enabled": self.supplier_lock_in_enabled,
            "production_feasibility_enabled": self.production_feasibility_enabled,
            "capability_accumulation_enabled": self.capability_accumulation_enabled,
            "brown_centrality_enabled": self.brown_centrality_enabled,
            "policy_regime_enabled": self.policy_regime_enabled,
            "identity_replay_enabled": self.identity_replay_enabled,
            "active_mechanism_families": list(self.active_mechanism_families()),
        }


def write_default_ablation_config(path: Path) -> None:
    config = AblationConfig()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
