from pathlib import Path

import pytest

from src.abm_v5 import (
    ABMV5OntologyRegistry,
    AgentIdentityOntology,
    FeedbackStatus,
    FunctionalRole,
    FunctionalRoleSpec,
    MechanismName,
    build_default_ontology_registry,
    get_core_mechanism_specs,
    get_core_state_variable_specs,
    get_functional_role_specs,
)


REQUIRED_STATE_VARIABLES = {
    "country_sector",
    "year",
    "output",
    "emissions",
    "emissions_intensity",
    "general_capability",
    "green_capability",
    "supplier_weights",
    "supplier_opportunity_set",
    "production_feasibility",
    "network_green_exposure",
    "brown_centrality",
    "energy_dependence",
    "capital_stock_inertia",
    "policy_exposure",
    "phase_space_position",
    "regime_membership",
    "regime_probability",
}


def test_agent_identity_requires_country_sector_primary_key() -> None:
    identity = AgentIdentityOntology(primary_key="country")

    with pytest.raises(ValueError, match="country_sector"):
        identity.validate()


def test_agent_identity_required_columns_include_auxiliary_keys() -> None:
    identity = AgentIdentityOntology()

    assert identity.required_columns() == (
        "country_sector",
        "country",
        "country_detail",
        "category",
        "sector",
    )


def test_core_state_variable_specs_are_unique() -> None:
    names = [spec.name for spec in get_core_state_variable_specs()]

    assert len(names) == len(set(names))


def test_core_state_variable_specs_validate() -> None:
    for spec in get_core_state_variable_specs():
        spec.validate()


def test_core_state_variable_specs_include_required_contract_variables() -> None:
    names = {spec.name for spec in get_core_state_variable_specs()}

    assert REQUIRED_STATE_VARIABLES.issubset(names)


def test_functional_roles_are_recomputed_yearly_and_not_mutually_exclusive() -> None:
    for role_spec in get_functional_role_specs():
        role_spec.validate()
        assert role_spec.recompute_each_year
        assert not role_spec.mutually_exclusive


def test_functional_roles_are_specs_not_agent_classes() -> None:
    ontology_path = Path("src") / "abm_v5" / "ontology.py"
    source_text = ontology_path.read_text()

    assert "EnergyAgent" not in source_text
    assert "SupplierAgent" not in source_text
    assert "GreenAgent" not in source_text

    class_names = []
    for line in source_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("class "):
            class_names.append(stripped.split("class ", 1)[1].split("(", 1)[0].split(":", 1)[0])

    assert [
        class_name
        for class_name in class_names
        if class_name.endswith("Agent") and class_name != "AgentIdentityOntology"
    ] == []


def test_core_mechanism_specs_are_unique() -> None:
    names = [spec.name for spec in get_core_mechanism_specs()]

    assert len(names) == len(set(names))
    assert set(names) == set(MechanismName)


def test_core_mechanism_specs_validate() -> None:
    for spec in get_core_mechanism_specs():
        spec.validate()


def test_mechanism_specs_reference_existing_state_variables() -> None:
    state_names = {spec.name for spec in get_core_state_variable_specs()}

    for mechanism in get_core_mechanism_specs():
        assert set(mechanism.input_variables).issubset(state_names)
        assert set(mechanism.update_targets).issubset(state_names)


def test_default_ontology_registry_validates() -> None:
    registry = build_default_ontology_registry()

    registry.validate()

    assert isinstance(registry, ABMV5OntologyRegistry)


def test_default_ontology_registry_get_state_variable() -> None:
    registry = build_default_ontology_registry()

    spec = registry.get_state_variable("emissions_intensity")

    assert spec.name == "emissions_intensity"


def test_default_ontology_registry_get_mechanism_accepts_string() -> None:
    registry = build_default_ontology_registry()

    mechanism = registry.get_mechanism("supplier_substitution")

    assert mechanism.name is MechanismName.SUPPLIER_SUBSTITUTION


def test_policy_mechanism_is_scenario_driven_not_foundation_executed() -> None:
    mechanisms = {spec.name: spec for spec in get_core_mechanism_specs()}

    policy_mechanism = mechanisms[MechanismName.POLICY_REGIME_EXPOSURE]

    assert policy_mechanism.feedback_status is FeedbackStatus.SCENARIO_DRIVEN
    for mechanism_name, mechanism in mechanisms.items():
        if mechanism_name is not MechanismName.POLICY_REGIME_EXPOSURE:
            assert mechanism.feedback_status is FeedbackStatus.DESIGN_TARGET


def test_init_exports_ontology_objects() -> None:
    import src.abm_v5 as abm_v5

    assert abm_v5.AgentIdentityOntology is AgentIdentityOntology
    assert abm_v5.FunctionalRole is FunctionalRole
    assert abm_v5.FunctionalRoleSpec is FunctionalRoleSpec
    assert abm_v5.MechanismName is MechanismName
    assert abm_v5.build_default_ontology_registry is build_default_ontology_registry
