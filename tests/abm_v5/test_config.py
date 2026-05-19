import pytest

from src.abm_v5 import (
    ABMV5Config,
    ComplexityLayer,
    ComplexityLevel,
    FeedbackStatus,
    HistoricalWindowConfig,
    ModelStage,
    OntologyConfig,
    SchemaConfig,
    SourceStatus,
    ValidationLayer,
)


def test_historical_window_years_are_inclusive() -> None:
    window = HistoricalWindowConfig(start_year=2000, end_year=2002)

    assert window.years() == [2000, 2001, 2002]


def test_historical_window_rejects_invalid_order() -> None:
    window = HistoricalWindowConfig(start_year=2016, end_year=1995)

    with pytest.raises(ValueError, match="start_year"):
        window.validate()


def test_abmv5_config_validates_default() -> None:
    config = ABMV5Config()

    config.validate()

    assert ComplexityLayer.SCENARIO not in config.enabled_layers


def test_abmv5_config_rejects_negative_seed() -> None:
    config = ABMV5Config(random_seed=-1)

    with pytest.raises(ValueError, match="random_seed"):
        config.validate()


def test_abmv5_config_rejects_scenario_layer_during_foundation() -> None:
    config = ABMV5Config(
        active_stage=ModelStage.FOUNDATION,
        enabled_layers=(ComplexityLayer.ACCOUNTING, ComplexityLayer.SCENARIO),
    )

    with pytest.raises(ValueError, match="SCENARIO"):
        config.validate()


def test_abmv5_config_accepts_string_layer_lookup() -> None:
    config = ABMV5Config(enabled_layers=(ComplexityLayer.ACCOUNTING,))

    assert config.is_layer_enabled("accounting")
    assert not config.is_layer_enabled(ComplexityLayer.CAPABILITY)


def test_schema_config_rejects_empty_keys() -> None:
    with pytest.raises(ValueError, match="primary_agent_key"):
        SchemaConfig(primary_agent_key="").validate()

    with pytest.raises(ValueError, match="time_key"):
        SchemaConfig(time_key="").validate()


def test_ontology_config_rejects_fixed_agent_classes() -> None:
    config = OntologyConfig(allow_fixed_agent_classes=True)

    with pytest.raises(ValueError, match="fixed_agent_classes"):
        config.validate()


def test_ontology_config_defaults_match_contract() -> None:
    config = OntologyConfig()

    config.validate()

    assert not config.allow_fixed_agent_classes
    assert config.allow_overlapping_roles
    assert config.recompute_roles_each_year
    assert config.agent_unit == "country_sector"


def test_enums_have_expected_values() -> None:
    assert [layer.value for layer in ComplexityLayer] == [
        "accounting",
        "production_network",
        "capability",
        "supplier_adaptation",
        "energy_inertia",
        "policy_regime",
        "regime_discovery",
        "scenario",
    ]
    assert [level.value for level in ComplexityLevel] == [
        "macro",
        "meso",
        "micro",
        "cross_level",
    ]
    assert [status.value for status in FeedbackStatus] == [
        "observed",
        "diagnostic",
        "exogenous",
        "endogenous",
        "scenario_driven",
        "design_target",
    ]
    assert [layer.value for layer in ValidationLayer] == [
        "accounting_validity",
        "structural_validity",
        "mechanism_validity",
        "historical_plausibility",
        "ablation_validity",
        "scenario_credibility",
    ]
    assert [status.value for status in SourceStatus] == [
        "raw_observed",
        "derived",
        "estimated",
        "proxy",
        "simulated",
        "placeholder",
    ]


def test_abmv5_init_exports_config_objects() -> None:
    import src.abm_v5 as abm_v5

    assert abm_v5.ABMV5Config is ABMV5Config
    assert abm_v5.HistoricalWindowConfig is HistoricalWindowConfig
    assert abm_v5.SchemaConfig is SchemaConfig
    assert abm_v5.OntologyConfig is OntologyConfig
    assert abm_v5.ComplexityLayer is ComplexityLayer
    assert abm_v5.ModelStage is ModelStage
