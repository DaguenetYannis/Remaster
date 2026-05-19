import pytest

from src.abm_v5 import (
    ABMV5SchemaRegistry,
    ColumnContract,
    ColumnDType,
    ComplexityLevel,
    FeedbackStatus,
    SchemaFamily,
    SourceStatus,
    TableSchemaContract,
    ValidationLayer,
    build_agent_identity_schema,
    build_agent_state_schema,
    build_default_schema_registry,
    build_edge_state_schema,
    build_phase_space_schema,
    build_regime_schema,
    build_simulation_output_schema,
    build_validation_report_schema,
)
from src.abm_v5.ontology import get_core_state_variable_specs


def toy_column(name: str = "country_sector") -> ColumnContract:
    return ColumnContract(
        name=name,
        dtype=ColumnDType.STRING,
        nullable=False,
        description="Toy column for schema contract tests.",
        source_status=SourceStatus.DERIVED,
        feedback_status=FeedbackStatus.DIAGNOSTIC,
        complexity_level=ComplexityLevel.MESO,
        validation_layers=(ValidationLayer.STRUCTURAL_VALIDITY,),
    )


def test_column_contract_rejects_empty_name() -> None:
    column = toy_column(name="")

    with pytest.raises(ValueError, match="name"):
        column.validate()


def test_column_contract_rejects_empty_validation_layers() -> None:
    column = ColumnContract(
        name="x",
        dtype=ColumnDType.FLOAT,
        nullable=True,
        description="Column without validation layers.",
        source_status=SourceStatus.DERIVED,
        feedback_status=FeedbackStatus.DIAGNOSTIC,
        complexity_level=ComplexityLevel.MESO,
        validation_layers=(),
    )

    with pytest.raises(ValueError, match="validation_layers"):
        column.validate()


def test_table_schema_contract_rejects_duplicate_columns() -> None:
    schema = TableSchemaContract(
        name="toy",
        family=SchemaFamily.AGENT_IDENTITY,
        columns=(toy_column("x"), toy_column("x")),
        primary_keys=("x",),
        description="Toy schema.",
    )

    with pytest.raises(ValueError, match="unique"):
        schema.validate()


def test_table_schema_contract_rejects_missing_primary_key() -> None:
    schema = TableSchemaContract(
        name="toy",
        family=SchemaFamily.AGENT_IDENTITY,
        columns=(toy_column("x"),),
        primary_keys=("missing",),
        description="Toy schema.",
    )

    with pytest.raises(ValueError, match="primary key"):
        schema.validate()


def test_agent_identity_schema_matches_country_sector_key() -> None:
    schema = build_agent_identity_schema()

    schema.validate()

    assert schema.family is SchemaFamily.AGENT_IDENTITY
    assert schema.primary_keys == ("country_sector",)
    assert schema.get_column("country_sector").dtype is ColumnDType.STRING
    assert not schema.get_column("country_sector").nullable


def test_agent_state_schema_contains_all_core_ontology_variables() -> None:
    schema = build_agent_state_schema()
    ontology_names = {spec.name for spec in get_core_state_variable_specs()}

    schema.validate()

    assert set(schema.column_names()) == ontology_names
    assert schema.get_column("year").dtype is ColumnDType.INTEGER
    assert schema.get_column("supplier_opportunity_set").dtype is ColumnDType.LIST
    assert schema.get_column("supplier_weights").dtype is ColumnDType.OBJECT


def test_agent_state_schema_uses_country_sector_year_unique_key() -> None:
    schema = build_agent_state_schema()

    assert schema.primary_keys == ("country_sector", "year")
    assert schema.unique_keys == (("country_sector", "year"),)
    assert schema.sorted_by == ("country_sector", "year")


def test_edge_state_schema_uses_supplier_buyer_year_key() -> None:
    schema = build_edge_state_schema()

    schema.validate()

    assert schema.family is SchemaFamily.EDGE_STATE
    assert schema.primary_keys == (
        "year",
        "supplier_country_sector",
        "buyer_country_sector",
    )
    assert schema.unique_keys == (
        ("year", "supplier_country_sector", "buyer_country_sector"),
    )


def test_phase_space_schema_contains_contract_variables() -> None:
    schema = build_phase_space_schema()

    assert {
        "country_sector",
        "year",
        "emissions_intensity_gap",
        "green_capability",
        "network_green_exposure",
        "phase_space_position",
    }.issubset(set(schema.column_names()))


def test_regime_schema_contains_probability_and_switch_fields() -> None:
    schema = build_regime_schema()

    assert schema.has_column("regime_probability")
    assert schema.has_column("regime_confidence")
    assert schema.has_column("regime_switch_flag")
    assert schema.get_column("regime_switch_flag").dtype is ColumnDType.BOOLEAN


def test_simulation_output_schema_is_scenario_run_country_sector_year_keyed() -> None:
    schema = build_simulation_output_schema()

    assert schema.family is SchemaFamily.SIMULATION_OUTPUT
    assert schema.primary_keys == ("scenario_id", "run_id", "country_sector", "year")
    assert schema.unique_keys == (("scenario_id", "run_id", "country_sector", "year"),)


def test_validation_report_schema_contains_passed_and_message() -> None:
    schema = build_validation_report_schema()

    assert schema.family is SchemaFamily.VALIDATION_REPORT
    assert schema.has_column("passed")
    assert schema.get_column("passed").dtype is ColumnDType.BOOLEAN
    assert schema.has_column("message")


def test_default_schema_registry_validates() -> None:
    registry = build_default_schema_registry()

    registry.validate()

    assert isinstance(registry, ABMV5SchemaRegistry)
    assert registry.schema_names() == (
        "agent_identity",
        "agent_state",
        "edge_state",
        "phase_space",
        "regime",
        "simulation_output",
        "validation_report",
    )


def test_default_schema_registry_retrieves_by_name() -> None:
    registry = build_default_schema_registry()

    schema = registry.get_schema("agent_state")

    assert schema.family is SchemaFamily.AGENT_STATE


def test_default_schema_registry_retrieves_by_family() -> None:
    registry = build_default_schema_registry()

    schema = registry.get_schema_by_family("edge_state")

    assert schema.name == "edge_state"
    assert registry.get_schema_by_family(SchemaFamily.REGIME).name == "regime"


def test_init_exports_schema_objects() -> None:
    import src.abm_v5 as abm_v5

    assert abm_v5.ColumnDType is ColumnDType
    assert abm_v5.SchemaFamily is SchemaFamily
    assert abm_v5.ColumnContract is ColumnContract
    assert abm_v5.TableSchemaContract is TableSchemaContract
    assert abm_v5.ABMV5SchemaRegistry is ABMV5SchemaRegistry
    assert abm_v5.build_default_schema_registry is build_default_schema_registry
