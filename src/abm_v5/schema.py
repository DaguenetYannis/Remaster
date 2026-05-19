from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.abm_v5.config import (
    ComplexityLevel,
    FeedbackStatus,
    SourceStatus,
    ValidationLayer,
)
from src.abm_v5.ontology import AgentStateLayer, get_core_state_variable_specs


class ColumnDType(str, Enum):
    """Portable dtype vocabulary for ABM v5 schema metadata."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    CATEGORY = "category"
    OBJECT = "object"
    LIST = "list"
    DICT = "dict"


class SchemaFamily(str, Enum):
    """Schema families expected by the ABM v5 contract."""

    AGENT_IDENTITY = "agent_identity"
    AGENT_STATE = "agent_state"
    EDGE_STATE = "edge_state"
    PHASE_SPACE = "phase_space"
    REGIME = "regime"
    MECHANISM = "mechanism"
    SIMULATION_OUTPUT = "simulation_output"
    VALIDATION_REPORT = "validation_report"


@dataclass(frozen=True)
class ColumnContract:
    """Metadata-only contract for one schema column."""

    name: str
    dtype: ColumnDType
    nullable: bool
    description: str
    source_status: SourceStatus
    feedback_status: FeedbackStatus
    complexity_level: ComplexityLevel
    validation_layers: tuple[ValidationLayer, ...]
    expected_range: str | None = None
    semantic_unit: str | None = None
    required: bool = True

    def validate(self) -> None:
        """Validate column metadata without inspecting data."""
        if not self.name:
            raise ValueError("name cannot be empty.")
        if not self.description:
            raise ValueError(f"description cannot be empty for {self.name}.")
        if not isinstance(self.dtype, ColumnDType):
            raise ValueError(f"dtype must be ColumnDType for {self.name}.")
        if not isinstance(self.nullable, bool):
            raise ValueError(f"nullable must be bool for {self.name}.")
        if not isinstance(self.source_status, SourceStatus):
            raise ValueError(f"source_status must be SourceStatus for {self.name}.")
        if not isinstance(self.feedback_status, FeedbackStatus):
            raise ValueError(f"feedback_status must be FeedbackStatus for {self.name}.")
        if not isinstance(self.complexity_level, ComplexityLevel):
            raise ValueError(f"complexity_level must be ComplexityLevel for {self.name}.")
        if not self.validation_layers:
            raise ValueError(f"validation_layers must not be empty for {self.name}.")
        if any(not isinstance(layer, ValidationLayer) for layer in self.validation_layers):
            raise ValueError(f"validation_layers must contain ValidationLayer values for {self.name}.")
        if not isinstance(self.required, bool):
            raise ValueError(f"required must be bool for {self.name}.")


@dataclass(frozen=True)
class TableSchemaContract:
    """Metadata-only contract for one ABM v5 table."""

    name: str
    family: SchemaFamily
    columns: tuple[ColumnContract, ...]
    primary_keys: tuple[str, ...]
    description: str
    unique_keys: tuple[tuple[str, ...], ...] = ()
    sorted_by: tuple[str, ...] = ()

    def validate(self) -> None:
        """Validate table metadata and column references."""
        if not self.name:
            raise ValueError("name cannot be empty.")
        if not self.description:
            raise ValueError(f"description cannot be empty for {self.name}.")
        if not isinstance(self.family, SchemaFamily):
            raise ValueError(f"family must be SchemaFamily for {self.name}.")
        if not self.columns:
            raise ValueError(f"columns must not be empty for {self.name}.")
        for column in self.columns:
            column.validate()

        column_names = self.column_names()
        known_columns = set(column_names)
        if len(column_names) != len(known_columns):
            raise ValueError(f"column names must be unique for {self.name}.")
        if not self.primary_keys:
            raise ValueError(f"primary_keys must not be empty for {self.name}.")
        for primary_key in self.primary_keys:
            if primary_key not in known_columns:
                raise ValueError(f"primary key {primary_key!r} is missing from {self.name}.")
        for unique_key in self.unique_keys:
            for column_name in unique_key:
                if column_name not in known_columns:
                    raise ValueError(f"unique key column {column_name!r} is missing from {self.name}.")
        for column_name in self.sorted_by:
            if column_name not in known_columns:
                raise ValueError(f"sorted_by column {column_name!r} is missing from {self.name}.")

    def column_names(self) -> tuple[str, ...]:
        """Return all column names in contract order."""
        return tuple(column.name for column in self.columns)

    def required_columns(self) -> tuple[str, ...]:
        """Return required column names in contract order."""
        return tuple(column.name for column in self.columns if column.required)

    def get_column(self, name: str) -> ColumnContract:
        """Return a column contract by name."""
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(f"Unknown column {name!r} in schema {self.name!r}.")

    def has_column(self, name: str) -> bool:
        """Return whether a column exists in the schema."""
        return name in self.column_names()


def _column(
    name: str,
    dtype: ColumnDType,
    nullable: bool,
    description: str,
    source_status: SourceStatus = SourceStatus.DERIVED,
    feedback_status: FeedbackStatus = FeedbackStatus.DIAGNOSTIC,
    complexity_level: ComplexityLevel = ComplexityLevel.MESO,
    validation_layers: tuple[ValidationLayer, ...] = (ValidationLayer.STRUCTURAL_VALIDITY,),
    expected_range: str | None = None,
    semantic_unit: str | None = None,
    required: bool = True,
) -> ColumnContract:
    return ColumnContract(
        name=name,
        dtype=dtype,
        nullable=nullable,
        description=description,
        source_status=source_status,
        feedback_status=feedback_status,
        complexity_level=complexity_level,
        validation_layers=validation_layers,
        expected_range=expected_range,
        semantic_unit=semantic_unit,
        required=required,
    )


def build_agent_identity_schema() -> TableSchemaContract:
    """Build the country-sector identity schema contract."""
    columns = (
        _column("country_sector", ColumnDType.STRING, False, "Stable country-sector primary key."),
        _column("country", ColumnDType.STRING, False, "Country code or label."),
        _column("country_detail", ColumnDType.STRING, True, "Detailed country label."),
        _column("category", ColumnDType.STRING, True, "Source category label."),
        _column("sector", ColumnDType.STRING, False, "Sector label."),
    )
    return TableSchemaContract(
        name="agent_identity",
        family=SchemaFamily.AGENT_IDENTITY,
        columns=columns,
        primary_keys=("country_sector",),
        description="Country-sector identity contract for ABM v5 agents.",
    )


def _state_dtype(name: str) -> ColumnDType:
    string_columns = {
        "country_sector",
        "country",
        "country_detail",
        "category",
        "sector",
        "regime_membership",
    }
    object_columns = {
        "supplier_weights",
        "input_requirements",
        "emissions_decomposition_terms",
        "phase_space_position",
    }
    if name in string_columns:
        return ColumnDType.STRING
    if name == "year":
        return ColumnDType.INTEGER
    if name == "supplier_opportunity_set":
        return ColumnDType.LIST
    if name in object_columns:
        return ColumnDType.OBJECT
    return ColumnDType.FLOAT


def _state_validation_layers(
    name: str,
    state_layer: AgentStateLayer,
) -> tuple[ValidationLayer, ...]:
    if state_layer is AgentStateLayer.IDENTITY:
        return (ValidationLayer.STRUCTURAL_VALIDITY,)
    if name in {"output", "emissions", "emissions_intensity"}:
        return (ValidationLayer.ACCOUNTING_VALIDITY, ValidationLayer.STRUCTURAL_VALIDITY)
    if name in {"production_feasibility", "supplier_weights", "input_requirements"}:
        return (ValidationLayer.ACCOUNTING_VALIDITY, ValidationLayer.MECHANISM_VALIDITY)
    if name in {"phase_space_position", "regime_membership", "regime_probability"}:
        return (
            ValidationLayer.STRUCTURAL_VALIDITY,
            ValidationLayer.MECHANISM_VALIDITY,
            ValidationLayer.HISTORICAL_PLAUSIBILITY,
        )
    return (ValidationLayer.STRUCTURAL_VALIDITY, ValidationLayer.MECHANISM_VALIDITY)


def build_agent_state_schema() -> TableSchemaContract:
    """Build the full country-sector state schema from ontology specs."""
    columns = tuple(
        _column(
            name=spec.name,
            dtype=_state_dtype(spec.name),
            nullable=spec.nullable,
            description=spec.description,
            source_status=spec.source_status,
            feedback_status=spec.feedback_status,
            complexity_level=spec.complexity_level,
            validation_layers=_state_validation_layers(spec.name, spec.layer),
            expected_range=spec.expected_range,
        )
        for spec in get_core_state_variable_specs()
    )
    return TableSchemaContract(
        name="agent_state",
        family=SchemaFamily.AGENT_STATE,
        columns=columns,
        primary_keys=("country_sector", "year"),
        description="Country-sector yearly state schema derived from the ABM v5 ontology.",
        unique_keys=(("country_sector", "year"),),
        sorted_by=("country_sector", "year"),
    )


def build_edge_state_schema() -> TableSchemaContract:
    """Build the supplier-buyer edge state schema contract."""
    columns = (
        _column("year", ColumnDType.INTEGER, False, "Historical or simulated edge year."),
        _column("supplier_country_sector", ColumnDType.STRING, False, "Supplier country-sector key."),
        _column("buyer_country_sector", ColumnDType.STRING, False, "Buyer country-sector key."),
        _column("transaction_value", ColumnDType.FLOAT, False, "Observed or simulated edge transaction value.", expected_range="non-negative"),
        _column("technical_coefficient", ColumnDType.FLOAT, True, "Input-output technical coefficient.", expected_range="non-negative"),
        _column("supplier_weight", ColumnDType.FLOAT, True, "Normalized supplier weight.", expected_range="0 to 1"),
        _column("embodied_emissions_flow", ColumnDType.FLOAT, True, "Emissions embodied in the supplier-buyer flow.", expected_range="non-negative"),
        _column("supplier_green_score", ColumnDType.FLOAT, True, "Green score associated with the supplier side of the edge."),
        _column("substitution_friction", ColumnDType.FLOAT, True, "Friction limiting substitution for this edge."),
        _column("compatibility_score", ColumnDType.FLOAT, True, "Compatibility score for supplier adaptation.", expected_range="0 to 1"),
    )
    return TableSchemaContract(
        name="edge_state",
        family=SchemaFamily.EDGE_STATE,
        columns=columns,
        primary_keys=("year", "supplier_country_sector", "buyer_country_sector"),
        description="Supplier-buyer edge schema for ABM v5 production-network metadata.",
        unique_keys=(("year", "supplier_country_sector", "buyer_country_sector"),),
        sorted_by=("year", "supplier_country_sector", "buyer_country_sector"),
    )


def _state_column(name: str, nullable: bool = True) -> ColumnContract:
    agent_state = build_agent_state_schema()
    column = agent_state.get_column(name)
    return ColumnContract(
        name=column.name,
        dtype=column.dtype,
        nullable=nullable,
        description=column.description,
        source_status=column.source_status,
        feedback_status=column.feedback_status,
        complexity_level=column.complexity_level,
        validation_layers=column.validation_layers,
        expected_range=column.expected_range,
        semantic_unit=column.semantic_unit,
        required=column.required,
    )


def build_phase_space_schema() -> TableSchemaContract:
    """Build the phase-space schema contract."""
    columns = (
        _state_column("country_sector", nullable=False),
        _state_column("year", nullable=False),
        _column("emissions_intensity_gap", ColumnDType.FLOAT, True, "Gap between node emissions intensity and a transition benchmark."),
        _state_column("green_capability"),
        _state_column("general_capability"),
        _state_column("network_green_exposure"),
        _state_column("brown_centrality"),
        _state_column("energy_dependence"),
        _state_column("capital_stock_inertia"),
        _state_column("policy_exposure"),
        _state_column("supplier_lock_in"),
        _state_column("directed_green_precedence"),
        _state_column("ecosystem_proximity"),
        _state_column("phase_space_position"),
    )
    return TableSchemaContract(
        name="phase_space",
        family=SchemaFamily.PHASE_SPACE,
        columns=columns,
        primary_keys=("country_sector", "year"),
        description="Phase-space schema for transition-position metadata.",
        unique_keys=(("country_sector", "year"),),
        sorted_by=("country_sector", "year"),
    )


def build_regime_schema() -> TableSchemaContract:
    """Build the regime-discovery output schema contract."""
    columns = (
        _state_column("country_sector", nullable=False),
        _state_column("year", nullable=False),
        _state_column("regime_membership"),
        _state_column("regime_probability"),
        _column("regime_confidence", ColumnDType.FLOAT, True, "Confidence score for diagnostic regime assignment.", expected_range="0 to 1"),
        _column("previous_regime_membership", ColumnDType.STRING, True, "Previous year's diagnostic regime assignment."),
        _column("regime_switch_flag", ColumnDType.BOOLEAN, True, "Flag indicating a diagnostic regime switch."),
        _column("threshold_rule_id", ColumnDType.STRING, True, "Identifier for the threshold rule used in regime diagnostics."),
    )
    return TableSchemaContract(
        name="regime",
        family=SchemaFamily.REGIME,
        columns=columns,
        primary_keys=("country_sector", "year"),
        description="Regime metadata schema for discovered transition regimes.",
        unique_keys=(("country_sector", "year"),),
        sorted_by=("country_sector", "year"),
    )


def build_simulation_output_schema() -> TableSchemaContract:
    """Build the future simulation output schema contract."""
    columns = (
        _column("scenario_id", ColumnDType.STRING, False, "Scenario identifier."),
        _column("run_id", ColumnDType.STRING, False, "Simulation run identifier."),
        _state_column("country_sector", nullable=False),
        _state_column("year", nullable=False),
        _state_column("output"),
        _state_column("realized_output"),
        _state_column("emissions"),
        _state_column("emissions_intensity"),
        _state_column("production_feasibility"),
        _state_column("general_capability"),
        _state_column("green_capability"),
        _state_column("network_green_exposure"),
        _state_column("supplier_lock_in"),
        _state_column("regime_membership"),
        _state_column("regime_probability"),
    )
    return TableSchemaContract(
        name="simulation_output",
        family=SchemaFamily.SIMULATION_OUTPUT,
        columns=columns,
        primary_keys=("scenario_id", "run_id", "country_sector", "year"),
        description="Metadata contract for future ABM v5 simulation outputs.",
        unique_keys=(("scenario_id", "run_id", "country_sector", "year"),),
        sorted_by=("scenario_id", "run_id", "country_sector", "year"),
    )


def build_validation_report_schema() -> TableSchemaContract:
    """Build the validation report schema contract."""
    columns = (
        _column("validation_id", ColumnDType.STRING, False, "Validation run identifier."),
        _column("validation_layer", ColumnDType.STRING, False, "Validation layer being checked."),
        _column("table_name", ColumnDType.STRING, False, "Schema or table name under validation."),
        _column("check_name", ColumnDType.STRING, False, "Name of the validation check."),
        _column("passed", ColumnDType.BOOLEAN, False, "Whether the check passed."),
        _column("severity", ColumnDType.STRING, False, "Severity assigned to the validation result."),
        _column("message", ColumnDType.STRING, False, "Human-readable validation message."),
        _column("n_failed", ColumnDType.INTEGER, True, "Number of failed rows or entities.", expected_range="non-negative"),
        _column("n_checked", ColumnDType.INTEGER, True, "Number of checked rows or entities.", expected_range="non-negative"),
    )
    return TableSchemaContract(
        name="validation_report",
        family=SchemaFamily.VALIDATION_REPORT,
        columns=columns,
        primary_keys=("validation_id", "check_name"),
        description="Schema for later metadata validation reports.",
        unique_keys=(("validation_id", "check_name"),),
        sorted_by=("validation_id", "check_name"),
    )


@dataclass(frozen=True)
class ABMV5SchemaRegistry:
    """Registry of validated ABM v5 schema contracts."""

    schemas: tuple[TableSchemaContract, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        """Validate all schemas and registry-level uniqueness."""
        if not self.schemas:
            raise ValueError("schemas must not be empty.")
        for schema in self.schemas:
            schema.validate()
        schema_names = self.schema_names()
        if len(schema_names) != len(set(schema_names)):
            raise ValueError("schema names must be unique.")
        schema_families = tuple(schema.family for schema in self.schemas)
        if len(schema_families) != len(set(schema_families)):
            raise ValueError("schema families must be unique.")

    def schema_names(self) -> tuple[str, ...]:
        """Return schema names in registry order."""
        return tuple(schema.name for schema in self.schemas)

    def get_schema(self, name: str) -> TableSchemaContract:
        """Return a schema by name."""
        for schema in self.schemas:
            if schema.name == name:
                return schema
        raise KeyError(f"Unknown schema: {name}")

    def get_schema_by_family(self, family: SchemaFamily | str) -> TableSchemaContract:
        """Return a schema by family enum or value."""
        schema_family = SchemaFamily(family)
        for schema in self.schemas:
            if schema.family is schema_family:
                return schema
        raise KeyError(f"Unknown schema family: {family}")


def build_default_schema_registry() -> ABMV5SchemaRegistry:
    """Build and validate the default Phase 1.4 schema registry."""
    registry = ABMV5SchemaRegistry(
        schemas=(
            build_agent_identity_schema(),
            build_agent_state_schema(),
            build_edge_state_schema(),
            build_phase_space_schema(),
            build_regime_schema(),
            build_simulation_output_schema(),
            build_validation_report_schema(),
        )
    )
    registry.validate()
    return registry
