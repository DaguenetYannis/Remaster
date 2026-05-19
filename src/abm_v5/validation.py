from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.abm_v5.config import FeedbackStatus, ValidationLayer
from src.abm_v5.ontology import (
    ABMV5OntologyRegistry,
    MechanismName,
    build_default_ontology_registry,
)
from src.abm_v5.schema import (
    ABMV5SchemaRegistry,
    SchemaFamily,
    build_default_schema_registry,
)


class ValidationSeverity(str, Enum):
    """Severity levels for ABM v5 metadata validation results."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationStatus(str, Enum):
    """Status values for ABM v5 validation checks."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class ValidationPrinciple:
    """Layered validation principle for ABM v5 model development."""

    layer: ValidationLayer
    title: str
    description: str
    required_before_scenarios: bool
    examples: tuple[str, ...] = ()

    def validate(self) -> None:
        """Validate principle metadata."""
        if not isinstance(self.layer, ValidationLayer):
            raise ValueError("layer must be ValidationLayer.")
        if not self.title:
            raise ValueError("title cannot be empty.")
        if not self.description:
            raise ValueError("description cannot be empty.")
        if not isinstance(self.required_before_scenarios, bool):
            raise ValueError("required_before_scenarios must be bool.")


def get_validation_principles() -> tuple[ValidationPrinciple, ...]:
    """Return the layered ABM v5 validation principles."""
    return (
        ValidationPrinciple(
            layer=ValidationLayer.ACCOUNTING_VALIDITY,
            title="Accounting validity",
            description=(
                "Output, emissions, supplier weights, input requirements, and "
                "identities must not appear or disappear without explanation."
            ),
            required_before_scenarios=True,
        ),
        ValidationPrinciple(
            layer=ValidationLayer.STRUCTURAL_VALIDITY,
            title="Structural validity",
            description=(
                "Country-sector structure, input-output links, supplier-buyer "
                "relations, and capability mappings must be preserved."
            ),
            required_before_scenarios=True,
        ),
        ValidationPrinciple(
            layer=ValidationLayer.MECHANISM_VALIDITY,
            title="Mechanism validity",
            description=(
                "Every behavioural or transition rule must have a theoretical "
                "interpretation and plausible local effect."
            ),
            required_before_scenarios=True,
        ),
        ValidationPrinciple(
            layer=ValidationLayer.HISTORICAL_PLAUSIBILITY,
            title="Historical plausibility",
            description=(
                "Simulated trajectories should reproduce broad historical "
                "patterns without arbitrary tuning."
            ),
            required_before_scenarios=True,
        ),
        ValidationPrinciple(
            layer=ValidationLayer.ABLATION_VALIDITY,
            title="Ablation validity",
            description=(
                "Added mechanisms should improve interpretation or diagnostics "
                "relative to simpler model variants."
            ),
            required_before_scenarios=True,
        ),
        ValidationPrinciple(
            layer=ValidationLayer.SCENARIO_CREDIBILITY,
            title="Scenario credibility",
            description=(
                "Counterfactual results are only interpreted after the previous "
                "validation layers pass."
            ),
            required_before_scenarios=True,
        ),
    )


@dataclass(frozen=True)
class ValidationResult:
    """Result from one lightweight metadata validation check."""

    check_name: str
    status: ValidationStatus
    severity: ValidationSeverity
    message: str
    layer: ValidationLayer
    n_failed: int = 0
    n_checked: int = 0

    def passed(self) -> bool:
        """Return True when the validation check passed."""
        return self.status is ValidationStatus.PASSED

    def validate(self) -> None:
        """Validate the result metadata."""
        if not self.check_name:
            raise ValueError("check_name cannot be empty.")
        if not self.message:
            raise ValueError("message cannot be empty.")
        if not isinstance(self.status, ValidationStatus):
            raise ValueError("status must be ValidationStatus.")
        if not isinstance(self.severity, ValidationSeverity):
            raise ValueError("severity must be ValidationSeverity.")
        if not isinstance(self.layer, ValidationLayer):
            raise ValueError("layer must be ValidationLayer.")
        if not isinstance(self.n_failed, int) or self.n_failed < 0:
            raise ValueError("n_failed must be a non-negative integer.")
        if not isinstance(self.n_checked, int) or self.n_checked < 0:
            raise ValueError("n_checked must be a non-negative integer.")
        if self.status is ValidationStatus.PASSED and self.n_failed != 0:
            raise ValueError("passed validation results must have n_failed equal to 0.")


def _result(
    check_name: str,
    status: ValidationStatus,
    severity: ValidationSeverity,
    message: str,
    layer: ValidationLayer,
    n_failed: int = 0,
    n_checked: int = 0,
) -> ValidationResult:
    validation_result = ValidationResult(
        check_name=check_name,
        status=status,
        severity=severity,
        message=message,
        layer=layer,
        n_failed=n_failed,
        n_checked=n_checked,
    )
    validation_result.validate()
    return validation_result


def _passed(
    check_name: str,
    message: str,
    layer: ValidationLayer,
    n_checked: int = 0,
) -> ValidationResult:
    return _result(
        check_name=check_name,
        status=ValidationStatus.PASSED,
        severity=ValidationSeverity.INFO,
        message=message,
        layer=layer,
        n_checked=n_checked,
    )


def _failed(
    check_name: str,
    message: str,
    layer: ValidationLayer,
    n_failed: int,
    n_checked: int,
    severity: ValidationSeverity = ValidationSeverity.ERROR,
) -> ValidationResult:
    return _result(
        check_name=check_name,
        status=ValidationStatus.FAILED,
        severity=severity,
        message=message,
        layer=layer,
        n_failed=n_failed,
        n_checked=n_checked,
    )


def validate_ontology_registry_metadata(
    registry: ABMV5OntologyRegistry,
) -> tuple[ValidationResult, ...]:
    """Validate ontology registry metadata without loading empirical data."""
    results: list[ValidationResult] = []

    try:
        registry.validate()
    except ValueError as error:
        results.append(
            _failed(
                "ontology_registry_validates",
                f"Ontology registry validation failed: {error}",
                ValidationLayer.STRUCTURAL_VALIDITY,
                n_failed=1,
                n_checked=1,
                severity=ValidationSeverity.CRITICAL,
            )
        )
    else:
        results.append(
            _passed(
                "ontology_registry_validates",
                "Ontology registry metadata validates.",
                ValidationLayer.STRUCTURAL_VALIDITY,
                n_checked=1,
            )
        )

    state_names = set(registry.state_variable_names())
    missing_inputs = sorted(
        {
            variable
            for mechanism in registry.mechanisms
            for variable in mechanism.input_variables
            if variable not in state_names
        }
    )
    results.append(
        _failed(
            "mechanism_inputs_exist",
            f"Mechanism input variables missing from state ontology: {missing_inputs}",
            ValidationLayer.MECHANISM_VALIDITY,
            n_failed=len(missing_inputs),
            n_checked=sum(len(mechanism.input_variables) for mechanism in registry.mechanisms),
        )
        if missing_inputs
        else _passed(
            "mechanism_inputs_exist",
            "Every mechanism input variable exists in the state ontology.",
            ValidationLayer.MECHANISM_VALIDITY,
            n_checked=sum(len(mechanism.input_variables) for mechanism in registry.mechanisms),
        )
    )

    missing_targets = sorted(
        {
            variable
            for mechanism in registry.mechanisms
            for variable in mechanism.update_targets
            if variable not in state_names
        }
    )
    results.append(
        _failed(
            "mechanism_targets_exist",
            f"Mechanism update targets missing from state ontology: {missing_targets}",
            ValidationLayer.MECHANISM_VALIDITY,
            n_failed=len(missing_targets),
            n_checked=sum(len(mechanism.update_targets) for mechanism in registry.mechanisms),
        )
        if missing_targets
        else _passed(
            "mechanism_targets_exist",
            "Every mechanism update target exists in the state ontology.",
            ValidationLayer.MECHANISM_VALIDITY,
            n_checked=sum(len(mechanism.update_targets) for mechanism in registry.mechanisms),
        )
    )

    exclusive_roles = [role.role.value for role in registry.functional_roles if role.mutually_exclusive]
    results.append(
        _failed(
            "functional_roles_not_mutually_exclusive",
            f"Functional roles marked mutually exclusive: {exclusive_roles}",
            ValidationLayer.STRUCTURAL_VALIDITY,
            n_failed=len(exclusive_roles),
            n_checked=len(registry.functional_roles),
        )
        if exclusive_roles
        else _passed(
            "functional_roles_not_mutually_exclusive",
            "No functional role is mutually exclusive.",
            ValidationLayer.STRUCTURAL_VALIDITY,
            n_checked=len(registry.functional_roles),
        )
    )

    static_roles = [role.role.value for role in registry.functional_roles if not role.recompute_each_year]
    results.append(
        _failed(
            "functional_roles_recomputed_yearly",
            f"Functional roles not recomputed yearly: {static_roles}",
            ValidationLayer.STRUCTURAL_VALIDITY,
            n_failed=len(static_roles),
            n_checked=len(registry.functional_roles),
        )
        if static_roles
        else _passed(
            "functional_roles_recomputed_yearly",
            "All functional roles are recomputed yearly.",
            ValidationLayer.STRUCTURAL_VALIDITY,
            n_checked=len(registry.functional_roles),
        )
    )

    fixed_class_signals = [
        role.role.value for role in registry.functional_roles if role.__class__.__name__.endswith("Agent")
    ]
    results.append(
        _failed(
            "no_fixed_agent_class_behaviour",
            f"Fixed agent-class signals found in role metadata: {fixed_class_signals}",
            ValidationLayer.STRUCTURAL_VALIDITY,
            n_failed=len(fixed_class_signals),
            n_checked=len(registry.functional_roles),
        )
        if fixed_class_signals
        else _passed(
            "no_fixed_agent_class_behaviour",
            "Ontology stores functional roles as metadata, not fixed agent classes.",
            ValidationLayer.STRUCTURAL_VALIDITY,
            n_checked=len(registry.functional_roles),
        )
    )

    return tuple(results)


def validate_schema_registry_metadata(
    registry: ABMV5SchemaRegistry,
) -> tuple[ValidationResult, ...]:
    """Validate schema registry metadata without validating any DataFrame."""
    results: list[ValidationResult] = []

    try:
        registry.validate()
    except ValueError as error:
        results.append(
            _failed(
                "schema_registry_validates",
                f"Schema registry validation failed: {error}",
                ValidationLayer.STRUCTURAL_VALIDITY,
                n_failed=1,
                n_checked=1,
                severity=ValidationSeverity.CRITICAL,
            )
        )
    else:
        results.append(
            _passed(
                "schema_registry_validates",
                "Schema registry metadata validates.",
                ValidationLayer.STRUCTURAL_VALIDITY,
                n_checked=1,
            )
        )

    schemas_without_keys = [schema.name for schema in registry.schemas if not schema.primary_keys]
    results.append(
        _failed(
            "schemas_have_primary_keys",
            f"Schemas without primary keys: {schemas_without_keys}",
            ValidationLayer.STRUCTURAL_VALIDITY,
            n_failed=len(schemas_without_keys),
            n_checked=len(registry.schemas),
        )
        if schemas_without_keys
        else _passed(
            "schemas_have_primary_keys",
            "Every schema has primary keys.",
            ValidationLayer.STRUCTURAL_VALIDITY,
            n_checked=len(registry.schemas),
        )
    )

    columns_without_validation = [
        f"{schema.name}.{column.name}"
        for schema in registry.schemas
        for column in schema.columns
        if not column.validation_layers
    ]
    results.append(
        _failed(
            "columns_have_validation_layers",
            f"Columns without validation layers: {columns_without_validation}",
            ValidationLayer.STRUCTURAL_VALIDITY,
            n_failed=len(columns_without_validation),
            n_checked=sum(len(schema.columns) for schema in registry.schemas),
        )
        if columns_without_validation
        else _passed(
            "columns_have_validation_layers",
            "Every schema column has at least one validation layer.",
            ValidationLayer.STRUCTURAL_VALIDITY,
            n_checked=sum(len(schema.columns) for schema in registry.schemas),
        )
    )

    agent_state = registry.get_schema_by_family(SchemaFamily.AGENT_STATE)
    agent_state_required = {"country_sector", "year"}
    missing_agent_state = sorted(agent_state_required - set(agent_state.column_names()))
    results.append(
        _failed(
            "agent_state_has_country_sector_year",
            f"agent_state is missing required keys: {missing_agent_state}",
            ValidationLayer.ACCOUNTING_VALIDITY,
            n_failed=len(missing_agent_state),
            n_checked=len(agent_state_required),
        )
        if missing_agent_state
        else _passed(
            "agent_state_has_country_sector_year",
            "agent_state has country_sector and year.",
            ValidationLayer.ACCOUNTING_VALIDITY,
            n_checked=len(agent_state_required),
        )
    )

    simulation_output = registry.get_schema_by_family(SchemaFamily.SIMULATION_OUTPUT)
    simulation_required = {"scenario_id", "run_id", "country_sector", "year"}
    missing_simulation = sorted(simulation_required - set(simulation_output.column_names()))
    results.append(
        _failed(
            "simulation_output_has_required_keys",
            f"simulation_output is missing required keys: {missing_simulation}",
            ValidationLayer.SCENARIO_CREDIBILITY,
            n_failed=len(missing_simulation),
            n_checked=len(simulation_required),
        )
        if missing_simulation
        else _passed(
            "simulation_output_has_required_keys",
            "simulation_output has scenario_id, run_id, country_sector, and year.",
            ValidationLayer.SCENARIO_CREDIBILITY,
            n_checked=len(simulation_required),
        )
    )

    validation_report = registry.get_schema_by_family(SchemaFamily.VALIDATION_REPORT)
    validation_columns = set(validation_report.column_names())
    has_validation_message = "message" in validation_columns and (
        "passed" in validation_columns or "status" in validation_columns
    )
    results.append(
        _passed(
            "validation_report_has_result_and_message",
            "validation_report has passed/message or status/message fields.",
            ValidationLayer.STRUCTURAL_VALIDITY,
            n_checked=2,
        )
        if has_validation_message
        else _failed(
            "validation_report_has_result_and_message",
            "validation_report must include passed/message or status/message fields.",
            ValidationLayer.STRUCTURAL_VALIDITY,
            n_failed=1,
            n_checked=2,
        )
    )

    return tuple(results)


EDGE_ONLY_VARIABLES = {
    "supplier_country_sector",
    "buyer_country_sector",
    "transaction_value",
    "technical_coefficient",
    "supplier_weight",
    "embodied_emissions_flow",
    "supplier_green_score",
    "compatibility_score",
}


def validate_theory_to_schema_alignment(
    ontology_registry: ABMV5OntologyRegistry,
    schema_registry: ABMV5SchemaRegistry,
) -> tuple[ValidationResult, ...]:
    """Validate ontology-to-schema alignment for Phase 1 metadata."""
    results: list[ValidationResult] = []
    agent_state = schema_registry.get_schema_by_family(SchemaFamily.AGENT_STATE)
    agent_state_columns = set(agent_state.column_names())
    state_variable_names = set(ontology_registry.state_variable_names())

    missing_state_variables = sorted(state_variable_names - agent_state_columns)
    results.append(
        _failed(
            "ontology_state_variables_in_agent_state_schema",
            f"Ontology state variables missing from agent_state schema: {missing_state_variables}",
            ValidationLayer.STRUCTURAL_VALIDITY,
            n_failed=len(missing_state_variables),
            n_checked=len(state_variable_names),
        )
        if missing_state_variables
        else _passed(
            "ontology_state_variables_in_agent_state_schema",
            "Every ontology state variable appears in the agent_state schema.",
            ValidationLayer.STRUCTURAL_VALIDITY,
            n_checked=len(state_variable_names),
        )
    )

    allowed_schema_variables = agent_state_columns | EDGE_ONLY_VARIABLES
    missing_inputs = sorted(
        {
            variable
            for mechanism in ontology_registry.mechanisms
            for variable in mechanism.input_variables
            if variable not in allowed_schema_variables
        }
    )
    results.append(
        _failed(
            "mechanism_inputs_in_agent_state_schema",
            f"Mechanism inputs missing from agent_state schema: {missing_inputs}",
            ValidationLayer.MECHANISM_VALIDITY,
            n_failed=len(missing_inputs),
            n_checked=sum(len(mechanism.input_variables) for mechanism in ontology_registry.mechanisms),
        )
        if missing_inputs
        else _passed(
            "mechanism_inputs_in_agent_state_schema",
            "Every mechanism input appears in agent_state schema or edge-only exceptions.",
            ValidationLayer.MECHANISM_VALIDITY,
            n_checked=sum(len(mechanism.input_variables) for mechanism in ontology_registry.mechanisms),
        )
    )

    missing_targets = sorted(
        {
            variable
            for mechanism in ontology_registry.mechanisms
            for variable in mechanism.update_targets
            if variable not in allowed_schema_variables
        }
    )
    results.append(
        _failed(
            "mechanism_targets_in_agent_state_schema",
            f"Mechanism update targets missing from agent_state schema: {missing_targets}",
            ValidationLayer.MECHANISM_VALIDITY,
            n_failed=len(missing_targets),
            n_checked=sum(len(mechanism.update_targets) for mechanism in ontology_registry.mechanisms),
        )
        if missing_targets
        else _passed(
            "mechanism_targets_in_agent_state_schema",
            "Every mechanism update target appears in agent_state schema or edge-only exceptions.",
            ValidationLayer.MECHANISM_VALIDITY,
            n_checked=sum(len(mechanism.update_targets) for mechanism in ontology_registry.mechanisms),
        )
    )

    mechanisms_without_validation = [
        mechanism.name.value for mechanism in ontology_registry.mechanisms if not mechanism.validation_layers
    ]
    results.append(
        _failed(
            "mechanisms_have_validation_layers",
            f"Mechanisms without validation layers: {mechanisms_without_validation}",
            ValidationLayer.MECHANISM_VALIDITY,
            n_failed=len(mechanisms_without_validation),
            n_checked=len(ontology_registry.mechanisms),
        )
        if mechanisms_without_validation
        else _passed(
            "mechanisms_have_validation_layers",
            "Every mechanism has at least one validation layer.",
            ValidationLayer.MECHANISM_VALIDITY,
            n_checked=len(ontology_registry.mechanisms),
        )
    )

    policy_mechanism = ontology_registry.get_mechanism(MechanismName.POLICY_REGIME_EXPOSURE)
    policy_is_scenario_driven = policy_mechanism.feedback_status is FeedbackStatus.SCENARIO_DRIVEN
    results.append(
        _passed(
            "policy_mechanism_is_scenario_driven",
            "Policy mechanism is scenario-driven, not endogenous in Phase 1.",
            ValidationLayer.SCENARIO_CREDIBILITY,
            n_checked=1,
        )
        if policy_is_scenario_driven
        else _failed(
            "policy_mechanism_is_scenario_driven",
            "Policy mechanism must be scenario-driven in Phase 1.",
            ValidationLayer.SCENARIO_CREDIBILITY,
            n_failed=1,
            n_checked=1,
        )
    )

    return tuple(results)


def run_phase1_metadata_validation() -> tuple[ValidationResult, ...]:
    """Run all Phase 1 metadata validators against default registries."""
    ontology_registry = build_default_ontology_registry()
    schema_registry = build_default_schema_registry()
    return (
        *validate_ontology_registry_metadata(ontology_registry),
        *validate_schema_registry_metadata(schema_registry),
        *validate_theory_to_schema_alignment(ontology_registry, schema_registry),
    )


def all_passed(results: tuple[ValidationResult, ...]) -> bool:
    """Return True if every validation result passed."""
    return all(result.passed() for result in results)


def failed_results(results: tuple[ValidationResult, ...]) -> tuple[ValidationResult, ...]:
    """Return only failed validation results."""
    return tuple(result for result in results if not result.passed())
