from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.abm_v5.config import (
    ComplexityLayer,
    ComplexityLevel,
    FeedbackStatus,
    SourceStatus,
    ValidationLayer,
)


AGENT_IDENTITY_DESCRIPTION = (
    "A country-sector agent is a meso-level productive position embedded in a "
    "global input-output system. It is not a firm, not a country, not a sector "
    "alone, and not a literal conscious decision-maker. It represents the "
    "adaptive behaviour of a productive position under production, capability, "
    "energy, policy, supplier, and emissions constraints."
)


@dataclass(frozen=True)
class AgentIdentityOntology:
    """Identity ontology for country-sector productive agents."""

    primary_key: str = "country_sector"
    auxiliary_keys: tuple[str, ...] = ("country", "country_detail", "category", "sector")
    stable_over_time: bool = True
    description: str = AGENT_IDENTITY_DESCRIPTION

    def required_columns(self) -> tuple[str, ...]:
        """Return identity columns required by the country-sector ontology."""
        return (self.primary_key, *self.auxiliary_keys)

    def validate(self) -> None:
        """Validate the country-sector identity contract."""
        if self.primary_key != "country_sector":
            raise ValueError('primary_key must equal "country_sector".')
        if "country" not in self.auxiliary_keys:
            raise ValueError('auxiliary_keys must include "country".')
        if "sector" not in self.auxiliary_keys:
            raise ValueError('auxiliary_keys must include "sector".')
        if not self.stable_over_time:
            raise ValueError("stable_over_time must be True.")
        if not self.description:
            raise ValueError("description must not be empty.")


class AgentStateLayer(str, Enum):
    """Conceptual layers of country-sector agent state."""

    IDENTITY = "identity"
    PRODUCTIVE_STATE = "productive_state"
    ECOLOGICAL_STATE = "ecological_state"
    CAPABILITY_ECOSYSTEM_STATE = "capability_ecosystem_state"
    CONSTRAINT_BEHAVIOUR_STATE = "constraint_behaviour_state"


@dataclass(frozen=True)
class StateVariableSpec:
    """Metadata for one ABM v5 country-sector state variable."""

    name: str
    layer: AgentStateLayer
    complexity_level: ComplexityLevel
    feedback_status: FeedbackStatus
    source_status: SourceStatus
    nullable: bool
    description: str
    expected_range: str | None = None
    required_from_phase: str = "phase_1"

    def validate(self) -> None:
        """Validate state variable metadata."""
        if not self.name:
            raise ValueError("name cannot be empty.")
        if not self.description:
            raise ValueError(f"description cannot be empty for {self.name}.")
        if not isinstance(self.layer, AgentStateLayer):
            raise ValueError(f"layer must be AgentStateLayer for {self.name}.")
        if not isinstance(self.complexity_level, ComplexityLevel):
            raise ValueError(f"complexity_level must be ComplexityLevel for {self.name}.")
        if not isinstance(self.feedback_status, FeedbackStatus):
            raise ValueError(f"feedback_status must be FeedbackStatus for {self.name}.")
        if not isinstance(self.source_status, SourceStatus):
            raise ValueError(f"source_status must be SourceStatus for {self.name}.")
        if not isinstance(self.nullable, bool):
            raise ValueError(f"nullable must be bool for {self.name}.")


def _state_variable(
    name: str,
    layer: AgentStateLayer,
    complexity_level: ComplexityLevel,
    feedback_status: FeedbackStatus,
    source_status: SourceStatus,
    nullable: bool,
    description: str,
    expected_range: str | None = None,
) -> StateVariableSpec:
    return StateVariableSpec(
        name=name,
        layer=layer,
        complexity_level=complexity_level,
        feedback_status=feedback_status,
        source_status=source_status,
        nullable=nullable,
        description=description,
        expected_range=expected_range,
    )


def get_core_state_variable_specs() -> tuple[StateVariableSpec, ...]:
    """Return contract-defined ABM v5 state variable metadata."""
    return (
        _state_variable(
            "country_sector",
            AgentStateLayer.IDENTITY,
            ComplexityLevel.MESO,
            FeedbackStatus.OBSERVED,
            SourceStatus.DERIVED,
            False,
            "Stable identifier for the acting country-sector node.",
        ),
        _state_variable(
            "country",
            AgentStateLayer.IDENTITY,
            ComplexityLevel.MESO,
            FeedbackStatus.OBSERVED,
            SourceStatus.RAW_OBSERVED,
            False,
            "Country code or country label associated with the productive position.",
        ),
        _state_variable(
            "country_detail",
            AgentStateLayer.IDENTITY,
            ComplexityLevel.MESO,
            FeedbackStatus.OBSERVED,
            SourceStatus.RAW_OBSERVED,
            True,
            "Detailed country label retained for traceable country-sector interpretation.",
        ),
        _state_variable(
            "category",
            AgentStateLayer.IDENTITY,
            ComplexityLevel.MESO,
            FeedbackStatus.OBSERVED,
            SourceStatus.RAW_OBSERVED,
            True,
            "Source category label associated with the country-sector row or column.",
        ),
        _state_variable(
            "sector",
            AgentStateLayer.IDENTITY,
            ComplexityLevel.MESO,
            FeedbackStatus.OBSERVED,
            SourceStatus.RAW_OBSERVED,
            False,
            "Sector label for the productive position.",
        ),
        _state_variable(
            "year",
            AgentStateLayer.IDENTITY,
            ComplexityLevel.MACRO,
            FeedbackStatus.OBSERVED,
            SourceStatus.RAW_OBSERVED,
            False,
            "Historical year indexing the country-sector state.",
        ),
        _state_variable(
            "output",
            AgentStateLayer.PRODUCTIVE_STATE,
            ComplexityLevel.MESO,
            FeedbackStatus.OBSERVED,
            SourceStatus.DERIVED,
            False,
            "Observed or reconstructed production output for the country-sector node.",
            "non-negative",
        ),
        _state_variable(
            "desired_output",
            AgentStateLayer.PRODUCTIVE_STATE,
            ComplexityLevel.MESO,
            FeedbackStatus.DESIGN_TARGET,
            SourceStatus.PLACEHOLDER,
            True,
            "Future target for demand-driven output before production constraints bind.",
            "non-negative",
        ),
        _state_variable(
            "realized_output",
            AgentStateLayer.PRODUCTIVE_STATE,
            ComplexityLevel.MESO,
            FeedbackStatus.DESIGN_TARGET,
            SourceStatus.SIMULATED,
            True,
            "Future mechanism output after feasibility, supplier, and inertia constraints.",
            "non-negative",
        ),
        _state_variable(
            "final_demand",
            AgentStateLayer.PRODUCTIVE_STATE,
            ComplexityLevel.MACRO,
            FeedbackStatus.OBSERVED,
            SourceStatus.DERIVED,
            True,
            "Final demand exposure attached to the country-sector accounting state.",
            "non-negative",
        ),
        _state_variable(
            "input_requirements",
            AgentStateLayer.PRODUCTIVE_STATE,
            ComplexityLevel.CROSS_LEVEL,
            FeedbackStatus.DIAGNOSTIC,
            SourceStatus.DERIVED,
            True,
            "Input bundle required to support the node's desired production.",
        ),
        _state_variable(
            "production_feasibility",
            AgentStateLayer.PRODUCTIVE_STATE,
            ComplexityLevel.CROSS_LEVEL,
            FeedbackStatus.DESIGN_TARGET,
            SourceStatus.SIMULATED,
            True,
            "Future diagnostic of whether available inputs and capacity can support output.",
            "0 to 1 or non-negative feasibility score",
        ),
        _state_variable(
            "capacity_proxy",
            AgentStateLayer.PRODUCTIVE_STATE,
            ComplexityLevel.MESO,
            FeedbackStatus.DIAGNOSTIC,
            SourceStatus.PROXY,
            True,
            "Proxy for productive capacity available to the country-sector node.",
            "non-negative",
        ),
        _state_variable(
            "inventory_or_buffer_proxy",
            AgentStateLayer.PRODUCTIVE_STATE,
            ComplexityLevel.MESO,
            FeedbackStatus.DIAGNOSTIC,
            SourceStatus.PROXY,
            True,
            "Proxy for short-run buffers that can absorb supplier or demand shocks.",
            "non-negative",
        ),
        _state_variable(
            "emissions",
            AgentStateLayer.ECOLOGICAL_STATE,
            ComplexityLevel.MESO,
            FeedbackStatus.OBSERVED,
            SourceStatus.DERIVED,
            True,
            "Emissions attributed to the country-sector node.",
            "non-negative",
        ),
        _state_variable(
            "emissions_intensity",
            AgentStateLayer.ECOLOGICAL_STATE,
            ComplexityLevel.MESO,
            FeedbackStatus.OBSERVED,
            SourceStatus.DERIVED,
            True,
            "Emissions per unit of output for the country-sector node.",
            "non-negative",
        ),
        _state_variable(
            "local_greenness",
            AgentStateLayer.ECOLOGICAL_STATE,
            ComplexityLevel.MESO,
            FeedbackStatus.DIAGNOSTIC,
            SourceStatus.DERIVED,
            True,
            "Local green-ness score derived from the node's emissions intensity.",
        ),
        _state_variable(
            "network_green_exposure",
            AgentStateLayer.ECOLOGICAL_STATE,
            ComplexityLevel.CROSS_LEVEL,
            FeedbackStatus.DESIGN_TARGET,
            SourceStatus.DERIVED,
            True,
            "Exposure to greener upstream or downstream positions through the production network.",
        ),
        _state_variable(
            "brown_centrality",
            AgentStateLayer.ECOLOGICAL_STATE,
            ComplexityLevel.CROSS_LEVEL,
            FeedbackStatus.DIAGNOSTIC,
            SourceStatus.DERIVED,
            True,
            "Network centrality of emissions-intensive or brown production positions.",
        ),
        _state_variable(
            "emissions_decomposition_terms",
            AgentStateLayer.ECOLOGICAL_STATE,
            ComplexityLevel.CROSS_LEVEL,
            FeedbackStatus.DIAGNOSTIC,
            SourceStatus.DERIVED,
            True,
            "Interpretable accounting terms used to explain emissions changes.",
        ),
        _state_variable(
            "general_capability",
            AgentStateLayer.CAPABILITY_ECOSYSTEM_STATE,
            ComplexityLevel.MESO,
            FeedbackStatus.DESIGN_TARGET,
            SourceStatus.ESTIMATED,
            True,
            "General productive capability available to the country-sector node.",
        ),
        _state_variable(
            "green_capability",
            AgentStateLayer.CAPABILITY_ECOSYSTEM_STATE,
            ComplexityLevel.MESO,
            FeedbackStatus.DESIGN_TARGET,
            SourceStatus.ESTIMATED,
            True,
            "Green productive capability relevant to lower-emissions transition.",
        ),
        _state_variable(
            "capability_density",
            AgentStateLayer.CAPABILITY_ECOSYSTEM_STATE,
            ComplexityLevel.MESO,
            FeedbackStatus.DIAGNOSTIC,
            SourceStatus.DERIVED,
            True,
            "Density of nearby productive capabilities in the node's ecosystem.",
        ),
        _state_variable(
            "green_capability_density",
            AgentStateLayer.CAPABILITY_ECOSYSTEM_STATE,
            ComplexityLevel.MESO,
            FeedbackStatus.DIAGNOSTIC,
            SourceStatus.DERIVED,
            True,
            "Density of nearby green capabilities in the node's ecosystem.",
        ),
        _state_variable(
            "directed_green_precedence",
            AgentStateLayer.CAPABILITY_ECOSYSTEM_STATE,
            ComplexityLevel.CROSS_LEVEL,
            FeedbackStatus.DESIGN_TARGET,
            SourceStatus.DERIVED,
            True,
            "Directed measure of whether greener productive positions precede the node.",
        ),
        _state_variable(
            "ecosystem_proximity",
            AgentStateLayer.CAPABILITY_ECOSYSTEM_STATE,
            ComplexityLevel.MESO,
            FeedbackStatus.DIAGNOSTIC,
            SourceStatus.DERIVED,
            True,
            "Distance or proximity to related productive ecosystem positions.",
        ),
        _state_variable(
            "reachable_green_complexity",
            AgentStateLayer.CAPABILITY_ECOSYSTEM_STATE,
            ComplexityLevel.CROSS_LEVEL,
            FeedbackStatus.DESIGN_TARGET,
            SourceStatus.DERIVED,
            True,
            "Green complexity that the node could plausibly reach through ecosystem movement.",
        ),
        _state_variable(
            "transition_sector_score",
            AgentStateLayer.CAPABILITY_ECOSYSTEM_STATE,
            ComplexityLevel.CROSS_LEVEL,
            FeedbackStatus.DIAGNOSTIC,
            SourceStatus.DERIVED,
            True,
            "Score identifying sector positions relevant to ecological transition pathways.",
        ),
        _state_variable(
            "supplier_weights",
            AgentStateLayer.CONSTRAINT_BEHAVIOUR_STATE,
            ComplexityLevel.MESO,
            FeedbackStatus.DESIGN_TARGET,
            SourceStatus.SIMULATED,
            True,
            "Weighted supplier relationships available to the country-sector node.",
        ),
        _state_variable(
            "supplier_opportunity_set",
            AgentStateLayer.CONSTRAINT_BEHAVIOUR_STATE,
            ComplexityLevel.CROSS_LEVEL,
            FeedbackStatus.DESIGN_TARGET,
            SourceStatus.SIMULATED,
            True,
            "Candidate supplier set available for future bounded adaptation mechanisms.",
        ),
        _state_variable(
            "supplier_lock_in",
            AgentStateLayer.CONSTRAINT_BEHAVIOUR_STATE,
            ComplexityLevel.MESO,
            FeedbackStatus.DESIGN_TARGET,
            SourceStatus.SIMULATED,
            True,
            "Degree to which the node is constrained by existing supplier relationships.",
        ),
        _state_variable(
            "substitution_friction",
            AgentStateLayer.CONSTRAINT_BEHAVIOUR_STATE,
            ComplexityLevel.MESO,
            FeedbackStatus.DESIGN_TARGET,
            SourceStatus.PROXY,
            True,
            "Friction limiting substitution across suppliers or input bundles.",
        ),
        _state_variable(
            "energy_dependence",
            AgentStateLayer.CONSTRAINT_BEHAVIOUR_STATE,
            ComplexityLevel.MESO,
            FeedbackStatus.DESIGN_TARGET,
            SourceStatus.PROXY,
            True,
            "Dependence of the node on energy-intensive or fuel-specific inputs.",
        ),
        _state_variable(
            "fuel_structure_proxy",
            AgentStateLayer.CONSTRAINT_BEHAVIOUR_STATE,
            ComplexityLevel.MESO,
            FeedbackStatus.DIAGNOSTIC,
            SourceStatus.PROXY,
            True,
            "Proxy for the fuel structure associated with the node's production.",
        ),
        _state_variable(
            "capital_stock_inertia",
            AgentStateLayer.CONSTRAINT_BEHAVIOUR_STATE,
            ComplexityLevel.MESO,
            FeedbackStatus.DESIGN_TARGET,
            SourceStatus.PROXY,
            True,
            "Inertia from long-lived capital stock that slows transition.",
        ),
        _state_variable(
            "policy_exposure",
            AgentStateLayer.CONSTRAINT_BEHAVIOUR_STATE,
            ComplexityLevel.CROSS_LEVEL,
            FeedbackStatus.SCENARIO_DRIVEN,
            SourceStatus.PLACEHOLDER,
            True,
            "Exposure to policy regimes or policy shocks in later scenario phases.",
        ),
        _state_variable(
            "phase_space_position",
            AgentStateLayer.CONSTRAINT_BEHAVIOUR_STATE,
            ComplexityLevel.CROSS_LEVEL,
            FeedbackStatus.DIAGNOSTIC,
            SourceStatus.DERIVED,
            True,
            "Position of the node in the model's transition phase space.",
        ),
        _state_variable(
            "regime_membership",
            AgentStateLayer.CONSTRAINT_BEHAVIOUR_STATE,
            ComplexityLevel.CROSS_LEVEL,
            FeedbackStatus.DIAGNOSTIC,
            SourceStatus.DERIVED,
            True,
            "Diagnostic regime label discovered from phase-space structure.",
        ),
        _state_variable(
            "regime_probability",
            AgentStateLayer.CONSTRAINT_BEHAVIOUR_STATE,
            ComplexityLevel.CROSS_LEVEL,
            FeedbackStatus.DIAGNOSTIC,
            SourceStatus.DERIVED,
            True,
            "Probability or confidence associated with diagnostic regime membership.",
            "0 to 1",
        ),
    )


class FunctionalRole(str, Enum):
    """Derived diagnostic roles, not fixed agent classes."""

    ENERGY_SYSTEM_NODE = "energy_system_node"
    INFRASTRUCTURE_INTENSIVE_NODE = "infrastructure_intensive_node"
    HIGH_CENTRALITY_SUPPLIER_NODE = "high_centrality_supplier_node"
    HIGH_BROWN_CENTRALITY_NODE = "high_brown_centrality_node"
    GREEN_CAPABILITY_READY_NODE = "green_capability_ready_node"
    SUPPLIER_LOCKED_NODE = "supplier_locked_node"
    TRANSITION_FRONTIER_NODE = "transition_frontier_node"
    VULNERABLE_IMPORT_DEPENDENT_NODE = "vulnerable_import_dependent_node"
    HIGH_OUTPUT_SYSTEMIC_NODE = "high_output_systemic_node"
    HIGH_EMISSIONS_SYSTEMIC_NODE = "high_emissions_systemic_node"
    POLICY_EXPOSED_NODE = "policy_exposed_node"
    LOW_FEASIBILITY_NODE = "low_feasibility_node"


@dataclass(frozen=True)
class FunctionalRoleSpec:
    """Metadata for a derived, overlapping country-sector role label."""

    role: FunctionalRole
    description: str
    derived_from_variables: tuple[str, ...]
    recompute_each_year: bool = True
    mutually_exclusive: bool = False

    def validate(self) -> None:
        """Validate role metadata as yearly overlapping diagnostics."""
        if not isinstance(self.role, FunctionalRole):
            raise ValueError("role must be FunctionalRole.")
        if not self.description:
            raise ValueError(f"description cannot be empty for {self.role}.")
        if not self.derived_from_variables:
            raise ValueError(f"derived_from_variables cannot be empty for {self.role}.")
        if not self.recompute_each_year:
            raise ValueError("recompute_each_year must be True.")
        if self.mutually_exclusive:
            raise ValueError("mutually_exclusive must be False.")


def get_functional_role_specs() -> tuple[FunctionalRoleSpec, ...]:
    """Return functional role metadata without derivation logic."""
    return (
        FunctionalRoleSpec(
            FunctionalRole.ENERGY_SYSTEM_NODE,
            "Node whose productive position is structurally tied to energy systems.",
            ("sector", "energy_dependence", "fuel_structure_proxy"),
        ),
        FunctionalRoleSpec(
            FunctionalRole.INFRASTRUCTURE_INTENSIVE_NODE,
            "Node with high capital or infrastructure inertia in transition.",
            ("capital_stock_inertia", "capacity_proxy", "sector"),
        ),
        FunctionalRoleSpec(
            FunctionalRole.HIGH_CENTRALITY_SUPPLIER_NODE,
            "Node that occupies a central supplier position in production networks.",
            ("supplier_weights", "output", "input_requirements"),
        ),
        FunctionalRoleSpec(
            FunctionalRole.HIGH_BROWN_CENTRALITY_NODE,
            "Node with central brown-network exposure relevant to lock-in.",
            ("brown_centrality", "emissions_intensity", "network_green_exposure"),
        ),
        FunctionalRoleSpec(
            FunctionalRole.GREEN_CAPABILITY_READY_NODE,
            "Node with capability conditions that may support green transition.",
            ("green_capability", "green_capability_density", "reachable_green_complexity"),
        ),
        FunctionalRoleSpec(
            FunctionalRole.SUPPLIER_LOCKED_NODE,
            "Node whose supplier relationships suggest constrained adaptation.",
            ("supplier_lock_in", "supplier_weights", "substitution_friction"),
        ),
        FunctionalRoleSpec(
            FunctionalRole.TRANSITION_FRONTIER_NODE,
            "Node near a frontier of reachable green productive complexity.",
            ("directed_green_precedence", "reachable_green_complexity", "transition_sector_score"),
        ),
        FunctionalRoleSpec(
            FunctionalRole.VULNERABLE_IMPORT_DEPENDENT_NODE,
            "Node exposed to constrained or fragile supplier opportunity sets.",
            ("supplier_opportunity_set", "supplier_lock_in", "input_requirements"),
        ),
        FunctionalRoleSpec(
            FunctionalRole.HIGH_OUTPUT_SYSTEMIC_NODE,
            "Node whose output and network position make it systemically important.",
            ("output", "supplier_weights", "final_demand"),
        ),
        FunctionalRoleSpec(
            FunctionalRole.HIGH_EMISSIONS_SYSTEMIC_NODE,
            "Node whose emissions and network position make it transition-critical.",
            ("emissions", "emissions_intensity", "brown_centrality"),
        ),
        FunctionalRoleSpec(
            FunctionalRole.POLICY_EXPOSED_NODE,
            "Node exposed to policy regime constraints or incentives.",
            ("policy_exposure", "emissions_intensity", "green_capability"),
        ),
        FunctionalRoleSpec(
            FunctionalRole.LOW_FEASIBILITY_NODE,
            "Node with weak production feasibility under input and capacity constraints.",
            ("production_feasibility", "input_requirements", "capacity_proxy"),
        ),
    )


class TheorySource(str, Enum):
    """Theory families used to motivate mechanism metadata."""

    COMPLEXITY_ECONOMICS = "complexity_economics"
    ECONOMIC_COMPLEXITY = "economic_complexity"
    DIRECTED_PRODUCTIVE_ECOSYSTEMS = "directed_productive_ecosystems"
    INPUT_OUTPUT_ECONOMICS = "input_output_economics"
    SUPPLY_CHAIN_ABM = "supply_chain_abm"
    POLICY_TRANSITION_THEORY = "policy_transition_theory"
    REMASTER_CONTRACT = "remaster_contract"


class MechanismName(str, Enum):
    """Mechanism names defined as metadata targets for later implementation."""

    PRODUCTION_FEASIBILITY = "production_feasibility"
    SUPPLIER_SUBSTITUTION = "supplier_substitution"
    BOUNDED_SUPPLIER_SEARCH = "bounded_supplier_search"
    CAPABILITY_ACCUMULATION = "capability_accumulation"
    DIRECTED_ECOSYSTEM_MOVEMENT = "directed_ecosystem_movement"
    EMISSIONS_INTENSITY_TRANSITION = "emissions_intensity_transition"
    ENERGY_DEPENDENCE_CONSTRAINT = "energy_dependence_constraint"
    CAPITAL_STOCK_INERTIA = "capital_stock_inertia"
    POLICY_REGIME_EXPOSURE = "policy_regime_exposure"
    PHASE_SPACE_REGIME_SWITCHING = "phase_space_regime_switching"
    NETWORK_GREEN_EXPOSURE_FEEDBACK = "network_green_exposure_feedback"
    BROWN_LOCK_IN_FEEDBACK = "brown_lock_in_feedback"


@dataclass(frozen=True)
class MechanismSpec:
    """Metadata for a mechanism linking inputs, targets, theory, and validation."""

    name: MechanismName
    theory_sources: tuple[TheorySource, ...]
    complexity_layer: ComplexityLayer
    complexity_level: ComplexityLevel
    input_variables: tuple[str, ...]
    update_targets: tuple[str, ...]
    feedback_status: FeedbackStatus
    validation_layers: tuple[ValidationLayer, ...]
    description: str

    def validate(self) -> None:
        """Validate mechanism metadata without executing mechanism logic."""
        if not isinstance(self.name, MechanismName):
            raise ValueError("name must be MechanismName.")
        if not self.theory_sources:
            raise ValueError(f"theory_sources cannot be empty for {self.name}.")
        if any(not isinstance(source, TheorySource) for source in self.theory_sources):
            raise ValueError(f"theory_sources must contain TheorySource values for {self.name}.")
        if not isinstance(self.complexity_layer, ComplexityLayer):
            raise ValueError(f"complexity_layer must be ComplexityLayer for {self.name}.")
        if not isinstance(self.complexity_level, ComplexityLevel):
            raise ValueError(f"complexity_level must be ComplexityLevel for {self.name}.")
        if not self.input_variables:
            raise ValueError(f"input_variables cannot be empty for {self.name}.")
        if not self.update_targets:
            raise ValueError(f"update_targets cannot be empty for {self.name}.")
        if not isinstance(self.feedback_status, FeedbackStatus):
            raise ValueError(f"feedback_status must be FeedbackStatus for {self.name}.")
        if not self.validation_layers:
            raise ValueError(f"validation_layers cannot be empty for {self.name}.")
        if any(not isinstance(layer, ValidationLayer) for layer in self.validation_layers):
            raise ValueError(f"validation_layers must contain ValidationLayer values for {self.name}.")
        if not self.description:
            raise ValueError(f"description cannot be empty for {self.name}.")


def get_core_mechanism_specs() -> tuple[MechanismSpec, ...]:
    """Return one metadata specification for each core ABM v5 mechanism."""
    return (
        MechanismSpec(
            MechanismName.PRODUCTION_FEASIBILITY,
            (
                TheorySource.INPUT_OUTPUT_ECONOMICS,
                TheorySource.SUPPLY_CHAIN_ABM,
                TheorySource.REMASTER_CONTRACT,
            ),
            ComplexityLayer.PRODUCTION_NETWORK,
            ComplexityLevel.CROSS_LEVEL,
            (
                "desired_output",
                "input_requirements",
                "supplier_weights",
                "capacity_proxy",
                "inventory_or_buffer_proxy",
            ),
            ("production_feasibility", "realized_output"),
            FeedbackStatus.DESIGN_TARGET,
            (
                ValidationLayer.ACCOUNTING_VALIDITY,
                ValidationLayer.STRUCTURAL_VALIDITY,
                ValidationLayer.MECHANISM_VALIDITY,
            ),
            "Maps demand and input requirements to feasible realized output under network constraints.",
        ),
        MechanismSpec(
            MechanismName.SUPPLIER_SUBSTITUTION,
            (
                TheorySource.SUPPLY_CHAIN_ABM,
                TheorySource.COMPLEXITY_ECONOMICS,
                TheorySource.REMASTER_CONTRACT,
            ),
            ComplexityLayer.SUPPLIER_ADAPTATION,
            ComplexityLevel.MESO,
            ("supplier_weights", "supplier_opportunity_set", "substitution_friction", "supplier_lock_in"),
            ("supplier_weights", "supplier_lock_in"),
            FeedbackStatus.DESIGN_TARGET,
            (
                ValidationLayer.STRUCTURAL_VALIDITY,
                ValidationLayer.MECHANISM_VALIDITY,
                ValidationLayer.ABLATION_VALIDITY,
            ),
            "Defines future substitution among suppliers without implementing the choice rule.",
        ),
        MechanismSpec(
            MechanismName.BOUNDED_SUPPLIER_SEARCH,
            (TheorySource.COMPLEXITY_ECONOMICS, TheorySource.SUPPLY_CHAIN_ABM),
            ComplexityLayer.SUPPLIER_ADAPTATION,
            ComplexityLevel.MICRO,
            ("supplier_opportunity_set", "supplier_lock_in", "substitution_friction", "policy_exposure"),
            ("supplier_opportunity_set", "supplier_weights"),
            FeedbackStatus.DESIGN_TARGET,
            (ValidationLayer.MECHANISM_VALIDITY, ValidationLayer.ABLATION_VALIDITY),
            "Represents bounded local search over supplier opportunities for later mechanism design.",
        ),
        MechanismSpec(
            MechanismName.CAPABILITY_ACCUMULATION,
            (TheorySource.ECONOMIC_COMPLEXITY, TheorySource.COMPLEXITY_ECONOMICS),
            ComplexityLayer.CAPABILITY,
            ComplexityLevel.CROSS_LEVEL,
            ("general_capability", "green_capability", "output", "ecosystem_proximity", "policy_exposure"),
            ("general_capability", "green_capability"),
            FeedbackStatus.DESIGN_TARGET,
            (
                ValidationLayer.STRUCTURAL_VALIDITY,
                ValidationLayer.MECHANISM_VALIDITY,
                ValidationLayer.HISTORICAL_PLAUSIBILITY,
            ),
            "Links production and ecosystem proximity to future capability accumulation metadata.",
        ),
        MechanismSpec(
            MechanismName.DIRECTED_ECOSYSTEM_MOVEMENT,
            (TheorySource.DIRECTED_PRODUCTIVE_ECOSYSTEMS, TheorySource.ECONOMIC_COMPLEXITY),
            ComplexityLayer.CAPABILITY,
            ComplexityLevel.CROSS_LEVEL,
            (
                "directed_green_precedence",
                "ecosystem_proximity",
                "transition_sector_score",
                "reachable_green_complexity",
            ),
            ("directed_green_precedence", "reachable_green_complexity", "transition_sector_score"),
            FeedbackStatus.DESIGN_TARGET,
            (
                ValidationLayer.STRUCTURAL_VALIDITY,
                ValidationLayer.MECHANISM_VALIDITY,
                ValidationLayer.HISTORICAL_PLAUSIBILITY,
            ),
            "Describes directed movement through productive ecosystems toward greener complexity.",
        ),
        MechanismSpec(
            MechanismName.EMISSIONS_INTENSITY_TRANSITION,
            (
                TheorySource.ECONOMIC_COMPLEXITY,
                TheorySource.COMPLEXITY_ECONOMICS,
                TheorySource.REMASTER_CONTRACT,
            ),
            ComplexityLayer.ACCOUNTING,
            ComplexityLevel.CROSS_LEVEL,
            (
                "emissions_intensity",
                "green_capability",
                "network_green_exposure",
                "brown_centrality",
                "energy_dependence",
                "capital_stock_inertia",
            ),
            ("emissions_intensity", "local_greenness"),
            FeedbackStatus.DESIGN_TARGET,
            (
                ValidationLayer.ACCOUNTING_VALIDITY,
                ValidationLayer.MECHANISM_VALIDITY,
                ValidationLayer.HISTORICAL_PLAUSIBILITY,
            ),
            "Defines the future transition target for emissions intensity and local green-ness.",
        ),
        MechanismSpec(
            MechanismName.ENERGY_DEPENDENCE_CONSTRAINT,
            (TheorySource.REMASTER_CONTRACT, TheorySource.INPUT_OUTPUT_ECONOMICS),
            ComplexityLayer.ENERGY_INERTIA,
            ComplexityLevel.MESO,
            ("energy_dependence", "fuel_structure_proxy", "output"),
            ("energy_dependence", "production_feasibility"),
            FeedbackStatus.DESIGN_TARGET,
            (ValidationLayer.STRUCTURAL_VALIDITY, ValidationLayer.MECHANISM_VALIDITY),
            "Captures energy dependence as a constraint on future production feasibility.",
        ),
        MechanismSpec(
            MechanismName.CAPITAL_STOCK_INERTIA,
            (TheorySource.COMPLEXITY_ECONOMICS, TheorySource.REMASTER_CONTRACT),
            ComplexityLayer.ENERGY_INERTIA,
            ComplexityLevel.MESO,
            ("capital_stock_inertia", "output", "emissions_intensity", "policy_exposure"),
            ("capital_stock_inertia",),
            FeedbackStatus.DESIGN_TARGET,
            (ValidationLayer.MECHANISM_VALIDITY, ValidationLayer.HISTORICAL_PLAUSIBILITY),
            "Represents slow adjustment from long-lived capital stock in later dynamics.",
        ),
        MechanismSpec(
            MechanismName.POLICY_REGIME_EXPOSURE,
            (TheorySource.POLICY_TRANSITION_THEORY, TheorySource.REMASTER_CONTRACT),
            ComplexityLayer.POLICY_REGIME,
            ComplexityLevel.CROSS_LEVEL,
            ("policy_exposure", "emissions_intensity", "brown_centrality", "green_capability"),
            ("policy_exposure", "supplier_opportunity_set", "green_capability"),
            FeedbackStatus.SCENARIO_DRIVEN,
            (ValidationLayer.MECHANISM_VALIDITY, ValidationLayer.SCENARIO_CREDIBILITY),
            "Defines scenario-driven policy exposure metadata without executing scenarios.",
        ),
        MechanismSpec(
            MechanismName.PHASE_SPACE_REGIME_SWITCHING,
            (TheorySource.COMPLEXITY_ECONOMICS, TheorySource.REMASTER_CONTRACT),
            ComplexityLayer.REGIME_DISCOVERY,
            ComplexityLevel.CROSS_LEVEL,
            ("phase_space_position", "regime_membership", "regime_probability"),
            ("regime_membership", "regime_probability"),
            FeedbackStatus.DESIGN_TARGET,
            (
                ValidationLayer.STRUCTURAL_VALIDITY,
                ValidationLayer.MECHANISM_VALIDITY,
                ValidationLayer.HISTORICAL_PLAUSIBILITY,
            ),
            "Defines future regime switching metadata from discovered phase-space positions.",
        ),
        MechanismSpec(
            MechanismName.NETWORK_GREEN_EXPOSURE_FEEDBACK,
            (
                TheorySource.COMPLEXITY_ECONOMICS,
                TheorySource.INPUT_OUTPUT_ECONOMICS,
                TheorySource.REMASTER_CONTRACT,
            ),
            ComplexityLayer.PRODUCTION_NETWORK,
            ComplexityLevel.CROSS_LEVEL,
            ("network_green_exposure", "supplier_weights", "emissions_intensity", "local_greenness"),
            ("network_green_exposure", "supplier_weights"),
            FeedbackStatus.DESIGN_TARGET,
            (
                ValidationLayer.STRUCTURAL_VALIDITY,
                ValidationLayer.MECHANISM_VALIDITY,
                ValidationLayer.ABLATION_VALIDITY,
            ),
            "Links green network exposure and supplier structure as a future feedback channel.",
        ),
        MechanismSpec(
            MechanismName.BROWN_LOCK_IN_FEEDBACK,
            (
                TheorySource.COMPLEXITY_ECONOMICS,
                TheorySource.SUPPLY_CHAIN_ABM,
                TheorySource.REMASTER_CONTRACT,
            ),
            ComplexityLayer.SUPPLIER_ADAPTATION,
            ComplexityLevel.CROSS_LEVEL,
            ("brown_centrality", "supplier_lock_in", "energy_dependence", "capital_stock_inertia"),
            ("supplier_lock_in", "production_feasibility", "regime_probability"),
            FeedbackStatus.DESIGN_TARGET,
            (
                ValidationLayer.MECHANISM_VALIDITY,
                ValidationLayer.HISTORICAL_PLAUSIBILITY,
                ValidationLayer.ABLATION_VALIDITY,
            ),
            "Defines brown lock-in as a feedback between centrality, supplier inertia, and feasibility.",
        ),
    )


@dataclass(frozen=True)
class ABMV5OntologyRegistry:
    """Validated ontology registry for Phase 1 ABM v5 metadata."""

    agent_identity: AgentIdentityOntology = field(default_factory=AgentIdentityOntology)
    state_variables: tuple[StateVariableSpec, ...] = field(default_factory=get_core_state_variable_specs)
    functional_roles: tuple[FunctionalRoleSpec, ...] = field(default_factory=get_functional_role_specs)
    mechanisms: tuple[MechanismSpec, ...] = field(default_factory=get_core_mechanism_specs)

    def validate(self) -> None:
        """Validate ontology components and cross-references."""
        self.agent_identity.validate()
        for state_variable in self.state_variables:
            state_variable.validate()
        for functional_role in self.functional_roles:
            functional_role.validate()
        for mechanism in self.mechanisms:
            mechanism.validate()

        state_variable_names = self.state_variable_names()
        if len(state_variable_names) != len(set(state_variable_names)):
            raise ValueError("state variable names must be unique.")

        role_values = tuple(role.role.value for role in self.functional_roles)
        if len(role_values) != len(set(role_values)):
            raise ValueError("functional role values must be unique.")

        mechanism_values = tuple(mechanism.name.value for mechanism in self.mechanisms)
        if len(mechanism_values) != len(set(mechanism_values)):
            raise ValueError("mechanism names must be unique.")

        known_variables = set(state_variable_names)
        for mechanism in self.mechanisms:
            missing_inputs = set(mechanism.input_variables) - known_variables
            if missing_inputs:
                raise ValueError(
                    f"{mechanism.name.value} references unknown input variables: "
                    f"{sorted(missing_inputs)}"
                )
            missing_targets = set(mechanism.update_targets) - known_variables
            if missing_targets:
                raise ValueError(
                    f"{mechanism.name.value} references unknown update targets: "
                    f"{sorted(missing_targets)}"
                )

    def state_variable_names(self) -> tuple[str, ...]:
        """Return state variable names in registry order."""
        return tuple(state_variable.name for state_variable in self.state_variables)

    def mechanism_names(self) -> tuple[str, ...]:
        """Return mechanism names in registry order."""
        return tuple(mechanism.name.value for mechanism in self.mechanisms)

    def get_state_variable(self, name: str) -> StateVariableSpec:
        """Return one state variable spec by name."""
        for state_variable in self.state_variables:
            if state_variable.name == name:
                return state_variable
        raise KeyError(f"Unknown state variable: {name}")

    def get_mechanism(self, name: MechanismName | str) -> MechanismSpec:
        """Return one mechanism spec by enum or string value."""
        mechanism_name = MechanismName(name)
        for mechanism in self.mechanisms:
            if mechanism.name is mechanism_name:
                return mechanism
        raise KeyError(f"Unknown mechanism: {name}")


def build_default_ontology_registry() -> ABMV5OntologyRegistry:
    """Build and validate the default Phase 1 ontology registry."""
    registry = ABMV5OntologyRegistry()
    registry.validate()
    return registry
