from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


DEFAULT_HISTORICAL_START_YEAR = 1995
DEFAULT_HISTORICAL_END_YEAR = 2016
DEFAULT_RANDOM_SEED = 42


class ComplexityLayer(str, Enum):
    """Complexity ladder layers defined by the ABM v5 contract."""

    ACCOUNTING = "accounting"
    PRODUCTION_NETWORK = "production_network"
    CAPABILITY = "capability"
    SUPPLIER_ADAPTATION = "supplier_adaptation"
    ENERGY_INERTIA = "energy_inertia"
    POLICY_REGIME = "policy_regime"
    REGIME_DISCOVERY = "regime_discovery"
    SCENARIO = "scenario"


class ComplexityLevel(str, Enum):
    """Level at which a variable or mechanism is interpreted."""

    MACRO = "macro"
    MESO = "meso"
    MICRO = "micro"
    CROSS_LEVEL = "cross_level"


class FeedbackStatus(str, Enum):
    """How a variable or mechanism participates in feedback."""

    OBSERVED = "observed"
    DIAGNOSTIC = "diagnostic"
    EXOGENOUS = "exogenous"
    ENDOGENOUS = "endogenous"
    SCENARIO_DRIVEN = "scenario_driven"
    DESIGN_TARGET = "design_target"


class ValidationLayer(str, Enum):
    """Validation layer categories for ABM v5 mechanisms."""

    ACCOUNTING_VALIDITY = "accounting_validity"
    STRUCTURAL_VALIDITY = "structural_validity"
    MECHANISM_VALIDITY = "mechanism_validity"
    HISTORICAL_PLAUSIBILITY = "historical_plausibility"
    ABLATION_VALIDITY = "ablation_validity"
    SCENARIO_CREDIBILITY = "scenario_credibility"


class SourceStatus(str, Enum):
    """Source status for state variables and later schema fields."""

    RAW_OBSERVED = "raw_observed"
    DERIVED = "derived"
    ESTIMATED = "estimated"
    PROXY = "proxy"
    SIMULATED = "simulated"
    PLACEHOLDER = "placeholder"


class ModelStage(str, Enum):
    """Implementation stages for the ABM v5 model layer."""

    FOUNDATION = "foundation"
    HISTORICAL_CONSTRUCTION = "historical_construction"
    DISCOVERY = "discovery"
    MECHANISM_IMPLEMENTATION = "mechanism_implementation"
    HISTORICAL_REPLAY = "historical_replay"
    SCENARIO_EXPERIMENT = "scenario_experiment"
    REPORTING = "reporting"


def _default_enabled_layers() -> tuple[ComplexityLayer, ...]:
    """Return foundation-enabled layers, excluding scenario execution."""
    return tuple(layer for layer in ComplexityLayer if layer is not ComplexityLayer.SCENARIO)


@dataclass(frozen=True)
class HistoricalWindowConfig:
    """Historical years used to construct and diagnose ABM v5."""

    start_year: int = DEFAULT_HISTORICAL_START_YEAR
    end_year: int = DEFAULT_HISTORICAL_END_YEAR

    def years(self) -> list[int]:
        """Return the inclusive historical year list."""
        return list(range(self.start_year, self.end_year + 1))

    def validate(self) -> None:
        """Validate that the historical window is positive and ordered."""
        if self.start_year <= 0:
            raise ValueError("start_year must be positive.")
        if self.end_year <= 0:
            raise ValueError("end_year must be positive.")
        if self.start_year > self.end_year:
            raise ValueError("start_year must be less than or equal to end_year.")


@dataclass(frozen=True)
class ABMV5Config:
    """Top-level ABM v5 foundation configuration."""

    historical_window: HistoricalWindowConfig = field(default_factory=HistoricalWindowConfig)
    strict_validation: bool = True
    allow_missing_optional_inputs: bool = False
    random_seed: int = DEFAULT_RANDOM_SEED
    active_stage: ModelStage = ModelStage.FOUNDATION
    enabled_layers: tuple[ComplexityLayer, ...] = field(default_factory=_default_enabled_layers)

    def validate(self) -> None:
        """Validate foundation-stage constraints and nested configuration."""
        self.historical_window.validate()
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative.")
        if not self.enabled_layers:
            raise ValueError("enabled_layers must not be empty.")
        for layer in self.enabled_layers:
            if not isinstance(layer, ComplexityLayer):
                raise ValueError(f"enabled_layers contains an invalid layer: {layer!r}")
        if (
            self.active_stage is ModelStage.FOUNDATION
            and ComplexityLayer.SCENARIO in self.enabled_layers
        ):
            raise ValueError("SCENARIO layer cannot be enabled during FOUNDATION stage.")

    def is_layer_enabled(self, layer: ComplexityLayer | str) -> bool:
        """Return whether a complexity layer is enabled."""
        return ComplexityLayer(layer) in self.enabled_layers


@dataclass(frozen=True)
class SchemaConfig:
    """Schema naming configuration for later ABM v5 data contracts."""

    primary_agent_key: str = "country_sector"
    time_key: str = "year"
    strict_dtypes: bool = True
    require_semantic_metadata: bool = True

    def validate(self) -> None:
        """Validate required schema key names."""
        if not self.primary_agent_key:
            raise ValueError("primary_agent_key must not be empty.")
        if not self.time_key:
            raise ValueError("time_key must not be empty.")


@dataclass(frozen=True)
class OntologyConfig:
    """Ontology constraints for country-sector agent interpretation."""

    allow_fixed_agent_classes: bool = False
    allow_overlapping_roles: bool = True
    recompute_roles_each_year: bool = True
    agent_unit: str = "country_sector"

    def validate(self) -> None:
        """Validate ontology constraints from the ABM v5 contract."""
        if self.allow_fixed_agent_classes:
            raise ValueError("allow_fixed_agent_classes must remain False.")
        if self.agent_unit != "country_sector":
            raise ValueError('agent_unit must equal "country_sector".')
