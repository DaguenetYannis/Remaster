from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CalibrationConfig:
    """Historical calibration settings for reproducing 1995-2016 dynamics."""

    start_year: int = 1995
    end_year: int = 2016
    validation_split_year: int = 2008
    production_features: tuple[str, ...] = (
        "log_X_lag1",
        "demand_gap",
        "sector_X_growth",
        "country_X_growth",
    )
    emissions_features: tuple[str, ...] = (
        "log_EI_lag1",
        "green_capability",
        "g_network",
    )


@dataclass(frozen=True)
class ProjectionConfig:
    """Projection settings for scenario runs after historical validation."""

    start_year: int = 2017
    end_year: int = 2027


@dataclass(frozen=True)
class GreennessConfig:
    """Green-ness transformation settings.

    ABM v3 uses negative-log emissions intensity for empirical variation and
    rescales the result to [0, 1] before using it in transition rules.
    """

    epsilon: float = 1e-12
    raw_output_column: str = "g_local_raw_negative_log"
    scaled_output_column: str = "g_local"


@dataclass(frozen=True)
class CapabilityConfig:
    """Atlas capability settings.

    ``smoothing_lambda`` is interpreted as an export-value smoothing mass in
    the same units as ``active_good_export_value``.
    """

    smoothing_lambda: float = 1.0
    green_capability_column: str = "green_capability"
    general_complexity_column: str = "general_complexity"


@dataclass(frozen=True)
class SubstitutionConfig:
    """Supplier substitution settings for quantity-constrained dynamics."""

    substitution_friction: float = 0.25
    same_sector_required: bool = True


@dataclass(frozen=True)
class OutputConfig:
    """Output locations under the ABM v3 output root."""

    output_root: Path = Path("data/abm_v3")
    calibration_dir_name: str = "calibration"
    validation_dir_name: str = "validation"
    scenarios_dir_name: str = "scenarios"
    diagnostics_dir_name: str = "diagnostics"


@dataclass(frozen=True)
class ABMV3Config:
    """Top-level configuration object passed into ABM v3 orchestrators."""

    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    greenness: GreennessConfig = field(default_factory=GreennessConfig)
    capability: CapabilityConfig = field(default_factory=CapabilityConfig)
    substitution: SubstitutionConfig = field(default_factory=SubstitutionConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)
