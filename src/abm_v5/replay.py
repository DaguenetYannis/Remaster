from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.abm_v5.config import DEFAULT_HISTORICAL_END_YEAR, DEFAULT_HISTORICAL_START_YEAR, ValidationLayer
from src.abm_v5.mechanisms import (
    build_default_mechanism_runtime_registry,
    write_default_ablation_config,
    write_mechanism_registry_json,
)
from src.abm_v5.paths import ABMV5Paths
from src.abm_v5.regime_handoff import HANDOFF_PANEL_FILENAME
from src.abm_v5.validation import ValidationResult, ValidationSeverity, ValidationStatus


HISTORICAL_START_YEAR = DEFAULT_HISTORICAL_START_YEAR
HISTORICAL_END_YEAR = DEFAULT_HISTORICAL_END_YEAR

REPLAY_PANEL_FILENAME = "replay_scaffold_panel_1995_2016.parquet"
REPLAY_METADATA_FILENAME = "replay_metadata.json"
MECHANISM_REGISTRY_FILENAME = "mechanism_registry.json"
ABLATION_CONFIG_FILENAME = "ablation_config.json"
REPLAY_VALIDATION_FILENAME = "replay_scaffold_validation.json"

DEFAULT_REPLAY_RUN_ID = "abm_v5_phase4_1_identity_replay_scaffold"
DEFAULT_REPLAY_MODE = "historical_replay_scaffold"


@dataclass(frozen=True)
class ReplayStateColumns:
    identity_columns: tuple[str, ...]
    time_column: str
    core_state_columns: tuple[str, ...]
    diagnostic_columns: tuple[str, ...]
    regime_columns: tuple[str, ...]


def build_default_replay_state_columns() -> ReplayStateColumns:
    return ReplayStateColumns(
        identity_columns=("country_sector", "country", "country_detail", "category", "sector"),
        time_column="year",
        core_state_columns=(
            "output",
            "final_demand",
            "emissions",
            "emissions_intensity",
            "local_greenness",
            "emissions_intensity_gap",
            "green_capability",
            "general_capability",
            "network_green_exposure",
            "brown_centrality",
            "supplier_lock_in",
        ),
        diagnostic_columns=(
            "supplier_count",
            "buyer_count",
            "supplier_concentration_hhi",
            "buyer_concentration_hhi",
            "import_dependence_proxy",
            "export_dependence_proxy",
            "phase_space_empirical_completeness",
            "phase_space_ready_for_regime_discovery_flag",
        ),
        regime_columns=(
            "regime_membership",
            "regime_confidence",
            "transition_state",
            "transition_confidence",
            "mechanism_learning_eligible_flag",
        ),
    )


@dataclass(frozen=True)
class ReplayMetadata:
    run_id: str
    start_year: int
    end_year: int
    mode: str
    active_mechanisms: tuple[str, ...]
    scaffold_only: bool
    notes: str

    def validate(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must not be empty.")
        if self.start_year != DEFAULT_HISTORICAL_START_YEAR:
            raise ValueError("start_year must be 1995.")
        if self.end_year != DEFAULT_HISTORICAL_END_YEAR:
            raise ValueError("end_year must be 2016.")
        if self.mode != DEFAULT_REPLAY_MODE:
            raise ValueError("mode must be historical_replay_scaffold.")
        if self.active_mechanisms != ("identity_replay",):
            raise ValueError("active_mechanisms must only include identity_replay in Phase 4.1.")
        if not self.scaffold_only:
            raise ValueError("scaffold_only must be True in Phase 4.1.")
        if not self.notes:
            raise ValueError("notes must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "run_id": self.run_id,
            "start_year": self.start_year,
            "end_year": self.end_year,
            "mode": self.mode,
            "active_mechanisms": list(self.active_mechanisms),
            "scaffold_only": self.scaffold_only,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ReplayScaffoldBuildResult:
    replay_panel_path: Path
    replay_metadata_path: Path
    mechanism_registry_path: Path
    ablation_config_path: Path
    validation_path: Path
    start_year: int
    end_year: int
    n_rows: int
    n_agents: int

    def validate(self) -> None:
        for field_name in (
            "replay_panel_path",
            "replay_metadata_path",
            "mechanism_registry_path",
            "ablation_config_path",
            "validation_path",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must not be empty.")
        if self.n_rows <= 0:
            raise ValueError("n_rows must be positive.")
        if self.n_agents <= 0:
            raise ValueError("n_agents must be positive.")
        if self.start_year != DEFAULT_HISTORICAL_START_YEAR:
            raise ValueError("start_year must be 1995.")
        if self.end_year != DEFAULT_HISTORICAL_END_YEAR:
            raise ValueError("end_year must be 2016.")


def _as_polars(df):
    import polars as pl

    return df if isinstance(df, pl.DataFrame) else pl.DataFrame(df)


def build_initial_replay_state(handoff_df):
    import polars as pl

    frame = _as_polars(handoff_df)
    copy_pairs = {
        "simulated_output": "output",
        "simulated_emissions": "emissions",
        "simulated_emissions_intensity": "emissions_intensity",
        "simulated_local_greenness": "local_greenness",
        "simulated_network_green_exposure": "network_green_exposure",
        "simulated_green_capability": "green_capability",
        "simulated_general_capability": "general_capability",
        "simulated_supplier_lock_in": "supplier_lock_in",
        "simulated_brown_centrality": "brown_centrality",
    }
    expressions = [
        pl.lit(DEFAULT_REPLAY_RUN_ID).alias("replay_run_id"),
        pl.lit(DEFAULT_REPLAY_MODE).alias("replay_mode"),
        pl.lit(True).alias("replay_scaffold_only_flag"),
    ]
    for simulated_column, observed_column in copy_pairs.items():
        expressions.append(
            pl.col(observed_column).alias(simulated_column)
            if observed_column in frame.columns
            else pl.lit(None).alias(simulated_column)
        )
    return frame.with_columns(expressions)


def run_identity_historical_replay(handoff_df, metadata: ReplayMetadata):
    import polars as pl

    metadata.validate()
    frame = _as_polars(handoff_df)
    if {"country_sector", "year"}.issubset(frame.columns):
        frame = frame.sort(["country_sector", "year"])
    replay = build_initial_replay_state(frame)
    return replay.with_columns(
        pl.lit(metadata.run_id).alias("replay_run_id"),
        pl.lit(metadata.mode).alias("replay_mode"),
        pl.lit(metadata.scaffold_only).alias("replay_scaffold_only_flag"),
    )


def _result(
    name: str,
    failed: int,
    checked: int,
    message: str,
    layer: ValidationLayer,
    severity: ValidationSeverity = ValidationSeverity.ERROR,
) -> ValidationResult:
    result = ValidationResult(
        name,
        ValidationStatus.FAILED if failed else ValidationStatus.PASSED,
        severity if failed else ValidationSeverity.INFO,
        message,
        layer,
        failed,
        checked,
    )
    result.validate()
    return result


def validate_replay_scaffold_output(replay_df) -> tuple[ValidationResult, ...]:
    import polars as pl

    frame = _as_polars(replay_df)
    required = (
        "country_sector",
        "year",
        "replay_run_id",
        "replay_mode",
        "replay_scaffold_only_flag",
        "simulated_output",
        "simulated_emissions",
        "simulated_emissions_intensity",
        "simulated_local_greenness",
        "simulated_network_green_exposure",
        "simulated_green_capability",
        "simulated_general_capability",
        "simulated_supplier_lock_in",
        "simulated_brown_centrality",
    )
    missing = [column for column in required if column not in frame.columns]
    results = [
        _result(
            "replay_required_columns",
            len(missing),
            len(required),
            f"Missing replay columns: {missing}." if missing else "Required replay columns are present.",
            ValidationLayer.STRUCTURAL_VALIDITY,
            ValidationSeverity.CRITICAL,
        )
    ]
    if missing:
        return tuple(results)

    duplicates = int(frame.group_by(["country_sector", "year"]).len().filter(pl.col("len") > 1)["len"].sum() or 0)
    results.append(
        _result(
            "replay_unique_country_sector_year",
            duplicates,
            frame.height,
            "Replay country_sector-year rows are unique.",
            ValidationLayer.STRUCTURAL_VALIDITY,
            ValidationSeverity.CRITICAL,
        )
    )
    years = sorted(frame["year"].drop_nulls().unique().to_list())
    out_years = [year for year in years if year < HISTORICAL_START_YEAR or year > HISTORICAL_END_YEAR]
    results.append(
        _result(
            "replay_year_range",
            len(out_years),
            len(years),
            "Replay years are within 1995-2016.",
            ValidationLayer.STRUCTURAL_VALIDITY,
        )
    )
    non_scaffold_rows = frame.filter(pl.col("replay_scaffold_only_flag") != True).height
    results.append(
        _result(
            "replay_scaffold_only_flag_true",
            non_scaffold_rows,
            frame.height,
            "Replay scaffold-only flag is True for all rows.",
            ValidationLayer.MECHANISM_VALIDITY,
            ValidationSeverity.CRITICAL,
        )
    )
    forbidden = [column for column in ("scenario_id", "policy_shock") if column in frame.columns]
    results.append(
        _result(
            "replay_no_scenario_or_policy_columns",
            len(forbidden),
            2,
            f"Forbidden scenario/policy columns present: {forbidden}." if forbidden else "No scenario or policy-shock columns are present.",
            ValidationLayer.STRUCTURAL_VALIDITY,
            ValidationSeverity.CRITICAL,
        )
    )
    copy_pairs = {
        "simulated_output": "output",
        "simulated_emissions": "emissions",
        "simulated_emissions_intensity": "emissions_intensity",
        "simulated_network_green_exposure": "network_green_exposure",
        "simulated_green_capability": "green_capability",
        "simulated_supplier_lock_in": "supplier_lock_in",
    }
    for simulated_column, observed_column in copy_pairs.items():
        if observed_column not in frame.columns:
            results.append(
                _result(
                    f"replay_{simulated_column}_matches_observed",
                    0,
                    0,
                    f"Observed column {observed_column} is absent; equality check skipped.",
                    ValidationLayer.ACCOUNTING_VALIDITY,
                    ValidationSeverity.WARNING,
                )
            )
            continue
        mismatches = frame.filter(
            pl.col(simulated_column).is_not_null()
            & pl.col(observed_column).is_not_null()
            & (pl.col(simulated_column) != pl.col(observed_column))
        ).height
        comparable = frame.filter(pl.col(simulated_column).is_not_null() & pl.col(observed_column).is_not_null()).height
        results.append(
            _result(
                f"replay_{simulated_column}_matches_observed",
                mismatches,
                comparable,
                f"{simulated_column} equals {observed_column} where both are non-null.",
                ValidationLayer.ACCOUNTING_VALIDITY,
                ValidationSeverity.CRITICAL,
            )
        )
    behavioural_columns = [
        column
        for column in (
            "mechanism_applied",
            "active_mechanism",
            "behavioral_rule_id",
            "stochastic_draw",
            "calibration_parameter",
        )
        if column in frame.columns
    ]
    results.append(
        _result(
            "replay_no_behavioural_mechanism_columns",
            len(behavioural_columns),
            5,
            f"Behavioural mechanism columns present: {behavioural_columns}."
            if behavioural_columns
            else "No behavioural mechanism columns are present.",
            ValidationLayer.MECHANISM_VALIDITY,
            ValidationSeverity.CRITICAL,
        )
    )
    return tuple(results)


def _validation_to_dict(results: tuple[ValidationResult, ...]) -> list[dict[str, Any]]:
    return [
        {
            "check_name": result.check_name,
            "status": result.status.value,
            "severity": result.severity.value,
            "message": result.message,
            "layer": result.layer.value,
            "n_failed": result.n_failed,
            "n_checked": result.n_checked,
        }
        for result in results
    ]


def _has_critical_failures(results: tuple[ValidationResult, ...]) -> bool:
    return any(result.status is ValidationStatus.FAILED and result.severity is ValidationSeverity.CRITICAL for result in results)


def build_replay_scaffold(project_root: Path) -> ReplayScaffoldBuildResult:
    import polars as pl

    paths = ABMV5Paths.from_project_root(project_root)
    paths.validate_project_root()
    paths.ensure_directories()
    handoff_path = paths.regimes / HANDOFF_PANEL_FILENAME
    if not handoff_path.exists():
        raise FileNotFoundError(f"Regime handoff panel missing: {handoff_path}. Run Phase 3.5 first.")

    handoff = pl.read_parquet(handoff_path)
    registry = build_default_mechanism_runtime_registry()
    mechanism_registry_path = paths.simulation / MECHANISM_REGISTRY_FILENAME
    ablation_config_path = paths.simulation / ABLATION_CONFIG_FILENAME
    write_mechanism_registry_json(registry, mechanism_registry_path)
    write_default_ablation_config(ablation_config_path)

    metadata = ReplayMetadata(
        run_id=DEFAULT_REPLAY_RUN_ID,
        start_year=HISTORICAL_START_YEAR,
        end_year=HISTORICAL_END_YEAR,
        mode=DEFAULT_REPLAY_MODE,
        active_mechanisms=("identity_replay",),
        scaffold_only=True,
        notes="Phase 4.1 scaffold-only replay. No behavioural mechanisms, scenarios, policy shocks, stochastic dynamics, calibration, or optimisation are implemented.",
    )
    replay = run_identity_historical_replay(handoff, metadata)
    results = validate_replay_scaffold_output(replay)

    replay_panel_path = paths.simulation / REPLAY_PANEL_FILENAME
    replay_metadata_path = paths.simulation / REPLAY_METADATA_FILENAME
    validation_path = paths.validation / REPLAY_VALIDATION_FILENAME
    replay.write_parquet(replay_panel_path)
    replay_metadata_path.write_text(json.dumps(metadata.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    validation_path.write_text(
        json.dumps({"validation_scope": "abm_v5_replay_scaffold", "results": _validation_to_dict(results)}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if _has_critical_failures(results):
        raise ValueError(f"Replay scaffold validation has critical failures: {validation_path}")

    result = ReplayScaffoldBuildResult(
        replay_panel_path=replay_panel_path,
        replay_metadata_path=replay_metadata_path,
        mechanism_registry_path=mechanism_registry_path,
        ablation_config_path=ablation_config_path,
        validation_path=validation_path,
        start_year=HISTORICAL_START_YEAR,
        end_year=HISTORICAL_END_YEAR,
        n_rows=replay.height,
        n_agents=replay["country_sector"].n_unique(),
    )
    result.validate()
    return result


def load_replay_scaffold_panel(path: Path):
    import polars as pl

    return pl.read_parquet(path)
