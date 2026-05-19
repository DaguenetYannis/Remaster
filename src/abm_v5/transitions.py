from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from src.abm_v5.config import DEFAULT_HISTORICAL_END_YEAR, DEFAULT_HISTORICAL_START_YEAR, ValidationLayer
from src.abm_v5.paths import ABMV5Paths
from src.abm_v5.regimes import REGIME_PANEL_FILENAME, RegimeLabel
from src.abm_v5.validation import ValidationResult, ValidationSeverity, ValidationStatus


HISTORICAL_START_YEAR = DEFAULT_HISTORICAL_START_YEAR
HISTORICAL_END_YEAR = DEFAULT_HISTORICAL_END_YEAR
TRANSITION_PANEL_FILENAME = "historical_transition_panel_1995_2016.parquet"
TRANSITION_MATRIX_FILENAME = "regime_transition_matrix.csv"
TRANSITION_SUMMARY_FILENAME = "transition_state_summary.csv"
TRANSITION_BY_YEAR_FILENAME = "transition_state_by_year.csv"
TRANSITION_BY_SECTOR_FILENAME = "transition_state_by_sector.csv"
TRANSITION_VALIDATION_FILENAME = "transition_encoding_validation.json"
TRANSITION_ASSIGNMENT_METHOD = "global_robust_delta_rules"
TRANSITION_RULE_ID = "abm_v5_phase3_transition_encoding_v1"

DELTA_SOURCE_VARIABLES = (
    "emissions_intensity_gap",
    "network_green_exposure",
    "green_capability",
    "general_capability",
    "supplier_lock_in",
    "brown_centrality",
    "local_greenness",
    "output",
    "emissions",
)
THRESHOLD_DELTA_VARIABLES = tuple(f"delta_{variable}" for variable in DELTA_SOURCE_VARIABLES[:7])


class TransitionState(str, Enum):
    NO_NEXT_YEAR = "no_next_year"
    INSUFFICIENT_DATA_TRANSITION = "insufficient_data_transition"
    STABLE_SAME_REGIME = "stable_same_regime"
    REGIME_SWITCH = "regime_switch"
    GREEN_EMBEDDING_IMPROVEMENT = "green_embedding_improvement"
    GREEN_EMBEDDING_DETERIORATION = "green_embedding_deterioration"
    CAPABILITY_GAIN = "capability_gain"
    CAPABILITY_LOSS = "capability_loss"
    SUPPLIER_LOCK_IN_INCREASE = "supplier_lock_in_increase"
    SUPPLIER_LOCK_IN_RELIEF = "supplier_lock_in_relief"
    DIRTY_GAP_IMPROVEMENT = "dirty_gap_improvement"
    DIRTY_GAP_WORSENING = "dirty_gap_worsening"
    MIXED_MOVEMENT = "mixed_movement"


@dataclass(frozen=True)
class TransitionEncodingBuildResult:
    transition_panel_path: Path
    transition_matrix_path: Path
    transition_state_summary_path: Path
    transition_state_by_year_path: Path
    transition_state_by_sector_path: Path
    validation_path: Path
    start_year: int
    end_year: int
    n_rows: int
    n_agents: int
    n_transitions: int

    def validate(self) -> None:
        for field_name in (
            "transition_panel_path",
            "transition_matrix_path",
            "transition_state_summary_path",
            "transition_state_by_year_path",
            "transition_state_by_sector_path",
            "validation_path",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must not be empty.")
        if self.n_rows <= 0:
            raise ValueError("n_rows must be positive.")
        if self.n_agents <= 0:
            raise ValueError("n_agents must be positive.")
        if self.n_transitions <= 0:
            raise ValueError("n_transitions must be positive.")
        if self.start_year != DEFAULT_HISTORICAL_START_YEAR:
            raise ValueError("start_year must be 1995.")
        if self.end_year != DEFAULT_HISTORICAL_END_YEAR:
            raise ValueError("end_year must be 2016.")


def add_forward_transition_columns(df):
    import polars as pl

    frame = df if isinstance(df, pl.DataFrame) else pl.DataFrame(df)
    sorted_frame = frame.sort(["country_sector", "year"])
    expressions = [
        pl.col("year").shift(-1).over("country_sector").alias("_observed_next_year"),
        pl.col("regime_membership").shift(-1).over("country_sector").alias("next_regime_membership"),
        pl.col("regime_membership").shift(1).over("country_sector").alias("previous_regime_membership"),
    ]
    for variable in DELTA_SOURCE_VARIABLES:
        expressions.append(
            (pl.col(variable).shift(-1).over("country_sector") - pl.col(variable)).alias(f"delta_{variable}")
        )
    return sorted_frame.with_columns(expressions).with_columns(
        pl.when(pl.col("_observed_next_year") == pl.col("year") + 1)
        .then(pl.col("_observed_next_year"))
        .otherwise(None)
        .alias("next_year")
    ).with_columns(
        pl.when(pl.col("next_year").is_null())
        .then(None)
        .otherwise(pl.col("next_regime_membership"))
        .alias("next_regime_membership")
    ).with_columns(
        pl.when(pl.col("next_regime_membership").is_null())
        .then(None)
        .otherwise(pl.col("next_regime_membership") != pl.col("regime_membership"))
        .alias("regime_switch_flag")
    ).drop("_observed_next_year")


def compute_transition_delta_thresholds(df) -> dict[str, Any]:
    import polars as pl

    frame = df if isinstance(df, pl.DataFrame) else pl.DataFrame(df)
    thresholds: dict[str, Any] = {}
    for variable in THRESHOLD_DELTA_VARIABLES:
        if variable not in frame.columns:
            thresholds[variable] = {
                "small_change_threshold": None,
                "meaningful_change_threshold": None,
                "large_change_threshold": None,
                "n_observations_used": 0,
            }
            continue
        values = frame.select(pl.col(variable).abs().alias(variable)).filter(pl.col(variable).is_not_null() & (pl.col(variable) > 0))
        if values.is_empty():
            thresholds[variable] = {
                "small_change_threshold": None,
                "meaningful_change_threshold": None,
                "large_change_threshold": None,
                "n_observations_used": 0,
            }
        else:
            thresholds[variable] = {
                "small_change_threshold": float(values.select(pl.col(variable).quantile(0.25)).item()),
                "meaningful_change_threshold": float(values.select(pl.col(variable).quantile(0.50)).item()),
                "large_change_threshold": float(values.select(pl.col(variable).quantile(0.75)).item()),
                "n_observations_used": values.height,
            }
    return {
        "method": "global_robust_delta_quantiles",
        "variables": list(THRESHOLD_DELTA_VARIABLES),
        "thresholds": thresholds,
        "notes": "Thresholds use absolute non-zero historical year-to-year deltas only.",
    }


def _meaningful(thresholds: dict[str, Any], variable: str) -> float | None:
    value = thresholds.get("thresholds", {}).get(variable, {}).get("meaningful_change_threshold")
    return float(value) if value is not None else None


def assign_transition_states(df, thresholds: dict[str, Any]):
    import polars as pl

    frame = df if isinstance(df, pl.DataFrame) else pl.DataFrame(df)

    def threshold(variable: str) -> pl.Expr:
        value = _meaningful(thresholds, variable)
        return pl.lit(value if value is not None else float("inf"))

    insufficient = RegimeLabel.INSUFFICIENT_DATA.value
    assigned = frame.with_columns(
        pl.when(pl.col("next_year").is_null())
        .then(pl.lit(TransitionState.NO_NEXT_YEAR.value))
        .when((pl.col("regime_membership") == insufficient) | (pl.col("next_regime_membership") == insufficient))
        .then(pl.lit(TransitionState.INSUFFICIENT_DATA_TRANSITION.value))
        .when(pl.col("regime_switch_flag") == True)
        .then(pl.lit(TransitionState.REGIME_SWITCH.value))
        .when(pl.col("delta_network_green_exposure") >= threshold("delta_network_green_exposure"))
        .then(pl.lit(TransitionState.GREEN_EMBEDDING_IMPROVEMENT.value))
        .when(pl.col("delta_network_green_exposure") <= -threshold("delta_network_green_exposure"))
        .then(pl.lit(TransitionState.GREEN_EMBEDDING_DETERIORATION.value))
        .when(
            (pl.col("delta_green_capability") >= threshold("delta_green_capability"))
            | (pl.col("green_capability").is_null() & (pl.col("delta_general_capability") >= threshold("delta_general_capability")))
        )
        .then(pl.lit(TransitionState.CAPABILITY_GAIN.value))
        .when(
            (pl.col("delta_green_capability") <= -threshold("delta_green_capability"))
            | (pl.col("green_capability").is_null() & (pl.col("delta_general_capability") <= -threshold("delta_general_capability")))
        )
        .then(pl.lit(TransitionState.CAPABILITY_LOSS.value))
        .when(pl.col("delta_supplier_lock_in") >= threshold("delta_supplier_lock_in"))
        .then(pl.lit(TransitionState.SUPPLIER_LOCK_IN_INCREASE.value))
        .when(pl.col("delta_supplier_lock_in") <= -threshold("delta_supplier_lock_in"))
        .then(pl.lit(TransitionState.SUPPLIER_LOCK_IN_RELIEF.value))
        .when(pl.col("delta_emissions_intensity_gap") <= -threshold("delta_emissions_intensity_gap"))
        .then(pl.lit(TransitionState.DIRTY_GAP_IMPROVEMENT.value))
        .when(pl.col("delta_emissions_intensity_gap") >= threshold("delta_emissions_intensity_gap"))
        .then(pl.lit(TransitionState.DIRTY_GAP_WORSENING.value))
        .when(pl.col("regime_switch_flag") == False)
        .then(pl.lit(TransitionState.STABLE_SAME_REGIME.value))
        .otherwise(pl.lit(TransitionState.MIXED_MOVEMENT.value))
        .alias("transition_state")
    ).with_columns(
        pl.when(pl.col("transition_state") == TransitionState.NO_NEXT_YEAR.value)
        .then(pl.lit(1.0))
        .when(pl.col("transition_state") == TransitionState.INSUFFICIENT_DATA_TRANSITION.value)
        .then(pl.lit(0.25))
        .when(pl.col("transition_state") == TransitionState.REGIME_SWITCH.value)
        .then(pl.lit(1.0))
        .when(pl.col("transition_state").is_in([
            TransitionState.GREEN_EMBEDDING_IMPROVEMENT.value,
            TransitionState.GREEN_EMBEDDING_DETERIORATION.value,
        ]))
        .then(pl.lit(0.9))
        .when(pl.col("transition_state").is_in([
            TransitionState.CAPABILITY_GAIN.value,
            TransitionState.CAPABILITY_LOSS.value,
            TransitionState.SUPPLIER_LOCK_IN_INCREASE.value,
            TransitionState.SUPPLIER_LOCK_IN_RELIEF.value,
            TransitionState.DIRTY_GAP_IMPROVEMENT.value,
            TransitionState.DIRTY_GAP_WORSENING.value,
        ]))
        .then(pl.lit(0.85))
        .when(pl.col("transition_state") == TransitionState.STABLE_SAME_REGIME.value)
        .then(pl.lit(0.75))
        .otherwise(pl.lit(0.5))
        .alias("transition_confidence"),
        pl.lit(TRANSITION_ASSIGNMENT_METHOD).alias("transition_assignment_method"),
        pl.lit(TRANSITION_RULE_ID).alias("transition_rule_id"),
    )
    forbidden = [column for column in ("regime_probability", "scenario_id", "run_id") if column in assigned.columns]
    return assigned.drop(forbidden) if forbidden else assigned


def build_regime_transition_matrix(transition_df):
    import polars as pl

    frame = transition_df if isinstance(transition_df, pl.DataFrame) else pl.DataFrame(transition_df)
    counts = frame.drop_nulls(["next_regime_membership"]).group_by(["regime_membership", "next_regime_membership"]).len()
    return counts.pivot(index="regime_membership", on="next_regime_membership", values="len", aggregate_function="sum").fill_null(0).sort("regime_membership")


def build_transition_state_summary(transition_df):
    import polars as pl

    frame = transition_df if isinstance(transition_df, pl.DataFrame) else pl.DataFrame(transition_df)
    total = frame.height
    delta_cols = (
        "delta_emissions_intensity_gap",
        "delta_network_green_exposure",
        "delta_green_capability",
        "delta_general_capability",
        "delta_supplier_lock_in",
        "delta_brown_centrality",
        "delta_local_greenness",
    )
    return frame.group_by("transition_state").agg(
        pl.len().alias("n_rows"),
        (pl.len() / total).alias("share_rows"),
        pl.n_unique("country_sector").alias("n_agents"),
        pl.col("transition_confidence").mean().alias("mean_transition_confidence"),
        *[pl.col(column).cast(pl.Float64, strict=False).mean().alias(f"mean_{column}") for column in delta_cols],
    ).sort("transition_state")


def build_transition_state_by_year(transition_df):
    import polars as pl

    frame = transition_df if isinstance(transition_df, pl.DataFrame) else pl.DataFrame(transition_df)
    return frame.group_by(["year", "transition_state"]).agg(
        pl.len().alias("n_rows"),
        pl.col("transition_confidence").mean().alias("mean_transition_confidence"),
    ).with_columns(
        (pl.col("n_rows") / pl.sum("n_rows").over("year")).alias("share_year_rows")
    ).select("year", "transition_state", "n_rows", "share_year_rows", "mean_transition_confidence")


def build_transition_state_by_sector(transition_df):
    import polars as pl

    frame = transition_df if isinstance(transition_df, pl.DataFrame) else pl.DataFrame(transition_df)
    return frame.group_by(["sector", "transition_state"]).agg(
        pl.len().alias("n_rows"),
        pl.col("transition_confidence").mean().alias("mean_transition_confidence"),
        pl.col("output").cast(pl.Float64, strict=False).mean().alias("mean_output"),
        pl.col("emissions").cast(pl.Float64, strict=False).mean().alias("mean_emissions"),
    ).with_columns(
        (pl.col("n_rows") / pl.sum("n_rows").over("sector")).alias("share_sector_rows")
    ).select("sector", "transition_state", "n_rows", "share_sector_rows", "mean_transition_confidence", "mean_output", "mean_emissions")


def _validation_result(name: str, failed: int, checked: int, message: str, layer: ValidationLayer, severity: ValidationSeverity = ValidationSeverity.ERROR) -> ValidationResult:
    result = ValidationResult(
        check_name=name,
        status=ValidationStatus.FAILED if failed else ValidationStatus.PASSED,
        severity=severity if failed else ValidationSeverity.INFO,
        message=message,
        layer=layer,
        n_failed=failed,
        n_checked=checked,
    )
    result.validate()
    return result


def validate_transition_encoding_outputs(transition_df, thresholds: dict[str, Any]) -> tuple[ValidationResult, ...]:
    import polars as pl

    frame = transition_df if isinstance(transition_df, pl.DataFrame) else pl.DataFrame(transition_df)
    results: list[ValidationResult] = []
    required = (
        "country_sector", "year", "next_year", "next_regime_membership", "previous_regime_membership",
        "regime_switch_flag", "transition_state", "transition_confidence",
        "transition_assignment_method", "transition_rule_id",
    )
    missing = [column for column in required if column not in frame.columns]
    results.append(_validation_result("transition_required_columns", len(missing), len(required), f"Missing required columns: {missing}." if missing else "Required transition columns are present.", ValidationLayer.STRUCTURAL_VALIDITY, ValidationSeverity.CRITICAL))
    if missing:
        return tuple(results)
    duplicates = int(frame.group_by(["country_sector", "year"]).len().filter(pl.col("len") > 1)["len"].sum() or 0)
    results.append(_validation_result("transition_unique_country_sector_year", duplicates, frame.height, "country_sector-year rows are unique." if not duplicates else f"Found {duplicates} duplicate rows.", ValidationLayer.STRUCTURAL_VALIDITY, ValidationSeverity.CRITICAL))
    years = sorted(frame["year"].drop_nulls().unique().to_list())
    out_years = [year for year in years if year < HISTORICAL_START_YEAR or year > HISTORICAL_END_YEAR]
    results.append(_validation_result("transition_year_range", len(out_years), len(years), "Transition years are within 1995-2016." if not out_years else f"Out-of-range years: {out_years}.", ValidationLayer.STRUCTURAL_VALIDITY))
    allowed = {state.value for state in TransitionState}
    bad_states = frame.filter(~pl.col("transition_state").is_in(allowed)).height
    results.append(_validation_result("transition_controlled_states", bad_states, frame.height, "Transition states are controlled.", ValidationLayer.STRUCTURAL_VALIDITY))
    bad_conf = frame.filter((pl.col("transition_confidence") < 0) | (pl.col("transition_confidence") > 1) | pl.col("transition_confidence").is_null()).height
    results.append(_validation_result("transition_confidence_bounds", bad_conf, frame.height, "Transition confidence is within [0, 1].", ValidationLayer.MECHANISM_VALIDITY))
    forbidden = [column for column in ("regime_probability", "simulated_output", "scenario_id", "run_id") if column in frame.columns]
    results.append(_validation_result("transition_no_probability_simulation_or_scenario_columns", len(forbidden), 4, f"Forbidden columns present: {forbidden}." if forbidden else "No probability, simulation, or scenario columns are created.", ValidationLayer.STRUCTURAL_VALIDITY, ValidationSeverity.CRITICAL))
    bad_no_next_switch = frame.filter(pl.col("next_year").is_null() & pl.col("regime_switch_flag").is_not_null()).height
    results.append(_validation_result("transition_switch_null_without_next_year", bad_no_next_switch, frame.height, "regime_switch_flag is null without next_year.", ValidationLayer.STRUCTURAL_VALIDITY))
    bad_final = frame.filter((pl.col("year") == HISTORICAL_END_YEAR) & (pl.col("transition_state") != TransitionState.NO_NEXT_YEAR.value)).height
    results.append(_validation_result("transition_final_year_no_next", bad_final, frame.filter(pl.col("year") == HISTORICAL_END_YEAR).height, "Final year rows are NO_NEXT_YEAR.", ValidationLayer.STRUCTURAL_VALIDITY, ValidationSeverity.CRITICAL))
    null_states = frame.filter(pl.col("transition_state").is_null()).height
    results.append(_validation_result("transition_state_non_null", null_states, frame.height, "transition_state is non-null.", ValidationLayer.STRUCTURAL_VALIDITY, ValidationSeverity.CRITICAL))
    has_thresholds = int(not thresholds.get("method") or not thresholds.get("thresholds"))
    results.append(_validation_result("transition_thresholds_available", has_thresholds, 2, "Threshold dictionary includes method and variable thresholds.", ValidationLayer.MECHANISM_VALIDITY, ValidationSeverity.CRITICAL))
    return tuple(results)


def _results_to_dict(results: tuple[ValidationResult, ...]) -> list[dict[str, Any]]:
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


def build_historical_transition_encoding(project_root: Path) -> TransitionEncodingBuildResult:
    import polars as pl

    paths = ABMV5Paths.from_project_root(project_root)
    paths.validate_project_root()
    paths.ensure_directories()
    regime_path = paths.regimes / REGIME_PANEL_FILENAME
    if not regime_path.exists():
        raise FileNotFoundError(f"Historical regime panel missing: {regime_path}. Run Phase 3.1-3.3 first.")
    regime_panel = pl.read_parquet(regime_path)
    transition_base = add_forward_transition_columns(regime_panel)
    thresholds = compute_transition_delta_thresholds(transition_base)
    transition_panel = assign_transition_states(transition_base, thresholds)
    results = validate_transition_encoding_outputs(transition_panel, thresholds)

    transition_panel_path = paths.regimes / TRANSITION_PANEL_FILENAME
    matrix_path = paths.regimes / TRANSITION_MATRIX_FILENAME
    summary_path = paths.regimes / TRANSITION_SUMMARY_FILENAME
    by_year_path = paths.regimes / TRANSITION_BY_YEAR_FILENAME
    by_sector_path = paths.regimes / TRANSITION_BY_SECTOR_FILENAME
    validation_path = paths.validation / TRANSITION_VALIDATION_FILENAME

    transition_panel.write_parquet(transition_panel_path)
    build_regime_transition_matrix(transition_panel).write_csv(matrix_path)
    build_transition_state_summary(transition_panel).write_csv(summary_path)
    build_transition_state_by_year(transition_panel).write_csv(by_year_path)
    build_transition_state_by_sector(transition_panel).write_csv(by_sector_path)
    validation_path.write_text(json.dumps({"validation_scope": "abm_v5_transition_encoding", "thresholds": thresholds, "results": _results_to_dict(results)}, indent=2, sort_keys=True), encoding="utf-8")
    if _has_critical_failures(results):
        raise ValueError(f"Transition encoding validation has critical failures: {validation_path}")
    result = TransitionEncodingBuildResult(
        transition_panel_path=transition_panel_path,
        transition_matrix_path=matrix_path,
        transition_state_summary_path=summary_path,
        transition_state_by_year_path=by_year_path,
        transition_state_by_sector_path=by_sector_path,
        validation_path=validation_path,
        start_year=HISTORICAL_START_YEAR,
        end_year=HISTORICAL_END_YEAR,
        n_rows=transition_panel.height,
        n_agents=transition_panel["country_sector"].n_unique(),
        n_transitions=transition_panel.filter(pl.col("next_year").is_not_null()).height,
    )
    result.validate()
    return result


def load_historical_transition_panel(path: Path):
    import polars as pl

    return pl.read_parquet(path)
