from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from src.abm_v5.config import DEFAULT_HISTORICAL_END_YEAR, DEFAULT_HISTORICAL_START_YEAR, ValidationLayer
from src.abm_v5.paths import ABMV5Paths
from src.abm_v5.regimes import REGIME_PANEL_FILENAME, RegimeLabel
from src.abm_v5.transitions import TRANSITION_PANEL_FILENAME, TransitionState
from src.abm_v5.validation import ValidationResult, ValidationSeverity, ValidationStatus


HISTORICAL_START_YEAR = DEFAULT_HISTORICAL_START_YEAR
HISTORICAL_END_YEAR = DEFAULT_HISTORICAL_END_YEAR
HANDOFF_PANEL_FILENAME = "regime_handoff_panel_1995_2016.parquet"
MECHANISM_TARGET_CANDIDATES_FILENAME = "mechanism_target_candidates.csv"
TRANSITION_TARGET_SUMMARY_FILENAME = "transition_target_summary.csv"
REGIME_STABILITY_SUMMARY_FILENAME = "regime_stability_summary.csv"
MECHANISM_LEARNING_SAMPLE_SUMMARY_FILENAME = "mechanism_learning_sample_summary.csv"
REGIME_HANDOFF_VALIDATION_FILENAME = "regime_handoff_validation.json"


class MechanismTargetFamily(str, Enum):
    EMISSIONS_INTENSITY = "emissions_intensity"
    NETWORK_EMBEDDING = "network_embedding"
    CAPABILITY = "capability"
    SUPPLIER_LOCK_IN = "supplier_lock_in"
    BROWN_CENTRALITY = "brown_centrality"
    REGIME_PERSISTENCE = "regime_persistence"
    REGIME_SWITCHING = "regime_switching"
    DATA_LIMITATION = "data_limitation"


class MechanismTargetStatus(str, Enum):
    PHASE4_TARGET = "phase4_target"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    DATA_LIMITED = "data_limited"
    RARE_EVENT = "rare_event"
    EXCLUDE_FROM_MECHANISM_LEARNING = "exclude_from_mechanism_learning"


@dataclass(frozen=True)
class RegimeHandoffBuildResult:
    handoff_panel_path: Path
    mechanism_target_candidates_path: Path
    transition_target_summary_path: Path
    regime_stability_summary_path: Path
    mechanism_learning_sample_summary_path: Path
    validation_path: Path
    start_year: int
    end_year: int
    n_rows: int
    n_agents: int
    n_mechanism_target_candidates: int

    def validate(self) -> None:
        for field_name in (
            "handoff_panel_path",
            "mechanism_target_candidates_path",
            "transition_target_summary_path",
            "regime_stability_summary_path",
            "mechanism_learning_sample_summary_path",
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


def build_regime_handoff_panel(regime_df, transition_df):
    import polars as pl

    regime = regime_df if isinstance(regime_df, pl.DataFrame) else pl.DataFrame(regime_df)
    transition = transition_df if isinstance(transition_df, pl.DataFrame) else pl.DataFrame(transition_df)
    transition_extra = [
        column
        for column in transition.columns
        if column not in regime.columns or column in {"country_sector", "year"}
    ]
    frame = regime.join(
        transition.select(*transition_extra),
        on=["country_sector", "year"],
        how="left",
    )
    if "accounting_ei_valid_flag" not in frame.columns:
        frame = frame.with_columns(pl.lit(True).alias("accounting_ei_valid_flag"))
    eligible = (
        (pl.col("transition_state") != TransitionState.NO_NEXT_YEAR.value)
        & (pl.col("transition_state") != TransitionState.INSUFFICIENT_DATA_TRANSITION.value)
        & (pl.col("regime_membership") != RegimeLabel.INSUFFICIENT_DATA.value)
        & (pl.col("next_regime_membership") != RegimeLabel.INSUFFICIENT_DATA.value)
        & (pl.col("transition_confidence") >= 0.5)
        & (pl.col("accounting_ei_valid_flag") == True)
    )
    return frame.with_columns(
        eligible.fill_null(False).alias("mechanism_learning_eligible_flag")
    ).with_columns(
        pl.when(pl.col("transition_state") == TransitionState.NO_NEXT_YEAR.value)
        .then(pl.lit("no_next_year"))
        .when(
            (pl.col("transition_state") == TransitionState.INSUFFICIENT_DATA_TRANSITION.value)
            | (pl.col("regime_membership") == RegimeLabel.INSUFFICIENT_DATA.value)
            | (pl.col("next_regime_membership") == RegimeLabel.INSUFFICIENT_DATA.value)
        )
        .then(pl.lit("insufficient_data"))
        .when(pl.col("transition_confidence") < 0.5)
        .then(pl.lit("low_transition_confidence"))
        .when(pl.col("accounting_ei_valid_flag") != True)
        .then(pl.lit("invalid_emissions_intensity"))
        .otherwise(pl.lit("eligible"))
        .alias("mechanism_learning_exclusion_reason"),
        (pl.col("regime_membership") != RegimeLabel.INSUFFICIENT_DATA.value).fill_null(False).alias("regime_is_observed_state_flag"),
        (~pl.col("transition_state").is_in([TransitionState.NO_NEXT_YEAR.value, TransitionState.INSUFFICIENT_DATA_TRANSITION.value])).fill_null(False).alias("transition_is_observed_movement_flag"),
        (pl.col("delta_emissions_intensity_gap").is_not_null() & pl.col("emissions_intensity_gap").is_not_null()).alias("usable_for_emissions_mechanism_flag"),
        (pl.col("delta_network_green_exposure").is_not_null() & pl.col("network_green_exposure").is_not_null()).alias("usable_for_network_mechanism_flag"),
        (pl.col("delta_green_capability").is_not_null() | pl.col("delta_general_capability").is_not_null()).alias("usable_for_capability_mechanism_flag"),
        (pl.col("delta_supplier_lock_in").is_not_null() & pl.col("supplier_lock_in").is_not_null()).alias("usable_for_supplier_lock_in_mechanism_flag"),
        (pl.col("delta_brown_centrality").is_not_null() & pl.col("brown_centrality").is_not_null()).alias("usable_for_brown_centrality_mechanism_flag"),
    ).drop([column for column in ("regime_probability", "scenario_id", "run_id") if column in frame.columns])


def build_regime_stability_summary(handoff_df):
    import polars as pl

    frame = handoff_df if isinstance(handoff_df, pl.DataFrame) else pl.DataFrame(handoff_df)
    denominator = (
        (pl.col("next_year").is_not_null())
        & (pl.col("transition_state") != TransitionState.INSUFFICIENT_DATA_TRANSITION.value)
    ).cast(pl.Int64).sum()
    return frame.group_by("regime_membership").agg(
        pl.len().alias("n_rows"),
        pl.n_unique("country_sector").alias("n_agents"),
        pl.col("mechanism_learning_eligible_flag").cast(pl.Int64).sum().alias("n_eligible_rows"),
        pl.col("mechanism_learning_eligible_flag").cast(pl.Float64).mean().alias("eligible_share"),
        (pl.col("transition_state") == TransitionState.STABLE_SAME_REGIME.value).cast(pl.Int64).sum().alias("stable_same_regime_rows"),
        (pl.col("transition_state") == TransitionState.REGIME_SWITCH.value).cast(pl.Int64).sum().alias("regime_switch_rows"),
        denominator.alias("_denominator"),
        pl.col("regime_confidence").mean().alias("mean_regime_confidence"),
        pl.col("transition_confidence").mean().alias("mean_transition_confidence"),
        *[pl.col(column).cast(pl.Float64, strict=False).mean().alias(f"mean_{column}") for column in (
            "output", "emissions", "network_green_exposure", "green_capability",
            "supplier_lock_in", "brown_centrality"
        )],
    ).with_columns(
        pl.when(pl.col("_denominator") > 0).then(pl.col("stable_same_regime_rows") / pl.col("_denominator")).otherwise(None).alias("stability_rate"),
        pl.when(pl.col("_denominator") > 0).then(pl.col("regime_switch_rows") / pl.col("_denominator")).otherwise(None).alias("switch_rate"),
    ).drop("_denominator")


def _family_for_state(state: str) -> MechanismTargetFamily:
    mapping = {
        TransitionState.GREEN_EMBEDDING_IMPROVEMENT.value: MechanismTargetFamily.NETWORK_EMBEDDING,
        TransitionState.GREEN_EMBEDDING_DETERIORATION.value: MechanismTargetFamily.NETWORK_EMBEDDING,
        TransitionState.CAPABILITY_GAIN.value: MechanismTargetFamily.CAPABILITY,
        TransitionState.CAPABILITY_LOSS.value: MechanismTargetFamily.CAPABILITY,
        TransitionState.SUPPLIER_LOCK_IN_INCREASE.value: MechanismTargetFamily.SUPPLIER_LOCK_IN,
        TransitionState.SUPPLIER_LOCK_IN_RELIEF.value: MechanismTargetFamily.SUPPLIER_LOCK_IN,
        TransitionState.DIRTY_GAP_IMPROVEMENT.value: MechanismTargetFamily.EMISSIONS_INTENSITY,
        TransitionState.DIRTY_GAP_WORSENING.value: MechanismTargetFamily.EMISSIONS_INTENSITY,
        TransitionState.STABLE_SAME_REGIME.value: MechanismTargetFamily.REGIME_PERSISTENCE,
        TransitionState.REGIME_SWITCH.value: MechanismTargetFamily.REGIME_SWITCHING,
        TransitionState.INSUFFICIENT_DATA_TRANSITION.value: MechanismTargetFamily.DATA_LIMITATION,
        TransitionState.NO_NEXT_YEAR.value: MechanismTargetFamily.DATA_LIMITATION,
    }
    return mapping.get(state, MechanismTargetFamily.DATA_LIMITATION)


def _status_for_state(state: str, n_eligible: int, eligible_share: float) -> MechanismTargetStatus:
    if eligible_share == 0:
        return MechanismTargetStatus.EXCLUDE_FROM_MECHANISM_LEARNING
    if state in {TransitionState.INSUFFICIENT_DATA_TRANSITION.value, TransitionState.NO_NEXT_YEAR.value}:
        return MechanismTargetStatus.DATA_LIMITED
    if state == TransitionState.MIXED_MOVEMENT.value:
        return MechanismTargetStatus.DIAGNOSTIC_ONLY
    if n_eligible >= 500:
        return MechanismTargetStatus.PHASE4_TARGET
    return MechanismTargetStatus.RARE_EVENT


def _recommended_use(status: MechanismTargetStatus) -> str:
    return {
        MechanismTargetStatus.PHASE4_TARGET: "learn mechanism target",
        MechanismTargetStatus.DIAGNOSTIC_ONLY: "diagnostic validation only",
        MechanismTargetStatus.RARE_EVENT: "diagnostic validation only",
        MechanismTargetStatus.DATA_LIMITED: "insufficient data boundary condition",
        MechanismTargetStatus.EXCLUDE_FROM_MECHANISM_LEARNING: "exclude from mechanism learning",
    }[status]


def build_transition_target_summary(handoff_df):
    import polars as pl

    frame = handoff_df if isinstance(handoff_df, pl.DataFrame) else pl.DataFrame(handoff_df)
    grouped = frame.group_by("transition_state").agg(
        pl.len().alias("n_rows"),
        pl.col("mechanism_learning_eligible_flag").cast(pl.Int64).sum().alias("n_eligible_rows"),
        pl.col("mechanism_learning_eligible_flag").cast(pl.Float64).mean().alias("eligible_share"),
        pl.col("transition_confidence").mean().alias("mean_transition_confidence"),
        *[pl.col(column).cast(pl.Float64, strict=False).mean().alias(f"mean_{column}") for column in (
            "delta_emissions_intensity_gap", "delta_network_green_exposure",
            "delta_green_capability", "delta_general_capability",
            "delta_supplier_lock_in", "delta_brown_centrality"
        )],
    ).with_columns((pl.col("n_rows") / frame.height).alias("share_rows"))
    rows = []
    for row in grouped.to_dicts():
        family = _family_for_state(row["transition_state"])
        status = _status_for_state(row["transition_state"], int(row["n_eligible_rows"]), float(row["eligible_share"] or 0.0))
        rows.append({
            **row,
            "mechanism_target_family": family.value,
            "target_status": status.value,
            "recommended_phase4_use": _recommended_use(status),
        })
    return pl.DataFrame(rows).select(
        "transition_state", "mechanism_target_family", "target_status", "n_rows", "share_rows",
        "n_eligible_rows", "eligible_share", "mean_transition_confidence",
        "mean_delta_emissions_intensity_gap", "mean_delta_network_green_exposure",
        "mean_delta_green_capability", "mean_delta_general_capability",
        "mean_delta_supplier_lock_in", "mean_delta_brown_centrality", "recommended_phase4_use",
    )


def build_mechanism_target_candidates(handoff_df, transition_target_summary_df):
    import polars as pl

    frame = handoff_df if isinstance(handoff_df, pl.DataFrame) else pl.DataFrame(handoff_df)
    summary = transition_target_summary_df if isinstance(transition_target_summary_df, pl.DataFrame) else pl.DataFrame(transition_target_summary_df)
    mapping = {
        MechanismTargetFamily.EMISSIONS_INTENSITY.value: ("emissions_intensity_gap", "delta_emissions_intensity_gap"),
        MechanismTargetFamily.NETWORK_EMBEDDING.value: ("network_green_exposure", "delta_network_green_exposure"),
        MechanismTargetFamily.CAPABILITY.value: ("green_capability", "delta_green_capability"),
        MechanismTargetFamily.SUPPLIER_LOCK_IN.value: ("supplier_lock_in", "delta_supplier_lock_in"),
        MechanismTargetFamily.BROWN_CENTRALITY.value: ("brown_centrality", "delta_brown_centrality"),
        MechanismTargetFamily.REGIME_PERSISTENCE.value: ("regime_membership", "regime_switch_flag"),
        MechanismTargetFamily.REGIME_SWITCHING.value: ("regime_membership", "regime_switch_flag"),
    }
    rows = []
    included = summary.filter(pl.col("target_status").is_in([
        MechanismTargetStatus.PHASE4_TARGET.value,
        MechanismTargetStatus.DIAGNOSTIC_ONLY.value,
    ]))
    for row in included.to_dicts():
        family = row["mechanism_target_family"]
        if family not in mapping:
            continue
        target, delta = mapping[family]
        subset = frame.filter((pl.col("transition_state") == row["transition_state"]) & pl.col("mechanism_learning_eligible_flag"))
        mean_delta = subset.select(pl.col(delta).cast(pl.Float64, strict=False).mean()).item() if delta in subset.columns and subset.height else None
        median_delta = subset.select(pl.col(delta).cast(pl.Float64, strict=False).median()).item() if delta in subset.columns and subset.height else None
        rows.append({
            "mechanism_target_family": family,
            "transition_state": row["transition_state"],
            "target_status": row["target_status"],
            "target_variable": target,
            "source_delta_variable": delta,
            "n_eligible_rows": row["n_eligible_rows"],
            "mean_delta": mean_delta,
            "median_delta": median_delta,
            "recommended_phase4_use": row["recommended_phase4_use"],
            "notes": "Candidate target only; Phase 4 must decide whether to implement a mechanism.",
        })
    schema = {
        "mechanism_target_family": pl.Utf8,
        "transition_state": pl.Utf8,
        "target_status": pl.Utf8,
        "target_variable": pl.Utf8,
        "source_delta_variable": pl.Utf8,
        "n_eligible_rows": pl.Int64,
        "mean_delta": pl.Float64,
        "median_delta": pl.Float64,
        "recommended_phase4_use": pl.Utf8,
        "notes": pl.Utf8,
    }
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def build_mechanism_learning_sample_summary(handoff_df):
    import polars as pl

    frame = handoff_df if isinstance(handoff_df, pl.DataFrame) else pl.DataFrame(handoff_df)
    flags = {
        MechanismTargetFamily.EMISSIONS_INTENSITY.value: "usable_for_emissions_mechanism_flag",
        MechanismTargetFamily.NETWORK_EMBEDDING.value: "usable_for_network_mechanism_flag",
        MechanismTargetFamily.CAPABILITY.value: "usable_for_capability_mechanism_flag",
        MechanismTargetFamily.SUPPLIER_LOCK_IN.value: "usable_for_supplier_lock_in_mechanism_flag",
        MechanismTargetFamily.BROWN_CENTRALITY.value: "usable_for_brown_centrality_mechanism_flag",
    }
    rows = []
    years = frame["year"].drop_nulls()
    for family, flag in flags.items():
        usable = frame.filter(pl.col(flag) & pl.col("mechanism_learning_eligible_flag"))
        exclusions = frame.filter(~pl.col("mechanism_learning_eligible_flag")).group_by("mechanism_learning_exclusion_reason").len().sort("len", descending=True)
        main_reason = exclusions["mechanism_learning_exclusion_reason"][0] if exclusions.height else "eligible"
        rows.append({
            "mechanism_target_family": family,
            "n_usable_rows": usable.height,
            "n_agents": usable["country_sector"].n_unique() if usable.height else 0,
            "start_year": int(years.min()) if not years.is_empty() else None,
            "end_year": int(years.max()) if not years.is_empty() else None,
            "share_of_total_rows": usable.height / frame.height if frame.height else 0.0,
            "main_exclusion_reason": main_reason,
            "notes": "Rows are descriptive historical samples, not mechanism rules.",
        })
    return pl.DataFrame(rows)


def _result(name: str, failed: int, checked: int, message: str, layer: ValidationLayer, severity: ValidationSeverity = ValidationSeverity.ERROR) -> ValidationResult:
    result = ValidationResult(name, ValidationStatus.FAILED if failed else ValidationStatus.PASSED, severity if failed else ValidationSeverity.INFO, message, layer, failed, checked)
    result.validate()
    return result


def validate_regime_handoff_outputs(handoff_df, target_candidates_df, transition_target_summary_df) -> tuple[ValidationResult, ...]:
    import polars as pl

    handoff = handoff_df if isinstance(handoff_df, pl.DataFrame) else pl.DataFrame(handoff_df)
    candidates = target_candidates_df if isinstance(target_candidates_df, pl.DataFrame) else pl.DataFrame(target_candidates_df)
    summary = transition_target_summary_df if isinstance(transition_target_summary_df, pl.DataFrame) else pl.DataFrame(transition_target_summary_df)
    required = (
        "country_sector", "year", "mechanism_learning_eligible_flag",
        "mechanism_learning_exclusion_reason", "regime_is_observed_state_flag",
        "transition_is_observed_movement_flag",
    )
    missing = [column for column in required if column not in handoff.columns]
    results = [_result("handoff_required_columns", len(missing), len(required), f"Missing handoff columns: {missing}." if missing else "Required handoff columns are present.", ValidationLayer.STRUCTURAL_VALIDITY, ValidationSeverity.CRITICAL)]
    if missing:
        return tuple(results)
    duplicates = int(handoff.group_by(["country_sector", "year"]).len().filter(pl.col("len") > 1)["len"].sum() or 0)
    results.append(_result("handoff_unique_country_sector_year", duplicates, handoff.height, "country_sector-year rows are unique.", ValidationLayer.STRUCTURAL_VALIDITY, ValidationSeverity.CRITICAL))
    years = sorted(handoff["year"].drop_nulls().unique().to_list())
    out_years = [year for year in years if year < HISTORICAL_START_YEAR or year > HISTORICAL_END_YEAR]
    results.append(_result("handoff_year_range", len(out_years), len(years), "Handoff years are within 1995-2016.", ValidationLayer.STRUCTURAL_VALIDITY))
    non_bool = handoff.filter(~pl.col("mechanism_learning_eligible_flag").is_in([True, False])).height
    results.append(_result("handoff_eligible_flag_boolean", non_bool, handoff.height, "Eligibility flag is boolean-like.", ValidationLayer.STRUCTURAL_VALIDITY))
    bad_no_next = handoff.filter((pl.col("transition_state") == TransitionState.NO_NEXT_YEAR.value) & pl.col("mechanism_learning_eligible_flag")).height
    bad_insufficient = handoff.filter((pl.col("transition_state") == TransitionState.INSUFFICIENT_DATA_TRANSITION.value) & pl.col("mechanism_learning_eligible_flag")).height
    results.append(_result("handoff_no_next_not_eligible", bad_no_next, handoff.height, "NO_NEXT_YEAR rows are not eligible.", ValidationLayer.MECHANISM_VALIDITY))
    results.append(_result("handoff_insufficient_not_eligible", bad_insufficient, handoff.height, "Insufficient-data transitions are not eligible.", ValidationLayer.MECHANISM_VALIDITY))
    forbidden = [column for column in ("regime_probability", "scenario_id", "run_id") if column in handoff.columns]
    results.append(_result("handoff_no_probability_or_scenario_columns", len(forbidden), 3, f"Forbidden columns present: {forbidden}." if forbidden else "No probability or scenario columns are created.", ValidationLayer.STRUCTURAL_VALIDITY, ValidationSeverity.CRITICAL))
    families = {family.value for family in MechanismTargetFamily}
    statuses = {status.value for status in MechanismTargetStatus}
    bad_families = candidates.filter(~pl.col("mechanism_target_family").is_in(families)).height if candidates.height else 0
    bad_statuses = summary.filter(~pl.col("target_status").is_in(statuses)).height if summary.height else 0
    data_limited_targets = summary.filter((pl.col("mechanism_target_family") == MechanismTargetFamily.DATA_LIMITATION.value) & (pl.col("target_status") == MechanismTargetStatus.PHASE4_TARGET.value)).height if summary.height else 0
    results.append(_result("handoff_candidate_families_controlled", bad_families, candidates.height, "Target candidate families are controlled.", ValidationLayer.MECHANISM_VALIDITY))
    results.append(_result("handoff_target_statuses_controlled", bad_statuses, summary.height, "Target statuses are controlled.", ValidationLayer.MECHANISM_VALIDITY))
    results.append(_result("handoff_data_limited_not_phase4_target", data_limited_targets, summary.height, "DATA_LIMITED transitions are not PHASE4_TARGET.", ValidationLayer.MECHANISM_VALIDITY, ValidationSeverity.CRITICAL))
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


def build_regime_handoff(project_root: Path) -> RegimeHandoffBuildResult:
    import polars as pl

    paths = ABMV5Paths.from_project_root(project_root)
    paths.validate_project_root()
    paths.ensure_directories()
    regime_path = paths.regimes / REGIME_PANEL_FILENAME
    transition_path = paths.regimes / TRANSITION_PANEL_FILENAME
    if not regime_path.exists():
        raise FileNotFoundError(f"Historical regime panel missing: {regime_path}. Run Phase 3.1-3.3 first.")
    if not transition_path.exists():
        raise FileNotFoundError(f"Historical transition panel missing: {transition_path}. Run Phase 3.4 first.")
    regime = pl.read_parquet(regime_path)
    transition = pl.read_parquet(transition_path)
    handoff = build_regime_handoff_panel(regime, transition)
    stability = build_regime_stability_summary(handoff)
    transition_summary = build_transition_target_summary(handoff)
    candidates = build_mechanism_target_candidates(handoff, transition_summary)
    sample_summary = build_mechanism_learning_sample_summary(handoff)
    results = validate_regime_handoff_outputs(handoff, candidates, transition_summary)

    handoff_path = paths.regimes / HANDOFF_PANEL_FILENAME
    candidates_path = paths.regimes / MECHANISM_TARGET_CANDIDATES_FILENAME
    transition_summary_path = paths.regimes / TRANSITION_TARGET_SUMMARY_FILENAME
    stability_path = paths.regimes / REGIME_STABILITY_SUMMARY_FILENAME
    sample_summary_path = paths.regimes / MECHANISM_LEARNING_SAMPLE_SUMMARY_FILENAME
    validation_path = paths.validation / REGIME_HANDOFF_VALIDATION_FILENAME
    handoff.write_parquet(handoff_path)
    candidates.write_csv(candidates_path)
    transition_summary.write_csv(transition_summary_path)
    stability.write_csv(stability_path)
    sample_summary.write_csv(sample_summary_path)
    validation_path.write_text(json.dumps({"validation_scope": "abm_v5_regime_handoff", "results": _validation_to_dict(results)}, indent=2, sort_keys=True), encoding="utf-8")
    if _has_critical_failures(results):
        raise ValueError(f"Regime handoff validation has critical failures: {validation_path}")
    result = RegimeHandoffBuildResult(
        handoff_panel_path=handoff_path,
        mechanism_target_candidates_path=candidates_path,
        transition_target_summary_path=transition_summary_path,
        regime_stability_summary_path=stability_path,
        mechanism_learning_sample_summary_path=sample_summary_path,
        validation_path=validation_path,
        start_year=HISTORICAL_START_YEAR,
        end_year=HISTORICAL_END_YEAR,
        n_rows=handoff.height,
        n_agents=handoff["country_sector"].n_unique(),
        n_mechanism_target_candidates=candidates.height,
    )
    result.validate()
    return result


def load_regime_handoff_panel(path: Path):
    import polars as pl

    return pl.read_parquet(path)
