from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from src.abm_v5.config import DEFAULT_HISTORICAL_END_YEAR, DEFAULT_HISTORICAL_START_YEAR, ValidationLayer
from src.abm_v5.paths import ABMV5Paths
from src.abm_v5.phase_space import PHASE_SPACE_OUTPUT_FILENAME
from src.abm_v5.validation import ValidationResult, ValidationSeverity, ValidationStatus


HISTORICAL_START_YEAR = DEFAULT_HISTORICAL_START_YEAR
HISTORICAL_END_YEAR = DEFAULT_HISTORICAL_END_YEAR
REGIME_PANEL_FILENAME = "historical_regime_panel_1995_2016.parquet"
VARIABLE_SELECTION_FILENAME = "regime_variable_selection.json"
THRESHOLDS_FILENAME = "regime_thresholds.json"
PROFILE_SUMMARY_FILENAME = "regime_profile_summary.csv"
SIZE_BY_YEAR_FILENAME = "regime_size_by_year.csv"
COMPOSITION_BY_SECTOR_FILENAME = "regime_composition_by_sector.csv"
COMPOSITION_BY_COUNTRY_FILENAME = "regime_composition_by_country.csv"
VALIDATION_FILENAME = "regime_discovery_validation.json"
THRESHOLD_RULE_ID = "abm_v5_phase3_global_quantile_v1"
ASSIGNMENT_METHOD = "global_interpretable_quantile_rules"

ELIGIBLE_VARIABLES = (
    "emissions_intensity_gap",
    "green_capability",
    "general_capability",
    "network_green_exposure",
    "brown_centrality",
    "supplier_lock_in",
    "local_greenness",
    "import_dependence_proxy",
    "export_dependence_proxy",
    "supplier_concentration_hhi",
    "buyer_concentration_hhi",
)
DESIGN_TARGET_VARIABLES = (
    "capability_density",
    "green_capability_density",
    "ecosystem_proximity",
    "directed_green_precedence",
    "reachable_green_complexity",
    "transition_sector_score",
)


class RegimeDimension(str, Enum):
    EMISSIONS_POSITION = "emissions_position"
    CAPABILITY_POSITION = "capability_position"
    NETWORK_POSITION = "network_position"
    BROWN_CENTRALITY_POSITION = "brown_centrality_position"
    SUPPLIER_LOCK_IN_POSITION = "supplier_lock_in_position"


class RegimeLabel(str, Enum):
    GREEN_CAPABLE_EMBEDDED = "green_capable_embedded"
    GREEN_CAPABLE_CONSTRAINED = "green_capable_constrained"
    BROWN_CENTRAL_CAPABLE = "brown_central_capable"
    BROWN_CENTRAL_CONSTRAINED = "brown_central_constrained"
    DIRTY_CAPABILITY_GAP = "dirty_capability_gap"
    CLEAN_LOW_CAPABILITY = "clean_low_capability"
    LOW_SIGNAL_PERIPHERAL = "low_signal_peripheral"
    MIXED_INTERMEDIATE = "mixed_intermediate"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class RegimeDiscoveryBuildResult:
    regime_panel_path: Path
    variable_selection_path: Path
    thresholds_path: Path
    profile_summary_path: Path
    size_by_year_path: Path
    composition_by_sector_path: Path
    composition_by_country_path: Path
    validation_path: Path
    start_year: int
    end_year: int
    n_rows: int
    n_agents: int
    n_regimes: int

    def validate(self) -> None:
        for field_name in (
            "regime_panel_path",
            "variable_selection_path",
            "thresholds_path",
            "profile_summary_path",
            "size_by_year_path",
            "composition_by_sector_path",
            "composition_by_country_path",
            "validation_path",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must not be empty.")
        if self.n_rows <= 0:
            raise ValueError("n_rows must be positive.")
        if self.n_agents <= 0:
            raise ValueError("n_agents must be positive.")
        if self.n_regimes <= 0:
            raise ValueError("n_regimes must be positive.")
        if self.start_year != DEFAULT_HISTORICAL_START_YEAR:
            raise ValueError("start_year must be 1995.")
        if self.end_year != DEFAULT_HISTORICAL_END_YEAR:
            raise ValueError("end_year must be 2016.")


def select_regime_discovery_variables(df) -> dict[str, Any]:
    import polars as pl

    frame = df if isinstance(df, pl.DataFrame) else pl.DataFrame(df)
    selected: list[str] = []
    excluded: list[str] = []
    reasons: dict[str, str] = {}
    availability: dict[str, float] = {}
    for variable in (*ELIGIBLE_VARIABLES, *DESIGN_TARGET_VARIABLES):
        if variable not in frame.columns:
            availability[variable] = 0.0
            excluded.append(variable)
            reasons[variable] = "missing_column"
            continue
        available_share = (frame.height - frame[variable].null_count()) / frame.height if frame.height else 0.0
        availability[variable] = available_share
        if variable in DESIGN_TARGET_VARIABLES:
            excluded.append(variable)
            reasons[variable] = "design_target_placeholder"
        elif available_share < 0.20:
            excluded.append(variable)
            reasons[variable] = "available_share_below_0.20"
        else:
            selected.append(variable)
    required_present = (
        "emissions_intensity_gap" in selected
        and "network_green_exposure" in selected
        and ("green_capability" in selected or "general_capability" in selected)
    )
    return {
        "selected_variables": selected,
        "excluded_variables": excluded,
        "exclusion_reasons": reasons,
        "availability_by_variable": availability,
        "design_target_variables": list(DESIGN_TARGET_VARIABLES),
        "required_core_variables_present": required_present,
    }


def discover_regime_thresholds(df, selected_variables: list[str]) -> dict[str, Any]:
    import polars as pl

    frame = df if isinstance(df, pl.DataFrame) else pl.DataFrame(df)
    specs = {
        "emissions_intensity_gap": {"low": 0.33, "high": 0.67},
        "green_capability": {"low": 0.33, "high": 0.67},
        "general_capability": {"low": 0.33, "high": 0.67},
        "network_green_exposure": {"low": 0.33, "high": 0.67},
        "brown_centrality": {"low": 0.50, "high": 0.95, "very_high": 0.99},
        "supplier_lock_in": {"low": 0.33, "high": 0.67},
        "local_greenness": {"low": 0.33, "high": 0.67},
    }
    output: dict[str, Any] = {}
    for variable in selected_variables:
        if variable not in specs or variable not in frame.columns:
            continue
        values = frame.select(pl.col(variable).cast(pl.Float64, strict=False)).drop_nulls()
        thresholds = {
            name: float(values.select(pl.col(variable).quantile(q)).item())
            for name, q in specs[variable].items()
            if values.height > 0
        }
        output[variable] = {
            "variable": variable,
            "thresholds": thresholds,
            "n_observations_used": values.height,
            "method": "global_quantile_rule",
            "notes": "Global 1995-2016 interpretable quantile thresholds; descriptive only.",
        }
    return output


def _threshold(thresholds: dict[str, Any], variable: str, key: str) -> float | None:
    value = thresholds.get(variable, {}).get("thresholds", {}).get(key)
    return float(value) if value is not None else None


def assign_interpretable_regimes(df, thresholds: dict[str, Any]):
    import polars as pl

    frame = df if isinstance(df, pl.DataFrame) else pl.DataFrame(df)
    eg_low = _threshold(thresholds, "emissions_intensity_gap", "low")
    eg_high = _threshold(thresholds, "emissions_intensity_gap", "high")
    gc_low = _threshold(thresholds, "green_capability", "low")
    gc_high = _threshold(thresholds, "green_capability", "high")
    gen_low = _threshold(thresholds, "general_capability", "low")
    ng_low = _threshold(thresholds, "network_green_exposure", "low")
    ng_high = _threshold(thresholds, "network_green_exposure", "high")
    bc_high = _threshold(thresholds, "brown_centrality", "high")
    bc_very_high = _threshold(thresholds, "brown_centrality", "very_high")
    sl_low = _threshold(thresholds, "supplier_lock_in", "low")
    sl_high = _threshold(thresholds, "supplier_lock_in", "high")

    def lit(value: float | None):
        return pl.lit(value if value is not None else float("nan"))

    assigned = frame.with_columns(
        (pl.col("emissions_intensity_gap") >= lit(eg_high)).fill_null(False).alias("is_dirty_gap"),
        (pl.col("emissions_intensity_gap") <= lit(eg_low)).fill_null(False).alias("is_clean_gap"),
        (pl.col("green_capability") >= lit(gc_high)).fill_null(False).alias("is_green_capable"),
        (
            (pl.col("green_capability") <= lit(gc_low))
            | (pl.col("green_capability").is_null() & (pl.col("general_capability") <= lit(gen_low)))
        ).fill_null(False).alias("is_low_capability"),
        (pl.col("network_green_exposure") >= lit(ng_high)).fill_null(False).alias("is_network_green"),
        (pl.col("network_green_exposure") <= lit(ng_low)).fill_null(False).alias("is_network_brown"),
        (pl.col("brown_centrality") >= lit(bc_high)).fill_null(False).alias("is_brown_central"),
        (pl.col("brown_centrality") >= lit(bc_very_high)).fill_null(False).alias("is_very_brown_central"),
        (pl.col("supplier_lock_in") >= lit(sl_high)).fill_null(False).alias("is_supplier_locked"),
        (pl.col("supplier_lock_in") <= lit(sl_low)).fill_null(False).alias("is_supplier_flexible"),
    ).with_columns(
        (
            pl.col("emissions_intensity_gap").is_null()
            | pl.col("network_green_exposure").is_null()
            | (pl.col("green_capability").is_null() & pl.col("general_capability").is_null())
        ).alias("_missing_required_core")
    )
    assigned = assigned.with_columns(
        pl.when(pl.col("_missing_required_core"))
        .then(pl.lit(RegimeLabel.INSUFFICIENT_DATA.value))
        .when(pl.col("is_green_capable") & pl.col("is_network_green") & ~pl.col("is_supplier_locked"))
        .then(pl.lit(RegimeLabel.GREEN_CAPABLE_EMBEDDED.value))
        .when(pl.col("is_green_capable") & pl.col("is_supplier_locked"))
        .then(pl.lit(RegimeLabel.GREEN_CAPABLE_CONSTRAINED.value))
        .when(pl.col("is_brown_central") & pl.col("is_green_capable"))
        .then(pl.lit(RegimeLabel.BROWN_CENTRAL_CAPABLE.value))
        .when(pl.col("is_brown_central") & ~pl.col("is_green_capable"))
        .then(pl.lit(RegimeLabel.BROWN_CENTRAL_CONSTRAINED.value))
        .when(pl.col("is_dirty_gap") & pl.col("is_low_capability"))
        .then(pl.lit(RegimeLabel.DIRTY_CAPABILITY_GAP.value))
        .when(pl.col("is_clean_gap") & pl.col("is_low_capability"))
        .then(pl.lit(RegimeLabel.CLEAN_LOW_CAPABILITY.value))
        .when(~pl.col("is_dirty_gap") & ~pl.col("is_green_capable") & ~pl.col("is_brown_central") & ~pl.col("is_supplier_locked"))
        .then(pl.lit(RegimeLabel.LOW_SIGNAL_PERIPHERAL.value))
        .otherwise(pl.lit(RegimeLabel.MIXED_INTERMEDIATE.value))
        .alias("regime_membership")
    ).with_columns(
        pl.when(pl.col("regime_membership") == RegimeLabel.INSUFFICIENT_DATA.value)
        .then(pl.lit(0.25))
        .when(pl.col("regime_membership") == RegimeLabel.MIXED_INTERMEDIATE.value)
        .then(pl.lit(0.5))
        .when(
            pl.col("brown_centrality").is_null()
            | pl.col("supplier_lock_in").is_null()
            | pl.col("green_capability").is_null()
        )
        .then(pl.lit(0.75))
        .otherwise(pl.lit(1.0))
        .alias("regime_confidence"),
        pl.lit(ASSIGNMENT_METHOD).alias("regime_assignment_method"),
        pl.lit(THRESHOLD_RULE_ID).alias("threshold_rule_id"),
    ).drop("_missing_required_core")
    forbidden = [column for column in ("regime_probability", "transition_state", "regime_switch_flag") if column in assigned.columns]
    return assigned.drop(forbidden) if forbidden else assigned


def build_regime_profile_summary(regime_df):
    import polars as pl

    frame = regime_df if isinstance(regime_df, pl.DataFrame) else pl.DataFrame(regime_df)
    total = frame.height
    return frame.group_by("regime_membership").agg(
        pl.len().alias("n_rows"),
        (pl.len() / total).alias("share_rows"),
        pl.n_unique("country_sector").alias("n_agents"),
        *[pl.col(column).cast(pl.Float64, strict=False).mean().alias(f"mean_{column}") for column in (
            "output", "emissions", "emissions_intensity", "emissions_intensity_gap",
            "local_greenness", "network_green_exposure", "green_capability",
            "general_capability", "brown_centrality", "supplier_lock_in", "regime_confidence"
        )],
    ).sort("regime_membership")


def build_regime_size_by_year(regime_df):
    import polars as pl

    frame = regime_df if isinstance(regime_df, pl.DataFrame) else pl.DataFrame(regime_df)
    return frame.group_by(["year", "regime_membership"]).agg(
        pl.len().alias("n_rows"),
        pl.col("regime_confidence").mean().alias("mean_regime_confidence"),
    ).with_columns(
        (pl.col("n_rows") / pl.sum("n_rows").over("year")).alias("share_year_rows")
    ).select("year", "regime_membership", "n_rows", "share_year_rows", "mean_regime_confidence").sort(["year", "regime_membership"])


def build_regime_composition_by_sector(regime_df):
    return _build_composition(regime_df, "sector")


def build_regime_composition_by_country(regime_df):
    return _build_composition(regime_df, "country")


def _build_composition(regime_df, group_column: str):
    import polars as pl

    frame = regime_df if isinstance(regime_df, pl.DataFrame) else pl.DataFrame(regime_df)
    return frame.group_by(["regime_membership", group_column]).agg(
        pl.len().alias("n_rows"),
        pl.col("output").cast(pl.Float64, strict=False).mean().alias("mean_output"),
        pl.col("emissions").cast(pl.Float64, strict=False).mean().alias("mean_emissions"),
    ).with_columns(
        (pl.col("n_rows") / pl.sum("n_rows").over("regime_membership")).alias("share_within_regime")
    ).select("regime_membership", group_column, "n_rows", "share_within_regime", "mean_output", "mean_emissions")


def _result(check_name: str, failed: int, checked: int, message: str, layer: ValidationLayer, severity: ValidationSeverity = ValidationSeverity.ERROR) -> ValidationResult:
    result = ValidationResult(
        check_name=check_name,
        status=ValidationStatus.FAILED if failed else ValidationStatus.PASSED,
        severity=severity if failed else ValidationSeverity.INFO,
        message=message,
        layer=layer,
        n_failed=failed,
        n_checked=checked,
    )
    result.validate()
    return result


def validate_regime_discovery_outputs(regime_df, thresholds: dict[str, Any], variable_selection: dict[str, Any]) -> tuple[ValidationResult, ...]:
    import polars as pl

    frame = regime_df if isinstance(regime_df, pl.DataFrame) else pl.DataFrame(regime_df)
    results: list[ValidationResult] = []
    required = ("country_sector", "year", "regime_membership", "regime_confidence", "regime_assignment_method", "threshold_rule_id")
    missing = [column for column in required if column not in frame.columns]
    results.append(_result("regime_required_columns", len(missing), len(required), f"Missing required columns: {missing}." if missing else "Required regime columns are present.", ValidationLayer.STRUCTURAL_VALIDITY, ValidationSeverity.CRITICAL))
    if missing:
        return tuple(results)
    duplicates = int(frame.group_by(["country_sector", "year"]).len().filter(pl.col("len") > 1)["len"].sum() or 0)
    results.append(_result("regime_unique_country_sector_year", duplicates, frame.height, "country_sector-year rows are unique." if not duplicates else f"Found {duplicates} duplicate rows.", ValidationLayer.STRUCTURAL_VALIDITY, ValidationSeverity.CRITICAL))
    years = sorted(frame["year"].drop_nulls().unique().to_list())
    out_years = [year for year in years if year < HISTORICAL_START_YEAR or year > HISTORICAL_END_YEAR]
    results.append(_result("regime_year_range", len(out_years), len(years), "Regime years are within 1995-2016." if not out_years else f"Out-of-range years: {out_years}.", ValidationLayer.STRUCTURAL_VALIDITY))
    labels = {label.value for label in RegimeLabel}
    bad_labels = frame.filter(~pl.col("regime_membership").is_in(labels)).height
    results.append(_result("regime_controlled_labels", bad_labels, frame.height, "Regime labels are controlled.", ValidationLayer.STRUCTURAL_VALIDITY))
    bad_conf = frame.filter((pl.col("regime_confidence") < 0) | (pl.col("regime_confidence") > 1) | pl.col("regime_confidence").is_null()).height
    results.append(_result("regime_confidence_bounds", bad_conf, frame.height, "Regime confidence is within [0, 1].", ValidationLayer.MECHANISM_VALIDITY))
    forbidden = [column for column in ("regime_probability", "transition_state") if column in frame.columns]
    if "regime_switch_flag" in frame.columns and frame["regime_switch_flag"].null_count() != frame.height:
        forbidden.append("regime_switch_flag")
    results.append(_result("regime_no_transition_or_probability_outputs", len(forbidden), 3, f"Forbidden Phase 3.1-3.3 columns present: {forbidden}." if forbidden else "No regime probabilities or transition states are created.", ValidationLayer.STRUCTURAL_VALIDITY, ValidationSeverity.CRITICAL))
    selected = set(variable_selection.get("selected_variables", []))
    design_selected = sorted(selected.intersection(DESIGN_TARGET_VARIABLES))
    results.append(_result("regime_no_design_target_selected_variables", len(design_selected), len(selected), f"Design-target variables selected: {design_selected}." if design_selected else "Selected variables exclude design-target placeholders.", ValidationLayer.MECHANISM_VALIDITY, ValidationSeverity.CRITICAL))
    missing_thresholds = [variable for variable in selected if variable in {"emissions_intensity_gap", "green_capability", "network_green_exposure", "brown_centrality", "supplier_lock_in", "local_greenness"} and variable not in thresholds]
    results.append(_result("regime_thresholds_present", len(missing_thresholds), len(selected), f"Missing thresholds: {missing_thresholds}." if missing_thresholds else "Thresholds are present for selected rule variables.", ValidationLayer.MECHANISM_VALIDITY, ValidationSeverity.CRITICAL))
    insufficient_share = frame.filter(pl.col("regime_membership") == RegimeLabel.INSUFFICIENT_DATA.value).height / frame.height if frame.height else 0.0
    results.append(_result("regime_insufficient_data_share", int(insufficient_share > 0.5), frame.height, f"INSUFFICIENT_DATA share is {insufficient_share:.3f}.", ValidationLayer.STRUCTURAL_VALIDITY, ValidationSeverity.WARNING))
    return tuple(results)


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(value) for value in obj]
    return obj


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


def build_historical_regime_discovery(project_root: Path) -> RegimeDiscoveryBuildResult:
    import polars as pl

    paths = ABMV5Paths.from_project_root(project_root)
    paths.validate_project_root()
    paths.ensure_directories()
    phase_space_path = paths.phase_space / PHASE_SPACE_OUTPUT_FILENAME
    if not phase_space_path.exists():
        raise FileNotFoundError(f"Phase-space panel missing: {phase_space_path}. Run Phase 2.6 first.")
    panel = pl.read_parquet(phase_space_path)
    variable_selection = select_regime_discovery_variables(panel)
    thresholds = discover_regime_thresholds(panel, variable_selection["selected_variables"])
    regime_panel = assign_interpretable_regimes(panel, thresholds)
    results = validate_regime_discovery_outputs(regime_panel, thresholds, variable_selection)

    variable_selection_path = paths.regimes / VARIABLE_SELECTION_FILENAME
    thresholds_path = paths.regimes / THRESHOLDS_FILENAME
    regime_panel_path = paths.regimes / REGIME_PANEL_FILENAME
    profile_path = paths.regimes / PROFILE_SUMMARY_FILENAME
    size_path = paths.regimes / SIZE_BY_YEAR_FILENAME
    sector_path = paths.regimes / COMPOSITION_BY_SECTOR_FILENAME
    country_path = paths.regimes / COMPOSITION_BY_COUNTRY_FILENAME
    validation_path = paths.validation / VALIDATION_FILENAME

    variable_selection_path.write_text(json.dumps(_to_jsonable(variable_selection), indent=2, sort_keys=True), encoding="utf-8")
    thresholds_path.write_text(json.dumps(_to_jsonable(thresholds), indent=2, sort_keys=True), encoding="utf-8")
    regime_panel.write_parquet(regime_panel_path)
    build_regime_profile_summary(regime_panel).write_csv(profile_path)
    build_regime_size_by_year(regime_panel).write_csv(size_path)
    build_regime_composition_by_sector(regime_panel).write_csv(sector_path)
    build_regime_composition_by_country(regime_panel).write_csv(country_path)
    validation_path.write_text(json.dumps({"validation_scope": "abm_v5_regime_discovery", "results": _validation_to_dict(results)}, indent=2, sort_keys=True), encoding="utf-8")
    if _has_critical_failures(results):
        raise ValueError(f"Regime discovery validation has critical failures: {validation_path}")
    result = RegimeDiscoveryBuildResult(
        regime_panel_path=regime_panel_path,
        variable_selection_path=variable_selection_path,
        thresholds_path=thresholds_path,
        profile_summary_path=profile_path,
        size_by_year_path=size_path,
        composition_by_sector_path=sector_path,
        composition_by_country_path=country_path,
        validation_path=validation_path,
        start_year=HISTORICAL_START_YEAR,
        end_year=HISTORICAL_END_YEAR,
        n_rows=regime_panel.height,
        n_agents=regime_panel["country_sector"].n_unique(),
        n_regimes=regime_panel["regime_membership"].n_unique(),
    )
    result.validate()
    return result


def load_historical_regime_panel(path: Path):
    import polars as pl

    return pl.read_parquet(path)
