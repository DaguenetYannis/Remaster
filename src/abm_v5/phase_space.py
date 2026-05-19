from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.abm_v5.accounting import ACCOUNTING_OUTPUT_FILENAME
from src.abm_v5.capabilities import CAPABILITY_OUTPUT_FILENAME
from src.abm_v5.config import DEFAULT_HISTORICAL_END_YEAR, DEFAULT_HISTORICAL_START_YEAR, ValidationLayer
from src.abm_v5.networks import NETWORK_OUTPUT_FILENAME
from src.abm_v5.paths import ABMV5Paths
from src.abm_v5.validation import ValidationResult, ValidationSeverity, ValidationStatus


HISTORICAL_START_YEAR = DEFAULT_HISTORICAL_START_YEAR
HISTORICAL_END_YEAR = DEFAULT_HISTORICAL_END_YEAR
PHASE_SPACE_OUTPUT_FILENAME = "historical_phase_space_panel_1995_2016.parquet"
PHASE_SPACE_VALIDATION_FILENAME = "phase_space_validation.json"
PHASE_SPACE_MISSINGNESS_FILENAME = "phase_space_missingness_summary.csv"
PHASE_SPACE_COVERAGE_FILENAME = "phase_space_variable_coverage_summary.json"

IDENTITY_COLUMNS = ("country_sector", "country", "country_detail", "category", "sector", "year")
ACCOUNTING_COLUMNS = (
    "output",
    "final_demand",
    "emissions",
    "emissions_intensity",
    "local_greenness",
    "accounting_output_positive_flag",
    "accounting_emissions_nonnegative_flag",
    "accounting_ei_valid_flag",
)
CAPABILITY_COLUMNS = (
    "general_capability",
    "green_capability",
    "capability_density",
    "green_capability_density",
    "ecosystem_proximity",
    "directed_green_precedence",
    "reachable_green_complexity",
    "transition_sector_score",
)
CAPABILITY_FLAG_COLUMNS = tuple(f"{column}_available_flag" for column in CAPABILITY_COLUMNS)
NETWORK_COLUMNS = (
    "supplier_count",
    "buyer_count",
    "total_inputs_from_suppliers",
    "total_outputs_to_buyers",
    "supplier_concentration_hhi",
    "buyer_concentration_hhi",
    "import_dependence_proxy",
    "export_dependence_proxy",
    "network_green_exposure",
    "incoming_network_green_exposure",
    "outgoing_network_green_exposure",
    "brown_centrality",
    "supplier_lock_in",
)
CORE_PHASE_SPACE_COLUMNS = (
    "emissions_intensity_gap",
    "phase_space_position",
    "phase_space_empirical_completeness",
    "phase_space_design_target_completeness",
    "phase_space_ready_for_regime_discovery_flag",
)
REGIME_PLACEHOLDER_COLUMNS = (
    "regime_membership",
    "regime_probability",
    "regime_confidence",
    "previous_regime_membership",
    "regime_switch_flag",
    "threshold_rule_id",
)
PHASE_SPACE_REQUIRED_COLUMNS = (
    *IDENTITY_COLUMNS,
    *ACCOUNTING_COLUMNS,
    *CAPABILITY_COLUMNS,
    *CAPABILITY_FLAG_COLUMNS,
    *NETWORK_COLUMNS,
    *CORE_PHASE_SPACE_COLUMNS,
    *REGIME_PLACEHOLDER_COLUMNS,
)
EMPIRICAL_PHASE_SPACE_VARIABLES = (
    "emissions_intensity_gap",
    "green_capability",
    "general_capability",
    "network_green_exposure",
    "brown_centrality",
    "supplier_lock_in",
)
DESIGN_TARGET_PHASE_SPACE_VARIABLES = (
    "capability_density",
    "green_capability_density",
    "ecosystem_proximity",
    "directed_green_precedence",
    "reachable_green_complexity",
    "transition_sector_score",
)
POSITION_EMPIRICAL_KEYS = (
    "emissions_intensity_gap",
    "green_capability",
    "general_capability",
    "network_green_exposure",
    "brown_centrality",
    "supplier_lock_in",
    "import_dependence_proxy",
    "export_dependence_proxy",
    "supplier_concentration_hhi",
    "buyer_concentration_hhi",
)


@dataclass(frozen=True)
class PhaseSpaceBuildResult:
    """Result metadata for the ABM v5 historical phase-space panel."""

    output_path: Path
    validation_path: Path
    missingness_summary_path: Path
    coverage_summary_path: Path
    start_year: int
    end_year: int
    n_rows: int
    n_agents: int

    def validate(self) -> None:
        """Validate phase-space build result metadata."""
        if not self.output_path:
            raise ValueError("output_path must not be empty.")
        if not self.validation_path:
            raise ValueError("validation_path must not be empty.")
        if not self.missingness_summary_path:
            raise ValueError("missingness_summary_path must not be empty.")
        if not self.coverage_summary_path:
            raise ValueError("coverage_summary_path must not be empty.")
        if self.n_rows <= 0:
            raise ValueError("n_rows must be positive.")
        if self.n_agents <= 0:
            raise ValueError("n_agents must be positive.")
        if self.start_year != DEFAULT_HISTORICAL_START_YEAR:
            raise ValueError("start_year must be 1995.")
        if self.end_year != DEFAULT_HISTORICAL_END_YEAR:
            raise ValueError("end_year must be 2016.")


def _validation_result(
    check_name: str,
    status: ValidationStatus,
    severity: ValidationSeverity,
    message: str,
    layer: ValidationLayer,
    n_failed: int,
    n_checked: int,
) -> ValidationResult:
    result = ValidationResult(
        check_name=check_name,
        status=status,
        severity=severity,
        message=message,
        layer=layer,
        n_failed=n_failed,
        n_checked=n_checked,
    )
    result.validate()
    return result


def compute_emissions_intensity_gap(df):
    """Compute sector-year emissions intensity gap without imputing invalid EI."""
    import polars as pl

    frame = df if isinstance(df, pl.DataFrame) else pl.DataFrame(df)
    return frame.with_columns(
        pl.when(pl.col("emissions_intensity").is_not_null() & (pl.col("emissions_intensity") >= 0))
        .then(pl.col("emissions_intensity").median().over(["sector", "year"]))
        .otherwise(None)
        .alias("_sector_year_median_emissions_intensity")
    ).with_columns(
        pl.when(
            pl.col("emissions_intensity").is_not_null()
            & pl.col("_sector_year_median_emissions_intensity").is_not_null()
        )
        .then(pl.col("emissions_intensity") - pl.col("_sector_year_median_emissions_intensity"))
        .otherwise(None)
        .alias("emissions_intensity_gap")
    ).drop("_sector_year_median_emissions_intensity")


def _build_position_record(row: dict[str, Any]) -> dict[str, Any]:
    position: dict[str, Any] = {}
    for key in POSITION_EMPIRICAL_KEYS:
        value = row.get(key)
        if value is not None:
            position[key] = value
    unavailable = [key for key in DESIGN_TARGET_PHASE_SPACE_VARIABLES if row.get(key) is None]
    position["unavailable_design_targets"] = unavailable
    return position


def build_phase_space_position(df):
    """Add observed phase-space coordinate records without assigning regimes."""
    import polars as pl

    frame = df if isinstance(df, pl.DataFrame) else pl.DataFrame(df)
    records = frame.to_dicts()
    positions = [_build_position_record(record) for record in records]
    return frame.with_columns(pl.Series("phase_space_position", positions))


def compute_phase_space_completeness(df):
    """Add empirical/design-target completeness and regime-discovery readiness flags."""
    import polars as pl

    frame = df if isinstance(df, pl.DataFrame) else pl.DataFrame(df)
    empirical_present = [pl.col(column).is_not_null().cast(pl.Float64) for column in EMPIRICAL_PHASE_SPACE_VARIABLES]
    design_present = [pl.col(column).is_not_null().cast(pl.Float64) for column in DESIGN_TARGET_PHASE_SPACE_VARIABLES]
    return frame.with_columns(
        (sum(empirical_present) / len(EMPIRICAL_PHASE_SPACE_VARIABLES)).alias(
            "phase_space_empirical_completeness"
        ),
        (sum(design_present) / len(DESIGN_TARGET_PHASE_SPACE_VARIABLES)).alias(
            "phase_space_design_target_completeness"
        ),
    ).with_columns(
        (
            (pl.col("accounting_ei_valid_flag") == True)
            & (pl.col("phase_space_empirical_completeness") >= 0.5)
            & pl.col("emissions_intensity_gap").is_not_null()
            & (pl.col("green_capability").is_not_null() | pl.col("general_capability").is_not_null())
            & pl.col("network_green_exposure").is_not_null()
        ).alias("phase_space_ready_for_regime_discovery_flag")
    )


def add_regime_placeholders(df):
    """Add nullable regime placeholder columns reserved for Phase 3."""
    import polars as pl

    frame = df if isinstance(df, pl.DataFrame) else pl.DataFrame(df)
    return frame.with_columns(
        pl.lit(None).cast(pl.Utf8).alias("regime_membership"),
        pl.lit(None).cast(pl.Float64).alias("regime_probability"),
        pl.lit(None).cast(pl.Float64).alias("regime_confidence"),
        pl.lit(None).cast(pl.Utf8).alias("previous_regime_membership"),
        pl.lit(None).cast(pl.Boolean).alias("regime_switch_flag"),
        pl.lit(None).cast(pl.Utf8).alias("threshold_rule_id"),
    )


def validate_phase_space_panel(df) -> tuple[ValidationResult, ...]:
    """Validate ABM v5 historical phase-space metadata."""
    import polars as pl

    frame = df if isinstance(df, pl.DataFrame) else pl.DataFrame(df)
    results: list[ValidationResult] = []
    missing_columns = sorted(column for column in PHASE_SPACE_REQUIRED_COLUMNS if column not in frame.columns)
    results.append(
        _validation_result(
            "phase_space_required_columns",
            ValidationStatus.FAILED if missing_columns else ValidationStatus.PASSED,
            ValidationSeverity.CRITICAL if missing_columns else ValidationSeverity.INFO,
            f"Missing required phase-space columns: {missing_columns}."
            if missing_columns
            else "All required phase-space columns are present.",
            ValidationLayer.STRUCTURAL_VALIDITY,
            len(missing_columns),
            len(PHASE_SPACE_REQUIRED_COLUMNS),
        )
    )
    if missing_columns:
        return tuple(results)

    duplicate_count = int(
        frame.group_by(["country_sector", "year"]).len().filter(pl.col("len") > 1)["len"].sum()
        or 0
    )
    results.append(
        _validation_result(
            "phase_space_unique_country_sector_year",
            ValidationStatus.FAILED if duplicate_count else ValidationStatus.PASSED,
            ValidationSeverity.CRITICAL if duplicate_count else ValidationSeverity.INFO,
            f"Found {duplicate_count} duplicate country_sector-year rows."
            if duplicate_count
            else "country_sector-year rows are unique.",
            ValidationLayer.STRUCTURAL_VALIDITY,
            duplicate_count,
            frame.height,
        )
    )

    years = sorted(frame["year"].drop_nulls().unique().to_list())
    expected_years = list(range(HISTORICAL_START_YEAR, HISTORICAL_END_YEAR + 1))
    is_full_panel = frame.height > frame["country_sector"].n_unique()
    year_failure = int(is_full_panel and years != expected_years)
    results.append(
        _validation_result(
            "phase_space_year_coverage",
            ValidationStatus.FAILED if year_failure else ValidationStatus.PASSED,
            ValidationSeverity.ERROR if year_failure else ValidationSeverity.INFO,
            f"Full panel years must be {expected_years}; observed {years}."
            if year_failure
            else "Year coverage is valid for the provided panel.",
            ValidationLayer.STRUCTURAL_VALIDITY,
            year_failure,
            len(years),
        )
    )

    bounded_columns = (
        "local_greenness",
        "network_green_exposure",
        "brown_centrality",
        "supplier_lock_in",
        "phase_space_empirical_completeness",
        "phase_space_design_target_completeness",
    )
    for column in bounded_columns:
        invalid_count = frame.filter(
            pl.col(column).is_not_null() & ((pl.col(column) < 0) | (pl.col(column) > 1))
        ).height
        results.append(
            _validation_result(
                f"phase_space_{column}_bounds",
                ValidationStatus.FAILED if invalid_count else ValidationStatus.PASSED,
                ValidationSeverity.ERROR if invalid_count else ValidationSeverity.INFO,
                f"Found {invalid_count} {column} values outside [0, 1]."
                if invalid_count
                else f"{column} values are within [0, 1] where present.",
                ValidationLayer.MECHANISM_VALIDITY,
                invalid_count,
                frame.height,
            )
        )

    non_null_placeholders = sum(frame[column].null_count() != frame.height for column in REGIME_PLACEHOLDER_COLUMNS)
    results.append(
        _validation_result(
            "phase_space_regime_placeholders_null",
            ValidationStatus.FAILED if non_null_placeholders else ValidationStatus.PASSED,
            ValidationSeverity.CRITICAL if non_null_placeholders else ValidationSeverity.INFO,
            f"{non_null_placeholders} regime placeholder columns contain non-null values."
            if non_null_placeholders
            else "Regime placeholder columns are fully null.",
            ValidationLayer.STRUCTURAL_VALIDITY,
            non_null_placeholders,
            len(REGIME_PLACEHOLDER_COLUMNS),
        )
    )

    missing_accounting_flags = [
        column
        for column in (
            "accounting_output_positive_flag",
            "accounting_emissions_nonnegative_flag",
            "accounting_ei_valid_flag",
        )
        if column not in frame.columns
    ]
    results.append(
        _validation_result(
            "phase_space_accounting_flags_preserved",
            ValidationStatus.FAILED if missing_accounting_flags else ValidationStatus.PASSED,
            ValidationSeverity.CRITICAL if missing_accounting_flags else ValidationSeverity.INFO,
            f"Missing accounting flags: {missing_accounting_flags}."
            if missing_accounting_flags
            else "Accounting validity flags are preserved.",
            ValidationLayer.ACCOUNTING_VALIDITY,
            len(missing_accounting_flags),
            3,
        )
    )
    return tuple(results)


def summarize_phase_space_missingness(df):
    """Return phase-space variable missingness by conceptual group."""
    import polars as pl

    frame = df if isinstance(df, pl.DataFrame) else pl.DataFrame(df)
    groups = {
        "accounting": ACCOUNTING_COLUMNS,
        "capability": (*CAPABILITY_COLUMNS, *CAPABILITY_FLAG_COLUMNS),
        "network": NETWORK_COLUMNS,
        "phase_space": CORE_PHASE_SPACE_COLUMNS,
        "regime_placeholder": REGIME_PLACEHOLDER_COLUMNS,
    }
    rows: list[dict[str, object]] = []
    for group, columns in groups.items():
        for column in columns:
            if column not in frame.columns:
                continue
            missing_share = frame[column].null_count() / frame.height if frame.height else 0.0
            rows.append(
                {
                    "variable": column,
                    "missing_share": missing_share,
                    "available_share": 1.0 - missing_share,
                    "variable_group": group,
                }
            )
    return pl.DataFrame(rows)


def summarize_phase_space_variable_coverage(df) -> dict[str, float | int | bool | None]:
    """Summarize empirical and placeholder coverage in the phase-space panel."""
    import polars as pl

    frame = df if isinstance(df, pl.DataFrame) else pl.DataFrame(df)
    years = sorted(frame["year"].drop_nulls().unique().to_list()) if "year" in frame.columns else []
    return {
        "n_rows": frame.height,
        "n_agents": frame["country_sector"].n_unique() if "country_sector" in frame.columns else 0,
        "start_year": int(min(years)) if years else None,
        "end_year": int(max(years)) if years else None,
        "mean_empirical_completeness": _mean_or_none(frame, "phase_space_empirical_completeness"),
        "mean_design_target_completeness": _mean_or_none(frame, "phase_space_design_target_completeness"),
        "share_ready_for_regime_discovery": _mean_bool_or_none(frame, "phase_space_ready_for_regime_discovery_flag"),
        "share_valid_emissions_intensity": _mean_bool_or_none(frame, "accounting_ei_valid_flag"),
        "share_green_capability_available": _mean_bool_or_none(frame, "green_capability_available_flag"),
        "share_general_capability_available": _mean_bool_or_none(frame, "general_capability_available_flag"),
        "share_network_green_exposure_available": _non_null_share(frame, "network_green_exposure"),
        "share_brown_centrality_available": _non_null_share(frame, "brown_centrality"),
        "share_supplier_lock_in_available": _non_null_share(frame, "supplier_lock_in"),
        "regime_placeholders_all_null": all(
            column in frame.columns and frame[column].null_count() == frame.height
            for column in REGIME_PLACEHOLDER_COLUMNS
        ),
    }


def _mean_or_none(frame, column: str) -> float | None:
    import polars as pl

    if column not in frame.columns:
        return None
    value = frame.select(pl.col(column).cast(pl.Float64, strict=False).mean()).item()
    return float(value) if value is not None else None


def _mean_bool_or_none(frame, column: str) -> float | None:
    import polars as pl

    if column not in frame.columns:
        return None
    value = frame.select(pl.col(column).cast(pl.Float64, strict=False).mean()).item()
    return float(value) if value is not None else None


def _non_null_share(frame, column: str) -> float | None:
    if column not in frame.columns or frame.height == 0:
        return None
    return (frame.height - frame[column].null_count()) / frame.height


def _validation_results_to_dict(results: tuple[ValidationResult, ...]) -> list[dict[str, Any]]:
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
    return any(
        result.status is ValidationStatus.FAILED and result.severity is ValidationSeverity.CRITICAL
        for result in results
    )


def build_historical_phase_space_panel(project_root: Path) -> PhaseSpaceBuildResult:
    """Build the unified observed ABM v5 historical phase-space panel."""
    import polars as pl

    paths = ABMV5Paths.from_project_root(project_root)
    paths.validate_project_root()
    paths.ensure_directories()

    accounting_path = paths.accounting / ACCOUNTING_OUTPUT_FILENAME
    capability_path = paths.capabilities / CAPABILITY_OUTPUT_FILENAME
    network_path = paths.supplier_network / NETWORK_OUTPUT_FILENAME
    if not accounting_path.exists():
        raise FileNotFoundError(f"Accounting panel missing: {accounting_path}. Run Phase 2.3 first.")
    if not capability_path.exists():
        raise FileNotFoundError(f"Capability panel missing: {capability_path}. Run Phase 2.4 first.")
    if not network_path.exists():
        raise FileNotFoundError(f"Network panel missing: {network_path}. Run Phase 2.5 first.")

    accounting = pl.read_parquet(accounting_path)
    capability = pl.read_parquet(capability_path)
    network = pl.read_parquet(network_path)
    capability_keep = [
        column
        for column in (*CAPABILITY_COLUMNS, *CAPABILITY_FLAG_COLUMNS)
        if column in capability.columns
    ]
    network_keep = [column for column in NETWORK_COLUMNS if column in network.columns]
    panel = (
        accounting.join(
            capability.select("country_sector", "year", *capability_keep),
            on=["country_sector", "year"],
            how="left",
        )
        .join(network.select("country_sector", "year", *network_keep), on=["country_sector", "year"], how="left")
    )
    panel = compute_emissions_intensity_gap(panel)
    panel = build_phase_space_position(panel)
    panel = compute_phase_space_completeness(panel)
    panel = add_regime_placeholders(panel)

    for column in PHASE_SPACE_REQUIRED_COLUMNS:
        if column not in panel.columns:
            panel = panel.with_columns(pl.lit(None).alias(column))
    panel = panel.select(PHASE_SPACE_REQUIRED_COLUMNS)

    results = validate_phase_space_panel(panel)
    missingness = summarize_phase_space_missingness(panel)
    coverage = summarize_phase_space_variable_coverage(panel)

    output_path = paths.phase_space / PHASE_SPACE_OUTPUT_FILENAME
    validation_path = paths.validation / PHASE_SPACE_VALIDATION_FILENAME
    missingness_summary_path = paths.diagnostics / PHASE_SPACE_MISSINGNESS_FILENAME
    coverage_summary_path = paths.diagnostics / PHASE_SPACE_COVERAGE_FILENAME
    panel.write_parquet(output_path)
    validation_path.write_text(
        json.dumps(
            {
                "validation_scope": "abm_v5_historical_phase_space_panel",
                "results": _validation_results_to_dict(results),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    missingness.write_csv(missingness_summary_path)
    coverage_summary_path.write_text(json.dumps(coverage, indent=2, sort_keys=True), encoding="utf-8")
    if _has_critical_failures(results):
        raise ValueError(f"Phase-space validation has critical failures: {validation_path}")

    result = PhaseSpaceBuildResult(
        output_path=output_path,
        validation_path=validation_path,
        missingness_summary_path=missingness_summary_path,
        coverage_summary_path=coverage_summary_path,
        start_year=HISTORICAL_START_YEAR,
        end_year=HISTORICAL_END_YEAR,
        n_rows=panel.height,
        n_agents=panel["country_sector"].n_unique(),
    )
    result.validate()
    return result


def load_historical_phase_space_panel(path: Path):
    """Load the ABM v5 historical phase-space panel."""
    import polars as pl

    return pl.read_parquet(path)
