from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.abm_v5.accounting import ACCOUNTING_OUTPUT_FILENAME
from src.abm_v5.config import DEFAULT_HISTORICAL_END_YEAR, DEFAULT_HISTORICAL_START_YEAR, ValidationLayer
from src.abm_v5.identity import IDENTITY_COLUMNS, load_agent_identity_panel
from src.abm_v5.paths import ABMV5Paths
from src.abm_v5.validation import ValidationResult, ValidationSeverity, ValidationStatus


HISTORICAL_START_YEAR = DEFAULT_HISTORICAL_START_YEAR
HISTORICAL_END_YEAR = DEFAULT_HISTORICAL_END_YEAR
CAPABILITY_SOURCE_NAME = "atlas_eora26_sector_capabilities_1995_2016"
CAPABILITY_OUTPUT_FILENAME = "capability_state_panel_1995_2016.parquet"
CAPABILITY_VALIDATION_FILENAME = "capability_state_validation.json"

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
ACCOUNTING_FLAG_COLUMNS = (
    "accounting_output_positive_flag",
    "accounting_emissions_nonnegative_flag",
    "accounting_ei_valid_flag",
)
CAPABILITY_ID_COLUMNS = ("country_sector", "country", "country_detail", "category", "sector", "year")
CAPABILITY_REQUIRED_COLUMNS = (
    *CAPABILITY_ID_COLUMNS,
    *CAPABILITY_COLUMNS,
    *CAPABILITY_FLAG_COLUMNS,
    "capability_source",
    *ACCOUNTING_FLAG_COLUMNS,
)

CANONICAL_COLUMN_VARIANTS: dict[str, tuple[str, ...]] = {
    "country_sector": ("country_sector", "country_sector", "eora_country_sector", "eora26_country_sector"),
    "country": ("country", "iso3code"),
    "country_detail": ("country_detail", "countryname"),
    "category": ("category",),
    "sector": ("sector", "eora26_sector"),
    "year": ("year",),
    "general_capability": (
        "general_capability",
        "complexity",
        "economic_complexity",
        "eci",
        "sector_complexity",
        "capability_score",
        "capability_export_weighted_pci",
        "capability_mean_pci",
        "active_good_count",
        "diversity",
    ),
    "green_capability": (
        "green_capability",
        "green_capability_export_share",
        "green_export_capability",
        "green_product_share",
        "green_capability_share",
        "green_active_good_count",
    ),
    "capability_density": ("capability_density", "density", "relatedness_density"),
    "green_capability_density": (
        "green_capability_density",
        "green_density",
        "green_relatedness_density",
    ),
    "ecosystem_proximity": (
        "ecosystem_proximity",
        "capability_ecosystem_proximity",
        "proximity_to_green",
    ),
    "directed_green_precedence": (
        "directed_green_precedence",
        "green_precedence",
        "directed_precedence",
    ),
    "reachable_green_complexity": ("reachable_green_complexity", "reachable_complexity"),
    "transition_sector_score": ("transition_sector_score", "transition_score"),
}


@dataclass(frozen=True)
class CapabilityBuildResult:
    """Result metadata for the ABM v5 capability state panel."""

    output_path: Path
    validation_path: Path
    start_year: int
    end_year: int
    n_rows: int
    n_agents: int
    source_used: str

    def validate(self) -> None:
        """Validate build result metadata."""
        if self.n_rows <= 0:
            raise ValueError("n_rows must be positive.")
        if self.n_agents <= 0:
            raise ValueError("n_agents must be positive.")
        if self.start_year != DEFAULT_HISTORICAL_START_YEAR:
            raise ValueError("start_year must be 1995.")
        if self.end_year != DEFAULT_HISTORICAL_END_YEAR:
            raise ValueError("end_year must be 2016.")
        if not self.source_used:
            raise ValueError("source_used must not be empty.")


def _normalize_name(column_name: str) -> str:
    return column_name.strip().lower().replace(" ", "_")


def inspect_capability_source_columns(path: Path) -> dict[str, list[str]]:
    """Inspect source columns for transparency without modelling logic."""
    import polars as pl

    if path.suffix.lower() == ".parquet":
        columns = list(pl.scan_parquet(path).collect_schema().names())
    elif path.suffix.lower() == ".csv":
        columns = list(pl.read_csv(path, n_rows=0).columns)
    else:
        columns = list(pl.read_parquet(path).columns)
    normalized = {_normalize_name(column): column for column in columns}
    identity_variants = {
        variant
        for canonical in ("country_sector", "country", "country_detail", "category", "sector", "year")
        for variant in CANONICAL_COLUMN_VARIANTS[canonical]
    }
    capability_variants = {
        variant
        for canonical in CAPABILITY_COLUMNS
        for variant in CANONICAL_COLUMN_VARIANTS[canonical]
    }
    return {
        "columns": columns,
        "candidate_identity_columns": [
            original for normalized_name, original in normalized.items() if normalized_name in identity_variants
        ],
        "candidate_capability_columns": [
            original for normalized_name, original in normalized.items() if normalized_name in capability_variants
        ],
    }


def normalize_capability_source_columns(df):
    """Normalize known Atlas capability column variants into ABM v5 canonical names."""
    import polars as pl

    frame = df if isinstance(df, pl.DataFrame) else pl.DataFrame(df)
    normalized_lookup = {_normalize_name(column): column for column in frame.columns}
    rename_map: dict[str, str] = {}
    existing_columns = set(frame.columns)
    for canonical_name, variants in CANONICAL_COLUMN_VARIANTS.items():
        if canonical_name in existing_columns:
            continue
        for variant in variants:
            source_column = normalized_lookup.get(variant)
            if source_column is not None:
                rename_map[source_column] = canonical_name
                existing_columns.add(canonical_name)
                break
    return frame.rename(rename_map)


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


def validate_capability_state_panel(df) -> tuple[ValidationResult, ...]:
    """Validate the ABM v5 capability state panel."""
    import polars as pl

    frame = df if isinstance(df, pl.DataFrame) else pl.DataFrame(df)
    results: list[ValidationResult] = []
    missing_columns = sorted(column for column in CAPABILITY_REQUIRED_COLUMNS if column not in frame.columns)
    results.append(
        _validation_result(
            "capability_required_columns",
            ValidationStatus.FAILED if missing_columns else ValidationStatus.PASSED,
            ValidationSeverity.CRITICAL if missing_columns else ValidationSeverity.INFO,
            f"Missing required capability columns: {missing_columns}."
            if missing_columns
            else "All required capability columns are present.",
            ValidationLayer.STRUCTURAL_VALIDITY,
            n_failed=len(missing_columns),
            n_checked=len(CAPABILITY_REQUIRED_COLUMNS),
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
            "capability_unique_country_sector_year",
            ValidationStatus.FAILED if duplicate_count else ValidationStatus.PASSED,
            ValidationSeverity.CRITICAL if duplicate_count else ValidationSeverity.INFO,
            f"Found {duplicate_count} duplicate country_sector-year rows."
            if duplicate_count
            else "country_sector-year rows are unique.",
            ValidationLayer.STRUCTURAL_VALIDITY,
            n_failed=duplicate_count,
            n_checked=frame.height,
        )
    )

    years = sorted(frame["year"].drop_nulls().unique().to_list())
    expected_years = list(range(HISTORICAL_START_YEAR, HISTORICAL_END_YEAR + 1))
    is_full_panel = frame.height > frame["country_sector"].n_unique()
    year_failure = int(is_full_panel and years != expected_years)
    results.append(
        _validation_result(
            "capability_year_coverage",
            ValidationStatus.FAILED if year_failure else ValidationStatus.PASSED,
            ValidationSeverity.ERROR if year_failure else ValidationSeverity.INFO,
            f"Full panel years must be {expected_years}; observed {years}."
            if year_failure
            else "Year coverage is valid for the provided panel.",
            ValidationLayer.STRUCTURAL_VALIDITY,
            n_failed=year_failure,
            n_checked=len(years),
        )
    )

    empty_country_sector_count = frame.filter(
        pl.col("country_sector").is_null() | (pl.col("country_sector").cast(pl.Utf8).str.strip_chars() == "")
    ).height
    results.append(
        _validation_result(
            "capability_country_sector_non_empty",
            ValidationStatus.FAILED if empty_country_sector_count else ValidationStatus.PASSED,
            ValidationSeverity.ERROR if empty_country_sector_count else ValidationSeverity.INFO,
            f"Found {empty_country_sector_count} empty country_sector values."
            if empty_country_sector_count
            else "country_sector values are non-empty.",
            ValidationLayer.STRUCTURAL_VALIDITY,
            n_failed=empty_country_sector_count,
            n_checked=frame.height,
        )
    )

    flag_failures = 0
    flag_checks = 0
    for capability_column, flag_column in zip(CAPABILITY_COLUMNS, CAPABILITY_FLAG_COLUMNS):
        flag_checks += frame.height
        mismatch_count = frame.filter(
            (pl.col(flag_column).cast(pl.Boolean, strict=False).fill_null(False) != pl.col(capability_column).is_not_null())
        ).height
        flag_failures += mismatch_count
    results.append(
        _validation_result(
            "capability_availability_flags_match_values",
            ValidationStatus.FAILED if flag_failures else ValidationStatus.PASSED,
            ValidationSeverity.ERROR if flag_failures else ValidationSeverity.INFO,
            f"Found {flag_failures} availability-flag mismatches."
            if flag_failures
            else "Availability flags match nullness of capability values.",
            ValidationLayer.MECHANISM_VALIDITY,
            n_failed=flag_failures,
            n_checked=flag_checks,
        )
    )

    infinite_count = 0
    for column in CAPABILITY_COLUMNS:
        infinite_count += frame.filter(pl.col(column).is_infinite().fill_null(False)).height
    results.append(
        _validation_result(
            "capability_values_finite_or_null",
            ValidationStatus.FAILED if infinite_count else ValidationStatus.PASSED,
            ValidationSeverity.ERROR if infinite_count else ValidationSeverity.INFO,
            f"Found {infinite_count} infinite capability values."
            if infinite_count
            else "Capability values are finite or null.",
            ValidationLayer.MECHANISM_VALIDITY,
            n_failed=infinite_count,
            n_checked=frame.height * len(CAPABILITY_COLUMNS),
        )
    )

    numeric_failures = 0
    for column in ("general_capability", "green_capability"):
        numeric_failures += frame.select(pl.col(column).cast(pl.Float64, strict=False).is_null().sum()).item() - frame[column].null_count()
    results.append(
        _validation_result(
            "capability_core_values_numeric_or_null",
            ValidationStatus.FAILED if numeric_failures else ValidationStatus.PASSED,
            ValidationSeverity.ERROR if numeric_failures else ValidationSeverity.INFO,
            f"Found {numeric_failures} nonnumeric general/green capability values."
            if numeric_failures
            else "general_capability and green_capability are numeric or null.",
            ValidationLayer.MECHANISM_VALIDITY,
            n_failed=int(numeric_failures),
            n_checked=frame.height * 2,
        )
    )

    missing_accounting_flags = sorted(column for column in ACCOUNTING_FLAG_COLUMNS if column not in frame.columns)
    results.append(
        _validation_result(
            "capability_accounting_flags_preserved",
            ValidationStatus.FAILED if missing_accounting_flags else ValidationStatus.PASSED,
            ValidationSeverity.CRITICAL if missing_accounting_flags else ValidationSeverity.INFO,
            f"Accounting flags were dropped: {missing_accounting_flags}."
            if missing_accounting_flags
            else "Accounting validity flags are preserved.",
            ValidationLayer.STRUCTURAL_VALIDITY,
            n_failed=len(missing_accounting_flags),
            n_checked=len(ACCOUNTING_FLAG_COLUMNS),
        )
    )

    return tuple(results)


def summarize_capability_missingness(df) -> dict[str, float]:
    """Return missingness shares for canonical capability variables."""
    import polars as pl

    frame = df if isinstance(df, pl.DataFrame) else pl.DataFrame(df)
    if frame.height == 0:
        return {column: 1.0 for column in CAPABILITY_COLUMNS}
    return {column: frame[column].null_count() / frame.height for column in CAPABILITY_COLUMNS}


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


def _write_validation_json(
    path: Path,
    source_used: str,
    results: tuple[ValidationResult, ...],
    missingness: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "validation_scope": "abm_v5_capability_state_panel",
                "source_used": source_used,
                "missingness": missingness,
                "results": _validation_results_to_dict(results),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def build_capability_state_panel(project_root: Path) -> CapabilityBuildResult:
    """Build the historical ABM v5 capability and ecosystem state panel."""
    import polars as pl

    paths = ABMV5Paths.from_project_root(project_root)
    paths.capabilities.mkdir(parents=True, exist_ok=True)
    paths.validation.mkdir(parents=True, exist_ok=True)

    identity_path = paths.inputs / "agent_identity.parquet"
    accounting_path = paths.accounting / ACCOUNTING_OUTPUT_FILENAME
    atlas_path = (
        project_root
        / "data"
        / "atlas"
        / "processed"
        / "atlas_eora26_sector_capabilities_1995_2016.parquet"
    )
    if not identity_path.exists():
        raise FileNotFoundError(f"Missing {identity_path}. Run ABM_v5 Phase 2.2 first.")
    if not accounting_path.exists():
        raise FileNotFoundError(f"Missing {accounting_path}. Run ABM_v5 Phase 2.3 first.")
    if not atlas_path.exists():
        raise FileNotFoundError(f"Missing Atlas capability source: {atlas_path}")

    identity_panel = load_agent_identity_panel(identity_path).select(list(IDENTITY_COLUMNS))
    accounting_panel = pl.read_parquet(accounting_path).select(
        "country_sector",
        "year",
        *ACCOUNTING_FLAG_COLUMNS,
    )
    atlas_panel = normalize_capability_source_columns(pl.read_parquet(atlas_path))
    if "year" not in atlas_panel.columns:
        raise ValueError("Atlas capability source must include year or a recognized variant.")
    if "country_sector" not in atlas_panel.columns:
        if {"country", "sector"}.issubset(set(atlas_panel.columns)):
            atlas_panel = atlas_panel.join(
                identity_panel.select("country_sector", "country", "sector"),
                on=["country", "sector"],
                how="left",
            )
        if "country_sector" not in atlas_panel.columns:
            raise ValueError(
                "Atlas capability source must include country_sector or recognized country/sector fields."
            )

    selected_columns = ["country_sector", "year", *[column for column in CAPABILITY_COLUMNS if column in atlas_panel.columns]]
    capability_source = atlas_panel.select(selected_columns).unique(subset=["country_sector", "year"], keep="first")
    base_panel = accounting_panel.join(identity_panel, on="country_sector", how="left")
    panel = base_panel.join(capability_source, on=["country_sector", "year"], how="left")

    for column in CAPABILITY_COLUMNS:
        if column not in panel.columns:
            panel = panel.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))
        else:
            panel = panel.with_columns(pl.col(column).cast(pl.Float64, strict=False).alias(column))
        panel = panel.with_columns(pl.col(column).is_not_null().alias(f"{column}_available_flag"))

    panel = panel.with_columns(pl.lit(CAPABILITY_SOURCE_NAME).alias("capability_source")).select(
        *CAPABILITY_ID_COLUMNS,
        *CAPABILITY_COLUMNS,
        *CAPABILITY_FLAG_COLUMNS,
        "capability_source",
        *ACCOUNTING_FLAG_COLUMNS,
    )

    output_path = paths.capabilities / CAPABILITY_OUTPUT_FILENAME
    validation_path = paths.validation / CAPABILITY_VALIDATION_FILENAME
    validation_results = validate_capability_state_panel(panel)
    missingness = summarize_capability_missingness(panel)
    _write_validation_json(validation_path, CAPABILITY_SOURCE_NAME, validation_results, missingness)
    if _has_critical_failures(validation_results):
        raise ValueError("ABM v5 capability state panel has critical validation failures.")

    panel.write_parquet(output_path)
    result = CapabilityBuildResult(
        output_path=output_path,
        validation_path=validation_path,
        start_year=HISTORICAL_START_YEAR,
        end_year=HISTORICAL_END_YEAR,
        n_rows=panel.height,
        n_agents=identity_panel["country_sector"].n_unique(),
        source_used=CAPABILITY_SOURCE_NAME,
    )
    result.validate()
    return result


def load_capability_state_panel(path: Path):
    """Load the generated ABM v5 capability state panel."""
    import polars as pl

    return pl.read_parquet(path)
