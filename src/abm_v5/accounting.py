from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.abm_v5.config import DEFAULT_HISTORICAL_END_YEAR, DEFAULT_HISTORICAL_START_YEAR, ValidationLayer
from src.abm_v5.identity import IDENTITY_COLUMNS, load_agent_identity_panel, parse_eora_country_sector_label
from src.abm_v5.paths import ABMV5Paths
from src.abm_v5.validation import ValidationResult, ValidationSeverity, ValidationStatus


HISTORICAL_START_YEAR = DEFAULT_HISTORICAL_START_YEAR
HISTORICAL_END_YEAR = DEFAULT_HISTORICAL_END_YEAR
CO2_ROW_LABEL = "Total CO2 emissions (Gg) from EDGAR | Total"
GREENNESS_EPSILON = 1e-12
ACCOUNTING_OUTPUT_FILENAME = "accounting_state_panel_1995_2016.parquet"
ACCOUNTING_VALIDATION_FILENAME = "accounting_state_validation.json"
ACCOUNTING_COLUMNS = (
    "country_sector",
    "year",
    "output",
    "final_demand",
    "emissions",
    "emissions_intensity",
    "local_greenness",
    "accounting_output_positive_flag",
    "accounting_emissions_nonnegative_flag",
    "accounting_ei_valid_flag",
)


@dataclass(frozen=True)
class AccountingBuildResult:
    """Result metadata for the ABM v5 historical accounting state panel."""

    output_path: Path
    validation_path: Path
    start_year: int
    end_year: int
    n_rows: int
    n_agents: int

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


def compute_local_greenness(emissions_intensity_series):
    """Compute local, within-year green-ness from emissions intensity."""
    import polars as pl

    series = (
        emissions_intensity_series
        if isinstance(emissions_intensity_series, pl.Series)
        else pl.Series("emissions_intensity", emissions_intensity_series)
    )
    raw_values: list[float | None] = []
    for value in series.to_list():
        if value is None:
            raw_values.append(None)
            continue
        value_float = float(value)
        if value_float <= 0 or math.isnan(value_float):
            raw_values.append(None)
        else:
            raw_values.append(-math.log(value_float + GREENNESS_EPSILON))

    valid_values = [value for value in raw_values if value is not None]
    if not valid_values:
        return pl.Series("local_greenness", [None] * len(raw_values), dtype=pl.Float64)
    min_value = min(valid_values)
    max_value = max(valid_values)
    if max_value == min_value:
        scaled = [0.5 if value is not None else None for value in raw_values]
    else:
        scaled = [
            ((value - min_value) / (max_value - min_value)) if value is not None else None
            for value in raw_values
        ]
    return pl.Series("local_greenness", scaled, dtype=pl.Float64)


def _read_label_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _canonical_q_label(label: str) -> str:
    parts = [part.strip() for part in label.strip().replace("\t", "|").split("|") if part.strip()]
    return " | ".join(parts)


def _value_columns(frame) -> list[str]:
    return [column for column in frame.columns if column != "__index_level_0__"]


def build_accounting_records_for_year(project_root: Path, year: int):
    """Build observed local accounting records for one historical year."""
    import polars as pl

    year_dir = project_root / "data" / "parquet" / str(year)
    raw_dir = project_root / "data" / "raw" / str(year)
    t_matrix = pl.read_parquet(year_dir / "T.parquet")
    fd_matrix = pl.read_parquet(year_dir / "FD.parquet")
    q_matrix = pl.read_parquet(year_dir / "Q.parquet")
    labels_t = _read_label_lines(raw_dir / "labels_T.txt")
    labels_q = _read_label_lines(raw_dir / "labels_Q.txt")

    t_value_columns = _value_columns(t_matrix)
    fd_value_columns = _value_columns(fd_matrix)
    q_value_columns = _value_columns(q_matrix)
    node_count = len(labels_t)
    if t_matrix.height != node_count or len(t_value_columns) != node_count:
        raise ValueError(f"T matrix dimensions do not align with labels_T for year {year}.")
    if fd_matrix.height != node_count:
        raise ValueError(f"FD matrix rows do not align with labels_T for year {year}.")
    if len(q_value_columns) != node_count:
        raise ValueError(f"Q matrix columns do not align with labels_T for year {year}.")

    co2_row_index = next(
        (index for index, label in enumerate(labels_q) if _canonical_q_label(label) == CO2_ROW_LABEL),
        None,
    )
    if co2_row_index is None:
        raise ValueError(f"CO2 row label not found in labels_Q for year {year}: {CO2_ROW_LABEL}")

    # Validated ABM v3 corrected orientation: T rows represent producing/selling nodes.
    # Local observed output is row-sum intermediate output plus row-sum final demand.
    intermediate_output = t_matrix.select(pl.sum_horizontal(t_value_columns).alias("intermediate_output"))[
        "intermediate_output"
    ]
    final_demand = fd_matrix.select(pl.sum_horizontal(fd_value_columns).alias("final_demand"))[
        "final_demand"
    ]
    emissions = q_matrix.row(co2_row_index, named=True)
    emissions_values = [emissions[column] for column in q_value_columns]
    output_values = (intermediate_output + final_demand).to_list()
    final_demand_values = final_demand.to_list()
    emissions_intensity_values: list[float | None] = []
    for output, emission in zip(output_values, emissions_values):
        output_valid = output is not None and float(output) > 0
        emission_valid = emission is not None and float(emission) >= 0
        emissions_intensity_values.append(
            float(emission) / float(output) if output_valid and emission_valid else None
        )
    local_greenness = compute_local_greenness(emissions_intensity_values).to_list()

    records: list[dict[str, object]] = []
    for index, label in enumerate(labels_t):
        parsed = parse_eora_country_sector_label(label)
        output = output_values[index]
        emission = emissions_values[index]
        emissions_intensity = emissions_intensity_values[index]
        records.append(
            {
                "country_sector": parsed["country_sector"],
                "year": year,
                "output": float(output) if output is not None else None,
                "final_demand": float(final_demand_values[index])
                if final_demand_values[index] is not None
                else None,
                "emissions": float(emission) if emission is not None else None,
                "emissions_intensity": emissions_intensity,
                "local_greenness": local_greenness[index],
                "accounting_output_positive_flag": output is not None and float(output) > 0,
                "accounting_emissions_nonnegative_flag": emission is not None and float(emission) >= 0,
                "accounting_ei_valid_flag": emissions_intensity is not None,
            }
        )
    return records


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


def validate_accounting_state_panel(df) -> tuple[ValidationResult, ...]:
    """Validate the ABM v5 accounting state panel."""
    import polars as pl

    frame = df if isinstance(df, pl.DataFrame) else pl.DataFrame(df)
    results: list[ValidationResult] = []
    missing_columns = sorted(column for column in ACCOUNTING_COLUMNS if column not in frame.columns)
    results.append(
        _validation_result(
            "accounting_required_columns",
            ValidationStatus.FAILED if missing_columns else ValidationStatus.PASSED,
            ValidationSeverity.CRITICAL if missing_columns else ValidationSeverity.INFO,
            f"Missing required accounting columns: {missing_columns}."
            if missing_columns
            else "All required accounting columns are present.",
            ValidationLayer.STRUCTURAL_VALIDITY,
            n_failed=len(missing_columns),
            n_checked=len(ACCOUNTING_COLUMNS),
        )
    )
    if missing_columns:
        return tuple(results)

    duplicate_count = (
        frame.group_by(["country_sector", "year"]).len().filter(pl.col("len") > 1)["len"].sum() or 0
    )
    duplicate_count = int(duplicate_count)
    results.append(
        _validation_result(
            "accounting_unique_country_sector_year",
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

    negative_output = frame.filter(pl.col("output").is_not_null() & (pl.col("output") < 0)).height
    results.append(
        _validation_result(
            "accounting_output_nonnegative_or_null",
            ValidationStatus.FAILED if negative_output else ValidationStatus.PASSED,
            ValidationSeverity.ERROR if negative_output else ValidationSeverity.INFO,
            f"Found {negative_output} rows with negative output."
            if negative_output
            else "Output is nonnegative or null.",
            ValidationLayer.ACCOUNTING_VALIDITY,
            n_failed=negative_output,
            n_checked=frame.height,
        )
    )

    negative_emissions = frame.filter(pl.col("emissions").is_not_null() & (pl.col("emissions") < 0)).height
    results.append(
        _validation_result(
            "accounting_emissions_nonnegative_or_null",
            ValidationStatus.FAILED if negative_emissions else ValidationStatus.PASSED,
            ValidationSeverity.ERROR if negative_emissions else ValidationSeverity.INFO,
            f"Found {negative_emissions} rows with negative emissions."
            if negative_emissions
            else "Emissions are nonnegative or null.",
            ValidationLayer.ACCOUNTING_VALIDITY,
            n_failed=negative_emissions,
            n_checked=frame.height,
        )
    )

    invalid_ei = frame.filter(
        (pl.col("output") > 0)
        & pl.col("emissions").is_not_null()
        & (pl.col("emissions") >= 0)
        & (pl.col("emissions_intensity").is_null() | (pl.col("emissions_intensity") < 0))
    ).height
    results.append(
        _validation_result(
            "accounting_emissions_intensity_valid",
            ValidationStatus.FAILED if invalid_ei else ValidationStatus.PASSED,
            ValidationSeverity.ERROR if invalid_ei else ValidationSeverity.INFO,
            f"Found {invalid_ei} rows with invalid emissions_intensity."
            if invalid_ei
            else "Emissions intensity is valid where output and emissions permit it.",
            ValidationLayer.ACCOUNTING_VALIDITY,
            n_failed=invalid_ei,
            n_checked=frame.height,
        )
    )

    invalid_greenness = frame.filter(
        pl.col("local_greenness").is_not_null()
        & ((pl.col("local_greenness") < 0) | (pl.col("local_greenness") > 1))
    ).height
    results.append(
        _validation_result(
            "accounting_local_greenness_range",
            ValidationStatus.FAILED if invalid_greenness else ValidationStatus.PASSED,
            ValidationSeverity.ERROR if invalid_greenness else ValidationSeverity.INFO,
            f"Found {invalid_greenness} rows with local_greenness outside [0, 1]."
            if invalid_greenness
            else "local_greenness is between 0 and 1 where present.",
            ValidationLayer.ACCOUNTING_VALIDITY,
            n_failed=invalid_greenness,
            n_checked=frame.height,
        )
    )

    years = sorted(frame["year"].drop_nulls().unique().to_list())
    expected_years = list(range(HISTORICAL_START_YEAR, HISTORICAL_END_YEAR + 1))
    is_full_panel = frame.height > len(frame["country_sector"].unique())
    year_failure = int(is_full_panel and years != expected_years)
    results.append(
        _validation_result(
            "accounting_year_coverage",
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

    nonpositive_output_invalid_ei = frame.filter(
        (pl.col("output").is_not_null())
        & (pl.col("output") <= 0)
        & pl.col("emissions_intensity").is_not_null()
    ).height
    results.append(
        _validation_result(
            "accounting_nonpositive_output_has_null_ei",
            ValidationStatus.FAILED if nonpositive_output_invalid_ei else ValidationStatus.PASSED,
            ValidationSeverity.WARNING if nonpositive_output_invalid_ei else ValidationSeverity.INFO,
            f"Found {nonpositive_output_invalid_ei} rows where nonpositive output has non-null EI."
            if nonpositive_output_invalid_ei
            else "Rows with nonpositive output do not silently produce valid EI.",
            ValidationLayer.ACCOUNTING_VALIDITY,
            n_failed=nonpositive_output_invalid_ei,
            n_checked=frame.height,
        )
    )

    return tuple(results)


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
    return any(result.status is ValidationStatus.FAILED and result.severity is ValidationSeverity.CRITICAL for result in results)


def _write_validation_json(path: Path, results: tuple[ValidationResult, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "validation_scope": "abm_v5_accounting_state_panel",
                "results": _validation_results_to_dict(results),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def build_accounting_state_panel(project_root: Path) -> AccountingBuildResult:
    """Build the observed ABM v5 accounting state panel for 1995-2016."""
    import polars as pl

    paths = ABMV5Paths.from_project_root(project_root)
    paths.accounting.mkdir(parents=True, exist_ok=True)
    paths.validation.mkdir(parents=True, exist_ok=True)
    identity_path = paths.inputs / "agent_identity.parquet"
    if not identity_path.exists():
        raise FileNotFoundError(
            f"Missing {identity_path}. Run ABM_v5 Phase 2.2 build_agent_identity_panel first."
        )

    records: list[dict[str, object]] = []
    for year in range(HISTORICAL_START_YEAR, HISTORICAL_END_YEAR + 1):
        records.extend(build_accounting_records_for_year(project_root, year))
    accounting_panel = pl.DataFrame(records)
    identity_panel = load_agent_identity_panel(identity_path).select(list(IDENTITY_COLUMNS))
    panel = identity_panel.join(accounting_panel, on="country_sector", how="left").select(
        [*IDENTITY_COLUMNS, *[column for column in ACCOUNTING_COLUMNS if column != "country_sector"]]
    )

    output_path = paths.accounting / ACCOUNTING_OUTPUT_FILENAME
    validation_path = paths.validation / ACCOUNTING_VALIDATION_FILENAME
    validation_results = validate_accounting_state_panel(panel)
    _write_validation_json(validation_path, validation_results)
    if _has_critical_failures(validation_results):
        raise ValueError("ABM v5 accounting state panel has critical validation failures.")

    panel.write_parquet(output_path)
    result = AccountingBuildResult(
        output_path=output_path,
        validation_path=validation_path,
        start_year=HISTORICAL_START_YEAR,
        end_year=HISTORICAL_END_YEAR,
        n_rows=panel.height,
        n_agents=identity_panel["country_sector"].n_unique(),
    )
    result.validate()
    return result


def load_accounting_state_panel(path: Path):
    """Load the generated ABM v5 accounting state panel."""
    import polars as pl

    return pl.read_parquet(path)
