from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.abm_v5.config import ValidationLayer
from src.abm_v5.paths import ABMV5Paths
from src.abm_v5.validation import ValidationResult, ValidationSeverity, ValidationStatus


IDENTITY_COLUMNS = ("country_sector", "country", "country_detail", "category", "sector")


@dataclass(frozen=True)
class AgentIdentityBuildResult:
    """Result metadata for building the ABM v5 agent identity panel."""

    output_path: Path
    validation_path: Path
    n_agents: int
    source_used: str

    def validate(self) -> None:
        """Validate build result metadata."""
        if not self.output_path:
            raise ValueError("output_path must not be empty.")
        if not self.validation_path:
            raise ValueError("validation_path must not be empty.")
        if self.n_agents <= 0:
            raise ValueError("n_agents must be positive.")


def parse_eora_country_sector_label(label: str) -> dict[str, str | None]:
    """Parse an Eora country-sector label into canonical identity fields."""
    country_sector = label.strip()
    separator = "|" if "|" in country_sector else "\t"
    parts = [part.strip() for part in country_sector.split(separator)]
    parsed: dict[str, str | None] = {
        "country_sector": country_sector,
        "country": None,
        "country_detail": None,
        "category": None,
        "sector": None,
    }
    if len(parts) == 4:
        parsed["country"], parsed["country_detail"], parsed["category"], parsed["sector"] = parts
    elif len(parts) < 4:
        for key, value in zip(IDENTITY_COLUMNS[1:], parts):
            parsed[key] = value
    else:
        parsed["country"] = parts[0]
        parsed["country_detail"] = parts[1]
        parsed["category"] = parts[2]
        parsed["sector"] = " | ".join(parts[3:])
    return parsed


def build_agent_identity_from_labels(labels: list[str]) -> list[dict[str, str | None]]:
    """Build unique country-sector identity records from raw labels."""
    records: list[dict[str, str | None]] = []
    seen_country_sectors: set[str] = set()
    for label in labels:
        stripped_label = label.strip()
        if not stripped_label:
            continue
        parsed = parse_eora_country_sector_label(stripped_label)
        country_sector = parsed["country_sector"]
        if country_sector is None or country_sector in seen_country_sectors:
            continue
        seen_country_sectors.add(country_sector)
        records.append({column: parsed.get(column) for column in IDENTITY_COLUMNS})
    return records


def _validation_result(
    check_name: str,
    status: ValidationStatus,
    severity: ValidationSeverity,
    message: str,
    n_failed: int,
    n_checked: int,
) -> ValidationResult:
    result = ValidationResult(
        check_name=check_name,
        status=status,
        severity=severity,
        message=message,
        layer=ValidationLayer.STRUCTURAL_VALIDITY,
        n_failed=n_failed,
        n_checked=n_checked,
    )
    result.validate()
    return result


def validate_agent_identity_records(records: list[dict[str, object]]) -> tuple[ValidationResult, ...]:
    """Validate canonical agent identity records."""
    results: list[ValidationResult] = []
    n_records = len(records)
    if n_records == 0:
        return (
            _validation_result(
                "agent_identity_non_empty",
                ValidationStatus.FAILED,
                ValidationSeverity.ERROR,
                "Agent identity records are empty.",
                n_failed=1,
                n_checked=1,
            ),
        )

    results.append(
        _validation_result(
            "agent_identity_non_empty",
            ValidationStatus.PASSED,
            ValidationSeverity.INFO,
            "Agent identity records are non-empty.",
            n_failed=0,
            n_checked=1,
        )
    )
    missing_columns = sorted(
        column for column in IDENTITY_COLUMNS if any(column not in record for record in records)
    )
    results.append(
        _validation_result(
            "agent_identity_required_columns",
            ValidationStatus.FAILED if missing_columns else ValidationStatus.PASSED,
            ValidationSeverity.ERROR if missing_columns else ValidationSeverity.INFO,
            f"Missing required identity columns: {missing_columns}."
            if missing_columns
            else "All required identity columns are present.",
            n_failed=len(missing_columns),
            n_checked=len(IDENTITY_COLUMNS),
        )
    )
    country_sectors = [str(record.get("country_sector", "")).strip() for record in records]
    duplicate_count = len(country_sectors) - len(set(country_sectors))
    results.append(
        _validation_result(
            "agent_identity_country_sector_unique",
            ValidationStatus.FAILED if duplicate_count else ValidationStatus.PASSED,
            ValidationSeverity.ERROR if duplicate_count else ValidationSeverity.INFO,
            f"Found {duplicate_count} duplicate country_sector values."
            if duplicate_count
            else "country_sector values are unique.",
            n_failed=duplicate_count,
            n_checked=n_records,
        )
    )
    empty_country_sectors = sum(1 for value in country_sectors if not value)
    results.append(
        _validation_result(
            "agent_identity_country_sector_non_empty",
            ValidationStatus.FAILED if empty_country_sectors else ValidationStatus.PASSED,
            ValidationSeverity.ERROR if empty_country_sectors else ValidationSeverity.INFO,
            f"Found {empty_country_sectors} empty country_sector values."
            if empty_country_sectors
            else "country_sector values are non-empty.",
            n_failed=empty_country_sectors,
            n_checked=n_records,
        )
    )
    country_present = sum(1 for record in records if str(record.get("country") or "").strip())
    sector_present = sum(1 for record in records if str(record.get("sector") or "").strip())
    country_share = country_present / n_records
    sector_share = sector_present / n_records
    coverage_failures = int(country_share < 0.95) + int(sector_share < 0.95)
    results.append(
        _validation_result(
            "agent_identity_country_sector_coverage",
            ValidationStatus.FAILED if coverage_failures else ValidationStatus.PASSED,
            ValidationSeverity.ERROR if coverage_failures else ValidationSeverity.INFO,
            "country and sector coverage must each be at least 95 percent; "
            f"observed country={country_share:.3f}, sector={sector_share:.3f}.",
            n_failed=coverage_failures,
            n_checked=2,
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


def _has_error_failures(results: tuple[ValidationResult, ...]) -> bool:
    return any(
        result.status is ValidationStatus.FAILED
        and result.severity in {ValidationSeverity.ERROR, ValidationSeverity.CRITICAL}
        for result in results
    )


def _normalize_column_name(column_name: str) -> str:
    return column_name.strip().lower().replace(" ", "_")


def _deduplicate_records(records: list[dict[str, str | None]]) -> list[dict[str, str | None]]:
    deduplicated: list[dict[str, str | None]] = []
    seen: set[str] = set()
    for record in records:
        country_sector = str(record.get("country_sector") or "").strip()
        if not country_sector or country_sector in seen:
            continue
        seen.add(country_sector)
        deduplicated.append({column: record.get(column) for column in IDENTITY_COLUMNS})
    return deduplicated


def _read_atlas_label_records(path: Path) -> list[dict[str, str | None]]:
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            return []
        normalized_names = {_normalize_column_name(name): name for name in reader.fieldnames}
        full_label_source = next(
            (
                normalized_names[name]
                for name in ("country_sector", "label", "eora_label", "full_label")
                if name in normalized_names
            ),
            None,
        )
        records: list[dict[str, str | None]] = []
        for row in reader:
            normalized_row = {
                _normalize_column_name(key): (value.strip() if isinstance(value, str) else value)
                for key, value in row.items()
                if key is not None
            }
            record = {column: normalized_row.get(column) or None for column in IDENTITY_COLUMNS}
            if not record["country_sector"] and full_label_source is not None:
                record["country_sector"] = (row.get(full_label_source) or "").strip() or None
            if record["country_sector"]:
                parsed = parse_eora_country_sector_label(str(record["country_sector"]))
                for column in IDENTITY_COLUMNS:
                    if not record.get(column):
                        record[column] = parsed.get(column)
            records.append(record)
    return _deduplicate_records(records)


def _read_labels_t_records(path: Path) -> list[dict[str, str | None]]:
    return build_agent_identity_from_labels(path.read_text(encoding="utf-8-sig").splitlines())


def _write_validation_json(
    path: Path,
    source_used: str,
    results: tuple[ValidationResult, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "validation_scope": "abm_v5_agent_identity",
                "source_used": source_used,
                "results": _validation_results_to_dict(results),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_records_parquet(records: list[dict[str, str | None]], path: Path) -> None:
    import polars as pl

    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(records).select(list(IDENTITY_COLUMNS)).write_parquet(path)


def build_agent_identity_panel(
    project_root: Path,
    preferred_source: str = "atlas_labels",
) -> AgentIdentityBuildResult:
    """Build the canonical ABM v5 country-sector identity panel."""
    paths = ABMV5Paths.from_project_root(project_root)
    paths.inputs.mkdir(parents=True, exist_ok=True)
    paths.validation.mkdir(parents=True, exist_ok=True)
    output_path = paths.inputs / "agent_identity.parquet"
    validation_path = paths.validation / "agent_identity_validation.json"
    atlas_path = project_root / "data" / "atlas" / "processed" / "eora26_country_sector_labels.csv"
    labels_t_path = project_root / "data" / "raw" / "1995" / "labels_T.txt"

    records: list[dict[str, str | None]] = []
    validation_results: tuple[ValidationResult, ...] = ()
    source_used = ""

    if preferred_source == "atlas_labels" and atlas_path.is_file():
        atlas_records = _read_atlas_label_records(atlas_path)
        atlas_results = validate_agent_identity_records([dict(record) for record in atlas_records])
        if not _has_error_failures(atlas_results):
            records = atlas_records
            validation_results = atlas_results
            source_used = "atlas_labels"

    if not records:
        if not labels_t_path.is_file():
            validation_results = (
                _validation_result(
                    "agent_identity_source_available",
                    ValidationStatus.FAILED,
                    ValidationSeverity.ERROR,
                    "No valid Atlas label file or fallback data/raw/1995/labels_T.txt source is available.",
                    n_failed=1,
                    n_checked=2,
                ),
            )
            _write_validation_json(validation_path, source_used or "none", validation_results)
            raise ValueError("No valid source is available for ABM v5 agent identity.")
        records = _read_labels_t_records(labels_t_path)
        validation_results = validate_agent_identity_records([dict(record) for record in records])
        source_used = "labels_T_1995"

    _write_validation_json(validation_path, source_used, validation_results)
    if _has_error_failures(validation_results):
        raise ValueError("ABM v5 agent identity validation failed.")

    _write_records_parquet(records, output_path)
    result = AgentIdentityBuildResult(
        output_path=output_path,
        validation_path=validation_path,
        n_agents=len(records),
        source_used=source_used,
    )
    result.validate()
    return result


def load_agent_identity_panel(path: Path):
    """Read a generated ABM v5 agent identity parquet file."""
    import polars as pl

    return pl.read_parquet(path)
