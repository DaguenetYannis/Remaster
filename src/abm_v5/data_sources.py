from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from src.abm_v5.config import DEFAULT_HISTORICAL_END_YEAR, DEFAULT_HISTORICAL_START_YEAR, ValidationLayer
from src.abm_v5.paths import ABMV5Paths
from src.abm_v5.validation import ValidationResult, ValidationSeverity, ValidationStatus


class DataSourceKind(str, Enum):
    """Kinds of ABM v5 source inputs and references."""

    EORA_MATRIX = "eora_matrix"
    EORA_LABELS = "eora_labels"
    ATLAS_PROCESSED = "atlas_processed"
    ABM_V3_REFERENCE = "abm_v3_reference"
    ABM_V4_REFERENCE = "abm_v4_reference"
    ABM_V5_OUTPUT = "abm_v5_output"


class DataSourceStatus(str, Enum):
    """Requiredness status for ABM v5 input registry sources."""

    REQUIRED = "required"
    OPTIONAL = "optional"
    DESIGN_TARGET = "design_target"


@dataclass(frozen=True)
class DataSourceSpec:
    """Metadata for one ABM v5 source path."""

    name: str
    kind: DataSourceKind
    relative_path_template: str
    status: DataSourceStatus
    description: str
    years_required: bool = False

    def validate(self) -> None:
        """Validate source metadata without checking filesystem state."""
        if not self.name:
            raise ValueError("name cannot be empty.")
        if not isinstance(self.kind, DataSourceKind):
            raise ValueError(f"kind must be DataSourceKind for {self.name}.")
        if not self.relative_path_template:
            raise ValueError(f"relative_path_template cannot be empty for {self.name}.")
        if not isinstance(self.status, DataSourceStatus):
            raise ValueError(f"status must be DataSourceStatus for {self.name}.")
        if not self.description:
            raise ValueError(f"description cannot be empty for {self.name}.")
        if self.years_required and "{year}" not in self.relative_path_template:
            raise ValueError(f"{self.name} requires a year placeholder.")
        if not self.years_required and "{year}" in self.relative_path_template:
            raise ValueError(f"{self.name} has a year placeholder but years_required is False.")

    def resolve(self, project_root: Path, year: int | None = None) -> Path:
        """Resolve this source path against a project root."""
        self.validate()
        if self.years_required:
            if year is None:
                raise ValueError(f"year is required to resolve {self.name}.")
            relative_path = self.relative_path_template.format(year=year)
        else:
            relative_path = self.relative_path_template
        return project_root / Path(relative_path)


@dataclass(frozen=True)
class InputRegistry:
    """ABM v5 historical input registry."""

    historical_start_year: int
    historical_end_year: int
    sources: tuple[DataSourceSpec, ...]

    def years(self) -> list[int]:
        """Return the inclusive historical years for ABM v5 Phase 2."""
        return list(range(self.historical_start_year, self.historical_end_year + 1))

    def validate(self) -> None:
        """Validate registry metadata without checking source existence."""
        if self.historical_start_year != DEFAULT_HISTORICAL_START_YEAR:
            raise ValueError("historical_start_year must be 1995.")
        if self.historical_end_year != DEFAULT_HISTORICAL_END_YEAR:
            raise ValueError("historical_end_year must be 2016.")
        source_names = self.source_names()
        if len(source_names) != len(set(source_names)):
            raise ValueError("source names must be unique.")
        for source in self.sources:
            source.validate()

    def source_names(self) -> tuple[str, ...]:
        """Return source names in registry order."""
        return tuple(source.name for source in self.sources)

    def get_source(self, name: str) -> DataSourceSpec:
        """Return a source specification by name."""
        for source in self.sources:
            if source.name == name:
                return source
        raise KeyError(f"Unknown input source: {name}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable registry dictionary."""
        return {
            "historical_start_year": self.historical_start_year,
            "historical_end_year": self.historical_end_year,
            "sources": [
                {
                    "name": source.name,
                    "kind": source.kind.value,
                    "relative_path_template": source.relative_path_template,
                    "status": source.status.value,
                    "description": source.description,
                    "years_required": source.years_required,
                }
                for source in self.sources
            ],
        }

    def write_json(self, path: Path) -> None:
        """Write the registry metadata to JSON."""
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def _source(
    name: str,
    kind: DataSourceKind,
    relative_path_template: str,
    status: DataSourceStatus,
    description: str,
    years_required: bool = False,
) -> DataSourceSpec:
    return DataSourceSpec(
        name=name,
        kind=kind,
        relative_path_template=relative_path_template,
        status=status,
        description=description,
        years_required=years_required,
    )


def build_default_input_registry() -> InputRegistry:
    """Build the default ABM v5 Phase 2.1 input registry."""
    registry = InputRegistry(
        historical_start_year=DEFAULT_HISTORICAL_START_YEAR,
        historical_end_year=DEFAULT_HISTORICAL_END_YEAR,
        sources=(
            _source(
                "eora_T",
                DataSourceKind.EORA_MATRIX,
                "data/parquet/{year}/T.parquet",
                DataSourceStatus.REQUIRED,
                "Eora transaction matrix by historical year.",
                years_required=True,
            ),
            _source(
                "eora_FD",
                DataSourceKind.EORA_MATRIX,
                "data/parquet/{year}/FD.parquet",
                DataSourceStatus.REQUIRED,
                "Eora final demand matrix by historical year.",
                years_required=True,
            ),
            _source(
                "eora_Q",
                DataSourceKind.EORA_MATRIX,
                "data/parquet/{year}/Q.parquet",
                DataSourceStatus.REQUIRED,
                "Eora environmental satellite account matrix by historical year.",
                years_required=True,
            ),
            _source(
                "eora_VA",
                DataSourceKind.EORA_MATRIX,
                "data/parquet/{year}/VA.parquet",
                DataSourceStatus.OPTIONAL,
                "Optional Eora value-added matrix by historical year.",
                years_required=True,
            ),
            _source(
                "eora_QY",
                DataSourceKind.EORA_MATRIX,
                "data/parquet/{year}/QY.parquet",
                DataSourceStatus.OPTIONAL,
                "Optional Eora final-demand satellite matrix by historical year.",
                years_required=True,
            ),
            _source(
                "labels_T",
                DataSourceKind.EORA_LABELS,
                "data/raw/{year}/labels_T.txt",
                DataSourceStatus.REQUIRED,
                "Raw Eora labels for transaction matrix rows and columns.",
                years_required=True,
            ),
            _source(
                "labels_FD",
                DataSourceKind.EORA_LABELS,
                "data/raw/{year}/labels_FD.txt",
                DataSourceStatus.REQUIRED,
                "Raw Eora labels for final demand columns.",
                years_required=True,
            ),
            _source(
                "labels_Q",
                DataSourceKind.EORA_LABELS,
                "data/raw/{year}/labels_Q.txt",
                DataSourceStatus.REQUIRED,
                "Raw Eora labels for environmental satellite rows.",
                years_required=True,
            ),
            _source(
                "atlas_sector_capabilities",
                DataSourceKind.ATLAS_PROCESSED,
                "data/atlas/processed/atlas_eora26_sector_capabilities_1995_2016.parquet",
                DataSourceStatus.REQUIRED,
                "Processed Atlas capability panel aggregated to Eora26 sectors.",
            ),
            _source(
                "eora26_country_sector_labels",
                DataSourceKind.ATLAS_PROCESSED,
                "data/atlas/processed/eora26_country_sector_labels.csv",
                DataSourceStatus.REQUIRED,
                "Processed country-sector labels for Eora26-aligned nodes.",
            ),
            _source(
                "abm_v3_phase_space",
                DataSourceKind.ABM_V3_REFERENCE,
                "data/abm_v3/phase_space/abm_v3_phase_space_state_panel_1995_2016.parquet",
                DataSourceStatus.OPTIONAL,
                "Optional ABM v3 phase-space panel used only as construction reference.",
            ),
            _source(
                "abm_v3_inputs_dir",
                DataSourceKind.ABM_V3_REFERENCE,
                "data/abm_v3/inputs/",
                DataSourceStatus.OPTIONAL,
                "Optional ABM v3 inputs directory used only as construction reference.",
            ),
            _source(
                "abm_v4_data_dir",
                DataSourceKind.ABM_V4_REFERENCE,
                "data/abm_v4/",
                DataSourceStatus.OPTIONAL,
                "Optional ABM v4 generated data directory used only as construction reference.",
            ),
            _source(
                "abm_v4_src_dir",
                DataSourceKind.ABM_V4_REFERENCE,
                "src/abm_v4/",
                DataSourceStatus.OPTIONAL,
                "Optional ABM v4 source directory used only to audit construction patterns.",
            ),
            _source(
                "abm_v4_tests_dir",
                DataSourceKind.ABM_V4_REFERENCE,
                "tests/abm_v4/",
                DataSourceStatus.OPTIONAL,
                "Optional ABM v4 tests directory used only to audit construction coverage.",
            ),
        ),
    )
    registry.validate()
    return registry


def _validation_result(
    source: DataSourceSpec,
    status: ValidationStatus,
    severity: ValidationSeverity,
    message: str,
    n_failed: int,
    n_checked: int,
) -> ValidationResult:
    result = ValidationResult(
        check_name=f"input_source_exists:{source.name}",
        status=status,
        severity=severity,
        message=message,
        layer=ValidationLayer.STRUCTURAL_VALIDITY,
        n_failed=n_failed,
        n_checked=n_checked,
    )
    result.validate()
    return result


def validate_input_registry_paths(
    registry: InputRegistry,
    project_root: Path,
    check_optional: bool = False,
) -> tuple[ValidationResult, ...]:
    """Validate registered source path existence without loading file contents."""
    registry.validate()
    results: list[ValidationResult] = []

    for source in registry.sources:
        if source.status is DataSourceStatus.OPTIONAL and not check_optional:
            results.append(
                _validation_result(
                    source=source,
                    status=ValidationStatus.SKIPPED,
                    severity=ValidationSeverity.INFO,
                    message=f"Optional source {source.name} was not checked.",
                    n_failed=0,
                    n_checked=0,
                )
            )
            continue

        paths = (
            tuple(source.resolve(project_root, year) for year in registry.years())
            if source.years_required
            else (source.resolve(project_root),)
        )
        missing_paths = tuple(path for path in paths if not path.exists())
        if missing_paths:
            severity = (
                ValidationSeverity.ERROR
                if source.status is DataSourceStatus.REQUIRED
                else ValidationSeverity.WARNING
            )
            results.append(
                _validation_result(
                    source=source,
                    status=ValidationStatus.FAILED,
                    severity=severity,
                    message=(
                        f"{source.status.value} source {source.name} is missing "
                        f"{len(missing_paths)} of {len(paths)} registered paths."
                    ),
                    n_failed=len(missing_paths),
                    n_checked=len(paths),
                )
            )
        else:
            results.append(
                _validation_result(
                    source=source,
                    status=ValidationStatus.PASSED,
                    severity=ValidationSeverity.INFO,
                    message=f"Source {source.name} exists for all registered paths.",
                    n_failed=0,
                    n_checked=len(paths),
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


def write_default_input_registry_files(project_root: Path) -> tuple[Path, Path]:
    """Write the default registry and path validation outputs to ABM v5 directories."""
    registry = build_default_input_registry()
    results = validate_input_registry_paths(registry, project_root=project_root, check_optional=False)
    paths = ABMV5Paths.from_project_root(project_root)
    registry_path = paths.inputs / "input_registry.json"
    validation_path = paths.validation / "input_registry_validation.json"

    registry.write_json(registry_path)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(
        json.dumps(
            {
                "validation_scope": "abm_v5_phase_2_1_input_registry",
                "check_optional": False,
                "results": _validation_results_to_dict(results),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return registry_path, validation_path
