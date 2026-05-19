import json
from pathlib import Path

import pytest

from src.abm_v5 import (
    DataSourceKind,
    DataSourceSpec,
    DataSourceStatus,
    InputRegistry,
    ValidationStatus,
    build_default_input_registry,
    validate_input_registry_paths,
    write_default_input_registry_files,
)


def test_data_source_spec_requires_year_placeholder_when_years_required() -> None:
    source = DataSourceSpec(
        name="bad_source",
        kind=DataSourceKind.EORA_MATRIX,
        relative_path_template="data/parquet/T.parquet",
        status=DataSourceStatus.REQUIRED,
        description="Bad source.",
        years_required=True,
    )

    with pytest.raises(ValueError, match="year placeholder"):
        source.validate()


def test_data_source_spec_resolves_year_path(tmp_path: Path) -> None:
    source = DataSourceSpec(
        name="eora_T",
        kind=DataSourceKind.EORA_MATRIX,
        relative_path_template="data/parquet/{year}/T.parquet",
        status=DataSourceStatus.REQUIRED,
        description="Eora T matrix.",
        years_required=True,
    )

    assert source.resolve(tmp_path, 1995) == tmp_path / "data" / "parquet" / "1995" / "T.parquet"


def test_input_registry_default_years_are_1995_2016() -> None:
    registry = build_default_input_registry()

    assert registry.years()[0] == 1995
    assert registry.years()[-1] == 2016
    assert len(registry.years()) == 22


def test_input_registry_source_names_are_unique() -> None:
    registry = build_default_input_registry()

    assert len(registry.source_names()) == len(set(registry.source_names()))


def test_default_input_registry_contains_required_eora_sources() -> None:
    registry = build_default_input_registry()

    assert {"eora_T", "eora_FD", "eora_Q", "labels_T", "labels_FD", "labels_Q"}.issubset(
        set(registry.source_names())
    )
    assert registry.get_source("eora_T").status is DataSourceStatus.REQUIRED
    assert registry.get_source("labels_Q").years_required


def test_default_input_registry_contains_atlas_sources() -> None:
    registry = build_default_input_registry()

    assert registry.get_source("atlas_sector_capabilities").kind is DataSourceKind.ATLAS_PROCESSED
    assert registry.get_source("eora26_country_sector_labels").status is DataSourceStatus.REQUIRED


def test_default_input_registry_contains_abm_v4_reference_sources() -> None:
    registry = build_default_input_registry()

    assert registry.get_source("abm_v4_data_dir").status is DataSourceStatus.OPTIONAL
    assert registry.get_source("abm_v4_src_dir").kind is DataSourceKind.ABM_V4_REFERENCE
    assert registry.get_source("abm_v4_tests_dir").kind is DataSourceKind.ABM_V4_REFERENCE


def test_validate_input_registry_paths_reports_missing_required_sources(tmp_path: Path) -> None:
    registry = InputRegistry(
        historical_start_year=1995,
        historical_end_year=2016,
        sources=(
            DataSourceSpec(
                name="required_file",
                kind=DataSourceKind.EORA_MATRIX,
                relative_path_template="data/parquet/{year}/T.parquet",
                status=DataSourceStatus.REQUIRED,
                description="Required yearly source.",
                years_required=True,
            ),
        ),
    )

    results = validate_input_registry_paths(registry, tmp_path)

    assert len(results) == 1
    assert results[0].status is ValidationStatus.FAILED
    assert results[0].n_failed == 22


def test_validate_input_registry_paths_skips_optional_sources_by_default(tmp_path: Path) -> None:
    registry = InputRegistry(
        historical_start_year=1995,
        historical_end_year=2016,
        sources=(
            DataSourceSpec(
                name="optional_dir",
                kind=DataSourceKind.ABM_V4_REFERENCE,
                relative_path_template="data/abm_v4/",
                status=DataSourceStatus.OPTIONAL,
                description="Optional reference directory.",
            ),
        ),
    )

    results = validate_input_registry_paths(registry, tmp_path)

    assert results[0].status is ValidationStatus.SKIPPED
    assert results[0].n_checked == 0


def test_write_default_input_registry_files_writes_json_outputs(tmp_path: Path) -> None:
    registry_path, validation_path = write_default_input_registry_files(tmp_path)

    assert registry_path == tmp_path / "data" / "abm_v5" / "inputs" / "input_registry.json"
    assert validation_path == tmp_path / "data" / "abm_v5" / "validation" / "input_registry_validation.json"
    assert registry_path.is_file()
    assert validation_path.is_file()

    registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    validation_payload = json.loads(validation_path.read_text(encoding="utf-8"))

    assert registry_payload["historical_start_year"] == 1995
    assert validation_payload["validation_scope"] == "abm_v5_phase_2_1_input_registry"
    assert validation_payload["results"]


def test_phase_2_source_audit_doc_exists() -> None:
    assert (Path("docs") / "abm_v5" / "PHASE_2_SOURCE_AUDIT.md").is_file()


def test_init_exports_data_source_objects() -> None:
    import src.abm_v5 as abm_v5

    assert abm_v5.DataSourceKind is DataSourceKind
    assert abm_v5.DataSourceStatus is DataSourceStatus
    assert abm_v5.DataSourceSpec is DataSourceSpec
    assert abm_v5.InputRegistry is InputRegistry
    assert abm_v5.build_default_input_registry is build_default_input_registry
    assert abm_v5.validate_input_registry_paths is validate_input_registry_paths
