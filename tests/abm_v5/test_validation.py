import pytest

from src.abm_v5 import (
    ValidationLayer,
    ValidationPrinciple,
    ValidationResult,
    ValidationSeverity,
    ValidationStatus,
    all_passed,
    build_default_ontology_registry,
    build_default_schema_registry,
    failed_results,
    get_validation_principles,
    run_phase1_metadata_validation,
    validate_ontology_registry_metadata,
    validate_schema_registry_metadata,
    validate_theory_to_schema_alignment,
)


def test_validation_principles_cover_all_layers() -> None:
    principles = get_validation_principles()

    for principle in principles:
        principle.validate()

    assert {principle.layer for principle in principles} == set(ValidationLayer)
    assert len(principles) == len(ValidationLayer)


def test_validation_principles_required_before_scenarios() -> None:
    principles = get_validation_principles()

    assert all(principle.required_before_scenarios for principle in principles)


def test_validation_result_passed_helper() -> None:
    result = ValidationResult(
        check_name="toy_check",
        status=ValidationStatus.PASSED,
        severity=ValidationSeverity.INFO,
        message="Toy check passed.",
        layer=ValidationLayer.STRUCTURAL_VALIDITY,
        n_failed=0,
        n_checked=1,
    )

    result.validate()

    assert result.passed()


def test_validation_result_rejects_passed_with_failures() -> None:
    result = ValidationResult(
        check_name="toy_check",
        status=ValidationStatus.PASSED,
        severity=ValidationSeverity.INFO,
        message="Invalid passed result.",
        layer=ValidationLayer.STRUCTURAL_VALIDITY,
        n_failed=1,
        n_checked=1,
    )

    with pytest.raises(ValueError, match="n_failed"):
        result.validate()


def test_validate_ontology_registry_metadata_passes_default_registry() -> None:
    registry = build_default_ontology_registry()
    results = validate_ontology_registry_metadata(registry)

    assert results
    assert all_passed(results)


def test_validate_schema_registry_metadata_passes_default_registry() -> None:
    registry = build_default_schema_registry()
    results = validate_schema_registry_metadata(registry)

    assert results
    assert all_passed(results)


def test_validate_theory_to_schema_alignment_passes_defaults() -> None:
    ontology_registry = build_default_ontology_registry()
    schema_registry = build_default_schema_registry()

    results = validate_theory_to_schema_alignment(ontology_registry, schema_registry)

    assert results
    assert all_passed(results)


def test_run_phase1_metadata_validation_all_passes() -> None:
    results = run_phase1_metadata_validation()

    assert results
    assert all_passed(results)
    assert failed_results(results) == ()


def test_all_passed_and_failed_results_helpers() -> None:
    passing = ValidationResult(
        check_name="passing",
        status=ValidationStatus.PASSED,
        severity=ValidationSeverity.INFO,
        message="Passing result.",
        layer=ValidationLayer.STRUCTURAL_VALIDITY,
    )
    failing = ValidationResult(
        check_name="failing",
        status=ValidationStatus.FAILED,
        severity=ValidationSeverity.ERROR,
        message="Failing result.",
        layer=ValidationLayer.MECHANISM_VALIDITY,
        n_failed=1,
        n_checked=1,
    )
    results = (passing, failing)

    assert not all_passed(results)
    assert failed_results(results) == (failing,)


def test_init_exports_validation_objects() -> None:
    import src.abm_v5 as abm_v5

    assert abm_v5.ValidationSeverity is ValidationSeverity
    assert abm_v5.ValidationStatus is ValidationStatus
    assert abm_v5.ValidationPrinciple is ValidationPrinciple
    assert abm_v5.ValidationResult is ValidationResult
    assert abm_v5.get_validation_principles is get_validation_principles
    assert abm_v5.run_phase1_metadata_validation is run_phase1_metadata_validation
    assert abm_v5.all_passed is all_passed
    assert abm_v5.failed_results is failed_results
