from pathlib import Path

from src.abm_v5 import (
    ComplexityLayer,
    MechanismName,
    TheoreticalPillar,
    TheoryMapping,
    get_theory_mappings,
    validate_theory_mappings,
)


DOCS_ROOT = Path("docs") / "abm_v5"


def test_theory_mappings_cover_all_pillars() -> None:
    mappings = get_theory_mappings()

    assert {mapping.pillar for mapping in mappings} == set(TheoreticalPillar)
    assert len(mappings) == len(TheoreticalPillar)


def test_theory_mappings_validate() -> None:
    for mapping in get_theory_mappings():
        mapping.validate()


def test_theory_mappings_reference_known_mechanisms() -> None:
    for mapping in get_theory_mappings():
        assert mapping.linked_mechanisms
        assert all(isinstance(mechanism, MechanismName) for mechanism in mapping.linked_mechanisms)


def test_theory_mappings_reference_known_layers() -> None:
    for mapping in get_theory_mappings():
        assert mapping.linked_complexity_layers
        assert all(isinstance(layer, ComplexityLayer) for layer in mapping.linked_complexity_layers)


def test_validate_theory_mappings_passes() -> None:
    validate_theory_mappings()


def test_theory_documents_exist() -> None:
    assert (DOCS_ROOT / "THEORY_MAP.md").is_file()
    assert (DOCS_ROOT / "METHODOLOGICAL_POSITION.md").is_file()
    assert (DOCS_ROOT / "PHASE_1_HANDOFF.md").is_file()


def test_methodological_position_contains_central_sentence() -> None:
    text = (DOCS_ROOT / "METHODOLOGICAL_POSITION.md").read_text()

    assert (
        "ABM_v5 should explain how ecological transition can emerge from explicit "
        "country-sector mechanisms, not merely reproduce green-transition indicators."
    ) in text


def test_phase_1_handoff_mentions_no_simulation_logic() -> None:
    text = (DOCS_ROOT / "PHASE_1_HANDOFF.md").read_text()

    assert "No empirical data loading has been implemented." in text
    assert "No simulation logic has been implemented." in text
    assert "No scenario logic has been implemented." in text
    assert "metadata contracts only" in text


def test_init_exports_theory_objects() -> None:
    import src.abm_v5 as abm_v5

    assert abm_v5.TheoreticalPillar is TheoreticalPillar
    assert abm_v5.TheoryMapping is TheoryMapping
    assert abm_v5.get_theory_mappings is get_theory_mappings
    assert abm_v5.validate_theory_mappings is validate_theory_mappings
