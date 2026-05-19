import json
from pathlib import Path

from src.abm_v5 import (
    AgentIdentityBuildResult,
    ValidationStatus,
    build_agent_identity_from_labels,
    build_agent_identity_panel,
    load_agent_identity_panel,
    parse_eora_country_sector_label,
    validate_agent_identity_records,
)


def write_pyproject(root: Path) -> None:
    (root / "pyproject.toml").write_text("[project]\nname = 'toy'\n", encoding="utf-8")


def test_parse_eora_country_sector_label_four_parts() -> None:
    parsed = parse_eora_country_sector_label("USA | United States | Industry | Electricity")

    assert parsed == {
        "country_sector": "USA | United States | Industry | Electricity",
        "country": "USA",
        "country_detail": "United States",
        "category": "Industry",
        "sector": "Electricity",
    }


def test_parse_eora_country_sector_label_more_than_four_parts_preserves_sector() -> None:
    parsed = parse_eora_country_sector_label("USA | United States | Industry | A | B")

    assert parsed["sector"] == "A | B"


def test_parse_eora_country_sector_label_missing_parts() -> None:
    parsed = parse_eora_country_sector_label("USA | United States")

    assert parsed["country"] == "USA"
    assert parsed["country_detail"] == "United States"
    assert parsed["category"] is None
    assert parsed["sector"] is None


def test_build_agent_identity_from_labels_deduplicates() -> None:
    records = build_agent_identity_from_labels(
        [
            "USA | United States | Industry | Electricity",
            "",
            "USA | United States | Industry | Electricity",
            "FRA | France | Industry | Steel",
        ]
    )

    assert len(records) == 2
    assert records[0]["country_sector"] == "USA | United States | Industry | Electricity"


def test_validate_agent_identity_records_passes_valid_records() -> None:
    records = [
        {
            "country_sector": "USA | United States | Industry | Electricity",
            "country": "USA",
            "country_detail": "United States",
            "category": "Industry",
            "sector": "Electricity",
        }
    ]

    results = validate_agent_identity_records(records)

    assert all(result.status is ValidationStatus.PASSED for result in results)


def test_validate_agent_identity_records_fails_duplicate_country_sector() -> None:
    records = [
        {
            "country_sector": "USA | United States | Industry | Electricity",
            "country": "USA",
            "country_detail": "United States",
            "category": "Industry",
            "sector": "Electricity",
        },
        {
            "country_sector": "USA | United States | Industry | Electricity",
            "country": "USA",
            "country_detail": "United States",
            "category": "Industry",
            "sector": "Electricity",
        },
    ]

    results = validate_agent_identity_records(records)

    assert any(result.status is ValidationStatus.FAILED for result in results)


def test_build_agent_identity_panel_prefers_atlas_labels(tmp_path: Path) -> None:
    write_pyproject(tmp_path)
    atlas_dir = tmp_path / "data" / "atlas" / "processed"
    atlas_dir.mkdir(parents=True)
    (atlas_dir / "eora26_country_sector_labels.csv").write_text(
        "country_sector,Country,Country_detail,Category,Sector\n"
        "USA | United States | Industry | Electricity,USA,United States,Industry,Electricity\n",
        encoding="utf-8",
    )
    labels_dir = tmp_path / "data" / "raw" / "1995"
    labels_dir.mkdir(parents=True)
    (labels_dir / "labels_T.txt").write_text(
        "FRA | France | Industry | Steel\n",
        encoding="utf-8",
    )

    result = build_agent_identity_panel(tmp_path)
    dataframe = load_agent_identity_panel(result.output_path)

    assert result.source_used == "atlas_labels"
    assert dataframe["country"][0] == "USA"


def test_build_agent_identity_panel_falls_back_to_labels_t(tmp_path: Path) -> None:
    write_pyproject(tmp_path)
    labels_dir = tmp_path / "data" / "raw" / "1995"
    labels_dir.mkdir(parents=True)
    (labels_dir / "labels_T.txt").write_text(
        "FRA | France | Industry | Steel\n",
        encoding="utf-8",
    )

    result = build_agent_identity_panel(tmp_path)
    dataframe = load_agent_identity_panel(result.output_path)

    assert result.source_used == "labels_T_1995"
    assert dataframe["sector"][0] == "Steel"


def test_build_agent_identity_panel_writes_output_and_validation(tmp_path: Path) -> None:
    write_pyproject(tmp_path)
    labels_dir = tmp_path / "data" / "raw" / "1995"
    labels_dir.mkdir(parents=True)
    (labels_dir / "labels_T.txt").write_text(
        "FRA | France | Industry | Steel\n",
        encoding="utf-8",
    )

    result = build_agent_identity_panel(tmp_path)
    payload = json.loads(result.validation_path.read_text(encoding="utf-8"))

    assert isinstance(result, AgentIdentityBuildResult)
    assert result.output_path.is_file()
    assert result.validation_path.is_file()
    assert result.n_agents == 1
    assert payload["validation_scope"] == "abm_v5_agent_identity"


def test_init_exports_identity_objects() -> None:
    import src.abm_v5 as abm_v5

    assert abm_v5.AgentIdentityBuildResult is AgentIdentityBuildResult
    assert abm_v5.parse_eora_country_sector_label is parse_eora_country_sector_label
    assert abm_v5.build_agent_identity_from_labels is build_agent_identity_from_labels
    assert abm_v5.validate_agent_identity_records is validate_agent_identity_records
    assert abm_v5.build_agent_identity_panel is build_agent_identity_panel
    assert abm_v5.load_agent_identity_panel is load_agent_identity_panel
