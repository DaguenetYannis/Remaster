from pathlib import Path
from uuid import uuid4

import polars as pl

from src.abm_v4.ecosystem import EcosystemAssignment, EcosystemMapper
from src.abm_v4.paths import ABMV4Paths


def toy_root() -> Path:
    return Path("tmp") / "abm_v4_ecosystem_tests" / uuid4().hex


def toy_state_panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "country_sector": [
                "FRA|Agriculture",
                "DEU|Agriculture",
                "FRA|Mining and Quarrying",
                "USA|Unknown Sector",
                "ROW|Missing",
            ],
            "Year": [1995, 1995, 1995, 1995, 1995],
            "Sector": [
                "Agriculture",
                "Agriculture",
                "Mining and Quarrying",
                "Unknown Sector",
                None,
            ],
            "X_observed": [1.0, 2.0, 3.0, 4.0, 5.0],
            "EI": [0.1, 0.2, 0.3, 0.4, 0.5],
            "ecosystem_id": ["fallback_unknown"] * 5,
            "ecosystem_label": ["Unknown ecosystem"] * 5,
            "ecosystem_source": ["fallback_unknown"] * 5,
        }
    )


def test_ecosystem_assignment_validates_source_vocabulary() -> None:
    assignment = EcosystemAssignment(
        sector="Agriculture",
        ecosystem_id="eco_food",
        ecosystem_label="Food system",
        ecosystem_source="eora_sector_manual_mapping",
    )

    assert assignment.is_known_source()


def test_manual_fallback_assigns_every_sector_and_source() -> None:
    mapper = EcosystemMapper(paths=ABMV4Paths(project_root=toy_root()))

    result = mapper.assign_ecosystems(toy_state_panel())

    assert result.ecosystem_source == "eora_sector_manual_mapping"
    assert result.state_panel["ecosystem_id"].null_count() == 0
    assert result.state_panel["ecosystem_label"].null_count() == 0
    assert result.state_panel["ecosystem_source"].null_count() == 0
    assert "fallback_unknown" in result.state_panel["ecosystem_source"].to_list()
    assert "eora_sector_manual_mapping" in result.state_panel["ecosystem_source"].to_list()


def test_adjacency_contains_same_adjacent_and_non_adjacent_relations() -> None:
    mapper = EcosystemMapper(
        paths=ABMV4Paths(project_root=toy_root()),
        eta_ecosystem_adjacent=0.42,
    )
    mapping = mapper.build_manual_sector_mapping(
        [
            "Agriculture",
            "Petroleum, Chemical and Non-Metallic Mineral Products",
            "Public Administration",
        ]
    )

    adjacency = mapper.build_adjacency(mapping)

    same = adjacency.filter(
        (pl.col("ecosystem_id_from") == "agriculture_food_biomass")
        & (pl.col("ecosystem_id_to") == "agriculture_food_biomass")
    )
    adjacent = adjacency.filter(
        (pl.col("ecosystem_id_from") == "agriculture_food_biomass")
        & (pl.col("ecosystem_id_to") == "basic_materials_chemicals")
    )
    non_adjacent = adjacency.filter(
        (pl.col("ecosystem_id_from") == "agriculture_food_biomass")
        & (pl.col("ecosystem_id_to") == "public_social_household_services")
    )

    assert same["relation_type"].item() == "same"
    assert same["proximity"].item() == 1.0
    assert adjacent["relation_type"].item() == "adjacent"
    assert adjacent["proximity"].item() == 0.42
    assert non_adjacent["relation_type"].item() == "non_adjacent"
    assert non_adjacent["proximity"].item() == 0.0


def test_report_counts_unmapped_nodes() -> None:
    mapper = EcosystemMapper(paths=ABMV4Paths(project_root=toy_root()))

    result = mapper.assign_ecosystems(toy_state_panel())
    report = result.assignment_report.to_dicts()[0]

    assert report["total_country_sector_nodes"] == 5
    assert report["unmapped_nodes"] == 1
    assert report["mapped_nodes"] == 4


def test_write_outputs_creates_only_ecosystem_and_state_files() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    state_panel_path = paths.state_panel_path(1995, 2016)
    mapper = EcosystemMapper(paths=paths)
    result = mapper.assign_ecosystems(toy_state_panel())

    mapper.write_outputs(result, state_panel_path=state_panel_path)

    assert paths.ecosystem_mapping_path.exists()
    assert paths.ecosystem_adjacency_path.exists()
    assert paths.ecosystem_assignment_report_path.exists()
    assert paths.ecosystem_sector_coverage_path.exists()
    assert state_panel_path.exists()
    assert not paths.simulations.exists()
    assert not paths.scenarios.exists()
    assert not paths.interim.exists()
    assert not paths.validation.exists()
