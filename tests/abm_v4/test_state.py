from pathlib import Path
from uuid import uuid4

import polars as pl

from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.state import (
    build_capability_coverage_by_sector,
    build_capability_coverage_by_year,
    build_capability_join_report,
    build_state_panel,
    discover_state_source,
    inspect_capability_columns,
    join_capability_variables,
    repair_capability_coverage,
    select_capability_join_keys,
)


def toy_root() -> Path:
    return Path("tmp") / "abm_v4_state_tests" / uuid4().hex


def toy_source_panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "country_sector": [
                "FRA|Agriculture",
                "DEU|Agriculture",
                "FRA|Manufacturing",
                "DEU|Manufacturing",
            ],
            "year": [1995, 1995, 1996, 2017],
            "output": [100.0, 200.0, 50.0, 90.0],
            "emissions_intensity": [0.5, 1.0, -0.5, 0.8],
            "country": ["FRA", "DEU", "FRA", "DEU"],
            "sector": ["Agriculture", "Agriculture", "Manufacturing", "Manufacturing"],
            "g_in": [0.3, 0.6, None, 0.2],
            "pagerank": [0.1, 0.2, 0.3, 0.4],
            "capability_mean_pci": [1.0, 2.0, 3.0, 4.0],
            "green_capability_share": [0.2, 0.4, 0.6, 0.8],
        }
    )


def write_toy_source(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    toy_source_panel().write_parquet(path)


def test_state_source_reports_missing_inputs() -> None:
    root = toy_root()
    report = discover_state_source(ABMV4Paths(project_root=root), 1995, 2016)

    assert not report.has_source
    assert report.selected_source is None
    assert "No valid" in report.message


def test_state_builder_selects_highest_priority_available_source() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    v3_source = paths.abm_v3_state_candidates(1995, 2016)[0]
    final_source = paths.data_final / "eora_atlas_dynamic_panel.parquet"
    write_toy_source(v3_source)
    write_toy_source(final_source)

    result = build_state_panel(paths, 1995, 2016)

    assert result.selected_source == v3_source


def test_state_builder_canonicalizes_column_variants() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_source(paths.final_state_candidates[0])

    result = build_state_panel(paths, 1995, 2016)

    assert {"Year", "X_observed", "EI", "Country", "Sector"}.issubset(
        set(result.state_panel.columns)
    )
    assert result.state_panel["Year"].to_list() == [1995, 1995, 1996]
    assert result.state_panel["X_observed"].to_list() == [100.0, 200.0, 50.0]


def test_state_builder_computes_g_local_v4_within_year() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_source(paths.final_state_candidates[0])

    state_panel = build_state_panel(paths, 1995, 2016).state_panel
    year_1995 = state_panel.filter(pl.col("Year") == 1995).sort("country_sector")

    assert year_1995["g_local_v4"].to_list() == [0.0, 1.0]


def test_state_builder_preserves_invalid_ei_without_zero_conversion() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_source(paths.final_state_candidates[0])

    state_panel = build_state_panel(paths, 1995, 2016).state_panel
    invalid_row = state_panel.filter(pl.col("country_sector") == "FRA|Manufacturing")

    assert invalid_row["EI"].item() == -0.5
    assert invalid_row["log_EI"].item() is None


def test_state_builder_computes_brown_centrality_when_inputs_exist() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_source(paths.final_state_candidates[0])

    state_panel = build_state_panel(paths, 1995, 2016).state_panel
    year_1995 = state_panel.filter(pl.col("Year") == 1995).sort("country_sector")

    assert year_1995["brown_centrality"].to_list() == [1.0, 0.25]


def test_state_builder_writes_diagnostics_only_when_enabled() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_source(paths.final_state_candidates[0])

    no_write_result = build_state_panel(paths, 1995, 2016, write_outputs=False)

    assert no_write_result.output_path is None
    assert not paths.data_abm_v4.exists()

    write_result = build_state_panel(paths, 1995, 2016, write_outputs=True)

    assert write_result.output_path == paths.state_panel_path(1995, 2016)
    assert paths.state_panel_path(1995, 2016).exists()
    assert (paths.diagnostics / "state_source_report.csv").exists()
    assert (paths.diagnostics / "state_missingness_report.csv").exists()
    assert (paths.diagnostics / "state_summary_by_year.csv").exists()
    assert (paths.diagnostics / "state_column_mapping.csv").exists()
    assert not paths.interim.exists()
    assert not paths.simulations.exists()
    assert not paths.scenarios.exists()
    assert not paths.validation.exists()


def toy_state_for_capability_join() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "country_sector": ["FRA|Agriculture", "DEU|Fishing", "ESP|Mining"],
            "Year": [2016, 2016, 2016],
            "Country": ["FRA", "DEU", "ESP"],
            "Sector": ["Agriculture", "Fishing", "Mining"],
            "general_capability": [None, 0.8, None],
            "green_capability": [None, 0.2, None],
        }
    )


def toy_atlas_capability_panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "iso3Code": ["FRA", "DEU"],
            "year": [2016, 2016],
            "eora26_sector": ["Agriculture", "Fishing"],
            "capability_export_weighted_pci": [1.5, 9.9],
            "capability_mean_pci": [1.0, 9.0],
            "active_good_count": [5.0, 8.0],
            "green_capability_export_share": [0.4, 0.9],
            "green_capability_share": [0.3, 0.8],
        }
    )


def test_capability_join_selects_valid_keys() -> None:
    selected = select_capability_join_keys(
        toy_state_for_capability_join(),
        toy_atlas_capability_panel(),
    )

    assert selected == (
        ("Country", "iso3Code"),
        ("Year", "year"),
        ("Sector", "eora26_sector"),
    )


def test_capability_join_improves_missingness_when_source_has_data() -> None:
    state = toy_state_for_capability_join()
    atlas = toy_atlas_capability_panel()
    selected = select_capability_join_keys(state, atlas)

    repaired = join_capability_variables(state, atlas, selected)

    assert state["general_capability"].null_count() == 2
    assert repaired["general_capability"].null_count() == 1
    assert repaired["green_capability"].null_count() == 1


def test_capability_join_preserves_existing_canonical_values() -> None:
    state = toy_state_for_capability_join()
    atlas = toy_atlas_capability_panel()
    selected = select_capability_join_keys(state, atlas)

    repaired = join_capability_variables(state, atlas, selected)
    deu = repaired.filter(pl.col("Country") == "DEU").row(0, named=True)

    assert deu["general_capability"] == 0.8
    assert deu["green_capability"] == 0.2
    assert deu["general_capability_source"] == "existing_state"


def test_capability_join_records_sources_and_unmatched_rows() -> None:
    state = toy_state_for_capability_join()
    atlas = toy_atlas_capability_panel()
    selected = select_capability_join_keys(state, atlas)
    repaired = join_capability_variables(state, atlas, selected)
    inspection = inspect_capability_columns(state, atlas)

    report = build_capability_join_report(
        state_before=state,
        state_after=repaired,
        source_file=Path("atlas.parquet"),
        selected_join_keys=selected,
        column_inspection=inspection,
    )
    row = report.to_dicts()[0]

    assert row["matched_rows"] == 2
    assert row["unmatched_rows"] == 1
    assert "Country=iso3Code" in row["selected_join_keys"]
    assert repaired.filter(pl.col("Country") == "FRA")["general_capability_source"].item() == "atlas_capability_join"


def test_capability_coverage_by_year_and_sector_is_computed() -> None:
    repaired = join_capability_variables(
        toy_state_for_capability_join(),
        toy_atlas_capability_panel(),
        select_capability_join_keys(toy_state_for_capability_join(), toy_atlas_capability_panel()),
    )

    by_year = build_capability_coverage_by_year(repaired)
    by_sector = build_capability_coverage_by_sector(repaired)

    assert by_year["rows"].item() == 3
    assert "general_capability_missing_share" in by_sector.columns
    assert by_sector.height == 3


def test_capability_repair_writes_only_when_enabled() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    state_path = paths.state_panel_path(1995, 2016)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    toy_state_for_capability_join().write_parquet(state_path)
    atlas_path = paths.data_atlas / "processed" / "atlas_eora26_sector_capabilities_1995_2016.parquet"
    atlas_path.parent.mkdir(parents=True, exist_ok=True)
    toy_atlas_capability_panel().write_parquet(atlas_path)

    no_write = repair_capability_coverage(paths, 1995, 2016, write_outputs=False)

    assert no_write.output_path is None
    assert not paths.capability_join_report_path.exists()

    write_result = repair_capability_coverage(paths, 1995, 2016, write_outputs=True)

    assert write_result.output_path == state_path
    assert paths.capability_join_report_path.exists()
    assert paths.capability_coverage_by_year_path.exists()
    assert paths.capability_coverage_by_sector_path.exists()
