from pathlib import Path
from uuid import uuid4

import polars as pl

from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.state import build_state_panel, discover_state_source


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
