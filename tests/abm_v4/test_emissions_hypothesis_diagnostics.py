from pathlib import Path
from uuid import uuid4

import polars as pl
import pytest

from src.abm_v4.config import ABMV4Config
from src.abm_v4.emissions import EmissionsTransitionHypothesisDiagnostics
from src.abm_v4.paths import ABMV4Paths


def _toy_paths() -> ABMV4Paths:
    return ABMV4Paths(project_root=Path("tmp") / "abm_v4_emissions_hypothesis_tests" / uuid4().hex)


def _toy_state_panel() -> pl.DataFrame:
    rows = []
    for year in range(1995, 2017):
        offset = year - 1995
        rows.extend(
            [
                {
                    "country_sector": "A_S1",
                    "Year": year,
                    "Country": "A",
                    "Sector": "S1",
                    "ecosystem_id": "eco1",
                    "EI": 2.0 - 0.03 * offset,
                    "general_capability_model": 0.8,
                    "green_capability_model": 0.7,
                    "general_capability_source": "atlas_observed",
                    "green_capability_source": "atlas_observed",
                    "network_green_exposure": 0.6,
                    "brown_centrality": 0.1,
                },
                {
                    "country_sector": "B_S1",
                    "Year": year,
                    "Country": "B",
                    "Sector": "S1",
                    "ecosystem_id": "eco1",
                    "EI": 3.0 - 0.01 * offset,
                    "general_capability_model": 0.4,
                    "green_capability_model": 0.2,
                    "general_capability_source": "io_imputed",
                    "green_capability_source": "io_imputed",
                    "network_green_exposure": 0.3,
                    "brown_centrality": 0.5,
                },
                {
                    "country_sector": "C_S2",
                    "Year": year,
                    "Country": "C",
                    "Sector": "S2",
                    "ecosystem_id": "eco2",
                    "EI": None if year == 2000 else 1.5 + 0.02 * offset,
                    "general_capability_model": 0.2,
                    "green_capability_model": 0.1,
                    "general_capability_source": "unavailable",
                    "green_capability_source": "unavailable",
                    "network_green_exposure": 0.2,
                    "brown_centrality": 0.2,
                },
            ]
        )
    return pl.DataFrame(rows)


def _write_toy_inputs(paths: ABMV4Paths, config: ABMV4Config) -> None:
    state_path = paths.state_panel_path(config.start_year, config.end_year)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    _toy_state_panel().write_parquet(state_path)
    paths.supplier_updated_weights_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "buyer_country_sector": ["A_S1", "B_S1"],
            "supplier_country_sector": ["B_S1", "A_S1"],
            "updated_weight": [1.0, 1.0],
        }
    ).write_parquet(paths.supplier_updated_weights_path)


def _diagnostics_with_panel() -> tuple[EmissionsTransitionHypothesisDiagnostics, pl.DataFrame]:
    paths = _toy_paths()
    config = ABMV4Config(start_year=1995, end_year=2016)
    _write_toy_inputs(paths, config)
    diagnostics = EmissionsTransitionHypothesisDiagnostics(paths, start_year=1995, end_year=2016)
    return diagnostics, diagnostics.build_base_transition_panel()


def test_hypothesis_targets_compute_one_three_and_five_year_annualized_rei() -> None:
    diagnostics, base = _diagnostics_with_panel()
    targets = diagnostics.compute_target_horizons(base)

    one = targets.filter((pl.col("target_name") == "one_year_rEI") & (pl.col("country_sector") == "A_S1") & (pl.col("year") == 1995))["target"].item()
    three = targets.filter((pl.col("target_name") == "three_year_rEI") & (pl.col("country_sector") == "A_S1") & (pl.col("year") == 1995))["target"].item()
    five = targets.filter((pl.col("target_name") == "five_year_rEI") & (pl.col("country_sector") == "A_S1") & (pl.col("year") == 1995))["target"].item()

    assert one == pytest.approx(__import__("math").log(2.0) - __import__("math").log(1.97))
    assert three == pytest.approx((__import__("math").log(2.0) - __import__("math").log(1.91)) / 3)
    assert five == pytest.approx((__import__("math").log(2.0) - __import__("math").log(1.85)) / 5)


def test_hypothesis_targets_exclude_invalid_ei_and_smoothed_target_exists() -> None:
    diagnostics, base = _diagnostics_with_panel()
    targets = diagnostics.compute_target_horizons(base)

    assert targets.filter((pl.col("country_sector") == "C_S2") & (pl.col("year") == 1999)).is_empty()
    assert "smoothed_one_year_rEI" in set(targets["target_name"].to_list())


def test_hypothesis_winsorized_target_clips_extremes() -> None:
    diagnostics, base = _diagnostics_with_panel()
    targets = diagnostics.compute_target_horizons(base)
    raw = targets.filter(pl.col("target_name") == "one_year_rEI")
    winsorized = targets.filter(pl.col("target_name") == "winsorized_one_year_rEI")

    assert winsorized["target"].max() <= raw["target"].quantile(0.99) + 1e-12


def test_hypothesis_sector_dominance_detects_background_vs_readiness() -> None:
    diagnostics, base = _diagnostics_with_panel()
    targets = diagnostics.compute_target_horizons(base)
    sector = diagnostics.test_h2_sector_dominance(targets)

    assert {"Sector", "sector_background_mae", "simple_readiness_mae", "readiness_improvement"} <= set(sector.columns)


def test_hypothesis_capability_source_diagnostics_split_sources() -> None:
    diagnostics, base = _diagnostics_with_panel()
    targets = diagnostics.compute_target_horizons(base)
    capability = diagnostics.test_h3_capability_measurement(targets)

    assert {"atlas_observed", "io_imputed", "unavailable"} <= set(capability["capability_source"].to_list())


def test_hypothesis_readiness_threshold_diagnostics_create_quantiles() -> None:
    diagnostics, base = _diagnostics_with_panel()
    targets = diagnostics.compute_target_horizons(base)
    threshold = diagnostics.test_h4_threshold_readiness(targets)

    assert {"readiness_quartile", "readiness_decile", "interaction"} <= set(threshold["quantile_type"].to_list())


def test_hypothesis_frontier_specs_compare_p10_p25_and_p50() -> None:
    diagnostics, base = _diagnostics_with_panel()
    targets = diagnostics.compute_target_horizons(base)
    frontier = diagnostics.test_h5_frontier_specification(base, targets)

    assert {"sector_year_p10", "sector_year_p25", "sector_year_p50"} <= set(frontier["frontier_specification"].to_list())


def test_hypothesis_macro_shocks_group_by_year_and_country() -> None:
    diagnostics, base = _diagnostics_with_panel()
    targets = diagnostics.compute_target_horizons(base)
    macro = diagnostics.test_h6_macro_shocks(targets)

    assert {"year", "country", "period"} <= set(macro["grouping_type"].to_list())


def test_hypothesis_diagnosis_table_contains_h1_to_h6() -> None:
    diagnostics, _base = _diagnostics_with_panel()
    result = diagnostics.run()

    assert set(result.hypothesis_diagnosis["hypothesis_id"].to_list()) == {"H1", "H2", "H3", "H4", "H5", "H6"}


def test_hypothesis_outputs_write_only_when_requested() -> None:
    diagnostics, _base = _diagnostics_with_panel()
    result = diagnostics.run()

    assert not diagnostics.paths.emissions_hypothesis_diagnosis_path.exists()
    diagnostics.write_outputs(result)
    assert diagnostics.paths.emissions_hypothesis_diagnosis_path.exists()
    assert diagnostics.paths.emissions_target_horizon_panel_path.exists()
    assert diagnostics.paths.emissions_hypothesis_diagnostic_report_path.exists()
