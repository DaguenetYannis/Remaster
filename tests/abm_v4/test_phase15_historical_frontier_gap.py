from pathlib import Path
from uuid import uuid4

import polars as pl
import pytest

from src.abm_v4.config import ABMV4Config
from src.abm_v4.emissions import (
    EmissionsUpdater,
    load_historical_frontier_gap_parameters,
)
from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.simulation import MultiYearBaseSimulator
from src.abm_v4.validation import build_multiyear_base_model_comparison


def _toy_paths() -> ABMV4Paths:
    return ABMV4Paths(project_root=Path("tmp") / "abm_v4_phase15_tests" / uuid4().hex)


def _state_panel() -> pl.DataFrame:
    rows = []
    for year in [1995, 1996, 1997]:
        rows.extend(
            [
                {
                    "country_sector": "A_S1",
                    "Year": year,
                    "Country": "A",
                    "Sector": "S1",
                    "ecosystem_id": "eco",
                    "X_observed": 100.0,
                    "EI": 1.0 + 0.1 * (year - 1995),
                    "emissions_observed": 100.0 * (1.0 + 0.1 * (year - 1995)),
                    "general_capability_model": 0.9,
                    "green_capability_model": 0.8,
                    "general_capability_source": "atlas_observed",
                    "green_capability_source": "atlas_observed",
                    "network_green_exposure": 0.9,
                    "brown_centrality": 0.1,
                },
                {
                    "country_sector": "B_S1",
                    "Year": year,
                    "Country": "B",
                    "Sector": "S1",
                    "ecosystem_id": "eco",
                    "X_observed": 100.0,
                    "EI": 2.0 + 0.1 * (year - 1995),
                    "emissions_observed": 100.0 * (2.0 + 0.1 * (year - 1995)),
                    "general_capability_model": 0.1,
                    "green_capability_model": 0.1,
                    "general_capability_source": "io_imputed",
                    "green_capability_source": "io_imputed",
                    "network_green_exposure": 0.1,
                    "brown_centrality": 0.8,
                },
                {
                    "country_sector": "C_S2",
                    "Year": year,
                    "Country": "C",
                    "Sector": "S2",
                    "ecosystem_id": "eco2",
                    "X_observed": 100.0,
                    "EI": 4.0 if year < 1997 else 0.5,
                    "emissions_observed": 100.0 * (4.0 if year < 1997 else 0.5),
                    "general_capability_model": 0.4,
                    "green_capability_model": 0.4,
                    "general_capability_source": "atlas_observed",
                    "green_capability_source": "atlas_observed",
                    "network_green_exposure": 0.4,
                    "brown_centrality": 0.4,
                },
            ]
        )
    return pl.DataFrame(rows)


def _write_multiyear_inputs(paths: ABMV4Paths) -> None:
    state_path = paths.state_panel_path(1995, 1997)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    _state_panel().write_parquet(state_path)


def test_historical_frontier_gap_only_computes_rei_without_readiness_terms() -> None:
    updater = EmissionsUpdater(paths=_toy_paths())
    panel = pl.DataFrame(
        {
            "invalid_EI_flag": [False, False],
            "sector_background_trend": [0.01, 0.01],
            "ei_gap": [1.0, 1.0],
            "readiness": [0.0, 99.0],
        }
    )
    out = updater.compute_historical_frontier_gap_rEI(panel, rho_gap=0.03, tau_gap=1.0)

    assert out["rEI_historical_frontier_gap_only"].to_list() == pytest.approx([0.025, 0.025])


def test_rolling_sector_p50_frontier_does_not_use_future_years() -> None:
    updater = EmissionsUpdater(paths=_toy_paths())
    state = _state_panel()
    frontier_full = updater.compute_rolling_sector_frontiers(state, 1996, quantile=0.50)
    frontier_truncated = updater.compute_rolling_sector_frontiers(
        state.filter(pl.col("Year") <= 1996),
        1996,
        quantile=0.50,
    )

    assert frontier_full.sort("Sector")["EI_frontier"].to_list() == pytest.approx(
        frontier_truncated.sort("Sector")["EI_frontier"].to_list()
    )


def test_parameter_file_is_loaded_when_provided() -> None:
    paths = _toy_paths()
    path = paths.validation / "params.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"parameter_records":{"one_year_rEI|rolling_sector_p50":{"global_parameters":{"rho_max":0.07,"tau_gap":2.5}}}}',
        encoding="utf-8",
    )

    loaded = load_historical_frontier_gap_parameters(path)
    assert loaded["rho_gap"] == pytest.approx(0.07)
    assert loaded["tau_gap"] == pytest.approx(2.5)
    assert not loaded["fallback_used"]


def test_fallback_parameters_are_reported_when_file_is_missing() -> None:
    loaded = load_historical_frontier_gap_parameters("missing_phase15_params.json")

    assert loaded["rho_gap"] == pytest.approx(0.03)
    assert loaded["tau_gap"] == pytest.approx(1.0)
    assert loaded["fallback_used"]


def test_calibrated_historical_outputs_use_separate_filenames() -> None:
    paths = _toy_paths()
    _write_multiyear_inputs(paths)
    config = ABMV4Config(start_year=1995, end_year=1997)
    base_simulator = MultiYearBaseSimulator(paths, config)
    historical = MultiYearBaseSimulator(
        paths,
        config,
        emissions_transition_mode="historical_frontier_gap_only",
    )
    base_result = base_simulator.run()
    historical_result = historical.run()

    base_simulator.write_outputs(base_result)
    original_bytes = paths.base_multiyear_state_panel_path.read_bytes()
    historical.write_outputs(historical_result)

    assert paths.base_multiyear_state_panel_path.read_bytes() == original_bytes
    assert paths.base_multiyear_state_panel_historical_frontier_gap_path.exists()
    assert paths.base_multiyear_summary_panel_historical_frontier_gap_path.exists()


def test_multiyear_comparison_report_compares_two_variants_when_both_exist() -> None:
    paths = _toy_paths()
    _write_multiyear_inputs(paths)
    config = ABMV4Config(start_year=1995, end_year=1997)
    base_simulator = MultiYearBaseSimulator(paths, config)
    historical = MultiYearBaseSimulator(
        paths,
        config,
        emissions_transition_mode="historical_frontier_gap_only",
    )
    base_simulator.write_outputs(base_simulator.run())
    historical.write_outputs(historical.run())

    comparison, markdown = build_multiyear_base_model_comparison(paths)
    assert comparison.height == 2
    assert "historical_frontier_gap_only" in set(comparison["model_variant"].to_list())
    assert "frontier_gap_readiness" in markdown


def test_config_py_is_not_overwritten_by_phase15_run() -> None:
    before = Path("src/abm_v4/config.py").read_text(encoding="utf-8")
    paths = _toy_paths()
    _write_multiyear_inputs(paths)
    simulator = MultiYearBaseSimulator(
        paths,
        ABMV4Config(start_year=1995, end_year=1997),
        emissions_transition_mode="historical_frontier_gap_only",
    )
    simulator.run()
    after = Path("src/abm_v4/config.py").read_text(encoding="utf-8")

    assert after == before


def test_outputs_are_written_only_with_explicit_output_flag() -> None:
    paths = _toy_paths()
    _write_multiyear_inputs(paths)
    simulator = MultiYearBaseSimulator(
        paths,
        ABMV4Config(start_year=1995, end_year=1997),
        emissions_transition_mode="historical_frontier_gap_only",
    )
    result = simulator.run()

    assert not paths.base_multiyear_state_panel_historical_frontier_gap_path.exists()
    simulator.write_outputs(result)
    assert paths.base_multiyear_state_panel_historical_frontier_gap_path.exists()
