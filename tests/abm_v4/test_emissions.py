import math
from pathlib import Path
from uuid import uuid4

import polars as pl

from src.abm_v4.config import EmissionsConfig
from src.abm_v4.emissions import EmissionsUpdater, emissions_identity, next_emissions_intensity
from src.abm_v4.paths import ABMV4Paths


def toy_root() -> Path:
    return Path("tmp") / "abm_v4_emissions_tests" / uuid4().hex


def toy_state_panel() -> pl.DataFrame:
    rows = [
        ("A", 2015, "S1", 100.0, 0.60, 0.10),
        ("B", 2015, "S1", 200.0, 0.10, 0.00),
        ("C", 2015, "S1", 50.0, -0.10, 0.00),
        ("D", 2015, "S2", 80.0, 0.80, 0.20),
        ("E", 2015, "S2", 120.0, 0.50, 0.10),
        ("A", 2016, "S1", 100.0, 0.50, 0.10),
        ("B", 2016, "S1", 200.0, 0.20, 0.00),
        ("C", 2016, "S1", 50.0, -0.10, 0.00),
        ("D", 2016, "S2", 80.0, 0.80, 0.20),
        ("E", 2016, "S2", 120.0, 0.40, 0.10),
        ("F", 2014, "S2", 60.0, 0.30, 0.00),
        ("F", 2016, "S2", 60.0, 0.20, 0.00),
    ]
    return pl.DataFrame(
        {
            "country_sector": [row[0] for row in rows],
            "Year": [row[1] for row in rows],
            "Sector": [row[2] for row in rows],
            "X_observed": [row[3] for row in rows],
            "EI": [row[4] for row in rows],
            "log_EI": [math.log(row[4]) if row[4] > 0 else None for row in rows],
            "brown_centrality": [row[5] for row in rows],
        }
    )


def toy_capability_update() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "country_sector": ["A", "B", "C", "D", "E", "F"],
            "cap_next": [0.5, 0.8, 0.2, 0.4, 0.7, 0.6],
            "gcap_next": [0.6, 0.7, 0.1, 0.2, 0.8, 0.3],
        }
    )


def toy_production_feasibility() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "country_sector": ["A", "B", "C", "D", "E", "F"],
            "X_feasible": [90.0, 200.0, 50.0, 80.0, 120.0, 60.0],
        }
    )


def toy_capability_exposure() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "country_sector": ["A", "B", "C", "D", "E", "F"],
            "network_green_exposure": [0.4, 0.7, 0.0, 0.2, 0.8, 0.3],
            "ecosystem_capability_exposure": [0.5, 0.6, 0.1, 0.3, 0.7, 0.4],
        }
    )


def toy_supplier_weights() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "buyer_country_sector": ["A", "A", "B", "B", "D"],
            "supplier_country_sector": ["B", "E", "A", "E", "A"],
            "updated_weight": [0.7, 0.3, 0.5, 0.5, 1.0],
        }
    )


def write_toy_emissions_inputs(paths: ABMV4Paths, write_weights: bool = True) -> None:
    state_path = paths.state_panel_path(1995, 2016)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    toy_state_panel().write_parquet(state_path)
    paths.capability_update_panel_path.parent.mkdir(parents=True, exist_ok=True)
    toy_capability_update().write_parquet(paths.capability_update_panel_path)
    toy_production_feasibility().write_parquet(paths.production_feasibility_panel_path)
    toy_capability_exposure().write_parquet(paths.capability_exposure_panel_path)
    if write_weights:
        toy_supplier_weights().write_parquet(paths.supplier_updated_weights_path)


def test_emissions_identity_uses_output_times_ei() -> None:
    assert emissions_identity(100.0, 0.2) == 20.0


def test_next_emissions_intensity_legacy_helper_respects_floor() -> None:
    config = EmissionsConfig(ei_min=0.01)

    updated = next_emissions_intensity(
        emissions_intensity=0.02,
        green_capability=1.0,
        network_green_exposure=1.0,
        general_capability=1.0,
        brown_centrality=0.0,
        config=config,
    )

    assert updated >= config.ei_min


def test_historical_rei_uses_log_difference_and_excludes_invalid_or_nonconsecutive() -> None:
    updater = EmissionsUpdater(paths=ABMV4Paths(project_root=toy_root()))
    historical = updater.compute_historical_rEI(toy_state_panel())

    a = historical.filter(pl.col("country_sector") == "A")["rEI_observed"].item()

    assert abs(a - (math.log(0.60) - math.log(0.50))) < 1e-12
    assert "C" not in historical["country_sector"].to_list()
    assert "F" not in historical["country_sector"].to_list()


def test_sector_frontier_uses_quantile_and_global_fallback() -> None:
    config = EmissionsConfig(min_frontier_nodes=4, ei_frontier_quantile=0.25)
    updater = EmissionsUpdater(paths=ABMV4Paths(project_root=toy_root()), config=config)
    latest = updater.prepare_latest_valid_state(toy_state_panel(), year=2016)

    frontier_panel = updater.compute_sector_frontiers(latest)
    valid = latest.filter(~pl.col("invalid_EI_flag"))
    global_frontier = valid["EI"].quantile(0.25)
    d = frontier_panel.filter(pl.col("country_sector") == "D").row(0, named=True)

    assert d["frontier_fallback_used"]
    assert d["EI_frontier"] == global_frontier


def test_ei_gap_is_positive_or_zero_never_negative() -> None:
    config = EmissionsConfig(min_frontier_nodes=2)
    updater = EmissionsUpdater(paths=ABMV4Paths(project_root=toy_root()), config=config)
    latest = updater.prepare_latest_valid_state(toy_state_panel(), year=2016)
    panel = updater.compute_ei_gap(updater.compute_sector_frontiers(latest))
    a = panel.filter(pl.col("country_sector") == "A")["ei_gap"].item()
    b = panel.filter(pl.col("country_sector") == "B")["ei_gap"].item()

    assert a > 0
    assert b == 0
    assert panel.filter(~pl.col("invalid_EI_flag"))["ei_gap"].min() >= 0


def test_supplier_lockin_is_hhi_and_fallback_is_flagged() -> None:
    updater = EmissionsUpdater(paths=ABMV4Paths(project_root=toy_root()))
    latest = updater.prepare_latest_valid_state(toy_state_panel(), year=2016)
    lockin = updater.compute_supplier_lockin(latest, supplier_weights=toy_supplier_weights())
    a = lockin.filter(pl.col("country_sector") == "A").row(0, named=True)
    e = lockin.filter(pl.col("country_sector") == "E").row(0, named=True)

    assert abs(a["supplier_lockin"] - (0.7**2 + 0.3**2)) < 1e-12
    assert 0 <= a["supplier_lockin"] <= 1
    assert e["supplier_lockin_fallback_used"]


def test_readiness_is_bounded_and_moves_with_capability_and_lockin() -> None:
    updater = EmissionsUpdater(paths=ABMV4Paths(project_root=toy_root()))
    panel = pl.DataFrame(
        {
            "gcap_next": [0.1, 0.9],
            "cap_next": [0.1, 0.9],
            "network_green_exposure": [0.1, 0.9],
            "ecosystem_capability_exposure": [0.1, 0.9],
            "brown_centrality": [0.8, 0.0],
            "supplier_lockin": [0.9, 0.0],
        }
    )

    out = updater.compute_transition_readiness(panel)

    assert out["readiness"].min() >= 0
    assert out["readiness"].max() <= updater.config.rho_max
    assert out["readiness"][1] > out["readiness"][0]


def test_gap_closure_is_zero_increasing_and_saturating() -> None:
    updater = EmissionsUpdater(paths=ABMV4Paths(project_root=toy_root()))
    panel = pl.DataFrame({"ei_gap": [0.0, 1.0, 1000.0], "readiness": [0.05, 0.05, 0.05]})

    out = updater.compute_gap_closure_potential(panel)

    assert out["gap_closure_potential"][0] == 0
    assert out["gap_closure_potential"][2] > out["gap_closure_potential"][1]
    assert out["gap_closure_potential"][2] < 0.05


def test_frontier_gap_rei_clips_and_preserves_raw() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_emissions_inputs(paths)
    config = EmissionsConfig(
        rEI_max=0.01,
        min_frontier_nodes=2,
        use_sector_background_trend=False,
        rho_max=0.50,
        theta_intercept=3.0,
    )

    panel = EmissionsUpdater(paths=paths, config=config).build_emissions_update_panel(year=2016)
    a = panel.filter(pl.col("country_sector") == "A").row(0, named=True)

    assert a["rEI_raw"] > a["rEI_used"]
    assert a["rEI_used"] == 0.01
    assert a["rEI_clipped_high_flag"]


def test_ei_update_positive_negative_and_invalid_behavior() -> None:
    updater = EmissionsUpdater(paths=ABMV4Paths(project_root=toy_root()))
    panel = pl.DataFrame(
        {
            "EI": [1.0, 1.0, -1.0],
            "rEI_used": [0.1, -0.1, None],
            "invalid_EI_flag": [False, False, True],
        }
    )

    out = updater.compute_emissions_update(panel)

    assert out["EI_next"][0] < 1.0
    assert out["EI_next"][1] > 1.0
    assert out["EI_next"][2] is None


def test_emissions_decomposition_sums_to_total_change() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_emissions_inputs(paths)

    panel = EmissionsUpdater(paths=paths).build_emissions_update_panel(year=2016)
    valid = panel.filter(~pl.col("invalid_EI_flag"))
    residual = valid["decomposition_residual_node"].abs().max()

    assert residual < 1e-9
    assert valid["production_scale_effect"].null_count() == 0
    assert valid["emissions_intensity_effect"].null_count() == 0


def test_bad_transition_flag_triggers_when_output_loss_dominates() -> None:
    updater = EmissionsUpdater(paths=ABMV4Paths(project_root=toy_root()))
    panel = pl.DataFrame(
        {
            "year": [2016],
            "invalid_EI_flag": [False],
            "EI_clipped_flag": [False],
            "rEI": [0.01],
            "emissions_observed_current": [100.0],
            "emissions_feasible_current_EI": [50.0],
            "emissions_feasible_next_EI": [49.0],
            "production_scale_effect": [-50.0],
            "emissions_intensity_effect": [-1.0],
            "interaction_effect": [0.0],
        }
    )

    report = updater.build_emissions_update_report(panel)

    assert report["bad_transition_flag"].item()


def test_legacy_mode_runs_and_default_is_frontier_gap() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_emissions_inputs(paths)

    default_updater = EmissionsUpdater(paths=paths)
    legacy = EmissionsUpdater(paths=paths, transition_mode="legacy_raw_log")
    default_panel = default_updater.build_emissions_update_panel(year=2016)
    legacy_panel = legacy.build_emissions_update_panel(year=2016)

    assert default_updater.transition_mode == "frontier_gap_readiness"
    assert "rEI_legacy_raw_log" in default_panel.columns
    assert default_panel["rEI_used"].to_list() != legacy_panel["rEI_used"].to_list()


def test_emissions_update_does_not_write_without_explicit_output() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_emissions_inputs(paths)

    EmissionsUpdater(paths=paths).build_emissions_update(year=2016)

    assert not paths.emissions_update_panel_path.exists()
    assert not paths.emissions_update_report_path.exists()
    assert not paths.emissions_decomposition_base_path.exists()
    assert not paths.emissions_historical_rEI_summary_path.exists()
    assert not paths.emissions_transition_comparison_path.exists()
