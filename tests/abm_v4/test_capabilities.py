from pathlib import Path
from uuid import uuid4

import polars as pl

from src.abm_v4.capabilities import (
    CapabilityUpdater,
    capability_increment,
    green_capability_increment,
    sigmoid,
)
from src.abm_v4.config import CapabilityConfig
from src.abm_v4.paths import ABMV4Paths


def toy_root() -> Path:
    return Path("tmp") / "abm_v4_capability_tests" / uuid4().hex


def toy_state_panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "country_sector": ["B1", "S1", "S2", "B1", "S1", "S2"],
            "Year": [1995, 1995, 1995, 1996, 1996, 1996],
            "Country": ["BUY", "SUP", "SUP2", "BUY", "SUP", "SUP2"],
            "Sector": ["Output", "Input", "Input", "Output", "Input", "Input"],
            "X_observed": [10.0, 20.0, 30.0, 20.0, 40.0, 60.0],
            "general_capability": [10.0, 20.0, None, 20.0, 40.0, None],
            "capability_export_weighted_pci": [None, None, None, None, None, None],
            "capability_mean_pci": [None, None, None, None, None, None],
            "active_good_count": [None, None, None, None, None, None],
            "green_capability": [0.2, 0.6, None, 0.3, 0.8, None],
            "green_capability_export_share": [None, None, None, None, None, None],
            "green_capability_share": [None, None, None, None, None, None],
            "g_local_v4": [0.2, 0.7, None, 0.3, 0.9, None],
            "ecosystem_id": ["eco_a", "eco_a", "eco_b", "eco_a", "eco_a", "eco_b"],
            "ecosystem_label": ["A", "A", "B", "A", "A", "B"],
        }
    )


def toy_supplier_weights() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "buyer_country_sector": ["B1", "B1"],
            "supplier_country_sector": ["S1", "S2"],
            "updated_weight": [0.25, 0.75],
        }
    )


def write_toy_capability_inputs(paths: ABMV4Paths) -> None:
    state_path = paths.state_panel_path(1995, 2016)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    toy_state_panel().write_parquet(state_path)
    paths.supplier_updated_weights_path.parent.mkdir(parents=True, exist_ok=True)
    toy_supplier_weights().write_parquet(paths.supplier_updated_weights_path)


def test_sigmoid_is_bounded() -> None:
    assert 0.0 < sigmoid(-10.0) < 0.5
    assert 0.5 < sigmoid(10.0) < 1.0


def test_capability_increments_do_not_exceed_remaining_stock() -> None:
    config = CapabilityConfig(delta_cap_param=0.1, delta_gcap_param=0.1)

    assert capability_increment(0.9, 1.0, config) <= 0.1
    assert green_capability_increment(0.9, 1.0, config) <= 0.1


def test_initial_capability_normalization_maps_values_to_unit_interval() -> None:
    updater = CapabilityUpdater(paths=ABMV4Paths(project_root=toy_root()))
    latest = updater.prepare_latest_state(toy_state_panel(), year=1996)

    normalized = updater.normalize_initial_capabilities(latest)

    assert normalized["cap"].min() >= 0.0
    assert normalized["cap"].max() <= 1.0


def test_missing_capabilities_are_flagged_not_zero_filled() -> None:
    updater = CapabilityUpdater(paths=ABMV4Paths(project_root=toy_root()))
    latest = updater.prepare_latest_state(toy_state_panel(), year=1996)

    normalized = updater.normalize_initial_capabilities(latest)

    assert normalized.filter(pl.col("country_sector") == "S2")[
        "general_capability_filled"
    ].item()
    assert normalized.filter(pl.col("country_sector") == "S2")["cap"].item() != 0.0


def test_supplier_weighted_exposure_excludes_missing_and_renormalizes() -> None:
    updater = CapabilityUpdater(paths=ABMV4Paths(project_root=toy_root()))
    latest = updater.prepare_latest_state(toy_state_panel(), year=1996)
    normalized = updater.normalize_initial_capabilities(latest).with_columns(
        pl.when(pl.col("country_sector") == "S2")
        .then(None)
        .otherwise(pl.col("cap"))
        .alias("cap")
    )

    exposure = updater.compute_supplier_weighted_exposures(normalized, toy_supplier_weights())

    b1 = exposure.filter(pl.col("country_sector") == "B1").row(0, named=True)
    s1_cap = normalized.filter(pl.col("country_sector") == "S1")["cap"].item()
    assert b1["network_capability_exposure"] == s1_cap
    assert b1["supplier_capability_coverage"] == 0.25


def test_ecosystem_exposure_is_mean_capability_within_ecosystem() -> None:
    updater = CapabilityUpdater(paths=ABMV4Paths(project_root=toy_root()))
    latest = updater.prepare_latest_state(toy_state_panel(), year=1996)
    normalized = updater.normalize_initial_capabilities(latest)

    exposure = updater.compute_ecosystem_exposure(normalized)
    expected = normalized.filter(pl.col("ecosystem_id") == "eco_a")["cap"].mean()

    assert exposure.filter(pl.col("country_sector") == "B1")[
        "ecosystem_capability_exposure"
    ].item() == expected


def test_sigmoid_is_monotonic() -> None:
    assert sigmoid(-1.0) < sigmoid(0.0) < sigmoid(1.0)


def test_higher_exposure_creates_higher_delta() -> None:
    config = CapabilityConfig()
    assert capability_increment(0.5, 0.8, config) > capability_increment(0.5, 0.2, config)


def test_capability_next_is_non_decreasing_and_bounded() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_capability_inputs(paths)
    updater = CapabilityUpdater(paths=paths, config=CapabilityConfig(delta_cap_param=0.5))

    _, update, _ = updater.build_capability_update(year=1996)

    assert (update["cap_next"] >= update["cap"]).all()
    assert update["cap_next"].max() <= 1.0


def test_green_capability_uses_same_bounded_logic() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_capability_inputs(paths)
    updater = CapabilityUpdater(paths=paths)

    _, update, _ = updater.build_capability_update(year=1996)

    assert (update["gcap_next"] >= update["gcap"]).all()
    assert update["gcap_next"].max() <= 1.0


def test_capability_report_includes_fill_shares_and_clipping_counts() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_capability_inputs(paths)
    updater = CapabilityUpdater(paths=paths)

    exposure, update, report = updater.build_capability_update(year=1996)
    row = report.row(0, named=True)

    assert row["share_general_capability_filled"] > 0.0
    assert row["share_green_capability_filled"] > 0.0
    assert "cap_clipped_count" in report.columns
    assert exposure.height == update.height


def test_capability_build_does_not_write_without_explicit_output() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_capability_inputs(paths)

    CapabilityUpdater(paths=paths).build_capability_update(year=1996)

    assert not paths.capability_exposure_panel_path.exists()
    assert not paths.capability_update_panel_path.exists()
    assert not paths.capability_update_report_path.exists()
