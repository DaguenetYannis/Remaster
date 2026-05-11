from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from src.abm_v3.phase_space.state_panel import ABMV3PhaseSpaceStatePanelBuilder


def toy_workspace() -> Path:
    """Create a workspace-local temporary folder for systems with locked OS temp dirs."""
    root = Path("tmp") / "phase_space_state_panel_tests" / uuid4().hex[:8]
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_small_base_panel(path: Path) -> None:
    """Write a small country-sector-year panel with intentional duplicate diagnostics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    panel = pd.DataFrame(
        {
            "country_sector": [
                "AAA | Agriculture",
                "AAA | Agriculture",
                "BBB | Manufacturing",
                "BBB | Manufacturing",
                "CCC | Energy",
                "CCC | Energy",
                "CCC | Energy",
            ],
            "Year": [1995, 1996, 1995, 1996, 1995, 1996, 1996],
            "Country": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC", "CCC"],
            "Sector": ["Agriculture", "Agriculture", "Manufacturing", "Manufacturing", "Energy", "Energy", "Energy"],
            "X_observed": [100.0, 120.0, 200.0, 210.0, 300.0, 330.0, 330.0],
            "EI": [0.5, 0.4, 0.8, 0.7, 1.5, 1.2, 1.2],
            "green_capability_export_share": [0.8, 0.85, 0.4, 0.45, 0.2, 0.25, 0.25],
            "brown_centrality": [0.2, 0.2, 0.5, 0.5, 0.9, 0.9, 0.9],
            "K": [150.0, 150.0, 250.0, 250.0, 350.0, 350.0, 350.0],
        }
    )
    panel.to_parquet(path, index=False)


def test_phase_space_state_panel_builder_writes_outputs_and_derives_variables() -> None:
    root = toy_workspace()
    base_panel = root / "inputs" / "base.parquet"
    output_dir = root / "phase_space"
    write_small_base_panel(base_panel)

    written = ABMV3PhaseSpaceStatePanelBuilder(
        base_panel=base_panel,
        output_dir=output_dir,
        top_n=1,
    ).build(start_year=1995, end_year=1996)

    assert written["panel"].exists()
    assert written["columns"].exists()
    assert written["diagnostics"].exists()
    assert written["summary"].exists()

    panel = pd.read_parquet(written["panel"])
    assert not panel.duplicated(["country_sector", "Year"]).any()
    assert {"country_sector", "Year"}.issubset(panel.columns)
    assert np.isclose(panel.loc[panel["X_observed"].eq(100.0), "log_X_observed"].iloc[0], np.log1p(100.0))
    assert np.isclose(panel.loc[panel["EI"].eq(0.5), "g_local"].iloc[0], 1.0 / 1.5)

    aaa_1995 = panel.loc[(panel["country_sector"].eq("AAA | Agriculture")) & (panel["Year"].eq(1995))].iloc[0]
    assert np.isclose(aaa_1995["EI_next"], 0.4)
    assert np.isclose(aaa_1995["delta_g_local"], (1.0 / 1.4) - (1.0 / 1.5))
    assert np.isclose(aaa_1995["rEI"], np.log(0.5) - np.log(0.4))
    assert "is_top25_by_output_over_period" in panel.columns
    assert "is_top25_by_output_in_year" in panel.columns


def test_phase_space_state_panel_diagnostics_report_missing_derived_and_duplicate_keys() -> None:
    root = toy_workspace()
    base_panel = root / "inputs" / "base.parquet"
    output_dir = root / "phase_space"
    write_small_base_panel(base_panel)

    written = ABMV3PhaseSpaceStatePanelBuilder(base_panel=base_panel, output_dir=output_dir, top_n=1).build(
        start_year=1995,
        end_year=1996,
    )

    diagnostics = pd.read_csv(written["diagnostics"])
    assert (
        diagnostics["diagnostic_type"].eq("availability")
        & diagnostics["variable"].eq("network_green_ness_available")
        & diagnostics["value"].astype(str).str.lower().eq("false")
    ).any()
    assert (
        diagnostics["diagnostic_type"].eq("derived_variable")
        & diagnostics["variable"].eq("g_local")
    ).any()
    assert diagnostics["diagnostic_type"].eq("duplicated_keys").any()

    columns = pd.read_csv(written["columns"])
    g_local = columns.loc[columns["column"].eq("g_local")].iloc[0]
    assert bool(g_local["derived"])
    assert "1 / (1 + EI)" in str(g_local["formula_or_transformation"])


def test_phase_space_state_panel_strict_mode_fails_on_missing_required_keys() -> None:
    root = toy_workspace()
    base_panel = root / "inputs" / "bad.parquet"
    base_panel.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Year": [1995], "EI": [0.5]}).to_parquet(base_panel, index=False)

    builder = ABMV3PhaseSpaceStatePanelBuilder(base_panel=base_panel, output_dir=root / "phase_space", strict=True)

    try:
        builder.build(start_year=1995, end_year=1996)
    except ValueError as error:
        assert "missing required columns" in str(error)
    else:
        raise AssertionError("Strict mode should fail when country_sector is missing.")
