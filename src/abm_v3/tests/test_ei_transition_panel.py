from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from src.abm_v3.ei_transition.outputs import EITransitionOutputWriter
from src.abm_v3.ei_transition.panel import EITransitionPanelBuilder
from src.abm_v3.paths import ABMV3Paths


def toy_paths() -> ABMV3Paths:
    root = Path("tmp") / "ei_transition_panel_tests" / uuid4().hex[:8]
    root.mkdir(parents=True, exist_ok=True)
    return ABMV3Paths(project_root=root)


def toy_raw_panel(include_g_network: bool = False) -> pd.DataFrame:
    rows = []
    for year, ei in [(1995, 10.0), (1996, 5.0), (1997, 0.0)]:
        row = {
            "country_sector": "AAA | AAA | Industries | Agriculture",
            "Year": year,
            "Country": "AAA",
            "Country_detail": "AAA",
            "Category": "Industries",
            "Sector": "Agriculture",
            "EI": ei,
            "green_capability": 0.2,
            "g_in": 0.4,
            "g_out": 0.8,
            "general_complexity": 0.5,
            "X": 100.0,
            "X_observed": 100.0,
        }
        if include_g_network:
            row["g_network"] = 0.9
        rows.append(row)
    return pd.DataFrame(rows)


def test_ei_transition_target_formula() -> None:
    result = EITransitionPanelBuilder(toy_paths()).build_from_dataframe(toy_raw_panel(), 1995, 1997)
    row = result.panel.loc[result.panel["Year"] == 1995].iloc[0]

    assert np.isclose(row["ei_reduction_next"], np.log(10.0) - np.log(5.0))


def test_non_positive_ei_rows_excluded() -> None:
    result = EITransitionPanelBuilder(toy_paths()).build_from_dataframe(toy_raw_panel(), 1995, 1997)
    row_with_non_positive_next = result.panel.loc[result.panel["Year"] == 1996].iloc[0]
    final_year = result.panel.loc[result.panel["Year"] == 1997].iloc[0]

    assert bool(row_with_non_positive_next["sample_included"]) is False
    assert "non_positive_EI_next" in row_with_non_positive_next["exclusion_reason"]
    assert bool(final_year["sample_included"]) is False
    assert "non_positive_EI" in final_year["exclusion_reason"]


def test_g_network_computed_when_missing() -> None:
    result = EITransitionPanelBuilder(toy_paths()).build_from_dataframe(toy_raw_panel(), 1995, 1997)

    assert np.isclose(result.panel["g_network"].iloc[0], 0.6)
    assert "g_network_source=computed_0.5_g_in_plus_g_out" in result.notes


def test_missing_required_columns_raise_clear_error() -> None:
    broken = toy_raw_panel().drop(columns=["EI"])

    try:
        EITransitionPanelBuilder(toy_paths()).build_from_dataframe(broken, 1995, 1997)
    except ValueError as error:
        message = str(error)
        assert "Missing required columns" in message
        assert "EI" in message
        assert "Available columns" in message
    else:
        raise AssertionError("Expected missing EI column to raise ValueError")


def test_transition_panel_drops_final_year_target() -> None:
    result = EITransitionPanelBuilder(toy_paths()).build_from_dataframe(toy_raw_panel(), 1995, 1997)
    final_year = result.panel.loc[result.panel["Year"] == 1997].iloc[0]

    assert pd.isna(final_year["EI_next"])
    assert bool(final_year["sample_included"]) is False
    assert "missing_EI_next" in final_year["exclusion_reason"]


def test_outputs_written_to_ei_transition_folder() -> None:
    paths = toy_paths()
    result = EITransitionPanelBuilder(paths).build_from_dataframe(toy_raw_panel(), 1995, 1997)

    written = EITransitionOutputWriter(paths).write_panel(1995, 1997, result)

    assert Path(written["panel"]).parent == paths.ei_transition_inputs_dir
    assert Path(written["sample_report"]).parent == paths.ei_transition_diagnostics_dir
    assert paths.ei_transition_panel_path(1995, 1997).exists()
