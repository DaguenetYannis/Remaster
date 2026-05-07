from __future__ import annotations

import pandas as pd
from pathlib import Path
from uuid import uuid4

from src.abm_v3.diagnostics.hypothesis_reports import HypothesisReportGenerator
from src.abm_v3.paths import ABMV3Paths


def workspace_tmp_path() -> Path:
    path = Path("tmp") / "abm_v3_tests" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def toy_hypothesis_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "country_sector": ["A", "B", "A", "B"],
            "Country": ["AAA", "BBB", "AAA", "BBB"],
            "Sector": ["S1", "S1", "S1", "S1"],
            "Year": [1995, 1995, 1996, 1996],
            "X": [100.0, 200.0, 110.0, 190.0],
            "EI": [2.0, 3.0, 1.8, 2.9],
            "delta_log_EI_next": [-0.1, -0.02, -0.05, -0.01],
            "log_EI_lag1": [0.7, 1.1, 0.6, 1.0],
            "green_capability": [0.8, 0.1, 0.9, 0.2],
            "general_complexity": [1.0, 0.5, 1.1, 0.6],
            "g_local": [0.3, 0.2, 0.4, 0.25],
            "g_in": [0.2, 0.1, 0.3, 0.15],
            "g_out": [0.25, 0.15, 0.35, 0.2],
            "g_network": [0.225, 0.125, 0.325, 0.175],
        }
    )


def test_hypothesis_report_regression_returns_coefficient_table() -> None:
    generator = HypothesisReportGenerator(ABMV3Paths(project_root=workspace_tmp_path()))

    table = generator.green_capability_ei_reduction(toy_hypothesis_panel(), write=False)

    assert "term" in table.columns
    assert "coefficient" in table.columns
    assert "green_capability" in table["term"].tolist()


def test_local_vs_network_report_returns_yearly_means() -> None:
    generator = HypothesisReportGenerator(ABMV3Paths(project_root=workspace_tmp_path()))

    yearly = generator.local_vs_network_greenness_yearly(toy_hypothesis_panel(), write=False)

    assert yearly["Year"].tolist() == [1995, 1996]
    assert "g_network" in yearly.columns
