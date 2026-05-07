from __future__ import annotations

import pandas as pd
from pathlib import Path
from uuid import uuid4

from src.abm_v3.paths import ABMV3Paths
from src.abm_v3.real_data_smoke_test import RealDataSmokeTester


def toy_smoke_panel() -> pd.DataFrame:
    rows = []
    for year in range(1995, 2017):
        rows.append(
            {
                "country_sector": "A",
                "Country": "AAA",
                "Sector": "Manufacturing",
                "Year": year,
                "X": 100.0,
                "D": 95.0,
                "EI": 2.0,
                "g_in": 0.2,
                "g_out": 0.3,
                "green_capability": 0.4,
                "general_complexity": 0.5,
            }
        )
    return pd.DataFrame(rows)


def workspace_tmp_path() -> Path:
    path = Path("tmp") / "abm_v3_tests" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_real_data_smoke_test_logic_on_toy_dataframe() -> None:
    report = RealDataSmokeTester(ABMV3Paths(project_root=workspace_tmp_path())).run(
        toy_smoke_panel(),
        write_report=False,
    )

    assert "required_columns" in report["check"].tolist()
    assert report.loc[report["check"] == "year_coverage", "passed"].iloc[0]
    assert report.loc[report["check"] == "duplicate_nodes", "passed"].iloc[0]


def test_smoke_test_writes_report_under_abm_v3() -> None:
    project_root = workspace_tmp_path()
    paths = ABMV3Paths(project_root=project_root)

    RealDataSmokeTester(paths).run(toy_smoke_panel(), write_report=True)

    assert (project_root / "data" / "abm_v3" / "diagnostics" / "real_data_smoke_test.csv").exists()
