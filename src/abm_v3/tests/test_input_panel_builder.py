from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from src.abm_v3.config import ABMV3Config
from src.abm_v3.data_loader import ABMV3DataLoader
from src.abm_v3.input_panel_builder import ABMV3InputPanelBuilder
from src.abm_v3.paths import ABMV3Paths
from src.abm_v3.real_data_smoke_test import RealDataSmokeTester


def workspace_tmp_path() -> Path:
    path = Path("tmp") / "abm_v3_input_builder_tests" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def toy_paths() -> ABMV3Paths:
    return ABMV3Paths(project_root=workspace_tmp_path())


def write_toy_raw_year(paths: ABMV3Paths, year: int = 1995) -> None:
    labels = [
        "AAA | AAA | Industries | Agriculture",
        "BBB | BBB | Industries | Manufacturing",
    ]
    matrix_dir = paths.parquet_root / str(year)
    raw_dir = paths.raw_root / str(year)
    matrix_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    t_matrix = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=labels,
        columns=labels,
    )
    fd_matrix = pd.DataFrame(
        [[10.0, 1.0], [20.0, 2.0]],
        index=labels,
        columns=["fd1", "fd2"],
    )
    t_matrix.to_parquet(matrix_dir / "T.parquet")
    fd_matrix.to_parquet(matrix_dir / "FD.parquet")
    raw_dir.joinpath("labels_T.txt").write_text(
        "AAA\tAAA\tIndustries\tAgriculture\t\n"
        "BBB\tBBB\tIndustries\tManufacturing\t\n",
        encoding="utf-8",
    )


def write_toy_metrics(paths: ABMV3Paths, rows: list[dict[str, object]] | None = None) -> None:
    paths.final_root.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(
        rows
        or [
            {
                "country_sector": "AAA | AAA | Industries | Agriculture",
                "Year": 1995,
                "emissions_intensity": 2.0,
                "g_base": 0.1,
                "g_in_network": 0.2,
                "g_out_network": 0.4,
                "green_active_good_export_value": 5.0,
                "active_good_export_value": 10.0,
                "green_capability_export_share": 0.5,
                "capability_export_weighted_pci": 1.2,
            }
        ]
    )
    metrics.to_parquet(paths.eora_atlas_merged_file, index=False)


def test_build_year_accounting_panel_creates_X_D_M() -> None:
    paths = toy_paths()
    write_toy_raw_year(paths)
    write_toy_metrics(paths)
    builder = ABMV3InputPanelBuilder(paths)

    panel, _report = builder.build_year_accounting_panel(1995)

    assert panel["intermediate_output"].tolist() == [4.0, 6.0]
    assert panel["intermediate_demand"].tolist() == [3.0, 7.0]
    assert panel["final_demand_total"].tolist() == [11.0, 22.0]
    assert panel["X"].tolist() == [15.0, 28.0]
    assert panel["D"].tolist() == [14.0, 29.0]
    assert panel["M"].tolist() == [4.0, 6.0]
    assert panel["available_inputs"].tolist() == [4.0, 6.0]


def test_country_sector_split() -> None:
    builder = ABMV3InputPanelBuilder(toy_paths())

    split = builder.split_country_sector_labels(["AAA | AAA | Industries | Agriculture"])

    assert split.iloc[0]["Country"] == "AAA"
    assert split.iloc[0]["Country_detail"] == "AAA"
    assert split.iloc[0]["Category"] == "Industries"
    assert split.iloc[0]["Sector"] == "Agriculture"


def test_abm_ready_panel_left_joins_metrics() -> None:
    paths = toy_paths()
    write_toy_raw_year(paths)
    write_toy_metrics(paths)
    builder = ABMV3InputPanelBuilder(paths)
    accounting, _report = builder.build_year_accounting_panel(1995)
    metrics = pd.read_parquet(paths.eora_atlas_merged_file)

    merged = builder.add_required_aliases(builder.merge_with_metrics(accounting, metrics))

    assert len(merged) == 2
    assert merged.loc[merged["country_sector"].str.startswith("BBB"), "EI"].isna().all()


def test_capacity_inventory_proxies() -> None:
    paths = toy_paths()
    write_toy_raw_year(paths)
    write_toy_metrics(paths)
    config = ABMV3Config()
    builder = ABMV3InputPanelBuilder(paths, config)

    panel, _report = builder.build_year_accounting_panel(1995)

    assert panel["K"].tolist() == [16.5, 30.800000000000004]
    assert np.allclose(panel["I"], [30 * 4 / 365, 30 * 6 / 365])


def test_input_panel_output_path_dynamic() -> None:
    path = ABMV3InputPanelBuilder(toy_paths()).output_path(1995, 2016)

    assert path.name == "abm_v3_historical_panel_1995_2016.parquet"


def test_loader_prefers_abm_ready_panel() -> None:
    paths = toy_paths()
    expected_path = paths.abm_v3_historical_panel_file(1995, 2016)
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"country_sector": ["A"], "Year": [1995], "X": [1.0]}).to_parquet(expected_path, index=False)

    loaded = ABMV3DataLoader(paths).load_abm_ready_historical_panel(1995, 2016)

    assert loaded["X"].tolist() == [1.0]


def test_loader_builds_if_missing() -> None:
    paths = toy_paths()
    write_toy_raw_year(paths)
    write_toy_metrics(paths)

    loaded = ABMV3DataLoader(paths).load_abm_ready_historical_panel(1995, 1995)

    assert paths.abm_v3_historical_panel_file(1995, 1995).exists()
    assert {"X", "D", "M", "available_inputs"}.issubset(loaded.columns)


def test_smoke_test_prefers_input_panel() -> None:
    paths = toy_paths()
    input_path = paths.abm_v3_historical_panel_file(1995, 2016)
    input_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for year in range(1995, 2017):
        rows.append(
            {
                "country_sector": "A",
                "Country": "AAA",
                "Sector": "Agriculture",
                "Year": year,
                "X": 1.0,
                "D": 1.0,
                "M": 1.0,
                "available_inputs": 1.0,
                "EI": 1.0,
                "g_in": 0.1,
                "g_out": 0.2,
                "g_network": 0.15,
                "green_capability": 0.3,
                "general_complexity": 0.4,
            }
        )
    pd.DataFrame(rows).to_parquet(input_path, index=False)

    report = RealDataSmokeTester(paths).run(write_report=False)

    assert report.loc[report["check"] == "abm_ready_input_panel_selected", "passed"].iloc[0]


def test_negative_handling() -> None:
    builder = ABMV3InputPanelBuilder(toy_paths())
    df = pd.DataFrame(
        {
            "X": [-1e-10, -0.01],
            "D": [1.0, 2.0],
            "M": [1.0, 2.0],
            "available_inputs": [1.0, 2.0],
        }
    )

    cleaned, report = builder.handle_negative_accounting_values(df)

    assert cleaned.loc[0, "X"] == 0.0
    assert np.isnan(cleaned.loc[1, "X"])
    assert report["tiny_negative_clipped_count"] == 1
    assert report["meaningful_negative_nan_count"] == 1


def test_column_dictionary_written() -> None:
    paths = toy_paths()

    ABMV3InputPanelBuilder(paths).write_column_dictionary()

    dictionary_path = paths.abm_v3_output_root / "diagnostics" / "abm_v3_input_panel_column_dictionary.csv"
    dictionary = pd.read_csv(dictionary_path)
    assert {"column", "description", "source", "observed_or_proxy", "notes"}.issubset(dictionary.columns)


def test_negative_ei_diagnostic_written() -> None:
    paths = toy_paths()
    panel = pd.DataFrame(
        {
            "Year": [1995],
            "country_sector": ["A"],
            "Country": ["AAA"],
            "Sector": ["Agriculture"],
            "X_observed": [100.0],
            "EI": [-1.0],
            "emissions_observed": [np.nan],
        }
    )

    ABMV3InputPanelBuilder(paths).write_negative_ei_rows(panel, 1995, 2016)

    path = paths.abm_v3_output_root / "diagnostics" / "negative_ei_rows_1995_2016.csv"
    rows = pd.read_csv(path)
    assert rows["country_sector"].tolist() == ["A"]


def test_input_intensity_summary_contains_fallback_shares() -> None:
    paths = toy_paths()
    panel = pd.DataFrame(
        {
            "Year": [1995, 1995],
            "observed_input_intensity": [0.4, 0.0],
            "effective_input_intensity": [0.4, 0.25],
            "input_intensity_source": ["node", "country_category"],
            "X_observed": [100.0, 100.0],
            "M_observed": [40.0, 0.0],
        }
    )

    ABMV3InputPanelBuilder(paths).write_input_intensity_summary(panel, 1995, 2016)

    path = paths.abm_v3_output_root / "diagnostics" / "input_intensity_summary_1995_2016.csv"
    summary = pd.read_csv(path)
    assert "node_ratio_used_share" in summary.columns
    assert "country_category_fallback_share" in summary.columns
