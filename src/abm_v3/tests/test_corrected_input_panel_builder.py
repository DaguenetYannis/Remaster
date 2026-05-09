from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from src.abm_v3.config import ABMV3Config
from src.abm_v3.input_panel_builder import CorrectedOrientationInputPanelBuilder
from src.abm_v3.paths import ABMV3Paths


def toy_paths() -> ABMV3Paths:
    root = Path("tmp") / "abm_v3_corrected_input_builder_tests" / uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return ABMV3Paths(project_root=root)


def toy_labels() -> list[str]:
    return [
        "AAA | AAA | Industries | Agriculture",
        "BBB | BBB | Industries | Manufacturing",
    ]


def write_toy_raw_year(paths: ABMV3Paths, year: int = 1995) -> None:
    labels = toy_labels()
    matrix_dir = paths.parquet_root / str(year)
    raw_dir = paths.raw_root / str(year)
    matrix_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=labels, columns=labels).to_parquet(matrix_dir / "T.parquet")
    pd.DataFrame(
        [[10.0, -1.0], [20.0, -2.0]],
        index=labels,
        columns=[
            "AAA | AAA | Final demand | Household final consumption",
            "AAA | AAA | Final demand | Changes in inventories P.52",
        ],
    ).to_parquet(matrix_dir / "FD.parquet")
    raw_dir.joinpath("labels_T.txt").write_text(
        "AAA\tAAA\tIndustries\tAgriculture\t\n"
        "BBB\tBBB\tIndustries\tManufacturing\t\n",
        encoding="utf-8",
    )
    raw_dir.joinpath("labels_FD.txt").write_text(
        "AAA\tAAA\tFinal demand\tHousehold final consumption\t\n"
        "AAA\tAAA\tFinal demand\tChanges in inventories P.52\t\n",
        encoding="utf-8",
    )


def write_toy_metrics(paths: ABMV3Paths) -> None:
    paths.final_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "country_sector": "AAA | AAA | Industries | Agriculture",
                "Year": 1995,
                "emissions_intensity": 2.0,
                "g_base": 0.1,
                "g_in_network": 0.2,
                "g_out_network": 0.4,
                "active_good_export_value": 10.0,
                "green_active_good_export_value": 5.0,
                "green_capability_export_share": 0.5,
                "capability_export_weighted_pci": 1.2,
            },
            {
                "country_sector": "BBB | BBB | Industries | Manufacturing",
                "Year": 1995,
                "emissions_intensity": 3.0,
                "g_base": 0.2,
                "g_in_network": 0.3,
                "g_out_network": 0.5,
                "active_good_export_value": 20.0,
                "green_active_good_export_value": 8.0,
                "green_capability_export_share": 0.4,
                "capability_export_weighted_pci": 1.5,
            },
        ]
    ).to_parquet(paths.eora_atlas_merged_file, index=False)


def build_corrected_panel(paths: ABMV3Paths, config: ABMV3Config | None = None) -> pd.DataFrame:
    write_toy_raw_year(paths)
    write_toy_metrics(paths)
    return CorrectedOrientationInputPanelBuilder(paths, config or ABMV3Config()).build(1995, 1995, overwrite=True)


def test_corrected_orientation_uses_row_sum_for_X() -> None:
    panel = build_corrected_panel(toy_paths())

    assert panel["X_corrected"].tolist() == [13.0, 27.0]


def test_corrected_orientation_uses_column_sum_for_M() -> None:
    panel = build_corrected_panel(toy_paths())

    assert panel["M_corrected"].tolist() == [4.0, 6.0]


def test_inventory_columns_excluded_from_corrected_Y() -> None:
    panel = build_corrected_panel(toy_paths())

    assert panel["Y_raw"].tolist() == [9.0, 18.0]
    assert panel["Y_no_inventory"].tolist() == [10.0, 20.0]


def test_raw_and_corrected_X_are_both_preserved() -> None:
    panel = build_corrected_panel(toy_paths())

    required = {
        "X_raw_current_convention",
        "X_row_raw",
        "X_row_no_inventory",
        "X_column_no_inventory",
        "X_corrected",
    }
    assert required.issubset(panel.columns)
    assert panel["X_raw_current_convention"].tolist() == [13.0, 24.0]


def test_corrected_aliases_use_corrected_values() -> None:
    panel = build_corrected_panel(toy_paths())

    assert panel["X"].tolist() == panel["X_corrected"].tolist()
    assert panel["X_observed"].tolist() == panel["X_corrected"].tolist()
    assert panel["M"].tolist() == panel["M_corrected"].tolist()
    assert panel["D"].tolist() == panel["D_proxy_corrected"].tolist()


def test_capacity_and_inventory_use_corrected_values() -> None:
    config = ABMV3Config()
    panel = build_corrected_panel(toy_paths(), config)

    assert np.allclose(panel["K"], 1.10 * panel["X_corrected"])
    assert np.allclose(panel["I"], 30 * panel["M_corrected"] / 365.0)


def test_input_intensity_uses_corrected_M_and_X() -> None:
    panel = build_corrected_panel(toy_paths())

    assert np.allclose(panel["observed_input_intensity"], panel["M_corrected"] / panel["X_corrected"])


def test_old_panel_not_overwritten() -> None:
    paths = toy_paths()
    old_path = paths.abm_v3_historical_panel_file(1995, 1995)
    old_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sentinel": [1]}).to_parquet(old_path, index=False)

    build_corrected_panel(paths)

    assert pd.read_parquet(old_path)["sentinel"].tolist() == [1]
    assert paths.abm_v3_corrected_historical_panel_file(1995, 1995).exists()


def test_orientation_comparison_diagnostics_written() -> None:
    paths = toy_paths()
    build_corrected_panel(paths)

    path = paths.abm_v3_output_root / "diagnostics" / "abm_v3_input_panel_orientation_comparison_1995_1995.csv"
    comparison = pd.read_csv(path)
    assert path.exists()
    assert {"Year", "old_X_total", "corrected_X_total", "correlation_old_X_corrected_X"}.issubset(comparison.columns)


def test_column_dictionary_mentions_corrected_orientation() -> None:
    paths = toy_paths()
    build_corrected_panel(paths)

    path = (
        paths.abm_v3_output_root
        / "diagnostics"
        / "abm_v3_input_panel_column_dictionary_transpose_row_fd_without_inventory.csv"
    )
    dictionary = pd.read_csv(path)
    assert dictionary["notes"].str.contains("corrected", case=False).any()
