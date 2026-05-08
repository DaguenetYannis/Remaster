from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from src.abm_v3.config import LeontiefPropagationConfig
from src.abm_v3.leontief.coefficients import LeontiefCoefficientBuilder
from src.abm_v3.leontief.propagation import LeontiefPropagationEngine
from src.abm_v3.leontief.validation import LeontiefPropagationValidator
from src.abm_v3.paths import ABMV3Paths


def toy_paths() -> ABMV3Paths:
    root = Path("tmp") / "abm_v3_leontief_mode_tests" / uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return ABMV3Paths(project_root=root)


def write_toy_year(paths: ABMV3Paths, t_values: list[list[float]], fd_values: list[list[float]]) -> None:
    labels = [
        "AAA | AAA | Industries | Agriculture",
        "BBB | BBB | Industries | Manufacturing",
    ]
    fd_labels = [
        "AAA | AAA | Final demand | Household final consumption",
        "AAA | AAA | Final demand | Changes in inventories P.52",
    ]
    matrix_dir = paths.parquet_root / "1995"
    raw_dir = paths.raw_root / "1995"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(t_values, index=labels, columns=labels).to_parquet(matrix_dir / "T.parquet")
    pd.DataFrame(fd_values, index=labels, columns=fd_labels).to_parquet(matrix_dir / "FD.parquet")
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


def load_mode(paths: ABMV3Paths, mode: str, cap: float = 0.99):
    config = LeontiefPropagationConfig(leontief_mode=mode, leontief_column_sum_cap=cap)
    return LeontiefCoefficientBuilder(paths, config).load_year(1995)


def test_raw_mode_matches_existing_formula() -> None:
    paths = toy_paths()
    write_toy_year(paths, [[10.0, 20.0], [30.0, 40.0]], [[50.0, 10.0], [20.0, 40.0]])

    year_data = load_mode(paths, "raw")

    assert year_data.Y_final_demand.tolist() == [60.0, 60.0]
    assert year_data.X_observed.tolist() == [100.0, 120.0]
    assert np.allclose(year_data.X_used_for_coefficients.tolist(), [100.0, 120.0])


def test_fd_without_inventory_excludes_inventory_columns() -> None:
    paths = toy_paths()
    write_toy_year(paths, [[1.0, 2.0], [3.0, 4.0]], [[10.0, -7.0], [20.0, -5.0]])

    year_data = load_mode(paths, "fd_without_inventory")

    assert year_data.Y_final_demand.tolist() == [10.0, 20.0]
    assert len(year_data.excluded_fd_columns) == 1
    assert "Changes in inventories" in year_data.excluded_fd_columns["fd_column_label"].iloc[0]


def test_positive_final_demand_only_clips_negative_fd() -> None:
    paths = toy_paths()
    write_toy_year(paths, [[1.0, 2.0], [3.0, 4.0]], [[10.0, -7.0], [-1.0, 5.0]])

    year_data = load_mode(paths, "positive_final_demand_only")

    assert year_data.Y_final_demand.tolist() == [10.0, 5.0]


def test_validation_still_uses_raw_observed_X() -> None:
    paths = toy_paths()
    write_toy_year(paths, [[1.0, 2.0], [3.0, 4.0]], [[10.0, -7.0], [20.0, -5.0]])
    year_data = load_mode(paths, "fd_without_inventory")
    result = LeontiefPropagationEngine(max_rounds=0).propagate(year_data)

    comparison = LeontiefPropagationValidator().build_node_comparison(year_data, result)

    assert comparison["X_observed"].tolist() == [7.0, 21.0]
    assert comparison["X_used_for_coefficients"].tolist() == [14.0, 26.0]


def test_column_rescaled_if_sum_above_one_caps_column_sum() -> None:
    paths = toy_paths()
    write_toy_year(paths, [[2.0, 0.0], [0.0, 1.0]], [[-1.0, 0.0], [0.0, 1.0]])

    year_data = load_mode(paths, "column_rescaled_if_sum_above_one", cap=0.99)

    assert np.isclose(year_data.A.toarray()[:, 0].sum(), 0.99)
    assert len(year_data.rescaled_columns) == 1


def test_abs_column_rescaled_caps_abs_column_sum() -> None:
    paths = toy_paths()
    write_toy_year(paths, [[2.0, 0.0], [0.0, 1.0]], [[-1.0, 0.0], [0.0, 1.0]])

    year_data = load_mode(paths, "column_rescaled_if_abs_sum_above_one", cap=0.99)

    assert np.isclose(np.abs(year_data.A.toarray()[:, 0]).sum(), 0.99)


def test_mode_diagnostics_reports_excluded_inventory() -> None:
    paths = toy_paths()
    write_toy_year(paths, [[1.0, 2.0], [3.0, 4.0]], [[10.0, -7.0], [20.0, -5.0]])

    year_data = load_mode(paths, "fd_without_inventory")

    diagnostics = year_data.mode_diagnostics.iloc[0]
    assert diagnostics["excluded_fd_column_count"] == 1
    assert diagnostics["total_excluded_inventory_value"] == -12.0


def test_mode_diagnostics_reports_rescaled_columns() -> None:
    paths = toy_paths()
    write_toy_year(paths, [[2.0, 0.0], [0.0, 1.0]], [[-1.0, 0.0], [0.0, 1.0]])

    year_data = load_mode(paths, "column_rescaled_if_sum_above_one")

    assert year_data.mode_diagnostics.loc[0, "rescaled_column_count"] == 1


def test_mode_specific_paths_do_not_overwrite() -> None:
    paths = toy_paths()

    assert paths.leontief_iterative_output_path(1995, "raw") != paths.leontief_iterative_output_path(
        1995,
        "positive_final_demand_only",
    )


def test_invalid_mode_raises_clear_error() -> None:
    paths = toy_paths()
    write_toy_year(paths, [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 0.0]])
    config = replace(LeontiefPropagationConfig(), leontief_mode="nonsense")

    try:
        LeontiefCoefficientBuilder(paths, config).load_year(1995)
    except ValueError as error:
        assert "Unknown Leontief coefficient mode" in str(error)
    else:
        raise AssertionError("Expected invalid Leontief mode to raise ValueError")
