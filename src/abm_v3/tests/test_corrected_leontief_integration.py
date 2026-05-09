from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from src.abm_v3.config import ABMV3Config, LeontiefPropagationConfig
from src.abm_v3.data_loader import ABMV3DataLoader
from src.abm_v3.leontief.coefficients import LeontiefCoefficientBuilder
from src.abm_v3.paths import ABMV3Paths
from src.abm_v3.runner import run_leontief_year


CORRECTED_ORIENTATION = "transpose_row_fd_without_inventory"


def toy_paths() -> ABMV3Paths:
    root = Path("tmp") / "abm_v3_corrected_leontief_tests" / uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return ABMV3Paths(project_root=root)


def labels() -> list[str]:
    return [
        "AAA | AAA | Industries | Agriculture",
        "BBB | BBB | Industries | Manufacturing",
    ]


def write_toy_year(
    paths: ABMV3Paths,
    t_values: list[list[float]] | None = None,
    fd_values: list[list[float]] | None = None,
) -> None:
    node_labels = labels()
    fd_labels = [
        "AAA | AAA | Final demand | Household final consumption",
        "AAA | AAA | Final demand | Changes in inventories P.52",
    ]
    matrix_dir = paths.parquet_root / "1995"
    raw_dir = paths.raw_root / "1995"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        t_values or [[1.0, 2.0], [3.0, 4.0]],
        index=node_labels,
        columns=node_labels,
    ).to_parquet(matrix_dir / "T.parquet")
    pd.DataFrame(
        fd_values or [[10.0, -1.0], [20.0, -2.0]],
        index=node_labels,
        columns=fd_labels,
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


def write_input_panel(paths: ABMV3Paths, orientation: str | None, x_values: list[float], k_values: list[float]) -> Path:
    path = (
        paths.abm_v3_historical_panel_file(1995, 2016)
        if orientation is None
        else paths.abm_v3_corrected_historical_panel_file(1995, 2016, orientation)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Year": [1995, 1995],
            "country_sector": labels(),
            "X_observed": x_values,
            "K": k_values,
        }
    ).to_parquet(path, index=False)
    return path


def test_transpose_row_output_fd_without_inventory_coefficient_mode_exists() -> None:
    config = LeontiefPropagationConfig()

    assert CORRECTED_ORIENTATION.replace("transpose_row_fd", "transpose_row_output_fd") in config.allowed_leontief_modes


def test_transpose_row_output_fd_without_inventory_builds_A_from_T_transpose() -> None:
    paths = toy_paths()
    write_toy_year(paths, t_values=[[1.0, 2.0], [3.0, 4.0]], fd_values=[[10.0, -1.0], [20.0, -2.0]])
    config = LeontiefPropagationConfig(leontief_mode="transpose_row_output_fd_without_inventory")

    year_data = LeontiefCoefficientBuilder(paths, config).load_year(1995)

    expected_x = np.array([1.0 + 2.0 + 10.0, 3.0 + 4.0 + 20.0])
    expected_a = np.array([[1.0, 3.0], [2.0, 4.0]]) / expected_x
    assert year_data.Y_final_demand.tolist() == [10.0, 20.0]
    assert np.allclose(year_data.X_used_for_coefficients.to_numpy(dtype=float), expected_x)
    assert np.allclose(year_data.A.toarray(), expected_a)


def test_corrected_input_panel_loader_uses_corrected_path() -> None:
    paths = toy_paths()
    corrected_path = write_input_panel(paths, CORRECTED_ORIENTATION, [13.0, 27.0], [100.0, 200.0])

    selected_path = ABMV3DataLoader(paths).input_panel_path_for_orientation(1995, 2016, CORRECTED_ORIENTATION)
    panel = ABMV3DataLoader(paths).load_input_panel_for_orientation(1995, 2016, CORRECTED_ORIENTATION)

    assert selected_path == corrected_path
    assert panel["X_observed"].tolist() == [13.0, 27.0]


def test_missing_corrected_panel_raises_clear_error() -> None:
    paths = toy_paths()

    try:
        ABMV3DataLoader(paths).load_input_panel_for_orientation(1995, 2016, CORRECTED_ORIENTATION)
    except FileNotFoundError as error:
        message = str(error)
        assert "Corrected ABM v3 input panel is missing" in message
        assert "build-corrected-input-panel --start-year 1995 --end-year 2016 --overwrite" in message
    else:
        raise AssertionError("Expected missing corrected panel to raise FileNotFoundError")


def test_pure_leontief_validation_uses_corrected_X() -> None:
    paths = toy_paths()
    write_toy_year(paths)
    write_input_panel(paths, None, [999.0, 999.0], [10.0, 10.0])
    write_input_panel(paths, CORRECTED_ORIENTATION, [13.0, 27.0], [100.0, 200.0])
    config = ABMV3Config(
        leontief=LeontiefPropagationConfig(
            leontief_mode="transpose_row_output_fd_without_inventory",
            input_panel_orientation=CORRECTED_ORIENTATION,
            max_rounds=0,
        )
    )

    output = run_leontief_year(1995, paths=paths, config=config)

    assert output["node_comparison"]["X_observed"].tolist() == [13.0, 27.0]
    assert output["summary"].loc[0, "validation_reference"] == f"input_panel:{CORRECTED_ORIENTATION}:X_observed"
    assert paths.leontief_summary_path(
        1995,
        "transpose_row_output_fd_without_inventory",
        CORRECTED_ORIENTATION,
    ).exists()


def test_output_paths_include_input_panel_orientation() -> None:
    paths = toy_paths()

    old_path = paths.leontief_iterative_output_path(1995, "fd_without_inventory")
    corrected_path = paths.leontief_iterative_output_path(
        1995,
        "transpose_row_output_fd_without_inventory",
        CORRECTED_ORIENTATION,
    )

    assert corrected_path != old_path
    assert CORRECTED_ORIENTATION in corrected_path.name
