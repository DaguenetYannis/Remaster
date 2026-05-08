from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from src.abm_v3.config import LeontiefPropagationConfig
from src.abm_v3.leontief.coefficients import LeontiefCoefficientBuilder
from src.abm_v3.paths import ABMV3Paths


def toy_paths() -> ABMV3Paths:
    root = Path("tmp") / "abm_v3_leontief_coefficients_tests" / uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return ABMV3Paths(project_root=root)


def write_toy_year(
    paths: ABMV3Paths,
    t_values: list[list[float]] | None = None,
    fd_values: list[list[float]] | None = None,
    year: int = 1995,
) -> None:
    labels = [
        "AAA | AAA | Industries | Agriculture",
        "BBB | BBB | Industries | Manufacturing",
    ]
    matrix_dir = paths.parquet_root / str(year)
    raw_dir = paths.raw_root / str(year)
    matrix_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        t_values or [[10.0, 20.0], [30.0, 40.0]],
        index=labels,
        columns=labels,
    ).to_parquet(matrix_dir / "T.parquet")
    pd.DataFrame(
        fd_values or [[50.0, 10.0], [20.0, 40.0]],
        index=labels,
        columns=[f"fd{i + 1}" for i in range(len(fd_values[0]) if fd_values else 2)],
    ).to_parquet(matrix_dir / "FD.parquet")
    raw_dir.joinpath("labels_T.txt").write_text(
        "AAA\tAAA\tIndustries\tAgriculture\t\n"
        "BBB\tBBB\tIndustries\tManufacturing\t\n",
        encoding="utf-8",
    )


def test_builds_technical_coefficients_by_column() -> None:
    paths = toy_paths()
    write_toy_year(paths)

    year_data = LeontiefCoefficientBuilder(paths, LeontiefPropagationConfig()).load_year(1995)

    x_values = np.array([10.0 + 30.0 + 60.0, 20.0 + 40.0 + 60.0])
    expected = np.array([[10.0, 20.0], [30.0, 40.0]]) / x_values
    assert np.allclose(year_data.A.toarray(), expected)


def test_final_demand_total() -> None:
    paths = toy_paths()
    write_toy_year(paths)

    year_data = LeontiefCoefficientBuilder(paths, LeontiefPropagationConfig()).load_year(1995)

    assert year_data.Y_final_demand.tolist() == [60.0, 60.0]


def test_invalid_output_columns_are_reported() -> None:
    paths = toy_paths()
    write_toy_year(paths, t_values=[[0.0, 2.0], [0.0, 3.0]], fd_values=[[0.0], [5.0]])

    year_data = LeontiefCoefficientBuilder(paths, LeontiefPropagationConfig()).load_year(1995)

    assert year_data.invalid_output_columns is not None
    assert year_data.invalid_output_columns["country_sector"].tolist() == [
        "AAA | AAA | Industries | Agriculture"
    ]
    assert np.isfinite(year_data.A.toarray()).all()
    assert np.allclose(year_data.A.toarray()[:, 0], [0.0, 0.0])
