from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from src.abm_v3.config import ABMV3Config, LeontiefPropagationConfig
from src.abm_v3.leontief.orientation import (
    ALL_ORIENTATION_MODES,
    LeontiefOrientationAuditor,
    build_orientation_coefficients,
    build_orientation_output,
    orientation_candidate_from_mode,
    safe_divide_array,
)
from src.abm_v3.paths import ABMV3Paths
from src.abm_v3.runner import run_leontief_orientation_audit


def toy_paths() -> ABMV3Paths:
    root = Path("tmp") / "abm_v3_leontief_orientation_tests" / uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return ABMV3Paths(project_root=root)


def toy_labels() -> list[str]:
    return [
        "AAA | AAA | Industries | Agriculture",
        "BBB | BBB | Industries | Manufacturing",
    ]


def toy_fd_labels() -> list[str]:
    return [
        "AAA | AAA | Final demand | Household final consumption",
        "AAA | AAA | Final demand | Changes in inventories P.52",
    ]


def write_toy_year(
    paths: ABMV3Paths,
    t_values: list[list[float]],
    fd_values: list[list[float]],
    write_panel: bool = False,
) -> None:
    labels = toy_labels()
    matrix_dir = paths.parquet_root / "1995"
    raw_dir = paths.raw_root / "1995"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(t_values, index=labels, columns=labels).to_parquet(matrix_dir / "T.parquet")
    pd.DataFrame(fd_values, index=labels, columns=toy_fd_labels()).to_parquet(matrix_dir / "FD.parquet")
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
    if write_panel:
        panel_path = paths.abm_v3_historical_panel_file(1995, 2016)
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "Year": [1995, 1995],
                "country_sector": labels,
                "X_observed": [100.0, 200.0],
            }
        ).to_parquet(panel_path, index=False)


def toy_matrices() -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = toy_labels()
    t_matrix = pd.DataFrame([[10.0, 20.0], [30.0, 40.0]], index=labels, columns=labels)
    fd_matrix = pd.DataFrame([[50.0, 10.0], [20.0, 40.0]], index=labels, columns=toy_fd_labels())
    return t_matrix, fd_matrix


def test_current_column_output_formula() -> None:
    t_matrix, fd_matrix = toy_matrices()

    x_values = build_orientation_output("current_column_output", t_matrix, fd_matrix)

    assert np.allclose(x_values, [100.0, 120.0])


def test_row_output_formula() -> None:
    t_matrix, fd_matrix = toy_matrices()

    x_values = build_orientation_output("row_output_standard_io", t_matrix, fd_matrix)

    assert np.allclose(x_values, [90.0, 130.0])


def test_transpose_orientation_formula() -> None:
    t_matrix, fd_matrix = toy_matrices()

    a_matrix = build_orientation_coefficients("transpose_row_output", t_matrix, fd_matrix).toarray()

    expected = np.array([[10.0 / 90.0, 30.0 / 130.0], [20.0 / 90.0, 40.0 / 130.0]])
    assert np.allclose(a_matrix, expected)


def test_fd_without_inventory_orientation_excludes_inventory() -> None:
    paths = toy_paths()
    write_toy_year(paths, [[10.0, 20.0], [30.0, 40.0]], [[50.0, 10.0], [20.0, 40.0]])

    audit = LeontiefOrientationAuditor(paths).audit_year(
        1995,
        max_rounds=1,
        include_fd_without_inventory=True,
        reference="current",
    )
    row = audit.summary.loc[audit.summary["orientation_mode"] == "row_output_fd_without_inventory"].iloc[0]

    assert row["inventory_excluded"]
    assert row["Y_total"] == 70.0
    assert row["X_total"] == 170.0


def test_orientation_summary_contains_all_modes() -> None:
    paths = toy_paths()
    write_toy_year(paths, [[0.0, 0.1], [0.2, 0.0]], [[1.0, -0.1], [2.0, -0.2]])
    config = ABMV3Config(leontief=LeontiefPropagationConfig(spectral_radius_max_iter=20))

    audit = LeontiefOrientationAuditor(paths, config).audit_year(
        1995,
        max_rounds=2,
        tolerance=1e-8,
        include_fd_without_inventory=True,
        reference="current",
    )

    assert audit.summary["orientation_mode"].tolist() == ALL_ORIENTATION_MODES


def test_orientation_node_comparison_detects_large_row_column_difference() -> None:
    paths = toy_paths()
    write_toy_year(paths, [[0.0, 100.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]])

    audit = LeontiefOrientationAuditor(paths).audit_year(
        1995,
        max_rounds=1,
        include_fd_without_inventory=False,
        reference="current",
    )

    assert bool(audit.node_comparison["largest_difference_flag"].any())


def test_safe_ratios_no_inf() -> None:
    ratios = safe_divide_array(np.array([1.0, 2.0]), np.array([0.0, 1.0]))

    assert np.isnan(ratios[0])
    assert np.isfinite(ratios[1])


def test_invalid_orientation_raises_clear_error() -> None:
    try:
        orientation_candidate_from_mode("not_a_mode")
    except ValueError as error:
        assert "Unknown Leontief orientation mode" in str(error)
    else:
        raise AssertionError("Expected invalid orientation to raise ValueError")


def test_runner_writes_orientation_audit() -> None:
    paths = toy_paths()
    write_toy_year(paths, [[0.0, 0.1], [0.2, 0.0]], [[1.0, -0.1], [2.0, -0.2]], write_panel=True)

    output = run_leontief_orientation_audit(
        1995,
        paths=paths,
        config=ABMV3Config(leontief=LeontiefPropagationConfig(spectral_radius_max_iter=20)),
        max_rounds=2,
        reference="abm_ready",
    )

    assert paths.leontief_orientation_summary_path(1995).exists()
    assert output["audit"].summary["notes"].str.contains("ABM-ready panel X_observed used").any()
