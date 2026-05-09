from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.abm_v3.config import ABMV3Config, LeontiefPropagationConfig
from src.abm_v3.paths import ABMV3Paths
from src.abm_v3.runner import run_behavioural_leontief_year, run_leontief_year


CORRECTED_MODE = "transpose_row_output_fd_without_inventory"
CORRECTED_ORIENTATION = "transpose_row_fd_without_inventory"


def toy_paths() -> ABMV3Paths:
    root = Path("tmp") / "abm_v3_output_structure_tests" / uuid4().hex[:8]
    root.mkdir(parents=True, exist_ok=True)
    return ABMV3Paths(project_root=root)


def expected_parts(paths: ABMV3Paths, relative_parts: list[str]) -> Path:
    return paths.project_root.joinpath(*relative_parts)


def test_pure_output_paths_are_structured() -> None:
    paths = toy_paths()

    path = paths.leontief_iterative_output_path(1995, "fd_without_inventory")

    assert path.parent == expected_parts(paths, ["data", "abm_v3", "leontief", "pure", "outputs"])
    assert path.name == "iterative_output_1995_fd_without_inventory.parquet"


def test_pure_propagation_diagnostics_paths_are_structured() -> None:
    paths = toy_paths()
    expected_parent = expected_parts(
        paths,
        ["data", "abm_v3", "leontief", "pure", "diagnostics", "propagation"],
    )

    assert paths.leontief_summary_path(1995, "fd_without_inventory").parent == expected_parent
    assert paths.leontief_rounds_path(1995, "fd_without_inventory").parent == expected_parent
    assert paths.leontief_node_comparison_path(1995, "fd_without_inventory").parent == expected_parent
    assert paths.leontief_summary_path(1995, "fd_without_inventory").name == "summary_1995_fd_without_inventory.csv"


def test_viability_diagnostics_paths_are_structured() -> None:
    paths = toy_paths()
    expected_parent = expected_parts(
        paths,
        ["data", "abm_v3", "leontief", "pure", "diagnostics", "viability"],
    )

    assert paths.leontief_viability_summary_path(1995, "fd_without_inventory").parent == expected_parent
    assert paths.leontief_viability_columns_path(1995, "fd_without_inventory").parent == expected_parent
    assert paths.leontief_negative_flows_path(1995, "fd_without_inventory").parent == expected_parent
    assert paths.leontief_spectral_diagnostics_path(1995, "fd_without_inventory").parent == expected_parent
    assert paths.leontief_top_unstable_nodes_path(1995, "fd_without_inventory").parent == expected_parent
    assert paths.leontief_invalid_output_columns_path(1995, "fd_without_inventory").parent == expected_parent


def test_mode_comparison_paths_are_structured() -> None:
    paths = toy_paths()
    expected_parent = expected_parts(
        paths,
        ["data", "abm_v3", "leontief", "pure", "diagnostics", "mode_comparison"],
    )

    assert paths.leontief_mode_comparison_path(1995).parent == expected_parent
    assert paths.leontief_mode_diagnostics_path(1995, "fd_without_inventory").parent == expected_parent
    assert paths.leontief_excluded_fd_columns_path(1995, "fd_without_inventory").parent == expected_parent
    assert paths.leontief_rescaled_columns_path(1995, "fd_without_inventory").parent == expected_parent


def test_orientation_audit_paths_are_structured() -> None:
    paths = toy_paths()
    expected_parent = expected_parts(
        paths,
        ["data", "abm_v3", "leontief", "pure", "diagnostics", "orientation_audit"],
    )

    assert paths.leontief_orientation_summary_path(1995).parent == expected_parent
    assert paths.leontief_orientation_summary_range_path(1995, 2016).parent == expected_parent
    assert paths.leontief_orientation_node_comparison_path(1995).parent == expected_parent
    assert paths.leontief_orientation_suspicious_nodes_path(1995).parent == expected_parent


def test_behavioural_summary_paths_are_structured() -> None:
    paths = toy_paths()

    path = paths.behavioural_leontief_summary_path(1995, "fd_without_inventory")

    assert path.parent == expected_parts(
        paths,
        ["data", "abm_v3", "leontief", "behavioural", "diagnostics", "summary"],
    )


def test_behavioural_round_paths_are_structured() -> None:
    paths = toy_paths()

    path = paths.behavioural_leontief_rounds_path(1995, "fd_without_inventory")

    assert path.parent == expected_parts(
        paths,
        ["data", "abm_v3", "leontief", "behavioural", "diagnostics", "rounds"],
    )


def test_behavioural_node_comparison_paths_are_structured() -> None:
    paths = toy_paths()

    path = paths.behavioural_leontief_node_comparison_path(1995, "fd_without_inventory")

    assert path.parent == expected_parts(
        paths,
        ["data", "abm_v3", "leontief", "behavioural", "diagnostics", "node_comparison"],
    )


def test_behavioural_node_round_paths_are_structured() -> None:
    paths = toy_paths()

    path = paths.behavioural_leontief_node_rounds_path(1995, "fd_without_inventory")

    assert path.parent == expected_parts(
        paths,
        ["data", "abm_v3", "leontief", "behavioural", "diagnostics", "node_rounds"],
    )


def test_suffix_includes_input_panel_orientation_when_provided() -> None:
    paths = toy_paths()

    suffix = paths.format_leontief_suffix(1995, CORRECTED_MODE, CORRECTED_ORIENTATION)

    assert suffix == "1995_transpose_row_output_fd_without_inventory__transpose_row_fd_without_inventory"
    assert paths.leontief_summary_path(1995, CORRECTED_MODE, CORRECTED_ORIENTATION).name == (
        "summary_1995_transpose_row_output_fd_without_inventory__transpose_row_fd_without_inventory.csv"
    )


def write_toy_year(paths: ABMV3Paths) -> None:
    labels = [
        "AAA | AAA | Industries | Agriculture",
        "BBB | BBB | Industries | Manufacturing",
    ]
    matrix_dir = paths.parquet_root / "1995"
    raw_dir = paths.raw_root / "1995"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([[0.1, 0.0], [0.0, 0.1]], index=labels, columns=labels).to_parquet(matrix_dir / "T.parquet")
    pd.DataFrame([[1.0, -0.2], [2.0, -0.1]], index=labels, columns=["fd", "Changes in inventories P.52"]).to_parquet(
        matrix_dir / "FD.parquet"
    )
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
    input_path = paths.abm_v3_historical_panel_file(1995, 2016)
    input_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Year": [1995, 1995],
            "country_sector": labels,
            "X_observed": [1.0, 2.0],
            "K": [10.0, 10.0],
        }
    ).to_parquet(input_path, index=False)


def test_old_commands_still_run_with_new_output_paths() -> None:
    paths = toy_paths()
    write_toy_year(paths)
    leontief_config = ABMV3Config(leontief=LeontiefPropagationConfig(leontief_mode="fd_without_inventory", max_rounds=0))
    behavioural_config = ABMV3Config(
        leontief=LeontiefPropagationConfig(
            leontief_mode="fd_without_inventory",
            behavioural_max_rounds=0,
            write_behavioural_node_rounds=False,
        )
    )

    run_leontief_year(1995, paths=paths, config=leontief_config)
    run_behavioural_leontief_year(1995, paths=paths, config=behavioural_config)

    assert paths.leontief_summary_path(1995, "fd_without_inventory").exists()
    assert paths.leontief_viability_summary_path(1995, "fd_without_inventory").exists()
    assert paths.behavioural_leontief_summary_path(1995, "fd_without_inventory").exists()
