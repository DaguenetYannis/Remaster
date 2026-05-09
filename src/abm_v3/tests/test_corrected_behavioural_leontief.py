from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.abm_v3.config import ABMV3Config, LeontiefPropagationConfig
from src.abm_v3.paths import ABMV3Paths
from src.abm_v3.runner import build_parser, run_behavioural_leontief_year, run_leontief_year


CORRECTED_ORIENTATION = "transpose_row_fd_without_inventory"
CORRECTED_MODE = "transpose_row_output_fd_without_inventory"


def toy_paths() -> ABMV3Paths:
    root = Path("tmp") / "ablb" / uuid4().hex[:8]
    root.mkdir(parents=True, exist_ok=True)
    return ABMV3Paths(project_root=root)


def labels() -> list[str]:
    return [
        "AAA | AAA | Industries | Agriculture",
        "BBB | BBB | Industries | Manufacturing",
    ]


def write_toy_year(paths: ABMV3Paths) -> None:
    node_labels = labels()
    matrix_dir = paths.parquet_root / "1995"
    raw_dir = paths.raw_root / "1995"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [[0.1, 0.0], [0.0, 0.1]],
        index=node_labels,
        columns=node_labels,
    ).to_parquet(matrix_dir / "T.parquet")
    pd.DataFrame(
        [[1.0, -0.2], [2.0, -0.1]],
        index=node_labels,
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


def write_input_panel(paths: ABMV3Paths, orientation: str | None, x_values: list[float], k_values: list[float]) -> None:
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


def test_behavioural_capacity_uses_corrected_K() -> None:
    paths = toy_paths()
    write_toy_year(paths)
    write_input_panel(paths, None, [10.0, 20.0], [1.0, 1.0])
    write_input_panel(paths, CORRECTED_ORIENTATION, [1.1, 2.1], [100.0, 200.0])
    config = ABMV3Config(
        leontief=LeontiefPropagationConfig(
            leontief_mode=CORRECTED_MODE,
            input_panel_orientation=CORRECTED_ORIENTATION,
            behavioural_max_rounds=0,
            write_behavioural_node_rounds=True,
        )
    )

    output = run_behavioural_leontief_year(1995, paths=paths, config=config)

    assert output["capacity"].tolist() == [100.0, 200.0]
    assert output["node_comparison"]["X_observed"].tolist() == [1.1, 2.1]
    assert output["result"].node_rounds["K"].tolist() == [100.0, 200.0]
    assert output["summary"].loc[0, "capacity_source"].startswith(
        f"input_panel:{CORRECTED_ORIENTATION}:K:"
    )
    assert paths.behavioural_leontief_summary_path(1995, CORRECTED_MODE, CORRECTED_ORIENTATION).exists()


def test_behavioural_output_paths_include_input_panel_orientation() -> None:
    paths = toy_paths()

    old_path = paths.behavioural_leontief_output_path(1995, "fd_without_inventory")
    corrected_path = paths.behavioural_leontief_output_path(1995, CORRECTED_MODE, CORRECTED_ORIENTATION)

    assert corrected_path != old_path
    assert CORRECTED_ORIENTATION in corrected_path.name


def test_existing_default_commands_still_work() -> None:
    parser = build_parser()
    leontief_args = parser.parse_args(["leontief-propagate", "--year", "1995", "--max-rounds", "0"])
    behavioural_args = parser.parse_args(["behavioural-leontief", "--year", "1995", "--max-rounds", "0"])
    assert leontief_args.input_panel_orientation is None
    assert behavioural_args.input_panel_orientation is None

    paths = toy_paths()
    write_toy_year(paths)
    write_input_panel(paths, None, [10.0, 20.0], [100.0, 100.0])
    leontief_config = ABMV3Config(leontief=LeontiefPropagationConfig(max_rounds=0))
    behavioural_config = ABMV3Config(
        leontief=LeontiefPropagationConfig(
            leontief_mode="fd_without_inventory",
            behavioural_max_rounds=0,
            write_behavioural_node_rounds=False,
        )
    )

    run_leontief_year(1995, paths=paths, config=leontief_config)
    run_behavioural_leontief_year(1995, paths=paths, config=behavioural_config)

    assert paths.leontief_summary_path(1995).exists()
    assert paths.behavioural_leontief_summary_path(1995, "fd_without_inventory").exists()
