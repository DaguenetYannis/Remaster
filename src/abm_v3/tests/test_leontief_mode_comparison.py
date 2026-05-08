from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.abm_v3.config import ABMV3Config, LeontiefPropagationConfig
from src.abm_v3.runner import run_leontief_mode_comparison
from src.abm_v3.paths import ABMV3Paths


def test_mode_comparison_contains_all_modes() -> None:
    paths = ABMV3Paths(project_root=Path("tmp") / "abm_v3_leontief_mode_comparison_tests" / uuid4().hex)
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
    pd.DataFrame([[0.1, 0.0], [0.0, 0.1]], index=labels, columns=labels).to_parquet(matrix_dir / "T.parquet")
    pd.DataFrame([[1.0, -0.2], [2.0, -0.1]], index=labels, columns=fd_labels).to_parquet(matrix_dir / "FD.parquet")
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
    modes = ["raw", "fd_without_inventory", "positive_final_demand_only"]
    config = ABMV3Config(leontief=LeontiefPropagationConfig(max_rounds=2))

    comparison = run_leontief_mode_comparison(1995, modes=modes, paths=paths, config=config)

    assert comparison["mode"].tolist() == modes
    assert paths.leontief_mode_comparison_path(1995).exists()
