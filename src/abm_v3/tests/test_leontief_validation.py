from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy import sparse

from src.abm_v3.config import ABMV3Config, LeontiefPropagationConfig
from src.abm_v3.leontief.coefficients import LeontiefYearData
from src.abm_v3.leontief.propagation import LeontiefPropagationEngine
from src.abm_v3.leontief.validation import LeontiefPropagationValidator
from src.abm_v3.paths import ABMV3Paths
from src.abm_v3.runner import build_parser, run_leontief_year


def make_year_data() -> LeontiefYearData:
    labels = pd.DataFrame(
        {
            "country_sector": [
                "AAA | AAA | Industries | Agriculture",
                "BBB | BBB | Industries | Manufacturing",
            ],
            "Country": ["AAA", "BBB"],
            "Country_detail": ["AAA", "BBB"],
            "Category": ["Industries", "Industries"],
            "Sector": ["Agriculture", "Manufacturing"],
        }
    )
    return LeontiefYearData(
        year=1995,
        labels=labels,
        X_observed=pd.Series([0.0, 10.0], index=labels["country_sector"]),
        Y_final_demand=pd.Series([3.0, 4.0], index=labels["country_sector"]),
        A=sparse.csr_matrix([[0.0, 0.0], [0.0, 0.0]]),
        invalid_output_columns=labels.iloc[[0]].assign(Year=1995, X_observed=0.0, reason="non_positive_output"),
    )


def test_node_comparison_safe_ratios() -> None:
    year_data = make_year_data()
    result = LeontiefPropagationEngine(max_rounds=0).propagate(year_data)

    comparison = LeontiefPropagationValidator().build_node_comparison(year_data, result)

    assert np.isnan(comparison.loc[0, "output_ratio"])
    assert np.isnan(comparison.loc[0, "absolute_percentage_error"])
    assert comparison.loc[1, "output_ratio"] == 0.4


def test_summary_metrics() -> None:
    year_data = make_year_data()
    result = LeontiefPropagationEngine(max_rounds=0).propagate(year_data)
    validator = LeontiefPropagationValidator()
    comparison = validator.build_node_comparison(year_data, result)

    summary = validator.build_summary(year_data, result, comparison)

    assert summary.loc[0, "observed_output_total"] == 10.0
    assert summary.loc[0, "accumulated_output_total"] == 7.0
    assert summary.loc[0, "absolute_error_total"] == 9.0
    assert summary.loc[0, "relative_error_total"] == 0.9
    assert summary.loc[0, "invalid_output_columns_count"] == 1


def test_runner_command_exists_or_smoke() -> None:
    parser = build_parser()
    args = parser.parse_args(["leontief-propagate", "--year", "1995", "--max-rounds", "2"])
    assert args.command == "leontief-propagate"

    paths = ABMV3Paths(project_root=Path("tmp") / "abm_v3_leontief_runner_tests" / uuid4().hex)
    labels = [
        "AAA | AAA | Industries | Agriculture",
        "BBB | BBB | Industries | Manufacturing",
    ]
    matrix_dir = paths.parquet_root / "1995"
    raw_dir = paths.raw_root / "1995"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([[0.0, 0.1], [0.2, 0.0]], index=labels, columns=labels).to_parquet(matrix_dir / "T.parquet")
    pd.DataFrame([[1.0], [2.0]], index=labels, columns=["fd1"]).to_parquet(matrix_dir / "FD.parquet")
    raw_dir.joinpath("labels_T.txt").write_text(
        "AAA\tAAA\tIndustries\tAgriculture\t\n"
        "BBB\tBBB\tIndustries\tManufacturing\t\n",
        encoding="utf-8",
    )
    config = ABMV3Config(leontief=LeontiefPropagationConfig(max_rounds=2))

    output = run_leontief_year(1995, paths=paths, config=config)

    assert paths.leontief_summary_path(1995).exists()
    assert output["summary"].loc[0, "rounds_used"] <= 2
