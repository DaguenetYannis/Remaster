from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy import sparse

from src.abm_v3.config import ABMV3Config, LeontiefPropagationConfig
from src.abm_v3.leontief.behavioural import (
    BehaviouralLeontiefEngine,
    BehaviouralLeontiefValidator,
)
from src.abm_v3.leontief.coefficients import LeontiefYearData
from src.abm_v3.paths import ABMV3Paths
from src.abm_v3.runner import run_behavioural_leontief_year


def labels(count: int = 2) -> pd.DataFrame:
    rows = [
        {
            "country_sector": "AAA | AAA | Industries | Agriculture",
            "Country": "AAA",
            "Country_detail": "AAA",
            "Category": "Industries",
            "Sector": "Agriculture",
        },
        {
            "country_sector": "BBB | BBB | Industries | Manufacturing",
            "Country": "BBB",
            "Country_detail": "BBB",
            "Category": "Industries",
            "Sector": "Manufacturing",
        },
    ]
    return pd.DataFrame(rows[:count])


def make_year_data(
    a_values: list[list[float]],
    y_values: list[float],
    x_observed: list[float] | None = None,
    mode: str = "fd_without_inventory",
) -> LeontiefYearData:
    node_labels = labels(len(y_values))
    return LeontiefYearData(
        year=1995,
        mode=mode,
        labels=node_labels,
        X_observed=pd.Series(x_observed or [10.0] * len(y_values), index=node_labels["country_sector"]),
        Y_final_demand=pd.Series(y_values, index=node_labels["country_sector"]),
        A=sparse.csr_matrix(np.array(a_values, dtype=float)),
    )


def test_soft_capacity_no_constraint_when_capacity_above_demand() -> None:
    engine = BehaviouralLeontiefEngine(LeontiefPropagationConfig(behavioural_capacity_eta=0.5))

    response = engine.apply_soft_capacity(np.array([100.0]), np.array([200.0]))

    assert response["capacity_stress"].tolist() == [1.0]
    assert response["realized_output"].tolist() == [100.0]
    assert response["capacity_binding"].tolist() == [False]


def test_soft_capacity_reduces_output_when_capacity_below_demand() -> None:
    engine = BehaviouralLeontiefEngine(LeontiefPropagationConfig(behavioural_capacity_eta=0.5))

    response = engine.apply_soft_capacity(np.array([100.0]), np.array([25.0]))

    assert response["capacity_ratio"].tolist() == [0.25]
    assert np.isclose(response["capacity_stress"][0], 0.5)
    assert np.isclose(response["realized_output"][0], 50.0)


def test_missing_capacity_does_not_collapse_node() -> None:
    engine = BehaviouralLeontiefEngine(LeontiefPropagationConfig())

    response = engine.apply_soft_capacity(np.array([100.0]), np.array([np.nan]))

    assert response["realized_output"].tolist() == [100.0]
    assert response["capacity_feasibility_missing"].tolist() == [True]
    assert response["capacity_stress"].tolist() == [1.0]


def test_behavioural_propagation_one_round() -> None:
    year_data = make_year_data([[0.0, 0.5], [0.25, 0.0]], [10.0, 20.0])
    capacity = pd.Series([100.0, 100.0], index=year_data.labels["country_sector"])

    result = BehaviouralLeontiefEngine(
        LeontiefPropagationConfig(behavioural_tolerance=0.0, behavioural_max_rounds=0)
    ).propagate(year_data, capacity)

    assert result.X_realized.tolist() == [10.0, 20.0]
    assert result.round_summaries.loc[0, "supplier_demand_total"] == 12.5


def test_behavioural_propagation_accumulates_realized_output() -> None:
    year_data = make_year_data([[0.0, 0.5], [0.25, 0.0]], [10.0, 20.0])
    capacity = pd.Series([100.0, 100.0], index=year_data.labels["country_sector"])

    result = BehaviouralLeontiefEngine(
        LeontiefPropagationConfig(behavioural_tolerance=0.0, behavioural_max_rounds=1)
    ).propagate(year_data, capacity)

    assert np.allclose(result.X_realized.to_numpy(dtype=float), [20.0, 22.5])


def test_behavioural_convergence_stops_on_residual() -> None:
    year_data = make_year_data([[0.1]], [1.0])
    capacity = pd.Series([100.0], index=year_data.labels["country_sector"])

    result = BehaviouralLeontiefEngine(
        LeontiefPropagationConfig(behavioural_tolerance=1e-3, behavioural_max_rounds=20)
    ).propagate(year_data, capacity)

    assert result.converged is True
    assert result.rounds_used < 20


def test_behavioural_node_comparison_uses_raw_observed_X() -> None:
    year_data = make_year_data([[0.0, 0.0], [0.0, 0.0]], [5.0, 5.0], x_observed=[100.0, 200.0])
    capacity = pd.Series([100.0, 100.0], index=year_data.labels["country_sector"])
    result = BehaviouralLeontiefEngine(LeontiefPropagationConfig(behavioural_max_rounds=0)).propagate(
        year_data,
        capacity,
    )

    comparison = BehaviouralLeontiefValidator().build_node_comparison(year_data, result)

    assert comparison["X_observed"].tolist() == [100.0, 200.0]


def test_behavioural_output_paths_include_mode() -> None:
    paths = ABMV3Paths(project_root=Path("tmp") / "behavioural_path_test" / uuid4().hex)

    assert paths.behavioural_leontief_output_path(1995, "raw") != paths.behavioural_leontief_output_path(
        1995,
        "fd_without_inventory",
    )


def test_runner_or_orchestration_smoke() -> None:
    paths = ABMV3Paths(project_root=Path("tmp") / "behavioural_runner_test" / uuid4().hex)
    node_labels = [
        "AAA | AAA | Industries | Agriculture",
        "BBB | BBB | Industries | Manufacturing",
    ]
    matrix_dir = paths.parquet_root / "1995"
    raw_dir = paths.raw_root / "1995"
    input_path = paths.abm_v3_historical_panel_file(1995, 2016)
    matrix_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    input_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([[0.1, 0.0], [0.0, 0.1]], index=node_labels, columns=node_labels).to_parquet(matrix_dir / "T.parquet")
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
    pd.DataFrame(
        {
            "Year": [1995, 1995],
            "country_sector": node_labels,
            "K": [10.0, 10.0],
            "X_observed": [1.0, 2.0],
        }
    ).to_parquet(input_path, index=False)
    config = ABMV3Config(
        leontief=LeontiefPropagationConfig(
            leontief_mode="fd_without_inventory",
            behavioural_max_rounds=1,
            write_behavioural_node_rounds=False,
        )
    )

    output = run_behavioural_leontief_year(1995, paths=paths, config=config)

    assert paths.behavioural_leontief_summary_path(1995, "fd_without_inventory").exists()
    assert output["summary"].loc[0, "rounds_used"] <= 1
