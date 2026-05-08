from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.abm_v3.diagnostics.production_bottlenecks import ProductionBottleneckReporter
from src.abm_v3.paths import ABMV3Paths


def reporter() -> ProductionBottleneckReporter:
    return ProductionBottleneckReporter(paths=ABMV3Paths(project_root=Path("tmp/production_bottleneck_tests")))


def observed_frame(x_value: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "country_sector": ["AAA | AAA | Industries | Manufacturing"],
            "Year": [2001],
            "X": [x_value],
            "Country": ["AAA"],
            "Country_detail": ["AAA"],
            "Category": ["Industries"],
            "Sector": ["Manufacturing"],
        }
    )


def predicted_frame(**overrides: object) -> pd.DataFrame:
    row = {
        "country_sector": "AAA | AAA | Industries | Manufacturing",
        "Year": 2001,
        "X": 80.0,
        "planned_output": 80.0,
        "D": 100.0,
        "K": 120.0,
        "desired_output": 80.0,
        "input_availability": 40.0,
        "adjusted_input_availability": 40.0,
        "substitution_gain": 0.0,
        "effective_input_intensity": 0.5,
        "input_intensity_source": "node",
        "input_feasible_output": 80.0,
        "input_stress_ratio": 1.0,
        "input_stress_factor": 1.0,
        "input_constraint_penalty": 0.0,
        "input_constraint_binding": False,
        "input_feasibility_missing": False,
        "realized_output": 80.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_node_report_computes_stage_ratios() -> None:
    report = reporter().build_node_report(
        predicted=predicted_frame(),
        observed=observed_frame(100.0),
        validation_year=2001,
    )

    assert report.loc[0, "planned_to_observed_ratio"] == 0.8
    assert report.loc[0, "realized_to_observed_ratio"] == 0.8
    assert report.loc[0, "planning_below_observed"] is True
    assert report.loc[0, "dominant_bottleneck"] == "planning"


def test_demand_bottleneck_classification() -> None:
    report = reporter().build_node_report(
        predicted=predicted_frame(
            X=70.0,
            planned_output=120.0,
            D=70.0,
            K=120.0,
            desired_output=70.0,
            input_feasible_output=100.0,
            realized_output=70.0,
        ),
        observed=observed_frame(100.0),
        validation_year=2001,
    )

    assert report.loc[0, "dominant_bottleneck"] == "demand"


def test_capacity_bottleneck_classification() -> None:
    report = reporter().build_node_report(
        predicted=predicted_frame(
            X=60.0,
            planned_output=120.0,
            D=120.0,
            K=60.0,
            desired_output=60.0,
            input_feasible_output=100.0,
            realized_output=60.0,
        ),
        observed=observed_frame(100.0),
        validation_year=2001,
    )

    assert report.loc[0, "dominant_bottleneck"] == "capacity"


def test_input_bottleneck_classification() -> None:
    report = reporter().build_node_report(
        predicted=predicted_frame(
            X=80.0,
            planned_output=120.0,
            D=120.0,
            K=120.0,
            desired_output=120.0,
            input_feasible_output=60.0,
            input_stress_factor=0.7,
            input_constraint_binding=True,
            input_constraint_penalty=40.0,
            realized_output=80.0,
        ),
        observed=observed_frame(100.0),
        validation_year=2001,
    )

    assert report.loc[0, "dominant_bottleneck"] == "input"


def test_no_underproduction_classification() -> None:
    report = reporter().build_node_report(
        predicted=predicted_frame(X=98.0, realized_output=98.0),
        observed=observed_frame(100.0),
        validation_year=2001,
    )

    assert report.loc[0, "dominant_bottleneck"] == "no_underproduction"


def test_zero_observed_output_safe_ratios() -> None:
    report = reporter().build_node_report(
        predicted=predicted_frame(X=10.0, realized_output=10.0),
        observed=observed_frame(0.0),
        validation_year=2001,
    )

    assert np.isnan(report.loc[0, "output_ratio"])
    assert np.isnan(report.loc[0, "planned_to_observed_ratio"])
    assert not np.isinf(report.loc[0, "output_ratio"])


def test_aggregate_report_sums_outputs() -> None:
    node_report = reporter().build_node_report(
        predicted=pd.concat(
            [
                predicted_frame(country_sector="A", X=80.0, planned_output=80.0, realized_output=80.0),
                predicted_frame(country_sector="B", X=98.0, planned_output=110.0, realized_output=98.0),
            ],
            ignore_index=True,
        ),
        observed=pd.DataFrame(
            {
                "country_sector": ["A", "B"],
                "Year": [2001, 2001],
                "X": [100.0, 100.0],
                "Country": ["AAA", "BBB"],
                "Sector": ["Manufacturing", "Services"],
            }
        ),
        validation_year=2001,
    )

    aggregate = reporter().build_aggregate_report(node_report, ["Year"])

    assert aggregate.loc[0, "observed_output_total"] == 200.0
    assert aggregate.loc[0, "simulated_output_total"] == 178.0
    assert aggregate.loc[0, "node_count"] == 2
    assert aggregate.loc[0, "share_bottleneck_planning"] == 0.5
    assert aggregate.loc[0, "share_no_underproduction"] == 0.5


def test_report_handles_missing_optional_columns() -> None:
    predicted = predicted_frame().drop(columns=["input_feasible_output"])

    report = reporter().build_node_report(
        predicted=predicted,
        observed=observed_frame(100.0),
        validation_year=2001,
    )

    assert "input_feasible_output" in report.columns
    assert np.isnan(report.loc[0, "input_feasible_output"])
