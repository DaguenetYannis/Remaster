from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy import sparse

from src.abm_v3.config import LeontiefPropagationConfig
from src.abm_v3.leontief.coefficients import LeontiefYearData
from src.abm_v3.leontief.outputs import LeontiefOutputWriter
from src.abm_v3.leontief.viability import LeontiefViabilityAnalyzer
from src.abm_v3.paths import ABMV3Paths


def labels(count: int = 2) -> pd.DataFrame:
    base = [
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
    return pd.DataFrame(base[:count])


def make_year_data(
    a_values: list[list[float]],
    x_values: list[float] | None = None,
    y_values: list[float] | None = None,
    negative_flows: pd.DataFrame | None = None,
) -> LeontiefYearData:
    node_labels = labels(len(a_values))
    return LeontiefYearData(
        year=1995,
        labels=node_labels,
        X_observed=pd.Series(x_values or [10.0] * len(a_values), index=node_labels["country_sector"]),
        Y_final_demand=pd.Series(y_values or [1.0] * len(a_values), index=node_labels["country_sector"]),
        A=sparse.csr_matrix(np.array(a_values, dtype=float)),
        negative_flows=negative_flows,
        total_negative_T_entries=1 if negative_flows is not None and (negative_flows["matrix"] == "T").any() else 0,
        total_negative_FD_entries=1 if negative_flows is not None and (negative_flows["matrix"] == "FD").any() else 0,
        most_negative_T_value=-2.0 if negative_flows is not None and (negative_flows["matrix"] == "T").any() else np.nan,
        most_negative_FD_value=-3.0 if negative_flows is not None and (negative_flows["matrix"] == "FD").any() else np.nan,
    )


def analyze(year_data: LeontiefYearData, config: LeontiefPropagationConfig | None = None):
    return LeontiefViabilityAnalyzer(config or LeontiefPropagationConfig()).analyze(year_data)


def test_column_diagnostics_detect_high_abs_column_sum() -> None:
    diagnostics = analyze(make_year_data([[0.7, 0.8], [0.6, 0.4]]))

    assert bool(diagnostics.columns.loc[0, "high_abs_column_sum"]) is True


def test_column_diagnostics_detect_large_coefficients() -> None:
    config = LeontiefPropagationConfig(large_coefficient_threshold=2.0)
    diagnostics = analyze(make_year_data([[3.0, 0.0], [0.0, 0.0]]), config)

    assert bool(diagnostics.columns.loc[0, "has_large_coefficients"]) is True


def test_column_diagnostics_detect_negative_coefficients() -> None:
    diagnostics = analyze(make_year_data([[-0.1, 0.0], [0.0, 0.0]]))

    assert bool(diagnostics.columns.loc[0, "has_negative_coefficients"]) is True


def test_near_zero_positive_output_flag() -> None:
    diagnostics = analyze(make_year_data([[0.0, 0.0], [0.0, 0.0]], x_values=[1e-12, 10.0]))

    assert bool(diagnostics.columns.loc[0, "near_zero_positive_output"]) is True


def test_negative_final_demand_flag() -> None:
    diagnostics = analyze(make_year_data([[0.0, 0.0], [0.0, 0.0]], y_values=[-1.0, 1.0]))

    assert bool(diagnostics.columns.loc[0, "negative_final_demand"]) is True


def test_invalid_output_column_flag() -> None:
    diagnostics = analyze(make_year_data([[0.0, 0.0], [0.0, 0.0]], x_values=[0.0, 10.0]))

    assert bool(diagnostics.columns.loc[0, "invalid_output_column"]) is True


def test_negative_flow_diagnostics_for_T_and_FD() -> None:
    negative_flows = pd.DataFrame(
        [
            {
                "Year": 1995,
                "matrix": "T",
                "row_country_sector": "AAA | AAA | Industries | Agriculture",
                "col_country_sector": "BBB | BBB | Industries | Manufacturing",
                "col_label": "",
                "value": -2.0,
            },
            {
                "Year": 1995,
                "matrix": "FD",
                "row_country_sector": "AAA | AAA | Industries | Agriculture",
                "col_country_sector": "",
                "col_label": "fd1",
                "value": -3.0,
            },
        ]
    )
    diagnostics = analyze(make_year_data([[0.0, 0.0], [0.0, 0.0]], negative_flows=negative_flows))

    assert set(diagnostics.negative_flows["matrix"]) == {"T", "FD"}
    assert diagnostics.summary.loc[0, "total_negative_T_entries"] == 1
    assert diagnostics.summary.loc[0, "total_negative_FD_entries"] == 1


def test_spectral_radius_estimate_below_one() -> None:
    config = LeontiefPropagationConfig(spectral_radius_tolerance=1e-10)
    diagnostics = analyze(make_year_data([[0.5, 0.0], [0.0, 0.2]]), config)

    spectral_a = diagnostics.spectral.loc[diagnostics.spectral["matrix"] == "A"].iloc[0]
    assert np.isclose(spectral_a["approximate_spectral_radius"], 0.5, atol=1e-6)
    assert bool(spectral_a["above_one"]) is False


def test_spectral_radius_estimate_above_one() -> None:
    diagnostics = analyze(make_year_data([[1.2, 0.0], [0.0, 0.2]]))

    spectral_a = diagnostics.spectral.loc[diagnostics.spectral["matrix"] == "A"].iloc[0]
    assert bool(spectral_a["above_one"]) is True


def test_summary_counts_suspicious_columns() -> None:
    diagnostics = analyze(
        make_year_data(
            [[1.2, -0.1], [0.0, 0.0]],
            x_values=[10.0, 0.0],
            y_values=[1.0, -1.0],
        )
    )

    assert diagnostics.summary.loc[0, "suspicious_column_count"] == 2


def test_output_writer_writes_viability_files() -> None:
    root = Path("tmp") / "abm_v3_leontief_viability_writer_tests" / uuid4().hex
    paths = ABMV3Paths(project_root=root)
    diagnostics = analyze(make_year_data([[0.5, 0.0], [0.0, 0.2]]))

    written = LeontiefOutputWriter(paths).write_viability(diagnostics)

    for path in written.values():
        assert Path(path).exists()
