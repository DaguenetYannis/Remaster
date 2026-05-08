from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from src.abm_v3.leontief.coefficients import LeontiefYearData
from src.abm_v3.leontief.propagation import LeontiefPropagationEngine


def year_data_from_arrays(a_values: object, y_values: list[float]) -> LeontiefYearData:
    labels = pd.DataFrame(
        {
            "country_sector": ["AAA | AAA | Industries | Agriculture", "BBB | BBB | Industries | Manufacturing"][: len(y_values)],
            "Country": ["AAA", "BBB"][: len(y_values)],
            "Country_detail": ["AAA", "BBB"][: len(y_values)],
            "Category": ["Industries", "Industries"][: len(y_values)],
            "Sector": ["Agriculture", "Manufacturing"][: len(y_values)],
        }
    )
    return LeontiefYearData(
        year=1995,
        labels=labels,
        X_observed=pd.Series(np.zeros(len(y_values)), index=labels["country_sector"]),
        Y_final_demand=pd.Series(y_values, index=labels["country_sector"]),
        A=a_values,
    )


def test_iterative_propagation_two_rounds() -> None:
    a_values = sparse.csr_matrix([[0.0, 0.5], [0.25, 0.0]])
    y_values = np.array([10.0, 20.0])
    year_data = year_data_from_arrays(a_values, y_values.tolist())

    result = LeontiefPropagationEngine(tolerance=0.0, max_rounds=2).propagate(year_data)

    expected = y_values + (a_values @ y_values) + (a_values @ (a_values @ y_values))
    assert result.rounds_used == 2
    assert np.allclose(result.X_iterative.to_numpy(dtype=float), expected)


def test_propagation_stops_on_tolerance() -> None:
    year_data = year_data_from_arrays(sparse.csr_matrix([[0.1]]), [1.0])

    result = LeontiefPropagationEngine(tolerance=1e-3, max_rounds=20).propagate(year_data)

    assert result.converged is True
    assert result.rounds_used < 20


def test_propagation_respects_max_rounds() -> None:
    year_data = year_data_from_arrays(sparse.csr_matrix([[0.9]]), [1.0])

    result = LeontiefPropagationEngine(tolerance=1e-20, max_rounds=2).propagate(year_data)

    assert result.converged is False
    assert result.rounds_used == 2


def test_sparse_and_dense_toy_results_match() -> None:
    dense_a = np.array([[0.0, 0.2], [0.3, 0.0]])
    sparse_a = sparse.csr_matrix(dense_a)
    y_values = [5.0, 7.0]

    sparse_result = LeontiefPropagationEngine(tolerance=0.0, max_rounds=3).propagate(
        year_data_from_arrays(sparse_a, y_values)
    )
    dense_result = LeontiefPropagationEngine(tolerance=0.0, max_rounds=3).propagate(
        year_data_from_arrays(dense_a, y_values)
    )

    assert np.allclose(sparse_result.X_iterative, dense_result.X_iterative)
