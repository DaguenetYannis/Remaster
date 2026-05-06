from __future__ import annotations

import pandas as pd

from src.abm_v3.diagnostics.collapse import CollapseThresholds, detect_bad_transition
from src.abm_v3.diagnostics.decomposition import emissions_decomposition


def previous_current() -> tuple[pd.DataFrame, pd.DataFrame]:
    previous = pd.DataFrame(
        {
            "country_sector": ["A", "B"],
            "X": [100.0, 100.0],
            "EI": [2.0, 2.0],
        }
    )
    current = pd.DataFrame(
        {
            "country_sector": ["A", "B"],
            "X": [50.0, 50.0],
            "EI": [1.9, 1.9],
        }
    )
    return previous, current


def test_emissions_decomposition_returns_expected_columns() -> None:
    previous, current = previous_current()
    decomposed, summary = emissions_decomposition(previous, current)
    expected = {"ei_effect", "output_effect", "interaction_effect", "delta_emissions_approx"}
    assert expected.issubset(decomposed.columns)
    assert expected.issubset(summary.keys())


def test_bad_transition_triggers_when_emissions_fall_because_output_collapses() -> None:
    previous, current = previous_current()
    result = detect_bad_transition(
        previous,
        current,
        thresholds=CollapseThresholds(output_loss_fraction=0.2),
    )
    assert result["bad_transition"]
