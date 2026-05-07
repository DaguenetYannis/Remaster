from __future__ import annotations

import numpy as np
import pandas as pd

from src.abm_v3.dynamics.production import realize_production_with_soft_input_constraint
from src.abm_v3.input_panel_builder import ABMV3InputPanelBuilder
from src.abm_v3.paths import ABMV3Paths


def builder() -> ABMV3InputPanelBuilder:
    return ABMV3InputPanelBuilder(ABMV3Paths(project_root="tmp/abm_v3_input_feasibility_tests"))


def intensity_panel(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_observed_input_intensity() -> None:
    panel = builder().add_input_intensity_features(
        intensity_panel(
            [
                {
                    "Country": "AAA",
                    "Category": "Industries",
                    "Sector": "Agriculture",
                    "Year": 1995,
                    "X_observed": 100.0,
                    "M_observed": 40.0,
                }
            ]
        )
    )

    assert panel.loc[0, "observed_input_intensity"] == 0.4
    assert panel.loc[0, "effective_input_intensity"] == 0.4
    assert panel.loc[0, "input_intensity_source"] == "node"


def test_zero_M_positive_X_uses_fallback() -> None:
    panel = builder().add_input_intensity_features(
        intensity_panel(
            [
                {"Country": "AAA", "Category": "Industries", "Sector": "A", "Year": 1995, "X_observed": 100.0, "M_observed": 0.0},
                {"Country": "AAA", "Category": "Industries", "Sector": "B", "Year": 1995, "X_observed": 100.0, "M_observed": 50.0},
            ]
        )
    )

    assert panel.loc[0, "observed_input_intensity"] == 0.0
    assert panel.loc[0, "effective_input_intensity"] == 0.25
    assert panel.loc[0, "input_intensity_source"] == "country_category"


def test_country_ecosystem_fallback_before_sector() -> None:
    panel = builder().add_input_intensity_features(
        intensity_panel(
            [
                {"Country": "AAA", "Category": "Cat1", "Sector": "S1", "Year": 1995, "X_observed": 100.0, "M_observed": 0.0},
                {"Country": "AAA", "Category": "Cat2", "Sector": "S2", "Year": 1995, "X_observed": 100.0, "M_observed": 50.0},
                {"Country": "BBB", "Category": "Cat9", "Sector": "S1", "Year": 1995, "X_observed": 100.0, "M_observed": 80.0},
            ]
        )
    )

    assert panel.loc[0, "input_intensity_source"] == "country_ecosystem"
    assert panel.loc[0, "effective_input_intensity"] == 0.25


def test_sector_fallback_only_after_country_fallbacks() -> None:
    panel = builder().add_input_intensity_features(
        intensity_panel(
            [
                {"Country": "AAA", "Category": "Cat1", "Sector": "S1", "Year": 1995, "X_observed": 100.0, "M_observed": 0.0},
                {"Country": "BBB", "Category": "Cat2", "Sector": "S1", "Year": 1995, "X_observed": 100.0, "M_observed": 50.0},
            ]
        )
    )

    assert panel.loc[0, "input_intensity_source"] == "sector"
    assert panel.loc[0, "effective_input_intensity"] == 0.25


def test_global_fallback() -> None:
    panel = builder().add_input_intensity_features(
        intensity_panel(
            [
                {"Country": "AAA", "Category": "Cat1", "Sector": "S1", "Year": 1995, "X_observed": 100.0, "M_observed": 0.0},
                {"Country": "BBB", "Category": "Cat2", "Sector": "S2", "Year": 1995, "X_observed": 100.0, "M_observed": 50.0},
            ]
        )
    )

    assert panel.loc[0, "input_intensity_source"] == "global"
    assert panel.loc[0, "effective_input_intensity"] == 0.25


def test_missing_input_intensity() -> None:
    panel = builder().add_input_intensity_features(
        intensity_panel(
            [
                {"Country": "AAA", "Category": "Cat1", "Sector": "S1", "Year": 1995, "X_observed": 100.0, "M_observed": 0.0},
            ]
        )
    )

    assert np.isnan(panel.loc[0, "effective_input_intensity"])
    assert panel.loc[0, "input_intensity_source"] == "missing"


def test_realize_production_soft_input_constraint() -> None:
    result = realize_production_with_soft_input_constraint(
        planned_output=pd.Series([100.0]),
        demand=pd.Series([100.0]),
        capacity=pd.Series([120.0]),
        adjusted_input_availability=pd.Series([40.0]),
        effective_input_intensity=pd.Series([0.4]),
        input_rigidity=0.5,
    )

    assert result.loc[0, "desired_output"] == 100.0
    assert result.loc[0, "input_feasible_output"] == 100.0
    assert result.loc[0, "input_stress_factor"] == 1.0
    assert result.loc[0, "realized_output"] == 100.0


def test_realize_production_shortage_softens() -> None:
    result = realize_production_with_soft_input_constraint(
        planned_output=pd.Series([100.0]),
        demand=pd.Series([100.0]),
        capacity=pd.Series([120.0]),
        adjusted_input_availability=pd.Series([10.0]),
        effective_input_intensity=pd.Series([0.4]),
        input_rigidity=0.5,
    )

    assert result.loc[0, "input_feasible_output"] == 25.0
    assert result.loc[0, "input_stress_ratio"] == 0.25
    assert result.loc[0, "input_stress_factor"] == 0.5
    assert result.loc[0, "realized_output"] == 50.0


def test_missing_input_feasibility_uses_desired_output_with_flag() -> None:
    result = realize_production_with_soft_input_constraint(
        planned_output=pd.Series([100.0]),
        demand=pd.Series([90.0]),
        capacity=pd.Series([120.0]),
        adjusted_input_availability=pd.Series([10.0]),
        effective_input_intensity=pd.Series([np.nan]),
        input_rigidity=0.5,
    )

    assert result.loc[0, "realized_output"] == 90.0
    assert result.loc[0, "input_feasibility_missing"]
    assert not result.loc[0, "input_constraint_binding"]
