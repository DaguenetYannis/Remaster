from __future__ import annotations

import pandas as pd

from src.abm_v3.calibration.emissions_model import EmissionsIntensityModel


def test_economic_only_uses_sector_average_trend() -> None:
    panel = pd.DataFrame(
        {
            "Sector": ["A", "A", "B"],
            "delta_log_EI_next": [-0.1, -0.3, 0.2],
            "green_capability": [1.0, 0.0, 1.0],
            "g_network": [1.0, 0.0, 1.0],
        }
    )

    model = EmissionsIntensityModel(mode="economic_only").fit(panel)
    predicted = model.predict_delta_ei(pd.DataFrame({"Sector": ["A", "B", "C"]}))

    assert predicted.tolist() == [-0.2, 0.2, 0.0]


def test_green_transition_uses_green_and_network_features() -> None:
    panel = pd.DataFrame(
        {
            "log_EI_lag1": [1.0, 1.1, 1.2, 1.3],
            "green_capability": [0.0, 0.5, 1.0, 1.5],
            "g_in": [0.1, 0.2, 0.3, 0.4],
            "g_out": [0.4, 0.3, 0.2, 0.1],
            "g_network": [0.25, 0.25, 0.25, 0.25],
            "general_complexity": [0.0, 0.1, 0.2, 0.3],
            "delta_log_EI_next": [-0.01, -0.02, -0.03, -0.04],
        }
    )

    model = EmissionsIntensityModel(mode="green_transition").fit(panel)

    assert "green_capability" in model.get_feature_names()
    assert "g_in" in model.get_feature_names()
    assert "g_out" in model.get_feature_names()
