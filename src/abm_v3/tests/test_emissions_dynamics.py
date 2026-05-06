from __future__ import annotations

import numpy as np
import pandas as pd

from src.abm_v3.dynamics.emissions import compute_emissions
from src.abm_v3.calibration.emissions_model import EmissionsIntensityModel


def test_emissions_identity_holds() -> None:
    output = pd.Series([10.0, 5.0])
    ei = pd.Series([2.0, 3.0])
    emissions = compute_emissions(output, ei)
    assert emissions.tolist() == [20.0, 15.0]


def test_ei_prediction_remains_non_negative() -> None:
    model = EmissionsIntensityModel(features=["feature"], target="delta_log_EI_next")
    fit_df = pd.DataFrame(
        {
            "feature": [0.0, 1.0, 2.0],
            "delta_log_EI_next": [-0.1, -0.2, -0.3],
        }
    )
    model.fit(fit_df)
    prediction_df = pd.DataFrame({"EI": [2.0, 3.0], "feature": [1.0, 2.0]})
    predicted = model.predict_next_ei(prediction_df)
    assert (predicted >= 0).all()
    assert np.isfinite(predicted).all()
