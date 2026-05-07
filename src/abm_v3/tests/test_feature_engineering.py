from __future__ import annotations

import numpy as np
import pandas as pd

from src.abm_v3.feature_engineering import FeatureEngineer


def toy_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "country_sector": ["A", "A", "B", "B"],
            "Country": ["AAA", "AAA", "BBB", "BBB"],
            "Sector": ["Agriculture", "Agriculture", "Manufacturing", "Manufacturing"],
            "Year": [1995, 1996, 1995, 1996],
            "X": [10.0, 12.0, 0.0, 3.0],
            "D": [9.0, 13.0, 1.0, 4.0],
            "EI": [2.0, 1.5, 3.0, 2.0],
        }
    )


def test_lag_creation_works() -> None:
    engineered = FeatureEngineer().create_lags(toy_panel(), ["X"])
    a_rows = engineered[engineered["country_sector"] == "A"].sort_values("Year")
    assert np.isnan(a_rows.iloc[0]["X_lag1"])
    assert a_rows.iloc[1]["X_lag1"] == 10.0


def test_growth_rates_do_not_produce_inf() -> None:
    engineered = FeatureEngineer().create_growth_rates(toy_panel(), ["X"])
    assert not np.isinf(engineered["X_growth"].dropna()).any()


def test_emissions_identity_is_created() -> None:
    engineered = FeatureEngineer().create_emissions(toy_panel())
    assert (engineered["emissions"] == engineered["X"] * engineered["EI"]).all()


def test_create_next_period_targets() -> None:
    toy_targets = pd.DataFrame(
        {
            "country_sector": ["A", "A", "B", "B"],
            "Year": [1995, 1996, 1995, 1996],
            "X": [100.0, 110.0, 0.0, 5.0],
            "EI": [2.0, 1.8, 3.0, 2.0],
        }
    )

    engineered = FeatureEngineer().create_next_period_targets(toy_targets)
    a_1995 = engineered[
        (engineered["country_sector"] == "A") & (engineered["Year"] == 1995)
    ].iloc[0]
    a_1996 = engineered[
        (engineered["country_sector"] == "A") & (engineered["Year"] == 1996)
    ].iloc[0]
    b_1995 = engineered[
        (engineered["country_sector"] == "B") & (engineered["Year"] == 1995)
    ].iloc[0]

    assert np.isfinite(a_1995["delta_log_X_next"])
    assert np.isfinite(a_1995["delta_log_EI_next"])
    assert np.isnan(a_1996["delta_log_X_next"])
    assert np.isnan(b_1995["delta_log_X_next"])
    assert not np.isinf(engineered[["delta_log_X_next", "delta_log_EI_next"]].to_numpy()).any()
