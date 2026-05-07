from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ProductionPlanningModel:
    """Interpretable ridge production model for adaptive agents.

    The target is ``delta_log_X_next``. Historical ``D`` is treated as an
    observed demand proxy and used alongside own output persistence,
    sector-level trends, country-level trends, and constraint indicators.
    Missing target rows are dropped; missing selected features are dropped
    explicitly rather than converted into zero dynamics.
    """

    features: list[str] = field(default_factory=lambda: [
        "log_X_lag1",
        "D_growth",
        "demand_gap",
        "sector_X_growth",
        "country_X_growth",
        "input_availability_ratio",
        "inventory_stress",
        "capacity_utilization",
    ])
    target: str = "delta_log_X_next"
    l2_penalty: float = 1.0
    coefficients_: pd.Series | None = None
    intercept_: float = 0.0

    def _available_features(self, df: pd.DataFrame) -> list[str]:
        return [feature for feature in self.features if feature in df.columns]

    def fit(self, df: pd.DataFrame) -> "ProductionPlanningModel":
        selected_features = self._available_features(df)
        required = [*selected_features, self.target]
        missing = [column for column in [self.target] if column not in df.columns]
        if missing:
            raise ValueError(f"Missing production calibration columns: {missing}")
        if not selected_features:
            raise ValueError("No available production features for calibration.")
        train = df[required].dropna()
        if train.empty:
            raise ValueError("No complete rows available for production planning fit.")
        x = train[selected_features].to_numpy(dtype=float)
        y = train[self.target].to_numpy(dtype=float)
        x_design = np.column_stack([np.ones(len(x)), x])
        penalty = np.eye(x_design.shape[1]) * self.l2_penalty
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(x_design.T @ x_design + penalty, x_design.T @ y)
        self.intercept_ = float(beta[0])
        self.coefficients_ = pd.Series(beta[1:], index=selected_features, dtype=float)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.coefficients_ is None:
            return pd.Series(0.0, index=df.index, name="predicted_delta_log_X_next")
        feature_names = self.get_feature_names()
        missing = [column for column in feature_names if column not in df.columns]
        if missing:
            raise ValueError(f"Missing production prediction columns: {missing}")
        values = df[feature_names].fillna(0.0).to_numpy(dtype=float)
        prediction = self.intercept_ + values @ self.coefficients_.to_numpy(dtype=float)
        return pd.Series(prediction, index=df.index, name="predicted_delta_log_X_next")

    def get_coefficients(self) -> pd.Series:
        if self.coefficients_ is None:
            return pd.Series(dtype=float)
        return self.coefficients_.copy()

    def get_feature_names(self) -> list[str]:
        if self.coefficients_ is not None:
            return list(self.coefficients_.index)
        return list(self.features)
