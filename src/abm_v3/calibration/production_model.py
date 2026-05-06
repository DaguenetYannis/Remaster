from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ProductionPlanningModel:
    """Interpretable ridge-style production planning scaffold."""

    features: list[str] = field(default_factory=list)
    target: str = "delta_log_X_next"
    l2_penalty: float = 1.0
    coefficients_: pd.Series | None = None
    intercept_: float = 0.0

    def fit(self, df: pd.DataFrame) -> "ProductionPlanningModel":
        required = [*self.features, self.target]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"Missing production calibration columns: {missing}")
        train = df[required].dropna()
        if train.empty:
            raise ValueError("No complete rows available for production planning fit.")
        x = train[self.features].to_numpy(dtype=float)
        y = train[self.target].to_numpy(dtype=float)
        x_design = np.column_stack([np.ones(len(x)), x])
        penalty = np.eye(x_design.shape[1]) * self.l2_penalty
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(x_design.T @ x_design + penalty, x_design.T @ y)
        self.intercept_ = float(beta[0])
        self.coefficients_ = pd.Series(beta[1:], index=self.features, dtype=float)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.coefficients_ is None:
            return pd.Series(0.0, index=df.index, name="predicted_delta_log_X_next")
        missing = [column for column in self.features if column not in df.columns]
        if missing:
            raise ValueError(f"Missing production prediction columns: {missing}")
        values = df[self.features].fillna(0.0).to_numpy(dtype=float)
        prediction = self.intercept_ + values @ self.coefficients_.to_numpy(dtype=float)
        return pd.Series(prediction, index=df.index, name="predicted_delta_log_X_next")

    def get_coefficients(self) -> pd.Series:
        if self.coefficients_ is None:
            return pd.Series(dtype=float)
        return self.coefficients_.copy()
