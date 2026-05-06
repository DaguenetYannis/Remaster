from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class EmissionsIntensityModel:
    """Estimate EI dynamics from previous EI, capability, and network green-ness.

    The scaffold models changes in log(EI), then exponentiates for next EI.
    This preserves non-negativity through the transform instead of clipping
    negative predictions after the fact.
    """

    features: list[str] = field(default_factory=lambda: ["log_EI_lag1", "green_capability", "g_network"])
    target: str = "delta_log_EI_next"
    l2_penalty: float = 1.0
    coefficients_: pd.Series | None = None
    intercept_: float = 0.0

    def fit(self, df: pd.DataFrame) -> "EmissionsIntensityModel":
        required = [*self.features, self.target]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"Missing emissions calibration columns: {missing}")
        train = df[required].dropna()
        if train.empty:
            raise ValueError("No complete rows available for emissions intensity fit.")
        x = train[self.features].to_numpy(dtype=float)
        y = train[self.target].to_numpy(dtype=float)
        x_design = np.column_stack([np.ones(len(x)), x])
        penalty = np.eye(x_design.shape[1]) * self.l2_penalty
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(x_design.T @ x_design + penalty, x_design.T @ y)
        self.intercept_ = float(beta[0])
        self.coefficients_ = pd.Series(beta[1:], index=self.features, dtype=float)
        return self

    def predict_delta_ei(self, df: pd.DataFrame) -> pd.Series:
        if self.coefficients_ is None:
            return pd.Series(0.0, index=df.index, name="predicted_delta_log_EI")
        missing = [column for column in self.features if column not in df.columns]
        if missing:
            raise ValueError(f"Missing emissions prediction columns: {missing}")
        x = df[self.features].fillna(0.0).to_numpy(dtype=float)
        predicted = self.intercept_ + x @ self.coefficients_.to_numpy(dtype=float)
        return pd.Series(predicted, index=df.index, name="predicted_delta_log_EI")

    def predict_next_ei(self, df: pd.DataFrame, current_ei_col: str = "EI") -> pd.Series:
        if current_ei_col not in df.columns:
            raise ValueError(f"Missing current EI column: {current_ei_col}")
        current_ei = pd.to_numeric(df[current_ei_col], errors="coerce")
        delta_log_ei = self.predict_delta_ei(df)
        next_ei = current_ei * np.exp(delta_log_ei)
        return pd.Series(next_ei, index=df.index, name="predicted_next_EI")
