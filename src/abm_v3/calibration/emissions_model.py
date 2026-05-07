from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class EmissionsIntensityModel:
    """Estimate EI dynamics under explicit ABM v3 modes.

    ``economic_only`` applies historical sector-average EI trends and does not
    use green capability or network green-ness. ``historical_ei`` learns
    observed EI dynamics from non-green controls. ``green_transition`` tests
    the theoretical mechanism linking EI reduction to green capability,
    network green exposure, and general complexity.
    """

    mode: str = "green_transition"
    features: list[str] | None = None
    target: str = "delta_log_EI_next"
    l2_penalty: float = 1.0
    coefficients_: pd.Series | None = None
    intercept_: float = 0.0
    sector_trends_: pd.Series | None = None

    def __post_init__(self) -> None:
        allowed_modes = {"economic_only", "historical_ei", "green_transition"}
        if self.mode not in allowed_modes:
            raise ValueError(f"Unsupported EI mode: {self.mode}")
        if self.features is None:
            if self.mode == "historical_ei":
                self.features = ["log_EI_lag1", "sector_EI_growth", "country_EI_growth"]
            elif self.mode == "green_transition":
                self.features = [
                    "log_EI_lag1",
                    "green_capability",
                    "g_in",
                    "g_out",
                    "g_network",
                    "general_complexity",
                ]
            else:
                self.features = []

    def _available_features(self, df: pd.DataFrame) -> list[str]:
        return [feature for feature in self.features or [] if feature in df.columns]

    def fit(self, df: pd.DataFrame) -> "EmissionsIntensityModel":
        if self.target not in df.columns:
            raise ValueError(f"Missing emissions calibration target: {self.target}")
        if self.mode == "economic_only":
            if "Sector" not in df.columns:
                raise ValueError("economic_only EI mode requires Sector column.")
            train = df[["Sector", self.target]].dropna()
            self.sector_trends_ = train.groupby("Sector")[self.target].mean()
            return self

        selected_features = self._available_features(df)
        required = [*selected_features, self.target]
        missing = [column for column in [self.target] if column not in df.columns]
        if missing:
            raise ValueError(f"Missing emissions calibration columns: {missing}")
        if not selected_features:
            raise ValueError(f"No available emissions features for mode {self.mode}.")
        train = df[required].dropna()
        if train.empty:
            raise ValueError("No complete rows available for emissions intensity fit.")
        x = train[selected_features].to_numpy(dtype=float)
        y = train[self.target].to_numpy(dtype=float)
        x_design = np.column_stack([np.ones(len(x)), x])
        penalty = np.eye(x_design.shape[1]) * self.l2_penalty
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(x_design.T @ x_design + penalty, x_design.T @ y)
        self.intercept_ = float(beta[0])
        self.coefficients_ = pd.Series(beta[1:], index=selected_features, dtype=float)
        return self

    def predict_delta_ei(self, df: pd.DataFrame) -> pd.Series:
        if self.mode == "economic_only":
            if self.sector_trends_ is None:
                return pd.Series(0.0, index=df.index, name="predicted_delta_log_EI")
            if "Sector" not in df.columns:
                raise ValueError("economic_only EI prediction requires Sector column.")
            prediction = df["Sector"].map(self.sector_trends_).fillna(0.0)
            return pd.Series(prediction.to_numpy(dtype=float), index=df.index, name="predicted_delta_log_EI")
        if self.coefficients_ is None:
            return pd.Series(0.0, index=df.index, name="predicted_delta_log_EI")
        feature_names = self.get_feature_names()
        missing = [column for column in feature_names if column not in df.columns]
        if missing:
            raise ValueError(f"Missing emissions prediction columns: {missing}")
        x = df[feature_names].fillna(0.0).to_numpy(dtype=float)
        predicted = self.intercept_ + x @ self.coefficients_.to_numpy(dtype=float)
        return pd.Series(predicted, index=df.index, name="predicted_delta_log_EI")

    def predict_next_ei(self, df: pd.DataFrame, current_ei_col: str = "EI") -> pd.Series:
        if current_ei_col not in df.columns:
            raise ValueError(f"Missing current EI column: {current_ei_col}")
        current_ei = pd.to_numeric(df[current_ei_col], errors="coerce")
        delta_log_ei = self.predict_delta_ei(df)
        next_ei = current_ei * np.exp(delta_log_ei)
        return pd.Series(next_ei, index=df.index, name="predicted_next_EI")

    def get_coefficients(self) -> pd.Series:
        if self.mode == "economic_only":
            if self.sector_trends_ is None:
                return pd.Series(dtype=float)
            return self.sector_trends_.copy()
        if self.coefficients_ is None:
            return pd.Series(dtype=float)
        return self.coefficients_.copy()

    def get_feature_names(self) -> list[str]:
        if self.coefficients_ is not None:
            return list(self.coefficients_.index)
        return list(self.features or [])
