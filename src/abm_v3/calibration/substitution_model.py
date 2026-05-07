from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class SubstitutionFrictionModel:
    """Store and calibrate scalar substitution friction ``sigma``.

    Sigma is a production-resilience parameter. Grid search therefore minimizes
    output validation loss and penalizes excessive output collapse; it is not
    selected from emissions reduction alone.
    """

    sigma: float = 0.25
    sigma_grid: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
    collapse_penalty_weight: float = 1.0
    results_: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame | None = None) -> "SubstitutionFrictionModel":
        if not 0.0 <= self.sigma <= 1.0:
            raise ValueError("Substitution friction sigma must be in [0, 1].")
        return self

    def fit_grid(
        self,
        evaluator: Callable[[float], dict[str, float]],
    ) -> "SubstitutionFrictionModel":
        rows = []
        for sigma in self.sigma_grid:
            if not 0.0 <= sigma <= 1.0:
                raise ValueError(f"Sigma grid value outside [0, 1]: {sigma}")
            metrics = evaluator(float(sigma))
            output_validation_loss = float(metrics.get("output_validation_loss", np.nan))
            collapse_penalty = float(metrics.get("collapse_penalty", 0.0))
            score = output_validation_loss + self.collapse_penalty_weight * collapse_penalty
            rows.append(
                {
                    "sigma": float(sigma),
                    "output_validation_loss": output_validation_loss,
                    "collapse_penalty": collapse_penalty,
                    "score": score,
                    **{key: value for key, value in metrics.items() if key not in {"output_validation_loss", "collapse_penalty"}},
                }
            )
        self.results_ = pd.DataFrame(rows)
        valid_results = self.results_.dropna(subset=["score"])
        if valid_results.empty:
            raise ValueError("Sigma grid produced no valid validation scores.")
        best_row = valid_results.sort_values(["score", "sigma"]).iloc[0]
        self.sigma = float(best_row["sigma"])
        return self

    def get_sigma(self) -> float:
        return float(self.sigma)

    def get_results(self) -> pd.DataFrame:
        if self.results_ is None:
            return pd.DataFrame(columns=["sigma", "output_validation_loss", "collapse_penalty", "score"])
        return self.results_.copy()
