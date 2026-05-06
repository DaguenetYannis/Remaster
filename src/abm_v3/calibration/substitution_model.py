from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SubstitutionFrictionModel:
    """Store or calibrate scalar substitution friction in [0, 1]."""

    sigma: float = 0.25

    def fit(self, df: pd.DataFrame | None = None) -> "SubstitutionFrictionModel":
        if not 0.0 <= self.sigma <= 1.0:
            raise ValueError("Substitution friction sigma must be in [0, 1].")
        return self

    def get_sigma(self) -> float:
        return float(self.sigma)
