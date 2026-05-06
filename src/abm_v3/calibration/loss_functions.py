from __future__ import annotations

import numpy as np
import pandas as pd


def _relative_rmse(simulated: pd.Series, observed: pd.Series, epsilon: float = 1e-12) -> float:
    aligned = pd.concat([simulated, observed], axis=1).dropna()
    if aligned.empty:
        return float("nan")
    error = aligned.iloc[:, 0].to_numpy(dtype=float) - aligned.iloc[:, 1].to_numpy(dtype=float)
    scale = np.maximum(np.abs(aligned.iloc[:, 1].to_numpy(dtype=float)), epsilon)
    return float(np.sqrt(np.mean((error / scale) ** 2)))


def output_loss(simulated: pd.Series, observed: pd.Series) -> float:
    return _relative_rmse(simulated, observed)


def emissions_loss(simulated: pd.Series, observed: pd.Series) -> float:
    return _relative_rmse(simulated, observed)


def ei_loss(simulated: pd.Series, observed: pd.Series) -> float:
    return _relative_rmse(simulated, observed)


def distribution_loss(simulated: pd.Series, observed: pd.Series) -> float:
    sim_share = simulated / simulated.sum() if simulated.sum() else simulated * np.nan
    obs_share = observed / observed.sum() if observed.sum() else observed * np.nan
    return _relative_rmse(sim_share, obs_share)


def composite_loss(losses: dict[str, float], weights: dict[str, float]) -> float:
    total_weight = sum(weights.get(name, 0.0) for name in losses)
    if total_weight == 0:
        return float("nan")
    weighted = sum(losses[name] * weights.get(name, 0.0) for name in losses)
    return float(weighted / total_weight)
