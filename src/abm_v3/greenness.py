from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.abm_v3.config import GreennessConfig

LOGGER = logging.getLogger(__name__)


def negative_log_greenness(EI: pd.Series, epsilon: float) -> pd.Series:
    """Transform emissions intensity into empirical green-ness variation."""

    numeric_ei = pd.to_numeric(EI, errors="coerce")
    return -np.log(numeric_ei + epsilon)


def minmax_rescale(series: pd.Series) -> pd.Series:
    """Rescale a series to [0, 1] while preserving missing values."""

    numeric = pd.to_numeric(series, errors="coerce")
    minimum = numeric.min(skipna=True)
    maximum = numeric.max(skipna=True)
    if pd.isna(minimum) or pd.isna(maximum):
        return pd.Series(np.nan, index=series.index)
    if maximum == minimum:
        return pd.Series(0.5, index=series.index)
    return (numeric - minimum) / (maximum - minimum)


def add_greenness_features(
    df: pd.DataFrame,
    config: GreennessConfig,
    ei_column: str = "EI",
) -> pd.DataFrame:
    """Add negative-log and bounded green-ness columns for ABM dynamics."""

    if ei_column not in df.columns:
        raise ValueError(f"Missing emissions intensity column: {ei_column}")
    result = df.copy()
    if config.raw_output_column in result.columns:
        LOGGER.warning("Overwriting explicit raw green-ness output column: %s", config.raw_output_column)
    if config.scaled_output_column in result.columns:
        LOGGER.warning("Overwriting explicit scaled green-ness output column: %s", config.scaled_output_column)
    result[config.raw_output_column] = negative_log_greenness(result[ei_column], config.epsilon)
    result[config.scaled_output_column] = minmax_rescale(result[config.raw_output_column])
    return result
