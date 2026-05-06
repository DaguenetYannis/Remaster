from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.abm_v3.config import CapabilityConfig

LOGGER = logging.getLogger(__name__)


def compute_smoothed_green_capability_export_share(
    df: pd.DataFrame,
    lambda_value: float,
    sector_col: str = "Sector",
    year_col: str = "Year",
) -> pd.Series:
    """Compute smoothed Atlas green capability export share.

    The smoothing mass ``lambda_value`` is interpreted in export-value units:

    ``(green_exports + lambda * sector_year_average_share) / (exports + lambda)``.

    Missing Atlas values are preserved as missing because missing capability
    evidence and zero capability are conceptually different in the ABM v3 data
    contract.
    """

    required = [
        sector_col,
        year_col,
        "green_active_good_export_value",
        "active_good_export_value",
        "green_capability_export_share",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing capability columns: {missing}")

    result = df.copy()
    sector_year_average = result.groupby([sector_col, year_col])[
        "green_capability_export_share"
    ].transform("mean")

    numerator = (
        result["green_active_good_export_value"].astype(float)
        + lambda_value * sector_year_average.astype(float)
    )
    denominator = result["active_good_export_value"].astype(float) + lambda_value

    smoothed = numerator / denominator
    smoothed = smoothed.replace([np.inf, -np.inf], np.nan)
    return smoothed


def add_capability_features(
    df: pd.DataFrame,
    config: CapabilityConfig,
) -> pd.DataFrame:
    """Add ABM-ready capability columns without overwriting Atlas originals."""

    result = df.copy()
    result[config.green_capability_column] = compute_smoothed_green_capability_export_share(
        result,
        lambda_value=config.smoothing_lambda,
    )
    if "capability_export_weighted_pci" not in result.columns:
        LOGGER.warning("Missing capability_export_weighted_pci; general_complexity set to NaN.")
        result[config.general_complexity_column] = np.nan
    else:
        result[config.general_complexity_column] = result[
            "capability_export_weighted_pci"
        ].astype(float)
    return result
