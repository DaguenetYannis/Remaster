from __future__ import annotations

import numpy as np
import pandas as pd


def compute_network_green_exposure(
    et: pd.DataFrame,
    local_greenness: pd.Series,
) -> pd.DataFrame:
    """Compute incoming and outgoing green exposure from a fixed ET network."""

    et_values = et.to_numpy(dtype=float)
    g = local_greenness.reindex(et.index).to_numpy(dtype=float)
    row_sums = et_values.sum(axis=1, keepdims=True)
    col_sums = et_values.sum(axis=0, keepdims=True)
    w_out = np.divide(et_values, row_sums, out=np.zeros_like(et_values), where=row_sums > 0)
    w_in = np.divide(et_values, col_sums, out=np.zeros_like(et_values), where=col_sums > 0)
    return pd.DataFrame(
        {
            "country_sector": et.index.astype(str),
            "g_out": w_out @ g,
            "g_in": w_in.T @ g,
        }
    )
