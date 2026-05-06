from __future__ import annotations

import pandas as pd


def select_eligible_suppliers(
    supplier_pool: pd.DataFrame,
    target_sector: str,
    sector_col: str = "Sector",
) -> pd.DataFrame:
    """Select same-sector suppliers eligible under base substitution rules."""

    if sector_col not in supplier_pool.columns:
        raise ValueError(f"Supplier pool missing sector column: {sector_col}")
    return supplier_pool[supplier_pool[sector_col] == target_sector].copy()


def apply_supplier_substitution(
    shortfall: float,
    eligible_suppliers: pd.DataFrame,
    sigma: float,
    capacity_col: str = "available_surplus",
) -> pd.DataFrame:
    """Allocate substitutable supply under scalar friction sigma in [0, 1]."""

    if not 0.0 <= sigma <= 1.0:
        raise ValueError("sigma must be in [0, 1].")
    result = eligible_suppliers.copy()
    result["substituted_amount"] = 0.0
    if sigma == 0.0 or shortfall <= 0 or result.empty:
        return result
    if capacity_col not in result.columns:
        raise ValueError(f"Eligible suppliers missing capacity column: {capacity_col}")
    available = result[capacity_col].clip(lower=0).astype(float)
    total_available = float(available.sum())
    allowed_substitution = min(float(shortfall) * sigma, total_available)
    if total_available > 0:
        result["substituted_amount"] = available / total_available * allowed_substitution
    return result
