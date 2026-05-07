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


def compute_substitution_adjusted_input_availability(
    nodes: pd.DataFrame,
    input_availability: pd.Series,
    demand: pd.Series,
    planned_output: pd.Series,
    sigma: float,
    node_col: str = "country_sector",
    sector_col: str = "Sector",
    surplus_col: str = "available_surplus",
) -> pd.Series:
    """Apply simplified node-level supplier substitution before production.

    This is a transparent placeholder for later edge-based IO substitution. It
    allows same-sector surplus to relax input constraints before final
    production realization, without introducing prices, green supplier
    preference, or anti-collapse output guardrails.
    """

    if not 0.0 <= sigma <= 1.0:
        raise ValueError("sigma must be in [0, 1].")
    if node_col not in nodes.columns:
        raise ValueError(f"Nodes missing stable node key column: {node_col}")
    if sector_col not in nodes.columns:
        raise ValueError(f"Nodes missing sector column: {sector_col}")

    input_availability = input_availability.reindex(nodes.index).astype(float)
    demand = demand.reindex(nodes.index).astype(float)
    planned_output = planned_output.reindex(nodes.index).astype(float)
    adjusted_input_availability = input_availability.copy()

    if sigma == 0.0:
        return pd.Series(
            adjusted_input_availability,
            index=nodes.index,
            name="adjusted_input_availability",
        )

    intended_output = pd.concat([planned_output, demand], axis=1).min(axis=1, skipna=False)
    shortfall = (intended_output - input_availability).clip(lower=0)

    if surplus_col in nodes.columns:
        remaining_surplus = pd.Series(
            nodes[surplus_col].to_numpy(dtype=float),
            index=nodes.index,
            name=surplus_col,
        ).clip(lower=0)
    else:
        remaining_surplus = (input_availability - intended_output).clip(lower=0)

    for constrained_index in nodes.index[shortfall > 0]:
        target_sector = nodes.at[constrained_index, sector_col]
        target_node = nodes.at[constrained_index, node_col]
        eligible_mask = (
            (nodes[sector_col] == target_sector)
            & (nodes[node_col] != target_node)
            & (remaining_surplus > 0)
        )
        eligible_surplus = remaining_surplus[eligible_mask]
        total_eligible_surplus = float(eligible_surplus.sum())
        if total_eligible_surplus <= 0:
            continue

        allowed_substitution = min(float(shortfall.at[constrained_index]) * sigma, total_eligible_surplus)
        if allowed_substitution <= 0:
            continue

        allocation = eligible_surplus / total_eligible_surplus * allowed_substitution
        remaining_surplus.loc[allocation.index] = remaining_surplus.loc[allocation.index] - allocation
        adjusted_input_availability.at[constrained_index] = (
            adjusted_input_availability.at[constrained_index] + allowed_substitution
        )

    return pd.Series(
        adjusted_input_availability,
        index=nodes.index,
        name="adjusted_input_availability",
    )
