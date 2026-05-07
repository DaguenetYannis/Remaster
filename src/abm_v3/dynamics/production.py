from __future__ import annotations

import numpy as np
import pandas as pd


def plan_production(
    current_nodes: pd.DataFrame,
    planning_model: object | None = None,
    current_output_col: str = "X",
) -> pd.Series:
    """Plan production from a calibrated model or hold current output constant."""

    if planning_model is None:
        return pd.Series(current_nodes[current_output_col].to_numpy(dtype=float), index=current_nodes.index, name="planned_output")
    delta_log = planning_model.predict(current_nodes)
    planned = current_nodes[current_output_col].astype(float) * np.exp(delta_log)
    return pd.Series(planned, index=current_nodes.index, name="planned_output")


def realize_production(
    planned_output: pd.Series,
    demand: pd.Series,
    input_availability: pd.Series,
) -> pd.Series:
    """Realize output after substitution subject to quantity constraints.

    This is the final feasibility check. It is not an anti-collapse guardrail:
    missing or binding constraints remain visible in the output instead of
    being reset upward.
    """

    combined = pd.concat([planned_output, demand, input_availability], axis=1).astype(float)
    realized = combined.min(axis=1, skipna=False)
    return pd.Series(realized, index=planned_output.index, name="realized_output")


def realize_production_with_soft_input_constraint(
    planned_output: pd.Series,
    demand: pd.Series,
    capacity: pd.Series | None,
    adjusted_input_availability: pd.Series,
    effective_input_intensity: pd.Series,
    input_rigidity: float = 0.5,
) -> pd.DataFrame:
    """Realize output through a soft monetary input-feasibility constraint.

    ``adjusted_input_availability`` and ``effective_input_intensity`` are both
    monetary quantities. A node with ``M=40`` and ``a=M/X=0.4`` can therefore
    support ``X=100`` before stress is applied; inputs are not treated as
    one-to-one physical output units. Missing input feasibility leaves desired
    output unchanged and is flagged explicitly, because missing data should not
    mechanically collapse a node.
    """

    if input_rigidity <= 0:
        raise ValueError("input_rigidity must be positive.")

    index = planned_output.index
    planned = planned_output.reindex(index).astype(float)
    demand_aligned = demand.reindex(index).astype(float)
    if capacity is None:
        capacity_aligned = pd.Series(np.inf, index=index, name="K")
    else:
        capacity_aligned = capacity.reindex(index).astype(float)
    adjusted_inputs = adjusted_input_availability.reindex(index).astype(float)
    input_intensity = effective_input_intensity.reindex(index).astype(float)

    desired_output = pd.concat([planned, demand_aligned, capacity_aligned], axis=1).min(axis=1, skipna=False)
    desired_output = desired_output.where(desired_output > 0, 0.0)
    input_feasible_output = adjusted_inputs / input_intensity
    input_feasible_output = input_feasible_output.replace([np.inf, -np.inf], np.nan)
    input_feasibility_missing = input_intensity.isna()

    positive_desired = desired_output > 0
    input_stress_ratio = pd.Series(np.nan, index=index, dtype=float)
    input_stress_ratio.loc[positive_desired] = (
        input_feasible_output.loc[positive_desired] / desired_output.loc[positive_desired]
    )
    input_stress_factor = pd.Series(1.0, index=index, dtype=float)
    measured_mask = positive_desired & ~input_feasibility_missing & input_stress_ratio.notna()
    bounded_ratio = input_stress_ratio.loc[measured_mask].clip(lower=0.0)
    input_stress_factor.loc[measured_mask] = np.minimum(1.0, bounded_ratio ** input_rigidity)
    input_stress_factor.loc[~positive_desired] = 1.0
    input_stress_factor.loc[input_feasibility_missing] = 1.0

    realized_output = desired_output * input_stress_factor
    input_constraint_penalty = desired_output - realized_output
    input_constraint_binding = input_stress_factor < 1.0
    return pd.DataFrame(
        {
            "desired_output": desired_output,
            "input_feasible_output": input_feasible_output,
            "input_stress_ratio": input_stress_ratio,
            "input_stress_factor": input_stress_factor,
            "input_constraint_penalty": input_constraint_penalty,
            "input_constraint_binding": input_constraint_binding,
            "input_feasibility_missing": input_feasibility_missing,
            "realized_output": realized_output,
        },
        index=index,
    )
