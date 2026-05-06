from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CollapseThresholds:
    """Thresholds for detecting bad transitions without preventing them."""

    output_loss_fraction: float = 0.2


def detect_bad_transition(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    thresholds: CollapseThresholds | None = None,
) -> dict[str, object]:
    thresholds = thresholds or CollapseThresholds()
    previous_output = float(previous["X"].sum())
    current_output = float(current["X"].sum())
    previous_emissions = float((previous["X"] * previous["EI"]).sum())
    current_emissions = float((current["X"] * current["EI"]).sum())
    output_loss_fraction = (
        (previous_output - current_output) / previous_output if previous_output > 0 else 0.0
    )
    emissions_fell = current_emissions < previous_emissions
    output_collapsed = output_loss_fraction > thresholds.output_loss_fraction
    return {
        "bad_transition": bool(emissions_fell and output_collapsed),
        "emissions_fell": bool(emissions_fell),
        "output_collapsed": bool(output_collapsed),
        "output_loss_fraction": float(output_loss_fraction),
        "previous_output": previous_output,
        "current_output": current_output,
        "previous_emissions": previous_emissions,
        "current_emissions": current_emissions,
    }
