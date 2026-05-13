from __future__ import annotations

import math
from dataclasses import dataclass

from src.abm_v4.config import EmissionsConfig


@dataclass(frozen=True)
class EmissionsDecomposition:
    """Aggregate emissions decomposition for one transition."""

    emissions_total_t: float
    emissions_total_t_plus_1: float
    delta_emissions_total: float
    emissions_intensity_effect: float
    production_scale_effect: float
    interaction_effect: float
    residual: float
    aggregate_output_loss_pct: float
    bad_transition_flag: bool


def emissions_identity(output: float, emissions_intensity: float) -> float:
    """Return emissions from output and emissions intensity."""
    return output * emissions_intensity


def next_emissions_intensity(
    emissions_intensity: float,
    green_capability: float,
    network_green_exposure: float,
    general_capability: float,
    brown_centrality: float,
    config: EmissionsConfig,
) -> float:
    """Update emissions intensity using the ABM v4 reduction equation."""
    reduction_rate = (
        config.beta_0
        + config.beta_log_ei * math.log(emissions_intensity)
        + config.beta_green_capability * green_capability
        + config.beta_network_green_exposure * network_green_exposure
        + config.beta_general_capability * general_capability
        - config.beta_brown_centrality * brown_centrality
    )
    return max(config.ei_min, emissions_intensity * math.exp(-reduction_rate))
