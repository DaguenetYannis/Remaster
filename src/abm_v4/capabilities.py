from __future__ import annotations

import math

from src.abm_v4.config import CapabilityConfig


def sigmoid(value: float) -> float:
    """Return a numerically stable logistic transform."""
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def capability_increment(capability: float, exposure: float, config: CapabilityConfig) -> float:
    """Compute general capability accumulation without depreciation."""
    remaining_capacity = config.cap_max - capability
    transition_signal = sigmoid(config.k_cap * (exposure - config.tau_cap))
    return config.delta_cap_param * remaining_capacity * transition_signal


def green_capability_increment(green_capability: float, exposure: float, config: CapabilityConfig) -> float:
    """Compute green capability accumulation without depreciation."""
    remaining_capacity = config.gcap_max - green_capability
    transition_signal = sigmoid(config.k_gcap * (exposure - config.tau_gcap))
    return config.delta_gcap_param * remaining_capacity * transition_signal
