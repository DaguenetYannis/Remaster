"""
Pure ABM transition rules.

This module contains no file loading, no plotting, and no scenario definitions.
Every function should be deterministic and easy to test.
"""

from __future__ import annotations

import numpy as np


def clean_array(values: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    array = np.where(np.isfinite(array), array, fill_value)
    return array


def update_production(
    demand: np.ndarray,
    capacity: np.ndarray,
    available_inputs: np.ndarray,
    inventory: np.ndarray,
) -> np.ndarray:
    """
    Constraint-based production rule.

    X[t+1] = min(D[t], K[t], M[t] + I[t])
    """
    demand = clean_array(demand)
    capacity = clean_array(capacity)
    available_inputs = clean_array(available_inputs)
    inventory = clean_array(inventory)

    return np.minimum.reduce([
        demand,
        capacity,
        available_inputs + inventory,
    ])


def update_inventory(
    previous_inventory: np.ndarray,
    available_inputs: np.ndarray,
    production: np.ndarray,
    inventory_replenishment_rate: float,
) -> np.ndarray:
    """
    Updates inventory after production.

    The rule is intentionally simple:
    - production consumes part of available inputs;
    - unused inputs can replenish inventory gradually.
    """
    previous_inventory = clean_array(previous_inventory)
    available_inputs = clean_array(available_inputs)
    production = clean_array(production)

    unused_inputs = np.maximum(available_inputs - production, 0.0)

    next_inventory = (
        (1.0 - inventory_replenishment_rate) * previous_inventory
        + inventory_replenishment_rate * unused_inputs
    )

    return np.maximum(next_inventory, 0.0)


def update_emissions_intensity(
    emissions_intensity: np.ndarray,
    capability_readiness: np.ndarray,
    network_green_exposure: np.ndarray,
    alpha: float,
    beta: float,
    shock: np.ndarray | None = None,
) -> np.ndarray:
    """
    EI[t+1] = EI[t] - alpha * C[t] * EI[t] - beta * NG[t] * EI[t] + shock[t]
    """
    emissions_intensity = clean_array(emissions_intensity)
    capability_readiness = clean_array(capability_readiness)
    network_green_exposure = clean_array(network_green_exposure)

    reduction_rate = (
        alpha * capability_readiness
        + beta * network_green_exposure
    )

    next_ei = emissions_intensity * (1.0 - reduction_rate)

    if shock is not None:
        next_ei = next_ei + clean_array(shock)

    return np.maximum(next_ei, 0.0)


def update_local_greeness(
    emissions_intensity: np.ndarray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """
    Converts EI into bounded local green-ness.

    g = 1 / (1 + EI)
    """
    emissions_intensity = clean_array(emissions_intensity)
    return 1.0 / (1.0 + emissions_intensity + epsilon)


def update_capability_readiness(
    local_greeness: np.ndarray,
    network_green_exposure: np.ndarray,
    ecosystem_exposure: np.ndarray | None = None,
    lambda_local: float = 0.5,
    lambda_network: float = 0.5,
    lambda_ecosystem: float = 0.0,
) -> np.ndarray:
    """
    C[t] = lambda_1 * local + lambda_2 * network + lambda_3 * ecosystem
    """
    local_greeness = clean_array(local_greeness)
    network_green_exposure = clean_array(network_green_exposure)

    if ecosystem_exposure is None:
        ecosystem_exposure = np.zeros_like(local_greeness)
    else:
        ecosystem_exposure = clean_array(ecosystem_exposure)

    total_lambda = lambda_local + lambda_network + lambda_ecosystem

    if total_lambda <= 0:
        raise ValueError("At least one lambda weight must be positive.")

    capability = (
        lambda_local * local_greeness
        + lambda_network * network_green_exposure
        + lambda_ecosystem * ecosystem_exposure
    ) / total_lambda

    return np.clip(capability, 0.0, 1.0)


def update_network_links(
    current_links: np.ndarray,
    feasible_links: np.ndarray,
    eta: float,
) -> np.ndarray:
    """
    A[t+1] = A[t] + eta * (F[t] - A[t])
    """
    current_links = clean_array(current_links)
    feasible_links = clean_array(feasible_links)

    next_links = current_links + eta * (feasible_links - current_links)

    return np.maximum(next_links, 0.0)


def compute_network_green_exposure(
    weights: np.ndarray,
    local_greeness: np.ndarray,
    axis: int = 1,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """
    Computes weighted network green exposure.

    If axis=1, rows are normalized.
    If axis=0, columns are normalized.
    """
    weights = clean_array(weights)
    local_greeness = clean_array(local_greeness)

    denominator = weights.sum(axis=axis, keepdims=True)
    normalized_weights = weights / np.maximum(denominator, epsilon)

    if axis == 1:
        return normalized_weights @ local_greeness

    if axis == 0:
        return normalized_weights.T @ local_greeness

    raise ValueError("axis must be 0 or 1.")


def compute_total_emissions(
    production: np.ndarray,
    emissions_intensity: np.ndarray,
) -> float:
    production = clean_array(production)
    emissions_intensity = clean_array(emissions_intensity)

    return float(np.sum(production * emissions_intensity))


def compute_mean_emissions_intensity(
    production: np.ndarray,
    emissions_intensity: np.ndarray,
    epsilon: float = 1e-12,
) -> float:
    production = clean_array(production)
    emissions_intensity = clean_array(emissions_intensity)

    weights = np.maximum(production, 0.0)
    total_weight = weights.sum()

    if total_weight <= epsilon:
        return 0.0

    return float(np.average(emissions_intensity, weights=weights))


def compute_mean_greeness(
    production: np.ndarray,
    local_greeness: np.ndarray,
    epsilon: float = 1e-12,
) -> float:
    production = clean_array(production)
    local_greeness = clean_array(local_greeness)

    weights = np.maximum(production, 0.0)
    total_weight = weights.sum()

    if total_weight <= epsilon:
        return 0.0

    return float(np.average(local_greeness, weights=weights))