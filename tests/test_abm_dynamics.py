import numpy as np
import pytest

from src.abm.dynamics import (
    update_production,
    update_inventory,
    update_emissions_intensity,
    update_local_greeness,
    update_capability_readiness,
    update_network_links,
    compute_network_green_exposure,
    compute_total_emissions,
)


def test_update_production_respects_all_constraints():
    demand = np.array([100.0, 80.0])
    capacity = np.array([90.0, 120.0])
    inputs = np.array([95.0, 60.0])
    inventory = np.array([10.0, 5.0])

    result = update_production(demand, capacity, inputs, inventory)

    np.testing.assert_array_equal(result, np.array([90.0, 65.0]))


def test_update_inventory_is_non_negative():
    previous_inventory = np.array([10.0, 5.0])
    inputs = np.array([100.0, 50.0])
    production = np.array([80.0, 70.0])

    result = update_inventory(
        previous_inventory=previous_inventory,
        available_inputs=inputs,
        production=production,
        inventory_replenishment_rate=0.5,
    )

    assert np.all(result >= 0)


def test_update_emissions_intensity_decreases_with_capability_and_network_exposure():
    ei = np.array([1.0, 2.0])
    capability = np.array([0.5, 0.5])
    network = np.array([0.5, 0.5])

    result = update_emissions_intensity(
        emissions_intensity=ei,
        capability_readiness=capability,
        network_green_exposure=network,
        alpha=0.1,
        beta=0.1,
    )

    assert result[0] < ei[0]
    assert result[1] < ei[1]


def test_update_local_greeness_is_bounded():
    ei = np.array([0.0, 1.0, 10.0])

    result = update_local_greeness(ei)

    assert np.all(result > 0)
    assert np.all(result <= 1)


def test_update_capability_readiness_raises_if_all_lambdas_zero():
    with pytest.raises(ValueError):
        update_capability_readiness(
            local_greeness=np.array([0.5]),
            network_green_exposure=np.array([0.5]),
            lambda_local=0.0,
            lambda_network=0.0,
            lambda_ecosystem=0.0,
        )


def test_update_network_links_moves_toward_feasible_links():
    current = np.array([[1.0, 0.0], [0.0, 1.0]])
    feasible = np.array([[0.0, 1.0], [1.0, 0.0]])

    result = update_network_links(current, feasible, eta=0.5)

    expected = np.array([[0.5, 0.5], [0.5, 0.5]])
    np.testing.assert_array_equal(result, expected)


def test_compute_network_green_exposure_row_normalized():
    weights = np.array([
        [0.0, 2.0],
        [1.0, 1.0],
    ])
    greeness = np.array([0.2, 0.8])

    result = compute_network_green_exposure(weights, greeness, axis=1)

    np.testing.assert_allclose(result, np.array([0.8, 0.5]))


def test_compute_total_emissions():
    production = np.array([100.0, 200.0])
    ei = np.array([0.5, 0.25])

    result = compute_total_emissions(production, ei)

    assert result == 100.0