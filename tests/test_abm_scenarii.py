import pytest

from src.abm_v2.scenarii import (
    BASELINE,
    get_scenario,
    list_scenarios,
    override_scenario,
)


def test_get_scenario_returns_baseline():
    scenario = get_scenario("baseline")

    assert scenario.name == "baseline"


def test_unknown_scenario_raises_error():
    with pytest.raises(ValueError):
        get_scenario("does_not_exist")


def test_list_scenarios_contains_baseline():
    scenarios = list_scenarios()

    assert "baseline" in scenarios


def test_override_scenario_does_not_mutate_original():
    modified = override_scenario(BASELINE, kappa=1.5)

    assert modified.kappa == 1.5
    assert BASELINE.kappa != 1.5