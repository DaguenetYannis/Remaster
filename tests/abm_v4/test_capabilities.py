from src.abm_v4.capabilities import capability_increment, green_capability_increment, sigmoid
from src.abm_v4.config import CapabilityConfig


def test_sigmoid_is_bounded() -> None:
    assert 0.0 < sigmoid(-10.0) < 0.5
    assert 0.5 < sigmoid(10.0) < 1.0


def test_capability_increments_do_not_exceed_remaining_stock() -> None:
    config = CapabilityConfig(delta_cap_param=0.1, delta_gcap_param=0.1)

    assert capability_increment(0.9, 1.0, config) <= 0.1
    assert green_capability_increment(0.9, 1.0, config) <= 0.1
