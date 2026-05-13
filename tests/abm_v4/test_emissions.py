from src.abm_v4.config import EmissionsConfig
from src.abm_v4.emissions import emissions_identity, next_emissions_intensity


def test_emissions_identity_uses_output_times_ei() -> None:
    assert emissions_identity(100.0, 0.2) == 20.0


def test_next_emissions_intensity_respects_floor() -> None:
    config = EmissionsConfig(ei_min=0.01)

    updated = next_emissions_intensity(
        emissions_intensity=0.02,
        green_capability=1.0,
        network_green_exposure=1.0,
        general_capability=1.0,
        brown_centrality=0.0,
        config=config,
    )

    assert updated >= config.ei_min
