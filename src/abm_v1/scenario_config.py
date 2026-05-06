from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScenarioConfig:
    """
    Configuration object for ABM scenario simulations.

    The parameters are intentionally mechanism-facing:
    each one corresponds to a theoretical lever in the ABM.
    """

    scenario_name: str = "baseline"

    agents_path: Path = Path("data/abm/agents_panel.parquet")
    transitions_path: Path = Path(
        "data/abm/diagnostics/transitions_with_clean_targets.parquet"
    )
    output_dir: Path = Path("data/abm/scenarios")

    n_steps: int = 10
    random_seed: int = 42

    transition_probability_scale: float = 0.08
    network_diffusion_boost: float = 1.0
    capability_policy_boost: float = 0.0
    brown_core_intervention: float = 0.0

    def output_panel_path(self) -> Path:
        return self.output_dir / f"{self.scenario_name}_simulation_panel.parquet"

    def output_summary_path(self) -> Path:
        return self.output_dir / f"{self.scenario_name}_summary_panel.parquet"


BASELINE_SCENARIO = ScenarioConfig(
    scenario_name="baseline",
    transition_probability_scale=0.08,
    network_diffusion_boost=1.0,
    capability_policy_boost=0.0,
    brown_core_intervention=0.0,
)


NETWORK_DIFFUSION_SCENARIO = ScenarioConfig(
    scenario_name="network_diffusion",
    transition_probability_scale=0.08,
    network_diffusion_boost=1.5,
    capability_policy_boost=0.0,
    brown_core_intervention=0.0,
)


CAPABILITY_POLICY_SCENARIO = ScenarioConfig(
    scenario_name="capability_policy",
    transition_probability_scale=0.08,
    network_diffusion_boost=1.0,
    capability_policy_boost=0.5,
    brown_core_intervention=0.0,
)


BROWN_CORE_INTERVENTION_SCENARIO = ScenarioConfig(
    scenario_name="brown_core_intervention",
    transition_probability_scale=0.08,
    network_diffusion_boost=1.0,
    capability_policy_boost=0.0,
    brown_core_intervention=0.5,
)


COMBINED_TRANSITION_SCENARIO = ScenarioConfig(
    scenario_name="combined_transition",
    transition_probability_scale=0.10,
    network_diffusion_boost=1.5,
    capability_policy_boost=0.5,
    brown_core_intervention=0.5,
)


DEFAULT_SCENARIOS: dict[str, ScenarioConfig] = {
    "baseline": BASELINE_SCENARIO,
    "network_diffusion": NETWORK_DIFFUSION_SCENARIO,
    "capability_policy": CAPABILITY_POLICY_SCENARIO,
    "brown_core_intervention": BROWN_CORE_INTERVENTION_SCENARIO,
    "combined_transition": COMBINED_TRANSITION_SCENARIO,
}