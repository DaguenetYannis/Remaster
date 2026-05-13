from __future__ import annotations

from dataclasses import dataclass

from src.abm_v4.config import ABMV4Config
from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.state import StateSourceDiagnostic, discover_state_source


@dataclass(frozen=True)
class SimulationReadinessReport:
    """Phase 1 readiness report for ABM v4 simulation inputs."""

    state_source: StateSourceDiagnostic
    can_run_base_model: bool


def inspect_base_model_readiness(
    paths: ABMV4Paths,
    config: ABMV4Config,
) -> SimulationReadinessReport:
    """Inspect whether local inputs exist for a future base-model run."""
    state_source = discover_state_source(paths, config.start_year, config.end_year)
    return SimulationReadinessReport(
        state_source=state_source,
        can_run_base_model=state_source.has_source,
    )
