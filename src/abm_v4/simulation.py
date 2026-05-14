from __future__ import annotations

from dataclasses import dataclass

from src.abm_v4.config import ABMV4Config
from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.state import StateSourceDiagnostic, discover_state_source
from src.abm_v4.validation import (
    OneStepBaseValidationResult,
    build_one_step_base_validation_report,
    missing_one_step_component_paths,
    write_one_step_base_validation_outputs,
)


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


@dataclass(frozen=True)
class OneStepBaseRunResult:
    """Result of the one-step base orchestration check."""

    validation: OneStepBaseValidationResult
    reused_existing_outputs: bool
    raw_t_rebuild_skipped: bool


def run_one_step_base_orchestration(
    paths: ABMV4Paths,
    config: ABMV4Config,
    *,
    reuse_existing: bool = True,
    force_rebuild_raw_t_edges: bool = False,
    write_outputs: bool = False,
) -> OneStepBaseRunResult:
    """Run the one-step base integration check from existing component outputs.

    This orchestration layer intentionally does not launch expensive component
    builders by default. It validates the outputs from Phases 2-7B and writes
    only consolidated validation artifacts when requested.
    """
    missing = missing_one_step_component_paths(paths, config)
    if missing:
        missing_lines = "\n".join(
            f"- {name}: {path}" for name, path in sorted(missing.items())
        )
        raw_t_hint = ""
        if (
            "raw_t_supplier_edges" in missing
            or "raw_t_supplier_edge_report" in missing
        ) and not force_rebuild_raw_t_edges:
            raw_t_hint = (
                "\nRaw T edges are expensive and were not rebuilt. Run "
                "`python scripts/run_abm_v4_base.py --build-raw-t-supplier-edges "
                "--create-output-dirs` first, or pass --force-rebuild-raw-t-edges "
                "after implementing/accepting that rebuild."
            )
        raise FileNotFoundError(
            "Cannot run one-step ABM v4 base validation because required component "
            f"outputs are missing:\n{missing_lines}{raw_t_hint}"
        )

    validation = build_one_step_base_validation_report(paths, config)
    if write_outputs:
        write_one_step_base_validation_outputs(paths, validation)
    return OneStepBaseRunResult(
        validation=validation,
        reused_existing_outputs=reuse_existing,
        raw_t_rebuild_skipped=not force_rebuild_raw_t_edges,
    )
