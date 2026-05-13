from __future__ import annotations

from src.abm_v4.config import ABMV4Config
from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.simulation import inspect_base_model_readiness


def main() -> None:
    """Inspect ABM v4 base-model readiness without generating simulation data."""
    config = ABMV4Config()
    paths = ABMV4Paths()
    paths.ensure_output_directories()
    report = inspect_base_model_readiness(paths=paths, config=config)

    print(report.state_source.message)
    print(f"Can run base model: {report.can_run_base_model}")


if __name__ == "__main__":
    main()
