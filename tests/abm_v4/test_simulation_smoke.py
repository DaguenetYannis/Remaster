from pathlib import Path
from uuid import uuid4

from src.abm_v4.config import ABMV4Config
from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.simulation import inspect_base_model_readiness


def test_simulation_readiness_fails_clearly_without_sources() -> None:
    root = Path("tmp") / "abm_v4_tests" / uuid4().hex
    report = inspect_base_model_readiness(
        paths=ABMV4Paths(project_root=root),
        config=ABMV4Config(),
    )

    assert not report.can_run_base_model
    assert not report.state_source.has_source
