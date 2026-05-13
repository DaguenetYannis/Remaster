from pathlib import Path
from uuid import uuid4

from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.state import discover_state_source


def test_state_source_reports_missing_inputs() -> None:
    root = Path("tmp") / "abm_v4_tests" / uuid4().hex
    report = discover_state_source(ABMV4Paths(project_root=root), 1995, 2016)

    assert not report.has_source
    assert report.selected_source is None
    assert "No valid" in report.message
