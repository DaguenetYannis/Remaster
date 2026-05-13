from pathlib import Path
from uuid import uuid4

from src.abm_v4.paths import ABMV4Paths


def toy_root() -> Path:
    return Path("tmp") / "abm_v4_tests" / uuid4().hex


def test_abm_v4_paths_are_project_relative() -> None:
    paths = ABMV4Paths(project_root=Path("toy_project"))

    assert paths.data_abm_v4 == Path("toy_project") / "data" / "abm_v4"
    assert paths.inputs == paths.data_abm_v4 / "inputs"
    assert paths.interim == paths.data_abm_v4 / "interim"
    assert paths.diagnostics == paths.data_abm_v4 / "diagnostics"
    assert paths.simulations == paths.data_abm_v4 / "simulations"
    assert paths.scenarios == paths.data_abm_v4 / "scenarios"
    assert paths.validation == paths.data_abm_v4 / "validation"


def test_abm_v4_paths_do_not_create_outputs_from_properties() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)

    _ = paths.inputs
    _ = paths.state_panel_path(1995, 2016)

    assert not paths.data_abm_v4.exists()


def test_ensure_output_directories_creates_only_v4_directories() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)

    paths.ensure_output_directories()

    for output_directory in paths.output_directories():
        assert output_directory.exists()

    assert not (root / "data" / "abm").exists()
    assert not (root / "data" / "abm_v3").exists()


def test_state_source_priority_prefers_abm_v3_before_final_and_legacy() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)

    candidates = paths.state_source_candidates(1995, 2016)

    assert candidates[0] == (
        root
        / "data"
        / "abm_v3"
        / "inputs"
        / "abm_v3_historical_panel_1995_2016_transpose_row_fd_without_inventory.parquet"
    )
    assert paths.final_state_candidates[0] in candidates
    assert candidates[-1] == root / "data" / "abm" / "agents_panel.parquet"
