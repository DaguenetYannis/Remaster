from pathlib import Path

import pytest

from src.abm_v5 import ABMV5Paths


EXPECTED_PATH_KEYS = {
    "project_root",
    "data_abm_v5",
    "inputs",
    "accounting",
    "phase_space",
    "regimes",
    "capabilities",
    "supplier_network",
    "energy_inertia",
    "policy",
    "simulation",
    "validation",
    "diagnostics",
    "reports",
    "plots",
    "plots_diagnostics",
    "plots_research",
    "plots_portfolio",
    "logs",
    "tmp",
    "docs",
}


def test_abmv5_paths_from_project_root_accepts_string_and_path(tmp_path: Path) -> None:
    from_string = ABMV5Paths.from_project_root(str(tmp_path))
    from_path = ABMV5Paths.from_project_root(tmp_path)

    assert from_string.project_root == tmp_path
    assert from_path.project_root == tmp_path
    assert from_string.data_abm_v5 == tmp_path / "data" / "abm_v5"


def test_abmv5_paths_validate_project_root_requires_pyproject(tmp_path: Path) -> None:
    paths = ABMV5Paths.from_project_root(tmp_path)

    with pytest.raises(FileNotFoundError):
        paths.validate_project_root()

    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'toy'\n")

    paths.validate_project_root()


def test_abmv5_paths_as_dict_contains_expected_keys(tmp_path: Path) -> None:
    paths = ABMV5Paths.from_project_root(tmp_path)

    path_dict = paths.as_dict()

    assert set(path_dict) == EXPECTED_PATH_KEYS
    assert all(isinstance(path_value, Path) for path_value in path_dict.values())


def test_abmv5_paths_ensure_directories_creates_expected_directories(
    tmp_path: Path,
) -> None:
    paths = ABMV5Paths.from_project_root(tmp_path)

    assert not paths.data_abm_v5.exists()
    assert not paths.plots.exists()
    assert not paths.logs.exists()

    paths.ensure_directories()

    for path_name, directory_path in paths.as_dict().items():
        if path_name == "project_root":
            continue
        assert directory_path.is_dir()

    assert not (tmp_path / "data" / "abm_v4").exists()


def test_abmv5_init_exports_paths() -> None:
    import src.abm_v5 as abm_v5

    assert "ABMV5Paths" in abm_v5.__all__
    assert abm_v5.ABMV5Paths is ABMV5Paths
