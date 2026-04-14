from src.config import DEFAULT_CONFIG
from src.paths import (
    INTERIM_DATA_DIR,
    OUTPUTS_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    ensure_project_dirs,
)


def test_standard_paths_are_rooted_in_project() -> None:
    assert RAW_DATA_DIR.parent == PROJECT_ROOT / "data"
    assert INTERIM_DATA_DIR.parent == PROJECT_ROOT / "data"
    assert PROCESSED_DATA_DIR.parent == PROJECT_ROOT / "data"
    assert OUTPUTS_DIR.parent == PROJECT_ROOT


def test_default_config_uses_standard_paths() -> None:
    assert DEFAULT_CONFIG.raw_data_dir == RAW_DATA_DIR
    assert DEFAULT_CONFIG.interim_data_dir == INTERIM_DATA_DIR
    assert DEFAULT_CONFIG.processed_data_dir == PROCESSED_DATA_DIR
    assert DEFAULT_CONFIG.outputs_dir == OUTPUTS_DIR


def test_ensure_project_dirs_is_idempotent() -> None:
    ensure_project_dirs()
    assert RAW_DATA_DIR.exists()
    assert INTERIM_DATA_DIR.exists()
    assert PROCESSED_DATA_DIR.exists()
    assert OUTPUTS_DIR.exists()
