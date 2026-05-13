from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PARQUET_DATA_DIR = DATA_DIR / "parquet"
METRICS_DATA_DIR = DATA_DIR / "metrics"
ATLAS_DATA_DIR = DATA_DIR / "atlas"
FINAL_DATA_DIR = DATA_DIR / "final"
ABM_DATA_DIR = DATA_DIR / "abm"
ABM_V3_DATA_DIR = DATA_DIR / "abm_v3"
ABM_V4_DATA_DIR = DATA_DIR / "abm_v4"

# Kept for older config/tests that still use broad data-stage names.
INTERIM_DATA_DIR = DATA_DIR / "parquet"
PROCESSED_DATA_DIR = DATA_DIR / "final"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"
TMP_DIR = PROJECT_ROOT / "tmp"


def ensure_project_dirs() -> None:
    """Create the standard local project directories if they are missing."""
    for directory_path in (
        RAW_DATA_DIR,
        PARQUET_DATA_DIR,
        METRICS_DATA_DIR,
        ATLAS_DATA_DIR,
        FINAL_DATA_DIR,
        ABM_DATA_DIR,
        ABM_V3_DATA_DIR,
        OUTPUTS_DIR,
        LOGS_DIR,
        TMP_DIR,
    ):
        directory_path.mkdir(parents=True, exist_ok=True)
