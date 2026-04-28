from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.paths import INTERIM_DATA_DIR, OUTPUTS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"

ATLAS_PROCESSED = DATA_DIR / "atlas" / "processed"
EORA_PROCESSED = DATA_DIR / "eora" / "processed"

FINAL_DIR = DATA_DIR / "final"

@dataclass(frozen=True)
class ProjectConfig:
    """Small, explicit config surface for the research project."""

    project_name: str = "eora26-research"
    years: tuple[int, ...] = ()
    raw_data_dir: Path = field(default=RAW_DATA_DIR)
    interim_data_dir: Path = field(default=INTERIM_DATA_DIR)
    processed_data_dir: Path = field(default=PROCESSED_DATA_DIR)
    outputs_dir: Path = field(default=OUTPUTS_DIR)


DEFAULT_CONFIG = ProjectConfig()
