from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.abm_v3.paths import ABMV3Paths

LOGGER = logging.getLogger(__name__)


@dataclass
class ABMV3OutputWriter:
    """Centralize explicit ABM v3 output writes under data/abm_v3."""

    paths: ABMV3Paths

    def output_dir(self, name: str) -> Path:
        return self.paths.abm_v3_output_root / name

    def write_dataframe(self, df: pd.DataFrame, subdir: str, filename: str) -> Path:
        target_dir = self.output_dir(subdir)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / filename
        LOGGER.info("Writing %s rows to %s", len(df), path)
        df.to_parquet(path, index=False)
        return path
