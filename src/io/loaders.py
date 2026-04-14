from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import polars as pl

from src.config import DEFAULT_CONFIG, ProjectConfig


def resolve_data_path(*parts: str | Path, config: ProjectConfig = DEFAULT_CONFIG) -> Path:
    """Resolve a path under the raw data directory."""
    return config.raw_data_dir.joinpath(*parts)


def load_parquet(path: str | Path, columns: Iterable[str] | None = None) -> pl.DataFrame:
    """Load a parquet file with Polars for fast analytical workflows."""
    return pl.read_parquet(path, columns=list(columns) if columns is not None else None)


def load_csv(path: str | Path, **kwargs: object) -> pl.DataFrame:
    """Load a CSV-like text file with Polars."""
    return pl.read_csv(path, **kwargs)


def load_pandas_parquet(path: str | Path, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Load parquet into pandas when compatibility matters more than speed."""
    return pd.read_parquet(path, columns=list(columns) if columns is not None else None)
