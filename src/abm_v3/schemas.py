from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

CORE_INDEX_COLUMNS: list[str] = ["country_sector", "Country", "Sector", "Year"]

STATE_COLUMNS: list[str] = [
    "X",
    "D",
    "EI",
    "g_local",
    "g_in",
    "g_out",
    "green_capability",
    "general_complexity",
    "planned_output",
    "realized_output",
    "emissions",
]

ATLAS_CAPABILITY_COLUMNS: list[str] = [
    "green_active_good_export_value",
    "active_good_export_value",
    "green_capability_export_share",
    "capability_export_weighted_pci",
    "active_good_count",
]

ATLAS_CAPABILITY_KEY_COLUMNS: list[str] = [
    "iso3Code",
    "countryName",
    "year",
    "eora26_sector",
]

MERGED_PANEL_KEY_COLUMNS: list[str] = ["Country", "Year", "Sector"]

EORA_MATRIX_LABEL_RULES: dict[str, tuple[str, str]] = {
    "T": ("labels_T", "labels_T"),
    "FD": ("labels_T", "labels_FD"),
    "Q": ("labels_Q", "labels_T"),
    "VA": ("labels_VA", "labels_T"),
    "QY": ("labels_Q", "labels_FD"),
}


def missing_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> list[str]:
    """Return required columns absent from ``df`` without raising."""

    return [column for column in required_columns if column not in df.columns]


def has_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> bool:
    """Return whether all required columns are present."""

    return not missing_columns(df, required_columns)


def describe_column_validation(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    name: str,
) -> dict[str, object]:
    """Build an inspectable column validation payload for flexible schemas."""

    required_list = list(required_columns)
    missing = missing_columns(df, required_list)
    return {
        "name": name,
        "passed": not missing,
        "required_columns": required_list,
        "missing_columns": missing,
        "available_columns": list(df.columns),
    }
