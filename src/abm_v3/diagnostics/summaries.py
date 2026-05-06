from __future__ import annotations

import pandas as pd


def _summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    required_values = ["X", "emissions"]
    missing = [column for column in [*group_cols, *required_values] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing summary columns: {missing}")
    return (
        df.groupby(group_cols, as_index=False)
        .agg(total_output=("X", "sum"), total_emissions=("emissions", "sum"), mean_ei=("EI", "mean"))
        .reset_index(drop=True)
    )


def global_summary(df: pd.DataFrame) -> pd.DataFrame:
    return _summary(df.assign(global_system="global"), ["global_system", "Year"])


def country_summary(df: pd.DataFrame) -> pd.DataFrame:
    return _summary(df, ["Country", "Year"])


def sector_summary(df: pd.DataFrame) -> pd.DataFrame:
    return _summary(df, ["Sector", "Year"])


def country_sector_summary(df: pd.DataFrame) -> pd.DataFrame:
    return _summary(df, ["country_sector", "Year"])
