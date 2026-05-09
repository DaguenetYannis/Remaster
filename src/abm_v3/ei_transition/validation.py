from __future__ import annotations

import pandas as pd


def validate_transition_split(
    panel: pd.DataFrame,
    train_end_year: int,
    validation_start_year: int,
    validation_end_year: int,
) -> None:
    """Validate the historical estimation split before model fitting."""
    if validation_start_year <= train_end_year:
        raise ValueError("validation_start_year must be greater than train_end_year.")
    if validation_end_year < validation_start_year:
        raise ValueError("validation_end_year must be greater than or equal to validation_start_year.")
    included = panel.loc[panel["sample_included"]].copy()
    train_rows = included.loc[included["Year"] <= train_end_year]
    validation_rows = included.loc[
        (included["Year"] >= validation_start_year)
        & (included["Year"] <= validation_end_year)
    ]
    if train_rows.empty:
        raise ValueError(f"No included EI transition training rows found through {train_end_year}.")
    if validation_rows.empty:
        raise ValueError(
            "No included EI transition validation rows found for "
            f"{validation_start_year}-{validation_end_year}."
        )
