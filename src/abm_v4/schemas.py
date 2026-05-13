from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import polars as pl


STATE_KEY_COLUMNS: tuple[str, ...] = ("country_sector", "Year")

STATE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "country_sector",
    "Year",
    "X_observed",
    "EI",
    "g_local",
    "brown_centrality",
    "general_capability",
    "green_capability",
    "ecosystem_id",
    "ecosystem_label",
    "ecosystem_source",
)

STATE_OPTIONAL_COLUMNS: tuple[str, ...] = (
    "Country",
    "Country_detail",
    "Category",
    "Sector",
    "emissions_observed",
    "g_in_network",
    "network_green_exposure",
    "g_out_network",
    "centrality",
    "pagerank",
    "log_x_observed",
    "log_EI",
    "g_local_v4",
)

ECOSYSTEM_MAPPING_REQUIRED_COLUMNS: tuple[str, ...] = (
    "Sector",
    "ecosystem_id",
    "ecosystem_label",
    "ecosystem_source",
)

ECOSYSTEM_ADJACENCY_REQUIRED_COLUMNS: tuple[str, ...] = (
    "source_ecosystem_id",
    "target_ecosystem_id",
    "ecosystem_proximity",
)

VALID_ECOSYSTEM_SOURCES: tuple[str, ...] = (
    "atlas_cluster_aggregated",
    "hs92_dominant_cluster",
    "eora_sector_manual_mapping",
    "fallback_unknown",
)


@dataclass(frozen=True)
class SchemaValidationResult:
    """Inspectable result for schema checks that should not fail silently."""

    schema_name: str
    required_columns: tuple[str, ...]
    present_columns: tuple[str, ...]
    missing_columns: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.missing_columns

    def to_dict(self) -> dict[str, object]:
        """Return a compact dictionary for diagnostics or tests."""
        return {
            "schema_name": self.schema_name,
            "required_columns": list(self.required_columns),
            "present_columns": list(self.present_columns),
            "missing_columns": list(self.missing_columns),
            "is_valid": self.is_valid,
        }


def missing_columns(
    dataframe: pl.DataFrame,
    required_columns: Iterable[str],
) -> tuple[str, ...]:
    """Return required columns absent from a Polars dataframe."""
    present_columns = set(dataframe.columns)
    return tuple(column for column in required_columns if column not in present_columns)


def validate_required_columns(
    dataframe: pl.DataFrame,
    required_columns: Iterable[str],
    schema_name: str,
) -> SchemaValidationResult:
    """Validate required columns without mutating or guessing missing fields."""
    required_tuple = tuple(required_columns)
    return SchemaValidationResult(
        schema_name=schema_name,
        required_columns=required_tuple,
        present_columns=tuple(dataframe.columns),
        missing_columns=missing_columns(dataframe, required_tuple),
    )


def validate_state_panel_schema(dataframe: pl.DataFrame) -> SchemaValidationResult:
    """Validate the ABM v4 state panel structural contract."""
    return validate_required_columns(
        dataframe=dataframe,
        required_columns=STATE_REQUIRED_COLUMNS,
        schema_name="abm_v4_state_panel",
    )


def validate_ecosystem_mapping_schema(dataframe: pl.DataFrame) -> SchemaValidationResult:
    """Validate the ABM v4 ecosystem mapping structural contract."""
    return validate_required_columns(
        dataframe=dataframe,
        required_columns=ECOSYSTEM_MAPPING_REQUIRED_COLUMNS,
        schema_name="abm_v4_ecosystem_mapping",
    )


def validate_ecosystem_adjacency_schema(dataframe: pl.DataFrame) -> SchemaValidationResult:
    """Validate the ABM v4 ecosystem adjacency structural contract."""
    return validate_required_columns(
        dataframe=dataframe,
        required_columns=ECOSYSTEM_ADJACENCY_REQUIRED_COLUMNS,
        schema_name="abm_v4_ecosystem_adjacency",
    )
