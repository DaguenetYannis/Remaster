import polars as pl

from src.abm_v4.schemas import (
    ECOSYSTEM_MAPPING_REQUIRED_COLUMNS,
    STATE_REQUIRED_COLUMNS,
    VALID_ECOSYSTEM_SOURCES,
    validate_ecosystem_mapping_schema,
    validate_state_panel_schema,
)


def test_state_schema_accepts_required_columns() -> None:
    dataframe = pl.DataFrame({column: [] for column in STATE_REQUIRED_COLUMNS})

    result = validate_state_panel_schema(dataframe)

    assert result.is_valid
    assert result.missing_columns == ()


def test_state_schema_reports_missing_columns_without_guessing() -> None:
    dataframe = pl.DataFrame({"country_sector": ["FRA|Agriculture"]})

    result = validate_state_panel_schema(dataframe)

    assert not result.is_valid
    assert "Year" in result.missing_columns
    assert "EI" in result.missing_columns


def test_ecosystem_mapping_schema_accepts_required_columns() -> None:
    dataframe = pl.DataFrame({column: [] for column in ECOSYSTEM_MAPPING_REQUIRED_COLUMNS})

    result = validate_ecosystem_mapping_schema(dataframe)

    assert result.is_valid


def test_ecosystem_source_vocabulary_is_explicit() -> None:
    assert VALID_ECOSYSTEM_SOURCES == (
        "atlas_cluster_aggregated",
        "hs92_dominant_cluster",
        "eora_sector_manual_mapping",
        "fallback_unknown",
    )
