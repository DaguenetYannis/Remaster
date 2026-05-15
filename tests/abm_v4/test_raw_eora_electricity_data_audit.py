from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import polars as pl
import pytest

from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.validation import RawEoraElectricityDataAudit


def _toy_paths() -> ABMV4Paths:
    return ABMV4Paths(project_root=Path("tmp") / "abm_v4_raw_eora_electricity_tests" / uuid4().hex)


def _source_with_columns() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "Country": "CHN",
                "Sector": "Electricity, Gas and Water",
                "year": 2000,
                "X_observed": 100.0,
                "emissions_observed": 20.0,
                "EI": 0.2,
            },
            {
                "Country": "CHN",
                "Sector": "Electricity, Gas and Water",
                "year": 2001,
                "X_observed": 120.0,
                "emissions_observed": 60.0,
                "EI": 0.5,
            },
            {
                "Country": "USA",
                "Sector": "Electricity, Gas and Water",
                "year": 2001,
                "X_observed": 200.0,
                "emissions_observed": 50.0,
                "EI": 0.25,
            },
        ]
    )


def _source_with_label(scale: float = 1.0) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "country_sector": "CHN | CHN | Commodities | Electricity, Gas and Water",
                "Year": 2000,
                "X": 100.0 * scale,
                "emissions_observed": 20.0 * scale,
            },
            {
                "country_sector": "CHN | CHN | Commodities | Electricity, Gas and Water",
                "Year": 2001,
                "X": 120.0 * scale,
                "emissions_observed": 60.0 * scale,
            },
        ]
    )


def test_candidate_source_inventory_detects_usable_schema() -> None:
    paths = _toy_paths()
    source = paths.data_final / "toy.parquet"
    source.parent.mkdir(parents=True, exist_ok=True)
    _source_with_columns().write_parquet(source)
    audit = RawEoraElectricityDataAudit(paths, start_year=2000, end_year=2001)

    row = audit.inspect_source_schema(source)

    assert row["usable_for_china_electricity"]
    assert row["has_output_candidate"]
    assert row["has_emissions_candidate"]


def test_china_electricity_records_extract_from_country_sector_columns() -> None:
    paths = _toy_paths()
    source = paths.data_final / "toy.parquet"
    source.parent.mkdir(parents=True, exist_ok=True)
    _source_with_columns().write_parquet(source)
    audit = RawEoraElectricityDataAudit(paths, start_year=2000, end_year=2001)

    out = audit.identify_china_electricity_records(source)

    assert out.height == 2
    assert out["country_value"].unique().to_list() == ["CHN"]


def test_china_electricity_records_extract_from_country_sector_string() -> None:
    paths = _toy_paths()
    source = paths.data_final / "toy_label.parquet"
    source.parent.mkdir(parents=True, exist_ok=True)
    _source_with_label().write_parquet(source)
    audit = RawEoraElectricityDataAudit(paths, start_year=2000, end_year=2001)

    out = audit.identify_china_electricity_records(source)

    assert out.height == 2
    assert "Electricity" in out["sector_value"].item(0)


def test_ei_recomputed_equals_emissions_divided_by_output() -> None:
    paths = _toy_paths()
    source = paths.data_final / "toy.parquet"
    source.parent.mkdir(parents=True, exist_ok=True)
    _source_with_columns().write_parquet(source)
    audit = RawEoraElectricityDataAudit(paths, start_year=2000, end_year=2001)

    row = audit.identify_china_electricity_records(source).filter(pl.col("year") == 2001).to_dicts()[0]

    assert row["EI_recomputed"] == pytest.approx(0.5)


def test_cross_source_comparison_detects_near_match() -> None:
    audit = RawEoraElectricityDataAudit(_toy_paths(), start_year=2000, end_year=2001)
    series = pl.concat(
        [
            audit._derive_series_fields(_source_with_label().rename({"Year": "year", "X": "X_candidate"}).with_columns(
                pl.lit("a.parquet").alias("source_path"),
                pl.lit("CHN").alias("country_value"),
                pl.lit("Electricity, Gas and Water").alias("sector_value"),
                pl.lit(None).cast(pl.Float64).alias("EI_candidate"),
            )),
            audit._derive_series_fields(_source_with_label().rename({"Year": "year", "X": "X_candidate"}).with_columns(
                pl.lit("b.parquet").alias("source_path"),
                pl.lit("CHN").alias("country_value"),
                pl.lit("Electricity, Gas and Water").alias("sector_value"),
                pl.lit(None).cast(pl.Float64).alias("EI_candidate"),
            )),
        ],
        how="diagonal_relaxed",
    )

    comparison = audit.compare_sources(series)

    assert "exact_or_near_match" in comparison["match_status"].to_list()


def test_cross_source_comparison_detects_possible_scale_factor() -> None:
    audit = RawEoraElectricityDataAudit(_toy_paths(), start_year=2000, end_year=2001)
    a = audit._derive_series_fields(_source_with_label().rename({"Year": "year", "X": "X_candidate"}).with_columns(
        pl.lit("a.parquet").alias("source_path"),
        pl.lit("CHN").alias("country_value"),
        pl.lit("Electricity, Gas and Water").alias("sector_value"),
        pl.lit(None).cast(pl.Float64).alias("EI_candidate"),
    ))
    b = audit._derive_series_fields(_source_with_label(scale=1000.0).rename({"Year": "year", "X": "X_candidate"}).with_columns(
        pl.lit("b.parquet").alias("source_path"),
        pl.lit("CHN").alias("country_value"),
        pl.lit("Electricity, Gas and Water").alias("sector_value"),
        pl.lit(None).cast(pl.Float64).alias("EI_candidate"),
    ))

    comparison = audit.compare_sources(pl.concat([a, b], how="diagonal_relaxed"))

    assert "scale_difference_possible" in comparison["match_status"].to_list()


def test_mapping_audit_detects_duplicate_records_and_missing_years() -> None:
    audit = RawEoraElectricityDataAudit(_toy_paths(), start_year=2000, end_year=2002)
    base = _source_with_label().filter(pl.col("Year") == 2000)
    end = _source_with_label().filter(pl.col("Year") == 2001).with_columns(pl.lit(2002).alias("Year"))
    series = pl.concat([audit._derive_series_fields(pl.concat([base, end], how="vertical_relaxed").rename({"Year": "year", "X": "X_candidate"}).with_columns(
        pl.lit("a.parquet").alias("source_path"),
        pl.lit("CHN").alias("country_value"),
        pl.lit("Electricity, Gas and Water").alias("sector_value"),
        pl.lit(None).cast(pl.Float64).alias("EI_candidate"),
    ))] * 2, how="diagonal_relaxed")

    mapping = audit.audit_mapping_consistency(series)

    assert "duplicate_records" in mapping["mapping_status"].to_list()
    assert "missing_year" in mapping["mapping_status"].to_list()


def test_breakpoint_audit_flags_large_log_changes() -> None:
    paths = _toy_paths()
    audit = RawEoraElectricityDataAudit(paths, start_year=2000, end_year=2001)
    source = audit._derive_series_fields(_source_with_columns().with_columns(
        pl.lit("a.parquet").alias("source_path"),
        pl.col("Country").alias("country_value"),
        pl.col("Sector").alias("sector_value"),
        (pl.col("Country") + pl.lit(" | ") + pl.col("Sector")).alias("country_sector"),
        pl.col("X_observed").alias("X_candidate"),
        pl.col("emissions_observed").alias("emissions_candidate"),
        pl.col("EI").alias("EI_candidate"),
    ))

    out = audit.detect_breakpoints_and_jumps(source)

    assert out.filter(pl.col("jump_flag")).height > 0


def test_recommendation_selects_repair_when_sources_mismatch() -> None:
    audit = RawEoraElectricityDataAudit(_toy_paths())
    comparison = pl.DataFrame({"match_status": ["mismatch", "mismatch", "exact_or_near_match"]})
    scaling = pl.DataFrame({"scale_flag": [False]})
    mapping = pl.DataFrame({"mapping_status": ["consistent"]})
    breakpoint = pl.DataFrame({"country_sector": ["CHN"], "jump_flag": [False]})
    major = pl.DataFrame({"country_sector": ["CHN"], "emissions_rank_within_electricity": [1.0]})

    rec = audit.build_recommendation(comparison, scaling, mapping, breakpoint, major)

    assert rec["recommended_next_action"].item(0) == "repair_mapping_or_scaling"


def test_recommendation_selects_sector_specific_when_consistent_but_jumps_are_severe() -> None:
    audit = RawEoraElectricityDataAudit(_toy_paths())
    comparison = pl.DataFrame({"match_status": ["exact_or_near_match", "exact_or_near_match"]})
    scaling = pl.DataFrame({"scale_flag": [False]})
    mapping = pl.DataFrame({"mapping_status": ["consistent"]})
    breakpoint = pl.DataFrame(
        {
            "country_sector": ["CHN", "CHN", "CHN", "USA", "RUS", "IND"],
            "jump_flag": [True, True, True, True, True, True],
        }
    )
    major = pl.DataFrame({"country_sector": ["CHN"], "emissions_rank_within_electricity": [1.0]})

    rec = audit.build_recommendation(comparison, scaling, mapping, breakpoint, major)

    assert rec["recommended_next_action"].item(0) == "treat_electricity_as_sector_specific_transition_case"


def test_outputs_are_written_only_with_explicit_output_flag() -> None:
    paths = _toy_paths()
    source = paths.data_final / "toy.parquet"
    source.parent.mkdir(parents=True, exist_ok=True)
    _source_with_columns().write_parquet(source)
    audit = RawEoraElectricityDataAudit(paths, start_year=2000, end_year=2001)
    result = audit.run()

    assert not paths.raw_eora_electricity_data_audit_report_path.exists()
    audit.write_outputs(result)
    assert paths.raw_eora_electricity_source_inventory_path.exists()
    assert paths.raw_eora_electricity_data_audit_report_path.exists()


def test_missing_usable_sources_fail_clearly() -> None:
    with pytest.raises(FileNotFoundError, match="No usable Eora-derived China electricity sources"):
        RawEoraElectricityDataAudit(_toy_paths()).run()
