from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import polars as pl
import pytest

from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.validation import ElectricityDataAudit


def _toy_paths() -> ABMV4Paths:
    return ABMV4Paths(project_root=Path("tmp") / "abm_v4_electricity_audit_tests" / uuid4().hex)


def _toy_observed() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "country_sector": "CHN electricity",
                "year": 2000,
                "Country": "People's Republic of China",
                "Sector": "Electricity, Gas and Water",
                "X_observed": 100.0,
                "EI_observed": 2.0,
                "emissions_observed": 200.0,
            },
            {
                "country_sector": "CHN electricity",
                "year": 2001,
                "Country": "People's Republic of China",
                "Sector": "Electricity, Gas and Water",
                "X_observed": 120.0,
                "EI_observed": 4.0,
                "emissions_observed": 480.0,
            },
            {
                "country_sector": "USA utilities",
                "year": 2000,
                "Country": "USA",
                "Sector": "power utilities",
                "X_observed": 100.0,
                "EI_observed": 1.0,
                "emissions_observed": 100.0,
            },
            {
                "country_sector": "USA utilities",
                "year": 2001,
                "Country": "USA",
                "Sector": "power utilities",
                "X_observed": 110.0,
                "EI_observed": 1.1,
                "emissions_observed": 121.0,
            },
            {
                "country_sector": "FRA services",
                "year": 2001,
                "Country": "FRA",
                "Sector": "Financial Intermediation",
                "X_observed": 50.0,
                "EI_observed": 0.5,
                "emissions_observed": 25.0,
            },
        ]
    )


def _toy_model(suffix: str) -> pl.DataFrame:
    more_aggressive = suffix == "frontier"
    return pl.DataFrame(
        [
            {
                "country_sector": "CHN electricity",
                "year": 2000,
                "EI_sim": 2.0,
                "emissions_sim": 200.0,
                "rEI_used": 0.05 if more_aggressive else 0.02,
                "ei_gap": 0.4,
                "readiness": 0.2,
                "emissions_observed": 200.0,
            },
            {
                "country_sector": "CHN electricity",
                "year": 2001,
                "EI_sim": 4.4 if more_aggressive else 4.1,
                "emissions_sim": 540.0 if more_aggressive else 500.0,
                "rEI_used": 0.08 if more_aggressive else 0.03,
                "ei_gap": 0.5,
                "readiness": 0.2,
                "emissions_observed": 480.0,
            },
            {
                "country_sector": "USA utilities",
                "year": 2000,
                "EI_sim": 1.0,
                "emissions_sim": 100.0,
                "rEI_used": 0.02,
                "ei_gap": 0.1,
                "readiness": 0.8,
                "emissions_observed": 100.0,
            },
            {
                "country_sector": "USA utilities",
                "year": 2001,
                "EI_sim": 1.1,
                "emissions_sim": 125.0,
                "rEI_used": 0.02,
                "ei_gap": 0.1,
                "readiness": 0.8,
                "emissions_observed": 121.0,
            },
        ]
    )


def _write_required_inputs(paths: ABMV4Paths) -> None:
    paths.inputs.mkdir(parents=True, exist_ok=True)
    paths.simulations.mkdir(parents=True, exist_ok=True)
    _toy_observed().rename({"year": "Year", "EI_observed": "EI"}).write_parquet(paths.state_panel_path(1995, 2016))
    _toy_model("readiness").write_parquet(paths.base_multiyear_state_panel_path)
    _toy_model("frontier").write_parquet(paths.base_multiyear_state_panel_historical_frontier_gap_path)


def test_electricity_like_sectors_are_detected_case_insensitively() -> None:
    audit = ElectricityDataAudit(_toy_paths())
    electricity = audit.identify_electricity_nodes(_toy_observed())

    assert set(electricity["country_sector"].to_list()) == {"CHN electricity", "USA utilities"}


def test_china_like_country_labels_are_detected() -> None:
    audit = ElectricityDataAudit(_toy_paths())
    china = audit.identify_china_electricity_nodes(_toy_observed())

    assert china["country_sector"].to_list() == ["CHN electricity", "CHN electricity"]


def test_observed_rei_and_pct_changes_are_computed_correctly() -> None:
    audit = ElectricityDataAudit(_toy_paths())
    series = audit.audit_all_electricity_observed_series(_toy_observed())
    row = series.filter((pl.col("country_sector") == "CHN electricity") & (pl.col("year") == 2001)).to_dicts()[0]

    assert row["observed_rEI"] == pytest.approx(0.69314718056)
    assert row["pct_change_X"] == pytest.approx(0.2)
    assert row["pct_change_EI"] == pytest.approx(1.0)
    assert row["pct_change_emissions"] == pytest.approx(1.4)


def test_large_ei_jumps_are_flagged_correctly() -> None:
    audit = ElectricityDataAudit(_toy_paths())
    observed = audit.audit_observed_series(_toy_observed())
    model = audit.audit_model_series(
        observed,
        audit._prepare_model_panel(_toy_model("readiness"), "readiness"),
        audit._prepare_model_panel(_toy_model("frontier"), "frontier_gap"),
    )
    flags = audit.build_data_quality_flags(observed, model)

    flag = flags.filter(
        (pl.col("country_sector") == "CHN electricity")
        & (pl.col("year") == 2001)
        & (pl.col("flag_name") == "EI_jump_large_flag")
    )
    assert flag["flag_value"].item()


def test_sector_percentile_ranks_are_computed_correctly() -> None:
    audit = ElectricityDataAudit(_toy_paths())
    series = audit.audit_all_electricity_observed_series(_toy_observed())
    row = series.filter((pl.col("country_sector") == "CHN electricity") & (pl.col("year") == 2001)).to_dicts()[0]

    assert row["EI_percentile_within_sector_year"] == pytest.approx(1.0)
    assert row["gap_to_sector_year_p50"] == pytest.approx(0.0)


def test_cross_country_comparison_includes_all_electricity_nodes() -> None:
    audit = ElectricityDataAudit(_toy_paths())
    observed = audit.audit_all_electricity_observed_series(_toy_observed())
    model = audit.audit_model_series(
        observed,
        audit._prepare_model_panel(_toy_model("readiness"), "readiness"),
        audit._prepare_model_panel(_toy_model("frontier"), "frontier_gap"),
    )
    comparison = audit.compare_electricity_nodes(model)

    assert set(comparison["country_sector"].to_list()) == {"CHN electricity", "USA utilities"}


def test_recommendation_selects_raw_data_inspection_when_anomaly_flags_are_strong() -> None:
    paths = _toy_paths()
    audit = ElectricityDataAudit(paths)
    observed = audit.audit_observed_series(_toy_observed())
    model = audit.audit_model_series(
        observed,
        audit._prepare_model_panel(_toy_model("readiness"), "readiness"),
        audit._prepare_model_panel(_toy_model("frontier"), "frontier_gap"),
    )
    flags = audit.build_data_quality_flags(observed, model)
    comparison = audit.compare_electricity_nodes(model)
    rec = audit.build_audit_recommendation(flags, comparison, model)

    assert rec["recommended_next_action"].to_list()[0] == "inspect_raw_eora_electricity_data"


def test_recommendation_selects_dampened_hybrid_when_data_are_clean_and_readiness_dampens() -> None:
    audit = ElectricityDataAudit(_toy_paths())
    model = pl.DataFrame(
        [
            {
                "country_sector": "CHN electricity",
                "year": 2001,
                "Country": "CHN",
                "Sector": "Electricity, Gas and Water",
                "observed_rEI": 0.01,
                "simulated_rEI_readiness": 0.02,
                "simulated_rEI_frontier_gap": 0.08,
                "rEI_error_readiness": 0.01,
                "rEI_error_frontier_gap": 0.07,
                "dampening_amount": 0.06,
                "frontier_gap_worsens_emissions_error": True,
            },
            {
                "country_sector": "CHN electricity",
                "year": 2002,
                "Country": "CHN",
                "Sector": "Electricity, Gas and Water",
                "observed_rEI": 0.00,
                "simulated_rEI_readiness": 0.01,
                "simulated_rEI_frontier_gap": 0.05,
                "rEI_error_readiness": 0.01,
                "rEI_error_frontier_gap": 0.05,
                "dampening_amount": 0.04,
                "frontier_gap_worsens_emissions_error": True,
            },
        ]
    )
    flags = pl.DataFrame(
        {
            "country_sector": ["CHN electricity"],
            "year": [2001],
            "flag_name": ["EI_jump_large_flag"],
            "flag_value": [False],
            "metric_value": [0.01],
            "threshold": [0.5],
            "interpretation": ["clean"],
        }
    )
    comparison = pl.DataFrame(
        {
            "country_sector": ["CHN electricity"],
            "Country": ["CHN"],
            "Sector": ["Electricity, Gas and Water"],
            "mean_frontier_gap": [0.1],
            "emissions_share": [0.1],
        }
    )
    rec = audit.build_audit_recommendation(flags, comparison, model)

    assert rec["recommended_next_action"].to_list()[0] == "test_dampened_frontier_gap_hybrid"


def test_outputs_are_written_only_with_explicit_output_flag() -> None:
    paths = _toy_paths()
    _write_required_inputs(paths)
    audit = ElectricityDataAudit(paths)
    result = audit.run()

    assert not paths.electricity_data_audit_report_path.exists()
    audit.write_outputs(result)
    assert paths.electricity_node_inventory_path.exists()
    assert paths.electricity_data_audit_report_path.exists()


def test_missing_required_inputs_fail_with_actionable_message() -> None:
    with pytest.raises(FileNotFoundError, match="build-state"):
        ElectricityDataAudit(_toy_paths()).load_observed_state()
