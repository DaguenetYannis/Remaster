from pathlib import Path
from uuid import uuid4

import polars as pl
import pytest

from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.validation import HighEmissionsDampeningDiagnostics


def _toy_paths() -> ABMV4Paths:
    return ABMV4Paths(project_root=Path("tmp") / "abm_v4_high_emissions_tests" / uuid4().hex)


def _toy_panel() -> pl.DataFrame:
    rows = [
        {
            "country_sector": "CHN electricity",
            "year": 2001,
            "Country": "CHN",
            "Sector": "Electricity, Gas and Water",
            "X_observed": 1000.0,
            "EI_observed": 2.0,
            "emissions_observed": 2000.0,
            "observed_rEI": -0.02,
            "EI_sim_readiness": 2.1,
            "EI_sim_frontier_gap": 1.8,
            "simulated_rEI_readiness": 0.01,
            "simulated_rEI_frontier_gap": 0.05,
            "rEI_error_readiness": 0.03,
            "rEI_error_frontier_gap": 0.07,
            "rEI_abs_error_readiness": 0.03,
            "rEI_abs_error_frontier_gap": 0.07,
            "rEI_sign_correct_readiness": False,
            "rEI_sign_correct_frontier_gap": False,
            "rEI_wrong_sign_readiness": True,
            "rEI_wrong_sign_frontier_gap": True,
            "emissions_error_readiness": 100.0,
            "emissions_error_frontier_gap": 500.0,
            "contribution_to_aggregate_error_difference": 400.0,
            "emissions_decile": "d10",
            "output_decile": "d10",
            "frontier_gap_decile": "d10",
            "readiness_decile": "d1",
            "output_weight": 0.8,
            "emissions_weight": 0.8,
        },
        {
            "country_sector": "USA electricity",
            "year": 2001,
            "Country": "USA",
            "Sector": "Electricity utilities",
            "X_observed": 500.0,
            "EI_observed": 1.5,
            "emissions_observed": 750.0,
            "observed_rEI": 0.03,
            "EI_sim_readiness": 1.4,
            "EI_sim_frontier_gap": 1.35,
            "simulated_rEI_readiness": 0.02,
            "simulated_rEI_frontier_gap": 0.04,
            "rEI_error_readiness": -0.01,
            "rEI_error_frontier_gap": 0.01,
            "rEI_abs_error_readiness": 0.01,
            "rEI_abs_error_frontier_gap": 0.01,
            "rEI_sign_correct_readiness": True,
            "rEI_sign_correct_frontier_gap": True,
            "rEI_wrong_sign_readiness": False,
            "rEI_wrong_sign_frontier_gap": False,
            "emissions_error_readiness": 50.0,
            "emissions_error_frontier_gap": 100.0,
            "contribution_to_aggregate_error_difference": 50.0,
            "emissions_decile": "d9",
            "output_decile": "d9",
            "frontier_gap_decile": "d8",
            "readiness_decile": "d5",
            "output_weight": 0.15,
            "emissions_weight": 0.15,
        },
        {
            "country_sector": "FRA services",
            "year": 2001,
            "Country": "FRA",
            "Sector": "Financial Intermediation",
            "X_observed": 100.0,
            "EI_observed": 0.5,
            "emissions_observed": 50.0,
            "observed_rEI": 0.01,
            "EI_sim_readiness": 0.49,
            "EI_sim_frontier_gap": 0.49,
            "simulated_rEI_readiness": 0.01,
            "simulated_rEI_frontier_gap": 0.01,
            "rEI_error_readiness": 0.0,
            "rEI_error_frontier_gap": 0.0,
            "rEI_abs_error_readiness": 0.0,
            "rEI_abs_error_frontier_gap": 0.0,
            "rEI_sign_correct_readiness": True,
            "rEI_sign_correct_frontier_gap": True,
            "rEI_wrong_sign_readiness": False,
            "rEI_wrong_sign_frontier_gap": False,
            "emissions_error_readiness": 10.0,
            "emissions_error_frontier_gap": 10.0,
            "contribution_to_aggregate_error_difference": 0.0,
            "emissions_decile": "d1",
            "output_decile": "d1",
            "frontier_gap_decile": "d1",
            "readiness_decile": "d9",
            "output_weight": 0.05,
            "emissions_weight": 0.05,
        },
    ]
    return pl.DataFrame(rows)


def _write_phase16_panel(paths: ABMV4Paths) -> None:
    paths.validation.mkdir(parents=True, exist_ok=True)
    _toy_panel().write_parquet(paths.transition_rule_sign_failure_panel_path)


def test_concentration_diagnostic_identifies_top_contributor_groups() -> None:
    paths = _toy_paths()
    _write_phase16_panel(paths)
    diag = HighEmissionsDampeningDiagnostics(paths)
    concentration = diag.identify_high_emissions_nodes(diag._add_dampening_columns(_toy_panel()), pl.DataFrame())

    electricity = concentration.filter((pl.col("grouping_type") == "Sector") & (pl.col("group") == "Electricity, Gas and Water"))
    assert electricity["aggregate_deterioration_share"].item() > 0.8


def test_electricity_diagnostic_filters_electricity_like_sectors() -> None:
    diagnostic = HighEmissionsDampeningDiagnostics(_toy_paths()).diagnose_electricity_sector(
        HighEmissionsDampeningDiagnostics(_toy_paths())._add_dampening_columns(_toy_panel())
    )

    assert set(diagnostic["country_sector"].to_list()) == {"CHN electricity", "USA electricity"}


def test_china_electricity_diagnostic_handles_country_and_label_matches() -> None:
    diagnostic = HighEmissionsDampeningDiagnostics(_toy_paths()).diagnose_china_electricity(
        HighEmissionsDampeningDiagnostics(_toy_paths())._add_dampening_columns(_toy_panel())
    )

    assert "CHN electricity" in diagnostic["country_sector"].to_list()
    assert diagnostic.filter(pl.col("country_sector") == "CHN electricity")["interpretation"].item() == "China electricity or utility node"


def test_readiness_dampening_amount_is_computed_correctly() -> None:
    out = HighEmissionsDampeningDiagnostics(_toy_paths())._add_dampening_columns(_toy_panel())
    row = out.filter(pl.col("country_sector") == "CHN electricity").to_dicts()[0]

    assert row["dampening_amount"] == pytest.approx(0.04)
    assert row["frontier_gap_more_aggressive"]


def test_simplified_model_selection_identifies_winners() -> None:
    diag = HighEmissionsDampeningDiagnostics(_toy_paths())
    panel = diag._add_dampening_columns(_toy_panel())
    table = diag.build_simplified_model_selection(
        panel,
        diag.diagnose_electricity_sector(panel),
        diag.diagnose_china_electricity(panel),
    )

    mae = table.filter(pl.col("metric") == "unweighted rEI MAE")
    high = table.filter(pl.col("metric") == "high-emissions-node emissions error")
    assert mae["winner"].item() == "frontier_gap_readiness"
    assert high["winner"].item() == "frontier_gap_readiness"


def test_recommendation_selects_dampened_hybrid_when_readiness_helps_high_emissions() -> None:
    diag = HighEmissionsDampeningDiagnostics(_toy_paths())
    panel = diag._add_dampening_columns(_toy_panel())
    concentration = diag.identify_high_emissions_nodes(panel, pl.DataFrame()).with_columns(
        pl.when(
            ((pl.col("grouping_type") == "Sector") & (pl.col("group") == "Electricity, Gas and Water"))
            | ((pl.col("grouping_type") == "Country") & (pl.col("group") == "CHN"))
            | ((pl.col("grouping_type") == "country_sector") & (pl.col("group") == "CHN electricity"))
        )
        .then(0.3)
        .otherwise(pl.col("aggregate_deterioration_share"))
        .alias("aggregate_deterioration_share")
    )
    rec = diag.build_phase17_recommendation(
        concentration=concentration,
        electricity=diag.diagnose_electricity_sector(panel),
        china_electricity=diag.diagnose_china_electricity(panel),
        dampening=diag.compute_readiness_dampening_metrics(panel),
        model_selection=diag.build_simplified_model_selection(panel, diag.diagnose_electricity_sector(panel), diag.diagnose_china_electricity(panel)),
    )

    assert rec["recommendation"].item() == "test_dampened_frontier_gap_hybrid"


def test_recommendation_selects_inspect_when_one_node_group_dominates() -> None:
    diag = HighEmissionsDampeningDiagnostics(_toy_paths())
    panel = diag._add_dampening_columns(_toy_panel())
    concentration = diag.identify_high_emissions_nodes(panel, pl.DataFrame())
    rec = diag.build_phase17_recommendation(
        concentration=concentration,
        electricity=diag.diagnose_electricity_sector(panel),
        china_electricity=diag.diagnose_china_electricity(panel),
        dampening=diag.compute_readiness_dampening_metrics(panel),
        model_selection=diag.build_simplified_model_selection(panel, diag.diagnose_electricity_sector(panel), diag.diagnose_china_electricity(panel)),
    )

    assert rec["recommendation"].item() == "inspect_electricity_data_before_hybrid"


def test_outputs_are_written_only_with_explicit_output_flag() -> None:
    paths = _toy_paths()
    _write_phase16_panel(paths)
    diag = HighEmissionsDampeningDiagnostics(paths)
    result = diag.run()

    assert not paths.phase17_high_emissions_dampening_report_path.exists()
    diag.write_outputs(result)
    assert paths.phase17_high_emissions_dampening_report_path.exists()
    assert paths.phase17_recommendation_path.exists()


def test_missing_phase16_panel_fails_with_actionable_message() -> None:
    with pytest.raises(FileNotFoundError, match="diagnose-transition-rule-tradeoffs"):
        HighEmissionsDampeningDiagnostics(_toy_paths()).load_tradeoff_panel()
