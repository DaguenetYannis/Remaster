from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import polars as pl
import pytest

from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.validation import ElectricityTransitionRegimeDiagnostics


def _toy_paths() -> ABMV4Paths:
    return ABMV4Paths(project_root=Path("tmp") / "abm_v4_electricity_regime_tests" / uuid4().hex)


def _observed() -> pl.DataFrame:
    rows = []
    for country, sector, eis, xs in [
        ("CHN", "Electricity, Gas and Water", [4.0, 2.0, 1.0, 1.0], [100.0, 200.0, 300.0, 320.0]),
        ("USA", "power utilities", [2.0, 1.8, 1.6, 1.5], [90.0, 95.0, 100.0, 105.0]),
        ("FRA", "Financial Intermediation", [0.5, 0.5, 0.5, 0.5], [20.0, 20.0, 20.0, 20.0]),
    ]:
        for offset, (ei, x) in enumerate(zip(eis, xs)):
            rows.append(
                {
                    "country_sector": f"{country} | {sector}",
                    "year": 2000 + offset,
                    "Country": country,
                    "Sector": sector,
                    "X_observed": x,
                    "EI_observed": ei,
                    "emissions_observed": x * ei,
                }
            )
    return pl.DataFrame(rows)


def _model(name: str) -> pl.DataFrame:
    rows = []
    for country, sector in [("CHN", "Electricity, Gas and Water"), ("USA", "power utilities")]:
        for year in range(2000, 2004):
            rows.append(
                {
                    "country_sector": f"{country} | {sector}",
                    "year": year,
                    "rEI_used": 0.20 if name == "readiness" else 0.40,
                    "EI_sim": 1.0,
                    "emissions_sim": 100.0,
                    "readiness": 0.1 if country == "CHN" else 0.9,
                }
            )
    return pl.DataFrame(rows)


def _write_phase19(paths: ABMV4Paths) -> None:
    paths.validation.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "finding": ["raw"],
            "recommended_next_action": ["treat_electricity_as_sector_specific_transition_case"],
        }
    ).write_csv(paths.raw_eora_electricity_data_audit_recommendation_path)
    pl.DataFrame(
        {
            "country_sector": ["CHN | Electricity, Gas and Water", "USA | power utilities"],
            "Country": ["CHN", "USA"],
            "year": [2001, 2001],
            "X": [200.0, 95.0],
        }
    ).write_csv(paths.raw_eora_major_electricity_comparison_path)
    pl.DataFrame(
        {
            "country_sector": ["CHN | Electricity, Gas and Water"],
            "year": [2001],
            "variable": ["EI"],
            "jump_flag": [True],
        }
    ).write_csv(paths.raw_eora_electricity_breakpoint_audit_path)


def _write_inputs(paths: ABMV4Paths) -> None:
    paths.inputs.mkdir(parents=True, exist_ok=True)
    paths.simulations.mkdir(parents=True, exist_ok=True)
    _observed().rename({"year": "Year", "EI_observed": "EI"}).write_parquet(paths.state_panel_path(1995, 2016))
    _model("readiness").write_parquet(paths.base_multiyear_state_panel_path)
    _model("frontier").write_parquet(paths.base_multiyear_state_panel_historical_frontier_gap_path)
    _write_phase19(paths)


def test_transition_panel_includes_only_electricity_like_nodes() -> None:
    diag = ElectricityTransitionRegimeDiagnostics(_toy_paths(), start_year=1995, end_year=2016)
    panel = diag.build_electricity_transition_panel(_observed(), diag._model_output(_model("readiness"), "readiness"), diag._model_output(_model("frontier"), "frontier_gap"))

    assert set(panel["Country"].unique().to_list()) == {"CHN", "USA"}


def test_targets_are_computed_correctly() -> None:
    diag = ElectricityTransitionRegimeDiagnostics(_toy_paths(), start_year=1995, end_year=2016)
    panel = diag.build_electricity_transition_panel(_observed(), diag._model_output(_model("readiness"), "readiness"), diag._model_output(_model("frontier"), "frontier_gap"))
    row = panel.filter((pl.col("Country") == "CHN") & (pl.col("year") == 2000)).to_dicts()[0]

    assert row["one_year_rEI"] == pytest.approx(0.69314718056)
    assert row["three_year_annualized_rEI"] == pytest.approx(0.46209812037)
    assert "winsorized_one_year_rEI" in panel.columns
    assert "smoothed_one_year_rEI" in panel.columns


def test_fixed_dampener_reduces_gap_closure() -> None:
    diag = ElectricityTransitionRegimeDiagnostics(_toy_paths())
    panel = diag.build_electricity_transition_panel(_observed(), diag._model_output(_model("readiness"), "readiness"), diag._model_output(_model("frontier"), "frontier_gap")).with_columns(pl.lit(1.0).alias("frontier_gap"))
    rules = diag.evaluate_candidate_rules(panel)
    low = rules.filter(pl.col("rule_name") == "electricity_dampened_frontier_gap_0_25")["simulated_rEI"].mean()
    high = rules.filter(pl.col("rule_name") == "electricity_dampened_frontier_gap_0_75")["simulated_rEI"].mean()

    assert high > low


def test_readiness_dampener_clips_readiness_between_bounds() -> None:
    diag = ElectricityTransitionRegimeDiagnostics(_toy_paths())
    panel = diag.build_electricity_transition_panel(_observed(), diag._model_output(_model("readiness"), "readiness"), diag._model_output(_model("frontier"), "frontier_gap")).with_columns(
        pl.when((pl.col("Country") == "CHN") & (pl.col("year") == 2000)).then(1).otherwise(0).alias("jump_flag_count")
    )
    rules = diag.evaluate_candidate_rules(panel)
    readiness_rule = rules.filter(pl.col("rule_name") == "electricity_readiness_dampened_frontier_gap")
    low = rules.filter(pl.col("rule_name") == "electricity_dampened_frontier_gap_0_25")
    high = rules.filter(pl.col("rule_name") == "electricity_dampened_frontier_gap_0_75")

    assert readiness_rule["simulated_rEI"].min() >= low["simulated_rEI"].min()
    assert readiness_rule["simulated_rEI"].max() <= high["simulated_rEI"].max()


def test_jump_filter_uses_background_in_jump_years() -> None:
    diag = ElectricityTransitionRegimeDiagnostics(_toy_paths())
    panel = diag.build_electricity_transition_panel(_observed(), diag._model_output(_model("readiness"), "readiness"), diag._model_output(_model("frontier"), "frontier_gap")).with_columns(
        pl.when((pl.col("Country") == "CHN") & (pl.col("year") == 2000)).then(1).otherwise(0).alias("jump_flag_count")
    )
    rules = diag.evaluate_candidate_rules(panel)
    jump = rules.filter((pl.col("rule_name") == "electricity_gap_with_jump_shock_filter") & (pl.col("jump_flag_count") > 0)).to_dicts()[0]

    assert jump["simulated_rEI"] == pytest.approx(jump["electricity_background"])


def test_high_emissions_rule_dampens_top_decile_more() -> None:
    diag = ElectricityTransitionRegimeDiagnostics(_toy_paths())
    panel = diag.build_electricity_transition_panel(_observed(), diag._model_output(_model("readiness"), "readiness"), diag._model_output(_model("frontier"), "frontier_gap"))
    rules = diag.evaluate_candidate_rules(panel)
    rule = rules.filter(pl.col("rule_name") == "electricity_high_emissions_dampened_gap")

    assert rule.filter(pl.col("emissions_decile").is_in(["d9", "d10"]))["simulated_rEI"].mean() <= rule["simulated_rEI"].max()


def test_rule_comparison_metrics_are_computed() -> None:
    diag = ElectricityTransitionRegimeDiagnostics(_toy_paths())
    panel = diag.build_electricity_transition_panel(_observed(), diag._model_output(_model("readiness"), "readiness"), diag._model_output(_model("frontier"), "frontier_gap"))
    comparison = diag.build_rule_comparison(diag.evaluate_candidate_rules(panel))

    assert {"rule_name", "unweighted_rEI_MAE", "electricity_aggregate_emissions_error"}.issubset(comparison.columns)


def test_china_comparison_is_produced_when_china_exists() -> None:
    diag = ElectricityTransitionRegimeDiagnostics(_toy_paths())
    panel = diag.build_electricity_transition_panel(_observed(), diag._model_output(_model("readiness"), "readiness"), diag._model_output(_model("frontier"), "frontier_gap"))
    china = diag.summarize_china_electricity(diag.evaluate_candidate_rules(panel))

    assert not china.is_empty()
    assert set(china["rule_name"].unique().to_list())


def test_recommendation_selects_dampened_when_it_improves_both() -> None:
    diag = ElectricityTransitionRegimeDiagnostics(_toy_paths())
    comparison = pl.DataFrame(
        {
            "rule_name": ["current_frontier_gap_readiness_reference", "electricity_dampened_frontier_gap_0_25"],
            "unweighted_rEI_MAE": [1.0, 0.5],
            "electricity_aggregate_emissions_error": [100.0, 50.0],
            "jump_year_rEI_MAE": [1.0, 1.0],
            "nonjump_year_rEI_MAE": [1.0, 1.0],
            "china_electricity_emissions_error": [100.0, 50.0],
        }
    )

    assert diag.build_recommendation(comparison)["recommendation"].item(0) == "test_electricity_dampened_frontier_gap_as_candidate_rule"


def test_recommendation_selects_jump_filter_when_jump_years_drive_improvement() -> None:
    diag = ElectricityTransitionRegimeDiagnostics(_toy_paths())
    comparison = pl.DataFrame(
        {
            "rule_name": ["current_frontier_gap_readiness_reference", "electricity_gap_with_jump_shock_filter"],
            "unweighted_rEI_MAE": [1.0, 1.0],
            "electricity_aggregate_emissions_error": [100.0, 100.0],
            "jump_year_rEI_MAE": [1.0, 0.5],
            "nonjump_year_rEI_MAE": [1.0, 1.0],
            "china_electricity_emissions_error": [100.0, 100.0],
        }
    )

    assert diag.build_recommendation(comparison)["recommendation"].item(0) == "test_electricity_jump_filtered_rule"


def test_outputs_are_written_only_with_explicit_output_flag() -> None:
    paths = _toy_paths()
    _write_inputs(paths)
    diag = ElectricityTransitionRegimeDiagnostics(paths)
    result = diag.run()

    assert not paths.electricity_transition_regime_report_path.exists()
    diag.write_outputs(result)
    assert paths.electricity_transition_rule_comparison_path.exists()
    assert paths.electricity_transition_regime_report_path.exists()


def test_missing_phase19_outputs_fail_clearly() -> None:
    with pytest.raises(FileNotFoundError, match="audit-raw-eora-electricity-data"):
        ElectricityTransitionRegimeDiagnostics(_toy_paths()).run()
