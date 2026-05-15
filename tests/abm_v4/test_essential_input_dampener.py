from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import polars as pl
import pytest

from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.validation import EssentialInputDampenerTester


def _toy_paths() -> ABMV4Paths:
    return ABMV4Paths(project_root=Path("tmp") / "abm_v4_essential_input_dampener_tests" / uuid4().hex)


def _write_phase22_metrics(paths: ABMV4Paths) -> None:
    paths.validation.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "country_sector": [
                "CHN | Electricity, Gas and Water",
                "USA | Electricity, Gas and Water",
                "DEU | Manufacturing",
                "FRA | Services",
                "BRA | Agriculture",
            ],
            "Country": ["China", "USA", "DEU", "FRA", "BRA"],
            "Sector": ["Electricity, Gas and Water", "Electricity, Gas and Water", "Manufacturing", "Services", "Agriculture"],
            "electricity_like": [True, True, False, False, False],
            "essential_input_score_diagnostic": [1.0, 0.8, 0.3, 0.1, None],
            "low_substitutability_score_diagnostic": [0.9, 0.7, 0.2, 0.1, 0.0],
            "systemic_dependence_score_diagnostic": [1.0, 0.6, 0.4, 0.2, 0.0],
            "structural_dependence_score_diagnostic": [0.95, 0.75, 0.35, 0.1, 0.0],
            "mean_brown_centrality": [0.8, 0.6, 0.2, 0.0, 0.0],
            "jump_frequency": [0.3, 0.2, 0.1, 0.0, 0.0],
        }
    ).write_csv(paths.essential_input_node_metrics_path)


def _write_transition_panel(paths: ABMV4Paths) -> None:
    paths.validation.mkdir(parents=True, exist_ok=True)
    nodes = [
        ("CHN | Electricity, Gas and Water", "China", "Electricity, Gas and Water", 1000.0, 100.0, "d10"),
        ("USA | Electricity, Gas and Water", "USA", "Electricity, Gas and Water", 800.0, 80.0, "d9"),
        ("DEU | Manufacturing", "DEU", "Manufacturing", 500.0, 25.0, "d5"),
        ("FRA | Services", "FRA", "Services", 300.0, 9.0, "d2"),
    ]
    rows = []
    for year in [2009, 2010, 2011, 2012]:
        for country_sector, country, sector, output, emissions, decile in nodes:
            is_electricity = "Electricity" in sector
            observed = 0.02 if is_electricity else 0.05
            frontier = 0.08 if is_electricity else 0.045
            readiness = 0.035 if is_electricity else 0.047
            rows.append(
                {
                    "country_sector": country_sector,
                    "year": year,
                    "Country": country,
                    "Sector": sector,
                    "observed_rEI": observed,
                    "simulated_rEI_frontier_gap": frontier,
                    "simulated_rEI_readiness": readiness,
                    "X_observed": output,
                    "EI_observed": emissions / output,
                    "emissions_observed": emissions,
                    "emissions_decile": decile,
                }
            )
    pl.DataFrame(rows).write_parquet(paths.transition_rule_sign_failure_panel_path)


def _write_inputs(paths: ABMV4Paths) -> None:
    _write_phase22_metrics(paths)
    _write_transition_panel(paths)


def test_eid_scores_are_normalized_with_robust_scaling() -> None:
    paths = _toy_paths()
    _write_phase22_metrics(paths)

    scores = EssentialInputDampenerTester(paths).normalize_eid_scores()
    score = scores.filter(pl.col("EID_score_name") == "essential_input_score_diagnostic")

    assert score["EID_norm"].drop_nulls().min() >= 0
    assert score["EID_norm"].drop_nulls().max() <= 1
    assert score.filter(pl.col("country_sector") == "CHN | Electricity, Gas and Water")["EID_norm"].item() == pytest.approx(1.0)


def test_missing_eid_scores_are_flagged_without_imputation() -> None:
    paths = _toy_paths()
    _write_phase22_metrics(paths)

    scores = EssentialInputDampenerTester(paths).normalize_eid_scores()
    missing = scores.filter(
        (pl.col("country_sector") == "BRA | Agriculture")
        & (pl.col("EID_score_name") == "essential_input_score_diagnostic")
    ).to_dicts()[0]

    assert missing["missing_flag"] is True
    assert missing["EID_norm"] is None


def test_d_eid_is_computed_for_lambda_and_floor() -> None:
    tester = EssentialInputDampenerTester(_toy_paths())
    out = pl.DataFrame({"eid": [0.0, 0.5, 1.0]}).select(tester.compute_d_eid(pl.col("eid"), 0.75, 0.5).alias("d"))

    assert out["d"].to_list() == pytest.approx([1.0, 0.625, 0.5])


def test_train_validation_split_is_respected() -> None:
    paths = _toy_paths()
    _write_transition_panel(paths)

    panel = EssentialInputDampenerTester(paths).build_evaluation_panel()

    assert set(panel.filter(pl.col("year") <= 2010)["train_or_validation"].unique()) == {"train"}
    assert set(panel.filter(pl.col("year") >= 2011)["train_or_validation"].unique()) == {"validation"}


def test_historical_residual_theta_is_computed_correctly() -> None:
    paths = _toy_paths()
    _write_transition_panel(paths)
    tester = EssentialInputDampenerTester(paths)
    pred = tester.build_evaluation_panel().with_columns(pl.col("simulated_rEI_frontier_gap").alias("predicted_without_residual"))
    candidate = {"candidate_id": "toy", "residual_level": "sector", "shrinkage_k": 5, "p_min": 0.5, "p_max": 1.5}

    residual = tester.calibrate_historical_residual(pred, candidate)
    electricity = residual.filter((pl.col("Sector") == "Electricity, Gas and Water") & (pl.col("year") == 2011)).to_dicts()[0]

    assert electricity["theta_shrunk"] == pytest.approx(-0.06)
    assert electricity["P_hist"] == pytest.approx(0.94)


def test_residual_shrinkage_toward_sector_mean_works() -> None:
    paths = _toy_paths()
    paths.validation.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "country_sector": ["a", "b"],
            "year": [2010, 2010],
            "Country": ["A", "B"],
            "Sector": ["S", "S"],
            "observed_rEI": [0.0, 0.0],
            "simulated_rEI_frontier_gap": [0.2, 0.0],
            "simulated_rEI_readiness": [0.0, 0.0],
            "X_observed": [1.0, 1.0],
            "EI_observed": [1.0, 1.0],
            "emissions_observed": [1.0, 1.0],
            "emissions_decile": ["d1", "d1"],
        }
    ).write_parquet(paths.transition_rule_sign_failure_panel_path)
    tester = EssentialInputDampenerTester(paths)
    pred = tester.build_evaluation_panel().with_columns(pl.col("simulated_rEI_frontier_gap").alias("predicted_without_residual"))
    candidate = {"candidate_id": "toy", "residual_level": "country_sector", "shrinkage_k": 1, "p_min": 0.5, "p_max": 1.5}

    residual = tester.calibrate_historical_residual(pred, candidate)
    row_a = residual.filter(pl.col("country_sector") == "a").to_dicts()[0]

    assert row_a["theta_raw"] == pytest.approx(-0.2)
    assert row_a["theta_sector_mean"] == pytest.approx(-0.1)
    assert row_a["theta_shrunk"] == pytest.approx(-0.15)


def test_p_hist_is_clipped_to_bounds() -> None:
    paths = _toy_paths()
    _write_transition_panel(paths)
    tester = EssentialInputDampenerTester(paths)
    pred = tester.build_evaluation_panel().with_columns(pl.lit(2.0).alias("predicted_without_residual"))
    candidate = {"candidate_id": "toy", "residual_level": "sector", "shrinkage_k": 5, "p_min": 0.75, "p_max": 1.25}

    residual = tester.calibrate_historical_residual(pred, candidate)

    assert residual["P_hist"].min() == pytest.approx(0.75)


def test_candidate_grid_includes_required_variant_families() -> None:
    grid = EssentialInputDampenerTester(_toy_paths()).build_candidate_grid()

    assert {
        "frontier_gap_readiness_baseline",
        "historical_frontier_gap_only_baseline",
        "essential_input_dampener_only",
        "historical_residual_only",
        "essential_input_dampener_plus_historical_residual",
    }.issubset(set(grid["variant_name"]))


def test_validation_metrics_are_computed_correctly() -> None:
    paths = _toy_paths()
    _write_inputs(paths)
    tester = EssentialInputDampenerTester(paths)
    panel = tester.build_evaluation_panel()
    scores = tester.normalize_eid_scores()
    grid = tester.build_candidate_grid().filter(pl.col("variant_name").is_in(["frontier_gap_readiness_baseline", "historical_frontier_gap_only_baseline"]))
    predictions, _ = tester.evaluate_candidates(panel, scores, grid)
    validation = tester.build_validation_results(predictions, grid)

    assert {"all_node_unweighted_rEI_MAE", "electricity_rEI_MAE", "mean_yearly_aggregate_emissions_pct_error"}.issubset(validation.columns)
    assert validation.filter(pl.col("train_or_validation") == "validation")["all_node_unweighted_rEI_MAE"].min() >= 0


def test_material_worsening_flags_are_triggered() -> None:
    tester = EssentialInputDampenerTester(_toy_paths())
    baseline = {
        "all_node_unweighted_rEI_MAE": 1.0,
        "all_node_emissions_weighted_rEI_MAE": 1.0,
        "all_node_wrong_sign_share": 0.10,
        "mean_yearly_aggregate_emissions_pct_error": 0.10,
    }
    candidate = {
        "all_node_unweighted_rEI_MAE": 1.03,
        "all_node_emissions_weighted_rEI_MAE": 1.03,
        "all_node_wrong_sign_share": 0.13,
        "mean_yearly_aggregate_emissions_pct_error": 0.13,
    }

    assert all(tester.material_worsening_flags(baseline, candidate).values())


def test_mechanism_decomposition_detects_residual_dominance() -> None:
    tester = EssentialInputDampenerTester(_toy_paths())
    validation = pl.DataFrame(
        {
            "candidate_id": ["base", "eid", "res", "combo"],
            "variant_name": [
                "historical_frontier_gap_only_baseline",
                "essential_input_dampener_only",
                "historical_residual_only",
                "essential_input_dampener_plus_historical_residual",
            ],
            "train_or_validation": ["validation"] * 4,
            "EID_score_name": [None, "essential", None, "essential"],
            "all_node_emissions_weighted_rEI_MAE": [1.0, 0.95, 0.5, 0.45],
        }
    )
    predictions = pl.DataFrame(
        {
            "candidate_id": ["combo"],
            "D_EID": [0.5],
            "P_hist": [1.1],
            "electricity_like": [True],
            "china_electricity": [False],
            "high_EID_decile": [True],
        }
    )

    mechanism = tester.build_mechanism_decomposition(validation, predictions)

    assert mechanism["residual_dominates_flag"].item() is True


def test_abm_v5_implication_table_marks_supported_high_eid_improvement() -> None:
    tester = EssentialInputDampenerTester(_toy_paths())
    validation = pl.DataFrame(
        {
            "candidate_id": ["base", "eid"],
            "variant_name": ["historical_frontier_gap_only_baseline", "essential_input_dampener_only"],
            "train_or_validation": ["validation", "validation"],
            "all_node_emissions_weighted_rEI_MAE": [1.0, 0.9],
            "high_EID_node_rEI_MAE": [0.5, 0.4],
        }
    )

    v5 = tester.build_abm_v5_implications(validation, pl.DataFrame())

    assert v5["agent_type_candidate"].item() == "essential_input_agent"


def test_recommendation_selects_proceed_when_eid_improves_without_material_worsening() -> None:
    tester = EssentialInputDampenerTester(_toy_paths())
    validation = pl.DataFrame(
        {
            "candidate_id": ["base", "eid"],
            "variant_name": ["historical_frontier_gap_only_baseline", "essential_input_dampener_only"],
            "train_or_validation": ["validation", "validation"],
            "all_node_emissions_weighted_rEI_MAE": [1.0, 0.99],
            "all_node_unweighted_rEI_MAE": [1.0, 0.99],
            "all_node_wrong_sign_share": [0.10, 0.10],
            "mean_yearly_aggregate_emissions_pct_error": [0.10, 0.10],
            "electricity_rEI_MAE": [0.5, 0.4],
            "high_EID_node_rEI_MAE": [0.6, 0.5],
        }
    )
    mechanism = pl.DataFrame({"residual_dominates_flag": [False]})

    recommendation = tester.build_recommendation(validation, mechanism)

    assert recommendation["recommendation"].item() == "proceed_to_multiyear_candidate_integration"


def test_recommendation_selects_residual_dominance() -> None:
    tester = EssentialInputDampenerTester(_toy_paths())
    validation = pl.DataFrame(
        {
            "candidate_id": ["base", "eid"],
            "variant_name": ["historical_frontier_gap_only_baseline", "essential_input_dampener_only"],
            "train_or_validation": ["validation", "validation"],
            "all_node_emissions_weighted_rEI_MAE": [1.0, 0.99],
            "all_node_unweighted_rEI_MAE": [1.0, 0.99],
            "all_node_wrong_sign_share": [0.10, 0.10],
            "mean_yearly_aggregate_emissions_pct_error": [0.10, 0.10],
            "electricity_rEI_MAE": [0.5, 0.4],
            "high_EID_node_rEI_MAE": [0.6, 0.5],
        }
    )
    mechanism = pl.DataFrame({"residual_dominates_flag": [True]})

    recommendation = tester.build_recommendation(validation, mechanism)

    assert recommendation["recommendation"].item() == "residual_dominates_mechanism"


def test_outputs_are_written_only_with_explicit_output_flag() -> None:
    paths = _toy_paths()
    _write_inputs(paths)
    tester = EssentialInputDampenerTester(paths)
    result = tester.run()

    assert not paths.essential_input_dampener_report_path.exists()
    tester.write_outputs(result)
    assert paths.essential_input_dampener_validation_results_path.exists()
    assert paths.essential_input_dampener_report_path.exists()


def test_missing_phase22_input_metrics_fail_clearly() -> None:
    paths = _toy_paths()
    _write_transition_panel(paths)

    with pytest.raises(FileNotFoundError, match="Missing Phase 22 node metrics"):
        EssentialInputDampenerTester(paths).normalize_eid_scores()
