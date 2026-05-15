from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import polars as pl
import pytest

from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.validation import AdaptiveEIDCalibrationDiagnostics


def _toy_paths() -> ABMV4Paths:
    return ABMV4Paths(project_root=Path("tmp") / "abm_v4_adaptive_eid_tests" / uuid4().hex)


def _diagnostic() -> AdaptiveEIDCalibrationDiagnostics:
    return AdaptiveEIDCalibrationDiagnostics(_toy_paths())


def _base_panel(years: range = range(1995, 2003)) -> pl.DataFrame:
    rows = []
    for year in years:
        for node, country, sector, eid, electricity, subtype, emissions in [
            ("CHN | Electricity, Gas and Water", "China", "Electricity, Gas and Water", 1.0, True, "infrastructure_energy", 100.0),
            ("USA | Transport", "USA", "Transport", 0.6, False, "transport_logistics_infrastructure", 60.0),
            ("ROW | TOTAL", "ROW", "TOTAL", 0.95, False, "accounting_or_pseudo_agent", 5.0),
        ]:
            observed = 0.10 if year <= 1999 else 0.025
            rows.append(
                {
                    "country_sector": node,
                    "Country": country,
                    "Sector": sector,
                    "year": year,
                    "X_observed": emissions * 2,
                    "EI_observed": 1.0,
                    "emissions_observed": emissions,
                    "sector_background_trend": 0.0,
                    "rEI_used": 0.10,
                    "observed_rEI": observed,
                    "EID_norm": eid,
                    "EID_fallback_flag": False,
                    "candidate_subtype": subtype,
                    "pseudo_agent_flag": subtype == "accounting_or_pseudo_agent",
                    "electricity_like": electricity,
                    "china_electricity": country == "China" and electricity,
                    "high_EID_flag": eid >= 0.9,
                    "_gap_closure_without_EID": 0.10,
                }
            )
    return pl.DataFrame(rows)


def _write_run_inputs(paths: ABMV4Paths) -> None:
    paths.simulations.mkdir(parents=True, exist_ok=True)
    paths.validation.mkdir(parents=True, exist_ok=True)
    rows = []
    for year in range(1995, 2017):
        for node, country, sector, eid, electricity, subtype, emissions in [
            ("CHN | Electricity, Gas and Water", "China", "Electricity, Gas and Water", 1.0, True, "infrastructure_energy", 100.0),
            ("USA | Transport", "USA", "Transport", 0.6, False, "transport_logistics_infrastructure", 60.0),
            ("DEU | Manufacturing", "DEU", "Manufacturing", 0.2, False, "manufacturing_system_core", 40.0),
        ]:
            rows.append(
                {
                    "country_sector": node,
                    "Country": country,
                    "Sector": sector,
                    "year": year,
                    "X_observed": emissions * 2,
                    "EI_observed": max(0.1, 1.0 - 0.01 * (year - 1995)),
                    "emissions_observed": emissions,
                    "sector_background_trend": 0.0,
                    "rEI_used": 0.08 if electricity else 0.04,
                }
            )
    pl.DataFrame(rows).write_parquet(paths.base_multiyear_state_panel_historical_frontier_gap_path)
    pl.DataFrame(
        {
            "country_sector": ["CHN | Electricity, Gas and Water", "USA | Transport", "DEU | Manufacturing"],
            "EID_score_name": ["structural_dependence_plus_brown_lockin"] * 3,
            "EID_norm": [1.0, 0.6, 0.2],
        }
    ).write_csv(paths.essential_input_dampener_scores_path)
    pl.DataFrame(
        {
            "country_sector": ["CHN | Electricity, Gas and Water", "USA | Transport", "DEU | Manufacturing"],
            "high_EID_definition": ["structural_dependence_plus_brown_lockin"] * 3,
            "candidate_subtype": ["infrastructure_energy", "transport_logistics_infrastructure", "manufacturing_system_core"],
            "pseudo_agent_flag": [False, False, False],
            "electricity_like": [True, False, False],
        }
    ).write_csv(paths.eid_high_node_heterogeneity_panel_path)
    pl.DataFrame(
        {
            "model_variant": ["frontier_gap_readiness", "historical_frontier_gap_only", "historical_frontier_gap_EID_diagnostic"],
            "rows": [100, 100, 100],
            "unweighted_rEI_MAE": [0.09, 0.08, 0.11],
            "output_weighted_rEI_MAE": [0.09, 0.08, 0.11],
            "emissions_weighted_rEI_MAE": [0.09, 0.08, 0.11],
            "wrong_sign_share": [0.3, 0.3, 0.35],
            "emissions_weighted_wrong_sign_share": [0.3, 0.3, 0.35],
            "validation_bias": [0.0, 0.0, 0.0],
            "validation_correlation": [0.1, 0.1, 0.1],
            "latest_year_aggregate_emissions_pct_error": [0.4, 0.5, 0.8],
            "mean_yearly_aggregate_emissions_pct_error": [0.4, 0.5, 0.8],
            "electricity_rEI_MAE": [0.1, 0.09, 0.12],
            "electricity_emissions_weighted_rEI_MAE": [0.1, 0.09, 0.12],
            "electricity_aggregate_emissions_error": [100.0, 120.0, 160.0],
            "china_electricity_rEI_MAE": [0.1, 0.09, 0.12],
            "china_electricity_emissions_error": [80.0, 100.0, 140.0],
            "high_EID_node_rEI_MAE": [0.1, 0.09, 0.12],
            "high_EID_node_emissions_error": [90.0, 110.0, 150.0],
        }
    ).write_csv(paths.multiyear_EID_diagnostic_comparison_path)


def test_parameter_grid_includes_required_lambda_values() -> None:
    grid = _diagnostic().build_parameter_grid()

    assert set(grid["lambda_EID"]) == {0.0, 0.25, 0.5, 0.75, 1.0}


def test_parameter_grid_includes_required_d_min_values() -> None:
    grid = _diagnostic().build_parameter_grid()

    assert set(grid["d_min"]) == {0.25, 0.5, 0.75, 1.0}


def test_rolling_5_year_windows_are_built_correctly() -> None:
    windows = _diagnostic().build_rolling_windows().filter(pl.col("design_name") == "rolling_5yr_cal_3yr_validation")
    first = windows.sort("calibration_start_year").row(0, named=True)

    assert first["calibration_start_year"] == 1995
    assert first["calibration_end_year"] == 1999
    assert first["validation_start_year"] == 2000
    assert first["validation_end_year"] == 2002


def test_rolling_3_year_windows_are_built_correctly() -> None:
    windows = _diagnostic().build_rolling_windows().filter(pl.col("design_name") == "rolling_3yr_cal_2yr_validation")
    first = windows.sort("calibration_start_year").row(0, named=True)

    assert first["calibration_start_year"] == 1995
    assert first["calibration_end_year"] == 1997
    assert first["validation_start_year"] == 1998
    assert first["validation_end_year"] == 1999


def test_calibration_selects_parameters_using_only_calibration_years() -> None:
    diagnostic = _diagnostic()
    grid = pl.DataFrame(
        {
            "parameter_id": ["no_eid", "strong_eid"],
            "lambda_EID": [0.0, 1.0],
            "d_min": [0.25, 0.25],
            "notes": ["toy", "toy"],
        }
    )
    windows = pl.DataFrame(
        {
            "window_id": ["w01"],
            "design_name": ["toy"],
            "calibration_start_year": [1995],
            "calibration_end_year": [1999],
            "validation_start_year": [2000],
            "validation_end_year": [2002],
            "calibration_years": [5],
            "validation_years": [3],
            "notes": ["toy"],
        }
    )

    results, _ = diagnostic.calibrate_and_validate(_base_panel(), grid, windows)

    selected = results.filter(pl.col("objective_name") == "transition_accuracy").row(0, named=True)

    assert selected["lambda_EID"] == pytest.approx(0.0)


def test_validation_evaluates_selected_parameters_on_future_years() -> None:
    diagnostic = _diagnostic()
    grid = diagnostic.build_parameter_grid().filter(pl.col("parameter_id").is_in(["p001", "p017"]))
    windows = diagnostic.build_rolling_windows().filter(pl.col("window_id") == "w01")

    _, panel = diagnostic.calibrate_and_validate(_base_panel(), grid, windows)
    validation_years = panel.filter(pl.col("validation_or_calibration") == "validation")["year"]

    assert validation_years.min() == 2000
    assert validation_years.max() == 2002


def test_d_eid_is_computed_for_each_grid_value() -> None:
    out = _diagnostic().apply_parameter(
        pl.DataFrame(
            {
                "EID_norm": [0.0, 0.5, 1.0],
                "sector_background_trend": [0.0, 0.0, 0.0],
                "_gap_closure_without_EID": [0.1, 0.1, 0.1],
                "observed_rEI": [0.1, 0.1, 0.1],
                "emissions_observed": [1.0, 1.0, 1.0],
            }
        ),
        {"lambda_EID": 0.5, "d_min": 0.75},
    )

    assert out["D_EID"].to_list() == pytest.approx([1.0, 0.75, 0.75])


def test_objective_scores_are_computed_correctly() -> None:
    frame = pl.DataFrame(
        {
            "cal_all_node_unweighted_rEI_MAE": [0.2, 0.1],
            "cal_emissions_weighted_rEI_MAE": [0.2, 0.1],
            "cal_wrong_sign_share": [0.4, 0.2],
            "cal_mean_yearly_aggregate_emissions_pct_error": [0.2, 0.1],
            "cal_electricity_emissions_error": [2.0, 1.0],
            "cal_high_EID_node_emissions_error": [2.0, 1.0],
        }
    )

    scored = _diagnostic().add_objective_score(frame, "balanced_policy_objective")

    assert scored["calibration_score"].to_list() == pytest.approx([1.0, 0.0])


def test_adaptive_validation_panel_includes_selected_lambda_and_d_min() -> None:
    diagnostic = _diagnostic()
    grid = diagnostic.build_parameter_grid().filter(pl.col("parameter_id").is_in(["p001"]))
    windows = diagnostic.build_rolling_windows().filter(pl.col("window_id") == "w01")

    _, panel = diagnostic.calibrate_and_validate(_base_panel(), grid, windows)

    assert {"lambda_EID", "d_min"}.issubset(panel.columns)


def test_parameter_stability_detects_stable_sequences() -> None:
    stability = _diagnostic().build_parameter_stability(
        pl.DataFrame(
            {
                "design_name": ["d", "d"],
                "objective_name": ["o", "o"],
                "window_id": ["w01", "w02"],
                "lambda_EID": [0.25, 0.25],
                "d_min": [0.75, 0.75],
            }
        )
    )

    assert stability["stable_parameter_flag"].item() is True


def test_parameter_stability_detects_regime_dependent_sequences() -> None:
    stability = _diagnostic().build_parameter_stability(
        pl.DataFrame(
            {
                "design_name": ["d", "d"],
                "objective_name": ["o", "o"],
                "window_id": ["w01", "w02"],
                "lambda_EID": [0.25, 1.0],
                "d_min": [0.75, 0.25],
            }
        )
    )

    assert stability["stable_parameter_flag"].item() is False


def test_hypothesis_summary_classifies_overfitting_when_forward_validation_fails() -> None:
    diagnostic = _diagnostic()
    calibration = pl.DataFrame(
        {
            "calibration_score": [0.1],
            "validation_score": [0.3],
            "lambda_EID": [1.0],
            "d_min": [0.25],
        }
    )
    comparison = pl.DataFrame(
        {
            "model_variant": ["adaptive_EID", "historical_frontier_gap_only", "historical_frontier_gap_EID_diagnostic"],
            "emissions_weighted_rEI_MAE": [0.2, 0.1, 0.3],
        }
    )
    stability = pl.DataFrame({"stable_parameter_flag": [False]})

    tests = diagnostic.build_hypothesis_tests(calibration, comparison, stability, pl.DataFrame())
    overfit = tests.filter(pl.col("hypothesis") == "adaptive_calibration_overfitting").row(0, named=True)

    assert overfit["result"] is True


def _comparison(adaptive: float, hist: float, fixed: float = 0.3, aggregate: float = 0.1) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "model_variant": ["adaptive_EID", "historical_frontier_gap_only", "historical_frontier_gap_EID_diagnostic"],
            "emissions_weighted_rEI_MAE": [adaptive, hist, fixed],
            "all_node_unweighted_rEI_MAE": [adaptive, hist, fixed],
            "wrong_sign_share": [0.2, 0.2, 0.2],
            "mean_yearly_aggregate_emissions_pct_error": [aggregate, 0.2, 0.3],
        }
    )


def test_recommendation_selects_stable_eid_parameter_when_stable_and_improves() -> None:
    rec = _diagnostic().build_recommendation(
        _comparison(0.1, 0.2, aggregate=0.1),
        pl.DataFrame({"stable_parameter_flag": [True]}),
        pl.DataFrame(),
        pl.DataFrame(),
        pl.DataFrame({"lambda_EID": [0.25], "d_min": [0.75]}),
    )

    assert rec["recommendation"].item() == "stable_EID_parameter_found"


def test_recommendation_selects_regime_dependent_eid_when_changing_parameters_improve() -> None:
    rec = _diagnostic().build_recommendation(
        _comparison(0.1, 0.2, aggregate=0.1),
        pl.DataFrame({"stable_parameter_flag": [False]}),
        pl.DataFrame(),
        pl.DataFrame(),
        pl.DataFrame({"lambda_EID": [0.25], "d_min": [0.75]}),
    )

    assert rec["recommendation"].item() == "regime_dependent_EID_found"


def test_recommendation_selects_reject_when_adaptive_eid_fails() -> None:
    rec = _diagnostic().build_recommendation(
        _comparison(0.3, 0.2, aggregate=0.25),
        pl.DataFrame({"stable_parameter_flag": [False]}),
        pl.DataFrame(),
        pl.DataFrame(),
        pl.DataFrame({"lambda_EID": [1.0], "d_min": [0.25]}),
    )

    assert rec["recommendation"].item() == "reject_EID_for_v4_confirmed"


def test_outputs_are_written_only_with_explicit_output_flag() -> None:
    paths = _toy_paths()
    _write_run_inputs(paths)
    diagnostic = AdaptiveEIDCalibrationDiagnostics(paths)

    result = diagnostic.run()

    assert not paths.adaptive_EID_report_path.exists()
    diagnostic.write_outputs(result)
    assert paths.adaptive_EID_parameter_grid_path.exists()
    assert paths.adaptive_EID_report_path.exists()


def test_missing_phase_25_or_23_inputs_fail_clearly() -> None:
    paths = _toy_paths()

    with pytest.raises(FileNotFoundError, match="Run Phase 15 first"):
        AdaptiveEIDCalibrationDiagnostics(paths).build_base_panel()
