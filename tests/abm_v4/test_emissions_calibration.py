from pathlib import Path
from uuid import uuid4

import polars as pl
import pytest

from src.abm_v4.config import ABMV4Config
from src.abm_v4.emissions import EmissionsTransitionCalibrator
from src.abm_v4.paths import ABMV4Paths


def _toy_paths() -> ABMV4Paths:
    return ABMV4Paths(project_root=Path("tmp") / "abm_v4_emissions_calibration_tests" / uuid4().hex)


def _toy_state_panel() -> pl.DataFrame:
    rows = []
    for year in range(1995, 2017):
        rows.extend(
            [
                {
                    "country_sector": "A_S1",
                    "Year": year,
                    "Country": "A",
                    "Sector": "S1",
                    "ecosystem_id": "eco1",
                    "EI": 2.0 - 0.02 * (year - 1995),
                    "general_capability_model": 0.8,
                    "green_capability_model": 0.7,
                    "general_capability_source": "atlas_observed",
                    "green_capability_source": "atlas_observed",
                    "network_green_exposure": 0.6,
                    "brown_centrality": 0.1,
                },
                {
                    "country_sector": "B_S1",
                    "Year": year,
                    "Country": "B",
                    "Sector": "S1",
                    "ecosystem_id": "eco1",
                    "EI": 3.0 - 0.01 * (year - 1995),
                    "general_capability_model": 0.4,
                    "green_capability_model": 0.2,
                    "general_capability_source": "io_imputed",
                    "green_capability_source": "io_imputed",
                    "network_green_exposure": 0.3,
                    "brown_centrality": 0.5,
                },
                {
                    "country_sector": "C_S2",
                    "Year": year,
                    "Country": "C",
                    "Sector": "S2",
                    "ecosystem_id": "eco2",
                    "EI": None if year == 2000 else 1.5 + 0.01 * (year - 1995),
                    "general_capability_model": 0.3,
                    "green_capability_model": 0.1,
                    "general_capability_source": "unavailable",
                    "green_capability_source": "unavailable",
                    "network_green_exposure": 0.2,
                    "brown_centrality": 0.2,
                },
            ]
        )
    return pl.DataFrame(rows)


def _write_toy_inputs(paths: ABMV4Paths, config: ABMV4Config) -> None:
    state_path = paths.state_panel_path(config.start_year, config.end_year)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    _toy_state_panel().write_parquet(state_path)
    paths.supplier_updated_weights_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "buyer_country_sector": ["A_S1", "A_S1", "B_S1"],
            "supplier_country_sector": ["B_S1", "C_S2", "A_S1"],
            "updated_weight": [0.7, 0.3, 1.0],
        }
    ).write_parquet(paths.supplier_updated_weights_path)


def test_calibration_dataset_computes_observed_rei_and_excludes_invalid_ei() -> None:
    paths = _toy_paths()
    config = ABMV4Config(start_year=1995, end_year=2016)
    _write_toy_inputs(paths, config)
    dataset = EmissionsTransitionCalibrator(paths, start_year=1995, end_year=2016).build_calibration_dataset()

    row = dataset.filter((pl.col("country_sector") == "A_S1") & (pl.col("year") == 1995)).to_dicts()[0]
    assert row["observed_rEI"] == pytest.approx(__import__("math").log(2.0) - __import__("math").log(1.98))
    assert dataset.filter((pl.col("country_sector") == "C_S2") & (pl.col("year") == 1999)).is_empty()


def test_calibration_temporal_split_assigns_years() -> None:
    dataset = EmissionsTransitionCalibrator(ABMV4Paths(project_root=Path("tmp"))).build_calibration_dataset(
        _toy_state_panel()
    )
    train, validation = EmissionsTransitionCalibrator(
        ABMV4Paths(project_root=Path("tmp")),
        train_end_year=2011,
        validation_start_year=2012,
    ).split_train_validation(dataset)

    assert train["year"].max() <= 2011
    assert validation["year"].min() >= 2012


def test_calibration_sampled_parameters_respect_bounds() -> None:
    calibrator = EmissionsTransitionCalibrator(
        ABMV4Paths(project_root=Path("tmp")),
        random_search_iterations=10,
        seed=7,
    )
    for params in calibrator.sample_parameter_sets():
        for name, value in params.items():
            lower, upper = calibrator.PARAMETER_BOUNDS[name]
            assert lower <= value <= upper
            if name != "theta_intercept":
                assert value >= 0


def test_calibration_metrics_include_required_loss_values() -> None:
    calibrator = EmissionsTransitionCalibrator(ABMV4Paths(project_root=Path("tmp")))
    predictions = pl.DataFrame(
        {
            "Sector": ["S1", "S1"],
            "observed_rEI": [0.1, -0.1],
            "simulated_rEI": [0.2, 0.1],
        }
    ).with_columns(
        (pl.col("simulated_rEI") - pl.col("observed_rEI")).alias("rEI_error"),
        (pl.col("simulated_rEI") - pl.col("observed_rEI")).abs().alias("rEI_abs_error"),
    )
    metrics = calibrator.compute_metrics(predictions)

    assert metrics["mae"] == pytest.approx(0.15)
    assert metrics["bias"] == pytest.approx(0.15)
    assert metrics["wrong_sign_share"] == pytest.approx(0.5)
    assert "correlation" in metrics


def test_calibration_best_parameter_selection_minimizes_validation_mae() -> None:
    calibrator = EmissionsTransitionCalibrator(ABMV4Paths(project_root=Path("tmp")))
    results = pl.DataFrame(
        {
            "validation_mae": [0.2, 0.1],
            **{name: [bounds[0], bounds[1]] for name, bounds in calibrator.PARAMETER_BOUNDS.items()},
        }
    )
    best = calibrator.select_best_parameters(results)

    assert best["rho_max"] == calibrator.PARAMETER_BOUNDS["rho_max"][1]


def test_calibration_model_comparison_includes_required_baselines() -> None:
    dataset = EmissionsTransitionCalibrator(ABMV4Paths(project_root=Path("tmp"))).build_calibration_dataset(
        _toy_state_panel()
    )
    calibrator = EmissionsTransitionCalibrator(
        ABMV4Paths(project_root=Path("tmp")),
        random_search_iterations=2,
    )
    params = calibrator._default_parameter_set()
    comparison = calibrator.evaluate_baseline_models(dataset, params)

    assert set(comparison["model_name"].to_list()) == {
        "frontier_gap_readiness",
        "sector_background_only",
        "frontier_gap_only",
        "readiness_without_capability",
        "legacy_raw_log",
    }


def test_calibration_plausibility_detects_boundary_solutions() -> None:
    calibrator = EmissionsTransitionCalibrator(ABMV4Paths(project_root=Path("tmp")))
    params = {name: bounds[0] for name, bounds in calibrator.PARAMETER_BOUNDS.items()}
    comparison = pl.DataFrame(
        {
            "model_name": ["frontier_gap_readiness", "sector_background_only", "readiness_without_capability"],
            "validation_mae": [0.1, 0.2, 0.11],
        }
    )
    summary = pl.DataFrame({"split": ["validation"], "wrong_sign_share": [0.2]})
    plausibility = calibrator.build_parameter_plausibility_report(params, comparison, summary)

    assert plausibility.filter(pl.col("parameter") == "rho_max")["near_lower_bound"].item()


def test_calibration_outputs_write_only_when_requested() -> None:
    paths = _toy_paths()
    config = ABMV4Config(start_year=1995, end_year=2016)
    _write_toy_inputs(paths, config)
    calibrator = EmissionsTransitionCalibrator(
        paths,
        start_year=1995,
        end_year=2016,
        random_search_iterations=3,
    )
    result = calibrator.run()

    assert not paths.emissions_calibration_report_path.exists()
    calibrator.write_outputs(result)
    assert paths.emissions_calibration_dataset_path.exists()
    assert paths.emissions_best_parameters_path.exists()
    assert paths.emissions_calibration_report_path.exists()


def test_calibration_report_includes_train_validation_and_baselines() -> None:
    paths = _toy_paths()
    config = ABMV4Config(start_year=1995, end_year=2016)
    _write_toy_inputs(paths, config)
    report = EmissionsTransitionCalibrator(
        paths,
        start_year=1995,
        end_year=2016,
        random_search_iterations=3,
    ).run().markdown

    assert "Train years" in report
    assert "Validation years" in report
    assert "sector_background_only" in report
    assert "legacy_raw_log" in report
