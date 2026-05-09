from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from src.abm_v3.ei_transition.models import (
    EITransitionModelSuite,
    build_expected_sign_table,
    build_prediction_frame,
)
from src.abm_v3.ei_transition.outputs import EITransitionOutputWriter
from src.abm_v3.paths import ABMV3Paths


def toy_paths() -> ABMV3Paths:
    root = Path("tmp") / "ei_transition_model_tests" / uuid4().hex[:8]
    root.mkdir(parents=True, exist_ok=True)
    return ABMV3Paths(project_root=root)


def toy_transition_panel() -> pd.DataFrame:
    rows = []
    sectors = ["Agriculture", "Manufacturing", "Services"]
    for node_index in range(6):
        sector = sectors[node_index % len(sectors)]
        country_sector = f"AAA | AAA | Industries | {sector} {node_index}"
        for year in range(1995, 2016):
            log_ei = 2.5 - 0.015 * (year - 1995) + 0.04 * node_index
            green_capability = 0.1 * node_index + 0.01 * (year - 1995)
            g_in = 0.2 + 0.02 * node_index
            g_out = 0.3 + 0.015 * (year - 1995)
            g_network = 0.5 * (g_in + g_out)
            general_complexity = 0.4 + 0.03 * node_index
            reduction = (
                0.02
                + 0.01 * log_ei
                + 0.02 * green_capability
                + 0.015 * g_network
                + 0.005 * general_complexity
            )
            ei = float(np.exp(log_ei))
            ei_next = float(ei * np.exp(-reduction))
            rows.append(
                {
                    "country_sector": country_sector,
                    "Year": year,
                    "Country": "AAA",
                    "Country_detail": "AAA",
                    "Category": "Industries",
                    "Sector": sector,
                    "EI": ei,
                    "EI_next": ei_next,
                    "log_EI": log_ei,
                    "log_EI_next": np.log(ei_next),
                    "ei_reduction_next": reduction,
                    "green_capability": green_capability,
                    "g_in": g_in,
                    "g_out": g_out,
                    "g_network": g_network,
                    "general_complexity": general_complexity,
                    "X": 100.0,
                    "X_observed": 100.0,
                    "sample_included": True,
                    "exclusion_reason": "",
                }
            )
    return pd.DataFrame(rows)


def test_economic_only_model_fits() -> None:
    panel = toy_transition_panel()

    result = EITransitionModelSuite().fit_one(
        panel,
        EITransitionModelSuite().model_specs()[0],
        train_end_year=2012,
        validation_start_year=2013,
        validation_end_year=2015,
    )

    assert result.spec.model_name == "economic_only"
    assert "log_EI" in result.coefficients["term"].tolist()


def test_green_transition_model_fits() -> None:
    panel = toy_transition_panel()

    result = EITransitionModelSuite().fit_one(
        panel,
        EITransitionModelSuite().model_specs()[1],
        train_end_year=2012,
        validation_start_year=2013,
        validation_end_year=2015,
    )

    terms = result.coefficients["term"].tolist()
    assert "green_capability" in terms
    assert "g_network" in terms
    assert "general_complexity" in terms


def test_network_robustness_model_fits() -> None:
    panel = toy_transition_panel()

    result = EITransitionModelSuite().fit_one(
        panel,
        EITransitionModelSuite().model_specs()[2],
        train_end_year=2012,
        validation_start_year=2013,
        validation_end_year=2015,
    )

    terms = result.coefficients["term"].tolist()
    assert "g_in" in terms
    assert "g_out" in terms


def test_predictions_compute_predicted_ei_next() -> None:
    panel = toy_transition_panel()
    fit_result = EITransitionModelSuite().fit_one(
        panel,
        EITransitionModelSuite().model_specs()[1],
        train_end_year=2012,
        validation_start_year=2013,
        validation_end_year=2015,
    )
    validation_rows = panel.loc[(panel["Year"] >= 2013) & (panel["Year"] <= 2015)].copy()

    predictions = build_prediction_frame(validation_rows, fit_result)

    expected = predictions["EI"] * np.exp(-predictions["predicted_ei_reduction_next"])
    assert np.allclose(predictions["predicted_EI_next"], expected)


def test_expected_sign_table_flags_matches() -> None:
    coefficients = pd.DataFrame(
        {
            "model_name": ["green_transition", "green_transition"],
            "term": ["log_EI", "g_network"],
            "coefficient": [0.2, -0.1],
        }
    )

    signs = build_expected_sign_table(coefficients)

    assert bool(signs.loc[signs["term"] == "log_EI", "matches_expected_sign"].iloc[0]) is True
    assert bool(signs.loc[signs["term"] == "g_network", "matches_expected_sign"].iloc[0]) is False


def test_model_suite_writes_outputs_to_ei_transition_folder() -> None:
    paths = toy_paths()
    panel = toy_transition_panel()
    output = EITransitionModelSuite().fit_all(
        panel,
        train_end_year=2012,
        validation_start_year=2013,
        validation_end_year=2015,
    )

    written = EITransitionOutputWriter(paths).write_fit_outputs(
        1995,
        2016,
        output["scores"],
        output["coefficients"],
        output["expected_signs"],
        output["predictions"],
    )

    assert Path(written["scores"]).parent == paths.ei_transition_diagnostics_dir
    assert Path(written["coefficients"]).parent == paths.ei_transition_models_dir
    assert Path(written["predictions"]).parent == paths.ei_transition_predictions_dir
    assert paths.ei_transition_model_scores_path(1995, 2016).exists()
