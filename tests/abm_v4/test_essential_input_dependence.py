from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import polars as pl
import pytest

from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.validation import EssentialInputDependenceDiagnostics


def _toy_paths() -> ABMV4Paths:
    return ABMV4Paths(project_root=Path("tmp") / "abm_v4_essential_input_tests" / uuid4().hex)


def _write_state(paths: ABMV4Paths) -> None:
    paths.inputs.mkdir(parents=True, exist_ok=True)
    rows = []
    nodes = [
        ("ELEC", "Electricity, Gas and Water", 100.0, 50.0),
        ("RAIL", "Transport", 80.0, 20.0),
        ("MANU", "Manufacturing", 120.0, 30.0),
        ("SERV", "Services", 200.0, 10.0),
        ("FOOD", "Food & Beverages", 90.0, 12.0),
    ]
    for country, sector, output, emissions in nodes:
        for year in [2000, 2001]:
            rows.append(
                {
                    "country_sector": f"{country} | {sector}",
                    "Country": country,
                    "Sector": sector,
                    "Year": year,
                    "ecosystem_label": "infrastructure" if country in {"ELEC", "RAIL"} else "other",
                    "X_observed": output,
                    "EI": emissions / output,
                    "emissions_observed": emissions,
                }
            )
    pl.DataFrame(rows).write_parquet(paths.state_panel_path(1995, 2016))


def _write_compact_supplier_data(paths: ABMV4Paths) -> None:
    paths.interim.mkdir(parents=True, exist_ok=True)
    relationships = [
        ("ELEC | Electricity, Gas and Water", "MANU | Manufacturing", 0.20),
        ("ELEC | Electricity, Gas and Water", "SERV | Services", 0.10),
        ("ELEC | Electricity, Gas and Water", "FOOD | Food & Beverages", 0.15),
        ("RAIL | Transport", "MANU | Manufacturing", 0.18),
        ("RAIL | Transport", "SERV | Services", 0.08),
        ("MANU | Manufacturing", "FOOD | Food & Beverages", 0.04),
    ]
    pl.DataFrame(
        {
            "supplier_country_sector": [r[0] for r in relationships],
            "buyer_country_sector": [r[1] for r in relationships],
            "supplier_type": ["historical"] * len(relationships),
            "candidate_sources": ["historical"] * len(relationships),
            "updated_weight": [r[2] for r in relationships],
            "choice_probability": [r[2] for r in relationships],
        }
    ).write_parquet(paths.supplier_updated_weights_path)
    pl.DataFrame(
        {
            "supplier_country_sector": [r[0] for r in relationships] + ["SERV | Services"],
            "buyer_country_sector": [r[1] for r in relationships] + ["MANU | Manufacturing"],
            "supplier_type": ["historical", "historical", "historical", "historical", "historical", "same_sector", "ecosystem"],
            "is_historical_candidate": [True, True, True, True, True, False, False],
            "is_same_sector_candidate": [False, False, False, False, False, True, False],
            "is_ecosystem_candidate": [False, False, False, False, False, False, True],
        }
    ).write_parquet(paths.supplier_opportunity_sets_path)
    hist = []
    for year in [2000, 2001]:
        for supplier, buyer, share in relationships:
            hist.append(
                {
                    "year": year,
                    "supplier_country_sector": supplier,
                    "buyer_country_sector": buyer,
                    "transaction_value": share * 100,
                    "historical_share": share,
                    "supplier_country": supplier.split(" | ")[0],
                    "supplier_sector": supplier.split(" | ", 1)[1],
                    "buyer_country": buyer.split(" | ")[0],
                    "buyer_sector": buyer.split(" | ", 1)[1],
                    "supplier_ecosystem_label": "infrastructure" if supplier.startswith(("ELEC", "RAIL")) else "other",
                    "buyer_ecosystem_label": "other",
                }
            )
    pl.DataFrame(hist).write_parquet(paths.historical_supplier_edges_path)


def _write_structural_panel(paths: ABMV4Paths) -> None:
    paths.validation.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "country_sector": [
                "ELEC | Electricity, Gas and Water",
                "RAIL | Transport",
                "MANU | Manufacturing",
                "SERV | Services",
                "FOOD | Food & Beverages",
            ],
            "Country": ["ELEC", "RAIL", "MANU", "SERV", "FOOD"],
            "Sector": ["Electricity, Gas and Water", "Transport", "Manufacturing", "Services", "Food & Beverages"],
            "ecosystem": ["infrastructure", "infrastructure", "other", "other", "other"],
            "electricity_like": [True, False, False, False, False],
            "cumulative_emissions_share": [0.4, 0.2, 0.2, 0.1, 0.1],
            "jump_frequency": [0.2, 0.1, 0.0, 0.0, 0.0],
            "share_frontier_gap_worsens_emissions_error": [0.8, 0.7, 0.1, 0.0, 0.0],
            "mean_contribution_to_aggregate_error_difference": [100.0, 80.0, 10.0, 5.0, 1.0],
            "needs_dampening_node": [True, True, False, False, False],
            "aggregate_sensitive_node": [True, True, False, False, False],
            "high_emissions_node": [True, True, False, False, False],
            "jump_prone_node": [True, True, False, False, False],
            "mean_brown_centrality": [0.8, 0.4, 0.2, 0.1, 0.1],
            "mean_log_EI_observed": [1.0, 0.2, 0.1, 0.0, 0.0],
        }
    ).write_parquet(paths.structural_signature_node_panel_path)


def _write_inputs(paths: ABMV4Paths) -> None:
    _write_state(paths)
    _write_compact_supplier_data(paths)
    _write_structural_panel(paths)


def test_supplier_buyer_dependence_panel_builds_from_compact_supplier_data() -> None:
    paths = _toy_paths()
    _write_inputs(paths)
    panel = EssentialInputDependenceDiagnostics(paths).build_supplier_buyer_dependence_panel()

    assert {"supplier_country_sector", "buyer_country_sector", "supplier_share_in_buyer_inputs"}.issubset(panel.columns)
    assert not panel.is_empty()


def test_buyer_reach_metrics_are_computed_correctly() -> None:
    paths = _toy_paths()
    _write_inputs(paths)
    diag = EssentialInputDependenceDiagnostics(paths)
    metrics = diag.compute_essential_input_metrics(diag.build_supplier_buyer_dependence_panel())
    elec = metrics.filter(pl.col("country_sector") == "ELEC | Electricity, Gas and Water").to_dicts()[0]

    assert elec["buyer_count"] == 3
    assert elec["buyer_sector_count"] == 3


def test_buyer_entropy_and_hhi_are_computed_correctly() -> None:
    paths = _toy_paths()
    _write_inputs(paths)
    diag = EssentialInputDependenceDiagnostics(paths)
    metrics = diag.compute_essential_input_metrics(diag.build_supplier_buyer_dependence_panel())
    elec = metrics.filter(pl.col("country_sector") == "ELEC | Electricity, Gas and Water").to_dicts()[0]

    assert elec["buyer_hhi"] > 0
    assert elec["buyer_entropy"] > 0


def test_input_universality_metrics_are_computed_correctly() -> None:
    paths = _toy_paths()
    _write_inputs(paths)
    diag = EssentialInputDependenceDiagnostics(paths)
    metrics = diag.compute_essential_input_metrics(diag.build_supplier_buyer_dependence_panel())
    elec = metrics.filter(pl.col("country_sector") == "ELEC | Electricity, Gas and Water").to_dicts()[0]

    assert elec["share_of_all_buyers"] == pytest.approx(1.0)
    assert elec["buyer_sector_coverage"] == pytest.approx(1.0)


def test_buyer_dependence_thresholds_are_computed_correctly() -> None:
    paths = _toy_paths()
    _write_inputs(paths)
    diag = EssentialInputDependenceDiagnostics(paths)
    metrics = diag.compute_essential_input_metrics(diag.build_supplier_buyer_dependence_panel())
    elec = metrics.filter(pl.col("country_sector") == "ELEC | Electricity, Gas and Water").to_dicts()[0]

    assert elec["buyers_dependency_above_1pct"] == 3
    assert elec["buyers_dependency_above_10pct"] == 3


def test_downstream_output_and_emissions_exposure_are_computed() -> None:
    paths = _toy_paths()
    _write_inputs(paths)
    diag = EssentialInputDependenceDiagnostics(paths)
    metrics = diag.compute_essential_input_metrics(diag.build_supplier_buyer_dependence_panel())
    elec = metrics.filter(pl.col("country_sector") == "ELEC | Electricity, Gas and Water").to_dicts()[0]

    assert elec["downstream_output_exposure"] > 0
    assert elec["downstream_emissions_exposure"] > 0


def test_electricity_contrast_detects_broad_buyer_reach() -> None:
    paths = _toy_paths()
    _write_inputs(paths)
    diag = EssentialInputDependenceDiagnostics(paths)
    metrics = diag.compute_essential_input_metrics(diag.build_supplier_buyer_dependence_panel())
    contrast = diag.contrast_electricity_vs_non_electricity(metrics)

    assert contrast.filter(pl.col("metric") == "buyer_count")["standardized_difference"].item() > 0


def test_dependence_vs_symptom_comparison_detects_high_correlation() -> None:
    diag = EssentialInputDependenceDiagnostics(_toy_paths())
    node = pl.DataFrame(
        {
            "essential_input_score_diagnostic": [0.0, 0.5, 1.0],
            "structural_dependence_score_diagnostic": [0.0, 0.5, 1.0],
            "systemic_dependence_score_diagnostic": [0.0, 0.5, 1.0],
            "mean_log_EI_observed": [0.0, 0.5, 1.0],
            "cumulative_emissions_share": [0.1, 0.2, 0.3],
        }
    )
    comparison = diag.compare_dependence_to_symptom_metrics(node)

    assert comparison["correlation"].max() == pytest.approx(1.0)


def test_dependence_based_lookalike_detection_identifies_synthetic_lookalike() -> None:
    diag = EssentialInputDependenceDiagnostics(_toy_paths())
    node = pl.DataFrame(
        {
            "country_sector": ["elec", "rail", "service"],
            "Country": ["ELEC", "RAIL", "SERV"],
            "Sector": ["Electricity", "Rail transport", "Services"],
            "ecosystem": ["infra", "infra", "service"],
            "electricity_like": [True, False, False],
            "structural_dependence_score_diagnostic": [1.0, 0.9, 0.1],
            "essential_input_score_diagnostic": [1.0, 0.8, 0.1],
            "low_substitutability_score_diagnostic": [0.8, 0.7, 0.1],
            "systemic_dependence_score_diagnostic": [1.0, 0.9, 0.1],
            "buyer_count": [10, 9, 1],
            "buyer_sector_coverage": [1.0, 0.9, 0.1],
            "buyer_ecosystem_coverage": [1.0, 0.9, 0.1],
            "downstream_output_exposure": [100.0, 90.0, 1.0],
            "downstream_emissions_exposure": [50.0, 40.0, 1.0],
            "cumulative_emissions_share": [0.4, 0.3, 0.1],
            "jump_frequency": [0.2, 0.1, 0.0],
            "share_frontier_gap_worsens_emissions_error": [0.8, 0.7, 0.0],
        }
    )
    lookalikes = diag.identify_structural_dependence_lookalikes(node)

    assert lookalikes["country_sector"].item(0) == "rail"


def test_recommendation_selects_structural_dependence_when_strong_and_not_redundant() -> None:
    diag = EssentialInputDependenceDiagnostics(_toy_paths())
    node = pl.DataFrame({"x": [1]})
    screening = pl.DataFrame(
        {
            "label_name": ["electricity_like", "needs_dampening_node"],
            "metric": ["structural_dependence_score_diagnostic", "structural_dependence_score_diagnostic"],
            "separation_score": [1.0, 1.0],
        }
    )
    lookalikes = pl.DataFrame({"structural_dependence_score": [0.8]})
    proxies = pl.DataFrame({"recommended_for_phase23": [True]})
    symptom = pl.DataFrame({"correlation": [0.2]})

    assert diag.build_recommendation(node, screening, lookalikes, proxies, symptom)["recommendation"].item(0) == "build_structural_dependence_dampener"


def test_recommendation_selects_insufficient_when_redundant_or_unavailable() -> None:
    diag = EssentialInputDependenceDiagnostics(_toy_paths())
    node = pl.DataFrame({"x": [1]})
    screening = pl.DataFrame({"label_name": ["electricity_like"], "separation_score": [0.5]})
    proxies = pl.DataFrame({"recommended_for_phase23": [False]})
    symptom = pl.DataFrame({"correlation": [0.95]})

    assert diag.build_recommendation(node, screening, pl.DataFrame(), proxies, symptom)["recommendation"].item(0) == "dependence_metrics_insufficient"


def test_outputs_are_written_only_with_explicit_output_flag() -> None:
    paths = _toy_paths()
    _write_inputs(paths)
    diag = EssentialInputDependenceDiagnostics(paths)
    result = diag.run()

    assert not paths.essential_input_dependence_report_path.exists()
    diag.write_outputs(result)
    assert paths.essential_input_node_metrics_path.exists()
    assert paths.essential_input_dependence_report_path.exists()


def test_missing_compact_supplier_data_fails_clearly() -> None:
    paths = _toy_paths()
    _write_state(paths)
    with pytest.raises(FileNotFoundError, match="No compact supplier-buyer dependence panel"):
        EssentialInputDependenceDiagnostics(paths).run()
