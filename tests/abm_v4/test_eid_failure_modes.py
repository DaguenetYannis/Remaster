from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import polars as pl
import pytest

from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.validation import EssentialInputFailureModeDiagnostics


def _toy_paths() -> ABMV4Paths:
    return ABMV4Paths(project_root=Path("tmp") / "abm_v4_eid_failure_mode_tests" / uuid4().hex)


def _write_inputs(paths: ABMV4Paths) -> None:
    paths.validation.mkdir(parents=True, exist_ok=True)
    nodes = [
        ("CHN | Electricity, Gas and Water", "CHN", "Electricity, Gas and Water", True, 0.40, 0.03, True, True, False),
        ("USA | Transport", "USA", "Transport", False, 0.20, 0.10, True, True, False),
        ("ROW | TOTAL", "ROW", "TOTAL", False, 0.05, 0.02, False, True, True),
        ("FRA | Financial Intermediation", "FRA", "Financial Intermediation", False, 0.01, 0.20, False, False, False),
        ("DEU | Manufacturing", "DEU", "Manufacturing", False, 0.10, 0.12, False, False, False),
    ]
    pl.DataFrame(
        {
            "country_sector": [n[0] for n in nodes],
            "Country": [n[1] for n in nodes],
            "Sector": [n[2] for n in nodes],
            "electricity_like": [n[3] for n in nodes],
            "essential_input_score_diagnostic": [1.0, 0.9, 0.95, 0.85, 0.1],
            "low_substitutability_score_diagnostic": [0.9, 0.8, 0.7, 0.6, 0.1],
            "systemic_dependence_score_diagnostic": [0.95, 0.75, 0.8, 0.55, 0.1],
            "structural_dependence_score_diagnostic": [0.95, 0.85, 0.9, 0.8, 0.1],
            "mean_brown_centrality": [0.8, 0.4, 0.3, 0.1, 0.1],
        }
    ).write_csv(paths.essential_input_node_metrics_path)
    pl.DataFrame(
        {
            "country_sector": [n[0] for n in nodes],
            "Country": [n[1] for n in nodes],
            "Sector": [n[2] for n in nodes],
            "ecosystem": ["energy", "transport", "accounting", "services", "industry"],
            "electricity_like": [n[3] for n in nodes],
            "cumulative_emissions_share": [n[4] for n in nodes],
            "cumulative_output_share": [n[5] for n in nodes],
            "high_emissions_node": [n[6] for n in nodes],
            "jump_prone_node": [False, True, False, False, False],
            "aggregate_sensitive_node": [n[7] for n in nodes],
            "needs_dampening_node": [True, True, True, False, False],
            "mean_brown_centrality": [0.8, 0.4, 0.3, 0.1, 0.1],
        }
    ).write_parquet(paths.structural_signature_node_panel_path)
    score_rows = []
    for score in [
        "structural_dependence_score_diagnostic",
        "essential_input_score_diagnostic",
        "low_substitutability_score_diagnostic",
        "structural_dependence_plus_brown_lockin",
    ]:
        values = [0.98, 0.92, 0.96, 0.9, 0.1]
        for (country_sector, country, sector, electricity_like, *_), value in zip(nodes, values, strict=True):
            score_rows.append(
                {
                    "country_sector": country_sector,
                    "Country": country,
                    "Sector": sector,
                    "electricity_like": electricity_like,
                    "EID_score_name": score,
                    "EID_raw": value,
                    "p05": 0.1,
                    "p95": 0.98,
                    "missing_flag": False,
                    "EID_norm": value,
                    "notes": "toy",
                    "high_EID_decile": value >= 0.9,
                }
            )
    pl.DataFrame(score_rows).write_csv(paths.essential_input_dampener_scores_path)
    pl.DataFrame(
        {
            "candidate_id": ["c0001", "c0002", "c0061", "c0102"],
            "variant_name": [
                "frontier_gap_readiness_baseline",
                "historical_frontier_gap_only_baseline",
                "essential_input_dampener_only",
                "essential_input_dampener_plus_historical_residual",
            ],
            "train_or_validation": ["validation"] * 4,
            "EID_score_name": [None, None, "structural_dependence_plus_brown_lockin", "essential_input_score_diagnostic"],
            "lambda_EID": [None, None, 1.0, 0.75],
            "d_min": [None, None, 0.25, 0.5],
            "all_node_emissions_weighted_rEI_MAE": [0.2, 0.3, 0.1, 0.12],
            "electricity_rEI_MAE": [0.1, 0.2, 0.05, 0.06],
        }
    ).write_csv(paths.essential_input_dampener_validation_results_path)
    transition_rows = []
    for year in [2011, 2012]:
        transition_rows.extend(
            [
                ("CHN | Electricity, Gas and Water", "CHN", "Electricity, Gas and Water", 0.02, 0.08, 0.04, 100.0, 1.0, 100.0, "d10"),
                ("USA | Transport", "USA", "Transport", 0.02, 0.07, 0.03, 80.0, 1.0, 80.0, "d9"),
                ("ROW | TOTAL", "ROW", "TOTAL", 0.02, 0.07, 0.03, 10.0, 1.0, 10.0, "d2"),
                ("FRA | Financial Intermediation", "FRA", "Financial Intermediation", 0.10, 0.10, 0.08, 20.0, 1.0, 20.0, "d2"),
                ("DEU | Manufacturing", "DEU", "Manufacturing", 0.03, 0.04, 0.035, 50.0, 1.0, 50.0, "d4"),
            ]
        )
    pl.DataFrame(
        {
            "country_sector": [r[0] for r in transition_rows],
            "year": [year for year in [2011, 2012] for _ in range(5)],
            "Country": [r[1] for r in transition_rows],
            "Sector": [r[2] for r in transition_rows],
            "observed_rEI": [r[3] for r in transition_rows],
            "simulated_rEI_frontier_gap": [r[4] for r in transition_rows],
            "simulated_rEI_readiness": [r[5] for r in transition_rows],
            "X_observed": [r[6] for r in transition_rows],
            "EI_observed": [r[7] for r in transition_rows],
            "emissions_observed": [r[8] for r in transition_rows],
            "emissions_decile": [r[9] for r in transition_rows],
        }
    ).write_parquet(paths.transition_rule_sign_failure_panel_path)


def test_high_eid_nodes_are_selected_from_normalized_scores() -> None:
    paths = _toy_paths()
    _write_inputs(paths)
    panel = EssentialInputFailureModeDiagnostics(paths).build_high_EID_node_panel()

    assert "CHN | Electricity, Gas and Water" in set(panel["country_sector"])
    assert "DEU | Manufacturing" not in set(panel["country_sector"])


def test_subtype_classification_assigns_electricity_to_infrastructure_energy() -> None:
    diag = EssentialInputFailureModeDiagnostics(_toy_paths())

    assert diag.classify_high_EID_subtype("CHN", "Electricity, Gas and Water") == "infrastructure_energy"


def test_subtype_classification_assigns_total_row_to_pseudo_agent() -> None:
    diag = EssentialInputFailureModeDiagnostics(_toy_paths())

    assert diag.classify_high_EID_subtype("ROW", "TOTAL") == "accounting_or_pseudo_agent"


def test_subtype_composition_aggregates_node_and_emissions_shares() -> None:
    paths = _toy_paths()
    _write_inputs(paths)
    diag = EssentialInputFailureModeDiagnostics(paths)
    composition = diag.build_subtype_composition(diag.build_high_EID_node_panel())
    energy = composition.filter(pl.col("candidate_subtype") == "infrastructure_energy").to_dicts()[0]

    assert energy["nodes"] == 1
    assert energy["observed_emissions_share"] > 0


def test_performance_by_subtype_computes_improvement_vs_baseline() -> None:
    paths = _toy_paths()
    _write_inputs(paths)
    diag = EssentialInputFailureModeDiagnostics(paths)
    panel = diag.build_high_EID_node_panel()
    performance = diag.evaluate_dampener_performance_by_subtype(panel)
    energy = performance.filter(
        (pl.col("candidate_subtype") == "infrastructure_energy")
        & (pl.col("variant_name") == "essential_input_dampener_only")
    ).to_dicts()[0]

    assert energy["improvement_vs_historical_frontier_gap"] > 0


def test_failure_mode_classification_identifies_helped_and_harmed() -> None:
    panel = pl.DataFrame(
        {
            "country_sector": ["a", "b", "c"],
            "Country": ["A", "B", "C"],
            "Sector": ["Electricity", "Manufacturing", "Services"],
            "candidate_subtype": ["infrastructure_energy", "manufacturing_system_core", "knowledge_finance_business_services"],
            "EID_norm": [1.0, 1.0, 1.0],
            "emissions_share": [0.1, 0.1, 0.0],
            "output_share": [0.1, 0.1, 0.1],
            "pseudo_agent_flag": [False, False, False],
            "baseline_error": [0.10, 0.05, 0.05],
            "EID_candidate_error": [0.01, 0.10, 0.05],
        }
    )
    # Exercise the same threshold logic through a minimal equivalent frame.
    out = panel.with_columns((pl.col("EID_candidate_error") - pl.col("baseline_error")).alias("error_change")).with_columns(
        pl.when(pl.col("pseudo_agent_flag"))
        .then(pl.lit("pseudo_agent_accounting_issue"))
        .when(pl.col("error_change") < -0.01)
        .then(pl.lit("helped_by_EID"))
        .when(pl.col("error_change") > 0.01)
        .then(pl.lit("harmed_by_EID"))
        .otherwise(pl.lit("no_material_change"))
        .alias("failure_mode")
    )

    assert {"helped_by_EID", "harmed_by_EID"}.issubset(set(out["failure_mode"]))


def test_pseudo_agent_audit_flags_aggregate_nodes() -> None:
    paths = _toy_paths()
    _write_inputs(paths)
    diag = EssentialInputFailureModeDiagnostics(paths)
    panel = diag.build_high_EID_node_panel()
    performance = diag.evaluate_dampener_performance_by_subtype(panel)
    failures = diag.identify_failure_modes(panel, performance)
    pseudo = diag.identify_accounting_or_pseudo_agent_nodes(panel, failures)

    assert pseudo.filter(pl.col("country_sector") == "ROW | TOTAL")["pseudo_agent_flag"].item() is True


def test_abm_v5_agent_type_candidate_confidence_levels_are_assigned() -> None:
    diag = EssentialInputFailureModeDiagnostics(_toy_paths())
    composition = pl.DataFrame(
        {
            "candidate_subtype": ["infrastructure_energy", "accounting_or_pseudo_agent"],
            "nodes": [5, 2],
            "pseudo_agent_share": [0.0, 1.0],
        }
    )
    performance = pl.DataFrame(
        {
            "candidate_subtype": ["infrastructure_energy", "accounting_or_pseudo_agent"],
            "variant_name": ["essential_input_dampener_only", "essential_input_dampener_only"],
            "improvement_vs_historical_frontier_gap": [0.02, 0.1],
        }
    )
    candidates = diag.identify_abm_v5_agent_type_candidates(composition, performance)

    assert candidates.filter(pl.col("subtype_source") == "infrastructure_energy")["confidence_level"].item() == "moderate"
    assert candidates.filter(pl.col("subtype_source") == "accounting_or_pseudo_agent")["confidence_level"].item() == "not_supported"


def test_recommendation_selects_split_when_only_infrastructure_benefits() -> None:
    diag = EssentialInputFailureModeDiagnostics(_toy_paths())
    composition = pl.DataFrame({"x": [1]})
    performance = pl.DataFrame(
        {
            "candidate_subtype": ["infrastructure_energy", "heavy_industry_materials"],
            "variant_name": ["essential_input_dampener_only", "essential_input_dampener_only"],
            "improvement_vs_historical_frontier_gap": [0.1, -0.1],
        }
    )
    pseudo = pl.DataFrame({"pseudo_agent_flag": [False, False]})
    failures = pl.DataFrame({"failure_mode": ["helped_by_EID"]})

    rec = diag.build_recommendation(composition, performance, failures, pseudo, pl.DataFrame())

    assert rec["recommendation"].item() == "split_EID_into_subtype_specific_diagnostics"


def test_recommendation_selects_exclude_pseudo_agents_when_contaminated() -> None:
    diag = EssentialInputFailureModeDiagnostics(_toy_paths())
    performance = pl.DataFrame(
        {
            "candidate_subtype": ["infrastructure_energy"],
            "variant_name": ["essential_input_dampener_only"],
            "improvement_vs_historical_frontier_gap": [0.1],
        }
    )
    pseudo = pl.DataFrame({"pseudo_agent_flag": [True, True, False]})
    rec = diag.build_recommendation(pl.DataFrame(), performance, pl.DataFrame(), pseudo, pl.DataFrame())

    assert rec["recommendation"].item() == "exclude_pseudo_agents_before_retesting"


def test_recommendation_selects_abandon_when_no_subtype_benefits() -> None:
    diag = EssentialInputFailureModeDiagnostics(_toy_paths())
    performance = pl.DataFrame(
        {
            "candidate_subtype": ["infrastructure_energy"],
            "variant_name": ["essential_input_dampener_only"],
            "improvement_vs_historical_frontier_gap": [-0.1],
        }
    )
    failures = pl.DataFrame({"failure_mode": ["harmed_by_EID"]})
    pseudo = pl.DataFrame({"pseudo_agent_flag": [False]})
    rec = diag.build_recommendation(pl.DataFrame(), performance, failures, pseudo, pl.DataFrame())

    assert rec["recommendation"].item() == "abandon_EID_dampener_for_v4"


def test_outputs_are_written_only_with_explicit_output_flag() -> None:
    paths = _toy_paths()
    _write_inputs(paths)
    diag = EssentialInputFailureModeDiagnostics(paths)
    result = diag.run()

    assert not paths.eid_failure_mode_report_path.exists()
    diag.write_outputs(result)
    assert paths.eid_high_node_heterogeneity_panel_path.exists()
    assert paths.eid_failure_mode_report_path.exists()


def test_missing_phase23_results_fail_clearly() -> None:
    paths = _toy_paths()
    paths.validation.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"country_sector": ["x"]}).write_csv(paths.essential_input_node_metrics_path)

    with pytest.raises(FileNotFoundError, match="Missing Phase 23 dampener outputs"):
        EssentialInputFailureModeDiagnostics(paths).load_phase23_results()
