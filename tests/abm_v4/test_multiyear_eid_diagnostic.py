from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import polars as pl
import pytest

from src.abm_v4.config import ABMV4Config
from src.abm_v4.emissions import HISTORICAL_FRONTIER_GAP_EID_DIAGNOSTIC_MODE
from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.simulation import MultiYearBaseSimulator
from src.abm_v4.validation import MultiYearEIDDiagnosticValidator


def _toy_paths() -> ABMV4Paths:
    return ABMV4Paths(project_root=Path("tmp") / "abm_v4_multiyear_eid_tests" / uuid4().hex)


def _write_state_and_scores(paths: ABMV4Paths) -> None:
    paths.inputs.mkdir(parents=True, exist_ok=True)
    paths.validation.mkdir(parents=True, exist_ok=True)
    rows = []
    for year in [2015, 2016]:
        rows.extend(
            [
                {
                    "country_sector": "CHN | Electricity, Gas and Water",
                    "Year": year,
                    "Country": "China",
                    "Sector": "Electricity, Gas and Water",
                    "X_observed": 100.0,
                    "EI": 1.0 if year == 2015 else 0.98,
                    "emissions_observed": 100.0 if year == 2015 else 98.0,
                    "general_capability": 0.5,
                    "green_capability": 0.5,
                    "network_green_exposure": 0.0,
                    "brown_centrality": 0.0,
                },
                {
                    "country_sector": "DEU | Manufacturing",
                    "Year": year,
                    "Country": "DEU",
                    "Sector": "Manufacturing",
                    "X_observed": 50.0,
                    "EI": 0.5 if year == 2015 else 0.49,
                    "emissions_observed": 25.0 if year == 2015 else 24.5,
                    "general_capability": 0.5,
                    "green_capability": 0.5,
                    "network_green_exposure": 0.0,
                    "brown_centrality": 0.0,
                },
            ]
        )
    pl.DataFrame(rows).write_parquet(paths.state_panel_path(2015, 2016))
    pl.DataFrame(
        {
            "country_sector": ["CHN | Electricity, Gas and Water"],
            "Country": ["China"],
            "Sector": ["Electricity, Gas and Water"],
            "electricity_like": [True],
            "EID_score_name": ["structural_dependence_plus_brown_lockin"],
            "EID_raw": [1.0],
            "p05": [0.0],
            "p95": [1.0],
            "missing_flag": [False],
            "EID_norm": [1.0],
            "notes": ["toy"],
            "high_EID_decile": [True],
        }
    ).write_csv(paths.essential_input_dampener_scores_path)


def _simulation_frame(variant: str, eid_multiplier: float = 1.0) -> pl.DataFrame:
    rows = []
    nodes = [
        ("CHN | Electricity, Gas and Water", "China", "Electricity, Gas and Water", True, "infrastructure_energy", False, 100.0),
        ("USA | Transport", "USA", "Transport", False, "transport_logistics_infrastructure", False, 80.0),
        ("ROW | TOTAL", "ROW", "TOTAL", False, "accounting_or_pseudo_agent", True, 10.0),
    ]
    for year in [2015, 2016]:
        for country_sector, country, sector, electricity, subtype, pseudo, emissions in nodes:
            observed_ei = 1.0 if year == 2015 else 0.9
            if variant == "historical_frontier_gap_only":
                sim_ei = 1.0 if year == 2015 else 0.8
            elif variant == "historical_frontier_gap_EID_diagnostic":
                sim_ei = 1.0 if year == 2015 else 0.85 * eid_multiplier
            else:
                sim_ei = 1.0 if year == 2015 else 0.82
            rows.append(
                {
                    "country_sector": country_sector,
                    "year": year,
                    "Country": country,
                    "Sector": sector,
                    "X_observed": emissions,
                    "EI_observed": observed_ei,
                    "EI_sim": sim_ei,
                    "emissions_observed": emissions * observed_ei,
                    "emissions_sim": emissions * sim_ei,
                    "rEI_used": 0.1,
                    "ei_gap": 0.5,
                    "EID_norm": 1.0 if country_sector != "ROW | TOTAL" else 0.95,
                    "D_EID": 0.25,
                    "EID_missing_flag": False,
                    "EID_fallback_flag": False,
                    "electricity_like": electricity,
                    "pseudo_agent_flag": pseudo,
                    "candidate_subtype": subtype,
                    "emissions_share": 0.1,
                    "output_share": 0.1,
                }
            )
    return pl.DataFrame(rows)


def _write_three_variant_outputs(paths: ABMV4Paths, eid_multiplier: float = 1.0) -> None:
    paths.simulations.mkdir(parents=True, exist_ok=True)
    paths.validation.mkdir(parents=True, exist_ok=True)
    _simulation_frame("frontier_gap_readiness").write_parquet(paths.base_multiyear_state_panel_path)
    _simulation_frame("historical_frontier_gap_only").write_parquet(paths.base_multiyear_state_panel_historical_frontier_gap_path)
    _simulation_frame("historical_frontier_gap_EID_diagnostic", eid_multiplier=eid_multiplier).write_parquet(paths.base_multiyear_state_panel_EID_diagnostic_path)
    pl.DataFrame(
        {
            "country_sector": ["CHN | Electricity, Gas and Water", "USA | Transport", "ROW | TOTAL"],
            "Country": ["China", "USA", "ROW"],
            "Sector": ["Electricity, Gas and Water", "Transport", "TOTAL"],
            "ecosystem": ["energy", "transport", "accounting"],
            "high_EID_definition": ["structural_dependence_plus_brown_lockin"] * 3,
            "high_EID_flag": [True, True, True],
            "EID_score_name": ["structural_dependence_plus_brown_lockin"] * 3,
            "EID_norm": [1.0, 1.0, 0.95],
            "electricity_like": [True, False, False],
            "candidate_subtype": ["infrastructure_energy", "transport_logistics_infrastructure", "accounting_or_pseudo_agent"],
            "pseudo_agent_flag": [False, False, True],
            "emissions_share": [0.5, 0.4, 0.1],
            "output_share": [0.3, 0.6, 0.1],
        }
    ).write_csv(paths.eid_high_node_heterogeneity_panel_path)


def test_eid_diagnostic_transition_mode_computes_d_eid_correctly() -> None:
    paths = _toy_paths()
    _write_state_and_scores(paths)
    sim = MultiYearBaseSimulator(
        paths,
        ABMV4Config(start_year=2015, end_year=2016),
        emissions_transition_mode=HISTORICAL_FRONTIER_GAP_EID_DIAGNOSTIC_MODE,
    )
    panel = pl.DataFrame({"country_sector": ["CHN | Electricity, Gas and Water"], "EID_norm": [1.0]})

    out = sim._add_eid_diagnostic_scores(panel)

    assert out["D_EID"].item() == pytest.approx(0.25)


def test_eid_fallback_is_reported_only_for_missing_scores() -> None:
    paths = _toy_paths()
    _write_state_and_scores(paths)
    sim = MultiYearBaseSimulator(paths, ABMV4Config(start_year=2015, end_year=2016), emissions_transition_mode=HISTORICAL_FRONTIER_GAP_EID_DIAGNOSTIC_MODE)
    panel = pl.DataFrame({"country_sector": ["missing node"]})

    out = sim._add_eid_diagnostic_scores(panel)

    assert out["D_EID"].item() == pytest.approx(1.0)
    assert out["EID_fallback_flag"].item() is True


def test_eid_diagnostic_simulation_writes_separate_paths() -> None:
    paths = _toy_paths()
    _write_state_and_scores(paths)
    sim = MultiYearBaseSimulator(paths, ABMV4Config(start_year=2015, end_year=2016), emissions_transition_mode=HISTORICAL_FRONTIER_GAP_EID_DIAGNOSTIC_MODE)
    result = sim.run()
    sim.write_outputs(result)

    assert paths.base_multiyear_state_panel_EID_diagnostic_path.exists()
    assert paths.base_multiyear_summary_panel_EID_diagnostic_path.exists()


def test_eid_diagnostic_mode_does_not_overwrite_base_outputs() -> None:
    paths = _toy_paths()
    _write_state_and_scores(paths)
    paths.simulations.mkdir(parents=True, exist_ok=True)
    paths.base_multiyear_state_panel_path.write_text("base", encoding="utf-8")
    sim = MultiYearBaseSimulator(paths, ABMV4Config(start_year=2015, end_year=2016), emissions_transition_mode=HISTORICAL_FRONTIER_GAP_EID_DIAGNOSTIC_MODE)
    sim.write_outputs(sim.run())

    assert paths.base_multiyear_state_panel_path.read_text(encoding="utf-8") == "base"


def test_validator_outputs_are_written_only_with_explicit_output_flag() -> None:
    paths = _toy_paths()
    _write_three_variant_outputs(paths)
    validator = MultiYearEIDDiagnosticValidator(paths)
    result = validator.run()

    assert not paths.multiyear_EID_diagnostic_report_path.exists()
    validator.write_outputs(result)
    assert paths.multiyear_EID_diagnostic_comparison_path.exists()
    assert paths.multiyear_EID_diagnostic_report_path.exists()


def test_validation_comparison_includes_required_three_variants() -> None:
    paths = _toy_paths()
    _write_three_variant_outputs(paths)
    comparison = MultiYearEIDDiagnosticValidator(paths).run().comparison

    assert {
        "frontier_gap_readiness",
        "historical_frontier_gap_only",
        "historical_frontier_gap_EID_diagnostic",
    }.issubset(set(comparison["model_variant"]))


def test_subtype_level_validation_computes_improvements() -> None:
    paths = _toy_paths()
    _write_three_variant_outputs(paths)
    subtype = MultiYearEIDDiagnosticValidator(paths).run().by_subtype
    row = subtype.filter(
        (pl.col("model_variant") == "historical_frontier_gap_EID_diagnostic")
        & (pl.col("candidate_subtype") == "infrastructure_energy")
    ).to_dicts()[0]

    assert "improvement_vs_historical_frontier_gap_only" in row


def test_pseudo_agent_sensitivity_excludes_pseudo_nodes() -> None:
    paths = _toy_paths()
    _write_three_variant_outputs(paths)
    pseudo = MultiYearEIDDiagnosticValidator(paths).run().pseudo_agent_sensitivity
    including = pseudo.filter(pl.col("pseudo_agent_scope") == "including_pseudo_agents")["rows"].max()
    excluding = pseudo.filter(pl.col("pseudo_agent_scope") == "excluding_pseudo_agents")["rows"].max()

    assert excluding < including


def test_mechanism_audit_computes_gap_closure_reduction() -> None:
    paths = _toy_paths()
    _write_three_variant_outputs(paths)
    mechanism = MultiYearEIDDiagnosticValidator(paths).run().mechanism_audit

    assert "mean_gap_closure_reduction" in mechanism.columns


def test_recommendation_selects_promote_when_all_criteria_pass() -> None:
    validator = MultiYearEIDDiagnosticValidator(_toy_paths())
    comparison = pl.DataFrame(
        {
            "model_variant": ["historical_frontier_gap_only", "historical_frontier_gap_EID_diagnostic"],
            "unweighted_rEI_MAE": [1.0, 0.9],
            "emissions_weighted_rEI_MAE": [1.0, 0.9],
            "wrong_sign_share": [0.4, 0.39],
            "mean_yearly_aggregate_emissions_pct_error": [1.0, 0.8],
            "electricity_rEI_MAE": [1.0, 0.8],
            "china_electricity_rEI_MAE": [1.0, 0.8],
        }
    )
    subtype = pl.DataFrame(
        {
            "model_variant": ["historical_frontier_gap_EID_diagnostic"],
            "candidate_subtype": ["infrastructure_energy"],
            "improvement_vs_historical_frontier_gap_only": [0.1],
            "material_worsening_flag": [False],
        }
    )
    pseudo = pl.DataFrame({"pseudo_agent_scope": [], "model_variant": [], "emissions_weighted_rEI_MAE": []})

    rec = validator.build_recommendation(comparison, subtype, pseudo)

    assert rec["recommendation"].item() == "promote_EID_to_provisional_base_candidate"


def test_recommendation_selects_split_when_some_subtypes_worsen() -> None:
    validator = MultiYearEIDDiagnosticValidator(_toy_paths())
    comparison = pl.DataFrame(
        {
            "model_variant": ["historical_frontier_gap_only", "historical_frontier_gap_EID_diagnostic"],
            "unweighted_rEI_MAE": [1.0, 0.99],
            "emissions_weighted_rEI_MAE": [1.0, 0.99],
            "wrong_sign_share": [0.4, 0.4],
            "mean_yearly_aggregate_emissions_pct_error": [1.0, 1.0],
            "electricity_rEI_MAE": [1.0, 1.0],
            "china_electricity_rEI_MAE": [1.0, 1.0],
        }
    )
    subtype = pl.DataFrame(
        {
            "model_variant": ["historical_frontier_gap_EID_diagnostic", "historical_frontier_gap_EID_diagnostic"],
            "candidate_subtype": ["infrastructure_energy", "heavy_industry_materials"],
            "improvement_vs_historical_frontier_gap_only": [0.1, -0.1],
            "material_worsening_flag": [False, True],
        }
    )

    rec = validator.build_recommendation(comparison, subtype, pl.DataFrame())

    assert rec["recommendation"].item() == "split_EID_by_subtype_before_integration"


def test_recommendation_selects_exclude_pseudo_agents_when_dependent() -> None:
    validator = MultiYearEIDDiagnosticValidator(_toy_paths())
    comparison = pl.DataFrame(
        {
            "model_variant": ["historical_frontier_gap_only", "historical_frontier_gap_EID_diagnostic"],
            "unweighted_rEI_MAE": [1.0, 0.99],
            "emissions_weighted_rEI_MAE": [1.0, 0.99],
            "wrong_sign_share": [0.4, 0.4],
            "mean_yearly_aggregate_emissions_pct_error": [1.0, 1.0],
            "electricity_rEI_MAE": [1.0, 1.0],
            "china_electricity_rEI_MAE": [1.0, 1.0],
        }
    )
    subtype = pl.DataFrame({"model_variant": ["historical_frontier_gap_EID_diagnostic"], "candidate_subtype": ["infrastructure_energy"], "improvement_vs_historical_frontier_gap_only": [0.1], "material_worsening_flag": [False]})
    pseudo = pl.DataFrame(
        {
            "pseudo_agent_scope": ["including_pseudo_agents", "including_pseudo_agents", "excluding_pseudo_agents", "excluding_pseudo_agents"],
            "model_variant": ["historical_frontier_gap_only", "historical_frontier_gap_EID_diagnostic"] * 2,
            "emissions_weighted_rEI_MAE": [1.0, 0.8, 1.0, 1.1],
        }
    )

    rec = validator.build_recommendation(comparison, subtype, pseudo)

    assert rec["recommendation"].item() == "exclude_pseudo_agents_and_retest"


def test_recommendation_selects_reject_when_material_worsening_occurs() -> None:
    validator = MultiYearEIDDiagnosticValidator(_toy_paths())
    comparison = pl.DataFrame(
        {
            "model_variant": ["historical_frontier_gap_only", "historical_frontier_gap_EID_diagnostic"],
            "unweighted_rEI_MAE": [1.0, 1.1],
            "emissions_weighted_rEI_MAE": [1.0, 1.1],
            "wrong_sign_share": [0.4, 0.5],
            "mean_yearly_aggregate_emissions_pct_error": [1.0, 1.1],
            "electricity_rEI_MAE": [1.0, 0.9],
            "china_electricity_rEI_MAE": [1.0, 0.9],
        }
    )

    rec = validator.build_recommendation(comparison, pl.DataFrame(), pl.DataFrame())

    assert rec["recommendation"].item() == "reject_EID_for_v4"


def test_abm_v5_implication_output_is_written_without_abm_v5_code() -> None:
    paths = _toy_paths()
    _write_three_variant_outputs(paths)
    validator = MultiYearEIDDiagnosticValidator(paths)
    result = validator.run()
    validator.write_outputs(result)

    assert paths.multiyear_EID_diagnostic_abm_v5_implications_path.exists()
    assert not (paths.project_root / "src" / "abm_v5").exists()


def test_missing_phase22_or_phase24_inputs_fail_clearly() -> None:
    paths = _toy_paths()
    paths.simulations.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="Missing Phase 24 heterogeneity panel"):
        MultiYearEIDDiagnosticValidator(paths).load_eid_metadata()
