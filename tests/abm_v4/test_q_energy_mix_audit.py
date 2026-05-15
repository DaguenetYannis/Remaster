from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import polars as pl
import pytest

from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.validation import QEnergyMixAudit


def _toy_paths() -> ABMV4Paths:
    return ABMV4Paths(project_root=Path("tmp") / "abm_v4_q_energy_tests" / uuid4().hex)


def _write_q_inputs(paths: ABMV4Paths, *, missing_component: bool = False, negative: bool = False) -> None:
    (paths.data_root / "parquet" / "1995").mkdir(parents=True, exist_ok=True)
    (paths.data_root / "raw" / "1995").mkdir(parents=True, exist_ok=True)
    paths.inputs.mkdir(parents=True, exist_ok=True)
    paths.validation.mkdir(parents=True, exist_ok=True)
    labels = [
        ("Energy Usage (TJ)", "Natural Gas"),
        ("Energy Usage (TJ)", "Coal"),
        ("Energy Usage (TJ)", "Petroleum"),
        ("Energy Usage (TJ)", "Nuclear Electricity"),
        ("Energy Usage (TJ)", "Hydroelectric Electricity"),
        ("Energy Usage (TJ)", "Geothermal Electricity"),
        ("Energy Usage (TJ)", "Wind Electricity"),
        ("Energy Usage (TJ)", "Solar, Tide and Wave Electricity"),
        ("Energy Usage (TJ)", "Biomass and Waste Electricity"),
    ]
    if missing_component:
        labels = labels[:-1]
    (paths.data_root / "raw" / "1995" / "labels_Q.txt").write_text(
        "\n".join(f"{family}\t{item}\t" for family, item in labels),
        encoding="utf-8",
    )
    rows = []
    for idx, _label in enumerate(labels):
        chn_value = [10, 80, 10, 0, 5, 0, 1, 2, 2][idx]
        usa_value = [40, 10, 30, 10, 5, 0, 3, 1, 1][idx]
        if negative and idx == 1:
            chn_value = -1
        rows.append(
            {
                "CHN | CHN | Industries | Electricity, Gas and Water": chn_value,
                "USA | USA | Industries | Manufacturing": usa_value,
            }
        )
    pl.DataFrame(rows).write_parquet(paths.data_root / "parquet" / "1995" / "Q.parquet")
    pl.DataFrame(
        {
            "country_sector": [
                "CHN | CHN | Industries | Electricity, Gas and Water",
                "USA | USA | Industries | Manufacturing",
            ],
            "Year": [1995, 1995],
            "Country": ["China", "USA"],
            "Sector": ["Electricity, Gas and Water", "Manufacturing"],
            "X_observed": [100.0, 200.0],
            "EI": [0.5, 0.2],
            "emissions_observed": [50.0, 40.0],
        }
    ).write_parquet(paths.state_panel_path(1995, 1995))
    pl.DataFrame(
        {
            "country_sector": [
                "CHN | CHN | Industries | Electricity, Gas and Water",
                "USA | USA | Industries | Manufacturing",
            ],
            "high_EID_definition": ["toy", "toy"],
            "high_EID_flag": [True, False],
            "candidate_subtype": ["infrastructure_energy", "manufacturing_system_core"],
            "pseudo_agent_flag": [False, False],
        }
    ).write_csv(paths.eid_high_node_heterogeneity_panel_path)


def _audit(paths: ABMV4Paths) -> QEnergyMixAudit:
    return QEnergyMixAudit(paths, start_year=1995, end_year=1995)


def test_q_energy_row_mapping_identifies_all_canonical_components() -> None:
    paths = _toy_paths()
    _write_q_inputs(paths)

    mapping = _audit(paths).build_row_mapping()

    assert set(mapping["canonical_energy_component"]) == set(QEnergyMixAudit.energy_columns)
    assert set(mapping["match_status"]) == {"matched"}


def test_missing_q_energy_rows_are_reported_clearly() -> None:
    paths = _toy_paths()
    _write_q_inputs(paths, missing_component=True)

    mapping = _audit(paths).build_row_mapping()

    assert "missing" in set(mapping["match_status"])


def test_energy_mix_panel_computes_fossil_clean_renewable_and_total() -> None:
    paths = _toy_paths()
    _write_q_inputs(paths)
    panel = _audit(paths).build_energy_mix_panel()
    row = panel.filter(pl.col("Country") == "China").row(0, named=True)

    assert row["fossil_energy_TJ"] == pytest.approx(100)
    assert row["clean_electricity_TJ"] == pytest.approx(10)
    assert row["renewable_electricity_TJ"] == pytest.approx(10)
    assert row["total_tracked_energy_TJ"] == pytest.approx(110)


def test_shares_are_computed_correctly_without_false_flags() -> None:
    paths = _toy_paths()
    _write_q_inputs(paths)
    row = _audit(paths).build_energy_mix_panel().filter(pl.col("Country") == "China").row(0, named=True)

    assert row["coal_share"] == pytest.approx(80 / 110)
    assert row["mapping_quality_flag"] == "ok"


def test_negative_energy_values_are_flagged() -> None:
    paths = _toy_paths()
    _write_q_inputs(paths, negative=True)
    panel = _audit(paths).build_energy_mix_panel()
    quality = _audit(paths).build_quality_audit(panel)

    assert quality.filter(pl.col("component") == "coal_TJ")["negative_count"].item() == 1


def test_nonfinite_values_are_flagged() -> None:
    audit = _audit(_toy_paths())
    frame = pl.DataFrame({"bad": [1.0, float("inf")]})

    assert audit._nonfinite_count(frame["bad"]) == 1


def test_energy_mix_hhi_and_entropy_are_computed_correctly() -> None:
    paths = _toy_paths()
    _write_q_inputs(paths)
    row = _audit(paths).build_energy_mix_panel().filter(pl.col("Country") == "China").row(0, named=True)

    shares = [10 / 110, 80 / 110, 10 / 110, 0, 5 / 110, 0, 1 / 110, 2 / 110, 2 / 110]
    expected_hhi = sum(value * value for value in shares)
    expected_entropy = -sum(value * __import__("math").log(value) for value in shares if value > 0)
    assert row["energy_mix_hhi"] == pytest.approx(expected_hhi)
    assert row["energy_mix_entropy"] == pytest.approx(expected_entropy)


def test_intensity_per_output_guards_against_nonpositive_output() -> None:
    paths = _toy_paths()
    _write_q_inputs(paths)
    state = pl.read_parquet(paths.state_panel_path(1995, 1995)).with_columns(
        pl.when(pl.col("Country") == "China").then(0.0).otherwise(pl.col("X_observed")).alias("X_observed")
    )
    state.write_parquet(paths.state_panel_path(1995, 1995))

    row = _audit(paths).build_energy_mix_panel().filter(pl.col("Country") == "China").row(0, named=True)

    assert row["total_tracked_energy_per_output"] is None


def test_quality_audit_computes_coverage_and_invalid_shares() -> None:
    paths = _toy_paths()
    _write_q_inputs(paths)
    panel = _audit(paths).build_energy_mix_panel()
    quality = _audit(paths).build_quality_audit(panel)

    assert quality.filter(pl.col("component") == "total_tracked_energy_TJ")["coverage_share"].item() == pytest.approx(1.0)
    assert quality.filter(pl.col("component") == "_share_validity")["nonfinite_count"].item() == 0


def test_aggregate_plausibility_flags_extreme_jumps() -> None:
    paths = _toy_paths()
    _write_q_inputs(paths)
    panel = _audit(paths).build_energy_mix_panel()
    second = panel.with_columns(pl.lit(1996).alias("year"), (pl.col("total_tracked_energy_TJ") * 100).alias("total_tracked_energy_TJ"))
    plausibility = _audit(paths).build_aggregate_plausibility(pl.concat([panel, second], how="diagonal_relaxed"))

    assert "extreme_jump" in set(plausibility["plausibility_flag"])


def test_china_electricity_focused_audit_selects_correct_node() -> None:
    paths = _toy_paths()
    _write_q_inputs(paths)
    audit = _audit(paths)
    panel = audit.build_energy_mix_panel()
    china = audit.build_china_electricity_audit(panel)

    assert china.height == 1
    assert china["Country"].to_list() if "Country" in china.columns else True
    assert "Electricity" in china["country_sector"].item()


def test_transition_error_join_preserves_model_error_columns() -> None:
    paths = _toy_paths()
    _write_q_inputs(paths)
    pl.DataFrame(
        {
            "country_sector": ["CHN | CHN | Industries | Electricity, Gas and Water"],
            "year": [1995],
            "observed_rEI": [0.1],
            "simulated_rEI_readiness": [0.08],
            "simulated_rEI_frontier_gap": [0.12],
            "rEI_error_readiness": [-0.02],
            "rEI_error_frontier_gap": [0.02],
            "rEI_abs_error_readiness": [0.02],
            "rEI_abs_error_frontier_gap": [0.02],
            "emissions_error_readiness": [1.0],
            "emissions_error_frontier_gap": [2.0],
            "emissions_decile": ["d10"],
        }
    ).write_parquet(paths.transition_rule_sign_failure_panel_path)

    joined = _audit(paths).build_transition_error_panel(_audit(paths).build_energy_mix_panel())

    assert {"rEI_error_frontier_gap_readiness", "rEI_error_historical_frontier_gap_only"}.issubset(joined.columns)


def test_predictor_screening_ranks_synthetic_strong_predictor() -> None:
    paths = _toy_paths()
    audit = _audit(paths)
    panel = pl.DataFrame(
        {
            "coal_share": [0.0, 0.2, 0.8, 1.0, 0.4],
            "rEI_abs_error_frontier_gap": [0.0, 0.2, 0.8, 1.0, 0.4],
        }
    )

    screening = audit.build_predictor_screening(panel)

    assert screening.filter(pl.col("predictor") == "coal_share")["recommended_for_phase28"].item() is True


def test_subtype_diagnostics_aggregate_correctly() -> None:
    paths = _toy_paths()
    _write_q_inputs(paths)
    panel = _audit(paths).build_transition_error_panel(_audit(paths).build_energy_mix_panel()).with_columns(
        pl.lit(0.1).alias("rEI_error_historical_frontier_gap_only"),
        pl.lit(0.2).alias("rEI_error_frontier_gap_readiness"),
    )
    subtype = _audit(paths).build_subtype_diagnostics(panel)

    assert {"infrastructure_energy", "manufacturing_system_core"}.issubset(set(subtype["candidate_subtype"]))


def test_hypothesis_logic_supports_usable_data_when_thresholds_pass() -> None:
    audit = _audit(_toy_paths())
    quality = pl.DataFrame(
        {
            "component": ["total_tracked_energy_TJ", "_share_validity"],
            "coverage_share": [0.9, 1.0],
            "negative_count": [0, 0],
            "nonfinite_count": [0, 0],
        }
    )
    hypotheses = audit.build_hypothesis_tests(
        quality,
        pl.DataFrame({"plausibility_flag": ["ok"]}),
        pl.DataFrame({"fossil_share": [0.8]}),
        pl.DataFrame({"target": ["x"], "abs_correlation": [0.3], "recommended_for_phase28": [True]}),
        pl.DataFrame({"candidate_subtype": ["infrastructure_energy"], "mean_fossil_share": [0.8]}),
    )

    assert hypotheses.filter(pl.col("hypothesis") == "H1_energy_mix_data_usable")["result"].item() is True


def test_hypothesis_logic_rejects_noisy_sparse_data() -> None:
    audit = _audit(_toy_paths())
    quality = pl.DataFrame(
        {
            "component": ["total_tracked_energy_TJ", "_share_validity"],
            "coverage_share": [0.1, 1.0],
            "negative_count": [2, 0],
            "nonfinite_count": [0, 2],
        }
    )
    hypotheses = audit.build_hypothesis_tests(quality, pl.DataFrame({"plausibility_flag": ["extreme_jump"] * 60}), pl.DataFrame(), pl.DataFrame(), pl.DataFrame())

    assert hypotheses.filter(pl.col("hypothesis") == "H5_energy_mix_too_sparse_or_noisy")["result"].item() is True


def test_recommendation_selects_energy_mix_augmented_candidate_when_evidence_is_strong() -> None:
    audit = _audit(_toy_paths())
    hypotheses = pl.DataFrame(
        {
            "hypothesis": ["H1_energy_mix_data_usable", "H6_energy_mix_may_resolve_frontier_tradeoff"],
            "result": [True, True],
        }
    )
    quality = pl.DataFrame({"component": ["total_tracked_energy_TJ"], "coverage_share": [0.95]})
    screening = pl.DataFrame({"recommended_for_phase28": [True]})

    rec = audit.build_recommendation(hypotheses, quality, screening)

    assert rec["recommendation"].item() == "test_energy_mix_augmented_transition_candidate"


def test_recommendation_selects_aggregate_only_when_country_sector_quality_is_partial() -> None:
    audit = _audit(_toy_paths())
    hypotheses = pl.DataFrame({"hypothesis": ["H5_energy_mix_too_sparse_or_noisy"], "result": [False]})
    quality = pl.DataFrame({"component": ["total_tracked_energy_TJ"], "coverage_share": [0.6]})

    rec = audit.build_recommendation(hypotheses, quality, pl.DataFrame())

    assert rec["recommendation"].item() == "aggregate_only_energy_mix_usable"


def test_outputs_are_written_only_with_explicit_output_flag() -> None:
    paths = _toy_paths()
    _write_q_inputs(paths)
    audit = _audit(paths)

    result = audit.run()

    assert not paths.q_energy_mix_report_path.exists()
    audit.write_outputs(result)
    assert paths.q_energy_source_inventory_path.exists()
    assert paths.q_energy_mix_report_path.exists()


def test_missing_q_source_candidates_fail_clearly() -> None:
    paths = _toy_paths()
    paths.inputs.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"country_sector": ["a"], "Year": [1995], "Country": ["A"], "Sector": ["S"], "X_observed": [1.0], "EI": [1.0]}).write_parquet(paths.state_panel_path(1995, 1995))

    with pytest.raises(FileNotFoundError, match="No usable Eora Q parquet sources"):
        _audit(paths).build_energy_mix_panel()
