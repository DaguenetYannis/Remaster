from pathlib import Path
from uuid import uuid4

import polars as pl
import pytest

from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.validation import TransitionRuleTradeoffDiagnostics


def _toy_paths() -> ABMV4Paths:
    return ABMV4Paths(project_root=Path("tmp") / "abm_v4_tradeoff_tests" / uuid4().hex)


def _variant_panel(scale: float, *, readiness_variant: bool = False) -> pl.DataFrame:
    rows = []
    specs = [
        ("A", "S1", "C1", 100.0, [2.0, 1.8, 1.7], "atlas_observed"),
        ("B", "S1", "C1", 1000.0, [2.0, 2.02, 2.03], "io_imputed"),
        ("C", "S2", "C2", 50.0, [1.0, 0.9, 0.95], "atlas_observed"),
    ]
    for node, sector, country, output, observed_ei, source in specs:
        if readiness_variant and node == "B":
            sim_ei = [2.0, 3.0, 3.1]
        elif node == "B":
            sim_ei = [2.0, 1.98, 1.99]
        else:
            sim_ei = [observed_ei[0], observed_ei[0] * scale, observed_ei[1] * scale]
        for index, year in enumerate([1995, 1996, 1997]):
            rows.append(
                {
                    "country_sector": node,
                    "year": year,
                    "Country": country,
                    "Sector": sector,
                    "ecosystem_id": "eco1" if sector == "S1" else "eco2",
                    "ecosystem_label": "Eco 1" if sector == "S1" else "Eco 2",
                    "general_capability_source": source,
                    "green_capability_source": source,
                    "X_observed": output,
                    "EI_observed": observed_ei[index],
                    "emissions_observed": output * observed_ei[index],
                    "EI_sim": sim_ei[index],
                    "emissions_sim": output * sim_ei[index],
                    "ei_gap": 1.0 if node == "B" else 0.2,
                    "readiness": 0.1 if node == "B" else 0.8,
                    "brown_centrality": 0.7 if node == "B" else 0.1,
                    "network_green_exposure": 0.2 if node == "B" else 0.7,
                    "production_feasibility_ratio": 1.0,
                }
            )
    return pl.DataFrame(rows)


def _write_variant_outputs(paths: ABMV4Paths) -> None:
    paths.simulations.mkdir(parents=True, exist_ok=True)
    _variant_panel(0.95, readiness_variant=True).write_parquet(paths.base_multiyear_state_panel_path)
    _variant_panel(0.85).write_parquet(paths.base_multiyear_state_panel_historical_frontier_gap_path)


def test_comparison_panel_aligns_observed_and_two_variants() -> None:
    paths = _toy_paths()
    _write_variant_outputs(paths)
    panel = TransitionRuleTradeoffDiagnostics(paths).build_node_year_comparison_panel()

    assert {"EI_sim_readiness", "EI_sim_frontier_gap", "observed_rEI"} <= set(panel.columns)
    assert panel.height == 6


def test_rei_is_computed_from_consecutive_simulated_ei_values() -> None:
    paths = _toy_paths()
    _write_variant_outputs(paths)
    panel = TransitionRuleTradeoffDiagnostics(paths).build_node_year_comparison_panel()
    row = panel.filter((pl.col("country_sector") == "A") & (pl.col("year") == 1995)).to_dicts()[0]

    assert row["simulated_rEI_readiness"] == pytest.approx(__import__("math").log(2.0) - __import__("math").log(1.9))


def test_sign_correctness_is_computed_correctly() -> None:
    paths = _toy_paths()
    _write_variant_outputs(paths)
    panel = TransitionRuleTradeoffDiagnostics(paths).build_node_year_comparison_panel()
    row = panel.filter((pl.col("country_sector") == "B") & (pl.col("year") == 1995)).to_dicts()[0]

    assert row["observed_rEI"] < 0
    assert row["simulated_rEI_frontier_gap"] > 0
    assert row["rEI_wrong_sign_frontier_gap"]


def test_delta_abs_error_is_computed_correctly() -> None:
    paths = _toy_paths()
    _write_variant_outputs(paths)
    panel = TransitionRuleTradeoffDiagnostics(paths).build_node_year_comparison_panel()
    row = panel.filter((pl.col("country_sector") == "A") & (pl.col("year") == 1995)).to_dicts()[0]

    expected = row["rEI_abs_error_frontier_gap"] - row["rEI_abs_error_readiness"]
    assert row["delta_abs_error"] == pytest.approx(expected)


def test_weighted_mae_differs_from_unweighted_when_weights_differ() -> None:
    paths = _toy_paths()
    _write_variant_outputs(paths)
    diagnostics = TransitionRuleTradeoffDiagnostics(paths)
    weighted = diagnostics.compute_weighted_errors(diagnostics.build_node_year_comparison_panel())
    frontier = weighted.filter(pl.col("model_variant") == "historical_frontier_gap_only").to_dicts()[0]

    assert frontier["output_weighted_rEI_MAE"] != pytest.approx(frontier["unweighted_rEI_MAE"])
    assert frontier["emissions_weighted_rEI_MAE"] != pytest.approx(frontier["unweighted_rEI_MAE"])


def test_group_diagnostics_detect_magnitude_improves_but_sign_worsens_sector() -> None:
    paths = _toy_paths()
    _write_variant_outputs(paths)
    diagnostics = TransitionRuleTradeoffDiagnostics(paths)
    grouped = diagnostics.summarize_by_group(diagnostics.build_node_year_comparison_panel(), ["Sector"])

    assert grouped.filter(
        pl.col("recommended_interpretation") == "magnitude improves but sign worsens"
    ).height >= 1


def test_aggregate_contribution_ranking_sums_to_expected_total_difference() -> None:
    paths = _toy_paths()
    _write_variant_outputs(paths)
    diagnostics = TransitionRuleTradeoffDiagnostics(paths)
    aggregate = diagnostics.compute_aggregate_contributions(diagnostics.build_node_year_comparison_panel())

    assert aggregate["rank"].min() == 1
    assert aggregate["contribution_share_to_total_difference"].sum() == pytest.approx(1.0)


def test_hypothesis_test_table_includes_h1_to_h10() -> None:
    paths = _toy_paths()
    _write_variant_outputs(paths)
    result = TransitionRuleTradeoffDiagnostics(paths).run()

    assert set(result.hypothesis_tests["hypothesis_id"].to_list()) == {
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
        "H7",
        "H8",
        "H9",
        "H10",
    }


def test_markdown_report_written_only_with_explicit_output_call() -> None:
    paths = _toy_paths()
    _write_variant_outputs(paths)
    diagnostics = TransitionRuleTradeoffDiagnostics(paths)
    result = diagnostics.run()

    assert not paths.transition_rule_error_tradeoff_report_path.exists()
    diagnostics.write_outputs(result)
    assert paths.transition_rule_error_tradeoff_report_path.exists()
    assert paths.transition_rule_sign_failure_panel_path.exists()


def test_missing_variant_outputs_fail_with_actionable_message() -> None:
    paths = _toy_paths()
    with pytest.raises(FileNotFoundError, match="Run the default multi-year base"):
        TransitionRuleTradeoffDiagnostics(paths).build_node_year_comparison_panel()
