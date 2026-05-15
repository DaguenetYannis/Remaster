from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import polars as pl
import pytest

from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.validation import StructuralSignatureDiagnostics


def _toy_paths() -> ABMV4Paths:
    return ABMV4Paths(project_root=Path("tmp") / "abm_v4_structural_signature_tests" / uuid4().hex)


def _state_rows() -> list[dict[str, object]]:
    nodes = [
        ("CHN | Electricity, Gas and Water", "CHN", "Electricity, Gas and Water", 1200.0, 0.40, 0.80),
        ("USA | power utilities", "USA", "power utilities", 900.0, 0.35, 0.70),
        ("KOR | utilities", "KOR", "utilities", 650.0, 0.30, 0.60),
        ("DEU | Manufacturing", "DEU", "Manufacturing", 120.0, 0.10, 0.20),
        ("FRA | Services", "FRA", "Services", 80.0, 0.05, 0.10),
        ("BRA | Agriculture", "BRA", "Agriculture", 60.0, 0.08, 0.10),
        ("IND | Construction", "IND", "Construction", 70.0, 0.09, 0.20),
        ("ZAF | Mining", "ZAF", "Mining", 90.0, 0.20, 0.40),
    ]
    rows = []
    for country_sector, country, sector, base_x, base_ei, brown in nodes:
        for offset, year in enumerate([2000, 2001, 2002]):
            jump_multiplier = 1.7 if "Electricity" in sector and year == 2001 else 1.0
            x = base_x * (1 + 0.1 * offset) * jump_multiplier
            ei = base_ei / (1 + 0.2 * offset)
            rows.append(
                {
                    "country_sector": country_sector,
                    "Year": year,
                    "Country": country,
                    "Sector": sector,
                    "ecosystem_label": "energy" if any(word in sector.lower() for word in ["electricity", "power", "utilities"]) else "other",
                    "X_observed": x,
                    "EI": ei,
                    "emissions_observed": x * ei,
                    "brown_centrality": brown,
                    "network_green_exposure": 1 - brown,
                    "general_capability_source": "atlas",
                    "green_capability_source": "io_imputed",
                }
            )
    return rows


def _write_inputs(paths: ABMV4Paths) -> None:
    paths.inputs.mkdir(parents=True, exist_ok=True)
    paths.validation.mkdir(parents=True, exist_ok=True)
    paths.interim.mkdir(parents=True, exist_ok=True)
    state = pl.DataFrame(_state_rows())
    state.write_parquet(paths.state_panel_path(1995, 2016))

    phase16 = state.select("country_sector", pl.col("Year").alias("year"), "Sector", "emissions_observed").with_columns(
        pl.when(pl.col("Sector").str.to_lowercase().str.contains("electricity|power|utilities"))
        .then(0.8)
        .otherwise(0.1)
        .alias("ei_gap"),
        pl.when(pl.col("Sector").str.to_lowercase().str.contains("electricity|power|utilities"))
        .then(0.3)
        .otherwise(0.6)
        .alias("readiness"),
        pl.when(pl.col("Sector").str.to_lowercase().str.contains("electricity|power|utilities"))
        .then(0.2)
        .otherwise(0.4)
        .alias("rEI_abs_error_readiness"),
        pl.when(pl.col("Sector").str.to_lowercase().str.contains("electricity|power|utilities"))
        .then(0.1)
        .otherwise(0.3)
        .alias("rEI_abs_error_frontier_gap"),
        pl.lit(-0.1).alias("delta_abs_error"),
        pl.lit(True).alias("frontier_gap_improves_abs_error"),
        pl.col("Sector").str.to_lowercase().str.contains("electricity|power|utilities").alias("frontier_gap_worsens_sign"),
        pl.col("Sector").str.to_lowercase().str.contains("electricity|power|utilities").alias(
            "frontier_gap_improves_magnitude_but_worsens_sign"
        ),
        pl.lit(10.0).alias("emissions_error_readiness"),
        pl.when(pl.col("Sector").str.to_lowercase().str.contains("electricity|power|utilities"))
        .then(50.0)
        .otherwise(5.0)
        .alias("emissions_error_frontier_gap"),
        pl.when(pl.col("Sector").str.to_lowercase().str.contains("electricity|power|utilities"))
        .then(pl.col("emissions_observed"))
        .otherwise(0.0)
        .alias("contribution_to_aggregate_error_difference"),
        pl.lit(0.2).alias("dampening_amount"),
    )
    phase16.write_parquet(paths.transition_rule_sign_failure_panel_path)

    pl.DataFrame(
        {
            "country_sector": state["country_sector"],
            "year": state["Year"],
            "cap_model": [0.5] * state.height,
            "gcap_model": [0.4] * state.height,
        }
    ).write_parquet(paths.capability_update_panel_path)
    pl.DataFrame(
        {
            "country_sector": state["country_sector"],
            "year": state["Year"],
            "exposure_cap": [0.6] * state.height,
            "exposure_gcap": [0.3] * state.height,
        }
    ).write_parquet(paths.capability_exposure_panel_path)
    pl.DataFrame(
        {
            "country_sector": [
                "CHN | Electricity, Gas and Water",
                "USA | power utilities",
                "KOR | utilities",
            ],
            "year": [2001, 2001, 2001],
            "jump_flag": [True, True, True],
        }
    ).write_csv(paths.raw_eora_electricity_breakpoint_audit_path)


def test_metric_inventory_classifies_available_columns() -> None:
    paths = _toy_paths()
    _write_inputs(paths)
    inventory = StructuralSignatureDiagnostics(paths).discover_available_metrics()

    state_row = inventory.filter(pl.col("source_path").str.contains("abm_v4_state_panel")).to_dicts()[0]
    assert "production_scale" in state_row["metric_family"]
    assert "emissions_scale" in state_row["metric_family"]


def test_node_year_feature_panel_is_built_from_multiple_sources() -> None:
    paths = _toy_paths()
    _write_inputs(paths)
    panel = StructuralSignatureDiagnostics(paths).build_node_year_feature_panel()

    assert {"cap_model", "exposure_gcap", "frontier_gap", "rEI_abs_error_readiness"}.issubset(panel.columns)
    assert panel.height == len(_state_rows())


def test_node_level_panel_aggregates_node_year_metrics() -> None:
    paths = _toy_paths()
    _write_inputs(paths)
    diag = StructuralSignatureDiagnostics(paths)
    node_year = diag.define_structural_labels(diag.build_node_year_feature_panel())
    node = diag.build_node_level_feature_panel(node_year)

    assert node.select("country_sector").n_unique() == 8
    assert "cumulative_emissions_share" in node.columns
    assert node["cumulative_emissions_share"].sum() == pytest.approx(1.0)


def test_electricity_like_label_is_assigned_correctly() -> None:
    paths = _toy_paths()
    _write_inputs(paths)
    panel = StructuralSignatureDiagnostics(paths).build_node_year_feature_panel()

    assert panel.filter(pl.col("Sector") == "power utilities")["electricity_like"].all()
    assert not panel.filter(pl.col("Sector") == "Manufacturing")["electricity_like"].any()


def test_high_emissions_and_jump_prone_labels_are_assigned() -> None:
    paths = _toy_paths()
    _write_inputs(paths)
    diag = StructuralSignatureDiagnostics(paths)
    node = diag.build_node_level_feature_panel(diag.define_structural_labels(diag.build_node_year_feature_panel()))

    china = node.filter(pl.col("Country") == "CHN").to_dicts()[0]
    assert china["high_emissions_node"]
    assert china["jump_prone_node"]


def test_needs_dampening_label_is_assigned_from_error_signatures() -> None:
    paths = _toy_paths()
    _write_inputs(paths)
    diag = StructuralSignatureDiagnostics(paths)
    node = diag.build_node_level_feature_panel(diag.define_structural_labels(diag.build_node_year_feature_panel()))

    assert node.filter(pl.col("Country") == "CHN")["needs_dampening_node"].item()


def test_electricity_contrast_computes_standardized_differences() -> None:
    paths = _toy_paths()
    _write_inputs(paths)
    diag = StructuralSignatureDiagnostics(paths)
    node = diag.build_node_level_feature_panel(diag.define_structural_labels(diag.build_node_year_feature_panel()))
    contrast = diag.compare_electricity_vs_non_electricity(node)

    assert "standardized_difference" in contrast.columns
    assert not contrast.filter(pl.col("metric") == "cumulative_emissions_share").is_empty()


def test_metric_screening_ranks_a_discriminating_metric_highest() -> None:
    diag = StructuralSignatureDiagnostics(_toy_paths())
    node = pl.DataFrame(
        {
            "country_sector": [f"n{i}" for i in range(8)],
            "electricity_like": [True, True, True, False, False, False, False, False],
            "high_emissions_node": [False] * 8,
            "jump_prone_node": [False] * 8,
            "aggregate_sensitive_node": [False] * 8,
            "needs_dampening_node": [False] * 8,
            "clear_metric": [10.0, 11.0, 12.0, 1.0, 1.1, 1.2, 1.3, 1.4],
            "noisy_metric": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
        }
    )
    screening = diag.screen_metric_discrimination(node)
    top = screening.filter(pl.col("label_name") == "electricity_like").sort("rank").to_dicts()[0]

    assert top["metric"] == "clear_metric"


def test_non_electricity_lookalike_detection_identifies_synthetic_lookalike() -> None:
    diag = StructuralSignatureDiagnostics(_toy_paths())
    node = pl.DataFrame(
        {
            "country_sector": ["elec", "infra", "service"],
            "Country": ["E", "I", "S"],
            "Sector": ["Electricity", "Transport infrastructure", "Services"],
            "ecosystem": ["energy", "infrastructure", "services"],
            "electricity_like": [True, False, False],
            "cumulative_emissions_share": [0.2, 0.19, 0.01],
            "cumulative_output_share": [0.2, 0.18, 0.02],
            "jump_frequency": [0.5, 0.45, 0.0],
            "share_frontier_gap_worsens_emissions_error": [0.8, 0.7, 0.0],
        }
    )
    screening = pl.DataFrame(
        {
            "label_name": ["electricity_like", "electricity_like"],
            "metric": ["cumulative_emissions_share", "cumulative_output_share"],
            "recommended_for_proxy": [True, True],
            "rank": [1, 2],
        }
    )
    lookalikes = diag.identify_non_electricity_lookalikes(node, screening)

    assert lookalikes.sort("similarity_score", descending=True)["country_sector"].item(0) == "infra"


def test_candidate_proxy_table_includes_availability_and_risk_fields() -> None:
    diag = StructuralSignatureDiagnostics(_toy_paths())
    node = pl.DataFrame(
        {
            "country_sector": ["elec", "other"],
            "electricity_like": [True, False],
            "cumulative_emissions_share": [0.2, 0.01],
            "cumulative_output_share": [0.2, 0.01],
            "jump_frequency": [0.5, 0.0],
        }
    )
    proxies = diag.build_candidate_proxy_table(node, pl.DataFrame(), pl.DataFrame())

    assert {"availability_status", "risks", "recommended_for_phase22"}.issubset(proxies.columns)
    assert "composite_transition_inertia_proxy" in proxies["candidate_proxy"].to_list()


def test_recommendation_selects_composite_proxy_for_multiple_structural_signals() -> None:
    diag = StructuralSignatureDiagnostics(_toy_paths())
    screening = pl.DataFrame(
        {
            "label_name": ["electricity_like", "electricity_like"],
            "metric_family": ["emissions_scale", "volatility_jump"],
            "separation_score": [1.2, 1.0],
        }
    )
    lookalikes = pl.DataFrame({"similarity_score": [0.8]})

    assert diag.build_recommendation(screening, lookalikes, pl.DataFrame())["recommendation"].item(0) == "build_composite_transition_inertia_proxy"


def test_recommendation_selects_insufficient_signature_when_metrics_do_not_distinguish() -> None:
    diag = StructuralSignatureDiagnostics(_toy_paths())
    screening = pl.DataFrame(
        {
            "label_name": ["electricity_like"],
            "metric_family": ["production_scale"],
            "separation_score": [0.2],
        }
    )

    assert diag.build_recommendation(screening, pl.DataFrame(), pl.DataFrame())["recommendation"].item(0) == "insufficient_signature"


def test_outputs_are_written_only_with_explicit_write_call() -> None:
    paths = _toy_paths()
    _write_inputs(paths)
    diag = StructuralSignatureDiagnostics(paths)
    result = diag.run()

    assert not paths.structural_signature_report_path.exists()
    diag.write_outputs(result)
    assert paths.structural_signature_node_year_panel_path.exists()
    assert paths.structural_signature_report_path.exists()


def test_missing_usable_feature_sources_fail_clearly() -> None:
    with pytest.raises(FileNotFoundError, match="state panel"):
        StructuralSignatureDiagnostics(_toy_paths()).run()
