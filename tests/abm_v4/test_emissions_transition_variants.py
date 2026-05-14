from pathlib import Path
from uuid import uuid4

import polars as pl
import pytest

from src.abm_v4.emissions import EmissionsTransitionVariantComparator
from src.abm_v4.paths import ABMV4Paths


def _toy_paths() -> ABMV4Paths:
    return ABMV4Paths(project_root=Path("tmp") / "abm_v4_transition_variant_tests" / uuid4().hex)


def _comparator() -> EmissionsTransitionVariantComparator:
    return EmissionsTransitionVariantComparator(
        _toy_paths(),
        random_search_iterations=2,
        minimum_family_observations=500,
    )


def _base_panel() -> pl.DataFrame:
    rows = []
    for year in [1995, 1996, 1997]:
        rows.extend(
            [
                {
                    "country_sector": "A",
                    "year": year,
                    "Country": "AA",
                    "Sector": "Agriculture",
                    "ecosystem_id": "eco1",
                    "EI": 1.0 + (year - 1995) * 0.1,
                    "cap_model": 0.8,
                    "gcap_model": 0.7,
                    "general_capability_source": "atlas_observed",
                    "green_capability_source": "atlas_observed",
                    "network_green_exposure": 0.6,
                    "ecosystem_capability_exposure": 0.7,
                    "brown_centrality": 0.1,
                    "supplier_lockin": 0.2,
                },
                {
                    "country_sector": "B",
                    "year": year,
                    "Country": "BB",
                    "Sector": "Agriculture",
                    "ecosystem_id": "eco1",
                    "EI": 2.0 + (year - 1995) * 0.1,
                    "cap_model": 0.2,
                    "gcap_model": 0.1,
                    "general_capability_source": "io_imputed",
                    "green_capability_source": "io_imputed",
                    "network_green_exposure": 0.2,
                    "ecosystem_capability_exposure": 0.7,
                    "brown_centrality": 0.4,
                    "supplier_lockin": 0.8,
                },
                {
                    "country_sector": "C",
                    "year": year,
                    "Country": "CC",
                    "Sector": "Mining",
                    "ecosystem_id": "eco2",
                    "EI": 4.0 if year < 1997 else 0.5,
                    "cap_model": 0.3,
                    "gcap_model": 0.2,
                    "general_capability_source": "unavailable",
                    "green_capability_source": "unavailable",
                    "network_green_exposure": 0.2,
                    "ecosystem_capability_exposure": 0.3,
                    "brown_centrality": 0.5,
                    "supplier_lockin": 0.5,
                },
            ]
        )
    return pl.DataFrame(rows)


def _target_panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "target_name": ["one_year_rEI"] * 4,
            "horizon_years": [1] * 4,
            "country_sector": ["A", "B", "A", "B"],
            "year": [1995, 1995, 1996, 1996],
            "next_year": [1996, 1996, 1997, 1997],
            "Country": ["AA", "BB", "AA", "BB"],
            "Sector": ["Agriculture", "Agriculture", "Agriculture", "Agriculture"],
            "ecosystem_id": ["eco1"] * 4,
            "target": [0.02, -0.01, 0.03, -0.02],
            "sector_background_prediction": [0.0, 0.0, 0.0, 0.0],
            "cap_model": [0.8, 0.2, 0.8, 0.2],
            "gcap_model": [0.7, 0.1, 0.7, 0.1],
            "general_capability_source": ["atlas_observed", "io_imputed", "atlas_observed", "io_imputed"],
            "green_capability_source": ["atlas_observed", "io_imputed", "atlas_observed", "io_imputed"],
            "network_green_exposure": [0.6, 0.2, 0.6, 0.2],
            "ecosystem_capability_exposure": [0.7, 0.7, 0.7, 0.7],
            "brown_centrality": [0.1, 0.4, 0.1, 0.4],
            "supplier_lockin": [0.2, 0.8, 0.2, 0.8],
        }
    )


def _write_run_inputs(paths: ABMV4Paths) -> None:
    rows = []
    for year in range(1995, 2017):
        offset = year - 1995
        rows.extend(
            [
                {
                    "country_sector": "AA_Agriculture",
                    "Year": year,
                    "Country": "AA",
                    "Sector": "Agriculture",
                    "ecosystem_id": "eco1",
                    "EI": 2.0 - 0.02 * offset,
                    "general_capability_model": 0.8,
                    "green_capability_model": 0.7,
                    "general_capability_source": "atlas_observed",
                    "green_capability_source": "atlas_observed",
                    "network_green_exposure": 0.6,
                    "brown_centrality": 0.1,
                },
                {
                    "country_sector": "BB_Mining",
                    "Year": year,
                    "Country": "BB",
                    "Sector": "Mining",
                    "ecosystem_id": "eco2",
                    "EI": 3.0 - 0.01 * offset,
                    "general_capability_model": 0.3,
                    "green_capability_model": 0.2,
                    "general_capability_source": "io_imputed",
                    "green_capability_source": "io_imputed",
                    "network_green_exposure": 0.2,
                    "brown_centrality": 0.5,
                },
            ]
        )
    state_path = paths.state_panel_path(1995, 2016)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(state_path)


def test_sector_family_mapping_is_created() -> None:
    mapped = _comparator().add_sector_family(
        pl.DataFrame({"Sector": ["Agriculture", "Mining and Quarrying", "Financial Intermediation"]})
    )

    assert mapped["sector_family"].to_list() == [
        "agriculture_biomass",
        "extractive_energy",
        "services_knowledge_finance",
    ]


def test_sector_year_p25_and_p50_frontiers_differ_correctly() -> None:
    comparator = _comparator()
    base = _base_panel()
    p25 = comparator.compute_frontier_gaps(base, "sector_year_p25")
    p50 = comparator.compute_frontier_gaps(base, "sector_year_p50")

    b_p25 = p25.filter((pl.col("country_sector") == "B") & (pl.col("year") == 1995))["frontier_gap"].item()
    b_p50 = p50.filter((pl.col("country_sector") == "B") & (pl.col("year") == 1995))["frontier_gap"].item()
    assert b_p25 > b_p50


def test_rolling_frontier_does_not_use_future_years() -> None:
    comparator = _comparator()
    base = _base_panel()
    rolling = comparator.compute_frontier_gaps(base, "rolling_sector_p50")
    current = comparator.compute_frontier_gaps(base.filter(pl.col("year") <= 1996), "rolling_sector_p50")

    c_1996_full = rolling.filter((pl.col("country_sector") == "C") & (pl.col("year") == 1996))["frontier_gap"].item()
    c_1996_truncated = current.filter((pl.col("country_sector") == "C") & (pl.col("year") == 1996))["frontier_gap"].item()
    assert c_1996_full == pytest.approx(c_1996_truncated)


def test_sector_family_variant_falls_back_to_global_when_observations_too_few() -> None:
    comparator = _comparator()
    dataset = comparator.build_variant_dataset(_base_panel(), _target_panel(), ["one_year_rEI"], ["sector_year_p25"])
    train, validation = comparator.split_train_validation(dataset)
    params = comparator._default_parameter_set()
    family_params, fallbacks = comparator.search_sector_family_parameters(
        train,
        validation,
        params,
        improved_sectors=set(),
    )

    assert family_params
    assert set(fallbacks.values()) == {"global_parameters_minimum_observations"}


def test_gated_readiness_uses_readiness_only_in_flagged_sectors() -> None:
    comparator = _comparator()
    dataset = comparator.build_variant_dataset(_base_panel(), _target_panel(), ["one_year_rEI"], ["sector_year_p25"])
    params = comparator._default_parameter_set()
    gated = comparator.predict_with_parameters(
        dataset,
        params,
        "gated_readiness_by_sector_signal",
        improved_sectors=set(),
    )
    gap_only = comparator.predict_with_parameters(
        dataset,
        params,
        "frontier_gap_only",
        improved_sectors=set(),
    )

    assert gated["simulated_rEI"].to_list() == pytest.approx(gap_only["simulated_rEI"].to_list())


def test_model_comparison_table_includes_all_required_variants() -> None:
    paths = _toy_paths()
    _write_run_inputs(paths)
    comparator = EmissionsTransitionVariantComparator(
        paths,
        random_search_iterations=1,
        targets=["one_year_rEI"],
        frontier_variants=["sector_year_p25"],
    )
    result = comparator.run()

    assert set(result.results["model_variant"].to_list()) == set(comparator.MODEL_VARIANTS)


def test_recommendation_logic_selects_sector_family_when_it_improves_enough() -> None:
    comparator = _comparator()
    results = pl.DataFrame(
        {
            "target_name": ["one_year_rEI", "one_year_rEI"],
            "frontier_variant": ["sector_year_p50", "sector_year_p50"],
            "model_variant": ["sector_background_only", "sector_family_frontier_gap_readiness"],
            "validation_mae": [1.0, 0.9],
            "validation_wrong_sign_share": [0.4, 0.4],
            "validation_correlation": [0.0, 0.2],
            "improvement_over_sector_background": [0.0, 0.1],
            "improvement_over_sector_background_pct": [0.0, 0.1],
        }
    )

    rec = comparator.build_recommendation(results)
    assert rec["recommended_model_variant"].item() == "sector_family_frontier_gap_readiness"


def test_recommendation_logic_selects_conservative_fallback_when_readiness_fails() -> None:
    comparator = _comparator()
    results = pl.DataFrame(
        {
            "target_name": ["one_year_rEI", "one_year_rEI", "one_year_rEI"],
            "frontier_variant": ["sector_year_p25", "sector_year_p25", "sector_year_p25"],
            "model_variant": [
                "sector_background_only",
                "frontier_gap_only",
                "global_frontier_gap_readiness",
            ],
            "validation_mae": [1.0, 0.99, 1.1],
            "validation_wrong_sign_share": [0.4, 0.4, 0.5],
            "validation_correlation": [0.0, 0.1, -0.1],
            "improvement_over_sector_background": [0.0, 0.01, -0.1],
            "improvement_over_sector_background_pct": [0.0, 0.01, -0.1],
        }
    )

    rec = comparator.build_recommendation(results)
    assert rec["recommended_model_variant"].item() == "frontier_gap_only"


def test_outputs_are_written_only_with_explicit_output_call() -> None:
    paths = _toy_paths()
    _write_run_inputs(paths)
    comparator = EmissionsTransitionVariantComparator(
        paths,
        random_search_iterations=1,
        targets=["one_year_rEI"],
        frontier_variants=["sector_year_p25"],
    )
    result = comparator.run()

    assert not comparator.paths.emissions_transition_variant_results_path.exists()
    comparator.write_outputs(result)
    assert comparator.paths.emissions_transition_variant_results_path.exists()
    assert comparator.paths.emissions_transition_variant_recommendation_path.exists()


def test_config_py_is_not_overwritten() -> None:
    before = Path("src/abm_v4/config.py").read_text(encoding="utf-8")
    paths = _toy_paths()
    _write_run_inputs(paths)
    comparator = EmissionsTransitionVariantComparator(
        paths,
        random_search_iterations=1,
        targets=["one_year_rEI"],
        frontier_variants=["sector_year_p25"],
    )
    _ = comparator.run()
    after = Path("src/abm_v4/config.py").read_text(encoding="utf-8")

    assert after == before
