from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import src.abm_v5 as abm_v5
from src.abm_v5.networks import (
    HISTORICAL_END_YEAR,
    HISTORICAL_START_YEAR,
    NETWORK_REQUIRED_COLUMNS,
    NetworkBuildResult,
    add_supplier_weights_and_coefficients,
    build_same_sector_fallback_candidates_for_year,
    build_network_state_for_year,
    build_network_state_panels,
    build_production_edges_for_year,
    build_supplier_candidates_for_year,
    load_edge_state_panel,
    load_network_state_panel,
    load_supplier_candidate_panel,
    summarize_supplier_candidate_coverage,
    summarize_network_state,
    validate_edge_state_panel,
    validate_network_state_panel,
    validate_supplier_candidate_panel,
)


LABELS = (
    "USA | United States | Industry | Manufacturing",
    "CHN | China | Industry | Manufacturing",
)


def _write_pyproject(project_root: Path) -> None:
    project_root.joinpath("pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")


def _write_t_inputs(project_root: Path, year: int, values: list[list[float]] | None = None) -> None:
    values = values or [[5.0, 10.0], [20.0, 0.0]]
    parquet_dir = project_root / "data" / "parquet" / str(year)
    raw_dir = project_root / "data" / "raw" / str(year)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "__index_level_0__": list(LABELS),
            LABELS[0]: [values[0][0], values[1][0]],
            LABELS[1]: [values[0][1], values[1][1]],
        }
    ).write_parquet(parquet_dir / "T.parquet")
    raw_dir.joinpath("labels_T.txt").write_text("\n".join(LABELS), encoding="utf-8")


def _accounting_for_year(year: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "country_sector": list(LABELS),
            "country": ["USA", "CHN"],
            "country_detail": ["United States", "China"],
            "category": ["Industry", "Industry"],
            "sector": ["Manufacturing", "Manufacturing"],
            "year": [year, year],
            "output": [100.0, 200.0],
            "local_greenness": [0.2, 0.8],
            "accounting_output_positive_flag": [True, True],
            "accounting_emissions_nonnegative_flag": [True, True],
            "accounting_ei_valid_flag": [True, True],
        }
    )


def _weighted_edges_for_year(year: int = 1995) -> pl.DataFrame:
    edges = pl.DataFrame(
        {
            "year": [year, year, year],
            "supplier_country_sector": [LABELS[0], LABELS[0], LABELS[1]],
            "buyer_country_sector": [LABELS[0], LABELS[1], LABELS[0]],
            "transaction_value": [5.0, 10.0, 20.0],
        }
    )
    return add_supplier_weights_and_coefficients(edges, _accounting_for_year(year))


def _candidate_accounting(year: int = 1995, n_suppliers: int = 12) -> pl.DataFrame:
    rows = [
        {
            "country_sector": "USA | United States | Industry | Manufacturing",
            "country": "USA",
            "sector": "Manufacturing",
            "year": year,
            "output": 100.0,
            "local_greenness": 0.2,
        }
    ]
    for index in range(n_suppliers):
        rows.append(
            {
                "country_sector": f"C{index:02d} | Country {index:02d} | Industry | Manufacturing",
                "country": f"C{index:02d}",
                "sector": "Manufacturing",
                "year": year,
                "output": float(200 - index),
                "local_greenness": 0.5,
            }
        )
    return pl.DataFrame(rows)


def _candidate_edges(year: int = 1995, values: list[float] | None = None) -> pl.DataFrame:
    values = values or [50.0, 20.0, 10.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.8, 0.6, 0.4, 0.2]
    buyer = "USA | United States | Industry | Manufacturing"
    edges = pl.DataFrame(
        {
            "year": [year] * len(values),
            "buyer_country_sector": [buyer] * len(values),
            "supplier_country_sector": [
                f"C{index:02d} | Country {index:02d} | Industry | Manufacturing"
                for index in range(len(values))
            ],
            "transaction_value": values,
        }
    )
    return add_supplier_weights_and_coefficients(edges, _candidate_accounting(year, len(values)))


def _write_phase2_backbone(project_root: Path) -> None:
    _write_pyproject(project_root)
    identity_dir = project_root / "data" / "abm_v5" / "inputs"
    accounting_dir = project_root / "data" / "abm_v5" / "accounting"
    identity_dir.mkdir(parents=True, exist_ok=True)
    accounting_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "country_sector": list(LABELS),
            "country": ["USA", "CHN"],
            "country_detail": ["United States", "China"],
            "category": ["Industry", "Industry"],
            "sector": ["Manufacturing", "Manufacturing"],
        }
    ).write_parquet(identity_dir / "agent_identity.parquet")
    accounting = pl.concat(
        [_accounting_for_year(year) for year in range(HISTORICAL_START_YEAR, HISTORICAL_END_YEAR + 1)]
    )
    accounting.write_parquet(accounting_dir / "accounting_state_panel_1995_2016.parquet")
    for year in range(HISTORICAL_START_YEAR, HISTORICAL_END_YEAR + 1):
        _write_t_inputs(project_root, year)


def test_build_production_edges_for_year_uses_positive_t_edges(tmp_path: Path) -> None:
    _write_t_inputs(tmp_path, 1995, values=[[0.0, 10.0], [20.0, 0.0]])
    edges = build_production_edges_for_year(tmp_path, 1995)
    assert edges.height == 2
    assert sorted(edges["transaction_value"].to_list()) == [10.0, 20.0]


def test_build_production_edges_for_year_preserves_supplier_buyer_labels(tmp_path: Path) -> None:
    _write_t_inputs(tmp_path, 1995, values=[[0.0, 10.0], [0.0, 0.0]])
    edges = build_production_edges_for_year(tmp_path, 1995)
    row = edges.row(0, named=True)
    assert row["supplier_country_sector"] == LABELS[0]
    assert row["buyer_country_sector"] == LABELS[1]


def test_add_supplier_weights_and_coefficients_normalizes_by_buyer() -> None:
    weighted = _weighted_edges_for_year()
    weight_sum = (
        weighted.filter(pl.col("buyer_country_sector") == LABELS[0])
        .select(pl.sum("supplier_weight"))
        .item()
    )
    assert weight_sum == pytest.approx(1.0)
    coefficient = weighted.filter(
        (pl.col("supplier_country_sector") == LABELS[0])
        & (pl.col("buyer_country_sector") == LABELS[1])
    )["technical_coefficient"].item()
    assert coefficient == pytest.approx(10.0 / 200.0)


def test_add_supplier_weights_and_coefficients_nulls_coefficient_for_nonpositive_buyer_output() -> None:
    accounting = _accounting_for_year(1995).with_columns(
        pl.when(pl.col("country_sector") == LABELS[1]).then(0.0).otherwise(pl.col("output")).alias("output")
    )
    weighted = add_supplier_weights_and_coefficients(_weighted_edges_for_year().select(
        "year", "supplier_country_sector", "buyer_country_sector", "transaction_value"
    ), accounting)
    assert (
        weighted.filter(pl.col("buyer_country_sector") == LABELS[1])["technical_coefficient"].item()
        is None
    )


def test_validate_edge_state_panel_requires_columns() -> None:
    results = validate_edge_state_panel(pl.DataFrame({"year": [1995]}))
    assert any(result.status.value == "failed" for result in results)


def test_validate_edge_state_panel_detects_duplicate_edges() -> None:
    edge = _weighted_edges_for_year().head(1)
    results = validate_edge_state_panel(pl.concat([edge, edge], how="vertical"))
    assert any(result.check_name == "edge_unique_year_supplier_buyer" and result.n_failed > 0 for result in results)


def test_validate_edge_state_panel_checks_supplier_weight_range() -> None:
    edge = _weighted_edges_for_year().with_columns(pl.lit(1.5).alias("supplier_weight"))
    results = validate_edge_state_panel(edge)
    assert any(result.check_name == "edge_supplier_weight_bounds" and result.n_failed > 0 for result in results)


def test_build_supplier_candidates_retains_top_minimum() -> None:
    candidates = build_supplier_candidates_for_year(
        _candidate_edges(),
        _candidate_accounting(),
        min_top_suppliers=10,
        supplier_weight_threshold=0.9,
        coverage_target=0.99,
    )
    historical = candidates.filter(pl.col("candidate_source") == "historical_t")
    assert historical.height >= 10
    assert historical.filter(pl.col("candidate_rank_within_buyer") <= 10).height == 10


def test_build_supplier_candidates_retains_weight_threshold_edges() -> None:
    candidates = build_supplier_candidates_for_year(
        _candidate_edges(values=[90.0, 5.0, 4.0, 1.0]),
        _candidate_accounting(n_suppliers=4),
        min_top_suppliers=1,
        supplier_weight_threshold=0.04,
        coverage_target=0.5,
    )
    retained_by_weight = candidates.filter(pl.col("retained_by_weight_threshold_flag"))
    assert retained_by_weight.height == 3


def test_build_supplier_candidates_retains_until_coverage_target() -> None:
    candidates = build_supplier_candidates_for_year(
        _candidate_edges(values=[50.0, 30.0, 20.0]),
        _candidate_accounting(n_suppliers=3),
        min_top_suppliers=1,
        supplier_weight_threshold=0.9,
        coverage_target=0.8,
    )
    historical = candidates.filter(pl.col("candidate_source") == "historical_t")
    assert historical.height == 2
    assert historical["buyer_input_cumulative_share"].max() >= 0.8


def test_build_supplier_candidates_respects_hard_cap_and_flags_unmet_coverage() -> None:
    candidates = build_supplier_candidates_for_year(
        _candidate_edges(values=[1.0] * 20),
        _candidate_accounting(n_suppliers=20),
        min_top_suppliers=1,
        hard_max_historical_suppliers=5,
        supplier_weight_threshold=0.0,
        coverage_target=0.9,
        fallback_candidates_per_buyer=0,
    )
    historical = candidates.filter(pl.col("candidate_source") == "historical_t")
    assert historical.height == 5
    assert historical["coverage_target_unmet_flag"].all()


def test_build_supplier_candidates_has_no_duplicate_buyer_supplier_rows() -> None:
    candidates = build_supplier_candidates_for_year(_candidate_edges(), _candidate_accounting())
    duplicates = candidates.group_by(["year", "buyer_country_sector", "supplier_country_sector"]).len().filter(
        pl.col("len") > 1
    )
    assert duplicates.is_empty()


def test_same_sector_fallback_only_added_when_needed() -> None:
    retained = build_supplier_candidates_for_year(
        _candidate_edges(values=[95.0, 5.0]),
        _candidate_accounting(n_suppliers=4),
        min_top_suppliers=1,
        coverage_target=0.9,
    )
    fallback = build_same_sector_fallback_candidates_for_year(
        retained.filter(pl.col("candidate_source") == "historical_t"),
        _candidate_accounting(n_suppliers=4),
        min_top_suppliers=1,
        coverage_target=0.9,
    )
    assert fallback.is_empty()


def test_same_sector_fallback_excludes_existing_historical_suppliers() -> None:
    retained = build_supplier_candidates_for_year(
        _candidate_edges(values=[50.0]),
        _candidate_accounting(n_suppliers=4),
        min_top_suppliers=10,
        coverage_target=0.95,
    )
    fallback = retained.filter(pl.col("candidate_source") == "same_sector_fallback")
    historical_pairs = retained.filter(pl.col("candidate_source") == "historical_t").select(
        "year", "buyer_country_sector", "supplier_country_sector"
    )
    assert fallback.join(historical_pairs, on=["year", "buyer_country_sector", "supplier_country_sector"], how="inner").is_empty()


def test_same_sector_fallback_has_null_transaction_values() -> None:
    retained = build_supplier_candidates_for_year(
        _candidate_edges(values=[50.0]),
        _candidate_accounting(n_suppliers=4),
        min_top_suppliers=10,
        coverage_target=0.95,
    )
    fallback = retained.filter(pl.col("candidate_source") == "same_sector_fallback")
    assert fallback["transaction_value"].null_count() == fallback.height
    assert fallback["supplier_weight"].null_count() == fallback.height
    assert fallback["technical_coefficient"].null_count() == fallback.height


def test_validate_supplier_candidate_panel_requires_columns() -> None:
    results = validate_supplier_candidate_panel(pl.DataFrame({"year": [1995]}))
    assert any(result.status.value == "failed" for result in results)


def test_validate_supplier_candidate_panel_rejects_duplicate_candidates() -> None:
    candidate = build_supplier_candidates_for_year(_candidate_edges(), _candidate_accounting()).head(1)
    results = validate_supplier_candidate_panel(pl.concat([candidate, candidate], how="vertical"))
    assert any(
        result.check_name == "supplier_candidate_unique_year_buyer_supplier" and result.n_failed > 0
        for result in results
    )


def test_validate_supplier_candidate_panel_checks_candidate_source_values() -> None:
    candidates = build_supplier_candidates_for_year(_candidate_edges(), _candidate_accounting()).with_columns(
        pl.lit("bad_source").alias("candidate_source")
    )
    results = validate_supplier_candidate_panel(candidates)
    assert any(result.check_name == "supplier_candidate_source_values" and result.n_failed > 0 for result in results)


def test_summarize_supplier_candidate_coverage_returns_expected_keys() -> None:
    edges = _candidate_edges()
    candidates = build_supplier_candidates_for_year(edges, _candidate_accounting())
    summary = summarize_supplier_candidate_coverage(edges, candidates)
    assert set(summary) == {
        "year",
        "raw_positive_edges",
        "retained_historical_candidate_rows",
        "fallback_candidate_rows",
        "total_candidate_rows",
        "retained_edge_share",
        "raw_transaction_value_total",
        "retained_historical_transaction_value_total",
        "retained_transaction_value_coverage",
        "mean_buyer_input_coverage",
        "share_buyer_years_reaching_coverage_target",
        "max_candidates_per_buyer_year",
        "buyers_with_coverage_target_unmet",
        "buyers_with_fallback_candidates",
    }


def test_build_network_state_for_year_computes_supplier_and_buyer_counts() -> None:
    network = build_network_state_for_year(_weighted_edges_for_year(), _accounting_for_year(1995))
    node_a = network.filter(pl.col("country_sector") == LABELS[0]).row(0, named=True)
    assert node_a["supplier_count"] == 2
    assert node_a["buyer_count"] == 2


def test_build_network_state_for_year_computes_hhi_values() -> None:
    network = build_network_state_for_year(_weighted_edges_for_year(), _accounting_for_year(1995))
    node_a = network.filter(pl.col("country_sector") == LABELS[0]).row(0, named=True)
    assert node_a["supplier_concentration_hhi"] == pytest.approx((5.0 / 25.0) ** 2 + (20.0 / 25.0) ** 2)
    assert node_a["supplier_lock_in"] == pytest.approx(node_a["supplier_concentration_hhi"])


def test_build_network_state_for_year_computes_network_green_exposure() -> None:
    network = build_network_state_for_year(_weighted_edges_for_year(), _accounting_for_year(1995))
    node_b = network.filter(pl.col("country_sector") == LABELS[1]).row(0, named=True)
    assert node_b["incoming_network_green_exposure"] == pytest.approx(0.2)
    assert 0.0 <= node_b["network_green_exposure"] <= 1.0


def test_build_network_state_for_year_computes_import_export_dependence() -> None:
    network = build_network_state_for_year(_weighted_edges_for_year(), _accounting_for_year(1995))
    node_a = network.filter(pl.col("country_sector") == LABELS[0]).row(0, named=True)
    assert node_a["import_dependence_proxy"] == pytest.approx(20.0 / 25.0)
    assert node_a["export_dependence_proxy"] == pytest.approx(10.0 / 15.0)


def test_validate_network_state_panel_requires_columns() -> None:
    results = validate_network_state_panel(pl.DataFrame({"country_sector": [LABELS[0]]}))
    assert any(result.status.value == "failed" for result in results)


def test_validate_network_state_panel_detects_duplicate_country_sector_year() -> None:
    network = build_network_state_for_year(_weighted_edges_for_year(), _accounting_for_year(1995))
    results = validate_network_state_panel(pl.concat([network.head(1), network.head(1)], how="vertical"))
    assert any(result.check_name == "network_unique_country_sector_year" and result.n_failed > 0 for result in results)


def test_validate_network_state_panel_checks_bounded_columns() -> None:
    network = build_network_state_for_year(_weighted_edges_for_year(), _accounting_for_year(1995)).with_columns(
        pl.lit(2.0).alias("network_green_exposure")
    )
    results = validate_network_state_panel(network)
    assert any(result.check_name == "network_network_green_exposure_bounds" and result.n_failed > 0 for result in results)


def test_build_network_state_panels_requires_accounting_panel(tmp_path: Path) -> None:
    _write_pyproject(tmp_path)
    identity_dir = tmp_path / "data" / "abm_v5" / "inputs"
    identity_dir.mkdir(parents=True)
    pl.DataFrame({"country_sector": [LABELS[0]]}).write_parquet(identity_dir / "agent_identity.parquet")
    with pytest.raises(FileNotFoundError, match="Phase 2.3"):
        build_network_state_panels(tmp_path)


def test_build_network_state_panels_writes_supplier_candidate_panel_not_dense_edge_panel(tmp_path: Path) -> None:
    _write_phase2_backbone(tmp_path)
    result = build_network_state_panels(tmp_path)
    assert isinstance(result, NetworkBuildResult)
    assert result.candidate_output_path.exists()
    assert result.network_output_path.exists()
    assert result.validation_path.exists()
    assert result.coverage_summary_path.exists()
    assert result.edge_output_path == result.candidate_output_path
    assert result.candidate_output_path.name == "supplier_candidate_panel_1995_2016.parquet"
    assert not (tmp_path / "data" / "abm_v5" / "supplier_network" / "edge_state_panel_1995_2016.parquet").exists()
    candidates = load_supplier_candidate_panel(result.candidate_output_path)
    assert candidates.height > 0
    assert candidates["compact_candidate_flag"].all()


def test_network_state_panel_still_uses_country_sector_year_backbone(tmp_path: Path) -> None:
    _write_phase2_backbone(tmp_path)
    result = build_network_state_panels(tmp_path)
    assert load_network_state_panel(result.network_output_path).height == 2 * 22


def test_summarize_network_state_returns_expected_keys() -> None:
    network = build_network_state_for_year(_weighted_edges_for_year(), _accounting_for_year(1995))
    summary = summarize_network_state(network)
    assert set(summary) == {
        "mean_supplier_count",
        "mean_buyer_count",
        "mean_supplier_concentration_hhi",
        "mean_buyer_concentration_hhi",
        "mean_import_dependence_proxy",
        "mean_export_dependence_proxy",
        "mean_network_green_exposure",
        "mean_brown_centrality",
        "mean_supplier_lock_in",
    }


def test_init_exports_supplier_candidate_objects() -> None:
    assert abm_v5.NetworkBuildResult is NetworkBuildResult
    assert abm_v5.build_production_edges_for_year is build_production_edges_for_year
    assert abm_v5.add_supplier_weights_and_coefficients is add_supplier_weights_and_coefficients
    assert abm_v5.build_supplier_candidates_for_year is build_supplier_candidates_for_year
    assert abm_v5.build_same_sector_fallback_candidates_for_year is build_same_sector_fallback_candidates_for_year
    assert abm_v5.build_network_state_for_year is build_network_state_for_year
    assert abm_v5.build_network_state_panels is build_network_state_panels
    assert abm_v5.validate_edge_state_panel is validate_edge_state_panel
    assert abm_v5.validate_network_state_panel is validate_network_state_panel
    assert abm_v5.validate_supplier_candidate_panel is validate_supplier_candidate_panel
    assert abm_v5.summarize_supplier_candidate_coverage is summarize_supplier_candidate_coverage
    assert abm_v5.summarize_network_state is summarize_network_state
