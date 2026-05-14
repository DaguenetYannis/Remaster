from pathlib import Path
from uuid import uuid4

import polars as pl

from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.suppliers import SupplierNetworkBuilder, SupplierOpportunity


def toy_root() -> Path:
    return Path("tmp") / "abm_v4_supplier_tests" / uuid4().hex


def toy_state_panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "country_sector": ["S1", "S2", "S3", "B1", "B2"],
            "Year": [1995, 1995, 1995, 1995, 1995],
            "Country": ["SUP", "SUP2", "BUY", "BUY", "BUY"],
            "Sector": ["Inputs", "Inputs", "Output", "Output", "Output"],
            "X_observed": [100.0, 200.0, 150.0, 50.0, 60.0],
            "green_capability": [0.2, 0.8, 0.5, 0.4, 0.3],
            "g_local_v4": [0.2, 0.8, 0.5, 0.4, 0.3],
            "ecosystem_id": ["eco_inputs", "eco_inputs", "eco_output", "eco_output", "eco_other"],
            "ecosystem_label": ["Input ecosystem", "Input ecosystem", "Output ecosystem", "Output ecosystem", "Other ecosystem"],
            "ecosystem_source": ["eora_sector_manual_mapping"] * 5,
        }
    )


def toy_edges_variant() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Year": [1995, 1995, 1995, 1996],
            "source_agent_id": ["S1", "S2", "S1", "S1"],
            "target_agent_id": ["B1", "B1", "B2", "B1"],
            "embedded_emissions": [10.0, 30.0, 5.0, 20.0],
        }
    )


def write_toy_state(paths: ABMV4Paths) -> None:
    path = paths.state_panel_path(1995, 2016)
    path.parent.mkdir(parents=True, exist_ok=True)
    toy_state_panel().write_parquet(path)

def write_toy_opportunity_state(paths: ABMV4Paths) -> None:
    path = paths.state_panel_path(1995, 2016)
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "country_sector": ["B1", "S1", "S2", "S3", "B1", "S1", "S2", "S3"],
            "Year": [1995, 1995, 1995, 1995, 1996, 1996, 1996, 1996],
            "Country": ["BUY", "SUP", "SUP2", "SUP3", "BUY", "SUP", "SUP2", "SUP3"],
            "Sector": ["Output", "Inputs", "Inputs", "Output", "Output", "Inputs", "Inputs", "Output"],
            "X_observed": [100.0, 100.0, 200.0, 100.0, 110.0, 90.0, 220.0, 100.0],
            "g_local_v4": [0.30, 0.20, 0.70, 0.50, 0.35, 0.25, 0.80, 0.55],
            "green_capability": [0.30, 0.20, 0.80, 0.60, 0.35, 0.30, 0.85, 0.65],
            "general_capability": [0.40, 0.50, 0.60, 0.70, 0.45, 0.55, 0.65, 0.75],
            "ecosystem_id": ["eco_output", "eco_inputs", "eco_inputs", "eco_output"] * 2,
            "ecosystem_label": ["Output ecosystem", "Input ecosystem", "Input ecosystem", "Output ecosystem"] * 2,
            "ecosystem_source": ["toy"] * 8,
        }
    ).write_parquet(path)


def toy_opportunity_candidate_pools() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    historical = pl.DataFrame(
        {
            "buyer_country_sector": ["B1", "B1"],
            "supplier_country_sector": ["S1", "S2"],
            "supplier_type": ["historical", "historical"],
            "historical_tie_strength": [0.7, 0.3],
            "mean_historical_share": [0.65, 0.35],
            "total_transaction_value": [70.0, 30.0],
            "supplier_country": ["SUP", "SUP2"],
            "buyer_country": ["BUY", "BUY"],
            "supplier_sector": ["Inputs", "Inputs"],
            "buyer_sector": ["Output", "Output"],
            "supplier_ecosystem_id": ["eco_inputs", "eco_inputs"],
            "buyer_ecosystem_id": ["eco_output", "eco_output"],
            "source_type": ["raw_eora_T_top_historical", "raw_eora_T_top_historical"],
        }
    )
    same_sector = pl.DataFrame(
        {
            "buyer_country_sector": ["B1"],
            "supplier_country_sector": ["S2"],
            "supplier_type": ["same_sector_foreign"],
            "supplier_rank": [1],
            "supplier_sector": ["Inputs"],
            "buyer_sector": ["Output"],
            "supplier_country": ["SUP2"],
            "buyer_country": ["BUY"],
            "supplier_ecosystem_id": ["eco_inputs"],
            "buyer_ecosystem_id": ["eco_output"],
            "total_supplier_output_or_transaction_proxy": [420.0],
            "domestic_fallback_used": [False],
            "source_type": ["same_sector_pool"],
        }
    )
    ecosystem = pl.DataFrame(
        {
            "buyer_country_sector": ["B1", "B1"],
            "supplier_country_sector": ["S2", "S3"],
            "supplier_type": ["ecosystem_feasible", "ecosystem_feasible"],
            "ecosystem_proximity": [0.35, 1.0],
            "supplier_rank": [1, 2],
            "supplier_ecosystem_id": ["eco_inputs", "eco_output"],
            "buyer_ecosystem_id": ["eco_output", "eco_output"],
            "supplier_ecosystem_label": ["Input ecosystem", "Output ecosystem"],
            "buyer_ecosystem_label": ["Output ecosystem", "Output ecosystem"],
            "total_supplier_output_or_transaction_proxy": [420.0, 200.0],
            "candidate_source_flags": ["historical|same_sector", ""],
            "source_type": ["ecosystem_pool", "ecosystem_pool"],
        }
    )
    return historical, same_sector, ecosystem


def write_toy_opportunity_inputs(paths: ABMV4Paths) -> None:
    historical, same_sector, ecosystem = toy_opportunity_candidate_pools()
    paths.interim.mkdir(parents=True, exist_ok=True)
    historical.write_parquet(paths.supplier_candidates_historical_top_path)
    same_sector.write_parquet(paths.supplier_pool_same_sector_path)
    ecosystem.write_parquet(paths.supplier_pool_ecosystem_path)


def toy_supplier_opportunities() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "buyer_country_sector": ["B1", "B1", "B1", "B2", "B2"],
            "supplier_country_sector": ["S1", "S2", "S3", "S1", "S2"],
            "supplier_type": [
                "historical",
                "historical",
                "ecosystem_feasible",
                "ecosystem_feasible",
                "same_sector_foreign",
            ],
            "candidate_sources": [
                "historical",
                "historical;same_sector",
                "ecosystem",
                "ecosystem",
                "same_sector",
            ],
            "choice_probability": [0.2, 0.7, 0.1, 0.4, 0.6],
            "historical_tie_strength": [0.25, 0.75, None, None, None],
        }
    )

def write_toy_t_matrix(paths: ABMV4Paths, year: int = 1995) -> None:
    path = paths.data_root / "parquet" / str(year) / "T.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "__index_level_0__": ["S1", "S2"],
            "B1": [10.0, 0.0],
            "B2": [5.0, 20.0],
        }
    ).write_parquet(path)


def write_toy_raw_t_edges(paths: ABMV4Paths) -> None:
    paths.raw_t_supplier_edges_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "year": [1995, 1995, 1995, 1995, 1995],
            "supplier_country_sector": ["S1", "S2", "S3", "S1", "S2"],
            "buyer_country_sector": ["B1", "B1", "B1", "B2", "B2"],
            "transaction_value": [10.0, 40.0, 20.0, 8.0, 12.0],
            "supplier_country": ["SUP", "SUP2", "BUY", "SUP", "SUP2"],
            "buyer_country": ["BUY", "BUY", "BUY", "BUY", "BUY"],
            "supplier_sector": ["Inputs", "Inputs", "Output", "Inputs", "Inputs"],
            "buyer_sector": ["Output", "Output", "Output", "Output", "Output"],
            "supplier_ecosystem_id": ["eco_inputs", "eco_inputs", "eco_output", "eco_inputs", "eco_inputs"],
            "buyer_ecosystem_id": ["eco_output", "eco_output", "eco_output", "eco_other", "eco_other"],
            "supplier_ecosystem_label": [
                "Input ecosystem",
                "Input ecosystem",
                "Output ecosystem",
                "Input ecosystem",
                "Input ecosystem",
            ],
            "buyer_ecosystem_label": [
                "Output ecosystem",
                "Output ecosystem",
                "Output ecosystem",
                "Other ecosystem",
                "Other ecosystem",
            ],
            "observed_edge": [True] * 5,
            "historical_tie_strength": [0.10, 0.50, 0.25, 0.40, 0.60],
            "historical_share": [0.10, 0.50, 0.25, 0.40, 0.60],
            "source_file": ["toy"] * 5,
            "source_type": ["raw_eora_T"] * 5,
        }
    ).write_parquet(paths.raw_t_supplier_edges_path)


def write_toy_ecosystem_adjacency(paths: ABMV4Paths) -> None:
    paths.ecosystem_adjacency_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "ecosystem_id_from": ["eco_output", "eco_output", "eco_output", "eco_other"],
            "ecosystem_id_to": ["eco_output", "eco_inputs", "eco_other", "eco_other"],
            "proximity": [1.0, 0.35, 0.0, 1.0],
            "relation_type": ["same", "adjacent", "non_adjacent", "same"],
        }
    ).write_csv(paths.ecosystem_adjacency_path)


def test_supplier_opportunity_uses_explicit_type_vocabulary() -> None:
    opportunity = SupplierOpportunity(
        buyer_country_sector="FRA|Agriculture",
        supplier_country_sector="DEU|Agriculture",
        supplier_type="same_sector_foreign",
        friction=0.50,
    )

    assert opportunity.is_supported_type()


def test_normalize_edge_schema_preserves_supplier_to_buyer_direction() -> None:
    builder = SupplierNetworkBuilder(paths=ABMV4Paths(project_root=toy_root()))

    normalized, mappings = builder.normalize_edge_schema(toy_edges_variant())

    assert normalized.columns == [
        "year",
        "supplier_country_sector",
        "buyer_country_sector",
        "transaction_value",
    ]
    assert normalized.row(0, named=True) == {
        "year": 1995,
        "supplier_country_sector": "S1",
        "buyer_country_sector": "B1",
        "transaction_value": 10.0,
    }
    assert any(mapping.source_column == "source_agent_id" for mapping in mappings)


def test_historical_share_and_tie_strength_are_computed() -> None:
    builder = SupplierNetworkBuilder(paths=ABMV4Paths(project_root=toy_root()))
    normalized, _ = builder.normalize_edge_schema(toy_edges_variant())

    edges = builder.compute_historical_ties(normalized)
    row = edges.filter(
        (pl.col("year") == 1995)
        & (pl.col("supplier_country_sector") == "S1")
        & (pl.col("buyer_country_sector") == "B1")
    ).row(0, named=True)

    assert row["historical_share"] == 0.25
    assert row["historical_tie_strength"] == 0.5


def test_supplier_metadata_is_attached_and_missing_metadata_reported() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    builder = SupplierNetworkBuilder(paths=paths)
    state_panel = toy_state_panel()
    edge_panel = pl.DataFrame(
        {
            "year": [1995, 1995],
            "supplier_country_sector": ["S1", "MISSING_SUPPLIER"],
            "buyer_country_sector": ["B1", "B1"],
            "transaction_value": [1.0, 2.0],
        }
    )

    attached = builder.attach_supplier_buyer_metadata(edge_panel, state_panel)
    report = builder.build_edge_report(
        attached.with_columns(
            pl.lit(True).alias("observed_edge"),
            pl.lit(1.0).alias("historical_tie_strength"),
            pl.lit(1.0).alias("historical_share"),
            pl.lit("toy").alias("source_file"),
            pl.lit("toy").alias("source_type"),
        ),
        state_panel,
        selected_source=builder.discover_edge_sources()[0]
        if builder.discover_edge_sources()
        else type("Source", (), {"path": Path("toy"), "source_type": "toy", "notes": "toy"})(),
    )

    assert attached["supplier_sector"].null_count() == 1
    assert report["share_edges_with_supplier_metadata"].item() == 0.5
    assert report["share_edges_with_buyer_metadata"].item() == 1.0


def test_raw_t_matrix_rows_are_suppliers_and_columns_are_buyers() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_state(paths)
    write_toy_t_matrix(paths)

    raw_edges = SupplierNetworkBuilder(paths=paths).build_edges_from_eora_T([1995])

    assert raw_edges.select(
        "year",
        "supplier_country_sector",
        "buyer_country_sector",
        "transaction_value",
    ).sort("supplier_country_sector", "buyer_country_sector").to_dicts() == [
        {
            "year": 1995,
            "supplier_country_sector": "S1",
            "buyer_country_sector": "B1",
            "transaction_value": 10.0,
        },
        {
            "year": 1995,
            "supplier_country_sector": "S1",
            "buyer_country_sector": "B2",
            "transaction_value": 5.0,
        },
        {
            "year": 1995,
            "supplier_country_sector": "S2",
            "buyer_country_sector": "B2",
            "transaction_value": 20.0,
        },
    ]


def test_raw_t_builder_converts_positive_entries_and_drops_zero_entries() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_state(paths)
    write_toy_t_matrix(paths)

    raw_edges = SupplierNetworkBuilder(paths=paths).build_edges_from_eora_T([1995])

    assert raw_edges.height == 3
    assert not raw_edges.filter(
        (pl.col("supplier_country_sector") == "S2")
        & (pl.col("buyer_country_sector") == "B1")
    ).height


def test_raw_t_edge_builder_does_not_use_embedded_emissions() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_state(paths)
    write_toy_t_matrix(paths)
    legacy_path = paths.data_abm_legacy / "edges_panel.parquet"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "Year": [1995],
            "source_agent_id": ["S1"],
            "target_agent_id": ["B1"],
            "embedded_emissions": [9999.0],
        }
    ).write_parquet(legacy_path)

    raw_edges = SupplierNetworkBuilder(paths=paths).build_edges_from_eora_T([1995])

    assert raw_edges.filter(
        (pl.col("supplier_country_sector") == "S1")
        & (pl.col("buyer_country_sector") == "B1")
    )["transaction_value"].item() == 10.0
    assert raw_edges["source_type"].unique().to_list() == ["raw_eora_T"]


def test_raw_t_historical_share_is_computed_by_buyer_year() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_state(paths)
    write_toy_t_matrix(paths)

    raw_edges = SupplierNetworkBuilder(paths=paths).build_edges_from_eora_T([1995])
    row = raw_edges.filter(
        (pl.col("supplier_country_sector") == "S1")
        & (pl.col("buyer_country_sector") == "B2")
    ).row(0, named=True)

    assert row["historical_share"] == 0.2


def test_raw_t_historical_tie_strength_is_computed_across_years() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_state(paths)
    write_toy_t_matrix(paths, year=1995)
    path_1996 = paths.data_root / "parquet" / "1996" / "T.parquet"
    path_1996.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "__index_level_0__": ["S1", "S2"],
            "B1": [30.0, 10.0],
            "B2": [0.0, 0.0],
        }
    ).write_parquet(path_1996)

    raw_edges = SupplierNetworkBuilder(paths=paths).build_edges_from_eora_T([1995, 1996])
    row = raw_edges.filter(
        (pl.col("year") == 1995)
        & (pl.col("supplier_country_sector") == "S1")
        & (pl.col("buyer_country_sector") == "B1")
    ).row(0, named=True)

    assert row["historical_tie_strength"] == 0.8


def test_raw_t_metadata_is_attached_correctly() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_state(paths)
    write_toy_t_matrix(paths)

    raw_edges = SupplierNetworkBuilder(paths=paths).build_edges_from_eora_T([1995])
    row = raw_edges.filter(
        (pl.col("supplier_country_sector") == "S1")
        & (pl.col("buyer_country_sector") == "B1")
    ).row(0, named=True)

    assert row["supplier_country"] == "SUP"
    assert row["buyer_country"] == "BUY"
    assert row["supplier_sector"] == "Inputs"
    assert row["buyer_sector"] == "Output"
    assert row["supplier_ecosystem_id"] == "eco_inputs"
    assert row["buyer_ecosystem_id"] == "eco_output"


def test_edge_source_comparison_distinguishes_raw_t_and_legacy_sources() -> None:
    builder = SupplierNetworkBuilder(paths=ABMV4Paths(project_root=toy_root()))
    raw_edges = pl.DataFrame(
        {
            "year": [1995, 1995],
            "supplier_country_sector": ["S1", "S2"],
            "buyer_country_sector": ["B1", "B1"],
            "transaction_value": [10.0, 20.0],
            "supplier_sector": ["Inputs", "Inputs"],
            "buyer_sector": ["Output", "Output"],
            "source_type": ["raw_eora_T", "raw_eora_T"],
        }
    )
    legacy_edges = pl.DataFrame(
        {
            "year": [1995],
            "supplier_country_sector": ["S1"],
            "buyer_country_sector": ["B1"],
            "transaction_value": [999.0],
            "supplier_sector": ["Inputs"],
            "buyer_sector": ["Output"],
            "source_type": ["legacy_abm_edges_embodied_emissions"],
        }
    )

    comparison = builder.compare_edge_sources(raw_edges, legacy_edges)

    assert comparison["source_type"].to_list() == [
        "raw_eora_T",
        "legacy_abm_edges_embodied_emissions",
    ]
    assert comparison["pair_overlap_count"].to_list() == [1, 1]
    assert comparison["pair_overlap_share_raw"].to_list() == [0.5, 0.5]
    assert comparison["pair_overlap_share_legacy"].to_list() == [1.0, 1.0]
    assert comparison["recommended_use"].to_list() == [
        "canonical_supplier_substitution_source",
        "carbon_diagnostics_not_input_sourcing_weight",
    ]


def test_raw_t_build_does_not_write_outputs_without_explicit_write() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_state(paths)
    write_toy_t_matrix(paths)

    SupplierNetworkBuilder(paths=paths).build_edges_from_eora_T([1995])

    assert not paths.raw_t_supplier_edges_path.exists()
    assert not paths.raw_t_supplier_edge_report_path.exists()
    assert not paths.supplier_edge_source_comparison_path.exists()


def test_historical_candidates_are_capped_by_buyer() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_state(paths)
    write_toy_raw_t_edges(paths)

    candidates = SupplierNetworkBuilder(paths=paths).build_historical_top_supplier_candidates(
        max_historical_suppliers_per_buyer=2
    )

    assert candidates.group_by("buyer_country_sector").len()["len"].max() == 2


def test_historical_candidates_are_ranked_by_tie_strength() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_state(paths)
    write_toy_raw_t_edges(paths)

    candidates = SupplierNetworkBuilder(paths=paths).build_historical_top_supplier_candidates(
        max_historical_suppliers_per_buyer=2
    )
    b1_suppliers = candidates.filter(pl.col("buyer_country_sector") == "B1")[
        "supplier_country_sector"
    ].to_list()

    assert b1_suppliers == ["S2", "S3"]


def test_same_sector_candidates_exclude_the_buyer_itself() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_state(paths)

    pool = SupplierNetworkBuilder(paths=paths).build_same_sector_supplier_pool()

    assert pool.filter(pl.col("buyer_country_sector") == pl.col("supplier_country_sector")).is_empty()


def test_same_sector_candidates_respect_candidate_cap() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_state(paths)

    pool = SupplierNetworkBuilder(paths=paths).build_same_sector_supplier_pool(
        max_same_sector_candidates_per_buyer=1
    )

    assert pool.group_by("buyer_country_sector").len()["len"].max() == 1


def test_ecosystem_candidates_use_same_and_adjacent_ecosystems_only() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_state(paths)
    write_toy_ecosystem_adjacency(paths)

    pool = SupplierNetworkBuilder(paths=paths).build_ecosystem_supplier_pool()
    b1_ecosystems = set(
        pool.filter(pl.col("buyer_country_sector") == "B1")[
            "supplier_ecosystem_id"
        ].to_list()
    )

    assert b1_ecosystems == {"eco_output", "eco_inputs"}
    assert "eco_other" not in b1_ecosystems


def test_ecosystem_candidates_respect_candidate_cap() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_state(paths)
    write_toy_ecosystem_adjacency(paths)

    pool = SupplierNetworkBuilder(paths=paths).build_ecosystem_supplier_pool(
        max_ecosystem_candidates_per_buyer=1
    )

    assert pool.group_by("buyer_country_sector").len()["len"].max() == 1


def test_ecosystem_duplicate_candidate_handling_is_explicit() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_state(paths)
    write_toy_ecosystem_adjacency(paths)
    historical = pl.DataFrame(
        {
            "buyer_country_sector": ["B1"],
            "supplier_country_sector": ["S3"],
        }
    )
    same_sector = pl.DataFrame(
        {
            "buyer_country_sector": ["B1"],
            "supplier_country_sector": ["S3"],
        }
    )

    pool = SupplierNetworkBuilder(paths=paths).build_ecosystem_supplier_pool(
        historical_candidates=historical,
        same_sector_candidates=same_sector,
    )
    duplicate_row = pool.filter(
        (pl.col("buyer_country_sector") == "B1")
        & (pl.col("supplier_country_sector") == "S3")
    ).row(0, named=True)

    assert duplicate_row["candidate_source_flags"] == "historical|same_sector"


def test_supplier_candidate_base_report_counts_candidate_types() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_state(paths)
    builder = SupplierNetworkBuilder(paths=paths)
    historical = pl.DataFrame(
        {"buyer_country_sector": ["B1"], "supplier_country_sector": ["S1"]}
    )
    same_sector = pl.DataFrame(
        {"buyer_country_sector": ["B1", "B2"], "supplier_country_sector": ["S3", "S3"]}
    )
    ecosystem = pl.DataFrame(
        {"buyer_country_sector": ["B1"], "supplier_country_sector": ["S2"]}
    )

    report = builder.build_supplier_candidate_base_report(historical, same_sector, ecosystem)
    row = report.row(0, named=True)

    assert row["historical_candidate_rows"] == 1
    assert row["same_sector_candidate_rows"] == 2
    assert row["ecosystem_candidate_rows"] == 1


def test_candidate_builders_do_not_create_all_to_all_matrix() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_state(paths)

    pool = SupplierNetworkBuilder(paths=paths).build_same_sector_supplier_pool(
        max_same_sector_candidates_per_buyer=1
    )

    assert pool.height <= toy_state_panel().height


def test_candidate_pools_merge_correctly() -> None:
    builder = SupplierNetworkBuilder(paths=ABMV4Paths(project_root=toy_root()))
    historical, same_sector, ecosystem = toy_opportunity_candidate_pools()

    merged = builder.merge_candidate_pools(historical, same_sector, ecosystem)

    assert merged.height == 5
    assert set(merged["candidate_source"].to_list()) == {"historical", "same_sector", "ecosystem"}


def test_duplicate_buyer_supplier_candidates_resolve_to_one_row() -> None:
    builder = SupplierNetworkBuilder(paths=ABMV4Paths(project_root=toy_root()))
    historical, same_sector, ecosystem = toy_opportunity_candidate_pools()
    merged = builder.merge_candidate_pools(historical, same_sector, ecosystem)

    deduplicated = builder.deduplicate_candidates(merged)

    assert deduplicated.filter(
        (pl.col("buyer_country_sector") == "B1")
        & (pl.col("supplier_country_sector") == "S2")
    ).height == 1


def test_candidate_source_flags_are_preserved() -> None:
    builder = SupplierNetworkBuilder(paths=ABMV4Paths(project_root=toy_root()))
    historical, same_sector, ecosystem = toy_opportunity_candidate_pools()
    deduplicated = builder.deduplicate_candidates(
        builder.merge_candidate_pools(historical, same_sector, ecosystem)
    )
    row = deduplicated.filter(pl.col("supplier_country_sector") == "S2").row(0, named=True)

    assert row["candidate_sources"] == "historical;same_sector;ecosystem"
    assert row["is_historical_candidate"]
    assert row["is_same_sector_candidate"]
    assert row["is_ecosystem_candidate"]


def test_supplier_type_priority_is_historical_then_same_sector_then_ecosystem() -> None:
    builder = SupplierNetworkBuilder(paths=ABMV4Paths(project_root=toy_root()))
    historical, same_sector, ecosystem = toy_opportunity_candidate_pools()
    same_only = same_sector.with_columns(pl.lit("S4").alias("supplier_country_sector"))
    deduplicated = builder.deduplicate_candidates(
        builder.merge_candidate_pools(historical, same_only, ecosystem)
    )

    assert deduplicated.filter(pl.col("supplier_country_sector") == "S2")[
        "supplier_type"
    ].item() == "historical"
    assert deduplicated.filter(pl.col("supplier_country_sector") == "S4")[
        "supplier_type"
    ].item() == "same_sector_foreign"
    assert deduplicated.filter(pl.col("supplier_country_sector") == "S3")[
        "supplier_type"
    ].item() == "ecosystem_feasible"


def test_supplier_friction_follows_hierarchy() -> None:
    builder = SupplierNetworkBuilder(paths=ABMV4Paths(project_root=toy_root()))
    frame = pl.DataFrame(
        {"supplier_type": ["historical", "same_sector_foreign", "ecosystem_feasible"]}
    )

    with_friction = builder.compute_supplier_friction(frame)

    assert with_friction["friction"].to_list() == [0.10, 0.50, 1.00]


def test_green_advantage_is_computed() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_opportunity_state(paths)
    write_toy_opportunity_inputs(paths)

    opportunities = SupplierNetworkBuilder(paths=paths).build_supplier_opportunity_sets()
    row = opportunities.filter(pl.col("supplier_country_sector") == "S2").row(0, named=True)

    assert row["green_advantage"] > 0


def test_supplier_reliability_is_computed_from_two_years_of_output() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_opportunity_state(paths)
    write_toy_opportunity_inputs(paths)

    opportunities = SupplierNetworkBuilder(paths=paths).build_supplier_opportunity_sets()
    s1 = opportunities.filter(pl.col("supplier_country_sector") == "S1").row(0, named=True)

    assert abs(s1["supplier_reliability"] - 0.9) < 1e-9


def test_supplier_attractiveness_penalizes_friction() -> None:
    builder = SupplierNetworkBuilder(paths=ABMV4Paths(project_root=toy_root()))
    base = pl.DataFrame(
        {
            "green_advantage": [0.0, 0.0],
            "supplier_reliability": [0.0, 0.0],
            "supplier_green_capability": [0.0, 0.0],
            "supplier_general_capability": [0.0, 0.0],
            "historical_tie_strength": [0.0, 0.0],
            "ecosystem_proximity": [0.0, 0.0],
            "friction": [0.1, 1.0],
        }
    )

    scored = builder.compute_supplier_attractiveness(base)

    assert scored["supplier_attractiveness"][0] > scored["supplier_attractiveness"][1]


def test_choice_probabilities_sum_to_one_by_buyer() -> None:
    builder = SupplierNetworkBuilder(paths=ABMV4Paths(project_root=toy_root()))
    frame = pl.DataFrame(
        {
            "buyer_country_sector": ["B1", "B1", "B2"],
            "supplier_attractiveness": [1.0, 2.0, 0.5],
        }
    )

    probabilities = builder.compute_choice_probabilities(frame)
    sums = probabilities.group_by("buyer_country_sector").agg(pl.sum("choice_probability"))

    assert all(abs(value - 1.0) < 1e-12 for value in sums["choice_probability"].to_list())


def test_opportunity_report_detects_probability_sum_errors() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_opportunity_state(paths)
    builder = SupplierNetworkBuilder(paths=paths)
    bad = pl.DataFrame(
        {
            "buyer_country_sector": ["B1", "B1"],
            "choice_probability": [0.2, 0.2],
            "is_historical_candidate": [True, False],
            "is_same_sector_candidate": [False, True],
            "is_ecosystem_candidate": [False, False],
            "duplicated_candidate_before_resolution": [False, False],
            "friction": [0.1, 0.5],
            "green_advantage": [0.0, 0.1],
            "supplier_attractiveness": [1.0, 2.0],
        }
    )

    report = builder.build_opportunity_set_report(bad)

    assert report["buyers_with_probability_sum_error"].item() == 1


def test_opportunity_build_does_not_write_without_explicit_write() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_opportunity_state(paths)
    write_toy_opportunity_inputs(paths)

    SupplierNetworkBuilder(paths=paths).build_supplier_opportunity_sets()

    assert not paths.supplier_opportunity_sets_path.exists()
    assert not paths.supplier_opportunity_set_report_path.exists()


def test_initial_weights_use_historical_tie_strength() -> None:
    builder = SupplierNetworkBuilder(paths=ABMV4Paths(project_root=toy_root()))

    weights = builder.build_supplier_initial_weights(toy_supplier_opportunities())

    assert weights.filter(
        (pl.col("buyer_country_sector") == "B1")
        & (pl.col("supplier_country_sector") == "S1")
    )["initial_weight"].item() == 0.25
    assert weights.filter(
        (pl.col("buyer_country_sector") == "B1")
        & (pl.col("supplier_country_sector") == "S2")
    )["initial_weight"].item() == 0.75


def test_non_historical_candidates_initialize_at_zero() -> None:
    builder = SupplierNetworkBuilder(paths=ABMV4Paths(project_root=toy_root()))

    weights = builder.build_supplier_initial_weights(toy_supplier_opportunities())

    assert weights.filter(
        (pl.col("buyer_country_sector") == "B1")
        & (pl.col("supplier_country_sector") == "S3")
    )["initial_weight"].item() == 0.0


def test_initial_weights_sum_to_one_by_buyer() -> None:
    builder = SupplierNetworkBuilder(paths=ABMV4Paths(project_root=toy_root()))

    weights = builder.build_supplier_initial_weights(toy_supplier_opportunities())
    sums = weights.group_by("buyer_country_sector").agg(pl.sum("initial_weight").alias("sum"))

    assert all(abs(value - 1.0) < 1e-12 for value in sums["sum"].to_list())


def test_initial_weight_fallback_uses_choice_probability_without_history() -> None:
    builder = SupplierNetworkBuilder(paths=ABMV4Paths(project_root=toy_root()))

    weights = builder.build_supplier_initial_weights(toy_supplier_opportunities())
    b2 = weights.filter(pl.col("buyer_country_sector") == "B2").sort("supplier_country_sector")

    assert b2["fallback_initialization_used"].to_list() == [True, True]
    assert b2["initial_weight"].to_list() == [0.4, 0.6]


def test_rewire_probability_is_clipped_to_unit_interval() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_opportunity_state(paths)
    paths.supplier_opportunity_sets_path.parent.mkdir(parents=True, exist_ok=True)
    toy_supplier_opportunities().write_parquet(paths.supplier_opportunity_sets_path)
    choice = type(
        "Choice",
        (),
        {"p_rewire_base": 2.0, "p_rewire_stress": 0.0, "p_rewire_green_gap": 0.0},
    )()

    flags = SupplierNetworkBuilder(paths=paths).build_supplier_rewiring_flags(
        supplier_choice=choice,
        random_seed=1,
    )

    assert flags["p_rewire"].max() == 1.0
    assert flags["p_rewire"].min() == 1.0


def test_seeded_rewiring_draw_is_reproducible() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_opportunity_state(paths)
    paths.supplier_opportunity_sets_path.parent.mkdir(parents=True, exist_ok=True)
    toy_supplier_opportunities().write_parquet(paths.supplier_opportunity_sets_path)
    builder = SupplierNetworkBuilder(paths=paths)

    first = builder.build_supplier_rewiring_flags(random_seed=7)
    second = builder.build_supplier_rewiring_flags(random_seed=7)

    assert first["random_draw"].to_list() == second["random_draw"].to_list()
    assert first["rewire_flag"].to_list() == second["rewire_flag"].to_list()


def test_updated_weight_equals_initial_weight_when_not_rewired() -> None:
    builder = SupplierNetworkBuilder(paths=ABMV4Paths(project_root=toy_root()))
    initial = builder.build_supplier_initial_weights(toy_supplier_opportunities())
    flags = pl.DataFrame(
        {"buyer_country_sector": ["B1", "B2"], "rewire_flag": [False, False], "p_rewire": [0.0, 0.0]}
    )

    updated = builder.build_supplier_updated_weights(initial, flags)

    assert updated["updated_weight"].to_list() == updated["initial_weight"].to_list()


def test_updated_weight_moves_toward_choice_probability_when_rewired() -> None:
    choice = type("Choice", (), {"lambda_weight_update": 0.5})()
    builder = SupplierNetworkBuilder(paths=ABMV4Paths(project_root=toy_root()))
    initial = builder.build_supplier_initial_weights(toy_supplier_opportunities())
    flags = pl.DataFrame(
        {"buyer_country_sector": ["B1", "B2"], "rewire_flag": [True, False], "p_rewire": [1.0, 0.0]}
    )

    updated = builder.build_supplier_updated_weights(initial, flags, supplier_choice=choice)
    row = updated.filter(
        (pl.col("buyer_country_sector") == "B1")
        & (pl.col("supplier_country_sector") == "S2")
    ).row(0, named=True)

    assert row["updated_weight"] < row["initial_weight"]
    assert row["updated_weight"] > row["choice_probability"]


def test_updated_weights_sum_to_one_by_buyer() -> None:
    builder = SupplierNetworkBuilder(paths=ABMV4Paths(project_root=toy_root()))
    initial = builder.build_supplier_initial_weights(toy_supplier_opportunities())
    flags = pl.DataFrame(
        {"buyer_country_sector": ["B1", "B2"], "rewire_flag": [True, True], "p_rewire": [1.0, 1.0]}
    )

    updated = builder.build_supplier_updated_weights(initial, flags)
    sums = updated.group_by("buyer_country_sector").agg(pl.sum("updated_weight").alias("sum"))

    assert all(abs(value - 1.0) < 1e-12 for value in sums["sum"].to_list())


def test_rewiring_diagnostics_detect_weight_sum_errors() -> None:
    builder = SupplierNetworkBuilder(paths=ABMV4Paths(project_root=toy_root()))
    initial = pl.DataFrame(
        {
            "buyer_country_sector": ["B1"],
            "initial_weight": [0.5],
            "fallback_initialization_used": [False],
        }
    )
    flags = pl.DataFrame(
        {
            "buyer_country_sector": ["B1"],
            "rewire_flag": [False],
            "p_rewire": [0.0],
            "fallback_stress_used": [True],
            "fallback_green_gap_used": [True],
        }
    )
    updated = pl.DataFrame(
        {"buyer_country_sector": ["B1"], "updated_weight": [0.5], "weight_delta": [0.0]}
    )

    report = builder.build_supplier_rewiring_report(initial, flags, updated)

    assert report["buyers_with_initial_weight_sum_error"].item() == 1
    assert report["buyers_with_updated_weight_sum_error"].item() == 1


def test_build_historical_edges_does_not_create_all_to_all_edges() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_state(paths)
    legacy_path = paths.data_abm_legacy / "edges_panel.parquet"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    toy_edges_variant().write_parquet(legacy_path)

    result = SupplierNetworkBuilder(paths=paths).build_historical_edges()

    assert result.edges.height == 4
    assert result.edges.select("supplier_country_sector", "buyer_country_sector").unique().height == 3


def test_write_historical_edges_only_writes_supplier_outputs() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_state(paths)
    legacy_path = paths.data_abm_legacy / "edges_panel.parquet"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    toy_edges_variant().write_parquet(legacy_path)
    builder = SupplierNetworkBuilder(paths=paths)

    result = builder.build_historical_edges()
    written = builder.write_historical_edges(result)

    assert written.output_path == paths.historical_supplier_edges_path
    assert paths.historical_supplier_edges_path.exists()
    assert paths.supplier_edge_report_path.exists()
    assert paths.supplier_edge_schema_report_path.exists()
    assert not paths.simulations.exists()
    assert not paths.scenarios.exists()
