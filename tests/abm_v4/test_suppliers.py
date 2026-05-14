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
            "country_sector": ["S1", "S2", "B1", "B2"],
            "Year": [1995, 1995, 1995, 1995],
            "Country": ["SUP", "SUP", "BUY", "BUY"],
            "Sector": ["Inputs", "Inputs", "Output", "Output"],
            "ecosystem_id": ["eco_inputs", "eco_inputs", "eco_output", "eco_output"],
            "ecosystem_label": ["Input ecosystem", "Input ecosystem", "Output ecosystem", "Output ecosystem"],
            "ecosystem_source": ["eora_sector_manual_mapping"] * 4,
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
