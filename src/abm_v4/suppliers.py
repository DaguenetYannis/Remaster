from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import heapq
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl

from src.abm_v4.paths import ABMV4Paths


SUPPLIER_TYPES: tuple[str, ...] = (
    "historical",
    "same_sector_foreign",
    "ecosystem_feasible",
)


EDGE_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "year": ("year", "Year"),
    "supplier_country_sector": (
        "supplier_country_sector",
        "source_agent_id",
        "source_country_sector",
        "source",
        "supplier",
    ),
    "buyer_country_sector": (
        "buyer_country_sector",
        "target_agent_id",
        "target_country_sector",
        "target",
        "buyer",
    ),
    "transaction_value": (
        "transaction_value",
        "value",
        "flow",
        "T_value",
        "embedded_emissions",
    ),
}


@dataclass(frozen=True)
class SupplierOpportunity:
    """Potential supplier relation considered by ABM v4."""

    buyer_country_sector: str
    supplier_country_sector: str
    supplier_type: str
    friction: float

    def is_supported_type(self) -> bool:
        return self.supplier_type in SUPPLIER_TYPES


@dataclass(frozen=True)
class EdgeSource:
    """Selected supplier-edge source metadata."""

    path: Path
    source_type: str
    notes: str


@dataclass(frozen=True)
class EdgeColumnMapping:
    """How a source edge column maps to the ABM v4 edge schema."""

    canonical_column: str
    source_column: str | None
    mapping_status: str
    notes: str


@dataclass(frozen=True)
class SupplierEdgeBuildResult:
    """Inspectable result of historical supplier-edge construction."""

    edges: pl.DataFrame
    selected_source: EdgeSource
    schema_report: pl.DataFrame
    edge_report: pl.DataFrame
    output_path: Path | None


class SupplierNetworkBuilder:
    """Build observed historical supplier-buyer edges for ABM v4."""

    def __init__(
        self,
        paths: ABMV4Paths,
        start_year: int = 1995,
        end_year: int = 2016,
        min_transaction_threshold: float = 0.0,
    ) -> None:
        self.paths = paths
        self.start_year = start_year
        self.end_year = end_year
        self.min_transaction_threshold = min_transaction_threshold

    def load_state_panel(self) -> pl.DataFrame:
        """Load the ABM v4 state panel required for node metadata."""
        state_path = self.paths.state_panel_path(self.start_year, self.end_year)
        if not state_path.exists():
            raise FileNotFoundError(f"ABM v4 state panel not found: {state_path}")
        return pl.read_parquet(state_path)

    def discover_edge_sources(self) -> tuple[EdgeSource, ...]:
        """Return available edge source candidates in priority order."""
        candidates: list[EdgeSource] = []

        # ABM v3 Leontief folders currently contain diagnostics/outputs, not a clear
        # supplier-buyer edge panel with transaction values.
        v3_edge_candidates = (
            self.paths.data_abm_v3 / "leontief" / "historical_supplier_edges.parquet",
            self.paths.data_abm_v3 / "edges_panel.parquet",
        )
        for candidate_path in v3_edge_candidates:
            if candidate_path.exists():
                candidates.append(
                    EdgeSource(
                        path=candidate_path,
                        source_type="abm_v3_edge_panel",
                        notes="ABM v3 candidate edge panel.",
                    )
                )

        legacy_edges = self.paths.data_abm_legacy / "edges_panel.parquet"
        if legacy_edges.exists():
            candidates.append(
                EdgeSource(
                    path=legacy_edges,
                    source_type="legacy_abm_edges_embodied_emissions",
                    notes=(
                        "Legacy ABM edge panel built from ET matrices. Direction is "
                        "source_agent_id -> target_agent_id; value is embedded_emissions, "
                        "not raw T transaction value."
                    ),
                )
            )

        t_candidates = [
            self.paths.data_root / "parquet" / str(year) / "T.parquet"
            for year in range(self.start_year, self.end_year + 1)
        ]
        if any(candidate_path.exists() for candidate_path in t_candidates):
            candidates.append(
                EdgeSource(
                    path=self.paths.data_root / "parquet",
                    source_type="eora_T_matrices",
                    notes="Raw Eora T fallback with row=supplier and column=buyer.",
                )
            )

        return tuple(candidates)

    def load_best_edge_source(self) -> tuple[pl.DataFrame, EdgeSource, tuple[EdgeColumnMapping, ...]]:
        """Load the best available supplier-edge source and normalize its schema."""
        for edge_source in self.discover_edge_sources():
            if edge_source.source_type in {"abm_v3_edge_panel", "legacy_abm_edges_embodied_emissions"}:
                source_edges = pl.read_parquet(edge_source.path)
                normalized_edges, mappings = self.normalize_edge_schema(source_edges)
                if normalized_edges.height > 0:
                    return normalized_edges, edge_source, mappings
            if edge_source.source_type == "eora_T_matrices":
                edges = self._build_core_edges_from_eora_T(
                    years=range(self.start_year, self.end_year + 1)
                )
                mappings = (
                    EdgeColumnMapping("year", "year", "derived", "Year from folder name."),
                    EdgeColumnMapping(
                        "supplier_country_sector",
                        "__index_level_0__",
                        "direct",
                        "Rows are suppliers in Eora T.",
                    ),
                    EdgeColumnMapping(
                        "buyer_country_sector",
                        "T matrix columns",
                        "unpivoted",
                        "Columns are buyers in Eora T.",
                    ),
                    EdgeColumnMapping(
                        "transaction_value",
                        "T matrix values",
                        "unpivoted",
                        "Observed transaction values from T_{supplier,buyer}.",
                    ),
                )
                if edges.height > 0:
                    return edges, edge_source, mappings

        raise FileNotFoundError("No interpretable ABM v4 supplier-edge source found.")

    def normalize_edge_schema(
        self,
        edges: pl.DataFrame,
    ) -> tuple[pl.DataFrame, tuple[EdgeColumnMapping, ...]]:
        """Normalize variant edge columns into supplier -> buyer direction."""
        source_columns = set(edges.columns)
        mappings: list[EdgeColumnMapping] = []
        select_expressions: list[pl.Expr] = []

        for canonical_column, candidates in EDGE_COLUMN_CANDIDATES.items():
            source_column = self._first_present_column(source_columns, candidates)
            mappings.append(self._build_mapping(canonical_column, source_column))
            if source_column is None:
                raise ValueError(f"Missing required edge column for {canonical_column}")
            select_expressions.append(pl.col(source_column).alias(canonical_column))

        normalized = (
            edges.select(select_expressions)
            .with_columns(
                pl.col("year").cast(pl.Int64, strict=False),
                pl.col("supplier_country_sector").cast(pl.Utf8),
                pl.col("buyer_country_sector").cast(pl.Utf8),
                pl.col("transaction_value").cast(pl.Float64, strict=False),
            )
            .filter(pl.col("year").is_between(self.start_year, self.end_year))
            .filter(pl.col("transaction_value") > self.min_transaction_threshold)
            .filter(pl.col("supplier_country_sector").is_not_null())
            .filter(pl.col("buyer_country_sector").is_not_null())
        )

        return normalized, tuple(mappings)

    def _empty_raw_t_edges(self) -> pl.DataFrame:
        """Return an empty raw-T edge frame with the canonical raw schema."""
        return pl.DataFrame(
            schema={
                "year": pl.Int64,
                "supplier_country_sector": pl.Utf8,
                "buyer_country_sector": pl.Utf8,
                "transaction_value": pl.Float64,
                "source_file": pl.Utf8,
                "source_type": pl.Utf8,
            }
        )

    def _build_core_edges_from_eora_T(self, years: Iterable[int]) -> pl.DataFrame:
        """Build sparse positive supplier-buyer edges from Eora T matrices."""
        frames: list[pl.DataFrame] = []
        for year in years:
            t_path = self.paths.data_root / "parquet" / str(year) / "T.parquet"
            if not t_path.exists():
                continue
            matrix = pl.read_parquet(t_path)
            if "__index_level_0__" not in matrix.columns:
                raise ValueError(f"Eora T matrix lacks row labels: {t_path}")
            value_columns = [
                column_name
                for column_name in matrix.columns
                if column_name != "__index_level_0__"
            ]
            supplier_labels = matrix["__index_level_0__"].cast(pl.Utf8).to_numpy()
            buyer_labels = np.array(value_columns, dtype=object)
            transaction_matrix = matrix.select(value_columns).to_numpy()
            positive_supplier_indices, positive_buyer_indices = np.nonzero(
                transaction_matrix > self.min_transaction_threshold
            )
            if len(positive_supplier_indices) == 0:
                continue
            sparse_edges = pl.DataFrame(
                {
                    "year": np.full(len(positive_supplier_indices), year),
                    "supplier_country_sector": supplier_labels[positive_supplier_indices],
                    "buyer_country_sector": buyer_labels[positive_buyer_indices],
                    "transaction_value": transaction_matrix[
                        positive_supplier_indices,
                        positive_buyer_indices,
                    ],
                    "source_file": np.full(len(positive_supplier_indices), str(t_path)),
                    "source_type": np.full(len(positive_supplier_indices), "raw_eora_T"),
                }
            )
            frames.append(sparse_edges)

        if not frames:
            return self._empty_raw_t_edges()
        return pl.concat(frames, how="vertical")

    def build_edges_from_eora_T(self, years: Iterable[int]) -> pl.DataFrame:
        """Build metadata-enriched ABM v4 supplier edges from raw Eora T matrices."""
        state_panel = self.load_state_panel()
        raw_edges = self._build_core_edges_from_eora_T(years)
        edges = self.attach_supplier_buyer_metadata(raw_edges, state_panel)
        edges = self.compute_historical_ties(edges)
        return edges.select(
            "year",
            "supplier_country_sector",
            "buyer_country_sector",
            "transaction_value",
            "supplier_country",
            "buyer_country",
            "supplier_sector",
            "buyer_sector",
            "supplier_ecosystem_id",
            "buyer_ecosystem_id",
            "supplier_ecosystem_label",
            "buyer_ecosystem_label",
            "observed_edge",
            "historical_tie_strength",
            "historical_share",
            "source_file",
            "source_type",
        )

    def attach_supplier_buyer_metadata(
        self,
        edges: pl.DataFrame,
        state_panel: pl.DataFrame,
    ) -> pl.DataFrame:
        """Attach country, sector, and ecosystem metadata to both edge endpoints."""
        metadata = (
            state_panel.select(
                "country_sector",
                "Country",
                "Sector",
                "ecosystem_id",
                "ecosystem_label",
            )
            .unique(subset=["country_sector"])
        )
        supplier_metadata = metadata.rename(
            {
                "country_sector": "supplier_country_sector",
                "Country": "supplier_country",
                "Sector": "supplier_sector",
                "ecosystem_id": "supplier_ecosystem_id",
                "ecosystem_label": "supplier_ecosystem_label",
            }
        )
        buyer_metadata = metadata.rename(
            {
                "country_sector": "buyer_country_sector",
                "Country": "buyer_country",
                "Sector": "buyer_sector",
                "ecosystem_id": "buyer_ecosystem_id",
                "ecosystem_label": "buyer_ecosystem_label",
            }
        )
        return edges.join(supplier_metadata, on="supplier_country_sector", how="left").join(
            buyer_metadata,
            on="buyer_country_sector",
            how="left",
        )

    def compute_historical_ties(self, edges: pl.DataFrame) -> pl.DataFrame:
        """Compute buyer-year shares and full-period buyer-supplier tie strengths."""
        return (
            edges.with_columns(
                pl.sum("transaction_value").over(["buyer_country_sector", "year"]).alias(
                    "_buyer_year_total"
                ),
                pl.sum("transaction_value").over(["buyer_country_sector"]).alias(
                    "_buyer_period_total"
                ),
                pl.sum("transaction_value").over(
                    ["buyer_country_sector", "supplier_country_sector"]
                ).alias("_supplier_buyer_period_total"),
            )
            .with_columns(
                pl.when(pl.col("_buyer_year_total") > 0)
                .then(pl.col("transaction_value") / pl.col("_buyer_year_total"))
                .otherwise(None)
                .alias("historical_share"),
                pl.when(pl.col("_buyer_period_total") > 0)
                .then(
                    pl.col("_supplier_buyer_period_total")
                    / pl.col("_buyer_period_total")
                )
                .otherwise(None)
                .alias("historical_tie_strength"),
                (pl.col("transaction_value") > 0).alias("observed_edge"),
            )
            .drop(
                [
                    "_buyer_year_total",
                    "_buyer_period_total",
                    "_supplier_buyer_period_total",
                ]
            )
        )

    def build_historical_edges(self) -> SupplierEdgeBuildResult:
        """Build ABM v4 historical supplier edges without writing outputs."""
        state_panel = self.load_state_panel()
        source_edges, selected_source, mappings = self.load_best_edge_source()
        edges = self.attach_supplier_buyer_metadata(source_edges, state_panel)
        edges = self.compute_historical_ties(edges)
        edges = edges.with_columns(
            pl.lit(str(selected_source.path)).alias("source_file"),
            pl.lit(selected_source.source_type).alias("source_type"),
        ).select(
            "year",
            "supplier_country_sector",
            "buyer_country_sector",
            "transaction_value",
            "supplier_country",
            "buyer_country",
            "supplier_sector",
            "buyer_sector",
            "supplier_ecosystem_id",
            "buyer_ecosystem_id",
            "supplier_ecosystem_label",
            "buyer_ecosystem_label",
            "observed_edge",
            "historical_tie_strength",
            "historical_share",
            "source_file",
            "source_type",
        )
        schema_report = self.build_edge_schema_report(mappings, selected_source)
        edge_report = self.build_edge_report(edges, state_panel, selected_source)
        return SupplierEdgeBuildResult(
            edges=edges,
            selected_source=selected_source,
            schema_report=schema_report,
            edge_report=edge_report,
            output_path=None,
        )

    def write_historical_edges(
        self,
        result: SupplierEdgeBuildResult,
    ) -> SupplierEdgeBuildResult:
        """Write historical supplier edges and diagnostics."""
        self.paths.interim.mkdir(parents=True, exist_ok=True)
        self.paths.diagnostics.mkdir(parents=True, exist_ok=True)
        result.edges.write_parquet(self.paths.historical_supplier_edges_path)
        result.edge_report.write_csv(self.paths.supplier_edge_report_path)
        result.schema_report.write_csv(self.paths.supplier_edge_schema_report_path)
        return SupplierEdgeBuildResult(
            edges=result.edges,
            selected_source=result.selected_source,
            schema_report=result.schema_report,
            edge_report=result.edge_report,
            output_path=self.paths.historical_supplier_edges_path,
        )

    def write_raw_supplier_edges(self, raw_edges: pl.DataFrame) -> Path:
        """Write raw Eora T supplier edges and their dedicated diagnostic report."""
        self.paths.interim.mkdir(parents=True, exist_ok=True)
        self.paths.diagnostics.mkdir(parents=True, exist_ok=True)
        state_panel = self.load_state_panel()
        raw_edges.write_parquet(self.paths.raw_t_supplier_edges_path)
        self.build_raw_t_edge_report(raw_edges, state_panel).write_csv(
            self.paths.raw_t_supplier_edge_report_path
        )
        return self.paths.raw_t_supplier_edges_path

    def write_edge_source_comparison(self, comparison: pl.DataFrame) -> Path:
        """Write raw T versus legacy embodied-emissions edge diagnostics."""
        self.paths.diagnostics.mkdir(parents=True, exist_ok=True)
        comparison.write_csv(self.paths.supplier_edge_source_comparison_path)
        return self.paths.supplier_edge_source_comparison_path

    def build_and_write_raw_t_supplier_edges(
        self,
        years: Iterable[int],
        legacy_edges: pl.DataFrame,
        row_chunk_size: int = 100,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Stream raw Eora T supplier edges to parquet and write comparison diagnostics."""
        self.paths.interim.mkdir(parents=True, exist_ok=True)
        self.paths.diagnostics.mkdir(parents=True, exist_ok=True)
        state_panel = self.load_state_panel()
        stats = self._collect_raw_t_streaming_stats(years, state_panel)
        self._write_raw_t_edges_streaming(stats, row_chunk_size=row_chunk_size)
        report = self._build_raw_t_report_from_stats(stats, state_panel)
        comparison = self._compare_edge_sources_from_raw_stats(stats, legacy_edges)
        report.write_csv(self.paths.raw_t_supplier_edge_report_path)
        comparison.write_csv(self.paths.supplier_edge_source_comparison_path)
        return report, comparison

    def build_historical_top_supplier_candidates(
        self,
        max_historical_suppliers_per_buyer: int = 25,
        debug_buyers: int | None = None,
        debug_years: tuple[int, int] | None = None,
    ) -> pl.DataFrame:
        """Build compact top historical raw-T suppliers for each buyer."""
        if not self.paths.raw_t_supplier_edges_path.exists():
            raise FileNotFoundError(
                f"Raw T supplier edge parquet not found: {self.paths.raw_t_supplier_edges_path}"
            )
        return self._build_historical_top_supplier_candidates_bounded(
            max_historical_suppliers_per_buyer=max_historical_suppliers_per_buyer,
            debug_buyers=debug_buyers,
            debug_years=debug_years,
        )

    def _build_historical_top_supplier_candidates_bounded(
        self,
        max_historical_suppliers_per_buyer: int,
        debug_buyers: int | None,
        debug_years: tuple[int, int] | None,
    ) -> pl.DataFrame:
        """Select top historical suppliers with bounded per-buyer heaps."""
        anchor_year = int(debug_years[0]) if debug_years is not None else int(self.start_year)
        selected_buyers: set[str] | None = None
        buyer_order: list[str] = []
        heaps: dict[str, list[tuple[float, float, str]]] = {}
        metadata_by_pair: dict[tuple[str, str], dict[str, object]] = {}
        parquet_file = pq.ParquetFile(self.paths.raw_t_supplier_edges_path)
        columns = [
            "year",
            "buyer_country_sector",
            "supplier_country_sector",
            "transaction_value",
            "historical_tie_strength",
            "supplier_country",
            "buyer_country",
            "supplier_sector",
            "buyer_sector",
            "supplier_ecosystem_id",
            "buyer_ecosystem_id",
        ]

        for batch in parquet_file.iter_batches(columns=columns, batch_size=250_000):
            frame = pl.from_arrow(batch).filter(pl.col("year") == anchor_year)
            if frame.is_empty():
                continue
            for row in frame.iter_rows(named=True):
                buyer = row["buyer_country_sector"]
                if selected_buyers is not None and buyer not in selected_buyers:
                    continue
                if debug_buyers is not None and selected_buyers is None:
                    if buyer not in heaps and len(buyer_order) >= debug_buyers:
                        continue
                    if buyer not in heaps:
                        buyer_order.append(buyer)
                        if len(buyer_order) == debug_buyers:
                            selected_buyers = set(buyer_order)
                supplier = row["supplier_country_sector"]
                score = (
                    float(row["historical_tie_strength"] or 0.0),
                    float(row["transaction_value"] or 0.0),
                    str(supplier),
                )
                heap = heaps.setdefault(buyer, [])
                existing_suppliers = {entry[2] for entry in heap}
                if supplier in existing_suppliers:
                    continue
                if len(heap) < max_historical_suppliers_per_buyer:
                    heapq.heappush(heap, score)
                    metadata_by_pair[(buyer, supplier)] = row
                elif score[:2] > heap[0][:2]:
                    removed = heapq.heapreplace(heap, score)
                    metadata_by_pair.pop((buyer, removed[2]), None)
                    metadata_by_pair[(buyer, supplier)] = row

        selected_rows: list[dict[str, object]] = []
        for buyer, heap in heaps.items():
            for tie_strength, transaction_value, supplier in sorted(
                heap,
                key=lambda item: (-item[0], -item[1], item[2]),
            ):
                metadata = metadata_by_pair[(buyer, supplier)]
                selected_rows.append(
                    {
                        "buyer_country_sector": buyer,
                        "supplier_country_sector": supplier,
                        "historical_tie_strength": tie_strength,
                        "anchor_transaction_value": transaction_value,
                        "supplier_country": metadata["supplier_country"],
                        "buyer_country": metadata["buyer_country"],
                        "supplier_sector": metadata["supplier_sector"],
                        "buyer_sector": metadata["buyer_sector"],
                        "supplier_ecosystem_id": metadata["supplier_ecosystem_id"],
                        "buyer_ecosystem_id": metadata["buyer_ecosystem_id"],
                    }
                )

        selected = pl.DataFrame(selected_rows)
        if selected.is_empty():
            return pl.DataFrame(
                schema={
                    "buyer_country_sector": pl.Utf8,
                    "supplier_country_sector": pl.Utf8,
                    "supplier_type": pl.Utf8,
                    "historical_tie_strength": pl.Float64,
                    "mean_historical_share": pl.Float64,
                    "total_transaction_value": pl.Float64,
                    "supplier_country": pl.Utf8,
                    "buyer_country": pl.Utf8,
                    "supplier_sector": pl.Utf8,
                    "buyer_sector": pl.Utf8,
                    "supplier_ecosystem_id": pl.Utf8,
                    "buyer_ecosystem_id": pl.Utf8,
                    "source_type": pl.Utf8,
                }
            )
        return self._aggregate_selected_historical_candidates_duckdb(selected, debug_years)

    def _aggregate_selected_historical_candidates_duckdb(
        self,
        selected: pl.DataFrame,
        debug_years: tuple[int, int] | None,
    ) -> pl.DataFrame:
        """Aggregate full-history totals for a compact selected-pair table."""
        import duckdb

        parquet_path = str(self.paths.raw_t_supplier_edges_path).replace("\\", "/").replace("'", "''")
        aggregate_where_sql = ""
        if debug_years is not None:
            aggregate_where_sql = (
                f"WHERE r.year BETWEEN {int(debug_years[0])} AND {int(debug_years[1])}"
            )
        query = f"""
            SELECT
                s.buyer_country_sector,
                s.supplier_country_sector,
                'historical' AS supplier_type,
                max(r.historical_tie_strength) AS historical_tie_strength,
                avg(r.historical_share) AS mean_historical_share,
                sum(r.transaction_value) AS total_transaction_value,
                any_value(s.supplier_country) AS supplier_country,
                any_value(s.buyer_country) AS buyer_country,
                any_value(s.supplier_sector) AS supplier_sector,
                any_value(s.buyer_sector) AS buyer_sector,
                any_value(s.supplier_ecosystem_id) AS supplier_ecosystem_id,
                any_value(s.buyer_ecosystem_id) AS buyer_ecosystem_id,
                'raw_eora_T_top_historical' AS source_type
            FROM read_parquet('{parquet_path}') AS r
            INNER JOIN selected_historical_candidates AS s
                ON r.buyer_country_sector = s.buyer_country_sector
                AND r.supplier_country_sector = s.supplier_country_sector
            {aggregate_where_sql}
            GROUP BY s.buyer_country_sector, s.supplier_country_sector
            ORDER BY s.buyer_country_sector, historical_tie_strength DESC, total_transaction_value DESC
        """
        with duckdb.connect(database=":memory:") as connection:
            duckdb_tmp = str((self.paths.project_root / "tmp" / "duckdb").resolve()).replace("\\", "/").replace("'", "''")
            Path(duckdb_tmp).mkdir(parents=True, exist_ok=True)
            connection.execute("SET preserve_insertion_order = false")
            connection.execute("SET memory_limit = '4GB'")
            connection.execute(f"SET temp_directory = '{duckdb_tmp}'")
            connection.register("selected_historical_candidates", selected)
            return pl.from_arrow(connection.execute(query).arrow())

    def _build_historical_top_supplier_candidates_duckdb(
        self,
        max_historical_suppliers_per_buyer: int,
        debug_buyers: int | None,
        debug_years: tuple[int, int] | None,
    ) -> pl.DataFrame:
        """Use DuckDB to reduce the large raw-T parquet without loading it into memory."""
        import duckdb

        parquet_path = str(self.paths.raw_t_supplier_edges_path).replace("\\", "/").replace("'", "''")
        where_clauses: list[str] = []
        anchor_year = int(debug_years[0]) if debug_years is not None else int(self.start_year)
        where_clauses.append(f"year = {anchor_year}")
        aggregate_where_clauses: list[str] = []
        if debug_years is not None:
            aggregate_where_clauses.append(
                f"r.year BETWEEN {int(debug_years[0])} AND {int(debug_years[1])}"
            )
        if debug_buyers is not None:
            where_clauses.append(
                "buyer_country_sector IN ("
                f"SELECT buyer_country_sector FROM read_parquet('{parquet_path}') "
                "GROUP BY buyer_country_sector "
                f"LIMIT {int(debug_buyers)}"
                ")"
            )
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        aggregate_where_sql = (
            f"WHERE {' AND '.join(aggregate_where_clauses)}"
            if aggregate_where_clauses
            else ""
        )
        selected_query = f"""
            CREATE TEMP TABLE selected_historical_candidates AS
            WITH ranked AS (
                SELECT
                    buyer_country_sector,
                    supplier_country_sector,
                    historical_tie_strength,
                    supplier_country,
                    buyer_country,
                    supplier_sector,
                    buyer_sector,
                    supplier_ecosystem_id,
                    buyer_ecosystem_id,
                    row_number() OVER (
                        PARTITION BY buyer_country_sector
                        ORDER BY historical_tie_strength DESC,
                                 transaction_value DESC,
                                 supplier_country_sector ASC
                    ) AS supplier_rank
                FROM read_parquet('{parquet_path}')
                {where_sql}
            )
            SELECT *
            FROM ranked
            WHERE supplier_rank <= {int(max_historical_suppliers_per_buyer)}
        """
        aggregate_query = f"""
            SELECT
                s.buyer_country_sector,
                s.supplier_country_sector,
                'historical' AS supplier_type,
                max(r.historical_tie_strength) AS historical_tie_strength,
                avg(r.historical_share) AS mean_historical_share,
                sum(r.transaction_value) AS total_transaction_value,
                any_value(s.supplier_country) AS supplier_country,
                any_value(s.buyer_country) AS buyer_country,
                any_value(s.supplier_sector) AS supplier_sector,
                any_value(s.buyer_sector) AS buyer_sector,
                any_value(s.supplier_ecosystem_id) AS supplier_ecosystem_id,
                any_value(s.buyer_ecosystem_id) AS buyer_ecosystem_id,
                'raw_eora_T_top_historical' AS source_type
            FROM read_parquet('{parquet_path}') AS r
            INNER JOIN selected_historical_candidates AS s
                ON r.buyer_country_sector = s.buyer_country_sector
                AND r.supplier_country_sector = s.supplier_country_sector
            {aggregate_where_sql}
            GROUP BY s.buyer_country_sector, s.supplier_country_sector, s.supplier_rank
            ORDER BY s.buyer_country_sector, s.supplier_rank
        """
        with duckdb.connect(database=":memory:") as connection:
            duckdb_tmp = str((self.paths.project_root / "tmp" / "duckdb").resolve()).replace("\\", "/").replace("'", "''")
            Path(duckdb_tmp).mkdir(parents=True, exist_ok=True)
            connection.execute("SET preserve_insertion_order = false")
            connection.execute("SET memory_limit = '4GB'")
            connection.execute(f"SET temp_directory = '{duckdb_tmp}'")
            connection.execute(selected_query)
            return pl.from_arrow(connection.execute(aggregate_query).arrow())

    def build_same_sector_supplier_pool(
        self,
        max_same_sector_candidates_per_buyer: int = 25,
        debug_buyers: int | None = None,
    ) -> pl.DataFrame:
        """Build compact same-sector supplier candidates, preferring foreign nodes."""
        nodes = self._node_candidate_profiles()
        if debug_buyers is not None:
            buyers = nodes.head(debug_buyers)
        else:
            buyers = nodes
        rows: list[dict[str, object]] = []
        suppliers_by_sector = {
            self._partition_key_value(sector): frame.sort(
                [
                    "total_supplier_output_or_transaction_proxy",
                    "green_rank_value",
                    "reliability_rank_value",
                    "country_sector",
                ],
                descending=[True, True, True, False],
            )
            for sector, frame in nodes.partition_by("Sector", as_dict=True).items()
        }

        for buyer in buyers.iter_rows(named=True):
            sector_suppliers = suppliers_by_sector.get(buyer["Sector"])
            if sector_suppliers is None:
                continue
            available = sector_suppliers.filter(
                pl.col("country_sector") != buyer["country_sector"]
            )
            foreign = available.filter(pl.col("Country") != buyer["Country"])
            domestic_fallback_used = foreign.is_empty()
            candidates = available if domestic_fallback_used else foreign
            for rank, supplier in enumerate(
                candidates.head(max_same_sector_candidates_per_buyer).iter_rows(named=True),
                start=1,
            ):
                rows.append(
                    {
                        "buyer_country_sector": buyer["country_sector"],
                        "supplier_country_sector": supplier["country_sector"],
                        "supplier_type": "same_sector_foreign",
                        "supplier_rank": rank,
                        "supplier_sector": supplier["Sector"],
                        "buyer_sector": buyer["Sector"],
                        "supplier_country": supplier["Country"],
                        "buyer_country": buyer["Country"],
                        "supplier_ecosystem_id": supplier["ecosystem_id"],
                        "buyer_ecosystem_id": buyer["ecosystem_id"],
                        "total_supplier_output_or_transaction_proxy": supplier[
                            "total_supplier_output_or_transaction_proxy"
                        ],
                        "domestic_fallback_used": domestic_fallback_used,
                        "source_type": "same_sector_pool",
                    }
                )
        return pl.DataFrame(rows, schema=self._same_sector_pool_schema())

    def build_ecosystem_supplier_pool(
        self,
        max_ecosystem_candidates_per_buyer: int = 25,
        historical_candidates: pl.DataFrame | None = None,
        same_sector_candidates: pl.DataFrame | None = None,
        debug_buyers: int | None = None,
    ) -> pl.DataFrame:
        """Build compact ecosystem-feasible supplier candidates from same/adjacent ecosystems."""
        if not self.paths.ecosystem_adjacency_path.exists():
            raise FileNotFoundError(
                f"Ecosystem adjacency file not found: {self.paths.ecosystem_adjacency_path}"
            )
        nodes = self._node_candidate_profiles()
        buyers = nodes.head(debug_buyers) if debug_buyers is not None else nodes
        adjacency = (
            pl.read_csv(self.paths.ecosystem_adjacency_path)
            .filter(pl.col("relation_type").is_in(["same", "adjacent"]))
            .select("ecosystem_id_from", "ecosystem_id_to", "proximity", "relation_type")
        )
        duplicate_pairs = self._candidate_pair_source_flags(
            historical_candidates=historical_candidates,
            same_sector_candidates=same_sector_candidates,
        )
        supplier_frames = {
            self._partition_key_value(ecosystem_id): frame.sort(
                [
                    "total_supplier_output_or_transaction_proxy",
                    "green_rank_value",
                    "country_sector",
                ],
                descending=[True, True, False],
            )
            for ecosystem_id, frame in nodes.partition_by("ecosystem_id", as_dict=True).items()
        }
        rows: list[dict[str, object]] = []
        for buyer in buyers.iter_rows(named=True):
            allowed_relations = adjacency.filter(
                pl.col("ecosystem_id_from") == buyer["ecosystem_id"]
            ).sort(["proximity", "ecosystem_id_to"], descending=[True, False])
            candidates_for_buyer: list[dict[str, object]] = []
            for relation in allowed_relations.iter_rows(named=True):
                supplier_frame = supplier_frames.get(relation["ecosystem_id_to"])
                if supplier_frame is None:
                    continue
                for supplier in supplier_frame.iter_rows(named=True):
                    if supplier["country_sector"] == buyer["country_sector"]:
                        continue
                    duplicate_flags = duplicate_pairs.get(
                        (buyer["country_sector"], supplier["country_sector"]),
                        [],
                    )
                    candidates_for_buyer.append(
                        {
                            "buyer_country_sector": buyer["country_sector"],
                            "supplier_country_sector": supplier["country_sector"],
                            "supplier_type": "ecosystem_feasible",
                            "ecosystem_proximity": relation["proximity"],
                            "supplier_ecosystem_id": supplier["ecosystem_id"],
                            "buyer_ecosystem_id": buyer["ecosystem_id"],
                            "supplier_ecosystem_label": supplier["ecosystem_label"],
                            "buyer_ecosystem_label": buyer["ecosystem_label"],
                            "total_supplier_output_or_transaction_proxy": supplier[
                                "total_supplier_output_or_transaction_proxy"
                            ],
                            "green_rank_value": supplier["green_rank_value"],
                            "candidate_source_flags": "|".join(duplicate_flags)
                            if duplicate_flags
                            else "",
                            "source_type": "ecosystem_pool",
                        }
                    )
            ranked = sorted(
                candidates_for_buyer,
                key=lambda row: (
                    -float(row["ecosystem_proximity"]),
                    -float(row["total_supplier_output_or_transaction_proxy"] or 0.0),
                    -float(row["green_rank_value"] or 0.0),
                    str(row["supplier_country_sector"]),
                ),
            )[:max_ecosystem_candidates_per_buyer]
            for rank, row in enumerate(ranked, start=1):
                row["supplier_rank"] = rank
                row.pop("green_rank_value", None)
                rows.append(row)

        return pl.DataFrame(rows, schema=self._ecosystem_pool_schema())

    def build_supplier_candidate_base_report(
        self,
        historical_candidates: pl.DataFrame,
        same_sector_candidates: pl.DataFrame,
        ecosystem_candidates: pl.DataFrame,
    ) -> pl.DataFrame:
        """Build compact candidate-base diagnostics."""
        state_buyers = self.load_state_panel().select("country_sector").unique()
        number_of_buyers = state_buyers.height
        counts = {
            "historical": self._candidate_count_summary(historical_candidates),
            "same_sector": self._candidate_count_summary(same_sector_candidates),
            "ecosystem": self._candidate_count_summary(ecosystem_candidates),
        }
        return pl.DataFrame(
            {
                "number_of_buyers": [number_of_buyers],
                "historical_candidate_rows": [historical_candidates.height],
                "same_sector_candidate_rows": [same_sector_candidates.height],
                "ecosystem_candidate_rows": [ecosystem_candidates.height],
                "median_historical_candidates_per_buyer": [counts["historical"]["median"]],
                "median_same_sector_candidates_per_buyer": [counts["same_sector"]["median"]],
                "median_ecosystem_candidates_per_buyer": [counts["ecosystem"]["median"]],
                "max_candidates_per_buyer_by_type": [
                    (
                        f"historical={counts['historical']['max']}; "
                        f"same_sector={counts['same_sector']['max']}; "
                        f"ecosystem={counts['ecosystem']['max']}"
                    )
                ],
                "share_buyers_with_no_historical_candidates": [
                    self._share_buyers_without_candidates(state_buyers, historical_candidates)
                ],
                "share_buyers_with_no_same_sector_candidates": [
                    self._share_buyers_without_candidates(state_buyers, same_sector_candidates)
                ],
                "share_buyers_with_no_ecosystem_candidates": [
                    self._share_buyers_without_candidates(state_buyers, ecosystem_candidates)
                ],
                "notes": [
                    (
                        "Historical candidates are reduced from raw Eora T via bounded parquet "
                        "batch scans plus selected-pair aggregation. Same-sector and ecosystem "
                        "pools use the compact ABM v4 state panel and do not create all-to-all "
                        "matrices."
                    )
                ],
            }
        )

    def write_supplier_candidate_base(
        self,
        historical_candidates: pl.DataFrame,
        same_sector_candidates: pl.DataFrame,
        ecosystem_candidates: pl.DataFrame,
        report: pl.DataFrame,
    ) -> None:
        """Write compact supplier candidate base outputs."""
        self.paths.interim.mkdir(parents=True, exist_ok=True)
        self.paths.diagnostics.mkdir(parents=True, exist_ok=True)
        historical_candidates.write_parquet(self.paths.supplier_candidates_historical_top_path)
        same_sector_candidates.write_parquet(self.paths.supplier_pool_same_sector_path)
        ecosystem_candidates.write_parquet(self.paths.supplier_pool_ecosystem_path)
        report.write_csv(self.paths.supplier_candidate_base_report_path)

    def build_supplier_opportunity_sets(
        self,
        supplier_friction: object | None = None,
        supplier_choice: object | None = None,
        opportunity_year: int | None = None,
        beta_supplier_choice: float = 1.0,
        epsilon: float = 1e-9,
    ) -> pl.DataFrame:
        """Build bounded supplier opportunity sets from compact candidate pools."""
        historical_candidates = pl.read_parquet(self.paths.supplier_candidates_historical_top_path)
        same_sector_candidates = pl.read_parquet(self.paths.supplier_pool_same_sector_path)
        ecosystem_candidates = pl.read_parquet(self.paths.supplier_pool_ecosystem_path)
        merged = self.merge_candidate_pools(
            historical_candidates,
            same_sector_candidates,
            ecosystem_candidates,
        )
        deduplicated = self.deduplicate_candidates(merged)
        profiles = self._state_profiles_for_opportunities(
            opportunity_year=opportunity_year,
            epsilon=epsilon,
        )
        enriched = self._attach_opportunity_state_profiles(deduplicated, profiles)
        enriched = self.compute_supplier_friction(enriched, supplier_friction=supplier_friction)
        enriched = self.compute_supplier_green_advantage(enriched)
        enriched = self.compute_supplier_reliability(enriched)
        enriched = self.compute_supplier_attractiveness(
            enriched,
            supplier_choice=supplier_choice,
        )
        enriched = self.compute_choice_probabilities(
            enriched,
            beta_supplier_choice=beta_supplier_choice,
        )
        return enriched.select(
            "buyer_country_sector",
            "supplier_country_sector",
            "supplier_type",
            "candidate_sources",
            "is_historical_candidate",
            "is_same_sector_candidate",
            "is_ecosystem_candidate",
            "buyer_country",
            "buyer_sector",
            "buyer_ecosystem_id",
            "buyer_ecosystem_label",
            "supplier_country",
            "supplier_sector",
            "supplier_ecosystem_id",
            "supplier_ecosystem_label",
            "historical_tie_strength",
            "mean_historical_share",
            "total_transaction_value",
            "ecosystem_proximity",
            "buyer_g_local_v4",
            "supplier_g_local_v4",
            "green_advantage",
            "supplier_green_capability",
            "supplier_general_capability",
            "supplier_reliability",
            "friction",
            "supplier_attractiveness",
            "choice_probability",
            "duplicated_candidate_before_resolution",
            "selected_priority_rule",
            "source_type",
        )

    def merge_candidate_pools(
        self,
        historical_candidates: pl.DataFrame,
        same_sector_candidates: pl.DataFrame,
        ecosystem_candidates: pl.DataFrame,
    ) -> pl.DataFrame:
        """Merge compact candidate pools into one inspectable long table."""
        schema_columns = {
            "buyer_country_sector": pl.Utf8,
            "supplier_country_sector": pl.Utf8,
            "supplier_type": pl.Utf8,
            "candidate_source": pl.Utf8,
            "is_historical_candidate": pl.Boolean,
            "is_same_sector_candidate": pl.Boolean,
            "is_ecosystem_candidate": pl.Boolean,
            "buyer_country": pl.Utf8,
            "buyer_sector": pl.Utf8,
            "buyer_ecosystem_id": pl.Utf8,
            "buyer_ecosystem_label": pl.Utf8,
            "supplier_country": pl.Utf8,
            "supplier_sector": pl.Utf8,
            "supplier_ecosystem_id": pl.Utf8,
            "supplier_ecosystem_label": pl.Utf8,
            "historical_tie_strength": pl.Float64,
            "mean_historical_share": pl.Float64,
            "total_transaction_value": pl.Float64,
            "ecosystem_proximity": pl.Float64,
            "source_type": pl.Utf8,
        }
        frames = [
            self._normalize_candidate_pool(
                historical_candidates,
                candidate_source="historical",
                is_historical=True,
                is_same_sector=False,
                is_ecosystem=False,
                schema_columns=schema_columns,
            ),
            self._normalize_candidate_pool(
                same_sector_candidates,
                candidate_source="same_sector",
                is_historical=False,
                is_same_sector=True,
                is_ecosystem=False,
                schema_columns=schema_columns,
            ),
            self._normalize_candidate_pool(
                ecosystem_candidates,
                candidate_source="ecosystem",
                is_historical=False,
                is_same_sector=False,
                is_ecosystem=True,
                schema_columns=schema_columns,
            ),
        ]
        return pl.concat(frames, how="vertical")

    def deduplicate_candidates(self, candidates: pl.DataFrame) -> pl.DataFrame:
        """Resolve duplicate buyer-supplier candidates while preserving source flags."""
        grouped = candidates.group_by(
            "buyer_country_sector",
            "supplier_country_sector",
        ).agg(
            pl.max("is_historical_candidate").alias("is_historical_candidate"),
            pl.max("is_same_sector_candidate").alias("is_same_sector_candidate"),
            pl.max("is_ecosystem_candidate").alias("is_ecosystem_candidate"),
            pl.len().alias("_candidate_pool_rows"),
            pl.col("buyer_country").drop_nulls().first().alias("buyer_country"),
            pl.col("buyer_sector").drop_nulls().first().alias("buyer_sector"),
            pl.col("buyer_ecosystem_id").drop_nulls().first().alias("buyer_ecosystem_id"),
            pl.col("buyer_ecosystem_label").drop_nulls().first().alias("buyer_ecosystem_label"),
            pl.col("supplier_country").drop_nulls().first().alias("supplier_country"),
            pl.col("supplier_sector").drop_nulls().first().alias("supplier_sector"),
            pl.col("supplier_ecosystem_id").drop_nulls().first().alias("supplier_ecosystem_id"),
            pl.col("supplier_ecosystem_label").drop_nulls().first().alias("supplier_ecosystem_label"),
            pl.max("historical_tie_strength").alias("historical_tie_strength"),
            pl.mean("mean_historical_share").alias("mean_historical_share"),
            pl.sum("total_transaction_value").alias("total_transaction_value"),
            pl.max("ecosystem_proximity").alias("ecosystem_proximity"),
        )
        rows: list[dict[str, object]] = []
        for row in grouped.iter_rows(named=True):
            sources: list[str] = []
            if row["is_historical_candidate"]:
                sources.append("historical")
            if row["is_same_sector_candidate"]:
                sources.append("same_sector")
            if row["is_ecosystem_candidate"]:
                sources.append("ecosystem")
            if row["is_historical_candidate"]:
                supplier_type = "historical"
                priority_rule = "historical_priority"
            elif row["is_same_sector_candidate"]:
                supplier_type = "same_sector_foreign"
                priority_rule = "same_sector_priority"
            else:
                supplier_type = "ecosystem_feasible"
                priority_rule = "ecosystem_priority"
            row["candidate_sources"] = ";".join(sources)
            row["supplier_type"] = supplier_type
            row["duplicated_candidate_before_resolution"] = row["_candidate_pool_rows"] > 1
            row["selected_priority_rule"] = priority_rule
            row["source_type"] = "supplier_opportunity_set"
            row.pop("_candidate_pool_rows")
            rows.append(row)
        return pl.DataFrame(rows)

    def compute_supplier_friction(
        self,
        opportunities: pl.DataFrame,
        supplier_friction: object | None = None,
    ) -> pl.DataFrame:
        """Attach friction according to the supplier-type hierarchy."""
        phi_historical = getattr(supplier_friction, "phi_historical", 0.10)
        phi_same_sector = getattr(supplier_friction, "phi_same_sector", 0.50)
        phi_ecosystem = getattr(supplier_friction, "phi_ecosystem", 1.00)
        return opportunities.with_columns(
            pl.when(pl.col("supplier_type") == "historical")
            .then(pl.lit(phi_historical))
            .when(pl.col("supplier_type") == "same_sector_foreign")
            .then(pl.lit(phi_same_sector))
            .otherwise(pl.lit(phi_ecosystem))
            .alias("friction")
        )

    def compute_supplier_green_advantage(self, opportunities: pl.DataFrame) -> pl.DataFrame:
        """Compute supplier green advantage relative to buyer historical baseline."""
        historical_baseline = (
            opportunities.filter(
                pl.col("is_historical_candidate")
                & pl.col("supplier_g_local_v4").is_not_null()
                & pl.col("historical_tie_strength").is_not_null()
            )
            .with_columns(
                (
                    pl.col("supplier_g_local_v4") * pl.col("historical_tie_strength")
                ).alias("_weighted_supplier_green")
            )
            .group_by("buyer_country_sector")
            .agg(
                pl.sum("_weighted_supplier_green").alias("_weighted_green_sum"),
                pl.sum("historical_tie_strength").alias("_tie_sum"),
            )
            .with_columns(
                pl.when(pl.col("_tie_sum") > 0)
                .then(pl.col("_weighted_green_sum") / pl.col("_tie_sum"))
                .otherwise(None)
                .alias("buyer_current_supplier_green_baseline")
            )
            .select("buyer_country_sector", "buyer_current_supplier_green_baseline")
        )
        return (
            opportunities.join(historical_baseline, on="buyer_country_sector", how="left")
            .with_columns(
                pl.coalesce(
                    ["buyer_current_supplier_green_baseline", "buyer_g_local_v4"]
                ).alias("buyer_current_supplier_green_baseline")
            )
            .with_columns(
                (
                    pl.col("supplier_g_local_v4")
                    - pl.col("buyer_current_supplier_green_baseline")
                ).alias("green_advantage")
            )
            .drop("buyer_current_supplier_green_baseline")
        )

    def compute_supplier_reliability(self, opportunities: pl.DataFrame) -> pl.DataFrame:
        """Return opportunities with an explicit supplier reliability column."""
        return opportunities.with_columns(
            pl.col("supplier_reliability").cast(pl.Float64, strict=False)
        )

    def compute_supplier_attractiveness(
        self,
        opportunities: pl.DataFrame,
        supplier_choice: object | None = None,
    ) -> pl.DataFrame:
        """Compute supplier attractiveness from transition and historical variables."""
        alpha_green_advantage = getattr(supplier_choice, "alpha_green_advantage", 1.0)
        alpha_reliability = getattr(supplier_choice, "alpha_reliability", 0.5)
        alpha_green_capability = getattr(supplier_choice, "alpha_green_capability", 0.5)
        alpha_general_capability = getattr(supplier_choice, "alpha_general_capability", 0.25)
        alpha_historical_tie = getattr(
            supplier_choice,
            "alpha_historical_tie",
            getattr(supplier_choice, "alpha_historical_tie_strength", 1.0),
        )
        alpha_ecosystem_proximity = getattr(supplier_choice, "alpha_ecosystem_proximity", 0.5)
        alpha_friction = getattr(supplier_choice, "alpha_friction", 1.0)
        return opportunities.with_columns(
            (
                alpha_green_advantage * pl.col("green_advantage").fill_null(0.0)
                + alpha_reliability * pl.col("supplier_reliability").fill_null(0.0)
                + alpha_green_capability * pl.col("supplier_green_capability").fill_null(0.0)
                + alpha_general_capability * pl.col("supplier_general_capability").fill_null(0.0)
                + alpha_historical_tie * pl.col("historical_tie_strength").fill_null(0.0)
                + alpha_ecosystem_proximity * pl.col("ecosystem_proximity").fill_null(0.0)
                - alpha_friction * pl.col("friction").fill_null(0.0)
            ).alias("supplier_attractiveness")
        )

    def compute_choice_probabilities(
        self,
        opportunities: pl.DataFrame,
        beta_supplier_choice: float = 1.0,
    ) -> pl.DataFrame:
        """Compute numerically stable softmax probabilities within each buyer."""
        return (
            opportunities.with_columns(
                (pl.lit(beta_supplier_choice) * pl.col("supplier_attractiveness")).alias("_choice_x")
            )
            .with_columns(
                pl.max("_choice_x").over("buyer_country_sector").alias("_choice_x_max")
            )
            .with_columns(
                (pl.col("_choice_x") - pl.col("_choice_x_max")).exp().alias("_choice_exp")
            )
            .with_columns(
                pl.sum("_choice_exp").over("buyer_country_sector").alias("_choice_exp_sum")
            )
            .with_columns(
                pl.when(pl.col("_choice_exp_sum") > 0)
                .then(pl.col("_choice_exp") / pl.col("_choice_exp_sum"))
                .otherwise(None)
                .alias("choice_probability")
            )
            .drop(["_choice_x", "_choice_x_max", "_choice_exp", "_choice_exp_sum"])
        )

    def build_opportunity_set_report(self, opportunities: pl.DataFrame) -> pl.DataFrame:
        """Build diagnostics for supplier opportunity sets."""
        state_buyers = self.load_state_panel().select("country_sector").unique()
        counts = opportunities.group_by("buyer_country_sector").len(name="candidate_count")
        probability_sums = opportunities.group_by("buyer_country_sector").agg(
            pl.sum("choice_probability").alias("probability_sum")
        ).with_columns((pl.col("probability_sum") - 1.0).abs().alias("probability_sum_error"))
        buyers_with_probability_sum_error = probability_sums.filter(
            pl.col("probability_sum_error") > 1e-8
        ).height
        row_count = opportunities.height
        return pl.DataFrame(
            {
                "number_of_buyers": [state_buyers.height],
                "opportunity_rows": [row_count],
                "median_candidates_per_buyer": [counts["candidate_count"].median()],
                "p95_candidates_per_buyer": [counts["candidate_count"].quantile(0.95)],
                "max_candidates_per_buyer": [counts["candidate_count"].max()],
                "share_historical_candidates": [
                    opportunities["is_historical_candidate"].sum() / row_count if row_count else 0.0
                ],
                "share_same_sector_candidates": [
                    opportunities["is_same_sector_candidate"].sum() / row_count if row_count else 0.0
                ],
                "share_ecosystem_candidates": [
                    opportunities["is_ecosystem_candidate"].sum() / row_count if row_count else 0.0
                ],
                "share_multi_source_candidates": [
                    opportunities["duplicated_candidate_before_resolution"].sum() / row_count
                    if row_count
                    else 0.0
                ],
                "mean_friction": [opportunities["friction"].mean()],
                "mean_green_advantage": [opportunities["green_advantage"].mean()],
                "mean_supplier_attractiveness": [
                    opportunities["supplier_attractiveness"].mean()
                ],
                "buyers_with_probability_sum_error": [buyers_with_probability_sum_error],
                "max_probability_sum_error": [
                    probability_sums["probability_sum_error"].max()
                    if probability_sums.height
                    else 0.0
                ],
                "notes": [
                    (
                        "Opportunity sets combine compact historical, same-sector, and ecosystem "
                        "candidate pools. They define feasible alternatives only, not realized "
                        "rewiring or updated supplier weights."
                    )
                ],
            }
        )

    def write_supplier_opportunity_sets(
        self,
        opportunities: pl.DataFrame,
        report: pl.DataFrame,
    ) -> None:
        """Write supplier opportunity sets and diagnostics."""
        self.paths.interim.mkdir(parents=True, exist_ok=True)
        self.paths.diagnostics.mkdir(parents=True, exist_ok=True)
        opportunities.write_parquet(self.paths.supplier_opportunity_sets_path)
        report.write_csv(self.paths.supplier_opportunity_set_report_path)

    def _collect_raw_t_streaming_stats(
        self,
        years: Iterable[int],
        state_panel: pl.DataFrame,
    ) -> dict[str, object]:
        year_list = list(years)
        node_metadata = self._node_metadata_by_country_sector(state_panel)
        supplier_labels: np.ndarray | None = None
        buyer_labels: np.ndarray | None = None
        pair_period_totals: np.ndarray | None = None
        pair_present_any: np.ndarray | None = None
        buyer_period_totals: np.ndarray | None = None
        buyer_edge_counts: np.ndarray | None = None
        supplier_present_any: np.ndarray | None = None
        buyer_present_any: np.ndarray | None = None
        buyer_year_totals: dict[int, np.ndarray] = {}
        existing_years: list[int] = []
        row_count = 0
        total_transaction_value = 0.0
        supplier_metadata_rows = 0
        buyer_metadata_rows = 0
        ecosystem_metadata_rows = 0

        for year in year_list:
            t_path = self.paths.data_root / "parquet" / str(year) / "T.parquet"
            if not t_path.exists():
                continue
            matrix, current_supplier_labels, current_buyer_labels = self._read_t_matrix_arrays(t_path)
            if supplier_labels is None:
                supplier_labels = current_supplier_labels
                buyer_labels = current_buyer_labels
                pair_period_totals = np.zeros(matrix.shape, dtype=np.float64)
                pair_present_any = np.zeros(matrix.shape, dtype=bool)
                buyer_period_totals = np.zeros(matrix.shape[1], dtype=np.float64)
                buyer_edge_counts = np.zeros(matrix.shape[1], dtype=np.int64)
                supplier_present_any = np.zeros(matrix.shape[0], dtype=bool)
                buyer_present_any = np.zeros(matrix.shape[1], dtype=bool)
                supplier_has_metadata = np.array(
                    [label in node_metadata for label in supplier_labels],
                    dtype=bool,
                )
                buyer_has_metadata = np.array(
                    [label in node_metadata for label in buyer_labels],
                    dtype=bool,
                )
                supplier_has_ecosystem = np.array(
                    [
                        node_metadata.get(label, {}).get("ecosystem_id") is not None
                        for label in supplier_labels
                    ],
                    dtype=bool,
                )
                buyer_has_ecosystem = np.array(
                    [
                        node_metadata.get(label, {}).get("ecosystem_id") is not None
                        for label in buyer_labels
                    ],
                    dtype=bool,
                )
            elif not (
                np.array_equal(supplier_labels, current_supplier_labels)
                and np.array_equal(buyer_labels, current_buyer_labels)
            ):
                raise ValueError(f"Eora T labels changed across years at {t_path}")

            positive_mask = matrix > self.min_transaction_threshold
            positive_values = np.where(positive_mask, matrix, 0.0)
            year_row_count = int(positive_mask.sum())
            year_buyer_total = positive_values.sum(axis=0)
            buyer_year_totals[year] = year_buyer_total
            existing_years.append(year)
            row_count += year_row_count
            total_transaction_value += float(positive_values.sum())
            pair_period_totals += positive_values
            pair_present_any |= positive_mask
            buyer_period_totals += year_buyer_total
            buyer_edge_counts += positive_mask.sum(axis=0)
            supplier_present_any |= positive_mask.any(axis=1)
            buyer_present_any |= positive_mask.any(axis=0)
            supplier_metadata_rows += int((positive_mask & supplier_has_metadata[:, None]).sum())
            buyer_metadata_rows += int((positive_mask & buyer_has_metadata[None, :]).sum())
            ecosystem_metadata_rows += int(
                (
                    positive_mask
                    & supplier_has_ecosystem[:, None]
                    & buyer_has_ecosystem[None, :]
                ).sum()
            )

        if supplier_labels is None or buyer_labels is None:
            supplier_labels = np.array([], dtype=object)
            buyer_labels = np.array([], dtype=object)
            pair_period_totals = np.zeros((0, 0), dtype=np.float64)
            pair_present_any = np.zeros((0, 0), dtype=bool)
            buyer_period_totals = np.array([], dtype=np.float64)
            buyer_edge_counts = np.array([], dtype=np.int64)
            supplier_present_any = np.array([], dtype=bool)
            buyer_present_any = np.array([], dtype=bool)

        return {
            "years": existing_years,
            "supplier_labels": supplier_labels,
            "buyer_labels": buyer_labels,
            "node_metadata": node_metadata,
            "pair_period_totals": pair_period_totals,
            "pair_present_any": pair_present_any,
            "buyer_period_totals": buyer_period_totals,
            "buyer_year_totals": buyer_year_totals,
            "buyer_edge_counts": buyer_edge_counts,
            "supplier_present_any": supplier_present_any,
            "buyer_present_any": buyer_present_any,
            "row_count": row_count,
            "total_transaction_value": total_transaction_value,
            "supplier_metadata_rows": supplier_metadata_rows,
            "buyer_metadata_rows": buyer_metadata_rows,
            "ecosystem_metadata_rows": ecosystem_metadata_rows,
        }

    def _write_raw_t_edges_streaming(
        self,
        stats: dict[str, object],
        row_chunk_size: int,
    ) -> None:
        supplier_labels = stats["supplier_labels"]
        buyer_labels = stats["buyer_labels"]
        pair_period_totals = stats["pair_period_totals"]
        buyer_period_totals = stats["buyer_period_totals"]
        buyer_year_totals = stats["buyer_year_totals"]
        node_metadata = stats["node_metadata"]
        writer: pq.ParquetWriter | None = None

        for year in stats["years"]:
            t_path = self.paths.data_root / "parquet" / str(year) / "T.parquet"
            matrix, current_supplier_labels, current_buyer_labels = self._read_t_matrix_arrays(t_path)
            if not (
                np.array_equal(supplier_labels, current_supplier_labels)
                and np.array_equal(buyer_labels, current_buyer_labels)
            ):
                raise ValueError(f"Eora T labels changed across years at {t_path}")
            for row_start in range(0, matrix.shape[0], row_chunk_size):
                row_end = min(row_start + row_chunk_size, matrix.shape[0])
                chunk = matrix[row_start:row_end, :]
                positive_rows, positive_cols = np.nonzero(
                    chunk > self.min_transaction_threshold
                )
                if len(positive_rows) == 0:
                    continue
                absolute_rows = positive_rows + row_start
                values = chunk[positive_rows, positive_cols]
                year_totals = buyer_year_totals[year][positive_cols]
                period_totals = buyer_period_totals[positive_cols]
                historical_share = np.divide(
                    values,
                    year_totals,
                    out=np.full_like(values, np.nan, dtype=np.float64),
                    where=year_totals > 0,
                )
                historical_tie_strength = np.divide(
                    pair_period_totals[absolute_rows, positive_cols],
                    period_totals,
                    out=np.full_like(values, np.nan, dtype=np.float64),
                    where=period_totals > 0,
                )
                table = pa.table(
                    self._raw_t_arrow_batch(
                        year=year,
                        t_path=t_path,
                        supplier_values=supplier_labels[absolute_rows],
                        buyer_values=buyer_labels[positive_cols],
                        transaction_values=values,
                        historical_share=historical_share,
                        historical_tie_strength=historical_tie_strength,
                        node_metadata=node_metadata,
                    )
                )
                if writer is None:
                    writer = pq.ParquetWriter(self.paths.raw_t_supplier_edges_path, table.schema)
                writer.write_table(table)

        if writer is None:
            self._empty_raw_t_edges().write_parquet(self.paths.raw_t_supplier_edges_path)
        else:
            writer.close()

    def _raw_t_arrow_batch(
        self,
        year: int,
        t_path: Path,
        supplier_values: np.ndarray,
        buyer_values: np.ndarray,
        transaction_values: np.ndarray,
        historical_share: np.ndarray,
        historical_tie_strength: np.ndarray,
        node_metadata: dict[str, dict[str, object]],
    ) -> dict[str, object]:
        supplier_metadata = [node_metadata.get(str(value), {}) for value in supplier_values]
        buyer_metadata = [node_metadata.get(str(value), {}) for value in buyer_values]
        row_count = len(transaction_values)
        return {
            "year": np.full(row_count, year, dtype=np.int64),
            "supplier_country_sector": supplier_values,
            "buyer_country_sector": buyer_values,
            "transaction_value": transaction_values,
            "supplier_country": [metadata.get("Country") for metadata in supplier_metadata],
            "buyer_country": [metadata.get("Country") for metadata in buyer_metadata],
            "supplier_sector": [metadata.get("Sector") for metadata in supplier_metadata],
            "buyer_sector": [metadata.get("Sector") for metadata in buyer_metadata],
            "supplier_ecosystem_id": [
                metadata.get("ecosystem_id") for metadata in supplier_metadata
            ],
            "buyer_ecosystem_id": [
                metadata.get("ecosystem_id") for metadata in buyer_metadata
            ],
            "supplier_ecosystem_label": [
                metadata.get("ecosystem_label") for metadata in supplier_metadata
            ],
            "buyer_ecosystem_label": [
                metadata.get("ecosystem_label") for metadata in buyer_metadata
            ],
            "observed_edge": np.full(row_count, True),
            "historical_tie_strength": historical_tie_strength,
            "historical_share": historical_share,
            "source_file": np.full(row_count, str(t_path)),
            "source_type": np.full(row_count, "raw_eora_T"),
        }

    def _build_raw_t_report_from_stats(
        self,
        stats: dict[str, object],
        state_panel: pl.DataFrame,
    ) -> pl.DataFrame:
        row_count = stats["row_count"]
        years = stats["years"]
        buyer_labels = stats["buyer_labels"]
        state_buyers = state_panel.select("country_sector").unique()
        edge_buyers = pl.DataFrame(
            {
                "country_sector": buyer_labels[stats["buyer_present_any"]].tolist()
                if len(buyer_labels)
                else []
            }
        )
        buyers_without_edges = state_buyers.join(edge_buyers, on="country_sector", how="anti").height
        return pl.DataFrame(
            {
                "selected_source": ["raw_eora_T"],
                "years_covered": [f"{min(years)}-{max(years)}" if years else ""],
                "row_count": [row_count],
                "unique_suppliers": [int(stats["supplier_present_any"].sum())],
                "unique_buyers": [int(stats["buyer_present_any"].sum())],
                "unique_supplier_buyer_pairs": [int(stats["pair_present_any"].sum())],
                "total_transaction_value": [stats["total_transaction_value"]],
                "share_edges_with_supplier_metadata": [
                    stats["supplier_metadata_rows"] / row_count if row_count else 0.0
                ],
                "share_edges_with_buyer_metadata": [
                    stats["buyer_metadata_rows"] / row_count if row_count else 0.0
                ],
                "share_edges_with_ecosystem_metadata": [
                    stats["ecosystem_metadata_rows"] / row_count if row_count else 0.0
                ],
                "buyers_without_edges": [buyers_without_edges],
                "notes": [
                    (
                        "Raw Eora T matrix edges with rows as suppliers and columns as buyers. "
                        "Recommended as canonical production-sourcing weights for supplier substitution."
                    )
                ],
            }
        )

    def _compare_edge_sources_from_raw_stats(
        self,
        stats: dict[str, object],
        legacy_edges: pl.DataFrame,
    ) -> pl.DataFrame:
        raw_pair_count = int(stats["pair_present_any"].sum())
        legacy_pairs = self._pair_frame(legacy_edges)
        supplier_index = {
            str(label): index for index, label in enumerate(stats["supplier_labels"])
        }
        buyer_index = {str(label): index for index, label in enumerate(stats["buyer_labels"])}
        pair_present_any = stats["pair_present_any"]
        pair_overlap_count = 0
        for pair in legacy_pairs.iter_rows(named=True):
            supplier_position = supplier_index.get(pair["supplier_country_sector"])
            buyer_position = buyer_index.get(pair["buyer_country_sector"])
            if (
                supplier_position is not None
                and buyer_position is not None
                and pair_present_any[supplier_position, buyer_position]
            ):
                pair_overlap_count += 1
        legacy_pair_count = legacy_pairs.height
        buyer_edge_counts = stats["buyer_edge_counts"]
        raw_metadata_coverage = (
            stats["ecosystem_metadata_rows"] / stats["row_count"]
            if stats["row_count"]
            else 0.0
        )
        overlap_values = {
            "pair_overlap_count": pair_overlap_count,
            "pair_overlap_share_raw": (
                pair_overlap_count / raw_pair_count if raw_pair_count else 0.0
            ),
            "pair_overlap_share_legacy": (
                pair_overlap_count / legacy_pair_count if legacy_pair_count else 0.0
            ),
        }
        raw_summary = {
            "source_type": "raw_eora_T",
            "row_count": stats["row_count"],
            "unique_supplier_buyer_pairs": raw_pair_count,
            "unique_suppliers": int(stats["supplier_present_any"].sum()),
            "unique_buyers": int(stats["buyer_present_any"].sum()),
            "total_weight": stats["total_transaction_value"],
            "median_edges_per_buyer": float(np.median(buyer_edge_counts))
            if len(buyer_edge_counts)
            else 0.0,
            "p95_edges_per_buyer": float(np.quantile(buyer_edge_counts, 0.95))
            if len(buyer_edge_counts)
            else 0.0,
            "max_edges_per_buyer": int(np.max(buyer_edge_counts))
            if len(buyer_edge_counts)
            else 0,
            "metadata_coverage": raw_metadata_coverage,
            "recommended_use": "canonical_supplier_substitution_source",
        }
        legacy_summary = self._source_summary(
            legacy_edges,
            source_type="legacy_abm_edges_embodied_emissions",
            recommended_use="carbon_diagnostics_not_input_sourcing_weight",
        )
        return pl.DataFrame(
            [
                {**raw_summary, **overlap_values},
                {**legacy_summary, **overlap_values},
            ]
        )

    def build_raw_t_edge_report(
        self,
        raw_edges: pl.DataFrame,
        state_panel: pl.DataFrame,
    ) -> pl.DataFrame:
        """Build the Phase 4A-bis raw Eora T supplier-edge report."""
        state_buyers = state_panel.select("country_sector").unique()
        edge_buyers = raw_edges.select(
            pl.col("buyer_country_sector").alias("country_sector")
        ).unique()
        buyers_without_edges = state_buyers.join(edge_buyers, on="country_sector", how="anti").height
        row_count = raw_edges.height
        supplier_metadata_rows = raw_edges.filter(pl.col("supplier_sector").is_not_null()).height
        buyer_metadata_rows = raw_edges.filter(pl.col("buyer_sector").is_not_null()).height
        ecosystem_metadata_rows = raw_edges.filter(
            pl.col("supplier_ecosystem_id").is_not_null()
            & pl.col("buyer_ecosystem_id").is_not_null()
        ).height
        years = raw_edges.select("year").unique().sort("year")["year"].to_list()

        return pl.DataFrame(
            {
                "selected_source": ["raw_eora_T"],
                "years_covered": [f"{min(years)}-{max(years)}" if years else ""],
                "row_count": [row_count],
                "unique_suppliers": [raw_edges["supplier_country_sector"].n_unique()],
                "unique_buyers": [raw_edges["buyer_country_sector"].n_unique()],
                "unique_supplier_buyer_pairs": [
                    raw_edges.select("supplier_country_sector", "buyer_country_sector").unique().height
                ],
                "total_transaction_value": [raw_edges["transaction_value"].sum()],
                "share_edges_with_supplier_metadata": [
                    supplier_metadata_rows / row_count if row_count else 0.0
                ],
                "share_edges_with_buyer_metadata": [
                    buyer_metadata_rows / row_count if row_count else 0.0
                ],
                "share_edges_with_ecosystem_metadata": [
                    ecosystem_metadata_rows / row_count if row_count else 0.0
                ],
                "buyers_without_edges": [buyers_without_edges],
                "notes": [
                    (
                        "Raw Eora T matrix edges with rows as suppliers and columns as buyers. "
                        "Recommended as canonical production-sourcing weights for supplier substitution."
                    )
                ],
            }
        )

    def compare_edge_sources(
        self,
        raw_edges: pl.DataFrame,
        legacy_edges: pl.DataFrame,
    ) -> pl.DataFrame:
        """Compare raw Eora T edges with legacy embodied-emissions edges."""
        raw_pairs = self._pair_frame(raw_edges)
        legacy_pairs = self._pair_frame(legacy_edges)
        pair_overlap_count = raw_pairs.join(
            legacy_pairs,
            on=["supplier_country_sector", "buyer_country_sector"],
            how="inner",
        ).height
        raw_pair_count = raw_pairs.height
        legacy_pair_count = legacy_pairs.height
        overlap_values = {
            "pair_overlap_count": pair_overlap_count,
            "pair_overlap_share_raw": (
                pair_overlap_count / raw_pair_count if raw_pair_count else 0.0
            ),
            "pair_overlap_share_legacy": (
                pair_overlap_count / legacy_pair_count if legacy_pair_count else 0.0
            ),
        }

        return pl.DataFrame(
            [
                {
                    **self._source_summary(
                        raw_edges,
                        source_type="raw_eora_T",
                        recommended_use="canonical_supplier_substitution_source",
                    ),
                    **overlap_values,
                },
                {
                    **self._source_summary(
                        legacy_edges,
                        source_type="legacy_abm_edges_embodied_emissions",
                        recommended_use="carbon_diagnostics_not_input_sourcing_weight",
                    ),
                    **overlap_values,
                },
            ]
        )

    def build_edge_report(
        self,
        edges: pl.DataFrame,
        state_panel: pl.DataFrame,
        selected_source: EdgeSource,
    ) -> pl.DataFrame:
        """Build supplier-edge coverage and provenance diagnostics."""
        state_buyers = state_panel.select("country_sector").unique()
        edge_buyers = edges.select(pl.col("buyer_country_sector").alias("country_sector")).unique()
        buyers_without_edges = state_buyers.join(edge_buyers, on="country_sector", how="anti").height
        row_count = edges.height
        supplier_metadata_rows = edges.filter(pl.col("supplier_sector").is_not_null()).height
        buyer_metadata_rows = edges.filter(pl.col("buyer_sector").is_not_null()).height
        ecosystem_metadata_rows = edges.filter(
            pl.col("supplier_ecosystem_id").is_not_null()
            & pl.col("buyer_ecosystem_id").is_not_null()
        ).height
        years = edges.select("year").unique().sort("year")["year"].to_list()

        return pl.DataFrame(
            {
                "selected_edge_source": [str(selected_source.path)],
                "source_type": [selected_source.source_type],
                "years_covered": [f"{min(years)}-{max(years)}" if years else ""],
                "row_count": [row_count],
                "unique_suppliers": [edges["supplier_country_sector"].n_unique()],
                "unique_buyers": [edges["buyer_country_sector"].n_unique()],
                "unique_supplier_buyer_pairs": [
                    edges.select("supplier_country_sector", "buyer_country_sector").unique().height
                ],
                "total_transaction_value": [edges["transaction_value"].sum()],
                "share_edges_with_supplier_metadata": [
                    supplier_metadata_rows / row_count if row_count else 0.0
                ],
                "share_edges_with_buyer_metadata": [
                    buyer_metadata_rows / row_count if row_count else 0.0
                ],
                "share_edges_with_ecosystem_metadata": [
                    ecosystem_metadata_rows / row_count if row_count else 0.0
                ],
                "buyers_without_edges": [buyers_without_edges],
                "notes": [
                    (
                        f"{selected_source.notes} Edge direction is "
                        "supplier_country_sector -> buyer_country_sector."
                    )
                ],
            }
        )

    def build_edge_schema_report(
        self,
        mappings: tuple[EdgeColumnMapping, ...],
        selected_source: EdgeSource,
    ) -> pl.DataFrame:
        """Build edge-schema mapping diagnostics."""
        return pl.DataFrame(
            [
                {
                    "canonical_column": mapping.canonical_column,
                    "source_column": mapping.source_column,
                    "mapping_status": mapping.mapping_status,
                    "notes": f"{mapping.notes} Source type: {selected_source.source_type}.",
                }
                for mapping in mappings
            ]
        )

    def _pair_frame(self, edges: pl.DataFrame) -> pl.DataFrame:
        return edges.select("supplier_country_sector", "buyer_country_sector").unique()

    def _source_summary(
        self,
        edges: pl.DataFrame,
        source_type: str,
        recommended_use: str,
    ) -> dict[str, object]:
        row_count = edges.height
        edge_counts_by_buyer = (
            edges.group_by("buyer_country_sector")
            .len(name="edge_count")
            .select("edge_count")
            if row_count
            else pl.DataFrame({"edge_count": []}, schema={"edge_count": pl.UInt32})
        )
        if row_count and {"supplier_sector", "buyer_sector"}.issubset(set(edges.columns)):
            metadata_rows = edges.filter(
                pl.col("supplier_sector").is_not_null()
                & pl.col("buyer_sector").is_not_null()
            ).height
            metadata_coverage = metadata_rows / row_count
        else:
            metadata_coverage = 0.0

        return {
            "source_type": source_type,
            "row_count": row_count,
            "unique_supplier_buyer_pairs": self._pair_frame(edges).height if row_count else 0,
            "unique_suppliers": edges["supplier_country_sector"].n_unique() if row_count else 0,
            "unique_buyers": edges["buyer_country_sector"].n_unique() if row_count else 0,
            "total_weight": edges["transaction_value"].sum() if row_count else 0.0,
            "median_edges_per_buyer": (
                edge_counts_by_buyer["edge_count"].median() if row_count else 0.0
            ),
            "p95_edges_per_buyer": (
                edge_counts_by_buyer["edge_count"].quantile(0.95) if row_count else 0.0
            ),
            "max_edges_per_buyer": (
                edge_counts_by_buyer["edge_count"].max() if row_count else 0
            ),
            "metadata_coverage": metadata_coverage,
            "recommended_use": recommended_use,
        }

    def _normalize_candidate_pool(
        self,
        candidates: pl.DataFrame,
        candidate_source: str,
        is_historical: bool,
        is_same_sector: bool,
        is_ecosystem: bool,
        schema_columns: dict[str, pl.DataType],
    ) -> pl.DataFrame:
        frame = candidates.clone()
        if "total_transaction_value" not in frame.columns:
            frame = frame.with_columns(pl.lit(None).cast(pl.Float64).alias("total_transaction_value"))
        for column_name, dtype in schema_columns.items():
            if column_name not in frame.columns:
                frame = frame.with_columns(pl.lit(None).cast(dtype).alias(column_name))
        return frame.with_columns(
            pl.lit(candidate_source).alias("candidate_source"),
            pl.lit(is_historical).alias("is_historical_candidate"),
            pl.lit(is_same_sector).alias("is_same_sector_candidate"),
            pl.lit(is_ecosystem).alias("is_ecosystem_candidate"),
        ).select(list(schema_columns))

    def _state_profiles_for_opportunities(
        self,
        opportunity_year: int | None,
        epsilon: float,
    ) -> pl.DataFrame:
        state_panel = self.load_state_panel()
        years = sorted(state_panel["Year"].drop_nulls().unique().to_list())
        if not years:
            raise ValueError("State panel has no available Year values.")
        latest_year = opportunity_year if opportunity_year is not None else max(years)
        previous_years = [year for year in years if year < latest_year]
        previous_year = max(previous_years) if previous_years else None
        latest = (
            state_panel.filter(pl.col("Year") == latest_year)
            .select(
                "country_sector",
                "Country",
                "Sector",
                "ecosystem_id",
                "ecosystem_label",
                "g_local_v4",
                "green_capability",
                "general_capability",
                "X_observed",
            )
            .unique(subset=["country_sector"])
        )
        if previous_year is None:
            previous = pl.DataFrame(
                schema={"country_sector": pl.Utf8, "X_observed_previous": pl.Float64}
            )
        else:
            previous = (
                state_panel.filter(pl.col("Year") == previous_year)
                .select(
                    "country_sector",
                    pl.col("X_observed").alias("X_observed_previous"),
                )
                .unique(subset=["country_sector"])
            )
        return (
            latest.join(previous, on="country_sector", how="left")
            .with_columns(
                (
                    1.0
                    - (
                        (pl.col("X_observed") - pl.col("X_observed_previous")).abs()
                        / (pl.col("X_observed_previous").abs() + epsilon)
                    )
                )
                .clip(0.0, 1.0)
                .alias("supplier_reliability")
            )
        )

    def _attach_opportunity_state_profiles(
        self,
        opportunities: pl.DataFrame,
        profiles: pl.DataFrame,
    ) -> pl.DataFrame:
        buyer_profiles = profiles.rename(
            {
                "country_sector": "buyer_country_sector",
                "Country": "_buyer_country_profile",
                "Sector": "_buyer_sector_profile",
                "ecosystem_id": "_buyer_ecosystem_id_profile",
                "ecosystem_label": "_buyer_ecosystem_label_profile",
                "g_local_v4": "buyer_g_local_v4",
                "green_capability": "_buyer_green_capability",
                "general_capability": "_buyer_general_capability",
                "X_observed": "_buyer_x_observed",
                "X_observed_previous": "_buyer_x_observed_previous",
                "supplier_reliability": "_buyer_reliability",
            }
        )
        supplier_profiles = profiles.rename(
            {
                "country_sector": "supplier_country_sector",
                "Country": "_supplier_country_profile",
                "Sector": "_supplier_sector_profile",
                "ecosystem_id": "_supplier_ecosystem_id_profile",
                "ecosystem_label": "_supplier_ecosystem_label_profile",
                "g_local_v4": "supplier_g_local_v4",
                "green_capability": "supplier_green_capability",
                "general_capability": "supplier_general_capability",
                "X_observed": "_supplier_x_observed",
                "X_observed_previous": "_supplier_x_observed_previous",
                "supplier_reliability": "supplier_reliability",
            }
        )
        return (
            opportunities.join(buyer_profiles, on="buyer_country_sector", how="left")
            .join(supplier_profiles, on="supplier_country_sector", how="left")
            .with_columns(
                pl.coalesce(["buyer_country", "_buyer_country_profile"]).alias("buyer_country"),
                pl.coalesce(["buyer_sector", "_buyer_sector_profile"]).alias("buyer_sector"),
                pl.coalesce(["buyer_ecosystem_id", "_buyer_ecosystem_id_profile"]).alias(
                    "buyer_ecosystem_id"
                ),
                pl.coalesce(
                    ["buyer_ecosystem_label", "_buyer_ecosystem_label_profile"]
                ).alias("buyer_ecosystem_label"),
                pl.coalesce(["supplier_country", "_supplier_country_profile"]).alias(
                    "supplier_country"
                ),
                pl.coalesce(["supplier_sector", "_supplier_sector_profile"]).alias(
                    "supplier_sector"
                ),
                pl.coalesce(
                    ["supplier_ecosystem_id", "_supplier_ecosystem_id_profile"]
                ).alias("supplier_ecosystem_id"),
                pl.coalesce(
                    ["supplier_ecosystem_label", "_supplier_ecosystem_label_profile"]
                ).alias("supplier_ecosystem_label"),
            )
            .drop(
                [
                    "_buyer_country_profile",
                    "_buyer_sector_profile",
                    "_buyer_ecosystem_id_profile",
                    "_buyer_ecosystem_label_profile",
                    "_buyer_green_capability",
                    "_buyer_general_capability",
                    "_buyer_x_observed",
                    "_buyer_x_observed_previous",
                    "_buyer_reliability",
                    "_supplier_country_profile",
                    "_supplier_sector_profile",
                    "_supplier_ecosystem_id_profile",
                    "_supplier_ecosystem_label_profile",
                    "_supplier_x_observed",
                    "_supplier_x_observed_previous",
                ]
            )
        )

    def _node_candidate_profiles(self) -> pl.DataFrame:
        """Return one compact ranking profile per country-sector node."""
        state_panel = self.load_state_panel()
        optional_rank_columns = [
            column_name
            for column_name in ("green_capability", "g_local_v4", "reliability")
            if column_name in state_panel.columns
        ]
        aggregations: list[pl.Expr] = [
            pl.first("Country").alias("Country"),
            pl.first("Sector").alias("Sector"),
            pl.first("ecosystem_id").alias("ecosystem_id"),
            pl.first("ecosystem_label").alias("ecosystem_label"),
        ]
        if "X_observed" in state_panel.columns:
            aggregations.append(
                pl.sum("X_observed").alias("total_supplier_output_or_transaction_proxy")
            )
        else:
            aggregations.append(
                pl.len().cast(pl.Float64).alias("total_supplier_output_or_transaction_proxy")
            )
        for column_name in optional_rank_columns:
            aggregations.append(pl.mean(column_name).alias(column_name))

        profiles = state_panel.group_by("country_sector").agg(aggregations)
        green_exprs: list[pl.Expr] = []
        green_sources = [
            pl.col(column_name)
            for column_name in ("green_capability", "g_local_v4")
            if column_name in profiles.columns
        ]
        if green_sources:
            green_exprs.append(
                pl.coalesce(green_sources).fill_null(0.0).alias("green_rank_value")
            )
        else:
            green_exprs.append(pl.lit(0.0).alias("green_rank_value"))
        if "reliability" in profiles.columns:
            green_exprs.append(
                pl.col("reliability").fill_null(0.0).alias("reliability_rank_value")
            )
        else:
            green_exprs.append(pl.lit(0.0).alias("reliability_rank_value"))
        return profiles.with_columns(green_exprs)

    def _candidate_pair_source_flags(
        self,
        historical_candidates: pl.DataFrame | None,
        same_sector_candidates: pl.DataFrame | None,
    ) -> dict[tuple[str, str], list[str]]:
        flags: dict[tuple[str, str], list[str]] = {}
        for source_name, candidates in (
            ("historical", historical_candidates),
            ("same_sector", same_sector_candidates),
        ):
            if candidates is None or candidates.is_empty():
                continue
            for row in candidates.select(
                "buyer_country_sector", "supplier_country_sector"
            ).iter_rows(named=True):
                pair = (row["buyer_country_sector"], row["supplier_country_sector"])
                flags.setdefault(pair, []).append(source_name)
        return flags

    def _candidate_count_summary(self, candidates: pl.DataFrame) -> dict[str, float]:
        if candidates.is_empty():
            return {"median": 0.0, "max": 0.0}
        counts = candidates.group_by("buyer_country_sector").len(name="candidate_count")
        return {
            "median": float(counts["candidate_count"].median() or 0.0),
            "max": float(counts["candidate_count"].max() or 0.0),
        }

    def _share_buyers_without_candidates(
        self,
        state_buyers: pl.DataFrame,
        candidates: pl.DataFrame,
    ) -> float:
        if state_buyers.is_empty():
            return 0.0
        candidate_buyers = candidates.select(
            pl.col("buyer_country_sector").alias("country_sector")
        ).unique()
        missing = state_buyers.join(candidate_buyers, on="country_sector", how="anti").height
        return missing / state_buyers.height

    def _same_sector_pool_schema(self) -> dict[str, pl.DataType]:
        return {
            "buyer_country_sector": pl.Utf8,
            "supplier_country_sector": pl.Utf8,
            "supplier_type": pl.Utf8,
            "supplier_rank": pl.Int64,
            "supplier_sector": pl.Utf8,
            "buyer_sector": pl.Utf8,
            "supplier_country": pl.Utf8,
            "buyer_country": pl.Utf8,
            "supplier_ecosystem_id": pl.Utf8,
            "buyer_ecosystem_id": pl.Utf8,
            "total_supplier_output_or_transaction_proxy": pl.Float64,
            "domestic_fallback_used": pl.Boolean,
            "source_type": pl.Utf8,
        }

    def _ecosystem_pool_schema(self) -> dict[str, pl.DataType]:
        return {
            "buyer_country_sector": pl.Utf8,
            "supplier_country_sector": pl.Utf8,
            "supplier_type": pl.Utf8,
            "ecosystem_proximity": pl.Float64,
            "supplier_rank": pl.Int64,
            "supplier_ecosystem_id": pl.Utf8,
            "buyer_ecosystem_id": pl.Utf8,
            "supplier_ecosystem_label": pl.Utf8,
            "buyer_ecosystem_label": pl.Utf8,
            "total_supplier_output_or_transaction_proxy": pl.Float64,
            "candidate_source_flags": pl.Utf8,
            "source_type": pl.Utf8,
        }

    def _partition_key_value(self, key: object) -> object:
        if isinstance(key, tuple) and len(key) == 1:
            return key[0]
        return key

    def _node_metadata_by_country_sector(
        self,
        state_panel: pl.DataFrame,
    ) -> dict[str, dict[str, object]]:
        metadata = (
            state_panel.select(
                "country_sector",
                "Country",
                "Sector",
                "ecosystem_id",
                "ecosystem_label",
            )
            .unique(subset=["country_sector"])
            .to_dicts()
        )
        return {
            str(row["country_sector"]): {
                "Country": row["Country"],
                "Sector": row["Sector"],
                "ecosystem_id": row["ecosystem_id"],
                "ecosystem_label": row["ecosystem_label"],
            }
            for row in metadata
        }

    def _read_t_matrix_arrays(self, t_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        matrix = pl.read_parquet(t_path)
        if "__index_level_0__" not in matrix.columns:
            raise ValueError(f"Eora T matrix lacks row labels: {t_path}")
        value_columns = [
            column_name
            for column_name in matrix.columns
            if column_name != "__index_level_0__"
        ]
        supplier_labels = matrix["__index_level_0__"].cast(pl.Utf8).to_numpy()
        buyer_labels = np.array(value_columns, dtype=object)
        transaction_matrix = matrix.select(value_columns).to_numpy()
        return transaction_matrix, supplier_labels, buyer_labels

    def _first_present_column(
        self,
        source_columns: set[str],
        candidate_columns: tuple[str, ...],
    ) -> str | None:
        for candidate_column in candidate_columns:
            if candidate_column in source_columns:
                return candidate_column
        return None

    def _build_mapping(
        self,
        canonical_column: str,
        source_column: str | None,
    ) -> EdgeColumnMapping:
        if source_column is None:
            return EdgeColumnMapping(
                canonical_column=canonical_column,
                source_column=None,
                mapping_status="missing",
                notes="No matching source column found.",
            )
        if source_column == canonical_column:
            return EdgeColumnMapping(
                canonical_column=canonical_column,
                source_column=source_column,
                mapping_status="direct",
                notes="Source column already uses the canonical name.",
            )
        notes = f"Mapped from {source_column}."
        if source_column == "source_agent_id":
            notes = "Mapped from source_agent_id; inspected ABM v1 builder uses source -> target."
        if source_column == "target_agent_id":
            notes = "Mapped from target_agent_id; inspected ABM v1 builder uses source -> target."
        if source_column == "embedded_emissions":
            notes = "Mapped from embedded_emissions; this is not raw Eora T transaction value."
        return EdgeColumnMapping(
            canonical_column=canonical_column,
            source_column=source_column,
            mapping_status="renamed",
            notes=notes,
        )
