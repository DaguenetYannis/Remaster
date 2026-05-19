from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.abm_v5.accounting import ACCOUNTING_OUTPUT_FILENAME
from src.abm_v5.config import DEFAULT_HISTORICAL_END_YEAR, DEFAULT_HISTORICAL_START_YEAR, ValidationLayer
from src.abm_v5.identity import load_agent_identity_panel
from src.abm_v5.paths import ABMV5Paths
from src.abm_v5.validation import ValidationResult, ValidationSeverity, ValidationStatus


HISTORICAL_START_YEAR = DEFAULT_HISTORICAL_START_YEAR
HISTORICAL_END_YEAR = DEFAULT_HISTORICAL_END_YEAR
MIN_EDGE_VALUE = 0.0

SUPPLIER_CANDIDATE_OUTPUT_FILENAME = "supplier_candidate_panel_1995_2016.parquet"
NETWORK_OUTPUT_FILENAME = "network_state_panel_1995_2016.parquet"
NETWORK_VALIDATION_FILENAME = "network_state_validation.json"
SUPPLIER_CANDIDATE_COVERAGE_SUMMARY_FILENAME = "supplier_candidate_coverage_summary.json"

EDGE_REQUIRED_COLUMNS = (
    "year",
    "supplier_country_sector",
    "buyer_country_sector",
    "transaction_value",
    "technical_coefficient",
    "supplier_weight",
)
NETWORK_REQUIRED_COLUMNS = (
    "country_sector",
    "year",
    "supplier_count",
    "buyer_count",
    "total_inputs_from_suppliers",
    "total_outputs_to_buyers",
    "supplier_concentration_hhi",
    "buyer_concentration_hhi",
    "import_dependence_proxy",
    "export_dependence_proxy",
    "network_green_exposure",
    "incoming_network_green_exposure",
    "outgoing_network_green_exposure",
    "brown_centrality",
    "supplier_lock_in",
)
SUPPLIER_CANDIDATE_REQUIRED_COLUMNS = (
    "year",
    "buyer_country_sector",
    "supplier_country_sector",
    "transaction_value",
    "supplier_weight",
    "technical_coefficient",
    "historical_supplier_flag",
    "same_sector_candidate_flag",
    "candidate_source",
    "candidate_rank_within_buyer",
    "buyer_input_cumulative_share",
    "retained_by_top_minimum_flag",
    "retained_by_weight_threshold_flag",
    "retained_by_coverage_flag",
    "coverage_target_unmet_flag",
    "compact_candidate_flag",
)
SUPPLIER_CANDIDATE_SOURCE_VALUES = ("historical_t", "same_sector_fallback")
SUMMARY_COLUMNS = (
    "supplier_count",
    "buyer_count",
    "supplier_concentration_hhi",
    "buyer_concentration_hhi",
    "import_dependence_proxy",
    "export_dependence_proxy",
    "network_green_exposure",
    "brown_centrality",
    "supplier_lock_in",
)


@dataclass(frozen=True)
class NetworkBuildResult:
    """Result metadata for ABM v5 historical production and network panels."""

    candidate_output_path: Path
    network_output_path: Path
    validation_path: Path
    coverage_summary_path: Path
    start_year: int
    end_year: int
    n_candidate_rows: int
    n_network_rows: int
    n_agents: int
    edge_output_path: Path | None = None

    def validate(self) -> None:
        """Validate network build result metadata."""
        if self.n_candidate_rows <= 0:
            raise ValueError("n_candidate_rows must be positive.")
        if self.n_network_rows <= 0:
            raise ValueError("n_network_rows must be positive.")
        if self.n_agents <= 0:
            raise ValueError("n_agents must be positive.")
        if self.start_year != DEFAULT_HISTORICAL_START_YEAR:
            raise ValueError("start_year must be 1995.")
        if self.end_year != DEFAULT_HISTORICAL_END_YEAR:
            raise ValueError("end_year must be 2016.")

    @property
    def n_edge_rows(self) -> int:
        """Backward-compatible alias for compact candidate rows."""
        return self.n_candidate_rows


def _read_label_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _value_columns(frame) -> list[str]:
    return [column for column in frame.columns if column != "__index_level_0__"]


def _country_from_country_sector(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if "\t" in text:
        return text.split("\t", 1)[0].strip() or None
    if "|" in text:
        return text.split("|", 1)[0].strip() or None
    return text.split(" ", 1)[0].strip() or None


def _validation_result(
    check_name: str,
    status: ValidationStatus,
    severity: ValidationSeverity,
    message: str,
    layer: ValidationLayer,
    n_failed: int,
    n_checked: int,
) -> ValidationResult:
    result = ValidationResult(
        check_name=check_name,
        status=status,
        severity=severity,
        message=message,
        layer=layer,
        n_failed=n_failed,
        n_checked=n_checked,
    )
    result.validate()
    return result


def build_production_edges_for_year(
    project_root: Path,
    year: int,
    min_edge_value: float = MIN_EDGE_VALUE,
):
    """Build positive production supplier-buyer edges from raw Eora T for one year."""
    import polars as pl

    year_dir = project_root / "data" / "parquet" / str(year)
    raw_dir = project_root / "data" / "raw" / str(year)
    t_matrix = pl.read_parquet(year_dir / "T.parquet")
    labels_t = _read_label_lines(raw_dir / "labels_T.txt")
    value_columns = _value_columns(t_matrix)
    node_count = len(labels_t)
    if t_matrix.height != node_count or len(value_columns) != node_count:
        raise ValueError(f"T matrix dimensions do not align with labels_T for year {year}.")

    # Repository ABM v4 supplier construction documents raw Eora T as
    # T[row, column] = production flow from supplier row to buyer column.
    renamed_matrix = t_matrix.rename(
        {column: labels_t[index] for index, column in enumerate(value_columns)}
    )
    return (
        renamed_matrix.with_columns(pl.Series("supplier_country_sector", labels_t))
        .select("supplier_country_sector", *labels_t)
        .unpivot(
            index="supplier_country_sector",
            variable_name="buyer_country_sector",
            value_name="transaction_value",
        )
        .with_columns(
            pl.lit(year).cast(pl.Int64).alias("year"),
            pl.col("transaction_value").cast(pl.Float64, strict=False),
        )
        .filter(pl.col("transaction_value") > min_edge_value)
        .select("year", "supplier_country_sector", "buyer_country_sector", "transaction_value")
    )


def add_supplier_weights_and_coefficients(edge_df, accounting_df_for_year):
    """Add production input coefficients and supplier shares to one year's T edges."""
    import polars as pl

    edges = edge_df if isinstance(edge_df, pl.DataFrame) else pl.DataFrame(edge_df)
    accounting = (
        accounting_df_for_year
        if isinstance(accounting_df_for_year, pl.DataFrame)
        else pl.DataFrame(accounting_df_for_year)
    )
    outputs = accounting.select(
        "country_sector",
        pl.col("output").cast(pl.Float64, strict=False).alias("output"),
    )
    buyer_outputs = outputs.rename(
        {"country_sector": "buyer_country_sector", "output": "buyer_output"}
    )
    supplier_outputs = outputs.rename(
        {"country_sector": "supplier_country_sector", "output": "supplier_output"}
    )
    return (
        edges.with_columns(pl.col("transaction_value").cast(pl.Float64, strict=False))
        .join(buyer_outputs, on="buyer_country_sector", how="left")
        .join(supplier_outputs, on="supplier_country_sector", how="left")
        .with_columns(
            pl.sum("transaction_value").over(["year", "buyer_country_sector"]).alias("_buyer_total")
        )
        .with_columns(
            pl.when(pl.col("buyer_output") > 0)
            .then(pl.col("transaction_value") / pl.col("buyer_output"))
            .otherwise(None)
            .alias("technical_coefficient"),
            pl.when(pl.col("_buyer_total") > 0)
            .then(pl.col("transaction_value") / pl.col("_buyer_total"))
            .otherwise(None)
            .alias("supplier_weight"),
        )
        .drop("_buyer_total")
    )


def build_supplier_candidates_for_year(
    edge_df_for_year,
    accounting_df_for_year,
    min_top_suppliers: int = 10,
    soft_max_historical_suppliers: int = 100,
    hard_max_historical_suppliers: int = 250,
    coverage_target: float = 0.95,
    supplier_weight_threshold: float = 0.001,
    fallback_candidates_per_buyer: int = 25,
):
    """Build a compact supplier candidate panel from one year's positive T edges."""
    import polars as pl

    edges = edge_df_for_year if isinstance(edge_df_for_year, pl.DataFrame) else pl.DataFrame(edge_df_for_year)
    if edges.is_empty():
        return _empty_supplier_candidate_frame()

    ranked = (
        edges.sort(["year", "buyer_country_sector", "transaction_value"], descending=[False, False, True])
        .with_columns(
            pl.int_range(1, pl.len() + 1).over(["year", "buyer_country_sector"]).alias(
                "candidate_rank_within_buyer"
            ),
            pl.col("supplier_weight").cum_sum().over(["year", "buyer_country_sector"]).alias(
                "buyer_input_cumulative_share"
            ),
        )
        .with_columns(
            pl.col("buyer_input_cumulative_share")
            .shift(1)
            .over(["year", "buyer_country_sector"])
            .fill_null(0.0)
            .alias("_previous_cumulative_share"),
            (pl.col("candidate_rank_within_buyer") <= min_top_suppliers).alias(
                "retained_by_top_minimum_flag"
            ),
            (pl.col("supplier_weight") >= supplier_weight_threshold).alias(
                "retained_by_weight_threshold_flag"
            ),
        )
        .with_columns(
            (pl.col("_previous_cumulative_share") < coverage_target).alias(
                "retained_by_coverage_flag"
            )
        )
        .filter(
            (
                pl.col("retained_by_top_minimum_flag")
                | pl.col("retained_by_weight_threshold_flag")
                | pl.col("retained_by_coverage_flag")
            )
            & (pl.col("candidate_rank_within_buyer") <= hard_max_historical_suppliers)
        )
    )

    coverage = ranked.group_by(["year", "buyer_country_sector"]).agg(
        pl.max("buyer_input_cumulative_share").alias("_retained_input_coverage"),
        pl.len().alias("_retained_supplier_count"),
    )
    ranked = (
        ranked.join(coverage, on=["year", "buyer_country_sector"], how="left")
        .with_columns(
            (pl.col("_retained_input_coverage") < coverage_target).alias(
                "coverage_target_unmet_flag"
            ),
            pl.lit(True).alias("historical_supplier_flag"),
            pl.lit(False).alias("same_sector_candidate_flag"),
            pl.lit("historical_t").alias("candidate_source"),
            pl.lit(True).alias("compact_candidate_flag"),
            (pl.col("_retained_supplier_count") > soft_max_historical_suppliers).alias(
                "soft_max_historical_suppliers_exceeded_flag"
            ),
        )
        .select(SUPPLIER_CANDIDATE_REQUIRED_COLUMNS)
    )

    fallback = build_same_sector_fallback_candidates_for_year(
        ranked,
        accounting_df_for_year,
        fallback_candidates_per_buyer=fallback_candidates_per_buyer,
        coverage_target=coverage_target,
        min_top_suppliers=min_top_suppliers,
    )
    if fallback.is_empty():
        return ranked
    return pl.concat([ranked, fallback], how="vertical").unique(
        subset=["year", "buyer_country_sector", "supplier_country_sector"], keep="first"
    )


def _empty_supplier_candidate_frame():
    import polars as pl

    return pl.DataFrame(
        schema={
            "year": pl.Int64,
            "buyer_country_sector": pl.Utf8,
            "supplier_country_sector": pl.Utf8,
            "transaction_value": pl.Float64,
            "supplier_weight": pl.Float64,
            "technical_coefficient": pl.Float64,
            "historical_supplier_flag": pl.Boolean,
            "same_sector_candidate_flag": pl.Boolean,
            "candidate_source": pl.Utf8,
            "candidate_rank_within_buyer": pl.Int64,
            "buyer_input_cumulative_share": pl.Float64,
            "retained_by_top_minimum_flag": pl.Boolean,
            "retained_by_weight_threshold_flag": pl.Boolean,
            "retained_by_coverage_flag": pl.Boolean,
            "coverage_target_unmet_flag": pl.Boolean,
            "compact_candidate_flag": pl.Boolean,
        }
    )


def build_same_sector_fallback_candidates_for_year(
    retained_candidates_for_year,
    accounting_df_for_year,
    fallback_candidates_per_buyer: int = 25,
    coverage_target: float = 0.95,
    min_top_suppliers: int = 10,
):
    """Build same-sector opportunity candidates without fabricating transaction values."""
    import polars as pl

    retained = (
        retained_candidates_for_year
        if isinstance(retained_candidates_for_year, pl.DataFrame)
        else pl.DataFrame(retained_candidates_for_year)
    )
    accounting = (
        accounting_df_for_year
        if isinstance(accounting_df_for_year, pl.DataFrame)
        else pl.DataFrame(accounting_df_for_year)
    )
    if retained.is_empty() or accounting.is_empty() or "sector" not in accounting.columns:
        return _empty_supplier_candidate_frame()

    retained_summary = retained.group_by(["year", "buyer_country_sector"]).agg(
        pl.max("buyer_input_cumulative_share").alias("_retained_input_coverage"),
        pl.len().alias("_retained_supplier_count"),
    )
    buyers_needing_fallback = retained_summary.filter(
        (pl.col("_retained_input_coverage") < coverage_target)
        | (pl.col("_retained_supplier_count") < min_top_suppliers)
    )
    if buyers_needing_fallback.is_empty():
        return _empty_supplier_candidate_frame()

    buyer_metadata = accounting.select(
        pl.col("year"),
        pl.col("country_sector").alias("buyer_country_sector"),
        pl.col("sector").alias("buyer_sector"),
    )
    supplier_pool = accounting.select(
        pl.col("year"),
        pl.col("country_sector").alias("supplier_country_sector"),
        pl.col("sector").alias("supplier_sector"),
        pl.col("output").cast(pl.Float64, strict=False).alias("supplier_output"),
    ).filter(
        pl.col("supplier_country_sector").is_not_null()
        & (pl.col("supplier_country_sector").cast(pl.Utf8).str.strip_chars() != "")
        & (pl.col("supplier_output") > 0)
    )
    existing_pairs = retained.select("year", "buyer_country_sector", "supplier_country_sector")
    fallback = (
        buyers_needing_fallback.select("year", "buyer_country_sector")
        .join(buyer_metadata, on=["year", "buyer_country_sector"], how="left")
        .join(supplier_pool, on="year", how="inner")
        .filter(pl.col("buyer_sector") == pl.col("supplier_sector"))
        .join(existing_pairs, on=["year", "buyer_country_sector", "supplier_country_sector"], how="anti")
        .sort(["year", "buyer_country_sector", "supplier_output"], descending=[False, False, True])
        .with_columns(
            pl.int_range(1, pl.len() + 1).over(["year", "buyer_country_sector"]).alias(
                "candidate_rank_within_buyer"
            )
        )
        .filter(pl.col("candidate_rank_within_buyer") <= fallback_candidates_per_buyer)
        .with_columns(
            pl.lit(None).cast(pl.Float64).alias("transaction_value"),
            pl.lit(None).cast(pl.Float64).alias("supplier_weight"),
            pl.lit(None).cast(pl.Float64).alias("technical_coefficient"),
            pl.lit(False).alias("historical_supplier_flag"),
            pl.lit(True).alias("same_sector_candidate_flag"),
            pl.lit("same_sector_fallback").alias("candidate_source"),
            pl.lit(None).cast(pl.Float64).alias("buyer_input_cumulative_share"),
            pl.lit(False).alias("retained_by_top_minimum_flag"),
            pl.lit(False).alias("retained_by_weight_threshold_flag"),
            pl.lit(False).alias("retained_by_coverage_flag"),
            pl.lit(False).alias("coverage_target_unmet_flag"),
            pl.lit(True).alias("compact_candidate_flag"),
        )
        .select(SUPPLIER_CANDIDATE_REQUIRED_COLUMNS)
    )
    return fallback


def summarize_supplier_candidate_coverage(raw_edge_df_for_year, candidate_df_for_year) -> dict[str, float | int]:
    """Summarize one year's raw-to-compact supplier candidate coverage."""
    import polars as pl

    raw_edges = raw_edge_df_for_year if isinstance(raw_edge_df_for_year, pl.DataFrame) else pl.DataFrame(raw_edge_df_for_year)
    candidates = (
        candidate_df_for_year
        if isinstance(candidate_df_for_year, pl.DataFrame)
        else pl.DataFrame(candidate_df_for_year)
    )
    year = int(raw_edges["year"].drop_nulls()[0]) if raw_edges.height else None
    historical = candidates.filter(pl.col("candidate_source") == "historical_t")
    fallback = candidates.filter(pl.col("candidate_source") == "same_sector_fallback")
    raw_total = raw_edges.select(pl.sum("transaction_value")).item() or 0.0
    retained_total = historical.select(pl.sum("transaction_value")).item() or 0.0
    buyer_coverage = historical.group_by(["year", "buyer_country_sector"]).agg(
        pl.max("buyer_input_cumulative_share").alias("_coverage")
    )
    candidate_counts = candidates.group_by(["year", "buyer_country_sector"]).len()
    buyers_with_fallback = fallback.select("year", "buyer_country_sector").unique().height
    buyers_unmet = historical.filter(pl.col("coverage_target_unmet_flag")).select(
        "year", "buyer_country_sector"
    ).unique().height
    return {
        "year": year,
        "raw_positive_edges": raw_edges.height,
        "retained_historical_candidate_rows": historical.height,
        "fallback_candidate_rows": fallback.height,
        "total_candidate_rows": candidates.height,
        "retained_edge_share": historical.height / raw_edges.height if raw_edges.height else 0.0,
        "raw_transaction_value_total": float(raw_total),
        "retained_historical_transaction_value_total": float(retained_total),
        "retained_transaction_value_coverage": float(retained_total / raw_total) if raw_total else 0.0,
        "mean_buyer_input_coverage": float(buyer_coverage.select(pl.mean("_coverage")).item() or 0.0)
        if buyer_coverage.height
        else 0.0,
        "share_buyer_years_reaching_coverage_target": float(
            buyer_coverage.filter(pl.col("_coverage") >= 0.95).height / buyer_coverage.height
        )
        if buyer_coverage.height
        else 0.0,
        "max_candidates_per_buyer_year": int(candidate_counts["len"].max() or 0) if candidate_counts.height else 0,
        "buyers_with_coverage_target_unmet": buyers_unmet,
        "buyers_with_fallback_candidates": buyers_with_fallback,
    }


def validate_edge_state_panel(edge_df) -> tuple[ValidationResult, ...]:
    """Validate ABM v5 production-network edge metadata."""
    import polars as pl

    frame = edge_df if isinstance(edge_df, pl.DataFrame) else pl.DataFrame(edge_df)
    results: list[ValidationResult] = []
    missing_columns = sorted(column for column in EDGE_REQUIRED_COLUMNS if column not in frame.columns)
    results.append(
        _validation_result(
            "edge_required_columns",
            ValidationStatus.FAILED if missing_columns else ValidationStatus.PASSED,
            ValidationSeverity.CRITICAL if missing_columns else ValidationSeverity.INFO,
            f"Missing required edge columns: {missing_columns}."
            if missing_columns
            else "All required edge columns are present.",
            ValidationLayer.STRUCTURAL_VALIDITY,
            len(missing_columns),
            len(EDGE_REQUIRED_COLUMNS),
        )
    )
    if missing_columns:
        return tuple(results)

    duplicate_count = int(
        frame.group_by(["year", "supplier_country_sector", "buyer_country_sector"])
        .len()
        .filter(pl.col("len") > 1)["len"]
        .sum()
        or 0
    )
    results.append(
        _validation_result(
            "edge_unique_year_supplier_buyer",
            ValidationStatus.FAILED if duplicate_count else ValidationStatus.PASSED,
            ValidationSeverity.CRITICAL if duplicate_count else ValidationSeverity.INFO,
            f"Found {duplicate_count} duplicate year-supplier-buyer rows."
            if duplicate_count
            else "year-supplier-buyer rows are unique.",
            ValidationLayer.STRUCTURAL_VALIDITY,
            duplicate_count,
            frame.height,
        )
    )

    nonpositive_edges = frame.filter(pl.col("transaction_value") <= 0).height
    results.append(
        _validation_result(
            "edge_transaction_value_positive",
            ValidationStatus.FAILED if nonpositive_edges else ValidationStatus.PASSED,
            ValidationSeverity.ERROR if nonpositive_edges else ValidationSeverity.INFO,
            f"Found {nonpositive_edges} nonpositive production edges."
            if nonpositive_edges
            else "Production edge transaction values are strictly positive.",
            ValidationLayer.ACCOUNTING_VALIDITY,
            nonpositive_edges,
            frame.height,
        )
    )

    invalid_weights = frame.filter(
        pl.col("supplier_weight").is_not_null()
        & ((pl.col("supplier_weight") < 0) | (pl.col("supplier_weight") > 1))
    ).height
    results.append(
        _validation_result(
            "edge_supplier_weight_bounds",
            ValidationStatus.FAILED if invalid_weights else ValidationStatus.PASSED,
            ValidationSeverity.ERROR if invalid_weights else ValidationSeverity.INFO,
            f"Found {invalid_weights} supplier weights outside [0, 1]."
            if invalid_weights
            else "Supplier weights are within [0, 1] where present.",
            ValidationLayer.STRUCTURAL_VALIDITY,
            invalid_weights,
            frame.height,
        )
    )

    invalid_coefficients = frame.filter(
        pl.col("technical_coefficient").is_not_null() & (pl.col("technical_coefficient") < 0)
    ).height
    results.append(
        _validation_result(
            "edge_technical_coefficient_nonnegative",
            ValidationStatus.FAILED if invalid_coefficients else ValidationStatus.PASSED,
            ValidationSeverity.ERROR if invalid_coefficients else ValidationSeverity.INFO,
            f"Found {invalid_coefficients} negative technical coefficients."
            if invalid_coefficients
            else "Technical coefficients are nonnegative where present.",
            ValidationLayer.ACCOUNTING_VALIDITY,
            invalid_coefficients,
            frame.height,
        )
    )

    weight_sums = frame.group_by(["year", "buyer_country_sector"]).agg(
        pl.sum("transaction_value").alias("_transaction_total"),
        pl.sum("supplier_weight").alias("_weight_sum"),
    )
    bad_weight_sums = weight_sums.filter(
        (pl.col("_transaction_total") > 0)
        & (pl.col("_weight_sum").is_null() | ((pl.col("_weight_sum") - 1.0).abs() > 1e-8))
    ).height
    results.append(
        _validation_result(
            "edge_supplier_weights_sum_to_one",
            ValidationStatus.FAILED if bad_weight_sums else ValidationStatus.PASSED,
            ValidationSeverity.ERROR if bad_weight_sums else ValidationSeverity.INFO,
            f"Found {bad_weight_sums} buyer-year groups with supplier weights not summing to 1."
            if bad_weight_sums
            else "Supplier weights sum to 1 by buyer-year where transaction totals are positive.",
            ValidationLayer.STRUCTURAL_VALIDITY,
            bad_weight_sums,
            weight_sums.height,
        )
    )

    years = sorted(frame["year"].drop_nulls().unique().to_list())
    out_of_range_years = [year for year in years if year < HISTORICAL_START_YEAR or year > HISTORICAL_END_YEAR]
    results.append(
        _validation_result(
            "edge_year_range",
            ValidationStatus.FAILED if out_of_range_years else ValidationStatus.PASSED,
            ValidationSeverity.ERROR if out_of_range_years else ValidationSeverity.INFO,
            f"Edge panel has years outside 1995-2016: {out_of_range_years}."
            if out_of_range_years
            else "Edge panel years are within 1995-2016.",
            ValidationLayer.STRUCTURAL_VALIDITY,
            len(out_of_range_years),
            len(years),
        )
    )
    return tuple(results)


def validate_supplier_candidate_panel(candidate_df) -> tuple[ValidationResult, ...]:
    """Validate compact ABM v5 supplier candidate metadata."""
    import polars as pl

    frame = candidate_df if isinstance(candidate_df, pl.DataFrame) else pl.DataFrame(candidate_df)
    results: list[ValidationResult] = []
    missing_columns = sorted(
        column for column in SUPPLIER_CANDIDATE_REQUIRED_COLUMNS if column not in frame.columns
    )
    results.append(
        _validation_result(
            "supplier_candidate_required_columns",
            ValidationStatus.FAILED if missing_columns else ValidationStatus.PASSED,
            ValidationSeverity.CRITICAL if missing_columns else ValidationSeverity.INFO,
            f"Missing required supplier candidate columns: {missing_columns}."
            if missing_columns
            else "All required supplier candidate columns are present.",
            ValidationLayer.STRUCTURAL_VALIDITY,
            len(missing_columns),
            len(SUPPLIER_CANDIDATE_REQUIRED_COLUMNS),
        )
    )
    if missing_columns:
        return tuple(results)

    duplicate_count = int(
        frame.group_by(["year", "buyer_country_sector", "supplier_country_sector"])
        .len()
        .filter(pl.col("len") > 1)["len"]
        .sum()
        or 0
    )
    results.append(
        _validation_result(
            "supplier_candidate_unique_year_buyer_supplier",
            ValidationStatus.FAILED if duplicate_count else ValidationStatus.PASSED,
            ValidationSeverity.CRITICAL if duplicate_count else ValidationSeverity.INFO,
            f"Found {duplicate_count} duplicate year-buyer-supplier candidate rows."
            if duplicate_count
            else "Supplier candidate year-buyer-supplier rows are unique.",
            ValidationLayer.STRUCTURAL_VALIDITY,
            duplicate_count,
            frame.height,
        )
    )

    non_compact = frame.filter(pl.col("compact_candidate_flag") != True).height
    invalid_sources = frame.filter(~pl.col("candidate_source").is_in(SUPPLIER_CANDIDATE_SOURCE_VALUES)).height
    both_flags = frame.filter(pl.col("historical_supplier_flag") & pl.col("same_sector_candidate_flag")).height
    invalid_weights = frame.filter(
        pl.col("supplier_weight").is_not_null()
        & ((pl.col("supplier_weight") < 0) | (pl.col("supplier_weight") > 1))
    ).height
    invalid_cumulative = frame.filter(
        pl.col("buyer_input_cumulative_share").is_not_null()
        & ((pl.col("buyer_input_cumulative_share") < 0) | (pl.col("buyer_input_cumulative_share") > 1))
    ).height
    historical_missing = frame.filter(
        (pl.col("candidate_source") == "historical_t")
        & (
            pl.col("transaction_value").is_null()
            | pl.col("supplier_weight").is_null()
        )
    ).height
    fallback_non_null = frame.filter(
        (pl.col("candidate_source") == "same_sector_fallback")
        & (
            pl.col("transaction_value").is_not_null()
            | pl.col("supplier_weight").is_not_null()
            | pl.col("technical_coefficient").is_not_null()
        )
    ).height
    historical_pairs = frame.filter(pl.col("candidate_source") == "historical_t").select(
        "year", "buyer_country_sector", "supplier_country_sector"
    )
    fallback_pairs = frame.filter(pl.col("candidate_source") == "same_sector_fallback").select(
        "year", "buyer_country_sector", "supplier_country_sector"
    )
    overlapping_fallback = fallback_pairs.join(
        historical_pairs,
        on=["year", "buyer_country_sector", "supplier_country_sector"],
        how="inner",
    ).height

    checks = (
        (
            "supplier_candidate_compact_flag_true",
            non_compact,
            "All supplier candidate rows are compact candidates.",
        ),
        (
            "supplier_candidate_source_values",
            invalid_sources,
            "Supplier candidate source values are controlled.",
        ),
        (
            "supplier_candidate_flags_not_both_true",
            both_flags,
            "Historical and same-sector fallback flags are not both true.",
        ),
        (
            "supplier_candidate_weight_bounds",
            invalid_weights,
            "Supplier weights are within [0, 1] where present.",
        ),
        (
            "supplier_candidate_cumulative_share_bounds",
            invalid_cumulative,
            "Buyer input cumulative shares are within [0, 1] where present.",
        ),
        (
            "supplier_candidate_historical_values_present",
            historical_missing,
            "Historical candidates have transaction values and supplier weights.",
        ),
        (
            "supplier_candidate_fallback_values_null",
            fallback_non_null,
            "Fallback candidates do not fabricate transaction values, weights, or coefficients.",
        ),
        (
            "supplier_candidate_fallback_excludes_historical_pairs",
            overlapping_fallback,
            "Fallback candidates do not duplicate retained historical supplier pairs.",
        ),
    )
    for check_name, n_failed, passed_message in checks:
        results.append(
            _validation_result(
                check_name,
                ValidationStatus.FAILED if n_failed else ValidationStatus.PASSED,
                ValidationSeverity.ERROR if n_failed else ValidationSeverity.INFO,
                f"{check_name} failed for {n_failed} rows." if n_failed else passed_message,
                ValidationLayer.STRUCTURAL_VALIDITY,
                int(n_failed),
                frame.height,
            )
        )
    return tuple(results)


def build_network_state_for_year(edge_df_for_year, accounting_df_for_year):
    """Build node-level production-network diagnostics for one year."""
    import polars as pl

    edges = edge_df_for_year if isinstance(edge_df_for_year, pl.DataFrame) else pl.DataFrame(edge_df_for_year)
    accounting = (
        accounting_df_for_year
        if isinstance(accounting_df_for_year, pl.DataFrame)
        else pl.DataFrame(accounting_df_for_year)
    )
    year = int(accounting["year"].drop_nulls()[0]) if accounting.height else None
    nodes = accounting.select(
        "country_sector",
        "year",
        pl.col("local_greenness").cast(pl.Float64, strict=False).alias("local_greenness"),
        pl.col("country").cast(pl.Utf8, strict=False).alias("country") if "country" in accounting.columns else pl.lit(None).cast(pl.Utf8).alias("country"),
    ).with_columns(
        pl.when(pl.col("country").is_not_null() & (pl.col("country").str.strip_chars() != ""))
        .then(pl.col("country"))
        .otherwise(pl.col("country_sector").map_elements(_country_from_country_sector, return_dtype=pl.Utf8))
        .alias("_node_country")
    )
    if edges.is_empty():
        return nodes.select("country_sector", "year").with_columns(
            pl.lit(0).cast(pl.Int64).alias("supplier_count"),
            pl.lit(0).cast(pl.Int64).alias("buyer_count"),
            pl.lit(0.0).alias("total_inputs_from_suppliers"),
            pl.lit(0.0).alias("total_outputs_to_buyers"),
            pl.lit(None).cast(pl.Float64).alias("supplier_concentration_hhi"),
            pl.lit(None).cast(pl.Float64).alias("buyer_concentration_hhi"),
            pl.lit(None).cast(pl.Float64).alias("import_dependence_proxy"),
            pl.lit(None).cast(pl.Float64).alias("export_dependence_proxy"),
            pl.lit(None).cast(pl.Float64).alias("network_green_exposure"),
            pl.lit(None).cast(pl.Float64).alias("incoming_network_green_exposure"),
            pl.lit(None).cast(pl.Float64).alias("outgoing_network_green_exposure"),
            pl.lit(None).cast(pl.Float64).alias("brown_centrality"),
            pl.lit(None).cast(pl.Float64).alias("supplier_lock_in"),
        )

    node_lookup = nodes.select("country_sector", "_node_country", "local_greenness")
    supplier_lookup = node_lookup.rename(
        {
            "country_sector": "supplier_country_sector",
            "_node_country": "supplier_country",
            "local_greenness": "supplier_local_greenness",
        }
    )
    buyer_lookup = node_lookup.rename(
        {
            "country_sector": "buyer_country_sector",
            "_node_country": "buyer_country",
            "local_greenness": "buyer_local_greenness",
        }
    )
    enriched_edges = (
        edges.join(supplier_lookup, on="supplier_country_sector", how="left")
        .join(buyer_lookup, on="buyer_country_sector", how="left")
        .with_columns(
            pl.sum("transaction_value").over(["year", "supplier_country_sector"]).alias(
                "_supplier_total_output"
            )
        )
        .with_columns(
            pl.when(pl.col("_supplier_total_output") > 0)
            .then(pl.col("transaction_value") / pl.col("_supplier_total_output"))
            .otherwise(None)
            .alias("_buyer_share")
        )
    )

    supplier_side = enriched_edges.group_by(["year", "buyer_country_sector"]).agg(
        pl.n_unique("supplier_country_sector").alias("supplier_count"),
        pl.sum("transaction_value").alias("total_inputs_from_suppliers"),
        (pl.col("supplier_weight") ** 2).sum().alias("supplier_concentration_hhi"),
        (
            pl.when(pl.col("supplier_country") != pl.col("buyer_country"))
            .then(pl.col("transaction_value"))
            .otherwise(0.0)
        ).sum().alias("_foreign_inputs"),
        (
            pl.when(pl.col("supplier_local_greenness").is_not_null())
            .then(pl.col("supplier_weight") * pl.col("supplier_local_greenness"))
            .otherwise(0.0)
        ).sum().alias("_incoming_green_weighted"),
        (
            pl.when(pl.col("supplier_local_greenness").is_not_null())
            .then(pl.col("supplier_weight"))
            .otherwise(0.0)
        ).sum().alias("_incoming_green_weight"),
    ).with_columns(
        pl.when(pl.col("total_inputs_from_suppliers") > 0)
        .then(pl.col("_foreign_inputs") / pl.col("total_inputs_from_suppliers"))
        .otherwise(None)
        .alias("import_dependence_proxy"),
        pl.when(pl.col("_incoming_green_weight") > 0)
        .then(pl.col("_incoming_green_weighted") / pl.col("_incoming_green_weight"))
        .otherwise(None)
        .alias("incoming_network_green_exposure"),
    ).rename({"buyer_country_sector": "country_sector"}).drop(
        "_foreign_inputs",
        "_incoming_green_weighted",
        "_incoming_green_weight",
    )

    buyer_side = enriched_edges.group_by(["year", "supplier_country_sector"]).agg(
        pl.n_unique("buyer_country_sector").alias("buyer_count"),
        pl.sum("transaction_value").alias("total_outputs_to_buyers"),
        (pl.col("_buyer_share") ** 2).sum().alias("buyer_concentration_hhi"),
        (
            pl.when(pl.col("supplier_country") != pl.col("buyer_country"))
            .then(pl.col("transaction_value"))
            .otherwise(0.0)
        ).sum().alias("_foreign_outputs"),
        (
            pl.when(pl.col("buyer_local_greenness").is_not_null())
            .then(pl.col("_buyer_share") * pl.col("buyer_local_greenness"))
            .otherwise(0.0)
        ).sum().alias("_outgoing_green_weighted"),
        (
            pl.when(pl.col("buyer_local_greenness").is_not_null())
            .then(pl.col("_buyer_share"))
            .otherwise(0.0)
        ).sum().alias("_outgoing_green_weight"),
    ).with_columns(
        pl.when(pl.col("total_outputs_to_buyers") > 0)
        .then(pl.col("_foreign_outputs") / pl.col("total_outputs_to_buyers"))
        .otherwise(None)
        .alias("export_dependence_proxy"),
        pl.when(pl.col("_outgoing_green_weight") > 0)
        .then(pl.col("_outgoing_green_weighted") / pl.col("_outgoing_green_weight"))
        .otherwise(None)
        .alias("outgoing_network_green_exposure"),
    ).rename({"supplier_country_sector": "country_sector"}).drop(
        "_foreign_outputs",
        "_outgoing_green_weighted",
        "_outgoing_green_weight",
    )

    network = (
        nodes.select("country_sector", "year", "local_greenness")
        .join(supplier_side, on=["country_sector", "year"], how="left")
        .join(buyer_side, on=["country_sector", "year"], how="left")
        .with_columns(
            pl.col("supplier_count").fill_null(0).cast(pl.Int64),
            pl.col("buyer_count").fill_null(0).cast(pl.Int64),
            pl.col("total_inputs_from_suppliers").fill_null(0.0),
            pl.col("total_outputs_to_buyers").fill_null(0.0),
            pl.when(
                pl.col("incoming_network_green_exposure").is_not_null()
                & pl.col("outgoing_network_green_exposure").is_not_null()
            )
            .then(
                0.5 * pl.col("incoming_network_green_exposure")
                + 0.5 * pl.col("outgoing_network_green_exposure")
            )
            .when(pl.col("incoming_network_green_exposure").is_not_null())
            .then(pl.col("incoming_network_green_exposure"))
            .otherwise(pl.col("outgoing_network_green_exposure"))
            .alias("network_green_exposure"),
            pl.col("supplier_concentration_hhi").alias("supplier_lock_in"),
        )
        .with_columns(
            pl.when(pl.col("local_greenness").is_not_null())
            .then(pl.col("total_outputs_to_buyers") * (1.0 - pl.col("local_greenness")))
            .otherwise(None)
            .alias("_brown_raw")
        )
    )
    raw_values = network["_brown_raw"].drop_nulls()
    if raw_values.is_empty():
        network = network.with_columns(pl.lit(None).cast(pl.Float64).alias("brown_centrality"))
    else:
        min_raw = float(raw_values.min())
        max_raw = float(raw_values.max())
        if min_raw == max_raw:
            network = network.with_columns(
                pl.when(pl.col("_brown_raw").is_not_null())
                .then(pl.lit(0.5))
                .otherwise(None)
                .alias("brown_centrality")
            )
        else:
            network = network.with_columns(
                pl.when(pl.col("_brown_raw").is_not_null())
                .then((pl.col("_brown_raw") - min_raw) / (max_raw - min_raw))
                .otherwise(None)
                .alias("brown_centrality")
            )
    return network.select(NETWORK_REQUIRED_COLUMNS).with_columns(pl.lit(year).alias("year"))


def validate_network_state_panel(network_df) -> tuple[ValidationResult, ...]:
    """Validate ABM v5 node-level production-network diagnostics."""
    import polars as pl

    frame = network_df if isinstance(network_df, pl.DataFrame) else pl.DataFrame(network_df)
    results: list[ValidationResult] = []
    missing_columns = sorted(column for column in NETWORK_REQUIRED_COLUMNS if column not in frame.columns)
    results.append(
        _validation_result(
            "network_required_columns",
            ValidationStatus.FAILED if missing_columns else ValidationStatus.PASSED,
            ValidationSeverity.CRITICAL if missing_columns else ValidationSeverity.INFO,
            f"Missing required network columns: {missing_columns}."
            if missing_columns
            else "All required network columns are present.",
            ValidationLayer.STRUCTURAL_VALIDITY,
            len(missing_columns),
            len(NETWORK_REQUIRED_COLUMNS),
        )
    )
    if missing_columns:
        return tuple(results)

    duplicate_count = int(
        frame.group_by(["country_sector", "year"]).len().filter(pl.col("len") > 1)["len"].sum()
        or 0
    )
    results.append(
        _validation_result(
            "network_unique_country_sector_year",
            ValidationStatus.FAILED if duplicate_count else ValidationStatus.PASSED,
            ValidationSeverity.CRITICAL if duplicate_count else ValidationSeverity.INFO,
            f"Found {duplicate_count} duplicate country_sector-year rows."
            if duplicate_count
            else "country_sector-year rows are unique.",
            ValidationLayer.STRUCTURAL_VALIDITY,
            duplicate_count,
            frame.height,
        )
    )

    negative_counts = frame.filter((pl.col("supplier_count") < 0) | (pl.col("buyer_count") < 0)).height
    results.append(
        _validation_result(
            "network_counts_nonnegative",
            ValidationStatus.FAILED if negative_counts else ValidationStatus.PASSED,
            ValidationSeverity.ERROR if negative_counts else ValidationSeverity.INFO,
            f"Found {negative_counts} rows with negative supplier or buyer counts."
            if negative_counts
            else "Supplier and buyer counts are nonnegative.",
            ValidationLayer.STRUCTURAL_VALIDITY,
            negative_counts,
            frame.height,
        )
    )

    bounded_columns = (
        "supplier_concentration_hhi",
        "buyer_concentration_hhi",
        "import_dependence_proxy",
        "export_dependence_proxy",
        "network_green_exposure",
        "incoming_network_green_exposure",
        "outgoing_network_green_exposure",
        "brown_centrality",
        "supplier_lock_in",
    )
    for column in bounded_columns:
        invalid_count = frame.filter(
            pl.col(column).is_not_null() & ((pl.col(column) < 0) | (pl.col(column) > 1))
        ).height
        results.append(
            _validation_result(
                f"network_{column}_bounds",
                ValidationStatus.FAILED if invalid_count else ValidationStatus.PASSED,
                ValidationSeverity.ERROR if invalid_count else ValidationSeverity.INFO,
                f"Found {invalid_count} {column} values outside [0, 1]."
                if invalid_count
                else f"{column} values are within [0, 1] where present.",
                ValidationLayer.MECHANISM_VALIDITY,
                invalid_count,
                frame.height,
            )
        )

    years = sorted(frame["year"].drop_nulls().unique().to_list())
    out_of_range_years = [year for year in years if year < HISTORICAL_START_YEAR or year > HISTORICAL_END_YEAR]
    results.append(
        _validation_result(
            "network_year_range",
            ValidationStatus.FAILED if out_of_range_years else ValidationStatus.PASSED,
            ValidationSeverity.ERROR if out_of_range_years else ValidationSeverity.INFO,
            f"Network panel has years outside 1995-2016: {out_of_range_years}."
            if out_of_range_years
            else "Network panel years are within 1995-2016.",
            ValidationLayer.STRUCTURAL_VALIDITY,
            len(out_of_range_years),
            len(years),
        )
    )
    return tuple(results)


def _validation_results_to_dict(results: tuple[ValidationResult, ...]) -> list[dict[str, Any]]:
    return [
        {
            "check_name": result.check_name,
            "status": result.status.value,
            "severity": result.severity.value,
            "message": result.message,
            "layer": result.layer.value,
            "n_failed": result.n_failed,
            "n_checked": result.n_checked,
        }
        for result in results
    ]


def _has_critical_failures(results: tuple[ValidationResult, ...]) -> bool:
    return any(
        result.status is ValidationStatus.FAILED and result.severity is ValidationSeverity.CRITICAL
        for result in results
    )


def build_network_state_panels(project_root: Path) -> NetworkBuildResult:
    """Build ABM v5 compact supplier candidates and network diagnostic panels."""
    import polars as pl

    paths = ABMV5Paths.from_project_root(project_root)
    paths.validate_project_root()
    paths.ensure_directories()

    identity_path = paths.inputs / "agent_identity.parquet"
    accounting_path = paths.accounting / ACCOUNTING_OUTPUT_FILENAME
    if not identity_path.exists():
        raise FileNotFoundError(f"Agent identity panel missing: {identity_path}. Run Phase 2.2 first.")
    if not accounting_path.exists():
        raise FileNotFoundError(
            f"Accounting state panel missing: {accounting_path}. Run Phase 2.3 first."
        )

    identity = load_agent_identity_panel(identity_path)
    accounting = pl.read_parquet(accounting_path)
    temp_candidate_dir = paths.tmp / "supplier_candidates_by_year"
    temp_candidate_dir.mkdir(parents=True, exist_ok=True)
    temp_candidate_paths: list[Path] = []
    candidate_results: list[ValidationResult] = []
    coverage_summaries: list[dict[str, float | int]] = []
    total_candidate_rows = 0
    network_frames: list[pl.DataFrame] = []
    for year in range(HISTORICAL_START_YEAR, HISTORICAL_END_YEAR + 1):
        year_accounting = accounting.filter(pl.col("year") == year)
        edges = build_production_edges_for_year(project_root, year)
        edges = add_supplier_weights_and_coefficients(edges, year_accounting)
        network = build_network_state_for_year(edges, year_accounting)
        candidates = build_supplier_candidates_for_year(edges, year_accounting)
        temp_candidate_path = temp_candidate_dir / f"supplier_candidate_panel_{year}.parquet"
        candidates.write_parquet(temp_candidate_path)
        temp_candidate_paths.append(temp_candidate_path)
        total_candidate_rows += candidates.height
        candidate_results.extend(validate_supplier_candidate_panel(candidates))
        coverage_summaries.append(summarize_supplier_candidate_coverage(edges, candidates))
        network_frames.append(network)

    network_panel = pl.concat(network_frames, how="vertical")

    backbone = accounting.select("country_sector", "year").unique(subset=["country_sector", "year"])
    network_panel = backbone.join(network_panel, on=["country_sector", "year"], how="left")
    network_results = validate_network_state_panel(network_panel)
    validation_results = (*tuple(candidate_results), *network_results)

    candidate_output_path = paths.supplier_network / SUPPLIER_CANDIDATE_OUTPUT_FILENAME
    network_output_path = paths.supplier_network / NETWORK_OUTPUT_FILENAME
    validation_path = paths.validation / NETWORK_VALIDATION_FILENAME
    coverage_summary_path = paths.validation / SUPPLIER_CANDIDATE_COVERAGE_SUMMARY_FILENAME
    if temp_candidate_paths:
        pl.scan_parquet([str(path) for path in temp_candidate_paths]).sink_parquet(candidate_output_path)
    network_panel.write_parquet(network_output_path)
    validation_payload = {
        "validation_scope": "abm_v5_network_state_panels",
        "supplier_candidate_results": _validation_results_to_dict(tuple(candidate_results)),
        "network_results": _validation_results_to_dict(network_results),
        "conceptual_boundary": (
            "Raw T is used internally for yearly production links and network diagnostics. "
            "The canonical supplier-network output is a compact candidate panel. Candidates "
            "are opportunity structure, not supplier choice or rewiring; ET/Leontief embodied "
            "carbon is not used for supplier edges."
        ),
    }
    validation_path.write_text(json.dumps(validation_payload, indent=2, sort_keys=True), encoding="utf-8")
    coverage_summary_path.write_text(
        json.dumps(
            {
                "validation_scope": "abm_v5_supplier_candidate_coverage",
                "yearly_coverage": coverage_summaries,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if _has_critical_failures(validation_results):
        raise ValueError(f"Network state panel validation has critical failures: {validation_path}")

    result = NetworkBuildResult(
        candidate_output_path=candidate_output_path,
        network_output_path=network_output_path,
        validation_path=validation_path,
        coverage_summary_path=coverage_summary_path,
        start_year=HISTORICAL_START_YEAR,
        end_year=HISTORICAL_END_YEAR,
        n_candidate_rows=total_candidate_rows,
        n_network_rows=network_panel.height,
        n_agents=identity.height,
        edge_output_path=candidate_output_path,
    )
    result.validate()
    return result


def load_edge_state_panel(path: Path):
    """Load a legacy/internal ABM v5 edge-like panel from parquet."""
    import polars as pl

    return pl.read_parquet(path)


def load_supplier_candidate_panel(path: Path):
    """Load the canonical ABM v5 compact supplier candidate panel."""
    import polars as pl

    return pl.read_parquet(path)


def load_network_state_panel(path: Path):
    """Load an ABM v5 node-level network state panel."""
    import polars as pl

    return pl.read_parquet(path)


def summarize_network_state(network_df) -> dict[str, float]:
    """Return compact means for network diagnostics."""
    import polars as pl

    frame = network_df if isinstance(network_df, pl.DataFrame) else pl.DataFrame(network_df)
    summary: dict[str, float] = {}
    for column in SUMMARY_COLUMNS:
        key = f"mean_{column}"
        if column not in frame.columns:
            summary[key] = float("nan")
            continue
        value = frame.select(pl.col(column).cast(pl.Float64, strict=False).mean()).item()
        summary[key] = float(value) if value is not None else float("nan")
    return summary
