from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.schemas import (
    STATE_DERIVED_COLUMNS,
    STATE_OPTIONAL_COLUMNS,
    STATE_REQUIRED_COLUMNS,
)


@dataclass(frozen=True)
class StateSourceDiagnostic:
    """Diagnostic for ABM v4 state-source discovery."""

    selected_source: Path | None
    checked_sources: tuple[Path, ...]
    message: str

    @property
    def has_source(self) -> bool:
        return self.selected_source is not None


@dataclass(frozen=True)
class ColumnMapping:
    """How a source column maps into an ABM v4 canonical column."""

    canonical_name: str
    source_column: str | None
    mapping_status: str
    notes: str


@dataclass(frozen=True)
class StateBuildResult:
    """Inspectable result of ABM v4 state-panel construction."""

    state_panel: pl.DataFrame
    selected_source: Path
    checked_sources: tuple[Path, ...]
    column_mappings: tuple[ColumnMapping, ...]
    source_report: pl.DataFrame
    missingness_report: pl.DataFrame
    summary_by_year: pl.DataFrame
    output_path: Path | None


@dataclass(frozen=True)
class CapabilityJoinResult:
    """Inspectable result of Atlas capability enrichment."""

    state_panel: pl.DataFrame
    source_file: Path
    selected_join_keys: tuple[tuple[str, str], ...]
    join_report: pl.DataFrame
    coverage_by_year: pl.DataFrame
    coverage_by_sector: pl.DataFrame
    output_path: Path | None


TEXT_COLUMNS = {
    "country_sector",
    "Country",
    "Country_detail",
    "Category",
    "Sector",
    "ecosystem_id",
    "ecosystem_label",
    "ecosystem_source",
    "general_capability_source",
    "green_capability_source",
}

GENERAL_CAPABILITY_CANDIDATES = (
    "capability_export_weighted_pci",
    "capability_mean_pci",
    "active_good_count",
    "diversity",
)

GREEN_CAPABILITY_CANDIDATES = (
    "green_capability_export_share",
    "green_capability_share",
    "green_active_good_count",
    "green_product_share",
)

CANONICAL_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "country_sector": ("country_sector", "country_sector_id", "node_id"),
    "Year": ("Year", "year"),
    "Country": ("Country", "country"),
    "Country_detail": ("Country_detail", "country_detail", "countryName"),
    "Category": ("Category", "category"),
    "Sector": ("Sector", "sector"),
    "X_observed": (
        "X_observed",
        "X_corrected",
        "X",
        "output",
        "production",
        "gross_output",
        "X_raw_current_convention",
    ),
    "EI": ("EI", "emissions_intensity", "ei"),
    "emissions_observed": ("emissions_observed", "emissions", "E_observed"),
    "g_local": ("g_local", "g_base"),
    "g_in_network": ("g_in_network", "g_in", "g_network"),
    "g_out_network": ("g_out_network", "g_out"),
    "pagerank": ("pagerank",),
    "centrality": ("centrality", "pagerank", "eigenvector_centrality"),
    "active_good_count": ("active_good_count",),
    "capability_mean_pci": ("capability_mean_pci",),
    "capability_export_weighted_pci": ("capability_export_weighted_pci",),
    "general_capability": ("general_capability",),
    "green_capability": ("green_capability",),
    "green_capability_export_share": ("green_capability_export_share",),
    "green_capability_share": ("green_capability_share",),
}


def discover_state_source(
    paths: ABMV4Paths,
    start_year: int,
    end_year: int,
) -> StateSourceDiagnostic:
    """Find the first available state source using the ABM v4 priority order."""
    checked_sources = paths.state_source_candidates(start_year, end_year)
    for source_path in checked_sources:
        if source_path.exists():
            return StateSourceDiagnostic(
                selected_source=source_path,
                checked_sources=checked_sources,
                message=f"Selected state source: {source_path}",
            )

    return StateSourceDiagnostic(
        selected_source=None,
        checked_sources=checked_sources,
        message="No valid ABM v4 state source was found.",
    )


def build_state_panel(
    paths: ABMV4Paths,
    start_year: int,
    end_year: int,
    *,
    write_outputs: bool = False,
    epsilon: float = 1e-9,
) -> StateBuildResult:
    """Build the ABM v4 state panel from the highest-priority available source."""
    state_source = discover_state_source(paths, start_year, end_year)
    if state_source.selected_source is None:
        checked = "; ".join(str(source_path) for source_path in state_source.checked_sources)
        raise FileNotFoundError(f"No valid ABM v4 state source found. Checked: {checked}")

    source_panel = pl.read_parquet(state_source.selected_source)
    state_panel, column_mappings = canonicalize_state_panel(
        source_panel,
        start_year=start_year,
        end_year=end_year,
        epsilon=epsilon,
    )

    source_report = build_state_source_report(
        state_panel=state_panel,
        source_panel=source_panel,
        selected_source=state_source.selected_source,
        checked_sources=state_source.checked_sources,
    )
    missingness_report = build_state_missingness_report(state_panel)
    summary_by_year = build_state_summary_by_year(state_panel)

    output_path: Path | None = None
    if write_outputs:
        output_path = paths.state_panel_path(start_year, end_year)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        paths.diagnostics.mkdir(parents=True, exist_ok=True)
        state_panel.write_parquet(output_path)
        source_report.write_csv(paths.diagnostics / "state_source_report.csv")
        missingness_report.write_csv(paths.diagnostics / "state_missingness_report.csv")
        summary_by_year.write_csv(paths.diagnostics / "state_summary_by_year.csv")
        column_mapping_frame(column_mappings).write_csv(
            paths.diagnostics / "state_column_mapping.csv"
        )

    return StateBuildResult(
        state_panel=state_panel,
        selected_source=state_source.selected_source,
        checked_sources=state_source.checked_sources,
        column_mappings=column_mappings,
        source_report=source_report,
        missingness_report=missingness_report,
        summary_by_year=summary_by_year,
        output_path=output_path,
    )


def canonicalize_state_panel(
    source_panel: pl.DataFrame,
    start_year: int,
    end_year: int,
    epsilon: float = 1e-9,
) -> tuple[pl.DataFrame, tuple[ColumnMapping, ...]]:
    """Rename source columns, filter years, and add derived ABM v4 variables."""
    source_columns = set(source_panel.columns)
    column_mappings: list[ColumnMapping] = []

    select_expressions: list[pl.Expr] = []
    base_columns = (
        *STATE_REQUIRED_COLUMNS,
        "Country",
        "Country_detail",
        "Category",
        "Sector",
        "emissions_observed",
        "g_local",
        "g_in_network",
        "g_out_network",
        "pagerank",
        "centrality",
        "active_good_count",
        "capability_mean_pci",
        "capability_export_weighted_pci",
        "general_capability",
        "green_capability",
        "green_capability_export_share",
        "green_capability_share",
    )

    for canonical_name in base_columns:
        source_column = _first_present_column(
            source_columns,
            CANONICAL_COLUMN_CANDIDATES.get(canonical_name, (canonical_name,)),
        )
        column_mappings.append(
            _build_column_mapping(canonical_name, source_column, source_columns)
        )
        select_expressions.append(_canonical_expression(canonical_name, source_column))

    state_panel = source_panel.select(select_expressions)
    state_panel = state_panel.with_columns(
        pl.col("Year").cast(pl.Int64, strict=False),
        pl.col("X_observed").cast(pl.Float64, strict=False),
        pl.col("EI").cast(pl.Float64, strict=False),
    ).filter(pl.col("Year").is_between(start_year, end_year))

    state_panel = add_derived_state_variables(state_panel, epsilon)
    derived_mappings = build_derived_column_mappings(state_panel)
    column_mappings.extend(derived_mappings)

    output_columns = (
        "country_sector",
        "Year",
        "Country",
        "Country_detail",
        "Category",
        "Sector",
        "X_observed",
        "EI",
        "emissions_observed",
        "g_local",
        "g_in_network",
        "g_out_network",
        "pagerank",
        "centrality",
        "brown_centrality",
        "general_capability",
        "green_capability",
        "active_good_count",
        "capability_mean_pci",
        "capability_export_weighted_pci",
        "green_capability_export_share",
        "green_capability_share",
        "network_green_exposure",
        "log_x_observed",
        "log_EI",
        "g_local_v4",
        "ecosystem_id",
        "ecosystem_label",
        "ecosystem_source",
    )
    return state_panel.select(output_columns), tuple(column_mappings)


def add_derived_state_variables(state_panel: pl.DataFrame, epsilon: float) -> pl.DataFrame:
    """Add ABM v4 derived variables without overwriting invalid source values."""
    panel = state_panel.with_columns(
        pl.col("X_observed").log1p().alias("log_x_observed"),
        pl.when(pl.col("EI") > 0)
        .then(pl.col("EI").log())
        .otherwise(None)
        .alias("log_EI"),
        pl.when(pl.col("EI") + epsilon > 0)
        .then(-(pl.col("EI") + epsilon).log())
        .otherwise(None)
        .alias("__g_local_v4_raw"),
        pl.coalesce(
            [
                pl.col("g_in_network"),
                pl.col("g_out_network"),
                pl.col("g_local"),
            ]
        ).alias("network_green_exposure"),
        pl.coalesce(
            [
                pl.col("general_capability"),
                pl.col("capability_export_weighted_pci"),
                pl.col("capability_mean_pci"),
                pl.col("active_good_count"),
            ]
        ).alias("general_capability"),
        pl.coalesce(
            [
                pl.col("green_capability"),
                pl.col("green_capability_export_share"),
                pl.col("green_capability_share"),
            ]
        ).alias("green_capability"),
        pl.lit("unknown").alias("ecosystem_id"),
        pl.lit("Unknown ecosystem").alias("ecosystem_label"),
        pl.lit("fallback_unknown").alias("ecosystem_source"),
    )

    panel = panel.with_columns(
        pl.col("__g_local_v4_raw").min().over("Year").alias("__g_local_v4_min"),
        pl.col("__g_local_v4_raw").max().over("Year").alias("__g_local_v4_max"),
        _percentile_rank("centrality").alias("__centrality_rank"),
        _percentile_rank("EI").alias("__ei_rank"),
    )
    panel = panel.with_columns(
        pl.when(pl.col("__g_local_v4_max") > pl.col("__g_local_v4_min"))
        .then(
            (pl.col("__g_local_v4_raw") - pl.col("__g_local_v4_min"))
            / (pl.col("__g_local_v4_max") - pl.col("__g_local_v4_min"))
        )
        .otherwise(None)
        .alias("g_local_v4"),
        (pl.col("__centrality_rank") * pl.col("__ei_rank")).alias("brown_centrality"),
    )

    return panel.drop(
        [
            "__g_local_v4_raw",
            "__g_local_v4_min",
            "__g_local_v4_max",
            "__centrality_rank",
            "__ei_rank",
        ]
    )


def build_state_source_report(
    state_panel: pl.DataFrame,
    source_panel: pl.DataFrame,
    selected_source: Path,
    checked_sources: tuple[Path, ...],
) -> pl.DataFrame:
    """Build a one-row report describing the selected source panel."""
    notes = []
    if state_panel["EI"].null_count() > 0:
        notes.append("EI has missing values.")
    if (state_panel["EI"] <= 0).sum() > 0:
        notes.append("EI has non-positive values; they were preserved.")
    if state_panel["network_green_exposure"].null_count() > 0:
        notes.append("Network green exposure is partially missing.")
    if not notes:
        notes.append("State source loaded and canonicalized.")

    return pl.DataFrame(
        {
            "selected_source": [str(selected_source)],
            "candidate_sources_checked": [
                "; ".join(str(source_path) for source_path in checked_sources)
            ],
            "row_count": [state_panel.height],
            "column_count": [source_panel.width],
            "year_min": [state_panel["Year"].min()],
            "year_max": [state_panel["Year"].max()],
            "country_sector_count": [state_panel["country_sector"].n_unique()],
            "notes": [" ".join(notes)],
        }
    )


def build_state_missingness_report(state_panel: pl.DataFrame) -> pl.DataFrame:
    """Report missingness for required, optional, and derived state columns."""
    report_columns = tuple(
        dict.fromkeys(
            (
                *STATE_REQUIRED_COLUMNS,
                *STATE_OPTIONAL_COLUMNS,
                *STATE_DERIVED_COLUMNS,
            )
        )
    )
    rows: list[dict[str, object]] = []
    for column_name in report_columns:
        if column_name not in state_panel.columns:
            missing_count = state_panel.height
        else:
            missing_count = state_panel[column_name].null_count()
        missing_share = missing_count / state_panel.height if state_panel.height else 0.0
        status = _column_status(column_name)
        rows.append(
            {
                "column": column_name,
                "missing_count": missing_count,
                "missing_share": missing_share,
                "status": status,
                "consequence": _missingness_consequence(column_name, status, missing_count),
            }
        )

    return pl.DataFrame(rows)


def build_state_summary_by_year(state_panel: pl.DataFrame) -> pl.DataFrame:
    """Summarize state-panel coverage and core values by year."""
    return (
        state_panel.group_by("Year")
        .agg(
            pl.len().alias("row_count"),
            pl.col("country_sector").n_unique().alias("country_sector_count"),
            pl.col("X_observed").sum().alias("x_observed_total"),
            pl.col("EI").mean().alias("ei_mean"),
            pl.col("EI").null_count().alias("ei_missing_count"),
            (pl.col("EI") <= 0).sum().alias("ei_nonpositive_count"),
            pl.col("network_green_exposure").null_count().alias(
                "network_green_exposure_missing_count"
            ),
            pl.col("g_local_v4").null_count().alias("g_local_v4_missing_count"),
        )
        .sort("Year")
    )


def capability_source_candidates(paths: ABMV4Paths) -> tuple[Path, ...]:
    """Return capability source candidates in explicit priority order."""
    return (
        paths.data_atlas / "processed" / "atlas_eora26_sector_capabilities_1995_2016.parquet",
        paths.data_final / "eora_atlas_dynamic_panel.parquet",
        paths.data_final / "eora_atlas_merged.parquet",
    )


def load_atlas_capability_panel(paths: ABMV4Paths) -> tuple[Path, pl.DataFrame]:
    """Load the highest-priority available Atlas capability source."""
    checked = capability_source_candidates(paths)
    for source_path in checked:
        if source_path.exists():
            return source_path, pl.read_parquet(source_path)
    checked_text = "; ".join(str(path) for path in checked)
    raise FileNotFoundError(f"No Atlas capability source found. Checked: {checked_text}")


def inspect_capability_columns(
    state_panel: pl.DataFrame,
    capability_panel: pl.DataFrame,
) -> pl.DataFrame:
    """Build a diagnostic inventory of state and capability key/value columns."""
    key_candidates = {
        "country": ("Country", "country", "iso3Code", "Country_detail"),
        "year": ("Year", "year"),
        "sector": ("Sector", "sector", "eora26_sector"),
        "country_sector": ("country_sector",),
    }
    capability_candidates = (*GENERAL_CAPABILITY_CANDIDATES, *GREEN_CAPABILITY_CANDIDATES)
    rows: list[dict[str, object]] = []
    for group, candidates in key_candidates.items():
        rows.append(
            {
                "column_group": group,
                "state_columns_available": ", ".join(
                    column for column in candidates if column in state_panel.columns
                ),
                "capability_columns_available": ", ".join(
                    column for column in candidates if column in capability_panel.columns
                ),
            }
        )
    rows.append(
        {
            "column_group": "capability_variables",
            "state_columns_available": ", ".join(
                column for column in capability_candidates if column in state_panel.columns
            ),
            "capability_columns_available": ", ".join(
                column for column in capability_candidates if column in capability_panel.columns
            ),
        }
    )
    return pl.DataFrame(rows)


def select_capability_join_keys(
    state_panel: pl.DataFrame,
    capability_panel: pl.DataFrame,
) -> tuple[tuple[str, str], ...]:
    """Select explicit state-to-capability join keys from available columns."""
    country_key = _first_join_key(
        state_panel.columns,
        capability_panel.columns,
        (("Country", "iso3Code"), ("Country", "Country"), ("Country_detail", "iso3Code")),
    )
    year_key = _first_join_key(
        state_panel.columns,
        capability_panel.columns,
        (("Year", "year"), ("Year", "Year")),
    )
    sector_key = _first_join_key(
        state_panel.columns,
        capability_panel.columns,
        (("Sector", "eora26_sector"), ("Sector", "Sector"), ("Sector", "sector")),
    )
    selected = tuple(key for key in (country_key, year_key, sector_key) if key is not None)
    if len(selected) < 3:
        raise ValueError(
            "Could not select a complete capability join key. "
            f"State columns: {state_panel.columns}; capability columns: {capability_panel.columns}"
        )
    return selected


def join_capability_variables(
    state_panel: pl.DataFrame,
    capability_panel: pl.DataFrame,
    selected_join_keys: tuple[tuple[str, str], ...],
) -> pl.DataFrame:
    """Fill missing canonical capabilities from Atlas variables without overwriting existing values."""
    state_keys = [state_key for state_key, _ in selected_join_keys]
    capability_keys = [capability_key for _, capability_key in selected_join_keys]
    selected_capability_columns = [
        column
        for column in (*GENERAL_CAPABILITY_CANDIDATES, *GREEN_CAPABILITY_CANDIDATES)
        if column in capability_panel.columns
    ]
    if not selected_capability_columns:
        return state_panel.with_columns(
            pl.lit(False).alias("capability_join_matched"),
            _source_column("general_capability", "existing_state").alias("general_capability_source"),
            _source_column("green_capability", "existing_state").alias("green_capability_source"),
        )

    capability_prepared = (
        capability_panel.select([*capability_keys, *selected_capability_columns])
        .unique(subset=capability_keys, keep="first")
        .rename(
            {
                capability_key: state_key
                for state_key, capability_key in selected_join_keys
                if capability_key != state_key
            }
        )
    )
    rename_values = {
        column: f"atlas_{column}"
        for column in selected_capability_columns
        if column in state_panel.columns or column in selected_capability_columns
    }
    capability_prepared = capability_prepared.rename(rename_values).with_columns(
        pl.lit(True).alias("capability_join_matched")
    )
    joined = state_panel.join(capability_prepared, on=state_keys, how="left").with_columns(
        pl.col("capability_join_matched").fill_null(False)
    )

    general_atlas_columns = [
        f"atlas_{column}"
        for column in GENERAL_CAPABILITY_CANDIDATES
        if f"atlas_{column}" in joined.columns
    ]
    green_atlas_columns = [
        f"atlas_{column}"
        for column in GREEN_CAPABILITY_CANDIDATES
        if f"atlas_{column}" in joined.columns
    ]
    expressions: list[pl.Expr] = []
    if general_atlas_columns:
        expressions.append(pl.coalesce(general_atlas_columns).alias("__atlas_general_capability"))
    else:
        expressions.append(pl.lit(None, dtype=pl.Float64).alias("__atlas_general_capability"))
    if green_atlas_columns:
        expressions.append(pl.coalesce(green_atlas_columns).alias("__atlas_green_capability"))
    else:
        expressions.append(pl.lit(None, dtype=pl.Float64).alias("__atlas_green_capability"))

    joined = joined.with_columns(expressions)
    if "general_capability_source" not in joined.columns:
        joined = joined.with_columns(
            _source_column("general_capability", "existing_state").alias("general_capability_source")
        )
    if "green_capability_source" not in joined.columns:
        joined = joined.with_columns(
            _source_column("green_capability", "existing_state").alias("green_capability_source")
        )

    return (
        joined.with_columns(
            pl.when(pl.col("general_capability").is_null())
            .then(pl.col("__atlas_general_capability"))
            .otherwise(pl.col("general_capability"))
            .alias("general_capability"),
            pl.when(pl.col("green_capability").is_null())
            .then(pl.col("__atlas_green_capability"))
            .otherwise(pl.col("green_capability"))
            .alias("green_capability"),
            pl.when(
                pl.col("general_capability").is_null()
                & pl.col("__atlas_general_capability").is_not_null()
            )
            .then(pl.lit("atlas_capability_join"))
            .otherwise(pl.col("general_capability_source"))
            .alias("general_capability_source"),
            pl.when(
                pl.col("green_capability").is_null()
                & pl.col("__atlas_green_capability").is_not_null()
            )
            .then(pl.lit("atlas_capability_join"))
            .otherwise(pl.col("green_capability_source"))
            .alias("green_capability_source"),
        )
        .drop(["__atlas_general_capability", "__atlas_green_capability"])
    )


def build_capability_join_report(
    state_before: pl.DataFrame,
    state_after: pl.DataFrame,
    source_file: Path,
    selected_join_keys: tuple[tuple[str, str], ...],
    column_inspection: pl.DataFrame,
) -> pl.DataFrame:
    """Build a one-row capability join diagnostic report."""
    matched_rows = (
        int(state_after["capability_join_matched"].sum())
        if "capability_join_matched" in state_after.columns
        else 0
    )
    before_general = state_before["general_capability"].is_not_null().sum()
    after_general = state_after["general_capability"].is_not_null().sum()
    before_green = state_before["green_capability"].is_not_null().sum()
    after_green = state_after["green_capability"].is_not_null().sum()
    selected_keys = "; ".join(f"{state}={capability}" for state, capability in selected_join_keys)
    notes = [
        "Existing non-missing canonical capability values were preserved.",
        "Atlas values were used only to fill missing canonical capability values.",
    ]
    if after_general <= before_general and after_green <= before_green:
        notes.append(
            "Canonical capability coverage did not improve; likely causes include source missingness or already-filled matched rows."
        )
    notes.append(
        "Column inspection: "
        + " | ".join(
            f"{row['column_group']}: state[{row['state_columns_available']}], capability[{row['capability_columns_available']}]"
            for row in column_inspection.to_dicts()
        )
    )
    return pl.DataFrame(
        {
            "source_file": [str(source_file)],
            "selected_join_keys": [selected_keys],
            "state_rows_before": [state_before.height],
            "state_rows_after": [state_after.height],
            "matched_rows": [matched_rows],
            "unmatched_rows": [state_after.height - matched_rows],
            "matched_share": [matched_rows / state_after.height if state_after.height else 0.0],
            "general_capability_nonmissing_before": [before_general],
            "general_capability_nonmissing_after": [after_general],
            "green_capability_nonmissing_before": [before_green],
            "green_capability_nonmissing_after": [after_green],
            "general_capability_fill_share_before": [
                1 - before_general / state_before.height if state_before.height else 0.0
            ],
            "general_capability_fill_share_after": [
                1 - after_general / state_after.height if state_after.height else 0.0
            ],
            "green_capability_fill_share_before": [
                1 - before_green / state_before.height if state_before.height else 0.0
            ],
            "green_capability_fill_share_after": [
                1 - after_green / state_after.height if state_after.height else 0.0
            ],
            "notes": [" ".join(notes)],
        }
    )


def build_capability_coverage_by_year(state_panel: pl.DataFrame) -> pl.DataFrame:
    """Summarize canonical capability coverage by year."""
    return (
        state_panel.group_by("Year")
        .agg(
            pl.len().alias("rows"),
            pl.col("general_capability").is_not_null().sum().alias("general_capability_nonmissing"),
            pl.col("green_capability").is_not_null().sum().alias("green_capability_nonmissing"),
        )
        .with_columns(
            (1 - pl.col("general_capability_nonmissing") / pl.col("rows")).alias(
                "general_capability_missing_share"
            ),
            (1 - pl.col("green_capability_nonmissing") / pl.col("rows")).alias(
                "green_capability_missing_share"
            ),
        )
        .rename({"Year": "year"})
        .sort("year")
    )


def build_capability_coverage_by_sector(state_panel: pl.DataFrame) -> pl.DataFrame:
    """Summarize canonical capability coverage by sector."""
    return (
        state_panel.group_by("Sector")
        .agg(
            pl.len().alias("rows"),
            pl.col("general_capability").is_not_null().sum().alias("__general_nonmissing"),
            pl.col("green_capability").is_not_null().sum().alias("__green_nonmissing"),
        )
        .with_columns(
            (1 - pl.col("__general_nonmissing") / pl.col("rows")).alias(
                "general_capability_missing_share"
            ),
            (1 - pl.col("__green_nonmissing") / pl.col("rows")).alias(
                "green_capability_missing_share"
            ),
        )
        .drop(["__general_nonmissing", "__green_nonmissing"])
        .sort("Sector")
    )


def repair_capability_coverage(
    paths: ABMV4Paths,
    start_year: int,
    end_year: int,
    *,
    write_outputs: bool = False,
) -> CapabilityJoinResult:
    """Repair ABM v4 state capability coverage using explicit Atlas joins."""
    state_path = paths.state_panel_path(start_year, end_year)
    if not state_path.exists():
        raise FileNotFoundError(f"ABM v4 state panel not found: {state_path}")
    state_panel = pl.read_parquet(state_path)
    source_file, capability_panel = load_atlas_capability_panel(paths)
    column_inspection = inspect_capability_columns(state_panel, capability_panel)
    selected_join_keys = select_capability_join_keys(state_panel, capability_panel)
    state_after = join_capability_variables(
        state_panel=state_panel,
        capability_panel=capability_panel,
        selected_join_keys=selected_join_keys,
    )
    join_report = build_capability_join_report(
        state_before=state_panel,
        state_after=state_after,
        source_file=source_file,
        selected_join_keys=selected_join_keys,
        column_inspection=column_inspection,
    )
    coverage_by_year = build_capability_coverage_by_year(state_after)
    coverage_by_sector = build_capability_coverage_by_sector(state_after)

    output_path: Path | None = None
    if write_outputs:
        paths.inputs.mkdir(parents=True, exist_ok=True)
        paths.diagnostics.mkdir(parents=True, exist_ok=True)
        output_path = state_path
        state_after.write_parquet(output_path)
        join_report.write_csv(paths.capability_join_report_path)
        coverage_by_year.write_csv(paths.capability_coverage_by_year_path)
        coverage_by_sector.write_csv(paths.capability_coverage_by_sector_path)

    return CapabilityJoinResult(
        state_panel=state_after,
        source_file=source_file,
        selected_join_keys=selected_join_keys,
        join_report=join_report,
        coverage_by_year=coverage_by_year,
        coverage_by_sector=coverage_by_sector,
        output_path=output_path,
    )


def column_mapping_frame(column_mappings: tuple[ColumnMapping, ...]) -> pl.DataFrame:
    """Convert column mappings to a diagnostics dataframe."""
    return pl.DataFrame(
        [
            {
                "canonical_name": mapping.canonical_name,
                "source_column": mapping.source_column,
                "mapping_status": mapping.mapping_status,
                "notes": mapping.notes,
            }
            for mapping in column_mappings
        ]
    )


def build_derived_column_mappings(state_panel: pl.DataFrame) -> tuple[ColumnMapping, ...]:
    """Describe derived and fallback columns added by ABM v4."""
    derived_specs = (
        ("log_x_observed", "derived", "Computed as log1p(X_observed)."),
        ("log_EI", "derived", "Computed as log(EI) only where EI > 0."),
        ("g_local_v4", "derived", "Computed as within-year rescaled -log(EI + epsilon)."),
        (
            "network_green_exposure",
            "derived",
            "Uses g_in_network, then g_out_network, then g_local where available.",
        ),
        (
            "brown_centrality",
            "derived",
            "Within-year percentile rank of centrality times percentile rank of EI.",
        ),
        (
            "general_capability",
            "derived",
            "Uses capability_export_weighted_pci, capability_mean_pci, then active_good_count.",
        ),
        (
            "green_capability",
            "derived",
            "Uses green_capability_export_share, then green_capability_share.",
        ),
        ("ecosystem_id", "fallback", "Set to unknown until ecosystem assignment is implemented."),
        (
            "ecosystem_label",
            "fallback",
            "Set to Unknown ecosystem until ecosystem assignment is implemented.",
        ),
        (
            "ecosystem_source",
            "fallback",
            "Set to fallback_unknown until ecosystem assignment is implemented.",
        ),
    )
    mappings: list[ColumnMapping] = []
    for canonical_name, mapping_status, notes in derived_specs:
        if canonical_name in state_panel.columns and state_panel[canonical_name].null_count() == state_panel.height:
            notes = f"{notes} Column is fully missing after derivation."
        mappings.append(
            ColumnMapping(
                canonical_name=canonical_name,
                source_column=None,
                mapping_status=mapping_status,
                notes=notes,
            )
        )
    return tuple(mappings)


def _first_present_column(
    source_columns: set[str],
    candidate_columns: tuple[str, ...],
) -> str | None:
    for candidate_column in candidate_columns:
        if candidate_column in source_columns:
            return candidate_column
    return None


def _first_join_key(
    state_columns: list[str],
    capability_columns: list[str],
    candidates: tuple[tuple[str, str], ...],
) -> tuple[str, str] | None:
    state_set = set(state_columns)
    capability_set = set(capability_columns)
    for state_column, capability_column in candidates:
        if state_column in state_set and capability_column in capability_set:
            return state_column, capability_column
    return None


def _source_column(capability_column: str, source_label: str) -> pl.Expr:
    return (
        pl.when(pl.col(capability_column).is_not_null())
        .then(pl.lit(source_label))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
    )


def _canonical_expression(canonical_name: str, source_column: str | None) -> pl.Expr:
    if source_column is not None:
        return pl.col(source_column).alias(canonical_name)
    if canonical_name in TEXT_COLUMNS:
        return pl.lit(None, dtype=pl.Utf8).alias(canonical_name)
    return pl.lit(None, dtype=pl.Float64).alias(canonical_name)


def _build_column_mapping(
    canonical_name: str,
    source_column: str | None,
    source_columns: set[str],
) -> ColumnMapping:
    if source_column is None:
        return ColumnMapping(
            canonical_name=canonical_name,
            source_column=None,
            mapping_status="missing",
            notes="No source column found; output column is missing unless derived later.",
        )
    if source_column == canonical_name:
        status = "direct"
        notes = "Source column already uses the canonical name."
    else:
        status = "renamed"
        notes = f"Mapped from source column {source_column}."
    if canonical_name == "centrality" and "centrality" not in source_columns and source_column == "pagerank":
        notes = "No centrality column found; using pagerank for centrality-dependent diagnostics."
    return ColumnMapping(
        canonical_name=canonical_name,
        source_column=source_column,
        mapping_status=status,
        notes=notes,
    )


def _percentile_rank(column_name: str) -> pl.Expr:
    return (
        pl.when(pl.col(column_name).is_not_null())
        .then(pl.col(column_name).rank("average").over("Year") / pl.col(column_name).count().over("Year"))
        .otherwise(None)
    )


def _column_status(column_name: str) -> str:
    if column_name in STATE_REQUIRED_COLUMNS:
        return "required"
    if column_name in STATE_DERIVED_COLUMNS:
        return "derived"
    return "optional"


def _missingness_consequence(
    column_name: str,
    status: str,
    missing_count: int,
) -> str:
    if missing_count == 0:
        return "No missing values detected."
    if status == "required":
        return "Required state construction input is missing for some rows."
    if column_name == "log_EI":
        return "EI is missing or non-positive for these rows; log_EI is intentionally missing."
    if column_name == "g_local_v4":
        return "EI is invalid or has no within-year variation for these rows."
    if column_name == "network_green_exposure":
        return "Network green exposure cannot be used for these rows."
    if column_name == "brown_centrality":
        return "Brown centrality cannot be used for these rows."
    if column_name in {"general_capability", "green_capability"}:
        return "Capability dynamics cannot use this variable for these rows."
    return "Optional source information is unavailable for these rows."
