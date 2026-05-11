from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.abm_v3.data_inventory import build_semantic_variable_map_rows


IDENTIFIER_COLUMNS = ["country_sector", "Year", "Country", "Country_detail", "Sector", "Category"]
BASE_CANONICAL_COLUMNS = [
    "X_observed",
    "log_X_observed",
    "final_demand_total",
    "Y_final_demand",
    "emissions_observed",
    "log_emissions_observed",
    "EI",
    "log_EI",
    "g_local",
    "green_capability_export_share",
    "green_capability_share",
    "green_capability_readiness",
    "capability_export_weighted_pci",
    "capability_mean_pci",
    "capability_ecosystem_exposure",
    "general_complexity",
    "network_green_exposure",
    "g_in_network",
    "g_out_network",
    "recursive_green",
    "pagerank",
    "eigenvector_centrality",
    "reverse_eigenvector_centrality",
    "brown_centrality",
    "K",
    "capacity_to_observed_ratio",
    "capacity_stress",
]
NEXT_COLUMNS = [
    "EI",
    "log_EI",
    "g_local",
    "green_capability_export_share",
    "green_capability_readiness",
    "network_green_exposure",
]
RANK_COLUMNS = {
    "EI": "EI_rank_pct",
    "green_capability_export_share": "green_capability_rank_pct",
    "network_green_exposure": "network_green_rank_pct",
    "brown_centrality": "brown_centrality_rank_pct",
    "X_observed": "X_observed_rank_pct",
    "emissions_observed": "emissions_observed_rank_pct",
}


@dataclass
class ABMV3PhaseSpaceStatePanelBuilder:
    """Build an auditable country-sector-year phase-space state panel."""

    base_panel: Path | str = Path(
        "data/abm_v3/inputs/abm_v3_historical_panel_1995_2016_transpose_row_fd_without_inventory.parquet"
    )
    output_dir: Path | str = Path("data/abm_v3/phase_space")
    include_ei_transition: bool = True
    include_scenario_overlays: bool = False
    strict: bool = False
    top_n: int = 25
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    column_sources: dict[str, str] = field(default_factory=dict)
    derived_columns: dict[str, str] = field(default_factory=dict)

    def build(self, start_year: int = 1995, end_year: int = 2016) -> dict[str, Path]:
        """Build the state panel and write panel, diagnostics, dictionary, and summary outputs."""
        self.diagnostics = []
        self.column_sources = {}
        self.derived_columns = {}

        base_path = Path(self.base_panel)
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        panel = self._read_table(base_path)
        self._record("row_count", "", "base_rows_before_filter", len(panel), str(base_path))
        panel = self._prepare_base_panel(panel, start_year, end_year, str(base_path))
        panel = self._derive_core_variables(panel)
        panel = self._derive_next_year_variables(panel)
        panel = self._derive_rank_percentiles(panel)
        panel = self._derive_selection_flags(panel)
        panel = self._derive_weights(panel)
        panel = self._order_columns(panel)

        missing_columns = [column for column in self.expected_output_columns() if column not in panel.columns]
        for column in missing_columns:
            self._record("missing_expected_variable", column, "Expected canonical variable is unavailable.", "", "")
        self._record_availability_flags(panel)

        panel_path = output_path / f"abm_v3_phase_space_state_panel_{start_year}_{end_year}.parquet"
        columns_path = output_path / f"abm_v3_phase_space_state_panel_{start_year}_{end_year}_columns.csv"
        diagnostics_path = output_path / f"abm_v3_phase_space_state_panel_{start_year}_{end_year}_diagnostics.csv"
        summary_path = output_path / f"abm_v3_phase_space_state_panel_{start_year}_{end_year}_summary.csv"

        panel.to_parquet(panel_path, index=False)
        self._build_column_dictionary(panel).to_csv(columns_path, index=False)
        pd.DataFrame(self.diagnostics).to_csv(diagnostics_path, index=False)
        self._build_summary(panel).to_csv(summary_path, index=False)

        return {
            "panel": panel_path,
            "columns": columns_path,
            "diagnostics": diagnostics_path,
            "summary": summary_path,
        }

    def expected_output_columns(self) -> list[str]:
        """Return the canonical state-panel columns expected when available."""
        next_names = [f"{column}_next" for column in NEXT_COLUMNS]
        deltas = [
            "delta_log_EI",
            "delta_g_local",
            "delta_green_capability_export_share",
            "delta_green_capability_readiness",
            "delta_network_green_exposure",
            "rEI",
        ]
        flags = [
            "is_top25_by_output_over_period",
            "is_top25_by_emissions_over_period",
            "is_top25_by_output_in_year",
            "is_top25_by_emissions_in_year",
            "is_low_EI",
            "is_high_EI",
            "is_high_green_capability",
            "is_transition_pivot",
            "is_clean_and_capable",
            "is_brown_lock_in_candidate",
        ]
        weights = ["trajectory_weight_output", "trajectory_weight_emissions", "trajectory_weight_unweighted"]
        return IDENTIFIER_COLUMNS + BASE_CANONICAL_COLUMNS + next_names + deltas + list(RANK_COLUMNS.values()) + flags + weights

    def _read_table(self, path: Path) -> pd.DataFrame:
        """Read a CSV or parquet input table."""
        if not path.exists():
            message = f"Base panel not found: {path}"
            self._record("read_error", "", message, "", str(path))
            raise FileNotFoundError(message)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        message = f"Unsupported base-panel format: {path.suffix}"
        self._record("read_error", "", message, "", str(path))
        raise ValueError(message)

    def _prepare_base_panel(self, panel: pd.DataFrame, start_year: int, end_year: int, source: str) -> pd.DataFrame:
        """Validate keys, filter years, and keep duplicate-key diagnostics visible."""
        missing_required = [column for column in ["country_sector", "Year"] if column not in panel.columns]
        if missing_required:
            for column in missing_required:
                self._record("missing_required_column", column, "Required key column is missing.", "", source)
            if self.strict:
                raise ValueError(f"Base panel is missing required columns: {missing_required}")
            return pd.DataFrame(columns=["country_sector", "Year"])

        prepared = panel.copy()
        prepared["country_sector"] = prepared["country_sector"].astype(str)
        prepared["Year"] = pd.to_numeric(prepared["Year"], errors="coerce").astype("Int64")
        invalid_years = int(prepared["Year"].isna().sum())
        if invalid_years:
            self._record("invalid_year", "Year", "Rows have non-integer years and will be dropped.", invalid_years, source)
        prepared = prepared.loc[prepared["Year"].notna()].copy()
        prepared["Year"] = prepared["Year"].astype(int)
        prepared = prepared.loc[(prepared["Year"] >= int(start_year)) & (prepared["Year"] <= int(end_year))].copy()
        self._record("row_count", "", "base_rows_after_year_filter", len(prepared), source)

        duplicate_count = int(prepared.duplicated(["country_sector", "Year"], keep=False).sum())
        if duplicate_count:
            self._record("duplicated_keys", "country_sector;Year", "Duplicate country_sector-year rows found.", duplicate_count, source)
            if self.strict:
                raise ValueError("Base panel contains duplicated country_sector-Year keys.")
            prepared = prepared.drop_duplicates(["country_sector", "Year"], keep="first").copy()
            self._record("row_count", "", "rows_after_duplicate_drop_keep_first", len(prepared), source)

        for column in prepared.columns:
            self.column_sources[column] = source
        years = sorted(prepared["Year"].dropna().astype(int).unique().tolist())
        self._record("years_covered", "Year", "Years covered after filtering.", f"{min(years)}-{max(years)}" if years else "", source)
        self._record("node_count", "country_sector", "Unique country-sector nodes after filtering.", prepared["country_sector"].nunique(), source)
        return prepared

    def _derive_core_variables(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Derive transparent canonical variables when source columns permit it."""
        output = panel.copy()
        self._numeric_in_place(output)

        if "log_X_observed" not in output.columns and "X_observed" in output.columns:
            output["log_X_observed"] = np.where(output["X_observed"].ge(0), np.log1p(output["X_observed"]), np.nan)
            self._mark_derived("log_X_observed", "log1p(X_observed)")
        if "emissions_observed" not in output.columns and {"EI", "X_observed"}.issubset(output.columns):
            output["emissions_observed"] = output["EI"] * output["X_observed"]
            self._mark_derived("emissions_observed", "EI * X_observed")
        if "log_emissions_observed" not in output.columns and "emissions_observed" in output.columns:
            output["log_emissions_observed"] = np.where(
                output["emissions_observed"].ge(0),
                np.log1p(output["emissions_observed"]),
                np.nan,
            )
            self._mark_derived("log_emissions_observed", "log1p(emissions_observed)")
        if "log_EI" not in output.columns and "EI" in output.columns:
            output["log_EI"] = np.where(output["EI"].gt(0), np.log(output["EI"]), np.nan)
            self._mark_derived("log_EI", "log(EI) for positive EI")
        if "g_local" not in output.columns and "EI" in output.columns:
            output["g_local"] = np.where(output["EI"].ge(0), 1.0 / (1.0 + output["EI"]), np.nan)
            self._mark_derived("g_local", "1 / (1 + EI) for non-negative EI")
        if "capacity_to_observed_ratio" not in output.columns and {"K", "X_observed"}.issubset(output.columns):
            output["capacity_to_observed_ratio"] = np.where(output["X_observed"].gt(0), output["K"] / output["X_observed"], np.nan)
            self._mark_derived("capacity_to_observed_ratio", "K / X_observed for positive X_observed")
        if "capacity_stress" not in output.columns and "capacity_to_observed_ratio" in output.columns:
            output["capacity_stress"] = np.where(
                output["capacity_to_observed_ratio"].gt(0),
                1.0 / output["capacity_to_observed_ratio"],
                np.nan,
            )
            self._mark_derived("capacity_stress", "1 / capacity_to_observed_ratio")

        return output

    def _derive_next_year_variables(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Add next-year variables and one-year movement components."""
        output = panel.sort_values(["country_sector", "Year"]).copy()
        grouped = output.groupby("country_sector", sort=False)
        for column in NEXT_COLUMNS:
            if column in output.columns:
                next_column = f"{column}_next"
                output[next_column] = grouped[column].shift(-1)
                self._mark_derived(next_column, f"next-year value of {column} within country_sector")

        delta_pairs = {
            "delta_log_EI": ("log_EI_next", "log_EI"),
            "delta_g_local": ("g_local_next", "g_local"),
            "delta_green_capability_export_share": (
                "green_capability_export_share_next",
                "green_capability_export_share",
            ),
            "delta_green_capability_readiness": ("green_capability_readiness_next", "green_capability_readiness"),
            "delta_network_green_exposure": ("network_green_exposure_next", "network_green_exposure"),
        }
        for delta_column, (next_column, current_column) in delta_pairs.items():
            if {next_column, current_column}.issubset(output.columns):
                output[delta_column] = output[next_column] - output[current_column]
                self._mark_derived(delta_column, f"{next_column} - {current_column}")
        if {"log_EI", "log_EI_next"}.issubset(output.columns):
            output["rEI"] = output["log_EI"] - output["log_EI_next"]
            self._mark_derived("rEI", "log_EI - log_EI_next")
        return output

    def _derive_rank_percentiles(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Add within-year rank-percentile columns used for selection flags."""
        output = panel.copy()
        for source_column, rank_column in RANK_COLUMNS.items():
            if source_column in output.columns:
                output[rank_column] = output.groupby("Year")[source_column].rank(pct=True, method="average")
                self._mark_derived(rank_column, f"within-year percentile rank of {source_column}")
        return output

    def _derive_selection_flags(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Add transparent node-selection flags for later trajectory filtering."""
        output = panel.copy()
        output_top_nodes = self._top_nodes_over_period(output, "X_observed")
        emissions_top_nodes = self._top_nodes_over_period(output, "emissions_observed")
        output["is_top25_by_output_over_period"] = output["country_sector"].isin(output_top_nodes)
        output["is_top25_by_emissions_over_period"] = output["country_sector"].isin(emissions_top_nodes)
        output["is_top25_by_output_in_year"] = self._top_in_year(output, "X_observed")
        output["is_top25_by_emissions_in_year"] = self._top_in_year(output, "emissions_observed")

        output["is_low_EI"] = self._flag_with_rank(output, "EI_rank_pct", "<=", 0.25) & output.get("EI", pd.Series(np.nan, index=output.index)).gt(0)
        output["is_high_EI"] = self._flag_with_rank(output, "EI_rank_pct", ">=", 0.75) & output.get("EI", pd.Series(np.nan, index=output.index)).gt(0)
        output["is_high_green_capability"] = self._flag_with_rank(output, "green_capability_rank_pct", ">=", 0.75)
        output["is_clean_and_capable"] = output["is_low_EI"] & output["is_high_green_capability"]
        output["is_transition_pivot"] = output["is_high_EI"] & output["is_high_green_capability"]
        brown_high = self._flag_with_rank(output, "brown_centrality_rank_pct", ">=", 0.75)
        output["is_brown_lock_in_candidate"] = output["is_high_EI"] & brown_high & ~output["is_high_green_capability"]
        for column in [column for column in output.columns if column.startswith("is_")]:
            self._mark_derived(column, "transparent rank-based trajectory-selection flag")
        return output

    def _derive_weights(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Add default aggregation and trajectory weights."""
        output = panel.copy()
        output["trajectory_weight_output"] = output["X_observed"] if "X_observed" in output.columns else np.nan
        output["trajectory_weight_emissions"] = output["emissions_observed"] if "emissions_observed" in output.columns else np.nan
        output["trajectory_weight_unweighted"] = 1.0
        self._mark_derived("trajectory_weight_output", "X_observed")
        self._mark_derived("trajectory_weight_emissions", "emissions_observed")
        self._mark_derived("trajectory_weight_unweighted", "constant 1.0")
        return output

    def _order_columns(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Return columns in a stable canonical-first order."""
        ordered = [column for column in self.expected_output_columns() if column in panel.columns]
        remaining = [column for column in panel.columns if column not in ordered]
        return panel[ordered + remaining].sort_values(["country_sector", "Year"]).reset_index(drop=True)

    def _numeric_in_place(self, panel: pd.DataFrame) -> None:
        """Convert expected metric columns to numeric when present."""
        excluded = set(IDENTIFIER_COLUMNS)
        for column in panel.columns:
            if column not in excluded:
                if pd.api.types.is_numeric_dtype(panel[column]):
                    continue
                panel[column] = pd.to_numeric(panel[column], errors="coerce")

    def _top_nodes_over_period(self, panel: pd.DataFrame, value_column: str) -> set[str]:
        """Return top nodes by all-period summed value."""
        if value_column not in panel.columns:
            return set()
        totals = panel.groupby("country_sector")[value_column].sum(min_count=1).sort_values(ascending=False)
        return set(totals.head(self.top_n).index.astype(str))

    def _top_in_year(self, panel: pd.DataFrame, value_column: str) -> pd.Series:
        """Return a boolean flag for top nodes within each year."""
        if value_column not in panel.columns:
            return pd.Series(False, index=panel.index)
        ranks = panel.groupby("Year")[value_column].rank(method="first", ascending=False)
        return ranks.le(self.top_n) & panel[value_column].notna()

    def _flag_with_rank(self, panel: pd.DataFrame, rank_column: str, operator: str, threshold: float) -> pd.Series:
        """Build rank-threshold flags without hiding missing columns."""
        if rank_column not in panel.columns:
            return pd.Series(False, index=panel.index)
        if operator == "<=":
            return panel[rank_column].le(threshold)
        return panel[rank_column].ge(threshold)

    def _build_summary(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Build annual and all-period summary rows."""
        rows = [self._summary_row(year, year_panel) for year, year_panel in panel.groupby("Year", sort=True)]
        if not panel.empty:
            rows.append(self._summary_row("all_period", panel))
        return pd.DataFrame(rows)

    def _summary_row(self, year: int | str, panel: pd.DataFrame) -> dict[str, Any]:
        """Build one summary row."""
        return {
            "Year": year,
            "node_count": panel["country_sector"].nunique() if "country_sector" in panel.columns else len(panel),
            "total_X_observed": self._sum(panel, "X_observed"),
            "total_emissions_observed": self._sum(panel, "emissions_observed"),
            "mean_EI": self._mean(panel, "EI"),
            "weighted_mean_EI": self._weighted_mean(panel, "EI", "X_observed"),
            "mean_g_local": self._mean(panel, "g_local"),
            "weighted_mean_g_local": self._weighted_mean(panel, "g_local", "X_observed"),
            "mean_green_capability_export_share": self._mean(panel, "green_capability_export_share"),
            "weighted_mean_green_capability_export_share": self._weighted_mean(panel, "green_capability_export_share", "X_observed"),
            "mean_network_green_exposure": self._mean(panel, "network_green_exposure"),
            "weighted_mean_network_green_exposure": self._weighted_mean(panel, "network_green_exposure", "X_observed"),
            "top25_output_coverage_share": self._coverage_share(panel, "X_observed", "is_top25_by_output_over_period"),
            "top25_emissions_coverage_share": self._coverage_share(panel, "emissions_observed", "is_top25_by_emissions_over_period"),
        }

    def _build_column_dictionary(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Build a semantic dictionary for the emitted columns."""
        semantic_rows = {row["canonical_variable"]: row for row in build_semantic_variable_map_rows()}
        rows: list[dict[str, Any]] = []
        for column in panel.columns:
            semantic = semantic_rows.get(column, {})
            rows.append(
                {
                    "column": column,
                    "semantic_category": semantic.get("semantic_category", self._fallback_semantic_category(column)),
                    "economic_meaning": semantic.get("economic_meaning", ""),
                    "source": self.column_sources.get(column, "derived"),
                    "derived": column in self.derived_columns,
                    "formula_or_transformation": self.derived_columns.get(column, ""),
                    "recommended_visual_role": semantic.get("suggested_axis_role", ""),
                    "caveats": semantic.get("caveats", ""),
                }
            )
        return pd.DataFrame(rows)

    def _fallback_semantic_category(self, column: str) -> str:
        """Classify columns produced by generic transformations."""
        if column.endswith("_next") or column.startswith("delta_") or column == "rEI":
            return "ei_transition"
        if column.endswith("_rank_pct") or column.startswith("is_") or column.startswith("trajectory_weight"):
            return "diagnostic"
        return ""

    def _record_availability_flags(self, panel: pd.DataFrame) -> None:
        """Record availability of major conceptual layers."""
        availability = {
            "network_green_ness_available": any(column in panel.columns for column in ["network_green_exposure", "g_in_network", "g_out_network", "recursive_green"]),
            "centrality_available": any(column in panel.columns for column in ["pagerank", "eigenvector_centrality", "reverse_eigenvector_centrality"]),
            "brown_centrality_available": "brown_centrality" in panel.columns,
            "ecosystem_capability_readiness_available": any(column in panel.columns for column in ["green_capability_readiness", "capability_ecosystem_exposure"]),
        }
        for name, value in availability.items():
            self._record("availability", name, name, bool(value), "")
        for column in panel.columns:
            missing_share = float(panel[column].isna().mean()) if len(panel) else np.nan
            self._record("missingness", column, "Missing share by canonical variable.", missing_share, self.column_sources.get(column, "derived"))

    def _sum(self, panel: pd.DataFrame, column: str) -> float:
        if column not in panel.columns:
            return np.nan
        return float(panel[column].sum(skipna=True))

    def _mean(self, panel: pd.DataFrame, column: str) -> float:
        if column not in panel.columns:
            return np.nan
        return float(panel[column].mean(skipna=True))

    def _weighted_mean(self, panel: pd.DataFrame, value_column: str, weight_column: str) -> float:
        if value_column not in panel.columns or weight_column not in panel.columns:
            return np.nan
        valid = panel[[value_column, weight_column]].dropna()
        valid = valid.loc[valid[weight_column] > 0]
        if valid.empty:
            return np.nan
        return float(np.average(valid[value_column], weights=valid[weight_column]))

    def _coverage_share(self, panel: pd.DataFrame, value_column: str, flag_column: str) -> float:
        if value_column not in panel.columns or flag_column not in panel.columns:
            return np.nan
        total = panel[value_column].sum(skipna=True)
        if not np.isfinite(total) or total == 0:
            return np.nan
        selected = panel.loc[panel[flag_column].astype(bool), value_column].sum(skipna=True)
        return float(selected / total)

    def _mark_derived(self, column: str, formula: str) -> None:
        """Track a derived column and add a diagnostic row."""
        self.column_sources[column] = "derived"
        self.derived_columns[column] = formula
        self._record("derived_variable", column, "Variable was derived during state-panel construction.", formula, "derived")

    def _record(self, diagnostic_type: str, variable: str, message: str, value: Any, source: str) -> None:
        """Append one auditable diagnostic row."""
        self.diagnostics.append(
            {
                "diagnostic_type": diagnostic_type,
                "variable": variable,
                "message": message,
                "value": value,
                "source": source,
            }
        )
