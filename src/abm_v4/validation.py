from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from src.abm_v4.config import ABMV4Config
from src.abm_v4.paths import ABMV4Paths


@dataclass(frozen=True)
class ValidationMessage:
    """Structured validation message that does not hide uncertainty."""

    check_name: str
    passed: bool
    message: str


@dataclass(frozen=True)
class OneStepValidationThresholds:
    """Transparent pass/warn thresholds for the one-step base validation."""

    weight_sum_error_max: float = 1e-8
    aggregate_feasibility_min: float = 0.95
    high_constrained_node_share: float = 0.80
    decomposition_residual_max_abs: float = 1e-4
    high_invalid_ei_share: float = 0.05
    high_capability_fill_share: float = 0.25


@dataclass(frozen=True)
class OneStepBaseValidationResult:
    """Consolidated validation outputs for the one-step base model."""

    report: pl.DataFrame
    markdown: str
    status: dict[str, Any]

    @property
    def passed(self) -> bool:
        """Return True when no layer has a blocking issue."""
        return bool(self.status["overall_passed"])


@dataclass(frozen=True)
class MultiYearHistoricalValidationResult:
    """Detailed historical validation outputs for a multi-year base run."""

    error_panel: pl.DataFrame
    error_summary: pl.DataFrame
    error_by_sector: pl.DataFrame
    error_by_country: pl.DataFrame
    error_by_ecosystem: pl.DataFrame
    error_by_capability_source: pl.DataFrame
    calibration_targets: pl.DataFrame
    markdown: str


@dataclass(frozen=True)
class TransitionRuleTradeoffDiagnosticResult:
    """Phase 16 transition-rule tradeoff diagnostic artifacts."""

    error_decomposition: pl.DataFrame
    sign_failure_panel: pl.DataFrame
    by_year: pl.DataFrame
    by_sector: pl.DataFrame
    by_country: pl.DataFrame
    by_ecosystem: pl.DataFrame
    by_capability_source: pl.DataFrame
    by_decile: pl.DataFrame
    aggregate_contribution: pl.DataFrame
    hypothesis_tests: pl.DataFrame
    markdown: str


@dataclass(frozen=True)
class HighEmissionsDampeningDiagnosticResult:
    """Phase 17 high-emissions and readiness-dampening diagnostic artifacts."""

    concentration: pl.DataFrame
    electricity: pl.DataFrame
    china_electricity: pl.DataFrame
    dampening: pl.DataFrame
    model_selection: pl.DataFrame
    recommendation: pl.DataFrame
    markdown: str


@dataclass(frozen=True)
class ElectricityDataAuditResult:
    """Phase 18 electricity and China EI data audit artifacts."""

    inventory: pl.DataFrame
    observed_series: pl.DataFrame
    model_series: pl.DataFrame
    anomaly_flags: pl.DataFrame
    cross_country: pl.DataFrame
    recommendation: pl.DataFrame
    markdown: str


class MultiYearHistoricalValidator:
    """Build calibration diagnostics from an existing ABM v4 multi-year run."""

    def __init__(self, paths: ABMV4Paths, config: ABMV4Config) -> None:
        self.paths = paths
        self.config = config

    def load_simulation_panel(self) -> pl.DataFrame:
        """Load the existing multi-year simulation state panel."""
        if not self.paths.base_multiyear_state_panel_path.exists():
            raise FileNotFoundError(
                "Missing multi-year simulation output: "
                f"{self.paths.base_multiyear_state_panel_path}. Run "
                "`python scripts/run_abm_v4_base.py --run-multiyear-base "
                "--start-year 1995 --end-year 2016 --create-output-dirs --reuse-existing` first."
            )
        return pl.read_parquet(self.paths.base_multiyear_state_panel_path)

    def load_summary_panel(self) -> pl.DataFrame:
        """Load the existing multi-year summary panel when present."""
        if not self.paths.base_multiyear_summary_panel_path.exists():
            return pl.DataFrame()
        return pl.read_csv(self.paths.base_multiyear_summary_panel_path)

    def build(self) -> MultiYearHistoricalValidationResult:
        """Build all multi-year historical validation artifacts in memory."""
        simulation = self.load_simulation_panel()
        summary = self.load_summary_panel()
        error_panel = self.build_error_panel(simulation)
        error_summary = self.build_error_summary(error_panel)
        error_by_sector = self.build_grouped_error_summary(error_panel, "Sector")
        error_by_country = self.build_grouped_error_summary(error_panel, "Country")
        error_by_ecosystem = self.build_grouped_error_summary(error_panel, "ecosystem_label")
        error_by_capability_source = self.build_capability_source_summary(error_panel)
        calibration_targets = self.build_calibration_targets(error_panel)
        markdown = self.build_markdown_report(
            error_panel=error_panel,
            error_summary=error_summary,
            error_by_sector=error_by_sector,
            error_by_country=error_by_country,
            error_by_ecosystem=error_by_ecosystem,
            error_by_capability_source=error_by_capability_source,
            calibration_targets=calibration_targets,
            summary_panel=summary,
        )
        return MultiYearHistoricalValidationResult(
            error_panel=error_panel,
            error_summary=error_summary,
            error_by_sector=error_by_sector,
            error_by_country=error_by_country,
            error_by_ecosystem=error_by_ecosystem,
            error_by_capability_source=error_by_capability_source,
            calibration_targets=calibration_targets,
            markdown=markdown,
        )

    def build_error_panel(self, simulation: pl.DataFrame) -> pl.DataFrame:
        """Compute node-year EI, emissions, and transition errors."""
        required = {
            "country_sector",
            "year",
            "Country",
            "Sector",
            "EI_sim",
            "EI_observed",
            "emissions_sim",
            "emissions_observed",
        }
        missing = sorted(required - set(simulation.columns))
        if missing:
            raise ValueError(f"Missing required multi-year simulation columns: {', '.join(missing)}")

        panel = simulation.sort(["country_sector", "year"]).with_columns(
            (pl.col("EI_sim") - pl.col("EI_observed")).alias("EI_error"),
            (pl.col("EI_sim") - pl.col("EI_observed")).abs().alias("EI_abs_error"),
            pl.when(pl.col("EI_observed").is_not_null() & (pl.col("EI_observed") != 0))
            .then((pl.col("EI_sim") - pl.col("EI_observed")) / pl.col("EI_observed"))
            .otherwise(None)
            .alias("EI_pct_error"),
            pl.when((pl.col("EI_sim") > 0) & (pl.col("EI_observed") > 0))
            .then(pl.col("EI_sim").log() - pl.col("EI_observed").log())
            .otherwise(None)
            .alias("log_EI_error"),
            (pl.col("emissions_sim") - pl.col("emissions_observed")).alias("emissions_error"),
            (pl.col("emissions_sim") - pl.col("emissions_observed")).abs().alias("emissions_abs_error"),
            pl.when(pl.col("emissions_observed").is_not_null() & (pl.col("emissions_observed") != 0))
            .then((pl.col("emissions_sim") - pl.col("emissions_observed")) / pl.col("emissions_observed"))
            .otherwise(None)
            .alias("emissions_pct_error"),
            pl.col("EI_observed").shift(-1).over("country_sector").alias("_EI_observed_next"),
            pl.col("EI_sim").shift(-1).over("country_sector").alias("_EI_sim_next"),
            pl.col("year").shift(-1).over("country_sector").alias("_year_next"),
        )
        panel = panel.with_columns(
            pl.when(
                (pl.col("_year_next") == pl.col("year") + 1)
                & (pl.col("EI_observed") > 0)
                & (pl.col("_EI_observed_next") > 0)
            )
            .then(pl.col("EI_observed").log() - pl.col("_EI_observed_next").log())
            .otherwise(None)
            .alias("observed_rEI"),
            pl.when(
                (pl.col("_year_next") == pl.col("year") + 1)
                & (pl.col("EI_sim") > 0)
                & (pl.col("_EI_sim_next") > 0)
            )
            .then(pl.col("EI_sim").log() - pl.col("_EI_sim_next").log())
            .otherwise(None)
            .alias("simulated_rEI"),
        ).with_columns(
            (pl.col("simulated_rEI") - pl.col("observed_rEI")).alias("rEI_error"),
            (pl.col("simulated_rEI") - pl.col("observed_rEI")).abs().alias("rEI_abs_error"),
        )
        panel = self._add_quartile_columns(panel)
        return panel.drop(["_EI_observed_next", "_EI_sim_next", "_year_next"])

    def build_error_summary(self, error_panel: pl.DataFrame) -> pl.DataFrame:
        """Build aggregate yearly error diagnostics."""
        return (
            error_panel.group_by("year")
            .agg(
                pl.col("emissions_observed").sum().alias("total_emissions_observed"),
                pl.col("emissions_sim").sum().alias("total_emissions_sim"),
                pl.col("EI_observed").mean().alias("mean_EI_observed"),
                pl.col("EI_sim").mean().alias("mean_EI_sim"),
                pl.col("EI_observed").median().alias("median_EI_observed"),
                pl.col("EI_sim").median().alias("median_EI_sim"),
                pl.col("observed_rEI").mean().alias("mean_rEI_observed"),
                pl.col("simulated_rEI").mean().alias("mean_rEI_sim"),
                pl.col("EI_abs_error").mean().alias("mean_EI_abs_error"),
                pl.col("log_EI_error").mean().alias("mean_log_EI_error"),
                pl.col("emissions_abs_error").mean().alias("mean_emissions_abs_error"),
                pl.col("rEI_abs_error").mean().alias("mean_rEI_abs_error"),
            )
            .with_columns(
                (pl.col("total_emissions_sim") - pl.col("total_emissions_observed")).alias(
                    "aggregate_emissions_error"
                ),
                pl.when(pl.col("total_emissions_observed") != 0)
                .then(
                    (pl.col("total_emissions_sim") - pl.col("total_emissions_observed"))
                    / pl.col("total_emissions_observed")
                )
                .otherwise(None)
                .alias("aggregate_emissions_pct_error"),
            )
            .sort("year")
        )

    def build_grouped_error_summary(self, error_panel: pl.DataFrame, group_column: str) -> pl.DataFrame:
        """Summarize validation errors by one grouping column."""
        if group_column not in error_panel.columns:
            return pl.DataFrame()
        return self._summarize_groups(error_panel, [group_column]).sort(
            "mean_emissions_abs_error", descending=True
        )

    def build_capability_source_summary(self, error_panel: pl.DataFrame) -> pl.DataFrame:
        """Summarize errors by general and green capability source."""
        frames: list[pl.DataFrame] = []
        for column in ("general_capability_source", "green_capability_source"):
            if column not in error_panel.columns:
                continue
            summary = self._summarize_groups(error_panel, [column]).rename({column: "capability_source"})
            summary = summary.with_columns(pl.lit(column).alias("capability_source_type"))
            frames.append(summary)
        return pl.concat(frames, how="diagonal") if frames else pl.DataFrame()

    def build_calibration_targets(self, error_panel: pl.DataFrame) -> pl.DataFrame:
        """Identify sector-level rEI calibration targets without changing parameters."""
        targets = (
            error_panel.group_by("Sector")
            .agg(
                pl.len().alias("rows"),
                pl.col("observed_rEI").count().alias("valid_rEI_rows"),
                pl.col("observed_rEI").mean().alias("mean_observed_rEI"),
                pl.col("observed_rEI").median().alias("median_observed_rEI"),
                pl.col("simulated_rEI").mean().alias("mean_simulated_rEI"),
                pl.col("simulated_rEI").median().alias("median_simulated_rEI"),
                pl.col("observed_rEI").quantile(0.25).alias("observed_rEI_p25"),
                pl.col("observed_rEI").quantile(0.50).alias("observed_rEI_p50"),
                pl.col("observed_rEI").quantile(0.75).alias("observed_rEI_p75"),
                pl.col("simulated_rEI").quantile(0.25).alias("simulated_rEI_p25"),
                pl.col("simulated_rEI").quantile(0.50).alias("simulated_rEI_p50"),
                pl.col("simulated_rEI").quantile(0.75).alias("simulated_rEI_p75"),
                pl.col("rEI_error").mean().alias("rEI_bias_by_sector"),
                pl.col("rEI_abs_error").mean().alias("mean_rEI_abs_error"),
                pl.col("invalid_EI_flag").mean().alias("invalid_EI_share")
                if "invalid_EI_flag" in error_panel.columns
                else pl.lit(None).alias("invalid_EI_share"),
                pl.col("capability_model_unavailable_flag").mean().alias("capability_unavailable_share")
                if "capability_model_unavailable_flag" in error_panel.columns
                else pl.lit(None).alias("capability_unavailable_share"),
                pl.col("ei_gap").mean().alias("mean_ei_gap")
                if "ei_gap" in error_panel.columns
                else pl.lit(None).alias("mean_ei_gap"),
            )
            .with_columns(
                pl.when(pl.col("invalid_EI_share") > 0.2)
                .then(pl.lit("inspect invalid EI"))
                .when(pl.col("capability_unavailable_share") > 0.25)
                .then(pl.lit("inspect capability source"))
                .when(pl.col("rEI_bias_by_sector") > 0.01)
                .then(pl.lit("decrease readiness"))
                .when(pl.col("rEI_bias_by_sector") < -0.01)
                .then(pl.lit("increase readiness"))
                .when(pl.col("mean_ei_gap").fill_null(0) <= 0)
                .then(pl.lit("inspect frontier gap"))
                .otherwise(pl.lit("adjust background trend"))
                .alias("suggested_direction")
            )
            .sort("mean_rEI_abs_error", descending=True)
        )
        return targets

    def build_markdown_report(
        self,
        *,
        error_panel: pl.DataFrame,
        error_summary: pl.DataFrame,
        error_by_sector: pl.DataFrame,
        error_by_country: pl.DataFrame,
        error_by_ecosystem: pl.DataFrame,
        error_by_capability_source: pl.DataFrame,
        calibration_targets: pl.DataFrame,
        summary_panel: pl.DataFrame,
    ) -> str:
        """Render a human-readable historical validation and calibration report."""
        latest = error_summary.sort("year").tail(1).to_dicts()[0] if not error_summary.is_empty() else {}
        aggregate_pct_error = _as_float(latest.get("aggregate_emissions_pct_error"))
        identity_error = (
            _as_float(summary_panel["emissions_identity_max_error"].max())
            if not summary_panel.is_empty() and "emissions_identity_max_error" in summary_panel.columns
            else 0.0
        )
        dynamic_valid = identity_error < 1e-8
        historically_calibrated = abs(aggregate_pct_error) < 0.25
        cap_source_rows = error_by_capability_source.to_dicts()
        io_rows = [
            row
            for row in cap_source_rows
            if str(row.get("capability_source", "")).lower() == "io_imputed"
        ]
        io_note = (
            "IO-imputed capability does not stand out as unavailable, but it should remain a calibration slice."
            if io_rows
            else "No IO-imputed capability source rows were available in this validation output."
        )
        if io_rows:
            io_log_errors = [
                _as_float(row.get("mean_log_EI_error"))
                for row in io_rows
                if row.get("mean_log_EI_error") is not None
            ]
            if io_log_errors and max(abs(value) for value in io_log_errors) > 0.5:
                io_note = "IO-imputed capability appears associated with large EI errors and should be inspected."

        lines = [
            "# ABM v4 Multi-Year Historical Validation",
            "",
            "## Executive Summary",
            "",
            f"- Dynamic validity: {'pass' if dynamic_valid else 'fail'}",
            f"- Historically calibrated: {'yes' if historically_calibrated else 'no'}",
            f"- Latest aggregate emissions pct error: {aggregate_pct_error}",
            f"- Max emissions identity error: {identity_error}",
            "- Production remains historically forced.",
            "",
            "## Aggregate Emissions Fit",
            "",
            self._markdown_table(
                error_summary.select(
                    [
                        "year",
                        "total_emissions_observed",
                        "total_emissions_sim",
                        "aggregate_emissions_pct_error",
                    ]
                ).tail(5)
            ),
            "",
            "## EI Fit",
            "",
            self._markdown_table(
                error_summary.select(
                    ["year", "mean_EI_observed", "mean_EI_sim", "median_EI_observed", "median_EI_sim"]
                ).tail(5)
            ),
            "",
            "## rEI Transition Fit",
            "",
            self._markdown_table(
                error_summary.select(["year", "mean_rEI_observed", "mean_rEI_sim", "mean_rEI_abs_error"]).tail(5)
            ),
            "",
            "## Largest Error Sectors",
            "",
            self._markdown_table(error_by_sector.head(10)),
            "",
            "## Largest Error Countries",
            "",
            self._markdown_table(error_by_country.head(10)),
            "",
            "## Error by Capability Source",
            "",
            self._markdown_table(error_by_capability_source),
            "",
            "## Error by Ecosystem",
            "",
            self._markdown_table(error_by_ecosystem.head(10)),
            "",
            "## Error by Initial EI Quartile",
            "",
            self._markdown_table(self._summarize_groups(error_panel, ["initial_EI_quartile"])),
            "",
            "## Error by Brown Centrality Quartile",
            "",
            self._markdown_table(self._summarize_groups(error_panel, ["brown_centrality_quartile"])),
            "",
            "## Error by Readiness Quartile",
            "",
            self._markdown_table(self._summarize_groups(error_panel, ["readiness_quartile"])),
            "",
            "## Error by Frontier Gap Quartile",
            "",
            self._markdown_table(self._summarize_groups(error_panel, ["frontier_gap_quartile"])),
            "",
            "## IO-Imputed Capability Assessment",
            "",
            io_note,
            "",
            "## Frontier-Gap Readiness Assessment",
            "",
            "Use the sector-level rEI bias table to distinguish over-greening from under-greening. "
            "Positive rEI bias indicates simulated EI reduction is too fast; negative bias indicates it is too slow.",
            "",
            "## Recommended Calibration Priorities",
            "",
            self._markdown_table(
                calibration_targets.select(
                    [
                        "Sector",
                        "mean_observed_rEI",
                        "mean_simulated_rEI",
                        "rEI_bias_by_sector",
                        "mean_rEI_abs_error",
                        "suggested_direction",
                    ]
                ).head(10)
            ),
            "",
            "## Caveat",
            "",
            "Production is historically forced in this Phase 11 validation. The run can be dynamically coherent "
            "while still not historically calibrated enough for scenarios.",
        ]
        return "\n".join(lines) + "\n"

    def write_outputs(self, result: MultiYearHistoricalValidationResult) -> None:
        """Write multi-year validation artifacts under data/abm_v4/validation."""
        self.paths.validation.mkdir(parents=True, exist_ok=True)
        result.error_panel.write_parquet(self.paths.multiyear_error_panel_path)
        result.error_summary.write_csv(self.paths.multiyear_error_summary_path)
        result.error_by_sector.write_csv(self.paths.multiyear_error_by_sector_path)
        result.error_by_country.write_csv(self.paths.multiyear_error_by_country_path)
        result.error_by_ecosystem.write_csv(self.paths.multiyear_error_by_ecosystem_path)
        result.error_by_capability_source.write_csv(self.paths.multiyear_error_by_capability_source_path)
        result.calibration_targets.write_csv(self.paths.multiyear_calibration_targets_path)
        self.paths.multiyear_validation_report_md_path.write_text(result.markdown, encoding="utf-8")

    def _summarize_groups(self, error_panel: pl.DataFrame, group_columns: list[str]) -> pl.DataFrame:
        return (
            error_panel.group_by(group_columns)
            .agg(
                pl.len().alias("rows"),
                pl.col("EI_abs_error").mean().alias("mean_EI_abs_error"),
                pl.col("EI_abs_error").median().alias("median_EI_abs_error"),
                pl.col("log_EI_error").mean().alias("mean_log_EI_error"),
                pl.col("emissions_abs_error").mean().alias("mean_emissions_abs_error"),
                pl.col("emissions_error").sum().alias("total_emissions_error"),
                pl.col("rEI_error").mean().alias("mean_rEI_error"),
                pl.col("rEI_abs_error").mean().alias("mean_rEI_abs_error"),
            )
            .with_columns(
                pl.col(group_columns[0]).cast(pl.Utf8).fill_null("missing").alias(group_columns[0])
            )
        )

    def _add_quartile_columns(self, panel: pl.DataFrame) -> pl.DataFrame:
        first_year = panel["year"].min()
        initial = (
            panel.filter(pl.col("year") == first_year)
            .select("country_sector", pl.col("EI_observed").alias("_initial_EI_observed"))
        )
        panel = panel.join(initial, on="country_sector", how="left")
        quartile_specs = {
            "_initial_EI_observed": "initial_EI_quartile",
            "brown_centrality": "brown_centrality_quartile",
            "readiness": "readiness_quartile",
            "ei_gap": "frontier_gap_quartile",
        }
        for source, target in quartile_specs.items():
            panel = self._add_quartile_column(panel, source, target)
        return panel.drop("_initial_EI_observed")

    def _add_quartile_column(self, panel: pl.DataFrame, source: str, target: str) -> pl.DataFrame:
        if source not in panel.columns:
            return panel.with_columns(pl.lit("missing").alias(target))
        valid = panel.filter(pl.col(source).is_not_null())
        if valid.is_empty():
            return panel.with_columns(pl.lit("missing").alias(target))
        q25 = valid[source].quantile(0.25)
        q50 = valid[source].quantile(0.50)
        q75 = valid[source].quantile(0.75)
        return panel.with_columns(
            pl.when(pl.col(source).is_null())
            .then(pl.lit("missing"))
            .when(pl.col(source) <= q25)
            .then(pl.lit("q1"))
            .when(pl.col(source) <= q50)
            .then(pl.lit("q2"))
            .when(pl.col(source) <= q75)
            .then(pl.lit("q3"))
            .otherwise(pl.lit("q4"))
            .alias(target)
        )

    def _markdown_table(self, frame: pl.DataFrame) -> str:
        if frame.is_empty():
            return "_No rows available._"
        rows = frame.to_dicts()
        columns = frame.columns
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for row in rows:
            values = [self._format_markdown_value(row.get(column)) for column in columns]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    def _format_markdown_value(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        if value is None:
            return ""
        return str(value).replace("|", "/")


def build_multiyear_base_model_comparison(paths: ABMV4Paths) -> tuple[pl.DataFrame, str]:
    """Compare available default and calibrated-historical multi-year base outputs."""
    rows: list[dict[str, Any]] = []
    specs = [
        (
            "frontier_gap_readiness",
            paths.base_multiyear_state_panel_path,
            paths.base_multiyear_summary_panel_path,
            paths.base_multiyear_validation_report_path,
            "previous base if available",
        ),
        (
            "historical_frontier_gap_only",
            paths.base_multiyear_state_panel_historical_frontier_gap_path,
            paths.base_multiyear_summary_panel_historical_frontier_gap_path,
            paths.base_multiyear_validation_report_historical_frontier_gap_csv_path,
            "Phase 15 calibrated-historical base",
        ),
    ]
    for model_name, state_path, summary_path, validation_path, note in specs:
        if not state_path.exists() or not summary_path.exists():
            continue
        state = pl.read_parquet(state_path)
        summary = pl.read_csv(summary_path)
        validation = pl.read_csv(validation_path) if validation_path.exists() else pl.DataFrame()
        rows.append(
            _build_multiyear_comparison_row(
                model_name=model_name,
                state=state,
                summary=summary,
                validation=validation,
                notes=note,
            )
        )
    comparison = pl.DataFrame(rows)
    markdown = _format_multiyear_base_model_comparison(comparison)
    return comparison, markdown


def write_multiyear_base_model_comparison(paths: ABMV4Paths, comparison: pl.DataFrame, markdown: str) -> None:
    """Write model comparison outputs under data/abm_v4/validation."""
    paths.validation.mkdir(parents=True, exist_ok=True)
    comparison.write_csv(paths.multiyear_base_model_comparison_csv_path)
    paths.multiyear_base_model_comparison_md_path.write_text(markdown, encoding="utf-8")


def _build_multiyear_comparison_row(
    *,
    model_name: str,
    state: pl.DataFrame,
    summary: pl.DataFrame,
    validation: pl.DataFrame,
    notes: str,
) -> dict[str, Any]:
    latest = summary.sort("year").tail(1).to_dicts()[0] if not summary.is_empty() else {}
    errors = _multiyear_state_errors(state)
    validation_row = validation.to_dicts()[0] if not validation.is_empty() else {}
    return {
        "model_variant": model_name,
        "latest_year": latest.get("year"),
        "latest_aggregate_emissions_pct_error": latest.get("aggregate_emissions_error_pct"),
        "mean_yearly_aggregate_emissions_pct_error": summary["aggregate_emissions_error_pct"].mean()
        if "aggregate_emissions_error_pct" in summary.columns and not summary.is_empty()
        else None,
        "mean_EI_error": errors.get("mean_EI_error"),
        "mean_log_EI_error": errors.get("mean_log_EI_error"),
        "rEI_MAE": errors.get("rEI_MAE"),
        "wrong_sign_share": errors.get("wrong_sign_share"),
        "max_emissions_identity_error": validation_row.get(
            "max_emissions_identity_error",
            summary["emissions_identity_max_error"].max()
            if "emissions_identity_max_error" in summary.columns and not summary.is_empty()
            else None,
        ),
        "max_supplier_weight_sum_error": validation_row.get("max_supplier_weight_sum_error"),
        "bad_transition_flag_count": summary["bad_transition_flag"].sum()
        if "bad_transition_flag" in summary.columns and not summary.is_empty()
        else None,
        "status": validation_row.get("status", ""),
        "notes": notes,
    }


def _multiyear_state_errors(state: pl.DataFrame) -> dict[str, float]:
    if state.is_empty():
        return {}
    panel = state.sort(["country_sector", "year"]).with_columns(
        (pl.col("EI_sim") - pl.col("EI_observed")).alias("_EI_error"),
        pl.when((pl.col("EI_sim") > 0) & (pl.col("EI_observed") > 0))
        .then(pl.col("EI_sim").log() - pl.col("EI_observed").log())
        .otherwise(None)
        .alias("_log_EI_error"),
        pl.col("EI_observed").shift(-1).over("country_sector").alias("_EI_observed_next"),
        pl.col("EI_sim").shift(-1).over("country_sector").alias("_EI_sim_next"),
        pl.col("year").shift(-1).over("country_sector").alias("_year_next"),
    )
    panel = panel.with_columns(
        pl.when(
            (pl.col("_year_next") == pl.col("year") + 1)
            & (pl.col("EI_observed") > 0)
            & (pl.col("_EI_observed_next") > 0)
        )
        .then(pl.col("EI_observed").log() - pl.col("_EI_observed_next").log())
        .otherwise(None)
        .alias("_observed_rEI"),
        pl.when(
            (pl.col("_year_next") == pl.col("year") + 1)
            & (pl.col("EI_sim") > 0)
            & (pl.col("_EI_sim_next") > 0)
        )
        .then(pl.col("EI_sim").log() - pl.col("_EI_sim_next").log())
        .otherwise(None)
        .alias("_simulated_rEI"),
    ).with_columns(
        (pl.col("_simulated_rEI") - pl.col("_observed_rEI")).abs().alias("_rEI_abs_error"),
        (
            ((pl.col("_simulated_rEI") > 0) & (pl.col("_observed_rEI") < 0))
            | ((pl.col("_simulated_rEI") < 0) & (pl.col("_observed_rEI") > 0))
        ).alias("_wrong_sign"),
    )
    row = panel.select(
        pl.col("_EI_error").mean().alias("mean_EI_error"),
        pl.col("_log_EI_error").mean().alias("mean_log_EI_error"),
        pl.col("_rEI_abs_error").mean().alias("rEI_MAE"),
        pl.col("_wrong_sign").mean().alias("wrong_sign_share"),
    ).to_dicts()[0]
    return {key: _as_float(value) for key, value in row.items()}


def _format_multiyear_base_model_comparison(comparison: pl.DataFrame) -> str:
    lines = [
        "# ABM v4 Multi-Year Base Model Comparison",
        "",
        "This comparison is historical validation only. It is not a scenario output.",
        "",
    ]
    if comparison.is_empty():
        lines.append("_No comparable multi-year base outputs were found._")
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "| Model | Latest emissions pct error | Mean yearly emissions pct error | rEI MAE | Wrong-sign share | Status | Notes |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in comparison.to_dicts():
        lines.append(
            "| {model} | {latest} | {mean} | {rei} | {wrong} | {status} | {notes} |".format(
                model=row.get("model_variant", ""),
                latest=_format_compact(row.get("latest_aggregate_emissions_pct_error")),
                mean=_format_compact(row.get("mean_yearly_aggregate_emissions_pct_error")),
                rei=_format_compact(row.get("rEI_MAE")),
                wrong=_format_compact(row.get("wrong_sign_share")),
                status=row.get("status", ""),
                notes=str(row.get("notes", "")).replace("|", "/"),
            )
        )
    return "\n".join(lines) + "\n"


def _format_compact(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


class TransitionRuleTradeoffDiagnostics:
    """Diagnose Phase 15 transition-rule magnitude, sign, and aggregate tradeoffs."""

    def __init__(self, paths: ABMV4Paths) -> None:
        self.paths = paths

    def run(self) -> TransitionRuleTradeoffDiagnosticResult:
        """Build all Phase 16 diagnostics in memory."""
        panel = self.build_node_year_comparison_panel()
        error_decomposition = self.compute_weighted_errors(panel)
        by_year = self.summarize_by_group(panel, ["year"])
        by_sector = self.summarize_by_group(panel, ["Sector"]).sort("aggregate_error_contribution", descending=True)
        by_country = self.summarize_by_group(panel, ["Country"]).sort("aggregate_error_contribution", descending=True)
        by_ecosystem = self.summarize_by_group(panel, ["ecosystem_id", "ecosystem_label"])
        by_capability_source = self.summarize_capability_sources(panel)
        by_decile = self.summarize_deciles(panel)
        aggregate = self.compute_aggregate_contributions(panel)
        hypotheses = self.test_hypotheses(panel, error_decomposition, by_sector, by_country, by_decile, aggregate)
        markdown = self.build_tradeoff_report(
            error_decomposition=error_decomposition,
            by_year=by_year,
            by_sector=by_sector,
            by_country=by_country,
            by_capability_source=by_capability_source,
            by_decile=by_decile,
            aggregate_contribution=aggregate,
            hypothesis_tests=hypotheses,
        )
        return TransitionRuleTradeoffDiagnosticResult(
            error_decomposition=error_decomposition,
            sign_failure_panel=panel,
            by_year=by_year,
            by_sector=by_sector,
            by_country=by_country,
            by_ecosystem=by_ecosystem,
            by_capability_source=by_capability_source,
            by_decile=by_decile,
            aggregate_contribution=aggregate,
            hypothesis_tests=hypotheses,
            markdown=markdown,
        )

    def write_outputs(self, result: TransitionRuleTradeoffDiagnosticResult) -> None:
        """Write Phase 16 diagnostics only when explicitly requested."""
        self.paths.validation.mkdir(parents=True, exist_ok=True)
        result.error_decomposition.write_csv(self.paths.transition_rule_error_decomposition_path)
        result.sign_failure_panel.write_parquet(self.paths.transition_rule_sign_failure_panel_path)
        result.by_year.write_csv(self.paths.transition_rule_sign_failure_by_year_path)
        result.by_sector.write_csv(self.paths.transition_rule_sign_failure_by_sector_path)
        result.by_country.write_csv(self.paths.transition_rule_sign_failure_by_country_path)
        result.by_ecosystem.write_csv(self.paths.transition_rule_sign_failure_by_ecosystem_path)
        result.by_capability_source.write_csv(
            self.paths.transition_rule_sign_failure_by_capability_source_path
        )
        result.by_decile.write_csv(self.paths.transition_rule_sign_failure_by_decile_path)
        result.aggregate_contribution.write_csv(self.paths.transition_rule_aggregate_contribution_path)
        result.hypothesis_tests.write_csv(self.paths.transition_rule_hypothesis_tests_path)
        self.paths.transition_rule_error_tradeoff_report_path.write_text(
            result.markdown,
            encoding="utf-8",
        )

    def load_variant_state_panels(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Load previous and calibrated-historical multi-year outputs."""
        missing: list[Path] = []
        for path in (
            self.paths.base_multiyear_state_panel_path,
            self.paths.base_multiyear_state_panel_historical_frontier_gap_path,
        ):
            if not path.exists():
                missing.append(path)
        if missing:
            missing_lines = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(
                "Missing transition-rule state panel(s):\n"
                f"{missing_lines}\n"
                "Run the default multi-year base and the Phase 15 "
                "`--emissions-transition-mode historical_frontier_gap_only` run first."
            )
        return (
            pl.read_parquet(self.paths.base_multiyear_state_panel_path),
            pl.read_parquet(self.paths.base_multiyear_state_panel_historical_frontier_gap_path),
        )

    def build_node_year_comparison_panel(self) -> pl.DataFrame:
        """Align observed transitions and both simulated variants by node-year."""
        readiness, frontier = self.load_variant_state_panels()
        readiness_prepared = self._prepare_variant_panel(readiness, "readiness")
        frontier_prepared = self._prepare_variant_panel(frontier, "frontier_gap")
        joined = readiness_prepared.join(
            frontier_prepared.select(
                "country_sector",
                "year",
                pl.col("EI_sim").alias("EI_sim_frontier_gap"),
                pl.col("emissions_sim").alias("emissions_sim_frontier_gap"),
                pl.col("simulated_rEI").alias("simulated_rEI_frontier_gap"),
                pl.col("rEI_error").alias("rEI_error_frontier_gap"),
                pl.col("rEI_abs_error").alias("rEI_abs_error_frontier_gap"),
                pl.col("rEI_sign_correct").alias("rEI_sign_correct_frontier_gap"),
                pl.col("rEI_wrong_sign").alias("rEI_wrong_sign_frontier_gap"),
                pl.col("emissions_error").alias("emissions_error_frontier_gap"),
                pl.col("ei_gap").alias("ei_gap_frontier_gap"),
                pl.col("readiness").alias("readiness_frontier_gap"),
            ),
            on=["country_sector", "year"],
            how="inner",
        )
        panel = joined.rename(
            {
                "EI_sim": "EI_sim_readiness",
                "emissions_sim": "emissions_sim_readiness",
                "simulated_rEI": "simulated_rEI_readiness",
                "rEI_error": "rEI_error_readiness",
                "rEI_abs_error": "rEI_abs_error_readiness",
                "rEI_sign_correct": "rEI_sign_correct_readiness",
                "rEI_wrong_sign": "rEI_wrong_sign_readiness",
                "emissions_error": "emissions_error_readiness",
                "ei_gap": "ei_gap_readiness",
                "readiness": "readiness_readiness",
            }
        )
        panel = panel.with_columns(
            (pl.col("rEI_abs_error_frontier_gap") - pl.col("rEI_abs_error_readiness")).alias("delta_abs_error"),
            (
                pl.col("rEI_sign_correct_frontier_gap").cast(pl.Int8)
                - pl.col("rEI_sign_correct_readiness").cast(pl.Int8)
            ).alias("delta_sign_correct"),
            (pl.col("rEI_abs_error_frontier_gap") < pl.col("rEI_abs_error_readiness")).alias(
                "frontier_gap_improves_abs_error"
            ),
            (
                pl.col("rEI_sign_correct_readiness") & ~pl.col("rEI_sign_correct_frontier_gap")
            ).alias("frontier_gap_worsens_sign"),
            (
                (pl.col("rEI_abs_error_frontier_gap") < pl.col("rEI_abs_error_readiness"))
                & pl.col("rEI_sign_correct_readiness")
                & ~pl.col("rEI_sign_correct_frontier_gap")
            ).alias("frontier_gap_improves_magnitude_but_worsens_sign"),
            (pl.col("emissions_error_frontier_gap").abs() - pl.col("emissions_error_readiness").abs()).alias(
                "contribution_to_aggregate_error_difference"
            ),
        )
        total_output = panel["X_observed"].sum() or 1.0
        total_emissions = panel["emissions_observed"].sum() or 1.0
        panel = panel.with_columns(
            (pl.col("X_observed") / total_output).alias("output_weight"),
            (pl.col("emissions_observed") / total_emissions).alias("emissions_weight"),
            pl.col("ei_gap_readiness").alias("ei_gap"),
            pl.col("readiness_readiness").alias("readiness"),
        )
        return self._add_deciles(panel)

    def compute_weighted_errors(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Compute unweighted, output-weighted, and emissions-weighted rule metrics."""
        rows: list[dict[str, Any]] = []
        for rule in ("readiness", "frontier_gap"):
            abs_error = f"rEI_abs_error_{rule}"
            wrong = f"rEI_wrong_sign_{rule}"
            rows.append(
                {
                    "model_variant": "frontier_gap_readiness" if rule == "readiness" else "historical_frontier_gap_only",
                    "unweighted_rEI_MAE": _as_float(panel[abs_error].mean()),
                    "output_weighted_rEI_MAE": self._weighted_mean(panel, abs_error, "output_weight"),
                    "emissions_weighted_rEI_MAE": self._weighted_mean(panel, abs_error, "emissions_weight"),
                    "unweighted_wrong_sign_share": _as_float(panel[wrong].mean()),
                    "output_weighted_wrong_sign_share": self._weighted_mean(panel, wrong, "output_weight"),
                    "emissions_weighted_wrong_sign_share": self._weighted_mean(panel, wrong, "emissions_weight"),
                    "mean_emissions_abs_error": _as_float(panel[f"emissions_error_{rule}"].abs().mean()),
                    "total_emissions_abs_error": _as_float(panel[f"emissions_error_{rule}"].abs().sum()),
                    "rows": panel.height,
                }
            )
        return pl.DataFrame(rows)

    def summarize_by_group(self, panel: pl.DataFrame, group_columns: list[str]) -> pl.DataFrame:
        """Summarize sign and magnitude tradeoffs by one grouping."""
        if panel.is_empty() or any(column not in panel.columns for column in group_columns):
            return pl.DataFrame()
        total_emissions = panel["emissions_observed"].sum() or 1.0
        total_contribution = panel["contribution_to_aggregate_error_difference"].sum() or 1.0
        summary = (
            panel.group_by(group_columns)
            .agg(
                pl.len().alias("rows"),
                pl.col("rEI_abs_error_frontier_gap").mean().alias("frontier_gap_abs_error_mean"),
                pl.col("rEI_abs_error_readiness").mean().alias("readiness_abs_error_mean"),
                pl.col("delta_abs_error").mean().alias("delta_abs_error_mean"),
                pl.col("rEI_wrong_sign_frontier_gap").mean().alias("frontier_gap_wrong_sign_share"),
                pl.col("rEI_wrong_sign_readiness").mean().alias("readiness_wrong_sign_share"),
                (
                    pl.col("rEI_wrong_sign_frontier_gap").mean()
                    - pl.col("rEI_wrong_sign_readiness").mean()
                ).alias("delta_wrong_sign_share"),
                (pl.col("emissions_observed").sum() / total_emissions).alias("observed_emissions_share"),
                (pl.col("contribution_to_aggregate_error_difference").sum() / total_contribution).alias(
                    "aggregate_error_contribution"
                ),
                pl.col("frontier_gap_improves_abs_error").mean().alias("frontier_gap_improves_magnitude_share"),
                pl.col("frontier_gap_worsens_sign").mean().alias("frontier_gap_worsens_sign_share"),
                pl.col("frontier_gap_improves_magnitude_but_worsens_sign").mean().alias(
                    "improves_magnitude_but_worsens_sign_share"
                ),
            )
            .with_columns(self._group_interpretation_expr().alias("recommended_interpretation"))
        )
        return summary

    def summarize_capability_sources(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Summarize by general and green capability source."""
        frames: list[pl.DataFrame] = []
        for column in ("general_capability_source", "green_capability_source"):
            if column not in panel.columns:
                continue
            frames.append(
                self.summarize_by_group(panel, [column])
                .rename({column: "capability_source"})
                .with_columns(pl.lit(column).alias("capability_source_type"))
            )
        return pl.concat(frames, how="diagonal") if frames else pl.DataFrame()

    def summarize_deciles(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Summarize by output, emissions, EI, frontier-gap, and readiness deciles."""
        frames: list[pl.DataFrame] = []
        for column, label in [
            ("output_decile", "output"),
            ("emissions_decile", "emissions"),
            ("EI_decile", "EI"),
            ("frontier_gap_decile", "frontier_gap"),
            ("readiness_decile", "readiness"),
        ]:
            if column not in panel.columns:
                continue
            frames.append(
                self.summarize_by_group(panel, [column])
                .rename({column: "decile"})
                .with_columns(pl.lit(label).alias("decile_type"))
            )
        return pl.concat(frames, how="diagonal") if frames else pl.DataFrame()

    def compute_aggregate_contributions(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Rank node-years by contribution to aggregate emissions-error deterioration."""
        contribution = panel.with_columns(
            (
                pl.col("emissions_error_frontier_gap").abs()
                - pl.col("emissions_error_readiness").abs()
            ).alias("abs_difference_in_emissions_error")
        )
        positive_total = contribution.filter(pl.col("abs_difference_in_emissions_error") > 0)[
            "abs_difference_in_emissions_error"
        ].sum()
        if positive_total is None or positive_total == 0:
            positive_total = contribution["abs_difference_in_emissions_error"].abs().sum() or 1.0
            share_expr = pl.col("abs_difference_in_emissions_error").abs() / positive_total
        else:
            share_expr = (
                pl.max_horizontal(pl.lit(0.0), pl.col("abs_difference_in_emissions_error"))
                / positive_total
            )
        return (
            contribution.select(
                "country_sector",
                "year",
                "Country",
                "Sector",
                "X_observed",
                "emissions_observed",
                "emissions_error_readiness",
                "emissions_error_frontier_gap",
                "abs_difference_in_emissions_error",
            )
            .with_columns(share_expr.alias("contribution_share_to_total_difference"))
            .sort("contribution_share_to_total_difference", descending=True)
            .with_row_index("rank", offset=1)
            .with_columns(
                pl.col("contribution_share_to_total_difference")
                .cum_sum()
                .alias("cumulative_contribution_share")
            )
        )

    def test_hypotheses(
        self,
        panel: pl.DataFrame,
        weighted: pl.DataFrame,
        by_sector: pl.DataFrame,
        by_country: pl.DataFrame,
        by_decile: pl.DataFrame,
        aggregate: pl.DataFrame,
    ) -> pl.DataFrame:
        """Build explicit H1-H10 evidence table."""
        rows: list[dict[str, Any]] = []
        fg_wrong = panel.filter(pl.col("rEI_wrong_sign_frontier_gap"))
        h1_value = self._share_frame(
            fg_wrong,
            (pl.col("ei_gap").fill_null(0.0) > 0) & (pl.col("observed_rEI") < 0),
        )
        low_readiness = panel.filter(pl.col("readiness") <= panel["readiness"].median())
        high_readiness = panel.filter(pl.col("readiness") > panel["readiness"].median())
        h2_value = (
            self._mean(low_readiness, "rEI_wrong_sign_frontier_gap")
            - self._mean(high_readiness, "rEI_wrong_sign_frontier_gap")
        )
        readiness_weighted = weighted.filter(pl.col("model_variant") == "frontier_gap_readiness").to_dicts()[0]
        frontier_weighted = weighted.filter(pl.col("model_variant") == "historical_frontier_gap_only").to_dicts()[0]
        h3_value = (
            frontier_weighted["emissions_weighted_rEI_MAE"]
            - frontier_weighted["unweighted_rEI_MAE"]
        )
        top_output = by_decile.filter((pl.col("decile_type") == "output") & (pl.col("decile") == "d10"))
        lower_output = by_decile.filter((pl.col("decile_type") == "output") & (pl.col("decile") != "d10"))
        h4_value = (
            top_output["delta_abs_error_mean"].mean() - lower_output["delta_abs_error_mean"].mean()
            if not top_output.is_empty() and not lower_output.is_empty()
            else None
        )
        h5_value = max(
            self._safe_std(by_country, "frontier_gap_wrong_sign_share"),
            self._safe_std(self.summarize_by_group(panel, ["year"]), "frontier_gap_wrong_sign_share"),
        )
        accurate_sectors = by_sector.sort("readiness_abs_error_mean").head(max(1, by_sector.height // 4))
        h6_value = accurate_sectors["delta_abs_error_mean"].mean() if not accurate_sectors.is_empty() else None
        h7_value = self._top_share(by_sector, "frontier_gap_wrong_sign_share", 5)
        sectors_readiness_better = by_sector.filter(pl.col("delta_abs_error_mean") > 0).height
        h8_value = sectors_readiness_better / max(by_sector.height, 1)
        cap_summary = self.summarize_capability_sources(panel)
        h9_value = self._safe_std(cap_summary, "frontier_gap_wrong_sign_share")
        h10_value = aggregate.head(20)["contribution_share_to_total_difference"].sum()
        rows.extend(
            [
                self._hypothesis_row("H1", "Frontier-gap-only behaves like mean reversion", "Wrong signs concentrate where gap is positive and observed rEI is negative.", "share wrong signs with positive gap and observed worsening", h1_value, h1_value > 0.35, "Add sign dampening or shock diagnostics."),
                self._hypothesis_row("H2", "Readiness was acting as shrinkage", "Gap-only sign failures should be worse among low-readiness nodes.", "low minus high readiness wrong-sign share", h2_value, h2_value is not None and h2_value > 0.03, "Test a weak readiness dampener later."),
                self._hypothesis_row("H3", "Metric optimized the wrong objective", "Weighted errors differ from unweighted errors.", "frontier emissions-weighted minus unweighted rEI MAE", h3_value, abs(h3_value) > 0.01, "Add weighted metrics to model selection."),
                self._hypothesis_row("H4", "p50 frontier is worse for central nodes", "Top-output decile deteriorates more than lower deciles.", "top-output delta minus lower-output delta", h4_value, h4_value is not None and h4_value > 0, "Inspect node-size-conditioned rules."),
                self._hypothesis_row("H5", "Annual direction is shock-dominated", "Wrong-sign failures cluster by year or country.", "max std of wrong-sign shares by year/country", h5_value, h5_value > 0.05, "Use smoothed/multi-year targets or diagnostic controls."),
                self._hypothesis_row("H6", "Sector background does most economic work", "Frontier term worsens already accurate sectors.", "delta abs error in best-readiness sectors", h6_value, h6_value is not None and h6_value > 0, "Gate or weaken frontier term by sector."),
                self._hypothesis_row("H7", "Frontier definition is incomplete", "Sign failures concentrate in sector-country groups.", "top five sector share of wrong-sign intensity", h7_value, h7_value > 0.25, "Consider conditioned frontiers later."),
                self._hypothesis_row("H8", "Readiness should be conditional", "Readiness beats gap-only in a subset of sectors.", "share sectors where readiness has lower abs error", h8_value, h8_value > 0.20, "Test gated readiness, not global readiness."),
                self._hypothesis_row("H9", "IO-imputed capability is not main issue", "Capability-source differences are present but indirect.", "std by capability-source wrong-sign share", h9_value, h9_value < 0.05, "Keep capability-source slices but avoid causal claims."),
                self._hypothesis_row("H10", "Production forcing amplifies EI errors", "Top node-years explain aggregate emissions-error worsening.", "top 20 cumulative contribution share", h10_value, h10_value > 0.50, "Use output/emissions-weighted validation."),
            ]
        )
        return pl.DataFrame(rows)

    def build_tradeoff_report(
        self,
        *,
        error_decomposition: pl.DataFrame,
        by_year: pl.DataFrame,
        by_sector: pl.DataFrame,
        by_country: pl.DataFrame,
        by_capability_source: pl.DataFrame,
        by_decile: pl.DataFrame,
        aggregate_contribution: pl.DataFrame,
        hypothesis_tests: pl.DataFrame,
    ) -> str:
        """Render a compact Phase 16 markdown report."""
        weighted_rows = error_decomposition.to_dicts()
        readiness = next(row for row in weighted_rows if row["model_variant"] == "frontier_gap_readiness")
        frontier = next(row for row in weighted_rows if row["model_variant"] == "historical_frontier_gap_only")
        magnitude_winner = (
            "historical_frontier_gap_only"
            if frontier["unweighted_rEI_MAE"] < readiness["unweighted_rEI_MAE"]
            else "frontier_gap_readiness"
        )
        sign_winner = (
            "historical_frontier_gap_only"
            if frontier["unweighted_wrong_sign_share"] < readiness["unweighted_wrong_sign_share"]
            else "frontier_gap_readiness"
        )
        strongest = hypothesis_tests.sort("evidence_strength").filter(pl.col("supports_hypothesis")).head(5)
        lines = [
            "# ABM v4 Phase 16 Transition Rule Tradeoff Diagnostics",
            "",
            "This is a diagnostic phase, not a scenario or calibration phase.",
            "",
            "## Phase 15 Recap",
            "",
            "The historical frontier-gap-only rule slightly improved magnitude-based rEI metrics but worsened aggregate emissions fit. Direct node-year sign diagnostics are recomputed below because sign conclusions depend on the exact transition panel and weighting used.",
            "",
            "## Core Answers",
            "",
            f"- Better rEI magnitude: `{magnitude_winner}`.",
            f"- Better direction/sign: `{sign_winner}`.",
            "- Better aggregate emissions fit: `frontier_gap_readiness` in the current comparison.",
            "- Scenario readiness: no; production remains historically forced and transition evidence is mixed.",
            "",
            "## Weighted Metrics",
            "",
            self._markdown_table(error_decomposition),
            "",
            "## Top Aggregate Worsening Contributors",
            "",
            self._markdown_table(aggregate_contribution.head(20)),
            "",
            "## Strongest Hypotheses",
            "",
            self._markdown_table(strongest if not strongest.is_empty() else hypothesis_tests.head(5)),
            "",
            "## Largest Sector Diagnostics",
            "",
            self._markdown_table(by_sector.head(15)),
            "",
            "## Largest Country Diagnostics",
            "",
            self._markdown_table(by_country.head(15)),
            "",
            "## Capability Source Diagnostics",
            "",
            self._markdown_table(by_capability_source),
            "",
            "## Decile Diagnostics",
            "",
            self._markdown_table(by_decile.head(30)),
            "",
            "## Recommended Phase 17",
            "",
            "Do not implement scenarios. Test a small set of validation-only objectives that combine unweighted rEI MAE, emissions-weighted rEI MAE, wrong-sign share, and aggregate emissions error. Then diagnose a weak dampened frontier-gap hybrid as a candidate, without changing the default rule.",
        ]
        return "\n".join(lines) + "\n"

    def _prepare_variant_panel(self, frame: pl.DataFrame, rule_name: str) -> pl.DataFrame:
        optional_defaults = {
            "ecosystem_id": pl.lit("missing"),
            "ecosystem_label": pl.lit("missing"),
            "general_capability_source": pl.lit("missing"),
            "green_capability_source": pl.lit("missing"),
            "ei_gap": pl.lit(None, dtype=pl.Float64),
            "readiness": pl.lit(None, dtype=pl.Float64),
            "supplier_lockin": pl.lit(None, dtype=pl.Float64),
            "brown_centrality": pl.lit(None, dtype=pl.Float64),
            "network_green_exposure": pl.lit(None, dtype=pl.Float64),
            "production_feasibility_ratio": pl.lit(None, dtype=pl.Float64),
        }
        for column, expr in optional_defaults.items():
            if column not in frame.columns:
                frame = frame.with_columns(expr.alias(column))
        prepared = frame.sort(["country_sector", "year"]).with_columns(
            pl.col("year").shift(-1).over("country_sector").alias("_next_year"),
            pl.col("EI_observed").shift(-1).over("country_sector").alias("_EI_observed_next"),
            pl.col("EI_sim").shift(-1).over("country_sector").alias("_EI_sim_next"),
        ).filter(
            (pl.col("_next_year") == pl.col("year") + 1)
            & (pl.col("EI_observed") > 0)
            & (pl.col("_EI_observed_next") > 0)
            & (pl.col("EI_sim") > 0)
            & (pl.col("_EI_sim_next") > 0)
        )
        prepared = prepared.with_columns(
            (pl.col("EI_observed").log() - pl.col("_EI_observed_next").log()).alias("observed_rEI"),
            (pl.col("EI_sim").log() - pl.col("_EI_sim_next").log()).alias("simulated_rEI"),
            (pl.col("emissions_sim") - pl.col("emissions_observed")).alias("emissions_error"),
        ).with_columns(
            (pl.col("simulated_rEI") - pl.col("observed_rEI")).alias("rEI_error"),
            (pl.col("simulated_rEI") - pl.col("observed_rEI")).abs().alias("rEI_abs_error"),
            self._sign_correct_expr("simulated_rEI", "observed_rEI").alias("rEI_sign_correct"),
        ).with_columns(
            (~pl.col("rEI_sign_correct")).alias("rEI_wrong_sign")
        )
        columns = [
            "country_sector",
            "year",
            "Country",
            "Sector",
            "ecosystem_id",
            "ecosystem_label",
            "general_capability_source",
            "green_capability_source",
            "X_observed",
            "EI_observed",
            "emissions_observed",
            "observed_rEI",
            "EI_sim",
            "emissions_sim",
            "simulated_rEI",
            "rEI_error",
            "rEI_abs_error",
            "rEI_sign_correct",
            "rEI_wrong_sign",
            "emissions_error",
            "ei_gap",
            "readiness",
            "supplier_lockin",
            "brown_centrality",
            "network_green_exposure",
            "production_feasibility_ratio",
        ]
        return prepared.select(columns).with_columns(pl.lit(rule_name).alias("_rule_name"))

    def _add_deciles(self, panel: pl.DataFrame) -> pl.DataFrame:
        specs = [
            ("X_observed", "output_decile"),
            ("emissions_observed", "emissions_decile"),
            ("EI_observed", "EI_decile"),
            ("ei_gap", "frontier_gap_decile"),
            ("readiness", "readiness_decile"),
        ]
        out = panel
        for source, target in specs:
            out = self._assign_decile(out, source, target)
        return out

    def _assign_decile(self, frame: pl.DataFrame, source: str, target: str) -> pl.DataFrame:
        if source not in frame.columns:
            return frame.with_columns(pl.lit("missing").alias(target))
        valid = frame.filter(pl.col(source).is_not_null())
        if valid.is_empty():
            return frame.with_columns(pl.lit("missing").alias(target))
        thresholds = [valid[source].quantile(i / 10) for i in range(1, 10)]
        expr = pl.when(pl.col(source).is_null()).then(pl.lit("missing"))
        for index, threshold in enumerate(thresholds, start=1):
            expr = expr.when(pl.col(source) <= threshold).then(pl.lit(f"d{index}"))
        return frame.with_columns(expr.otherwise(pl.lit("d10")).alias(target))

    def _sign_correct_expr(self, simulated: str, observed: str) -> pl.Expr:
        return (
            ((pl.col(simulated).abs() <= 1e-12) & (pl.col(observed).abs() <= 1e-12))
            | ((pl.col(simulated) > 0) & (pl.col(observed) > 0))
            | ((pl.col(simulated) < 0) & (pl.col(observed) < 0))
        )

    def _weighted_mean(self, frame: pl.DataFrame, value_column: str, weight_column: str) -> float:
        if frame.is_empty():
            return float("nan")
        value = frame.select(
            (pl.col(value_column).cast(pl.Float64) * pl.col(weight_column)).sum()
            / pl.col(weight_column).sum()
        ).item()
        return _as_float(value)

    def _group_interpretation_expr(self) -> pl.Expr:
        return (
            pl.when((pl.col("delta_abs_error_mean") < 0) & (pl.col("delta_wrong_sign_share") > 0))
            .then(pl.lit("magnitude improves but sign worsens"))
            .when(pl.col("delta_abs_error_mean") < 0)
            .then(pl.lit("frontier gap improves magnitude"))
            .when(pl.col("delta_wrong_sign_share") > 0)
            .then(pl.lit("frontier gap worsens sign"))
            .otherwise(pl.lit("readiness remains preferable or mixed"))
        )

    def _hypothesis_row(
        self,
        hypothesis_id: str,
        hypothesis_name: str,
        expected_pattern: str,
        evidence_metric: str,
        evidence_value: Any,
        supports: bool,
        recommended_fix: str,
    ) -> dict[str, Any]:
        if evidence_value is None:
            strength = "inconclusive"
        elif supports:
            strength = "strong" if abs(float(evidence_value)) > 0.5 else "moderate"
        else:
            strength = "weak"
        return {
            "hypothesis_id": hypothesis_id,
            "hypothesis_name": hypothesis_name,
            "expected_pattern": expected_pattern,
            "evidence_metric": evidence_metric,
            "evidence_value": evidence_value,
            "evidence_strength": strength,
            "supports_hypothesis": supports,
            "interpretation": (
                "Evidence supports this mechanism." if supports else "Evidence is weak or mixed."
            ),
            "recommended_model_fix": recommended_fix,
        }

    def _share_frame(self, frame: pl.DataFrame, expr: pl.Expr) -> float:
        if frame.is_empty():
            return float("nan")
        return _as_float(frame.select(expr.mean()).item())

    def _mean(self, frame: pl.DataFrame, column: str) -> float:
        if frame.is_empty() or column not in frame.columns:
            return float("nan")
        return _as_float(frame[column].mean())

    def _safe_std(self, frame: pl.DataFrame, column: str) -> float:
        if frame.is_empty() or column not in frame.columns:
            return 0.0
        return _as_float(frame[column].std(), 0.0)

    def _top_share(self, frame: pl.DataFrame, column: str, top_n: int) -> float:
        if frame.is_empty() or column not in frame.columns:
            return 0.0
        total = frame[column].sum() or 1.0
        return _as_float(frame.sort(column, descending=True).head(top_n)[column].sum() / total)

    def _markdown_table(self, frame: pl.DataFrame) -> str:
        if frame.is_empty():
            return "_No rows available._"
        columns = frame.columns
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for row in frame.to_dicts():
            lines.append(
                "| "
                + " | ".join(self._format_markdown_value(row.get(column)) for column in columns)
                + " |"
            )
        return "\n".join(lines)

    def _format_markdown_value(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        if value is None:
            return ""
        return str(value).replace("|", "/")


class HighEmissionsDampeningDiagnostics:
    """Diagnose whether Phase 15 tradeoffs are driven by high-emissions dampening."""

    def __init__(self, paths: ABMV4Paths) -> None:
        self.paths = paths

    def run(self) -> HighEmissionsDampeningDiagnosticResult:
        """Build all Phase 17 diagnostics in memory."""
        panel = self.load_tradeoff_panel()
        panel = self._add_dampening_columns(panel)
        aggregate = self.load_aggregate_contribution()
        concentration = self.identify_high_emissions_nodes(panel, aggregate)
        electricity = self.diagnose_electricity_sector(panel)
        china_electricity = self.diagnose_china_electricity(panel)
        dampening = self.compute_readiness_dampening_metrics(panel)
        model_selection = self.build_simplified_model_selection(panel, electricity, china_electricity)
        recommendation = self.build_phase17_recommendation(
            concentration=concentration,
            electricity=electricity,
            china_electricity=china_electricity,
            dampening=dampening,
            model_selection=model_selection,
        )
        markdown = self.build_markdown_report(
            concentration=concentration,
            electricity=electricity,
            china_electricity=china_electricity,
            dampening=dampening,
            model_selection=model_selection,
            recommendation=recommendation,
        )
        return HighEmissionsDampeningDiagnosticResult(
            concentration=concentration,
            electricity=electricity,
            china_electricity=china_electricity,
            dampening=dampening,
            model_selection=model_selection,
            recommendation=recommendation,
            markdown=markdown,
        )

    def write_outputs(self, result: HighEmissionsDampeningDiagnosticResult) -> None:
        """Write Phase 17 outputs only when explicitly requested."""
        self.paths.validation.mkdir(parents=True, exist_ok=True)
        result.concentration.write_csv(self.paths.high_emissions_concentration_diagnostic_path)
        result.electricity.write_csv(self.paths.electricity_sector_dampening_diagnostic_path)
        result.china_electricity.write_csv(self.paths.china_electricity_transition_diagnostic_path)
        result.dampening.write_csv(self.paths.readiness_dampening_diagnostic_path)
        result.model_selection.write_csv(self.paths.simplified_model_selection_tradeoff_path)
        result.recommendation.write_csv(self.paths.phase17_recommendation_path)
        self.paths.phase17_high_emissions_dampening_report_path.write_text(
            result.markdown,
            encoding="utf-8",
        )

    def load_tradeoff_panel(self) -> pl.DataFrame:
        """Load the Phase 16 transition-rule comparison panel."""
        if not self.paths.transition_rule_sign_failure_panel_path.exists():
            raise FileNotFoundError(
                "Missing Phase 16 transition-rule panel: "
                f"{self.paths.transition_rule_sign_failure_panel_path}. Run "
                "`python scripts/run_abm_v4_base.py --diagnose-transition-rule-tradeoffs "
                "--create-output-dirs` first."
            )
        return pl.read_parquet(self.paths.transition_rule_sign_failure_panel_path)

    def load_aggregate_contribution(self) -> pl.DataFrame:
        """Load Phase 16 aggregate contribution ranking if present."""
        if not self.paths.transition_rule_aggregate_contribution_path.exists():
            return pl.DataFrame()
        return pl.read_csv(self.paths.transition_rule_aggregate_contribution_path)

    def identify_high_emissions_nodes(self, panel: pl.DataFrame, aggregate: pl.DataFrame) -> pl.DataFrame:
        """Summarize aggregate deterioration concentration by required groupings."""
        rows: list[dict[str, Any]] = []
        for grouping_type, column in [
            ("Sector", "Sector"),
            ("Country", "Country"),
            ("country_sector", "country_sector"),
            ("emissions_decile", "emissions_decile"),
            ("output_decile", "output_decile"),
        ]:
            if column not in panel.columns:
                continue
            rows.extend(self._concentration_rows(panel, grouping_type, column))
        rows.extend(self._top_contributor_bucket_rows(panel, aggregate))
        return pl.DataFrame(rows)

    def diagnose_electricity_sector(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Return node-year diagnostics for electricity-like sectors."""
        electricity = panel.filter(self._electricity_expr())
        if electricity.is_empty():
            return pl.DataFrame()
        return electricity.select(
            "country_sector",
            "Country",
            "Sector",
            "year",
            "X_observed",
            "EI_observed",
            "emissions_observed",
            "observed_rEI",
            "EI_sim_readiness",
            "EI_sim_frontier_gap",
            "simulated_rEI_readiness",
            "simulated_rEI_frontier_gap",
            "emissions_error_readiness",
            "emissions_error_frontier_gap",
            "rEI_abs_error_readiness",
            "rEI_abs_error_frontier_gap",
            pl.col("rEI_sign_correct_readiness").alias("sign_correct_readiness"),
            pl.col("rEI_sign_correct_frontier_gap").alias("sign_correct_frontier_gap"),
            pl.col("frontier_gap_improves_abs_error").alias("frontier_gap_rule_improves_rEI_abs_error"),
            (
                pl.col("emissions_error_frontier_gap").abs()
                > pl.col("emissions_error_readiness").abs()
            ).alias("frontier_gap_rule_worsens_emissions_error"),
            pl.col("contribution_to_aggregate_error_difference").alias(
                "contribution_to_aggregate_deterioration"
            ),
        ).with_columns(
            pl.when(pl.col("frontier_gap_rule_improves_rEI_abs_error") & pl.col("frontier_gap_rule_worsens_emissions_error"))
            .then(pl.lit("transition improves but emissions error worsens"))
            .when(pl.col("frontier_gap_rule_worsens_emissions_error"))
            .then(pl.lit("frontier gap worsens emissions error"))
            .otherwise(pl.lit("mixed or frontier gap improves"))
            .alias("notes")
        )

    def diagnose_china_electricity(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Return focused diagnostics for China and China electricity node-years."""
        china = panel.filter(
            (pl.col("Country").cast(pl.Utf8).str.to_uppercase() == "CHN")
            | pl.col("country_sector").cast(pl.Utf8).str.contains("CHN")
        )
        if china.is_empty():
            return pl.DataFrame()
        return china.select(
            "country_sector",
            "year",
            "Sector",
            "X_observed",
            "EI_observed",
            "emissions_observed",
            "observed_rEI",
            "simulated_rEI_readiness",
            "simulated_rEI_frontier_gap",
            "rEI_error_readiness",
            "rEI_error_frontier_gap",
            "emissions_error_readiness",
            "emissions_error_frontier_gap",
            pl.col("contribution_to_aggregate_error_difference").alias(
                "contribution_to_aggregate_deterioration"
            ),
        ).with_columns(
            pl.when(self._electricity_expr())
            .then(pl.lit("China electricity or utility node"))
            .when(pl.col("emissions_error_frontier_gap").abs() > pl.col("emissions_error_readiness").abs())
            .then(pl.lit("China non-electricity deterioration"))
            .otherwise(pl.lit("China mixed or improved"))
            .alias("interpretation")
        )

    def compute_readiness_dampening_metrics(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Summarize whether readiness dampens frontier-gap predictions."""
        group_specs = [
            ("all", None),
            ("Sector", "Sector"),
            ("Country", "Country"),
            ("emissions_decile", "emissions_decile"),
            ("output_decile", "output_decile"),
            ("frontier_gap_decile", "frontier_gap_decile"),
            ("readiness_decile", "readiness_decile"),
        ]
        frames: list[pl.DataFrame] = []
        for grouping_type, column in group_specs:
            if column is None:
                frames.append(self._dampening_summary(panel, grouping_type, None))
            elif column in panel.columns:
                frames.append(self._dampening_summary(panel, grouping_type, column))
        return pl.concat(frames, how="diagonal") if frames else pl.DataFrame()

    def build_simplified_model_selection(
        self,
        panel: pl.DataFrame,
        electricity: pl.DataFrame,
        china: pl.DataFrame,
    ) -> pl.DataFrame:
        """Build compact two-rule model-selection tradeoff table."""
        comparison = self._load_model_comparison()
        readiness_summary = self._rule_summary(panel, "readiness")
        frontier_summary = self._rule_summary(panel, "frontier_gap")
        metrics = [
            ("unweighted rEI MAE", readiness_summary["unweighted_rEI_MAE"], frontier_summary["unweighted_rEI_MAE"], "lower is better"),
            ("output-weighted rEI MAE", readiness_summary["output_weighted_rEI_MAE"], frontier_summary["output_weighted_rEI_MAE"], "lower is better"),
            ("emissions-weighted rEI MAE", readiness_summary["emissions_weighted_rEI_MAE"], frontier_summary["emissions_weighted_rEI_MAE"], "lower is better"),
            ("unweighted wrong-sign share", readiness_summary["unweighted_wrong_sign_share"], frontier_summary["unweighted_wrong_sign_share"], "lower is better"),
            ("output-weighted wrong-sign share", readiness_summary["output_weighted_wrong_sign_share"], frontier_summary["output_weighted_wrong_sign_share"], "lower is better"),
            ("emissions-weighted wrong-sign share", readiness_summary["emissions_weighted_wrong_sign_share"], frontier_summary["emissions_weighted_wrong_sign_share"], "lower is better"),
            ("latest-year aggregate emissions pct error", self._comparison_metric(comparison, "frontier_gap_readiness", "latest_aggregate_emissions_pct_error"), self._comparison_metric(comparison, "historical_frontier_gap_only", "latest_aggregate_emissions_pct_error"), "absolute lower is better"),
            ("mean yearly aggregate emissions pct error", self._comparison_metric(comparison, "frontier_gap_readiness", "mean_yearly_aggregate_emissions_pct_error"), self._comparison_metric(comparison, "historical_frontier_gap_only", "mean_yearly_aggregate_emissions_pct_error"), "absolute lower is better"),
            ("high-emissions-node emissions error", self._high_emissions_error(panel, "readiness"), self._high_emissions_error(panel, "frontier_gap"), "lower is better"),
            ("electricity-sector emissions error", self._emissions_abs_error(electricity, "readiness"), self._emissions_abs_error(electricity, "frontier_gap"), "lower is better"),
            ("China emissions error", self._emissions_abs_error(china, "readiness"), self._emissions_abs_error(china, "frontier_gap"), "lower is better"),
        ]
        rows = []
        for metric, readiness_value, frontier_value, interpretation in metrics:
            winner = self._winner(readiness_value, frontier_value, absolute="absolute" in interpretation)
            rows.append(
                {
                    "metric": metric,
                    "frontier_gap_readiness_value": readiness_value,
                    "historical_frontier_gap_only_value": frontier_value,
                    "winner": winner,
                    "interpretation": interpretation,
                }
            )
        return pl.DataFrame(rows)

    def build_phase17_recommendation(
        self,
        *,
        concentration: pl.DataFrame,
        electricity: pl.DataFrame,
        china_electricity: pl.DataFrame,
        dampening: pl.DataFrame,
        model_selection: pl.DataFrame,
    ) -> pl.DataFrame:
        """Apply Phase 17 decision rules."""
        electricity_share = self._share_for_group(concentration, "Sector", "Electricity, Gas and Water")
        china_share = self._share_for_group(concentration, "Country", "CHN")
        top10_share = self._share_for_group(concentration, "top_contributor_bucket", "top_10")
        electricity_worse = self._emissions_abs_error(electricity, "frontier_gap") > self._emissions_abs_error(electricity, "readiness")
        china_worse = self._emissions_abs_error(china_electricity, "frontier_gap") > self._emissions_abs_error(china_electricity, "readiness")
        all_dampening = dampening.filter((pl.col("grouping_type") == "all") & (pl.col("group") == "all"))
        dampening_positive = (
            not all_dampening.is_empty()
            and all_dampening["mean_dampening_amount"].item() > 0
            and all_dampening["share_frontier_gap_more_aggressive"].item() > 0.5
        )
        transition_wins = model_selection.filter(
            pl.col("metric").str.contains("rEI MAE")
            & (pl.col("winner") == "historical_frontier_gap_only")
        ).height
        high_emissions_losses = model_selection.filter(
            pl.col("metric").str.contains("emissions error|aggregate emissions")
            & (pl.col("winner") == "frontier_gap_readiness")
        ).height

        if electricity_share > 0.5 or china_share > 0.5:
            recommendation = "inspect_electricity_data_before_hybrid"
            evidence = f"electricity_share={electricity_share}; china_share={china_share}; top10_share={top10_share}"
            phase18 = "Inspect high-emissions electricity nodes, especially China, before adding a mechanism."
        elif dampening_positive and (electricity_worse or china_worse):
            recommendation = "test_dampened_frontier_gap_hybrid"
            evidence = "readiness dampening is positive and frontier gap worsens high-emissions errors"
            phase18 = "Test a dampened frontier-gap hybrid as a diagnostic candidate."
        elif transition_wins >= 3 and high_emissions_losses >= 2:
            recommendation = "keep_both_rules_until_weighted_validation_objective"
            evidence = "frontier gap wins transition metrics but readiness wins high-emissions/aggregate metrics"
            phase18 = "Build weighted validation objective before choosing a base rule."
        else:
            recommendation = "inconclusive"
            evidence = "decision rules did not identify a dominant mechanism"
            phase18 = "Continue diagnostics before rule changes."
        return pl.DataFrame(
            [
                {
                    "recommendation": recommendation,
                    "evidence": evidence,
                    "interpretation": (
                        "High-emissions concentration remains the immediate blocker."
                        if recommendation == "inspect_electricity_data_before_hybrid"
                        else "Rule choice remains exploratory."
                    ),
                    "recommended_phase18": phase18,
                    "scenario_readiness": "premature",
                }
            ]
        )

    def build_markdown_report(
        self,
        *,
        concentration: pl.DataFrame,
        electricity: pl.DataFrame,
        china_electricity: pl.DataFrame,
        dampening: pl.DataFrame,
        model_selection: pl.DataFrame,
        recommendation: pl.DataFrame,
    ) -> str:
        """Render Phase 17 high-emissions dampening report."""
        electricity_share = self._share_for_group(concentration, "Sector", "Electricity, Gas and Water")
        china_share = self._share_for_group(concentration, "Country", "CHN")
        china_electricity = china_electricity.sort("contribution_to_aggregate_deterioration", descending=True)
        lines = [
            "# ABM v4 Phase 17 High-Emissions Dampening Diagnostics",
            "",
            "This is a diagnostic phase, not a scenario, calibration, or hybrid-rule implementation phase.",
            "",
            "## Phase 16 Recap",
            "",
            "Phase 16 showed that `historical_frontier_gap_only` improves average transition metrics, while aggregate emissions fit worsens because deterioration is concentrated in high-emissions nodes.",
            "",
            "## Core Answers",
            "",
            f"- Electricity deterioration share: {electricity_share:.6g}.",
            f"- China deterioration share: {china_share:.6g}.",
            "- China Electricity, Gas and Water is the dominant repeated node if it appears at the top of the contribution table.",
            "- Readiness dampening is diagnosed below, but no hybrid rule is implemented here.",
            "- Scenarios remain premature.",
            "",
            "## Recommendation",
            "",
            self._markdown_table(recommendation),
            "",
            "## Simplified Model Selection",
            "",
            self._markdown_table(model_selection),
            "",
            "## High-Emissions Concentration",
            "",
            self._markdown_table(concentration.head(40)),
            "",
            "## Electricity Diagnostics",
            "",
            self._markdown_table(electricity.head(30)),
            "",
            "## China Diagnostics",
            "",
            self._markdown_table(china_electricity.head(30)),
            "",
            "## Readiness Dampening",
            "",
            self._markdown_table(dampening.head(40)),
            "",
            "## Recommended Phase 18",
            "",
            str(recommendation["recommended_phase18"].item()) if not recommendation.is_empty() else "Continue diagnostics before scenarios.",
        ]
        return "\n".join(lines) + "\n"

    def _add_dampening_columns(self, panel: pl.DataFrame) -> pl.DataFrame:
        return panel.with_columns(
            (pl.col("rEI_abs_error_frontier_gap") - pl.col("rEI_abs_error_readiness")).alias("delta_abs_error"),
            (pl.col("rEI_sign_correct_frontier_gap").cast(pl.Int8) - pl.col("rEI_sign_correct_readiness").cast(pl.Int8)).alias(
                "delta_sign_correct"
            ),
            (pl.col("rEI_abs_error_frontier_gap") < pl.col("rEI_abs_error_readiness")).alias(
                "frontier_gap_improves_abs_error"
            ),
            (pl.col("rEI_sign_correct_readiness") & ~pl.col("rEI_sign_correct_frontier_gap")).alias(
                "frontier_gap_worsens_sign"
            ),
            (
                (pl.col("rEI_abs_error_frontier_gap") < pl.col("rEI_abs_error_readiness"))
                & pl.col("rEI_sign_correct_readiness")
                & ~pl.col("rEI_sign_correct_frontier_gap")
            ).alias("frontier_gap_improves_magnitude_but_worsens_sign"),
            (pl.col("simulated_rEI_frontier_gap") - pl.col("simulated_rEI_readiness")).alias(
                "simulated_rEI_difference"
            ),
            (pl.col("simulated_rEI_frontier_gap") > pl.col("simulated_rEI_readiness")).alias(
                "frontier_gap_more_aggressive"
            ),
            pl.max_horizontal(
                pl.lit(0.0),
                pl.col("simulated_rEI_frontier_gap") - pl.col("simulated_rEI_readiness"),
            ).alias("dampening_amount"),
            (pl.col("simulated_rEI_frontier_gap") - pl.col("simulated_rEI_readiness"))
            .abs()
            .alias("absolute_dampening_amount"),
        )

    def _concentration_rows(self, panel: pl.DataFrame, grouping_type: str, column: str) -> list[dict[str, Any]]:
        total_emissions = panel["emissions_observed"].sum() or 1.0
        positive_total = self._positive_deterioration_total(panel)
        rows: list[dict[str, Any]] = []
        grouped = (
            panel.group_by(column)
            .agg(
                pl.len().alias("rows"),
                pl.col("emissions_observed").sum().alias("_observed_emissions"),
                pl.max_horizontal(pl.lit(0.0), pl.col("contribution_to_aggregate_error_difference")).sum().alias("_deterioration"),
                pl.col("emissions_error_readiness").abs().sum().alias("readiness_rule_emissions_error"),
                pl.col("emissions_error_frontier_gap").abs().sum().alias("frontier_gap_rule_emissions_error"),
            )
            .with_columns(
                (pl.col("_observed_emissions") / total_emissions).alias("observed_emissions_share"),
                (pl.col("_deterioration") / positive_total).alias("aggregate_deterioration_share"),
                (pl.col("frontier_gap_rule_emissions_error") - pl.col("readiness_rule_emissions_error")).alias("error_difference"),
            )
            .sort("aggregate_deterioration_share", descending=True)
            .with_columns(pl.col("aggregate_deterioration_share").cum_sum().alias("cumulative_deterioration_share"))
        )
        for row in grouped.to_dicts():
            rows.append(
                {
                    "grouping_type": grouping_type,
                    "group": str(row[column]),
                    "rows": row["rows"],
                    "observed_emissions_share": row["observed_emissions_share"],
                    "aggregate_deterioration_share": row["aggregate_deterioration_share"],
                    "cumulative_deterioration_share": row["cumulative_deterioration_share"],
                    "readiness_rule_emissions_error": row["readiness_rule_emissions_error"],
                    "frontier_gap_rule_emissions_error": row["frontier_gap_rule_emissions_error"],
                    "error_difference": row["error_difference"],
                    "interpretation": self._concentration_interpretation(row["aggregate_deterioration_share"]),
                }
            )
        return rows

    def _top_contributor_bucket_rows(self, panel: pl.DataFrame, aggregate: pl.DataFrame) -> list[dict[str, Any]]:
        if aggregate.is_empty():
            positive_total = self._positive_deterioration_total(panel)
            aggregate = panel.select(
                "country_sector",
                "year",
                (
                    pl.max_horizontal(pl.lit(0.0), pl.col("contribution_to_aggregate_error_difference"))
                    / positive_total
                ).alias("contribution_share_to_total_difference"),
            )
        ranked = aggregate.sort("contribution_share_to_total_difference", descending=True)
        keys = {
            "top_10": ranked.head(10).select("country_sector", "year"),
            "top_20": ranked.head(20).select("country_sector", "year"),
            "top_50": ranked.head(50).select("country_sector", "year"),
        }
        rows: list[dict[str, Any]] = []
        for label, key_frame in keys.items():
            subset = panel.join(key_frame, on=["country_sector", "year"], how="inner")
            rows.append(self._bucket_row(subset, panel, label))
        top50 = keys["top_50"]
        rest = panel.join(top50, on=["country_sector", "year"], how="anti")
        rows.append(self._bucket_row(rest, panel, "rest"))
        out = []
        for row in rows:
            row["cumulative_deterioration_share"] = (
                1.0 if row["group"] == "rest" else row["aggregate_deterioration_share"]
            )
            out.append(row)
        return out

    def _bucket_row(self, subset: pl.DataFrame, panel: pl.DataFrame, label: str) -> dict[str, Any]:
        total_emissions = panel["emissions_observed"].sum() or 1.0
        positive_total = self._positive_deterioration_total(panel)
        deterioration = subset.select(
            pl.max_horizontal(pl.lit(0.0), pl.col("contribution_to_aggregate_error_difference")).sum()
        ).item() if not subset.is_empty() else 0.0
        readiness_error = subset["emissions_error_readiness"].abs().sum() if not subset.is_empty() else 0.0
        frontier_error = subset["emissions_error_frontier_gap"].abs().sum() if not subset.is_empty() else 0.0
        return {
            "grouping_type": "top_contributor_bucket",
            "group": label,
            "rows": subset.height,
            "observed_emissions_share": (subset["emissions_observed"].sum() or 0.0) / total_emissions,
            "aggregate_deterioration_share": deterioration / positive_total,
            "cumulative_deterioration_share": None,
            "readiness_rule_emissions_error": readiness_error,
            "frontier_gap_rule_emissions_error": frontier_error,
            "error_difference": frontier_error - readiness_error,
            "interpretation": self._concentration_interpretation(deterioration / positive_total),
        }

    def _dampening_summary(self, panel: pl.DataFrame, grouping_type: str, column: str | None) -> pl.DataFrame:
        group_exprs = [pl.lit("all").alias("group")] if column is None else [pl.col(column).cast(pl.Utf8).alias("group")]
        grouped = (
            panel.with_columns(group_exprs)
            .group_by("group")
            .agg(
                pl.len().alias("rows"),
                pl.col("simulated_rEI_readiness").mean().alias("mean_simulated_rEI_readiness"),
                pl.col("simulated_rEI_frontier_gap").mean().alias("mean_simulated_rEI_frontier_gap"),
                pl.col("dampening_amount").mean().alias("mean_dampening_amount"),
                pl.col("frontier_gap_more_aggressive").mean().alias("share_frontier_gap_more_aggressive"),
                pl.col("observed_rEI").mean().alias("mean_observed_rEI"),
                pl.col("rEI_wrong_sign_frontier_gap").mean().alias("frontier_gap_wrong_sign_share"),
                pl.col("rEI_wrong_sign_readiness").mean().alias("readiness_wrong_sign_share"),
                pl.col("emissions_error_frontier_gap").abs().sum().alias("frontier_gap_emissions_error"),
                pl.col("emissions_error_readiness").abs().sum().alias("readiness_emissions_error"),
            )
            .with_columns(
                pl.lit(grouping_type).alias("grouping_type"),
                pl.when(pl.col("mean_dampening_amount") > 0)
                .then(pl.lit("readiness dampens frontier-gap update"))
                .otherwise(pl.lit("readiness does not dampen on average"))
                .alias("interpretation"),
            )
            .select(
                "grouping_type",
                "group",
                "rows",
                "mean_simulated_rEI_readiness",
                "mean_simulated_rEI_frontier_gap",
                "mean_dampening_amount",
                "share_frontier_gap_more_aggressive",
                "mean_observed_rEI",
                "frontier_gap_wrong_sign_share",
                "readiness_wrong_sign_share",
                "frontier_gap_emissions_error",
                "readiness_emissions_error",
                "interpretation",
            )
        )
        return grouped

    def _rule_summary(self, panel: pl.DataFrame, rule: str) -> dict[str, float]:
        return {
            "unweighted_rEI_MAE": _as_float(panel[f"rEI_abs_error_{rule}"].mean()),
            "output_weighted_rEI_MAE": self._weighted_mean(panel, f"rEI_abs_error_{rule}", "output_weight"),
            "emissions_weighted_rEI_MAE": self._weighted_mean(panel, f"rEI_abs_error_{rule}", "emissions_weight"),
            "unweighted_wrong_sign_share": _as_float(panel[f"rEI_wrong_sign_{rule}"].mean()),
            "output_weighted_wrong_sign_share": self._weighted_mean(panel, f"rEI_wrong_sign_{rule}", "output_weight"),
            "emissions_weighted_wrong_sign_share": self._weighted_mean(panel, f"rEI_wrong_sign_{rule}", "emissions_weight"),
        }

    def _load_model_comparison(self) -> pl.DataFrame:
        if not self.paths.multiyear_base_model_comparison_csv_path.exists():
            return pl.DataFrame()
        return pl.read_csv(self.paths.multiyear_base_model_comparison_csv_path)

    def _comparison_metric(self, comparison: pl.DataFrame, model: str, metric: str) -> float:
        if comparison.is_empty() or metric not in comparison.columns:
            return float("nan")
        row = comparison.filter(pl.col("model_variant") == model)
        return float("nan") if row.is_empty() else _as_float(row[metric].item())

    def _high_emissions_error(self, panel: pl.DataFrame, rule: str) -> float:
        if "emissions_decile" not in panel.columns:
            return self._emissions_abs_error(panel, rule)
        return self._emissions_abs_error(panel.filter(pl.col("emissions_decile") == "d10"), rule)

    def _emissions_abs_error(self, frame: pl.DataFrame, rule: str) -> float:
        if frame.is_empty():
            return float("nan")
        return _as_float(frame[f"emissions_error_{rule}"].abs().sum())

    def _winner(self, readiness_value: float, frontier_value: float, *, absolute: bool = False) -> str:
        if math.isnan(readiness_value) or math.isnan(frontier_value):
            return "inconclusive"
        left = abs(readiness_value) if absolute else readiness_value
        right = abs(frontier_value) if absolute else frontier_value
        if left < right:
            return "frontier_gap_readiness"
        if right < left:
            return "historical_frontier_gap_only"
        return "tie"

    def _weighted_mean(self, frame: pl.DataFrame, value_column: str, weight_column: str) -> float:
        if frame.is_empty() or value_column not in frame.columns or weight_column not in frame.columns:
            return float("nan")
        value = frame.select(
            (pl.col(value_column).cast(pl.Float64) * pl.col(weight_column)).sum()
            / pl.col(weight_column).sum()
        ).item()
        return _as_float(value)

    def _positive_deterioration_total(self, panel: pl.DataFrame) -> float:
        total = panel.select(
            pl.max_horizontal(pl.lit(0.0), pl.col("contribution_to_aggregate_error_difference")).sum()
        ).item()
        return float(total or 1.0)

    def _share_for_group(self, concentration: pl.DataFrame, grouping_type: str, group: str) -> float:
        if concentration.is_empty():
            return 0.0
        row = concentration.filter(
            (pl.col("grouping_type") == grouping_type) & (pl.col("group") == group)
        )
        return 0.0 if row.is_empty() else _as_float(row["aggregate_deterioration_share"].item())

    def _electricity_expr(self) -> pl.Expr:
        return pl.col("Sector").cast(pl.Utf8).str.to_lowercase().str.contains(
            "electricity|gas and water|utilities"
        )

    def _concentration_interpretation(self, share: float) -> str:
        if share >= 0.5:
            return "dominant aggregate deterioration driver"
        if share >= 0.2:
            return "major aggregate deterioration driver"
        if share >= 0.05:
            return "meaningful contributor"
        return "lower contributor"

    def _markdown_table(self, frame: pl.DataFrame) -> str:
        if frame.is_empty():
            return "_No rows available._"
        columns = frame.columns
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for row in frame.to_dicts():
            lines.append(
                "| "
                + " | ".join(self._format_markdown_value(row.get(column)) for column in columns)
                + " |"
            )
        return "\n".join(lines)

    def _format_markdown_value(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        if value is None:
            return ""
        return str(value).replace("|", "/")


class ElectricityDataAudit:
    """Inspect electricity and China EI series before changing transition rules."""

    def __init__(self, paths: ABMV4Paths, start_year: int = 1995, end_year: int = 2016) -> None:
        self.paths = paths
        self.start_year = start_year
        self.end_year = end_year

    def run(self) -> ElectricityDataAuditResult:
        """Build all Phase 18 audit artifacts in memory."""
        observed = self.load_observed_state()
        readiness, frontier = self.load_model_variants()
        electricity = self.identify_electricity_nodes(observed)
        inventory = self.build_electricity_inventory(electricity)
        all_electricity_series = self.audit_all_electricity_observed_series(observed)
        observed_series = self.audit_observed_series(observed)
        model_series = self.audit_model_series(observed_series, readiness, frontier)
        anomaly_flags = self.build_data_quality_flags(observed_series, model_series)
        cross_country = self.compare_electricity_nodes(
            self.audit_model_series(all_electricity_series, readiness, frontier)
        )
        recommendation = self.build_audit_recommendation(anomaly_flags, cross_country, model_series)
        markdown = self.build_markdown_report(
            inventory=inventory,
            observed_series=observed_series,
            model_series=model_series,
            anomaly_flags=anomaly_flags,
            cross_country=cross_country,
            recommendation=recommendation,
        )
        return ElectricityDataAuditResult(
            inventory=inventory,
            observed_series=observed_series,
            model_series=model_series,
            anomaly_flags=anomaly_flags,
            cross_country=cross_country,
            recommendation=recommendation,
            markdown=markdown,
        )

    def write_outputs(self, result: ElectricityDataAuditResult) -> None:
        """Write Phase 18 outputs. The caller controls whether this is invoked."""
        self.paths.validation.mkdir(parents=True, exist_ok=True)
        result.inventory.write_csv(self.paths.electricity_node_inventory_path)
        result.observed_series.write_csv(self.paths.china_electricity_observed_series_audit_path)
        result.model_series.write_csv(self.paths.china_electricity_model_series_audit_path)
        result.anomaly_flags.write_csv(self.paths.electricity_anomaly_flags_path)
        result.cross_country.write_csv(self.paths.electricity_cross_country_comparison_path)
        result.recommendation.write_csv(self.paths.electricity_data_audit_recommendation_path)
        self.paths.electricity_data_audit_report_path.write_text(result.markdown, encoding="utf-8")

    def load_observed_state(self) -> pl.DataFrame:
        """Load observed ABM v4 state and standardize names used by the audit."""
        path = self.paths.state_panel_path(self.start_year, self.end_year)
        if not path.exists():
            raise FileNotFoundError(
                "Missing observed ABM v4 state panel for the electricity audit: "
                f"{path}. Run `python scripts/run_abm_v4_base.py --build-state "
                "--create-output-dirs` first."
            )
        frame = pl.read_parquet(path)
        if "Year" in frame.columns and "year" not in frame.columns:
            frame = frame.rename({"Year": "year"})
        if "EI_observed" not in frame.columns and "EI" in frame.columns:
            frame = frame.rename({"EI": "EI_observed"})
        required = {"country_sector", "year", "Country", "Sector", "X_observed", "EI_observed", "emissions_observed"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Observed state panel is missing required electricity audit columns: {missing}")
        keep = [
            column
            for column in [
                "country_sector",
                "year",
                "Country",
                "Sector",
                "X_observed",
                "EI_observed",
                "emissions_observed",
                "readiness",
                "ei_gap",
                "general_capability_source",
                "green_capability_source",
            ]
            if column in frame.columns
        ]
        return frame.select(keep)

    def load_model_variants(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Load previous readiness and historical frontier-gap multi-year state panels."""
        readiness_path = self.paths.base_multiyear_state_panel_path
        frontier_path = self.paths.base_multiyear_state_panel_historical_frontier_gap_path
        missing = [path for path in (readiness_path, frontier_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing model variant state panel(s) for the electricity audit: "
                f"{missing}. Run the default and `historical_frontier_gap_only` multi-year base runs first."
            )
        return self._prepare_model_panel(pl.read_parquet(readiness_path), "readiness"), self._prepare_model_panel(
            pl.read_parquet(frontier_path), "frontier_gap"
        )

    def identify_electricity_nodes(self, observed: pl.DataFrame) -> pl.DataFrame:
        """Return observed rows whose sector label is electricity-like."""
        return observed.filter(self._electricity_expr())

    def identify_china_electricity_nodes(self, observed: pl.DataFrame) -> pl.DataFrame:
        """Return observed rows that match both China-like and electricity-like labels."""
        return observed.filter(self._electricity_expr() & self._china_expr())

    def build_electricity_inventory(self, electricity: pl.DataFrame) -> pl.DataFrame:
        """Summarize all electricity-like nodes present in ABM v4."""
        if electricity.is_empty():
            return pl.DataFrame(
                schema={
                    "country_sector": pl.Utf8,
                    "Country": pl.Utf8,
                    "Sector": pl.Utf8,
                    "years_available": pl.Int64,
                    "first_year": pl.Int64,
                    "last_year": pl.Int64,
                    "total_observed_emissions": pl.Float64,
                    "mean_observed_EI": pl.Float64,
                    "max_observed_EI": pl.Float64,
                    "notes": pl.Utf8,
                }
            )
        return (
            electricity.group_by("country_sector", "Country", "Sector")
            .agg(
                pl.col("year").n_unique().alias("years_available"),
                pl.col("year").min().alias("first_year"),
                pl.col("year").max().alias("last_year"),
                pl.col("emissions_observed").sum().alias("total_observed_emissions"),
                pl.col("EI_observed").mean().alias("mean_observed_EI"),
                pl.col("EI_observed").max().alias("max_observed_EI"),
            )
            .with_columns(
                pl.when(self._china_expr())
                .then(pl.lit("China electricity-like node"))
                .otherwise(pl.lit("electricity-like node"))
                .alias("notes")
            )
            .sort("total_observed_emissions", descending=True)
        )

    def audit_observed_series(self, observed: pl.DataFrame) -> pl.DataFrame:
        """Build yearly observed China/electricity series with frontier comparability diagnostics."""
        electricity = self.audit_all_electricity_observed_series(observed)
        china_electricity_nodes = electricity.filter(self._china_expr()).select("country_sector").unique()
        top_nodes = self._top_contributor_nodes()
        focus_nodes = pl.concat([china_electricity_nodes, top_nodes], how="diagonal").unique()
        if focus_nodes.is_empty():
            return electricity.filter(self._china_expr())
        return electricity.join(focus_nodes, on="country_sector", how="inner").sort("country_sector", "year")

    def audit_all_electricity_observed_series(self, observed: pl.DataFrame) -> pl.DataFrame:
        """Build yearly observed series for all electricity-like nodes."""
        valid = observed.with_columns(
            pl.when(pl.col("EI_observed") > 0).then(pl.col("EI_observed").log()).otherwise(None).alias("log_EI_observed")
        )
        sector_year = (
            valid.filter(pl.col("EI_observed") > 0)
            .group_by("Sector", "year")
            .agg(
                pl.col("EI_observed").quantile(0.25, interpolation="nearest").alias("sector_year_p25_frontier"),
                pl.col("EI_observed").quantile(0.50, interpolation="nearest").alias("sector_year_p50_frontier"),
                pl.len().alias("_sector_year_count"),
            )
            .sort("Sector", "year")
            .with_columns(
                pl.col("sector_year_p50_frontier").cum_min().over("Sector").alias("rolling_sector_p50_frontier")
            )
        )
        ranked = valid.join(sector_year, on=["Sector", "year"], how="left").with_columns(
            pl.col("EI_observed").rank(method="average").over("Sector", "year").alias("EI_level_rank_within_sector_year"),
        )
        ranked = ranked.with_columns(
            (pl.col("EI_level_rank_within_sector_year") / pl.col("_sector_year_count")).alias(
                "EI_percentile_within_sector_year"
            ),
            (pl.col("log_EI_observed") - pl.col("rolling_sector_p50_frontier").log()).alias("gap_to_rolling_p50"),
            (pl.col("log_EI_observed") - pl.col("sector_year_p50_frontier").log()).alias("gap_to_sector_year_p50"),
            (pl.col("log_EI_observed") - pl.col("sector_year_p25_frontier").log()).alias("gap_to_sector_year_p25"),
        )
        electricity = ranked.filter(self._electricity_expr()).sort("country_sector", "year")
        return electricity.with_columns(
            (pl.col("log_EI_observed") - pl.col("log_EI_observed").shift(1).over("country_sector")).alias("observed_rEI"),
            (pl.col("X_observed") / pl.col("X_observed").shift(1).over("country_sector") - 1.0).alias("pct_change_X"),
            (pl.col("EI_observed") / pl.col("EI_observed").shift(1).over("country_sector") - 1.0).alias("pct_change_EI"),
            (pl.col("emissions_observed") / pl.col("emissions_observed").shift(1).over("country_sector") - 1.0).alias(
                "pct_change_emissions"
            ),
        ).select(
            "country_sector",
            "year",
            "Country",
            "Sector",
            "X_observed",
            "EI_observed",
            "log_EI_observed",
            "emissions_observed",
            "observed_rEI",
            "pct_change_X",
            "pct_change_EI",
            "pct_change_emissions",
            "EI_level_rank_within_sector_year",
            "EI_percentile_within_sector_year",
            "rolling_sector_p50_frontier",
            "gap_to_rolling_p50",
            "gap_to_sector_year_p50",
            "gap_to_sector_year_p25",
        )

    def audit_model_series(
        self,
        observed_series: pl.DataFrame,
        readiness: pl.DataFrame,
        frontier: pl.DataFrame,
    ) -> pl.DataFrame:
        """Compare observed China/electricity series with both simulated variants."""
        model = (
            observed_series.join(readiness, on=["country_sector", "year"], how="left")
            .join(frontier, on=["country_sector", "year"], how="left")
            .with_columns(
                (pl.col("simulated_rEI_frontier_gap") - pl.col("simulated_rEI_readiness")).alias(
                    "simulated_rEI_difference"
                ),
                pl.max_horizontal(
                    pl.lit(0.0),
                    pl.col("simulated_rEI_frontier_gap") - pl.col("simulated_rEI_readiness"),
                ).alias("dampening_amount"),
                (pl.col("simulated_rEI_frontier_gap") - pl.col("observed_rEI")).alias("rEI_error_frontier_gap"),
                (pl.col("simulated_rEI_readiness") - pl.col("observed_rEI")).alias("rEI_error_readiness"),
            )
        )
        return model.with_columns(
            (pl.col("rEI_error_frontier_gap").abs() < pl.col("rEI_error_readiness").abs()).alias(
                "frontier_gap_improves_rEI_abs_error"
            ),
            (pl.col("emissions_error_frontier_gap").abs() > pl.col("emissions_error_readiness").abs()).alias(
                "frontier_gap_worsens_emissions_error"
            ),
            pl.when(
                (pl.col("simulated_rEI_frontier_gap") > pl.col("observed_rEI"))
                & (pl.col("simulated_rEI_frontier_gap") > pl.col("simulated_rEI_readiness"))
            )
            .then(pl.lit("frontier_gap_only over-improves relative to observed and readiness dampens"))
            .when(pl.col("simulated_rEI_frontier_gap") < pl.col("observed_rEI"))
            .then(pl.lit("frontier_gap_only under-improves relative to observed"))
            .otherwise(pl.lit("frontier_gap_only close to observed or not more aggressive"))
            .alias("interpretation"),
        )

    def build_data_quality_flags(self, observed_series: pl.DataFrame, model_series: pl.DataFrame) -> pl.DataFrame:
        """Create row-level structural-break and anomaly flags."""
        electricity_abs_log = observed_series["observed_rEI"].abs().drop_nulls()
        ei_jump_threshold = _as_float(electricity_abs_log.quantile(0.95), 0.0)
        x_jump_threshold = _as_float(observed_series["pct_change_X"].abs().drop_nulls().quantile(0.95), 0.0)
        emissions_jump_threshold = _as_float(
            observed_series["pct_change_emissions"].abs().drop_nulls().quantile(0.95), 0.0
        )
        gap_threshold = _as_float(observed_series["gap_to_rolling_p50"].drop_nulls().quantile(0.95), 0.0)
        base = model_series.with_columns(
            (pl.col("EI_observed").is_null()).alias("EI_missing_flag"),
            ((pl.col("EI_observed").is_not_null()) & (pl.col("EI_observed") <= 0)).alias("EI_nonpositive_flag"),
            (pl.col("observed_rEI").abs() >= ei_jump_threshold).alias("EI_jump_large_flag"),
            (pl.col("pct_change_X").abs() >= x_jump_threshold).alias("X_jump_large_flag"),
            (pl.col("pct_change_emissions").abs() >= emissions_jump_threshold).alias("emissions_jump_large_flag"),
            (pl.col("gap_to_rolling_p50") >= gap_threshold).alias("frontier_gap_extreme_flag"),
            (
                (pl.col("observed_rEI").sign() != pl.col("simulated_rEI_readiness").sign())
                & (pl.col("observed_rEI").sign() != pl.col("simulated_rEI_frontier_gap").sign())
            ).alias("sign_reversal_flag"),
            ((pl.col("simulated_rEI_frontier_gap") - pl.col("simulated_rEI_readiness")).abs() > 0.02).alias(
                "model_disagreement_flag"
            ),
        )
        flag_specs = [
            ("EI_missing_flag", "EI_observed", None, "Observed EI is missing."),
            ("EI_nonpositive_flag", "EI_observed", 0.0, "Observed EI is non-positive."),
            ("EI_jump_large_flag", "observed_rEI", ei_jump_threshold, "Observed EI log change is unusually large."),
            ("X_jump_large_flag", "pct_change_X", x_jump_threshold, "Observed production change is unusually large."),
            (
                "emissions_jump_large_flag",
                "pct_change_emissions",
                emissions_jump_threshold,
                "Observed emissions change is unusually large.",
            ),
            (
                "frontier_gap_extreme_flag",
                "gap_to_rolling_p50",
                gap_threshold,
                "Gap to rolling p50 frontier is unusually large.",
            ),
            (
                "sign_reversal_flag",
                "observed_rEI",
                None,
                "Observed rEI sign differs from both model variants.",
            ),
            (
                "model_disagreement_flag",
                "simulated_rEI_difference",
                0.02,
                "The two rules produce substantially different rEI.",
            ),
        ]
        frames = []
        for flag_name, metric_column, threshold, interpretation in flag_specs:
            frames.append(
                base.select(
                    "country_sector",
                    "year",
                    pl.lit(flag_name).alias("flag_name"),
                    pl.col(flag_name).fill_null(False).alias("flag_value"),
                    pl.col(metric_column).alias("metric_value"),
                    pl.lit(threshold).cast(pl.Float64).alias("threshold"),
                    pl.lit(interpretation).alias("interpretation"),
                )
            )
        return pl.concat(frames, how="vertical_relaxed").sort("country_sector", "year", "flag_name")

    def compare_electricity_nodes(self, model_series: pl.DataFrame) -> pl.DataFrame:
        """Compare electricity-like nodes across countries."""
        total_emissions = model_series["emissions_observed"].sum() or 1.0
        return (
            model_series.group_by("country_sector", "Country", "Sector")
            .agg(
                pl.col("EI_observed").mean().alias("mean_EI"),
                pl.col("EI_observed").median().alias("median_EI"),
                pl.col("observed_rEI").mean().alias("mean_observed_rEI"),
                pl.col("observed_rEI").std().alias("rEI_volatility"),
                (pl.col("emissions_observed").sum() / total_emissions).alias("emissions_share"),
                pl.col("gap_to_rolling_p50").mean().alias("mean_frontier_gap"),
                pl.col("readiness").mean().alias("mean_readiness"),
                pl.col("rEI_error_readiness").abs().mean().alias("MAE_readiness"),
                pl.col("rEI_error_frontier_gap").abs().mean().alias("MAE_frontier_gap"),
                pl.col("emissions_error_readiness").abs().sum().alias("emissions_error_readiness"),
                pl.col("emissions_error_frontier_gap").abs().sum().alias("emissions_error_frontier_gap"),
            )
            .sort("emissions_share", descending=True)
        )

    def build_audit_recommendation(
        self,
        anomaly_flags: pl.DataFrame,
        cross_country: pl.DataFrame,
        model_series: pl.DataFrame,
    ) -> pl.DataFrame:
        """Apply Phase 18 decision rules."""
        china = model_series.filter(self._china_expr())
        china_flags = anomaly_flags.join(
            china.select("country_sector", "year").unique(), on=["country_sector", "year"], how="inner"
        )
        severe_flags = {
            "EI_missing_flag",
            "EI_nonpositive_flag",
            "EI_jump_large_flag",
            "X_jump_large_flag",
            "emissions_jump_large_flag",
        }
        severe_count = china_flags.filter(
            pl.col("flag_name").is_in(sorted(severe_flags)) & pl.col("flag_value")
        ).height
        both_poor = self._both_rules_poor_for_china(china)
        over_improve_share = self._share(
            china,
            (pl.col("simulated_rEI_frontier_gap") > pl.col("observed_rEI"))
            & (pl.col("simulated_rEI_frontier_gap") > pl.col("simulated_rEI_readiness")),
        )
        dampening_help_share = self._share(
            china,
            (pl.col("dampening_amount") > 0) & pl.col("frontier_gap_worsens_emissions_error"),
        )
        if severe_count >= 3 or both_poor:
            action = "inspect_raw_eora_electricity_data"
            interpretation = "China electricity has anomaly flags or both rules perform poorly."
        elif over_improve_share >= 0.5 and dampening_help_share >= 0.5:
            action = "test_dampened_frontier_gap_hybrid"
            interpretation = "Data look usable and readiness dampens frontier-gap over-improvement."
        elif not china.is_empty() and self._is_china_unusual(cross_country):
            action = "treat_electricity_as_sector_specific_transition_case"
            interpretation = "China electricity is structurally unusual relative to other electricity nodes."
        elif not china.is_empty():
            action = "proceed_to_validation_objective_design"
            interpretation = "The aggregate issue is concentrated but not clearly a raw data break."
        else:
            action = "inconclusive"
            interpretation = "China electricity could not be isolated cleanly."
        return pl.DataFrame(
            [
                {
                    "finding": "electricity_china_data_audit",
                    "evidence": (
                        f"severe_china_flags={severe_count}; "
                        f"over_improve_share={over_improve_share:.3f}; "
                        f"dampening_help_share={dampening_help_share:.3f}"
                    ),
                    "interpretation": interpretation,
                    "recommended_next_action": action,
                },
                {
                    "finding": "scenario_readiness",
                    "evidence": "No emissions rule yet performs well on both transition metrics and high-emissions aggregate fit.",
                    "interpretation": "Scenario use remains premature.",
                    "recommended_next_action": "keep_both_rules_for_now",
                },
            ]
        )

    def build_markdown_report(
        self,
        *,
        inventory: pl.DataFrame,
        observed_series: pl.DataFrame,
        model_series: pl.DataFrame,
        anomaly_flags: pl.DataFrame,
        cross_country: pl.DataFrame,
        recommendation: pl.DataFrame,
    ) -> str:
        """Build the Phase 18 audit report."""
        china_inventory = inventory.filter(self._china_expr())
        active_flags = anomaly_flags.filter(pl.col("flag_value"))
        action = recommendation["recommended_next_action"].item(0) if not recommendation.is_empty() else "inconclusive"
        lines = [
            "# ABM v4 Phase 18 Electricity and China EI Data Audit",
            "",
            "This diagnostic audits electricity-like EI data and model behavior before any hybrid rule is implemented.",
            "",
            "## Summary",
            "",
            f"- Electricity-like nodes identified: {inventory.height}.",
            f"- China electricity-like nodes identified: {china_inventory.height}.",
            f"- Active anomaly flags in focused audit rows: {active_flags.height}.",
            f"- Recommended next action: `{action}`.",
            "- Scenarios remain premature.",
            "",
            "## Recommendation",
            "",
            self._markdown_table(recommendation),
            "",
            "## Electricity Inventory",
            "",
            self._markdown_table(inventory.head(30)),
            "",
            "## China Electricity Observed Series",
            "",
            self._markdown_table(observed_series.filter(self._china_expr()).head(40)),
            "",
            "## China Electricity Model Series",
            "",
            self._markdown_table(model_series.filter(self._china_expr()).head(40)),
            "",
            "## Active Anomaly Flags",
            "",
            self._markdown_table(active_flags.head(80)),
            "",
            "## Cross-Country Electricity Comparison",
            "",
            self._markdown_table(cross_country.head(40)),
        ]
        return "\n".join(lines) + "\n"

    def _prepare_model_panel(self, frame: pl.DataFrame, suffix: str) -> pl.DataFrame:
        if "EI_observed" not in frame.columns and "EI" in frame.columns:
            frame = frame.rename({"EI": "EI_observed"})
        cols = [
            "country_sector",
            "year",
            "EI_sim",
            "emissions_sim",
            "rEI_used",
            "ei_gap",
            "readiness",
            "emissions_observed",
        ]
        keep = [column for column in cols if column in frame.columns]
        out = frame.select(keep).sort("country_sector", "year")
        rename = {
            "EI_sim": f"EI_sim_{suffix}",
            "emissions_sim": f"emissions_sim_{suffix}",
            "rEI_used": f"simulated_rEI_{suffix}",
            "emissions_observed": f"emissions_observed_{suffix}",
        }
        if suffix == "frontier_gap":
            rename.update({"ei_gap": "frontier_gap", "readiness": "readiness_frontier_gap_rule"})
        else:
            rename.update({"ei_gap": "ei_gap_readiness_rule"})
        out = out.rename({old: new for old, new in rename.items() if old in out.columns})
        if f"emissions_sim_{suffix}" in out.columns:
            observed_column = f"emissions_observed_{suffix}"
            out = out.with_columns(
                (pl.col(f"emissions_sim_{suffix}") - pl.col(observed_column)).alias(f"emissions_error_{suffix}")
            ).drop(observed_column)
        if suffix == "frontier_gap" and "readiness_frontier_gap_rule" in out.columns:
            out = out.drop("readiness_frontier_gap_rule")
        return out

    def _top_contributor_nodes(self) -> pl.DataFrame:
        path = self.paths.transition_rule_aggregate_contribution_path
        if not path.exists():
            return pl.DataFrame({"country_sector": []}, schema={"country_sector": pl.Utf8})
        frame = pl.read_csv(path)
        if "country_sector" not in frame.columns:
            return pl.DataFrame({"country_sector": []}, schema={"country_sector": pl.Utf8})
        return frame.sort("contribution_share_to_total_difference", descending=True).head(50).select("country_sector").unique()

    def _electricity_expr(self) -> pl.Expr:
        return pl.col("Sector").cast(pl.Utf8).str.to_lowercase().str.contains(
            "electricity|gas and water|utilities|power|water|gas"
        )

    def _china_expr(self) -> pl.Expr:
        country = pl.col("Country").cast(pl.Utf8).str.to_lowercase()
        node = pl.col("country_sector").cast(pl.Utf8).str.to_lowercase()
        return country.str.contains("china|chn|people's republic of china") | node.str.contains(
            "china|chn|people's republic of china"
        )

    def _share(self, frame: pl.DataFrame, expr: pl.Expr) -> float:
        if frame.is_empty():
            return 0.0
        return _as_float(frame.select(expr.mean()).item())

    def _both_rules_poor_for_china(self, china: pl.DataFrame) -> bool:
        if china.is_empty():
            return False
        readiness = _as_float(china["rEI_error_readiness"].abs().mean())
        frontier = _as_float(china["rEI_error_frontier_gap"].abs().mean())
        return readiness > 0.10 and frontier > 0.10

    def _is_china_unusual(self, cross_country: pl.DataFrame) -> bool:
        china = cross_country.filter(self._china_expr())
        if china.is_empty() or cross_country.height < 5:
            return False
        china_gap = _as_float(china["mean_frontier_gap"].abs().max())
        gap_threshold = _as_float(cross_country["mean_frontier_gap"].abs().quantile(0.90))
        china_share = _as_float(china["emissions_share"].sum())
        return china_gap >= gap_threshold or china_share >= 0.20

    def _markdown_table(self, frame: pl.DataFrame) -> str:
        if frame.is_empty():
            return "_No rows available._"
        columns = frame.columns
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for row in frame.to_dicts():
            lines.append(
                "| "
                + " | ".join(self._format_markdown_value(row.get(column)) for column in columns)
                + " |"
            )
        return "\n".join(lines)

    def _format_markdown_value(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        if value is None:
            return ""
        return str(value).replace("|", "/")


def required_one_step_component_paths(
    paths: ABMV4Paths,
    config: ABMV4Config,
) -> dict[str, Path]:
    """Return component outputs needed for a reuse-based one-step validation."""
    return {
        "state_panel": paths.state_panel_path(config.start_year, config.end_year),
        "ecosystem_assignment_report": paths.ecosystem_assignment_report_path,
        "raw_t_supplier_edges": paths.raw_t_supplier_edges_path,
        "raw_t_supplier_edge_report": paths.raw_t_supplier_edge_report_path,
        "supplier_candidate_base_report": paths.supplier_candidate_base_report_path,
        "supplier_opportunity_set_report": paths.supplier_opportunity_set_report_path,
        "supplier_rewiring_report": paths.supplier_rewiring_report_path,
        "capability_update_report": paths.capability_update_report_path,
        "production_feasibility_report": paths.production_feasibility_report_path,
        "emissions_update_report": paths.emissions_update_report_path,
    }


def missing_one_step_component_paths(
    paths: ABMV4Paths,
    config: ABMV4Config,
) -> dict[str, Path]:
    """Return required component paths that are currently missing."""
    return {
        name: path
        for name, path in required_one_step_component_paths(paths, config).items()
        if not path.exists()
    }


def _read_first_csv_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    frame = pl.read_csv(path)
    if frame.is_empty():
        return {}
    return frame.to_dicts()[0]


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def _status_from_checks(passed: bool, warnings: list[str], blocking: list[str]) -> str:
    if not passed or blocking:
        return "fail"
    if warnings:
        return "warning"
    return "pass"


def _join_messages(messages: list[str]) -> str:
    return "; ".join(messages) if messages else ""


def _base_row(layer: str, passed: bool, warnings: list[str], blocking: list[str]) -> dict[str, Any]:
    return {
        "layer": layer,
        "status": _status_from_checks(passed, warnings, blocking),
        "passed": passed and not blocking,
        "warnings": _join_messages(warnings),
        "blocking_issues": _join_messages(blocking),
    }


def _build_state_row(paths: ABMV4Paths, config: ABMV4Config) -> dict[str, Any]:
    state_path = paths.state_panel_path(config.start_year, config.end_year)
    warnings: list[str] = []
    blocking: list[str] = []
    required_columns = {"country_sector", "Year", "X_observed", "EI", "Sector"}
    state = pl.read_parquet(state_path) if state_path.exists() else pl.DataFrame()
    source_report = _read_first_csv_row(paths.diagnostics / "state_source_report.csv")
    missing_required = sorted(required_columns - set(state.columns))
    if state.is_empty():
        blocking.append("State panel is missing or empty.")
    if missing_required:
        blocking.append(f"Missing required state columns: {', '.join(missing_required)}.")
    year_min = state["Year"].min() if "Year" in state.columns and not state.is_empty() else None
    year_max = state["Year"].max() if "Year" in state.columns and not state.is_empty() else None
    node_count = (
        state.select("country_sector").unique().height
        if "country_sector" in state.columns and not state.is_empty()
        else 0
    )
    passed = node_count > 0 and year_max is not None and year_max >= config.end_year and not blocking
    row = _base_row("state", passed, warnings, blocking)
    row.update(
        {
            "state_row_count": state.height,
            "year_coverage": f"{year_min}-{year_max}" if year_min is not None else "",
            "country_sector_node_count": node_count,
            "selected_source": source_report.get("selected_source", "existing_state_panel"),
            "missing_required_variables": ", ".join(missing_required),
        }
    )
    return row


def _build_ecosystem_row(paths: ABMV4Paths) -> dict[str, Any]:
    report = _read_first_csv_row(paths.ecosystem_assignment_report_path)
    warnings: list[str] = []
    blocking: list[str] = []
    unmapped = _as_int(report.get("unmapped_nodes"))
    if not report:
        blocking.append("Ecosystem assignment report is missing.")
    if unmapped != 0:
        blocking.append(f"{unmapped} nodes are unmapped.")
    passed = bool(report) and unmapped == 0
    row = _base_row("ecosystem", passed, warnings, blocking)
    row.update(
        {
            "ecosystem_source": report.get("ecosystem_source_counts", ""),
            "mapped_node_count": report.get("mapped_nodes"),
            "unmapped_node_count": report.get("unmapped_nodes"),
            "ecosystem_count": report.get("number_of_ecosystems"),
        }
    )
    return row


def _build_supplier_row(paths: ABMV4Paths, thresholds: OneStepValidationThresholds) -> dict[str, Any]:
    edge = _read_first_csv_row(paths.raw_t_supplier_edge_report_path)
    candidates = _read_first_csv_row(paths.supplier_candidate_base_report_path)
    opportunities = _read_first_csv_row(paths.supplier_opportunity_set_report_path)
    rewiring = _read_first_csv_row(paths.supplier_rewiring_report_path)
    warnings: list[str] = []
    blocking: list[str] = []
    if not edge:
        blocking.append("Raw T supplier edge report is missing.")
    if not candidates:
        blocking.append("Supplier candidate base report is missing.")
    if not opportunities:
        blocking.append("Supplier opportunity report is missing.")
    if not rewiring:
        blocking.append("Supplier rewiring report is missing.")
    max_initial_error = _as_float(rewiring.get("max_initial_weight_sum_error"))
    max_updated_error = _as_float(rewiring.get("max_updated_weight_sum_error"))
    if max_updated_error >= thresholds.weight_sum_error_max:
        blocking.append(f"Updated supplier weight sum error is {max_updated_error}.")
    if _as_float(rewiring.get("rewired_buyer_share")) == 0:
        warnings.append("Rewired buyer share is zero.")
    number_of_buyers = max(_as_float(rewiring.get("number_of_buyers")), 1.0)
    fallback_stress_share = _as_float(rewiring.get("fallback_stress_buyers")) / number_of_buyers
    if fallback_stress_share > 0.5:
        warnings.append("Fallback stress was used for most buyers.")
    passed = not blocking
    row = _base_row("supplier", passed, warnings, blocking)
    row.update(
        {
            "raw_t_edge_source_status": edge.get("selected_source", "missing"),
            "historical_candidate_rows": candidates.get("historical_candidate_rows"),
            "same_sector_candidate_rows": candidates.get("same_sector_candidate_rows"),
            "ecosystem_candidate_rows": candidates.get("ecosystem_candidate_rows"),
            "opportunity_rows": opportunities.get("opportunity_rows"),
            "median_candidates_per_buyer": opportunities.get("median_candidates_per_buyer"),
            "rewired_buyer_share": rewiring.get("rewired_buyer_share"),
            "max_initial_weight_sum_error": max_initial_error,
            "max_updated_weight_sum_error": max_updated_error,
        }
    )
    return row


def _build_capability_row(paths: ABMV4Paths, thresholds: OneStepValidationThresholds) -> dict[str, Any]:
    report = _read_first_csv_row(paths.capability_update_report_path)
    warnings: list[str] = []
    blocking: list[str] = []
    if not report:
        blocking.append("Capability update report is missing.")
    if _as_int(report.get("cap_clipped_count")) != 0 or _as_int(report.get("gcap_clipped_count")) != 0:
        blocking.append("Capability clipping was detected.")
    general_fill = _as_float(report.get("share_general_capability_filled"))
    green_fill = _as_float(report.get("share_green_capability_filled"))
    if max(general_fill, green_fill) > thresholds.high_capability_fill_share:
        warnings.append("Capability fill share is above threshold.")
    passed = not blocking
    row = _base_row("capability", passed, warnings, blocking)
    row.update(
        {
            "selected_year": report.get("year"),
            "mean_cap": report.get("mean_cap"),
            "mean_gcap": report.get("mean_gcap"),
            "mean_exposure_cap": report.get("mean_exposure_cap"),
            "mean_exposure_gcap": report.get("mean_exposure_gcap"),
            "mean_delta_cap": report.get("mean_delta_cap"),
            "mean_delta_gcap": report.get("mean_delta_gcap"),
            "share_general_capability_filled": general_fill,
            "share_green_capability_filled": green_fill,
        }
    )
    return row


def _build_production_row(paths: ABMV4Paths, thresholds: OneStepValidationThresholds) -> dict[str, Any]:
    report = _read_first_csv_row(paths.production_feasibility_report_path)
    warnings: list[str] = []
    blocking: list[str] = []
    if not report:
        blocking.append("Production feasibility report is missing.")
    aggregate_feasibility = _as_float(report.get("aggregate_feasibility_ratio"))
    constrained_share = _as_float(report.get("share_nodes_with_input_feasibility_below_1"))
    if aggregate_feasibility <= thresholds.aggregate_feasibility_min:
        blocking.append(f"Aggregate feasibility ratio is {aggregate_feasibility}.")
    if constrained_share > thresholds.high_constrained_node_share and aggregate_feasibility > thresholds.aggregate_feasibility_min:
        warnings.append("Many nodes are marginally constrained despite high aggregate feasibility.")
    passed = not blocking
    row = _base_row("production", passed, warnings, blocking)
    row.update(
        {
            "selected_year": report.get("year"),
            "aggregate_feasibility_ratio": aggregate_feasibility,
            "mean_input_feasibility": report.get("mean_input_feasibility"),
            "constrained_node_share": constrained_share,
            "p95_supplier_pressure_max": report.get("p95_supplier_pressure_max"),
            "share_nodes_with_supplier_pressure_above_1": report.get("share_nodes_with_supplier_pressure_above_1"),
        }
    )
    return row


def _build_emissions_row(paths: ABMV4Paths, thresholds: OneStepValidationThresholds) -> dict[str, Any]:
    report = _read_first_csv_row(paths.emissions_update_report_path)
    warnings: list[str] = []
    blocking: list[str] = []
    if not report:
        blocking.append("Emissions update report is missing.")
    residual = abs(_as_float(report.get("decomposition_residual")))
    node_count = max(_as_float(report.get("node_count")), 1.0)
    invalid_share = _as_float(report.get("invalid_EI_nodes")) / node_count
    if residual >= thresholds.decomposition_residual_max_abs:
        blocking.append(f"Emissions decomposition residual is {residual}.")
    if _as_bool(report.get("bad_transition_flag")):
        blocking.append("Bad transition flag is true.")
    if invalid_share > thresholds.high_invalid_ei_share:
        warnings.append("Invalid EI share is above threshold.")
    passed = not blocking
    row = _base_row("emissions", passed, warnings, blocking)
    row.update(
        {
            "transition_mode": report.get("emissions_transition_mode"),
            "valid_EI_nodes": report.get("valid_EI_nodes"),
            "invalid_EI_nodes": report.get("invalid_EI_nodes"),
            "mean_rEI_used": report.get("mean_rEI_used"),
            "median_rEI_used": report.get("median_rEI_used"),
            "aggregate_delta_emissions": report.get("aggregate_delta_emissions"),
            "decomposition_residual": report.get("decomposition_residual"),
            "bad_transition_flag": report.get("bad_transition_flag"),
        }
    )
    return row


def build_one_step_base_validation_report(
    paths: ABMV4Paths,
    config: ABMV4Config,
    thresholds: OneStepValidationThresholds | None = None,
) -> OneStepBaseValidationResult:
    """Aggregate existing one-step ABM v4 diagnostics into one validation report."""
    thresholds = thresholds or OneStepValidationThresholds()
    rows = [
        _build_state_row(paths, config),
        _build_ecosystem_row(paths),
        _build_supplier_row(paths, thresholds),
        _build_capability_row(paths, thresholds),
        _build_production_row(paths, thresholds),
        _build_emissions_row(paths, thresholds),
    ]
    report = pl.DataFrame(rows)
    blocking_issues = [
        row["blocking_issues"] for row in rows if row.get("blocking_issues")
    ]
    warnings = [row["warnings"] for row in rows if row.get("warnings")]
    failed_layers = [row["layer"] for row in rows if row["status"] == "fail"]
    warning_layers = [row["layer"] for row in rows if row["status"] == "warning"]
    overall_passed = not failed_layers
    status = {
        "overall_status": "pass" if overall_passed and not warning_layers else ("warning" if overall_passed else "fail"),
        "overall_passed": overall_passed,
        "failed_layers": failed_layers,
        "warning_layers": warning_layers,
        "warnings": warnings,
        "blocking_issues": blocking_issues,
        "recommended_next_phase": (
            "Phase 9: multi-year base simulation design"
            if overall_passed
            else "Resolve blocking one-step validation issues before multi-year simulation."
        ),
    }
    return OneStepBaseValidationResult(
        report=report,
        markdown=format_one_step_validation_markdown(report, status),
        status=status,
    )


def format_one_step_validation_markdown(
    report: pl.DataFrame,
    status: dict[str, Any],
) -> str:
    """Render the one-step validation report as a compact Markdown summary."""
    lines = [
        "# ABM v4 One-Step Base Validation",
        "",
        f"Overall status: **{status['overall_status']}**",
        "",
        "| Layer | Status | Warnings | Blocking issues |",
        "| --- | --- | --- | --- |",
    ]
    for row in report.to_dicts():
        lines.append(
            "| {layer} | {status} | {warnings} | {blocking} |".format(
                layer=row["layer"],
                status=row["status"],
                warnings=row.get("warnings") or "",
                blocking=row.get("blocking_issues") or "",
            )
        )
    lines.extend(
        [
            "",
            "## Recommended Next Phase",
            "",
            str(status["recommended_next_phase"]),
        ]
    )
    if status["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in status["warnings"])
    if status["blocking_issues"]:
        lines.extend(["", "## Blocking Issues", ""])
        lines.extend(f"- {issue}" for issue in status["blocking_issues"])
    return "\n".join(lines) + "\n"


def write_one_step_base_validation_outputs(
    paths: ABMV4Paths,
    result: OneStepBaseValidationResult,
) -> None:
    """Write one-step validation outputs under data/abm_v4/validation."""
    paths.validation.mkdir(parents=True, exist_ok=True)
    result.report.write_csv(paths.one_step_base_validation_report_csv_path)
    paths.one_step_base_validation_report_md_path.write_text(result.markdown, encoding="utf-8")
    paths.one_step_base_status_json_path.write_text(
        json.dumps(result.status, indent=2),
        encoding="utf-8",
    )
