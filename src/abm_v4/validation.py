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


@dataclass(frozen=True)
class RawEoraElectricityDataAuditResult:
    """Phase 19 raw Eora-derived electricity data path audit artifacts."""

    source_inventory: pl.DataFrame
    china_series_by_source: pl.DataFrame
    cross_source_comparison: pl.DataFrame
    scaling_audit: pl.DataFrame
    mapping_audit: pl.DataFrame
    breakpoint_audit: pl.DataFrame
    major_electricity_comparison: pl.DataFrame
    recommendation: pl.DataFrame
    markdown: str


@dataclass(frozen=True)
class ElectricityTransitionRegimeDiagnosticResult:
    """Phase 20 electricity transition regime diagnostic artifacts."""

    target_diagnostics: pl.DataFrame
    rule_comparison: pl.DataFrame
    by_country: pl.DataFrame
    by_year: pl.DataFrame
    by_decile: pl.DataFrame
    by_jump_status: pl.DataFrame
    china_comparison: pl.DataFrame
    recommendation: pl.DataFrame
    markdown: str


@dataclass(frozen=True)
class StructuralSignatureDiagnosticResult:
    """Phase 21 structural-signature diagnostic artifacts."""

    metric_inventory: pl.DataFrame
    node_year_panel: pl.DataFrame
    node_panel: pl.DataFrame
    label_summary: pl.DataFrame
    electricity_contrast: pl.DataFrame
    metric_screening: pl.DataFrame
    lookalikes: pl.DataFrame
    candidate_proxies: pl.DataFrame
    recommendation: pl.DataFrame
    markdown: str


@dataclass(frozen=True)
class EssentialInputDependenceDiagnosticResult:
    """Phase 22 essential-input and structural-dependence diagnostic artifacts."""

    supplier_buyer_panel: pl.DataFrame
    node_metrics: pl.DataFrame
    electricity_contrast: pl.DataFrame
    symptom_comparison: pl.DataFrame
    metric_screening: pl.DataFrame
    lookalikes: pl.DataFrame
    candidate_proxies: pl.DataFrame
    recommendation: pl.DataFrame
    markdown: str


@dataclass(frozen=True)
class EssentialInputDampenerTestResult:
    """Phase 23 essential-input dampener candidate-test artifacts."""

    candidate_grid: pl.DataFrame
    scores: pl.DataFrame
    residual_panel: pl.DataFrame
    residual_summary: pl.DataFrame
    validation_results: pl.DataFrame
    by_sector: pl.DataFrame
    by_country: pl.DataFrame
    by_electricity: pl.DataFrame
    china_electricity: pl.DataFrame
    by_eid_decile: pl.DataFrame
    mechanism_decomposition: pl.DataFrame
    abm_v5_implications: pl.DataFrame
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


class RawEoraElectricityDataAudit:
    """Trace China electricity observations through existing Eora-derived files."""

    def __init__(self, paths: ABMV4Paths, start_year: int = 1995, end_year: int = 2016) -> None:
        self.paths = paths
        self.start_year = start_year
        self.end_year = end_year

    def run(self) -> RawEoraElectricityDataAuditResult:
        """Build all Phase 19 raw-data-path audit artifacts in memory."""
        inventory = self.discover_candidate_sources()
        usable = inventory.filter(pl.col("usable_for_china_electricity"))
        if usable.is_empty():
            raise FileNotFoundError(
                "No usable Eora-derived China electricity sources were found. Run Phase 18 first and ensure "
                "ABM v4 state/final parquet sources exist under data/abm_v4, data/abm_v3, or data/final."
            )
        series = self.extract_china_electricity_series(usable)
        if series.is_empty():
            raise FileNotFoundError(
                "Usable schemas were found, but no China electricity records could be extracted. Inspect "
                "`raw_eora_electricity_source_inventory.csv` for available country/sector columns."
            )
        comparison = self.compare_sources(series)
        scaling = self.audit_units_and_scaling(series)
        mapping = self.audit_mapping_consistency(series)
        preferred_source = self._preferred_source_path(series)
        electricity_records = self.extract_major_electricity_records(preferred_source)
        breakpoint = self.detect_breakpoints_and_jumps(electricity_records)
        major = self.build_major_electricity_comparison(electricity_records, breakpoint)
        recommendation = self.build_recommendation(comparison, scaling, mapping, breakpoint, major)
        markdown = self.build_markdown_report(
            inventory=inventory,
            series=series,
            comparison=comparison,
            scaling=scaling,
            mapping=mapping,
            breakpoint=breakpoint,
            major=major,
            recommendation=recommendation,
        )
        return RawEoraElectricityDataAuditResult(
            source_inventory=inventory,
            china_series_by_source=series,
            cross_source_comparison=comparison,
            scaling_audit=scaling,
            mapping_audit=mapping,
            breakpoint_audit=breakpoint,
            major_electricity_comparison=major,
            recommendation=recommendation,
            markdown=markdown,
        )

    def write_outputs(self, result: RawEoraElectricityDataAuditResult) -> None:
        """Write Phase 19 outputs. The caller controls whether this is invoked."""
        self.paths.validation.mkdir(parents=True, exist_ok=True)
        result.source_inventory.write_csv(self.paths.raw_eora_electricity_source_inventory_path)
        result.china_series_by_source.write_csv(self.paths.raw_eora_china_electricity_series_by_source_path)
        result.cross_source_comparison.write_csv(self.paths.raw_eora_china_electricity_cross_source_comparison_path)
        result.scaling_audit.write_csv(self.paths.raw_eora_electricity_scaling_audit_path)
        result.mapping_audit.write_csv(self.paths.raw_eora_electricity_mapping_audit_path)
        result.breakpoint_audit.write_csv(self.paths.raw_eora_electricity_breakpoint_audit_path)
        result.major_electricity_comparison.write_csv(self.paths.raw_eora_major_electricity_comparison_path)
        result.recommendation.write_csv(self.paths.raw_eora_electricity_data_audit_recommendation_path)
        self.paths.raw_eora_electricity_data_audit_report_path.write_text(result.markdown, encoding="utf-8")

    def discover_candidate_sources(self) -> pl.DataFrame:
        """Discover and inspect schemas for plausible Eora-derived tabular files."""
        rows = []
        for path in self._candidate_source_paths():
            rows.append(self.inspect_source_schema(path))
        if not rows:
            return pl.DataFrame(
                schema={
                    "source_path": pl.Utf8,
                    "exists": pl.Boolean,
                    "file_type": pl.Utf8,
                    "rows_if_known": pl.Int64,
                    "columns": pl.Utf8,
                    "has_country": pl.Boolean,
                    "has_sector": pl.Boolean,
                    "has_year": pl.Boolean,
                    "has_output_candidate": pl.Boolean,
                    "has_emissions_candidate": pl.Boolean,
                    "has_ei_candidate": pl.Boolean,
                    "usable_for_china_electricity": pl.Boolean,
                    "notes": pl.Utf8,
                }
            )
        return pl.DataFrame(rows).sort("source_path")

    def inspect_source_schema(self, path: Path) -> dict[str, Any]:
        """Return a defensive schema inventory row for one candidate source."""
        exists = path.exists()
        file_type = path.suffix.lower().lstrip(".")
        columns: list[str] = []
        rows_if_known: int | None = None
        notes = ""
        if exists:
            try:
                if file_type == "parquet":
                    scan = pl.scan_parquet(path)
                    columns = list(scan.collect_schema().names())
                    try:
                        rows_if_known = int(scan.select(pl.len()).collect().item())
                    except Exception:
                        rows_if_known = None
                elif file_type == "csv":
                    sample = pl.read_csv(path, n_rows=50, infer_schema_length=50)
                    columns = sample.columns
                else:
                    notes = "unsupported file type"
            except Exception as exc:
                notes = f"schema inspection failed: {exc}"
        profile = self._column_profile(columns)
        usable = bool(
            exists
            and profile["year_column"]
            and (profile["country_sector_column"] or (profile["country_column"] and profile["sector_column"]))
            and (profile["output_column"] or profile["emissions_column"] or profile["ei_column"])
        )
        return {
            "source_path": self._rel(path),
            "exists": exists,
            "file_type": file_type,
            "rows_if_known": rows_if_known,
            "columns": ";".join(columns),
            "has_country": bool(profile["country_column"] or profile["country_sector_column"]),
            "has_sector": bool(profile["sector_column"] or profile["country_sector_column"]),
            "has_year": bool(profile["year_column"]),
            "has_output_candidate": bool(profile["output_column"]),
            "has_emissions_candidate": bool(profile["emissions_column"]),
            "has_ei_candidate": bool(profile["ei_column"]),
            "usable_for_china_electricity": usable,
            "notes": notes if notes else ("usable schema" if usable else "missing one or more required semantic columns"),
        }

    def extract_china_electricity_series(self, usable_sources: pl.DataFrame) -> pl.DataFrame:
        """Extract China electricity records from every usable source."""
        frames = []
        for row in usable_sources.to_dicts():
            frame = self.identify_china_electricity_records(self.paths.project_root / row["source_path"])
            if not frame.is_empty():
                frames.append(frame)
        return self._empty_china_series() if not frames else pl.concat(frames, how="diagonal_relaxed").sort(
            "source_path", "year"
        )

    def identify_china_electricity_records(self, source_path: Path) -> pl.DataFrame:
        """Extract normalized China electricity rows from one source, if possible."""
        frame = self._read_source(source_path)
        if frame.is_empty():
            return self._empty_china_series()
        profile = self._column_profile(frame.columns)
        if not profile["year_column"]:
            return self._empty_china_series()
        prepared = self._normalize_source_frame(frame, profile)
        filtered = prepared.filter(self._china_expr() & self._electricity_expr()).sort("country_sector", "year")
        if filtered.is_empty():
            return self._empty_china_series()
        return self._derive_series_fields(filtered.with_columns(pl.lit(self._rel(source_path)).alias("source_path")))

    def extract_major_electricity_records(self, source_path_text: str) -> pl.DataFrame:
        """Extract all electricity-like records from the preferred usable source."""
        source_path = self.paths.project_root / source_path_text
        frame = self._read_source(source_path)
        if frame.is_empty():
            return pl.DataFrame()
        profile = self._column_profile(frame.columns)
        prepared = self._normalize_source_frame(frame, profile)
        electricity = prepared.filter(self._electricity_expr()).sort("country_sector", "year")
        if electricity.is_empty():
            return electricity
        return self._derive_series_fields(electricity.with_columns(pl.lit(self._rel(source_path)).alias("source_path")))

    def compare_sources(self, series: pl.DataFrame) -> pl.DataFrame:
        """Compare China electricity variables across source pairs by year."""
        sources = sorted(series["source_path"].unique().to_list())
        variables = ["X_candidate", "emissions_candidate", "EI_candidate", "EI_recomputed"]
        rows: list[dict[str, Any]] = []
        for year in sorted(series["year"].drop_nulls().unique().to_list()):
            year_frame = series.filter(pl.col("year") == year)
            for variable in variables:
                for i, source_a in enumerate(sources):
                    for source_b in sources[i + 1 :]:
                        value_a = self._source_year_value(year_frame, source_a, variable)
                        value_b = self._source_year_value(year_frame, source_b, variable)
                        rows.append(self._comparison_row(year, variable, source_a, source_b, value_a, value_b))
        return pl.DataFrame(rows) if rows else self._empty_cross_source_comparison()

    def audit_units_and_scaling(self, series: pl.DataFrame) -> pl.DataFrame:
        """Check for stable source-to-source scale factors."""
        sources = sorted(series["source_path"].unique().to_list())
        variables = ["X_candidate", "emissions_candidate", "EI_recomputed"]
        rows = []
        for variable in variables:
            for i, source in enumerate(sources):
                for comparison_source in sources[i + 1 :]:
                    joined = (
                        series.filter(pl.col("source_path") == source)
                        .select("year", pl.col(variable).alias("value"))
                        .join(
                            series.filter(pl.col("source_path") == comparison_source).select(
                                "year", pl.col(variable).alias("comparison_value")
                            ),
                            on="year",
                            how="inner",
                        )
                        .filter((pl.col("value").is_not_null()) & (pl.col("comparison_value").is_not_null()) & (pl.col("comparison_value") != 0))
                        .with_columns((pl.col("value") / pl.col("comparison_value")).alias("ratio"))
                    )
                    if joined.is_empty():
                        ratio = p25 = p75 = float("nan")
                    else:
                        ratio = _as_float(joined["ratio"].median(), float("nan"))
                        p25 = _as_float(joined["ratio"].quantile(0.25), float("nan"))
                        p75 = _as_float(joined["ratio"].quantile(0.75), float("nan"))
                    factor = self._possible_scale_factor(ratio)
                    rows.append(
                        {
                            "source_path": source,
                            "variable": variable,
                            "comparison_source": comparison_source,
                            "median_ratio": ratio,
                            "p25_ratio": p25,
                            "p75_ratio": p75,
                            "possible_scale_factor": factor,
                            "scale_flag": factor != "none" and factor != "1",
                            "notes": "stable scale factor candidate" if factor not in {"none", "1"} else "no scale issue detected",
                        }
                    )
        return pl.DataFrame(rows)

    def audit_mapping_consistency(self, series: pl.DataFrame) -> pl.DataFrame:
        """Check missing years, duplicate source-year records, and mapping consistency."""
        rows: list[dict[str, Any]] = []
        for source in sorted(series["source_path"].unique().to_list()):
            source_frame = series.filter(pl.col("source_path") == source)
            counts = source_frame.group_by("year").len().rename({"len": "duplicate_records_count"})
            years_present = set(source_frame["year"].drop_nulls().to_list())
            if years_present:
                expected_years = set(range(int(min(years_present)), int(max(years_present)) + 1))
            else:
                expected_years = set()
            missing_years = sorted(expected_years - years_present)
            for row in counts.to_dicts():
                duplicate_count = int(row["duplicate_records_count"])
                year_frame = source_frame.filter(pl.col("year") == row["year"])
                country_value = str(year_frame["country_value"].drop_nulls().head(1).item()) if year_frame["country_value"].drop_nulls().len() else ""
                sector_value = str(year_frame["sector_value"].drop_nulls().head(1).item()) if year_frame["sector_value"].drop_nulls().len() else ""
                status = "duplicate_records" if duplicate_count > 1 else "consistent"
                if not self._is_china_text(country_value + " " + year_frame["country_sector"].head(1).item()):
                    status = "country_mismatch"
                elif not self._is_electricity_text(sector_value + " " + year_frame["country_sector"].head(1).item()):
                    status = "sector_mismatch"
                rows.append(
                    {
                        "source_path": source,
                        "year": row["year"],
                        "country_value": country_value,
                        "sector_value": sector_value,
                        "country_sector": year_frame["country_sector"].head(1).item(),
                        "mapped_country": "CHN",
                        "mapped_sector": "Electricity, Gas and Water",
                        "mapping_status": status,
                        "duplicate_records_count": duplicate_count,
                        "missing_records_count": 0,
                        "notes": "multiple records for source-year" if duplicate_count > 1 else "",
                    }
                )
            for year in missing_years:
                rows.append(
                    {
                        "source_path": source,
                        "year": year,
                        "country_value": None,
                        "sector_value": None,
                        "country_sector": None,
                        "mapped_country": "CHN",
                        "mapped_sector": "Electricity, Gas and Water",
                        "mapping_status": "missing_year",
                        "duplicate_records_count": 0,
                        "missing_records_count": 1,
                        "notes": "expected audit year missing from source extraction",
                    }
                )
        return pl.DataFrame(rows).sort("source_path", "year") if rows else self._empty_mapping_audit()

    def detect_breakpoints_and_jumps(self, electricity: pl.DataFrame) -> pl.DataFrame:
        """Flag large transparent jumps for major electricity nodes."""
        if electricity.is_empty():
            return self._empty_breakpoint_audit()
        top_nodes = (
            electricity.group_by("country_sector")
            .agg(pl.col("emissions_candidate").sum().alias("_emissions"))
            .sort("_emissions", descending=True)
            .head(12)
            .select("country_sector")
        )
        focus = electricity.join(top_nodes, on="country_sector", how="inner").sort("country_sector", "year")
        focus = focus.with_columns(
            (pl.col("X_candidate").log() - pl.col("X_candidate").log().shift(1).over("country_sector")).alias("_X_log_change"),
            (pl.col("emissions_candidate").log() - pl.col("emissions_candidate").log().shift(1).over("country_sector")).alias("_emissions_log_change"),
            (pl.col("EI_recomputed").log() - pl.col("EI_recomputed").log().shift(1).over("country_sector")).alias("_EI_log_change"),
        )
        frames = []
        variable_specs = [
            ("X", "X_candidate", "_X_log_change"),
            ("emissions", "emissions_candidate", "_emissions_log_change"),
            ("EI", "EI_recomputed", "_EI_log_change"),
            ("log_EI", "log_EI_recomputed", "_EI_log_change"),
            ("rEI", "observed_rEI_recomputed", "observed_rEI_recomputed"),
        ]
        for variable, value_column, change_column in variable_specs:
            threshold = _as_float(focus[change_column].abs().drop_nulls().quantile(0.95), 0.0)
            frames.append(
                focus.select(
                    "country_sector",
                    pl.col("country_value").alias("Country"),
                    pl.col("sector_value").alias("Sector"),
                    "year",
                    pl.lit(variable).alias("variable"),
                    pl.col(value_column).alias("value"),
                    pl.col(change_column).alias("log_change"),
                    pl.when(pl.col(value_column).shift(1).over("country_sector") != 0)
                    .then(pl.col(value_column) / pl.col(value_column).shift(1).over("country_sector") - 1.0)
                    .otherwise(None)
                    .alias("pct_change"),
                    (pl.col(change_column).abs().rank(method="average").over("year") / pl.len().over("year")).alias(
                        "sector_percentile_abs_change"
                    ),
                    (pl.col(change_column).abs() >= threshold).fill_null(False).alias("jump_flag"),
                    pl.when(pl.col(change_column).abs() >= threshold)
                    .then(pl.lit("large"))
                    .otherwise(pl.lit("normal"))
                    .alias("jump_severity"),
                    pl.when(pl.col(change_column).abs() >= threshold)
                    .then(pl.lit("top electricity-node jump by transparent threshold"))
                    .otherwise(pl.lit(""))
                    .alias("notes"),
                )
            )
        return pl.concat(frames, how="vertical_relaxed").sort("country_sector", "year", "variable")

    def build_major_electricity_comparison(self, electricity: pl.DataFrame, breakpoint: pl.DataFrame) -> pl.DataFrame:
        """Compare China with other large electricity nodes in the preferred source."""
        if electricity.is_empty():
            return self._empty_major_comparison()
        top_nodes = (
            electricity.group_by("country_sector")
            .agg(pl.col("emissions_candidate").sum().alias("_total_emissions"))
            .sort("_total_emissions", descending=True)
            .head(12)
            .select("country_sector")
        )
        jump_counts = (
            breakpoint.filter(pl.col("jump_flag"))
            .group_by("country_sector", "year")
            .len()
            .rename({"len": "jump_flags_count"})
        )
        return (
            electricity.join(top_nodes, on="country_sector", how="inner")
            .join(jump_counts, on=["country_sector", "year"], how="left")
            .with_columns(
                pl.col("jump_flags_count").fill_null(0),
                pl.col("X_candidate").rank(method="ordinal", descending=True).over("year").alias("X_rank_within_electricity"),
                pl.col("emissions_candidate").rank(method="ordinal", descending=True).over("year").alias(
                    "emissions_rank_within_electricity"
                ),
                (pl.col("EI_recomputed").rank(method="average").over("year") / pl.len().over("year")).alias(
                    "EI_percentile_within_electricity"
                ),
                (pl.col("observed_rEI_recomputed").rank(method="average").over("year") / pl.len().over("year")).alias(
                    "rEI_percentile_within_electricity"
                ),
                pl.when(pl.col("jump_flags_count") > 0)
                .then(pl.lit("jump flagged"))
                .otherwise(pl.lit(""))
                .alias("notes"),
            )
            .select(
                "country_sector",
                pl.col("country_value").alias("Country"),
                "year",
                pl.col("X_candidate").alias("X"),
                pl.col("emissions_candidate").alias("emissions"),
                pl.col("EI_recomputed").alias("EI"),
                pl.col("observed_rEI_recomputed").alias("observed_rEI"),
                "X_rank_within_electricity",
                "emissions_rank_within_electricity",
                "EI_percentile_within_electricity",
                "rEI_percentile_within_electricity",
                "jump_flags_count",
                "notes",
            )
            .sort("year", "emissions_rank_within_electricity")
        )

    def build_recommendation(
        self,
        comparison: pl.DataFrame,
        scaling: pl.DataFrame,
        mapping: pl.DataFrame,
        breakpoint: pl.DataFrame,
        major: pl.DataFrame,
    ) -> pl.DataFrame:
        """Apply Phase 19 decision rules to audit outputs."""
        mismatch_share = self._share(comparison, pl.col("match_status") == "mismatch")
        scale_flags = scaling.filter(pl.col("scale_flag")).height if not scaling.is_empty() else 0
        mapping_problems = mapping.filter(pl.col("mapping_status").is_in(["duplicate_records", "ambiguous"])).height
        china_jumps = breakpoint.filter(pl.col("country_sector").str.contains("CHN") & pl.col("jump_flag")).height
        other_jump_nodes = (
            breakpoint.filter(~pl.col("country_sector").str.contains("CHN") & pl.col("jump_flag"))
            .select("country_sector")
            .n_unique()
            if not breakpoint.is_empty()
            else 0
        )
        china_dominates = False if major.is_empty() else _as_float(
            major.filter(pl.col("country_sector").str.contains("CHN"))["emissions_rank_within_electricity"].mean(),
            99.0,
        ) <= 2.0
        if mapping_problems > 0 or (mismatch_share > 0.2 and scale_flags == 0):
            action = "repair_mapping_or_scaling"
            interpretation = "Source lineage has mapping gaps, duplicates, or unexplained mismatches."
        elif scale_flags > 0 and mismatch_share > 0.2:
            action = "inspect_raw_eora_source_manually"
            interpretation = "Cross-source differences look scale-related and need manual source inspection."
        elif china_jumps >= 3 and other_jump_nodes >= 3:
            action = "treat_electricity_as_sector_specific_transition_case"
            interpretation = "China jumps are severe, but other major electricity nodes also show jumps."
        elif china_jumps >= 3 and china_dominates:
            action = "treat_china_electricity_as_historical_structural_break"
            interpretation = "China electricity is cleanly mapped but historically extreme and aggregate-important."
        elif mismatch_share <= 0.2:
            action = "proceed_to_validation_objective_design"
            interpretation = "Data path looks broadly consistent; next decision should be validation-objective design."
        else:
            action = "inconclusive"
            interpretation = "Evidence is mixed."
        return pl.DataFrame(
            [
                {
                    "finding": "raw_eora_electricity_data_path",
                    "evidence": (
                        f"mismatch_share={mismatch_share:.3f}; scale_flags={scale_flags}; "
                        f"mapping_problems={mapping_problems}; china_jumps={china_jumps}; "
                        f"other_jump_nodes={other_jump_nodes}"
                    ),
                    "interpretation": interpretation,
                    "recommended_next_action": action,
                },
                {
                    "finding": "scenario_readiness",
                    "evidence": "Phase 19 is diagnostic only; no final emissions-transition mechanism has been selected.",
                    "interpretation": "Scenarios remain premature.",
                    "recommended_next_action": "keep_frontier_gap_readiness_as_aggregate_safe_base",
                },
            ]
        )

    def build_markdown_report(
        self,
        *,
        inventory: pl.DataFrame,
        series: pl.DataFrame,
        comparison: pl.DataFrame,
        scaling: pl.DataFrame,
        mapping: pl.DataFrame,
        breakpoint: pl.DataFrame,
        major: pl.DataFrame,
        recommendation: pl.DataFrame,
    ) -> str:
        """Build the Phase 19 markdown audit report."""
        action = recommendation["recommended_next_action"].item(0) if not recommendation.is_empty() else "inconclusive"
        lines = [
            "# ABM v4 Phase 19 Raw Eora Electricity Data Path Audit",
            "",
            "This diagnostic traces China Electricity, Gas and Water through existing Eora-derived files.",
            "",
            "## Summary",
            "",
            f"- Candidate sources inspected: {inventory.height}.",
            f"- Usable China electricity sources: {inventory.filter(pl.col('usable_for_china_electricity')).height}.",
            f"- Extracted China electricity rows: {series.height}.",
            f"- Recommended next action: `{action}`.",
            "- No scenarios, recalibration, or hybrid rule were implemented.",
            "",
            "## Recommendation",
            "",
            self._markdown_table(recommendation),
            "",
            "## Source Inventory",
            "",
            self._markdown_table(inventory.select([c for c in inventory.columns if c != "columns"]).head(60)),
            "",
            "## China Electricity Series by Source",
            "",
            self._markdown_table(series.head(80)),
            "",
            "## Cross-Source Comparison",
            "",
            self._markdown_table(comparison.head(80)),
            "",
            "## Scaling Audit",
            "",
            self._markdown_table(scaling.head(80)),
            "",
            "## Mapping Audit",
            "",
            self._markdown_table(mapping.filter(pl.col("mapping_status") != "consistent").head(80)),
            "",
            "## Breakpoint and Jump Audit",
            "",
            self._markdown_table(breakpoint.filter(pl.col("jump_flag")).head(80)),
            "",
            "## Major Electricity Comparison",
            "",
            self._markdown_table(major.head(80)),
        ]
        return "\n".join(lines) + "\n"

    def _candidate_source_paths(self) -> list[Path]:
        roots = [
            self.paths.inputs,
            self.paths.interim,
            self.paths.simulations,
            self.paths.data_abm_v3 / "inputs",
            self.paths.data_final,
            self.paths.data_root / "processed",
        ]
        known = [
            self.paths.state_panel_path(self.start_year, self.end_year),
            self.paths.data_abm_v3 / "inputs" / "abm_v3_historical_panel_1995_2016_transpose_row_fd_without_inventory.parquet",
            self.paths.data_abm_v3 / "inputs" / "abm_v3_historical_panel_1995_2016.parquet",
            self.paths.data_final / "eora_atlas_merged.parquet",
            self.paths.data_final / "eora_atlas_dynamic_panel.parquet",
            self.paths.data_final / "eora_metrics_panel.parquet",
        ]
        paths: set[Path] = {path for path in known if path.exists()}
        for root in roots:
            if not root.exists():
                continue
            for suffix in ("*.parquet", "*.csv"):
                for path in root.rglob(suffix):
                    if "raw_T" in path.name or path.stat().st_size > 600_000_000:
                        continue
                    paths.add(path)
        return sorted(paths)

    def _read_source(self, path: Path) -> pl.DataFrame:
        if not path.exists():
            return pl.DataFrame()
        try:
            if path.suffix.lower() == ".parquet":
                return pl.read_parquet(path)
            if path.suffix.lower() == ".csv" and path.stat().st_size < 200_000_000:
                return pl.read_csv(path, infer_schema_length=200)
        except Exception:
            return pl.DataFrame()
        return pl.DataFrame()

    def _column_profile(self, columns: list[str]) -> dict[str, str | None]:
        lower = {column.lower(): column for column in columns}
        country = self._first_matching_column(lower, ["country", "country_code", "country_detail", "mapped_country"])
        sector = self._first_matching_column(lower, ["sector", "category", "mapped_sector"])
        year = self._first_matching_column(lower, ["year"])
        country_sector = self._first_contains_column(lower, ["country_sector", "node"])
        output = self._first_matching_column(lower, ["x_observed", "x", "output", "gross_output", "production"])
        emissions = self._first_contains_column(lower, ["emissions_observed", "emissions", "co2"])
        ei = self._first_matching_column(lower, ["ei_observed", "ei", "emissions_intensity"])
        return {
            "country_column": country,
            "sector_column": sector,
            "year_column": year,
            "country_sector_column": country_sector,
            "output_column": output,
            "emissions_column": emissions,
            "ei_column": ei,
        }

    def _normalize_source_frame(self, frame: pl.DataFrame, profile: dict[str, str | None]) -> pl.DataFrame:
        year_col = profile["year_column"]
        if year_col is None:
            return pl.DataFrame()
        out = frame.with_columns(pl.col(year_col).cast(pl.Int64, strict=False).alias("year"))
        country_col = profile["country_column"]
        sector_col = profile["sector_column"]
        cs_col = profile["country_sector_column"]
        if country_col:
            out = out.with_columns(pl.col(country_col).cast(pl.Utf8).alias("country_value"))
        else:
            out = out.with_columns(pl.lit(None).cast(pl.Utf8).alias("country_value"))
        if sector_col:
            out = out.with_columns(pl.col(sector_col).cast(pl.Utf8).alias("sector_value"))
        else:
            out = out.with_columns(pl.lit(None).cast(pl.Utf8).alias("sector_value"))
        if cs_col:
            out = out.with_columns(pl.col(cs_col).cast(pl.Utf8).alias("country_sector"))
        else:
            out = out.with_columns(
                (pl.col("country_value").fill_null("") + pl.lit(" | ") + pl.col("sector_value").fill_null("")).alias(
                    "country_sector"
                )
            )
        if not country_col and cs_col:
            out = out.with_columns(
                pl.col("country_sector").str.split("|").list.first().str.strip_chars().alias("country_value")
            )
        if not sector_col and cs_col:
            out = out.with_columns(
                pl.col("country_sector").str.split("|").list.last().str.strip_chars().alias("sector_value")
            )
        for source_name, target in [
            (profile["output_column"], "X_candidate"),
            (profile["emissions_column"], "emissions_candidate"),
            (profile["ei_column"], "EI_candidate"),
        ]:
            if source_name:
                out = out.with_columns(pl.col(source_name).cast(pl.Float64, strict=False).alias(target))
            else:
                out = out.with_columns(pl.lit(None).cast(pl.Float64).alias(target))
        return out.select(
            "year",
            "country_value",
            "sector_value",
            "country_sector",
            "X_candidate",
            "emissions_candidate",
            "EI_candidate",
        ).filter((pl.col("year") >= self.start_year) & (pl.col("year") <= self.end_year))

    def _derive_series_fields(self, frame: pl.DataFrame) -> pl.DataFrame:
        for fallback, target in [
            ("X_observed", "X_candidate"),
            ("emissions_observed", "emissions_candidate"),
            ("EI", "EI_candidate"),
            ("EI_observed", "EI_candidate"),
        ]:
            if target not in frame.columns and fallback in frame.columns:
                frame = frame.with_columns(pl.col(fallback).cast(pl.Float64, strict=False).alias(target))
        for target in ["X_candidate", "emissions_candidate", "EI_candidate"]:
            if target not in frame.columns:
                frame = frame.with_columns(pl.lit(None).cast(pl.Float64).alias(target))
        for target in ["source_path", "country_value", "sector_value", "country_sector"]:
            if target not in frame.columns:
                frame = frame.with_columns(pl.lit("").cast(pl.Utf8).alias(target))
        return (
            frame.with_columns(
                pl.when((pl.col("X_candidate") > 0) & pl.col("emissions_candidate").is_not_null())
                .then(pl.col("emissions_candidate") / pl.col("X_candidate"))
                .otherwise(pl.col("EI_candidate"))
                .alias("EI_recomputed"),
                pl.when(pl.col("X_candidate").is_not_null()).then(pl.lit("extracted")).otherwise(pl.lit("missing X")).alias(
                    "notes"
                ),
            )
            .sort("source_path", "country_sector", "year")
            .with_columns(
                pl.when(pl.col("EI_recomputed") > 0).then(pl.col("EI_recomputed").log()).otherwise(None).alias(
                    "log_EI_recomputed"
                ),
                (pl.col("X_candidate") / pl.col("X_candidate").shift(1).over("source_path", "country_sector") - 1.0).alias(
                    "X_pct_change"
                ),
                (
                    pl.col("emissions_candidate")
                    / pl.col("emissions_candidate").shift(1).over("source_path", "country_sector")
                    - 1.0
                ).alias("emissions_pct_change"),
                (
                    pl.col("EI_recomputed") / pl.col("EI_recomputed").shift(1).over("source_path", "country_sector")
                    - 1.0
                ).alias("EI_pct_change"),
                (
                    pl.col("EI_recomputed").log()
                    - pl.col("EI_recomputed").log().shift(1).over("source_path", "country_sector")
                ).alias("observed_rEI_recomputed"),
                pl.lit("country_value").alias("country_column"),
                pl.lit("sector_value").alias("sector_column"),
            )
            .select(
                "source_path",
                "year",
                "country_column",
                "country_value",
                "sector_column",
                "sector_value",
                "country_sector",
                "X_candidate",
                "emissions_candidate",
                "EI_candidate",
                "EI_recomputed",
                "log_EI_recomputed",
                "X_pct_change",
                "emissions_pct_change",
                "EI_pct_change",
                "observed_rEI_recomputed",
                "notes",
            )
        )

    def _source_year_value(self, frame: pl.DataFrame, source: str, variable: str) -> float | None:
        value_frame = frame.filter(pl.col("source_path") == source)
        if value_frame.is_empty() or variable not in value_frame.columns:
            return None
        values = value_frame[variable].drop_nulls()
        return None if values.is_empty() else _as_float(values.mean(), float("nan"))

    def _comparison_row(
        self,
        year: int,
        variable: str,
        source_a: str,
        source_b: str,
        value_a: float | None,
        value_b: float | None,
    ) -> dict[str, Any]:
        if value_a is None:
            status = "missing_in_a"
            diff = rel = None
        elif value_b is None:
            status = "missing_in_b"
            diff = rel = None
        else:
            diff = abs(value_a - value_b)
            rel = diff / max(abs(value_a), abs(value_b), 1e-12)
            ratio = value_a / value_b if value_b else float("nan")
            if rel <= 1e-8:
                status = "exact_or_near_match"
            elif self._possible_scale_factor(ratio) not in {"none", "1"}:
                status = "scale_difference_possible"
            else:
                status = "mismatch"
        return {
            "year": year,
            "variable": variable,
            "source_a": source_a,
            "source_b": source_b,
            "value_a": value_a,
            "value_b": value_b,
            "absolute_difference": diff,
            "relative_difference": rel,
            "match_status": status,
            "notes": "transparent source-pair comparison",
        }

    def _preferred_source_path(self, series: pl.DataFrame) -> str:
        preferred = self._rel(self.paths.state_panel_path(self.start_year, self.end_year))
        if preferred in series["source_path"].unique().to_list():
            return preferred
        return (
            series.group_by("source_path")
            .len()
            .sort("len", descending=True)
            .head(1)["source_path"]
            .item()
        )

    def _possible_scale_factor(self, ratio: float) -> str:
        if ratio is None or math.isnan(ratio) or ratio == 0:
            return "none"
        for factor in [1.0, 1e3, 1e-3, 1e6, 1e-6, 1e9, 1e-9]:
            if abs(ratio / factor - 1.0) <= 0.02:
                return f"{factor:g}"
        return "none"

    def _first_matching_column(self, lower_columns: dict[str, str], names: list[str]) -> str | None:
        for name in names:
            if name in lower_columns:
                return lower_columns[name]
        return None

    def _first_contains_column(self, lower_columns: dict[str, str], fragments: list[str]) -> str | None:
        for key, original in lower_columns.items():
            if any(fragment in key for fragment in fragments):
                return original
        return None

    def _electricity_expr(self) -> pl.Expr:
        text = (pl.col("sector_value").fill_null("") + pl.lit(" ") + pl.col("country_sector").fill_null("")).str.to_lowercase()
        return text.str.contains("electricity|gas and water|utilities|power|water|gas")

    def _china_expr(self) -> pl.Expr:
        text = (pl.col("country_value").fill_null("") + pl.lit(" ") + pl.col("country_sector").fill_null("")).str.to_lowercase()
        return text.str.contains("china|chn|people's republic of china")

    def _is_china_text(self, text: str) -> bool:
        return any(token in text.lower() for token in ["china", "chn", "people's republic of china"])

    def _is_electricity_text(self, text: str) -> bool:
        return any(token in text.lower() for token in ["electricity", "gas and water", "utilities", "power", "water", "gas"])

    def _share(self, frame: pl.DataFrame, expr: pl.Expr) -> float:
        if frame.is_empty():
            return 0.0
        return _as_float(frame.select(expr.mean()).item())

    def _rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.paths.project_root))
        except ValueError:
            return str(path)

    def _markdown_table(self, frame: pl.DataFrame) -> str:
        if frame.is_empty():
            return "_No rows available._"
        columns = frame.columns
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
        for row in frame.to_dicts():
            lines.append("| " + " | ".join(self._format_markdown_value(row.get(column)) for column in columns) + " |")
        return "\n".join(lines)

    def _format_markdown_value(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        if value is None:
            return ""
        return str(value).replace("|", "/")

    def _empty_china_series(self) -> pl.DataFrame:
        return pl.DataFrame(schema={column: pl.Utf8 for column in ["source_path", "country_column", "country_value", "sector_column", "sector_value", "country_sector", "notes"]}).with_columns(
            pl.lit(None).cast(pl.Int64).alias("year"),
            pl.lit(None).cast(pl.Float64).alias("X_candidate"),
            pl.lit(None).cast(pl.Float64).alias("emissions_candidate"),
            pl.lit(None).cast(pl.Float64).alias("EI_candidate"),
            pl.lit(None).cast(pl.Float64).alias("EI_recomputed"),
            pl.lit(None).cast(pl.Float64).alias("log_EI_recomputed"),
            pl.lit(None).cast(pl.Float64).alias("X_pct_change"),
            pl.lit(None).cast(pl.Float64).alias("emissions_pct_change"),
            pl.lit(None).cast(pl.Float64).alias("EI_pct_change"),
            pl.lit(None).cast(pl.Float64).alias("observed_rEI_recomputed"),
        ).head(0)

    def _empty_cross_source_comparison(self) -> pl.DataFrame:
        return pl.DataFrame(schema={"year": pl.Int64, "variable": pl.Utf8, "source_a": pl.Utf8, "source_b": pl.Utf8, "value_a": pl.Float64, "value_b": pl.Float64, "absolute_difference": pl.Float64, "relative_difference": pl.Float64, "match_status": pl.Utf8, "notes": pl.Utf8})

    def _empty_mapping_audit(self) -> pl.DataFrame:
        return pl.DataFrame(schema={"source_path": pl.Utf8, "year": pl.Int64, "country_value": pl.Utf8, "sector_value": pl.Utf8, "country_sector": pl.Utf8, "mapped_country": pl.Utf8, "mapped_sector": pl.Utf8, "mapping_status": pl.Utf8, "duplicate_records_count": pl.Int64, "missing_records_count": pl.Int64, "notes": pl.Utf8})

    def _empty_breakpoint_audit(self) -> pl.DataFrame:
        return pl.DataFrame(schema={"country_sector": pl.Utf8, "Country": pl.Utf8, "Sector": pl.Utf8, "year": pl.Int64, "variable": pl.Utf8, "value": pl.Float64, "log_change": pl.Float64, "pct_change": pl.Float64, "sector_percentile_abs_change": pl.Float64, "jump_flag": pl.Boolean, "jump_severity": pl.Utf8, "notes": pl.Utf8})

    def _empty_major_comparison(self) -> pl.DataFrame:
        return pl.DataFrame(schema={"country_sector": pl.Utf8, "Country": pl.Utf8, "year": pl.Int64, "X": pl.Float64, "emissions": pl.Float64, "EI": pl.Float64, "observed_rEI": pl.Float64, "X_rank_within_electricity": pl.Float64, "emissions_rank_within_electricity": pl.Float64, "EI_percentile_within_electricity": pl.Float64, "rEI_percentile_within_electricity": pl.Float64, "jump_flags_count": pl.Int64, "notes": pl.Utf8})


class ElectricityTransitionRegimeDiagnostics:
    """Compare diagnostic electricity-specific transition rule candidates."""

    def __init__(self, paths: ABMV4Paths, start_year: int = 1995, end_year: int = 2016) -> None:
        self.paths = paths
        self.start_year = start_year
        self.end_year = end_year

    def run(self) -> ElectricityTransitionRegimeDiagnosticResult:
        """Build all Phase 20 diagnostic outputs in memory."""
        self._require_phase19_outputs()
        observed = self.load_observed_state()
        readiness, frontier = self.load_existing_model_outputs()
        panel = self.build_electricity_transition_panel(observed, readiness, frontier)
        targets = self.compute_electricity_targets(panel)
        rule_panel = self.evaluate_candidate_rules(panel)
        comparison = self.build_rule_comparison(rule_panel)
        by_country = self.summarize_by_country(rule_panel)
        by_year = self.summarize_by_year(rule_panel)
        by_decile = self.summarize_by_decile(rule_panel)
        by_jump = self.summarize_by_jump_status(rule_panel)
        china = self.summarize_china_electricity(rule_panel)
        recommendation = self.build_recommendation(comparison)
        markdown = self.build_markdown_report(
            target_diagnostics=targets,
            rule_comparison=comparison,
            by_country=by_country,
            by_year=by_year,
            by_decile=by_decile,
            by_jump_status=by_jump,
            china_comparison=china,
            recommendation=recommendation,
            node_count=panel["country_sector"].n_unique(),
            row_count=panel.height,
        )
        return ElectricityTransitionRegimeDiagnosticResult(
            target_diagnostics=targets,
            rule_comparison=comparison,
            by_country=by_country,
            by_year=by_year,
            by_decile=by_decile,
            by_jump_status=by_jump,
            china_comparison=china,
            recommendation=recommendation,
            markdown=markdown,
        )

    def write_outputs(self, result: ElectricityTransitionRegimeDiagnosticResult) -> None:
        """Write Phase 20 outputs. The caller controls whether this is invoked."""
        self.paths.validation.mkdir(parents=True, exist_ok=True)
        result.target_diagnostics.write_csv(self.paths.electricity_transition_target_diagnostics_path)
        result.rule_comparison.write_csv(self.paths.electricity_transition_rule_comparison_path)
        result.by_country.write_csv(self.paths.electricity_transition_rule_by_country_path)
        result.by_year.write_csv(self.paths.electricity_transition_rule_by_year_path)
        result.by_decile.write_csv(self.paths.electricity_transition_rule_by_decile_path)
        result.by_jump_status.write_csv(self.paths.electricity_transition_rule_by_jump_status_path)
        result.china_comparison.write_csv(self.paths.china_electricity_rule_comparison_path)
        result.recommendation.write_csv(self.paths.electricity_transition_regime_recommendation_path)
        self.paths.electricity_transition_regime_report_path.write_text(result.markdown, encoding="utf-8")

    def load_observed_state(self) -> pl.DataFrame:
        """Load and standardize the ABM v4 observed state panel."""
        path = self.paths.state_panel_path(self.start_year, self.end_year)
        if not path.exists():
            raise FileNotFoundError(f"Missing observed state panel: {path}. Run --build-state first.")
        frame = pl.read_parquet(path)
        if "Year" in frame.columns and "year" not in frame.columns:
            frame = frame.rename({"Year": "year"})
        if "EI_observed" not in frame.columns and "EI" in frame.columns:
            frame = frame.rename({"EI": "EI_observed"})
        keep = [
            c
            for c in [
                "country_sector",
                "year",
                "Country",
                "Sector",
                "X_observed",
                "EI_observed",
                "emissions_observed",
                "readiness",
                "general_capability_source",
                "green_capability_source",
            ]
            if c in frame.columns
        ]
        return frame.select(keep)

    def load_existing_model_outputs(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Load default readiness and historical frontier-gap multi-year outputs."""
        readiness_path = self.paths.base_multiyear_state_panel_path
        frontier_path = self.paths.base_multiyear_state_panel_historical_frontier_gap_path
        missing = [p for p in [readiness_path, frontier_path] if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing existing model outputs for Phase 20: {missing}. Run Phase 15 bases first.")
        return self._model_output(pl.read_parquet(readiness_path), "readiness"), self._model_output(
            pl.read_parquet(frontier_path), "frontier_gap"
        )

    def build_electricity_transition_panel(
        self,
        observed: pl.DataFrame,
        readiness: pl.DataFrame,
        frontier: pl.DataFrame,
    ) -> pl.DataFrame:
        """Create one row per electricity node-year transition from t to t+1."""
        electricity = observed.filter(self._electricity_expr()).sort("country_sector", "year")
        electricity = electricity.with_columns(
            pl.when(pl.col("EI_observed") > 0).then(pl.col("EI_observed").log()).otherwise(None).alias("log_EI"),
            pl.col("EI_observed").shift(-1).over("country_sector").alias("EI_next"),
            pl.col("X_observed").shift(-1).over("country_sector").alias("X_next"),
            pl.col("emissions_observed").shift(-1).over("country_sector").alias("emissions_next"),
            pl.col("year").shift(-1).over("country_sector").alias("target_year"),
        ).with_columns(
            pl.when(pl.col("EI_next") > 0).then(pl.col("EI_next").log()).otherwise(None).alias("log_EI_next"),
            pl.col("log_EI").rolling_mean(window_size=3, min_samples=1).over("country_sector").alias("_smooth_log_EI"),
        )
        sector_year = (
            electricity.filter(pl.col("EI_observed") > 0)
            .group_by("Sector", "year")
            .agg(pl.col("EI_observed").quantile(0.50, interpolation="nearest").alias("sector_year_p50"))
            .sort("Sector", "year")
            .with_columns(pl.col("sector_year_p50").cum_min().over("Sector").alias("rolling_sector_p50"))
        )
        jumps = self._jump_counts().rename({"year": "target_year"})
        panel = (
            electricity.join(sector_year, on=["Sector", "year"], how="left")
            .join(readiness, on=["country_sector", "year"], how="left")
            .join(frontier, on=["country_sector", "year"], how="left")
            .join(jumps, on=["country_sector", "target_year"], how="left")
            .with_columns(
                (pl.col("log_EI") - pl.col("log_EI_next")).alias("one_year_rEI"),
                (pl.col("_smooth_log_EI") - pl.col("_smooth_log_EI").shift(-1).over("country_sector")).alias(
                    "smoothed_one_year_rEI"
                ),
                ((pl.col("log_EI") - pl.col("log_EI").shift(-3).over("country_sector")) / 3.0).alias(
                    "three_year_annualized_rEI"
                ),
                pl.max_horizontal(pl.lit(0.0), pl.col("log_EI") - pl.col("rolling_sector_p50").log()).alias(
                    "frontier_gap"
                ),
                pl.col("jump_flag_count").fill_null(0).cast(pl.Int64),
            )
            .filter(pl.col("target_year") == pl.col("year") + 1)
        )
        p05 = _as_float(panel["one_year_rEI"].drop_nulls().quantile(0.05), float("nan"))
        p95 = _as_float(panel["one_year_rEI"].drop_nulls().quantile(0.95), float("nan"))
        panel = panel.with_columns(
            pl.col("one_year_rEI").clip(p05, p95).alias("winsorized_one_year_rEI"),
            (
                (pl.col("X_observed").rank(method="ordinal").over("year") * 10 / pl.len().over("year"))
                .ceil()
                .clip(1, 10)
                .cast(pl.Int64)
            ).alias("_output_decile_num"),
            (
                (pl.col("emissions_observed").rank(method="ordinal").over("year") * 10 / pl.len().over("year"))
                .ceil()
                .clip(1, 10)
                .cast(pl.Int64)
            ).alias("_emissions_decile_num"),
            (pl.col("emissions_observed").rank(method="ordinal", descending=True).over("year") <= 20).alias(
                "emissions_rank_top20"
            ),
        ).with_columns(
            (pl.lit("d") + pl.col("_output_decile_num").cast(pl.Utf8)).alias("output_decile"),
            (pl.lit("d") + pl.col("_emissions_decile_num").cast(pl.Utf8)).alias("emissions_decile"),
        )
        background = panel.group_by("Sector", "year").agg(pl.col("one_year_rEI").mean().alias("electricity_background"))
        return panel.join(background, on=["Sector", "year"], how="left").filter(pl.col("one_year_rEI").is_not_null())

    def compute_electricity_targets(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Summarize target volatility and composition."""
        rows = []
        for target in [
            "one_year_rEI",
            "smoothed_one_year_rEI",
            "three_year_annualized_rEI",
            "winsorized_one_year_rEI",
        ]:
            valid = panel.filter(pl.col(target).is_not_null())
            total_emissions = valid["emissions_observed"].sum() or 1.0
            rows.append(
                {
                    "target_name": target,
                    "rows": valid.height,
                    "mean": _as_float(valid[target].mean()),
                    "median": _as_float(valid[target].median()),
                    "std": _as_float(valid[target].std()),
                    "p05": _as_float(valid[target].quantile(0.05)),
                    "p95": _as_float(valid[target].quantile(0.95)),
                    "share_positive": self._share(valid, pl.col(target) > 0),
                    "share_negative": self._share(valid, pl.col(target) < 0),
                    "jump_flag_share": self._share(valid, pl.col("jump_flag_count") > 0),
                    "china_row_share": self._share(valid, self._china_expr()),
                    "china_emissions_share": _as_float(
                        valid.filter(self._china_expr())["emissions_observed"].sum() / total_emissions
                    ),
                    "recommended_use": "primary" if target == "one_year_rEI" else "diagnostic sensitivity",
                }
            )
        return pl.DataFrame(rows)

    def evaluate_candidate_rules(self, panel: pl.DataFrame, target_name: str = "one_year_rEI") -> pl.DataFrame:
        """Create long panel of diagnostic candidate-rule predictions."""
        base = panel.with_columns(
            pl.col(target_name).alias("observed_rEI"),
            pl.when(pl.col("readiness").is_null()).then(0.5).otherwise(pl.col("readiness")).alias("_readiness_fill"),
        )
        rho = 0.03
        tau = 1.0
        gap_term = rho * pl.col("frontier_gap") / (pl.col("frontier_gap") + tau)
        rule_exprs = [
            ("current_frontier_gap_readiness_reference", pl.col("simulated_rEI_readiness"), "existing previous base rule"),
            ("historical_frontier_gap_only_reference", pl.col("simulated_rEI_frontier_gap"), "existing historical rule"),
            ("electricity_sector_background_only", pl.col("electricity_background"), "electricity sector-time background"),
            ("electricity_rolling_frontier_gap_only", pl.col("electricity_background") + gap_term, "rolling p50 gap only"),
            ("electricity_dampened_frontier_gap_0_25", pl.col("electricity_background") + pl.lit(0.25) * gap_term, "fixed dampener 0.25"),
            ("electricity_dampened_frontier_gap_0_50", pl.col("electricity_background") + pl.lit(0.50) * gap_term, "fixed dampener 0.50"),
            ("electricity_dampened_frontier_gap_0_75", pl.col("electricity_background") + pl.lit(0.75) * gap_term, "fixed dampener 0.75"),
            (
                "electricity_readiness_dampened_frontier_gap",
                pl.col("electricity_background") + pl.col("_readiness_fill").clip(0.25, 0.75) * gap_term,
                "bounded readiness dampener",
            ),
            (
                "electricity_gap_with_jump_shock_filter",
                pl.when(pl.col("jump_flag_count") > 0)
                .then(pl.col("electricity_background"))
                .otherwise(pl.col("electricity_background") + gap_term),
                "gap closure only in non-jump years",
            ),
            (
                "electricity_high_emissions_dampened_gap",
                pl.col("electricity_background")
                + pl.when(pl.col("emissions_decile").cast(pl.Utf8).is_in(["d9", "d10"]))
                .then(pl.lit(0.25))
                .otherwise(pl.lit(0.75))
                * gap_term,
                "top-emissions nodes dampened more strongly",
            ),
        ]
        frames = []
        for name, expr, notes in rule_exprs:
            frames.append(
                base.with_columns(
                    pl.lit(name).alias("rule_name"),
                    pl.lit(target_name).alias("target_name"),
                    expr.alias("simulated_rEI"),
                    pl.lit(notes).alias("notes"),
                )
                .with_columns(
                    (pl.col("observed_rEI") - pl.col("simulated_rEI")).alias("rEI_error"),
                    (pl.col("observed_rEI") - pl.col("simulated_rEI")).abs().alias("rEI_abs_error"),
                    (
                        (pl.col("observed_rEI").sign() != pl.col("simulated_rEI").sign())
                        & (pl.col("observed_rEI") != 0)
                        & (pl.col("simulated_rEI") != 0)
                    ).alias("wrong_sign"),
                    (pl.col("EI_observed") * (-pl.col("simulated_rEI")).exp()).alias("simulated_EI"),
                    (pl.col("X_next") * pl.col("EI_observed") * (-pl.col("simulated_rEI")).exp()).alias(
                        "simulated_emissions"
                    ),
                )
                .with_columns((pl.col("simulated_emissions") - pl.col("emissions_next")).alias("emissions_error"))
            )
        return pl.concat(frames, how="vertical_relaxed").filter(pl.col("simulated_rEI").is_not_null())

    def build_rule_comparison(self, rule_panel: pl.DataFrame) -> pl.DataFrame:
        """Build top-level electricity candidate-rule metrics."""
        return (
            rule_panel.group_by("rule_name", "target_name", "notes")
            .agg(
                pl.len().alias("rows"),
                pl.col("rEI_abs_error").mean().alias("unweighted_rEI_MAE"),
                self._weighted_expr("rEI_abs_error", "X_observed").alias("output_weighted_rEI_MAE"),
                self._weighted_expr("rEI_abs_error", "emissions_observed").alias("emissions_weighted_rEI_MAE"),
                pl.col("wrong_sign").mean().alias("unweighted_wrong_sign_share"),
                self._weighted_expr("wrong_sign", "emissions_observed").alias("emissions_weighted_wrong_sign_share"),
                pl.corr("observed_rEI", "simulated_rEI").alias("validation_correlation"),
                pl.col("rEI_error").mean().alias("validation_bias"),
                pl.col("emissions_error").abs().sum().alias("electricity_aggregate_emissions_error"),
                pl.when(self._china_expr()).then(pl.col("emissions_error").abs()).otherwise(0.0).sum().alias(
                    "china_electricity_emissions_error"
                ),
                pl.when(pl.col("emissions_rank_top20")).then(pl.col("emissions_error").abs()).otherwise(0.0).sum().alias(
                    "top20_electricity_node_year_emissions_error"
                ),
                self._latest_pct_error_expr().alias("latest_year_electricity_emissions_pct_error"),
                self._mean_yearly_pct_error_expr().alias("mean_yearly_electricity_emissions_pct_error"),
                pl.when(pl.col("jump_flag_count") > 0).then(pl.col("rEI_abs_error")).otherwise(None).mean().alias(
                    "jump_year_rEI_MAE"
                ),
                pl.when(pl.col("jump_flag_count") == 0).then(pl.col("rEI_abs_error")).otherwise(None).mean().alias(
                    "nonjump_year_rEI_MAE"
                ),
                pl.when(pl.col("jump_flag_count") > 0).then(pl.col("wrong_sign")).otherwise(None).mean().alias(
                    "jump_year_wrong_sign_share"
                ),
                pl.when(pl.col("jump_flag_count") == 0).then(pl.col("wrong_sign")).otherwise(None).mean().alias(
                    "nonjump_year_wrong_sign_share"
                ),
            )
            .sort("electricity_aggregate_emissions_error")
        )

    def summarize_by_country(self, rule_panel: pl.DataFrame) -> pl.DataFrame:
        total = rule_panel.select("country_sector", "year", "emissions_observed").unique()["emissions_observed"].sum() or 1.0
        return (
            rule_panel.group_by("rule_name", "Country")
            .agg(
                pl.len().alias("rows"),
                (pl.col("emissions_observed").sum() / total).alias("observed_emissions_share"),
                pl.col("rEI_abs_error").mean().alias("unweighted_rEI_MAE"),
                self._weighted_expr("rEI_abs_error", "emissions_observed").alias("emissions_weighted_rEI_MAE"),
                pl.col("wrong_sign").mean().alias("wrong_sign_share"),
                pl.col("emissions_error").abs().sum().alias("emissions_error"),
                (pl.col("jump_flag_count") > 0).mean().alias("jump_flag_share"),
            )
            .with_columns(
                pl.when(pl.col("emissions_error") > pl.col("emissions_error").median().over("rule_name"))
                .then(pl.lit("high electricity aggregate-error contributor"))
                .otherwise(pl.lit("lower contributor"))
                .alias("recommended_interpretation")
            )
        )

    def summarize_by_year(self, rule_panel: pl.DataFrame) -> pl.DataFrame:
        return (
            rule_panel.group_by("rule_name", "target_year")
            .agg(
                pl.len().alias("rows"),
                pl.col("emissions_next").sum().alias("observed_emissions"),
                pl.col("simulated_emissions").sum().alias("simulated_emissions"),
                pl.col("emissions_error").sum().alias("emissions_error"),
                pl.col("rEI_abs_error").mean().alias("rEI_MAE"),
                pl.col("wrong_sign").mean().alias("wrong_sign_share"),
                pl.col("jump_flag_count").sum().alias("jump_flag_count"),
            )
            .rename({"target_year": "year"})
            .with_columns(
                pl.when(pl.col("jump_flag_count") > 0)
                .then(pl.lit("jump-year affected"))
                .otherwise(pl.lit("non-jump year"))
                .alias("recommended_interpretation")
            )
        )

    def summarize_by_decile(self, rule_panel: pl.DataFrame) -> pl.DataFrame:
        frames = []
        total = rule_panel.select("country_sector", "year", "emissions_observed").unique()["emissions_observed"].sum() or 1.0
        for decile_type, column in [("output_decile", "output_decile"), ("emissions_decile", "emissions_decile")]:
            frames.append(
                rule_panel.group_by("rule_name", column)
                .agg(
                    pl.len().alias("rows"),
                    (pl.col("emissions_observed").sum() / total).alias("observed_emissions_share"),
                    pl.col("rEI_abs_error").mean().alias("rEI_MAE"),
                    pl.col("wrong_sign").mean().alias("wrong_sign_share"),
                    pl.col("emissions_error").abs().sum().alias("emissions_error"),
                )
                .with_columns(
                    pl.lit(decile_type).alias("decile_type"),
                    pl.col(column).cast(pl.Utf8).alias("decile"),
                    pl.when(pl.col(column).cast(pl.Utf8).is_in(["d9", "d10"]))
                    .then(pl.lit("high-importance electricity nodes"))
                    .otherwise(pl.lit("lower-importance electricity nodes"))
                    .alias("recommended_interpretation"),
                )
                .select(
                    "rule_name",
                    "decile_type",
                    "decile",
                    "rows",
                    "observed_emissions_share",
                    "rEI_MAE",
                    "wrong_sign_share",
                    "emissions_error",
                    "recommended_interpretation",
                )
            )
        return pl.concat(frames, how="vertical_relaxed")

    def summarize_by_jump_status(self, rule_panel: pl.DataFrame) -> pl.DataFrame:
        total = rule_panel.select("country_sector", "year", "emissions_observed").unique()["emissions_observed"].sum() or 1.0
        return (
            rule_panel.with_columns(
                pl.when(pl.col("jump_flag_count") > 0).then(pl.lit("jump_year")).otherwise(pl.lit("nonjump_year")).alias(
                    "jump_status"
                )
            )
            .group_by("rule_name", "jump_status")
            .agg(
                pl.len().alias("rows"),
                (pl.col("emissions_observed").sum() / total).alias("observed_emissions_share"),
                pl.col("rEI_abs_error").mean().alias("rEI_MAE"),
                pl.col("wrong_sign").mean().alias("wrong_sign_share"),
                pl.col("emissions_error").abs().sum().alias("emissions_error"),
            )
            .with_columns(
                pl.when(pl.col("jump_status") == "jump_year")
                .then(pl.lit("structural-break sensitive"))
                .otherwise(pl.lit("ordinary electricity transition"))
                .alias("recommended_interpretation")
            )
        )

    def summarize_china_electricity(self, rule_panel: pl.DataFrame) -> pl.DataFrame:
        return (
            rule_panel.filter(self._china_expr())
            .select(
                "rule_name",
                "target_year",
                "X_observed",
                "EI_observed",
                "emissions_observed",
                "observed_rEI",
                "simulated_rEI",
                "simulated_EI",
                "simulated_emissions",
                "rEI_error",
                "emissions_error",
                "jump_flag_count",
                "notes",
            )
            .rename({"target_year": "year"})
            .sort("rule_name", "year")
        )

    def build_recommendation(self, comparison: pl.DataFrame) -> pl.DataFrame:
        """Apply Phase 20 decision rules."""
        reference = self._row_for_rule(comparison, "current_frontier_gap_readiness_reference")
        historical = self._row_for_rule(comparison, "historical_frontier_gap_only_reference")
        fixed = comparison.filter(pl.col("rule_name").str.contains("electricity_dampened_frontier_gap")).sort(
            "electricity_aggregate_emissions_error"
        )
        jump = self._row_for_rule(comparison, "electricity_gap_with_jump_shock_filter")
        high = self._row_for_rule(comparison, "electricity_high_emissions_dampened_gap")
        action = "inconclusive"
        evidence = ""
        interpretation = "Evidence is mixed."
        if not fixed.is_empty() and not reference.is_empty():
            best_fixed = fixed.head(1)
            if (
                best_fixed["unweighted_rEI_MAE"].item() <= reference["unweighted_rEI_MAE"].item()
                and best_fixed["electricity_aggregate_emissions_error"].item()
                <= reference["electricity_aggregate_emissions_error"].item()
            ):
                action = "test_electricity_dampened_frontier_gap_as_candidate_rule"
                evidence = f"best_fixed={best_fixed['rule_name'].item()}"
                interpretation = "Fixed dampening improves both transition and aggregate electricity metrics."
        if action == "inconclusive" and not jump.is_empty() and not reference.is_empty():
            if (
                jump["jump_year_rEI_MAE"].item() < reference["jump_year_rEI_MAE"].item()
                and jump["nonjump_year_rEI_MAE"].item() <= reference["nonjump_year_rEI_MAE"].item() * 1.05
            ):
                action = "test_electricity_jump_filtered_rule"
                evidence = "jump-filter improves jump-year MAE without materially worsening non-jump years"
                interpretation = "Jump years drive enough error to justify a jump-filter diagnostic candidate."
        if action == "inconclusive" and not high.is_empty() and not reference.is_empty():
            if (
                high["china_electricity_emissions_error"].item() < reference["china_electricity_emissions_error"].item()
                and high["unweighted_rEI_MAE"].item() <= reference["unweighted_rEI_MAE"].item() * 1.05
            ):
                action = "test_electricity_dampened_frontier_gap_as_candidate_rule"
                evidence = "high-emissions dampening improves China fit with limited average transition cost"
                interpretation = "High-emissions electricity dampening is a plausible candidate."
        if action == "inconclusive" and not reference.is_empty():
            best_aggregate = comparison.sort("electricity_aggregate_emissions_error").head(1)
            if best_aggregate["rule_name"].item() == "current_frontier_gap_readiness_reference":
                action = "keep_frontier_gap_readiness_as_temporary_aggregate_safe_base"
                evidence = "current readiness reference has best aggregate electricity emissions fit"
                interpretation = "Diagnostic candidates do not beat the aggregate-safe reference."
        if action == "inconclusive" and not historical.is_empty():
            action = "develop_energy_system_submodel_later"
            evidence = "no simple electricity candidate dominates transition and aggregate metrics"
            interpretation = "Electricity remains structurally difficult for generic transition rules."
        return pl.DataFrame(
            [
                {
                    "recommendation": action,
                    "evidence": evidence,
                    "interpretation": interpretation,
                    "recommended_phase21": "Test only the selected electricity candidate as diagnostic, while keeping scenarios blocked.",
                    "scenario_readiness": "premature",
                }
            ]
        )

    def build_markdown_report(
        self,
        *,
        target_diagnostics: pl.DataFrame,
        rule_comparison: pl.DataFrame,
        by_country: pl.DataFrame,
        by_year: pl.DataFrame,
        by_decile: pl.DataFrame,
        by_jump_status: pl.DataFrame,
        china_comparison: pl.DataFrame,
        recommendation: pl.DataFrame,
        node_count: int,
        row_count: int,
    ) -> str:
        action = recommendation["recommendation"].item(0) if not recommendation.is_empty() else "inconclusive"
        lines = [
            "# ABM v4 Phase 20 Electricity Transition Regime Diagnostics",
            "",
            f"Electricity-like nodes included: {node_count}; transition rows: {row_count}.",
            f"Recommendation: `{action}`.",
            "",
            "## Recommendation",
            self._markdown_table(recommendation),
            "",
            "## Target Diagnostics",
            self._markdown_table(target_diagnostics),
            "",
            "## Rule Comparison",
            self._markdown_table(rule_comparison.head(20)),
            "",
            "## China Electricity Rule Comparison",
            self._markdown_table(china_comparison.head(80)),
            "",
            "## Jump Status",
            self._markdown_table(by_jump_status.head(40)),
            "",
            "## Major Countries",
            self._markdown_table(by_country.sort("observed_emissions_share", descending=True).head(60)),
            "",
            "## Years",
            self._markdown_table(by_year.head(60)),
            "",
            "## Deciles",
            self._markdown_table(by_decile.head(60)),
            "",
            "Scenarios remain premature.",
        ]
        return "\n".join(lines) + "\n"

    def _model_output(self, frame: pl.DataFrame, suffix: str) -> pl.DataFrame:
        return frame.select(
            "country_sector",
            "year",
            pl.col("rEI_used").alias(f"simulated_rEI_{suffix}"),
            pl.col("EI_sim").alias(f"EI_sim_{suffix}"),
            pl.col("emissions_sim").alias(f"emissions_sim_{suffix}"),
            *([pl.col("readiness").alias("readiness")] if suffix == "readiness" and "readiness" in frame.columns else []),
        )

    def _jump_counts(self) -> pl.DataFrame:
        path = self.paths.raw_eora_electricity_breakpoint_audit_path
        if not path.exists():
            return pl.DataFrame(
                schema={"country_sector": pl.Utf8, "year": pl.Int64, "jump_flag_count": pl.Int64}
            )
        return (
            pl.read_csv(path)
            .filter(pl.col("jump_flag"))
            .group_by("country_sector", "year")
            .len()
            .rename({"len": "jump_flag_count"})
        )

    def _require_phase19_outputs(self) -> None:
        required = [
            self.paths.raw_eora_electricity_data_audit_recommendation_path,
            self.paths.raw_eora_major_electricity_comparison_path,
            self.paths.raw_eora_electricity_breakpoint_audit_path,
        ]
        missing = [path for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing Phase 19 audit outputs for electricity regime diagnostics: {missing}. "
                "Run --audit-raw-eora-electricity-data first; do not rerun it unless these files are absent."
            )

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

    def _weighted_expr(self, value_column: str, weight_column: str) -> pl.Expr:
        return (pl.col(value_column).cast(pl.Float64) * pl.col(weight_column)).sum() / pl.col(weight_column).sum()

    def _latest_pct_error_expr(self) -> pl.Expr:
        latest = self.end_year
        return (
            pl.when(pl.col("target_year") == latest).then(pl.col("simulated_emissions")).otherwise(0.0).sum()
            - pl.when(pl.col("target_year") == latest).then(pl.col("emissions_next")).otherwise(0.0).sum()
        ) / pl.when(pl.col("target_year") == latest).then(pl.col("emissions_next")).otherwise(0.0).sum()

    def _mean_yearly_pct_error_expr(self) -> pl.Expr:
        # A compact approximation at group level: total absolute error divided by total observed emissions.
        return pl.col("emissions_error").abs().sum() / pl.col("emissions_next").sum()

    def _row_for_rule(self, comparison: pl.DataFrame, rule_name: str) -> pl.DataFrame:
        return comparison.filter(pl.col("rule_name") == rule_name)

    def _share(self, frame: pl.DataFrame, expr: pl.Expr) -> float:
        if frame.is_empty():
            return 0.0
        return _as_float(frame.select(expr.mean()).item())

    def _markdown_table(self, frame: pl.DataFrame) -> str:
        if frame.is_empty():
            return "_No rows available._"
        columns = frame.columns
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
        for row in frame.to_dicts():
            lines.append("| " + " | ".join(self._format_markdown_value(row.get(column)) for column in columns) + " |")
        return "\n".join(lines)

    def _format_markdown_value(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        if value is None:
            return ""
        return str(value).replace("|", "/")


class StructuralSignatureDiagnostics:
    """Discover observable ABM v4 structural signatures for transition inertia."""

    def __init__(self, paths: ABMV4Paths, start_year: int = 1995, end_year: int = 2016) -> None:
        self.paths = paths
        self.start_year = start_year
        self.end_year = end_year

    def run(self) -> StructuralSignatureDiagnosticResult:
        """Build all Phase 21 diagnostic artifacts in memory."""
        inventory = self.discover_available_metrics()
        node_year = self.build_node_year_feature_panel()
        if node_year.is_empty():
            raise FileNotFoundError(
                "No usable structural-signature feature panel could be built. Run ABM v4 state construction "
                "and Phase 16 diagnostics first."
            )
        node_year = self.define_structural_labels(node_year)
        node_panel = self.build_node_level_feature_panel(node_year)
        label_summary = self.build_label_summary(node_year, node_panel)
        contrast = self.compare_electricity_vs_non_electricity(node_panel)
        screening = self.screen_metric_discrimination(node_panel)
        lookalikes = self.identify_non_electricity_lookalikes(node_panel, screening)
        proxies = self.build_candidate_proxy_table(node_panel, screening, lookalikes)
        recommendation = self.build_recommendation(screening, lookalikes, proxies)
        markdown = self.build_markdown_report(
            inventory=inventory,
            label_summary=label_summary,
            contrast=contrast,
            screening=screening,
            lookalikes=lookalikes,
            proxies=proxies,
            recommendation=recommendation,
            node_year_rows=node_year.height,
            node_rows=node_panel.height,
        )
        return StructuralSignatureDiagnosticResult(
            metric_inventory=inventory,
            node_year_panel=node_year,
            node_panel=node_panel,
            label_summary=label_summary,
            electricity_contrast=contrast,
            metric_screening=screening,
            lookalikes=lookalikes,
            candidate_proxies=proxies,
            recommendation=recommendation,
            markdown=markdown,
        )

    def write_outputs(self, result: StructuralSignatureDiagnosticResult) -> None:
        """Write Phase 21 outputs. The caller controls whether this is invoked."""
        self.paths.validation.mkdir(parents=True, exist_ok=True)
        result.metric_inventory.write_csv(self.paths.structural_signature_metric_inventory_path)
        result.node_year_panel.write_parquet(self.paths.structural_signature_node_year_panel_path)
        result.node_panel.write_parquet(self.paths.structural_signature_node_panel_path)
        result.label_summary.write_csv(self.paths.structural_signature_label_summary_path)
        result.electricity_contrast.write_csv(self.paths.electricity_structural_signature_contrast_path)
        result.metric_screening.write_csv(self.paths.structural_signature_metric_screening_path)
        result.lookalikes.write_csv(self.paths.structural_signature_non_electricity_lookalikes_path)
        result.candidate_proxies.write_csv(self.paths.candidate_transition_inertia_proxies_path)
        result.recommendation.write_csv(self.paths.structural_signature_recommendation_path)
        self.paths.structural_signature_report_path.write_text(result.markdown, encoding="utf-8")

    def discover_available_metrics(self) -> pl.DataFrame:
        """Inventory available compact ABM v4 files and classify metric families."""
        candidates = [
            self.paths.state_panel_path(self.start_year, self.end_year),
            self.paths.transition_rule_sign_failure_panel_path,
            self.paths.transition_rule_error_decomposition_path,
            self.paths.transition_rule_aggregate_contribution_path,
            self.paths.high_emissions_concentration_diagnostic_path,
            self.paths.readiness_dampening_diagnostic_path,
            self.paths.electricity_transition_rule_comparison_path,
            self.paths.electricity_transition_rule_by_country_path,
            self.paths.electricity_transition_rule_by_decile_path,
            self.paths.electricity_transition_rule_by_jump_status_path,
            self.paths.raw_eora_major_electricity_comparison_path,
            self.paths.raw_eora_electricity_breakpoint_audit_path,
            self.paths.capability_exposure_panel_path,
            self.paths.capability_update_panel_path,
            self.paths.production_feasibility_panel_path,
            self.paths.supplier_opportunity_sets_path,
            self.paths.supplier_updated_weights_path,
            self.paths.ecosystem_mapping_path,
            self.paths.ecosystem_adjacency_path,
        ]
        rows = []
        for path in candidates:
            exists = path.exists()
            columns: list[str] = []
            rows_if_known: int | None = None
            notes = ""
            if exists:
                try:
                    if path.suffix.lower() == ".parquet":
                        scan = pl.scan_parquet(path)
                        columns = list(scan.collect_schema().names())
                        rows_if_known = int(scan.select(pl.len()).collect().item())
                    elif path.suffix.lower() == ".csv":
                        frame = pl.read_csv(path, n_rows=50, infer_schema_length=50)
                        columns = frame.columns
                except Exception as exc:
                    notes = f"schema inspection failed: {exc}"
            families = self._metric_families(columns, path)
            rows.append(
                {
                    "source_path": self._rel(path),
                    "exists": exists,
                    "rows_if_known": rows_if_known,
                    "columns": ";".join(columns),
                    "metric_family": ";".join(families),
                    "candidate_metrics": ";".join(self._candidate_metric_columns(columns)),
                    "usable_for_node_year_panel": bool({"country_sector", "year"} <= set(columns) or {"country_sector", "Year"} <= set(columns)),
                    "usable_for_node_level_panel": bool("country_sector" in columns or "Sector" in columns),
                    "notes": notes if notes else ("available" if exists else "missing"),
                }
            )
        return pl.DataFrame(rows)

    def build_node_year_feature_panel(self) -> pl.DataFrame:
        """Build a broad node-year feature panel from compact existing outputs."""
        state_path = self.paths.state_panel_path(self.start_year, self.end_year)
        if not state_path.exists():
            raise FileNotFoundError(f"Missing ABM v4 state panel: {state_path}. Run --build-state first.")
        state = pl.read_parquet(state_path)
        if "Year" in state.columns and "year" not in state.columns:
            state = state.rename({"Year": "year"})
        if "EI_observed" not in state.columns and "EI" in state.columns:
            state = state.rename({"EI": "EI_observed"})
        panel = state.select(
            [
                c
                for c in [
                    "country_sector",
                    "Country",
                    "Sector",
                    "year",
                    "ecosystem_id",
                    "ecosystem_label",
                    "X_observed",
                    "EI_observed",
                    "emissions_observed",
                    "network_green_exposure",
                    "brown_centrality",
                    "general_capability_model",
                    "green_capability_model",
                    "general_capability_source",
                    "green_capability_source",
                    "input_feasibility",
                    "production_feasibility_ratio",
                ]
                if c in state.columns
            ]
        )
        panel = self._join_optional(panel, self._phase16_panel(), ["country_sector", "year"])
        panel = self._join_optional(panel, self._production_panel(), ["country_sector", "year"])
        panel = self._join_optional(panel, self._capability_panel(), ["country_sector", "year"])
        panel = self._join_optional(panel, self._supplier_features(), ["country_sector"])
        panel = self._join_optional(panel, self._jump_features(), ["country_sector", "year"])
        panel = panel.sort("country_sector", "year")
        total_by_year = panel.group_by("year").agg(
            pl.col("X_observed").sum().alias("_year_output_total"),
            pl.col("emissions_observed").sum().alias("_year_emissions_total"),
        )
        panel = panel.join(total_by_year, on="year", how="left").with_columns(
            self._electricity_expr().alias("electricity_like"),
            pl.when(pl.col("X_observed") > 0).then(pl.col("X_observed").log()).otherwise(None).alias("log_X_observed"),
            pl.when(pl.col("emissions_observed") > 0).then(pl.col("emissions_observed").log()).otherwise(None).alias(
                "log_emissions_observed"
            ),
            pl.when(pl.col("EI_observed") > 0).then(pl.col("EI_observed").log()).otherwise(None).alias("log_EI_observed"),
            (pl.col("X_observed") / pl.col("_year_output_total")).alias("output_share_year"),
            (pl.col("emissions_observed") / pl.col("_year_emissions_total")).alias("emissions_share_year"),
            pl.col("X_observed").rank(method="ordinal", descending=True).over("year").alias("output_rank_year"),
            pl.col("emissions_observed").rank(method="ordinal", descending=True).over("year").alias("emissions_rank_year"),
            pl.col("EI_observed").rank(method="average").over("Sector", "year").alias("EI_rank_within_sector_year"),
            (pl.col("EI_observed").rank(method="average").over("Sector", "year") / pl.len().over("Sector", "year")).alias(
                "EI_percentile_within_sector_year"
            ),
        ).with_columns(
            (pl.col("log_EI_observed") - pl.col("log_EI_observed").shift(1).over("country_sector")).alias("EI_growth"),
            (pl.col("log_X_observed") - pl.col("log_X_observed").shift(1).over("country_sector")).alias("X_growth"),
            (pl.col("log_emissions_observed") - pl.col("log_emissions_observed").shift(1).over("country_sector")).alias(
                "emissions_growth"
            ),
        ).with_columns(
            (-pl.col("EI_growth")).alias("observed_rEI"),
            pl.col("EI_growth").abs().alias("abs_rEI"),
            pl.col("EI_growth").sign().alias("rEI_sign"),
            pl.col("EI_growth").rolling_std(window_size=3, min_samples=2).over("country_sector").alias(
                "rolling_3yr_EI_volatility"
            ),
            pl.col("X_growth").rolling_std(window_size=3, min_samples=2).over("country_sector").alias(
                "rolling_3yr_X_volatility"
            ),
            pl.col("emissions_growth").rolling_std(window_size=3, min_samples=2).over("country_sector").alias(
                "rolling_3yr_emissions_volatility"
            ),
        )
        for column in [
            "ecosystem_label",
            "jump_flag",
            "jump_count_recent",
            "supplier_count",
            "supplier_opportunity_count",
            "supplier_weight_concentration",
            "mean_supplier_green_exposure",
            "cap_model",
            "gcap_model",
            "exposure_cap",
            "exposure_gcap",
            "production_feasibility_ratio",
            "input_feasibility",
        ]:
            if column not in panel.columns:
                dtype = pl.Utf8 if column == "ecosystem_label" else pl.Float64
                panel = panel.with_columns(pl.lit(None).cast(dtype).alias(column))
        return panel.drop(["_year_output_total", "_year_emissions_total"])

    def build_node_level_feature_panel(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Aggregate node-year signatures to country-sector level."""
        total_emissions = panel["emissions_observed"].sum() or 1.0
        total_output = panel["X_observed"].sum() or 1.0
        explicit_node_metrics = {
            "frontier_gap",
            "readiness",
            "dampening_amount",
            "frontier_gap_improves_abs_error",
            "frontier_gap_worsens_sign",
            "emissions_error_frontier_gap",
            "emissions_error_readiness",
            "jump_flag",
        }
        numeric = [
            c
            for c in self._screenable_metrics(panel)
            if c not in explicit_node_metrics
            and c in panel.columns
            and panel.schema.get(c) in {pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt32, pl.UInt64}
        ]
        agg_exprs = [
            pl.col("Country").drop_nulls().first().alias("Country"),
            pl.col("Sector").drop_nulls().first().alias("Sector"),
            pl.col("ecosystem_label").drop_nulls().first().alias("ecosystem"),
            pl.col("electricity_like").max().alias("electricity_like"),
            (pl.col("emissions_observed").sum() / total_emissions).alias("cumulative_emissions_share"),
            (pl.col("X_observed").sum() / total_output).alias("cumulative_output_share"),
            pl.col("emissions_rank_year").mean().alias("mean_emissions_rank"),
            pl.col("emissions_rank_year").min().alias("best_emissions_rank"),
            pl.col("output_rank_year").mean().alias("mean_output_rank"),
            pl.col("output_rank_year").min().alias("best_output_rank"),
            pl.col("jump_flag").fill_null(False).mean().alias("jump_frequency"),
            pl.when(pl.col("jump_flag").fill_null(False)).then(pl.col("emissions_observed")).otherwise(0.0).sum().alias(
                "_jump_emissions"
            ),
            pl.col("observed_rEI").std().alias("rEI_volatility"),
            pl.col("X_growth").std().alias("X_volatility"),
            pl.col("emissions_growth").std().alias("emissions_volatility"),
            pl.col("frontier_gap").mean().alias("mean_frontier_gap") if "frontier_gap" in panel.columns else pl.lit(None).alias("mean_frontier_gap"),
            pl.col("readiness").mean().alias("mean_readiness") if "readiness" in panel.columns else pl.lit(None).alias("mean_readiness"),
            pl.col("dampening_amount").mean().alias("mean_dampening_amount") if "dampening_amount" in panel.columns else pl.lit(None).alias("mean_dampening_amount"),
            pl.col("frontier_gap_improves_abs_error").mean().alias("share_frontier_gap_improves_abs_error")
            if "frontier_gap_improves_abs_error" in panel.columns
            else pl.lit(None).alias("share_frontier_gap_improves_abs_error"),
            pl.col("frontier_gap_worsens_sign").mean().alias("share_frontier_gap_worsens_sign")
            if "frontier_gap_worsens_sign" in panel.columns
            else pl.lit(None).alias("share_frontier_gap_worsens_sign"),
            pl.when(pl.col("emissions_error_frontier_gap").abs() > pl.col("emissions_error_readiness").abs())
            .then(1.0)
            .otherwise(0.0)
            .mean()
            .alias("share_frontier_gap_worsens_emissions_error")
            if {"emissions_error_frontier_gap", "emissions_error_readiness"} <= set(panel.columns)
            else pl.lit(None).alias("share_frontier_gap_worsens_emissions_error"),
        ]
        for metric in numeric:
            agg_exprs.extend(
                [
                    pl.col(metric).mean().alias(f"mean_{metric}"),
                    pl.col(metric).median().alias(f"median_{metric}"),
                    pl.col(metric).std().alias(f"std_{metric}"),
                    pl.col(metric).quantile(0.95).alias(f"p95_{metric}"),
                    pl.col(metric).max().alias(f"max_{metric}"),
                ]
            )
        node = panel.group_by("country_sector").agg(agg_exprs).with_columns(
            (pl.col("_jump_emissions") / (pl.col("cumulative_emissions_share") * total_emissions)).fill_nan(0).alias(
                "jump_emissions_share"
            )
        ).drop("_jump_emissions")
        return self.define_node_labels(node)

    def define_structural_labels(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Add transparent node-year structural labels."""
        contribution_threshold = self._quantile_or_zero(panel, "contribution_to_aggregate_error_difference", 0.90)
        emissions_threshold = self._quantile_or_zero(panel, "emissions_share_year", 0.90)
        output_threshold = self._quantile_or_zero(panel, "output_share_year", 0.90)
        return panel.with_columns(
            (pl.col("emissions_share_year") >= emissions_threshold).alias("high_emissions_node_year"),
            (pl.col("output_share_year") >= output_threshold).alias("high_output_node_year"),
            (
                pl.col("contribution_to_aggregate_error_difference").fill_null(0) >= contribution_threshold
                if "contribution_to_aggregate_error_difference" in panel.columns
                else pl.lit(False)
            ).alias("aggregate_sensitive_node_year"),
            (
                (pl.col("contribution_to_aggregate_error_difference").fill_null(0) > contribution_threshold)
                | (
                    (pl.col("emissions_error_frontier_gap").abs() > pl.col("emissions_error_readiness").abs())
                    if {"emissions_error_frontier_gap", "emissions_error_readiness"} <= set(panel.columns)
                    else pl.lit(False)
                )
            ).alias("needs_dampening_node_year"),
        )

    def define_node_labels(self, node: pl.DataFrame) -> pl.DataFrame:
        """Add transparent node-level structural labels."""
        emissions_threshold = self._quantile_or_zero(node, "cumulative_emissions_share", 0.90)
        output_threshold = self._quantile_or_zero(node, "cumulative_output_share", 0.90)
        jump_threshold = self._quantile_or_zero(node, "jump_frequency", 0.90)
        contribution_threshold = self._quantile_or_zero(node, "mean_contribution_to_aggregate_error_difference", 0.90)
        jump_expr = (
            pl.col("jump_frequency") > 0
            if jump_threshold <= 0
            else pl.col("jump_frequency") >= jump_threshold
        )
        node = node.with_columns(
            (pl.col("cumulative_emissions_share") >= emissions_threshold).alias("high_emissions_node"),
            (pl.col("cumulative_output_share") >= output_threshold).alias("high_output_node"),
            jump_expr.alias("jump_prone_node"),
            (pl.col("mean_contribution_to_aggregate_error_difference").fill_null(0) >= contribution_threshold).alias(
                "aggregate_sensitive_node"
            )
            if "mean_contribution_to_aggregate_error_difference" in node.columns
            else pl.lit(False).alias("aggregate_sensitive_node"),
        )
        return node.with_columns(
            (
                (pl.col("share_frontier_gap_worsens_emissions_error").fill_null(0) > 0.5)
                | pl.col("aggregate_sensitive_node")
            ).alias("needs_dampening_node"),
        )

    def build_label_summary(self, node_year: pl.DataFrame, node: pl.DataFrame) -> pl.DataFrame:
        """Summarize structural label sizes and composition."""
        definitions = {
            "electricity_like": "Sector label contains electricity, gas, water, utilities, or power.",
            "high_emissions_node": "Top 10 percent by cumulative emissions share.",
            "high_output_node": "Top 10 percent by cumulative output share.",
            "jump_prone_node": "Jump frequency above p90.",
            "aggregate_sensitive_node": "Top 10 percent aggregate error contribution signature.",
            "needs_dampening_node": "Frontier-gap rule worsens emissions error or aggregate contribution.",
        }
        rows = []
        for label, definition in definitions.items():
            label_col = label if label in node.columns else f"{label}_year"
            node_subset = node.filter(pl.col(label_col)) if label_col in node.columns else node.head(0)
            year_col = label if label in node_year.columns else f"{label}_year"
            year_subset = node_year.filter(pl.col(year_col)) if year_col in node_year.columns else node_year.join(node_subset.select("country_sector"), on="country_sector", how="inner")
            rows.append(
                {
                    "label_name": label,
                    "definition": definition,
                    "rows_node_year": year_subset.height,
                    "nodes": node_subset.height,
                    "electricity_share": self._share(node_subset, pl.col("electricity_like")),
                    "emissions_share": _as_float(node_subset["cumulative_emissions_share"].sum()),
                    "output_share": _as_float(node_subset["cumulative_output_share"].sum()),
                    "notes": "diagnostic label, not calibration target",
                }
            )
        return pl.DataFrame(rows)

    def compare_electricity_vs_non_electricity(self, node: pl.DataFrame) -> pl.DataFrame:
        """Contrast electricity-like nodes against all others across usable metrics."""
        rows = []
        for metric in self._screenable_metrics(node):
            if metric not in node.columns:
                continue
            series = node.select(metric, "electricity_like").drop_nulls()
            if series.height < 5:
                continue
            elec = series.filter(pl.col("electricity_like"))[metric]
            other = series.filter(~pl.col("electricity_like"))[metric]
            if elec.is_empty() or other.is_empty():
                continue
            std = _as_float(series[metric].std(), 0.0)
            elec_mean = _as_float(elec.mean())
            other_mean = _as_float(other.mean())
            elec_median = _as_float(elec.median())
            percentile = self._percentile_of_value(series[metric], elec_median)
            rows.append(
                {
                    "metric": metric,
                    "metric_family": self._metric_family_for_metric(metric),
                    "electricity_mean": elec_mean,
                    "non_electricity_mean": other_mean,
                    "electricity_median": elec_median,
                    "non_electricity_median": _as_float(other.median()),
                    "electricity_p25": _as_float(elec.quantile(0.25)),
                    "electricity_p75": _as_float(elec.quantile(0.75)),
                    "non_electricity_p25": _as_float(other.quantile(0.25)),
                    "non_electricity_p75": _as_float(other.quantile(0.75)),
                    "standardized_difference": 0.0 if std == 0 else (elec_mean - other_mean) / std,
                    "electricity_median_percentile_in_all_nodes": percentile,
                    "interpretation": "strong contrast" if abs(0.0 if std == 0 else (elec_mean - other_mean) / std) >= 0.5 else "weak/moderate contrast",
                    "candidate_mechanism": self._candidate_mechanism(metric),
                }
            )
        if not rows:
            return pl.DataFrame()
        return pl.DataFrame(rows).sort("standardized_difference", descending=True)

    def screen_metric_discrimination(self, node: pl.DataFrame) -> pl.DataFrame:
        """Screen metrics for label separation using transparent univariate scores."""
        labels = [
            "electricity_like",
            "high_emissions_node",
            "jump_prone_node",
            "aggregate_sensitive_node",
            "needs_dampening_node",
        ]
        rows = []
        for label in labels:
            if label not in node.columns:
                continue
            for metric in self._screenable_metrics(node):
                if metric not in node.columns:
                    continue
                frame = node.select(label, metric).drop_nulls()
                if frame.height < 5 or frame[label].n_unique() < 2:
                    continue
                positive = frame.filter(pl.col(label))[metric]
                negative = frame.filter(~pl.col(label))[metric]
                if positive.is_empty() or negative.is_empty():
                    continue
                std = _as_float(frame[metric].std(), 0.0)
                sd = 0.0 if std == 0 else (_as_float(positive.mean()) - _as_float(negative.mean())) / std
                corr = _as_float(frame.select(pl.corr(pl.col(metric), pl.col(label).cast(pl.Float64))).item())
                separation = abs(sd) + abs(corr)
                rows.append(
                    {
                        "label_name": label,
                        "metric": metric,
                        "metric_family": self._metric_family_for_metric(metric),
                        "usable_rows": frame.height,
                        "separation_score": separation,
                        "abs_correlation_or_auc": abs(corr),
                        "standardized_difference": sd,
                        "interpretation": "strong discriminator" if separation >= 0.8 else "screening signal",
                        "recommended_for_proxy": separation >= 0.8 and not metric.startswith("mean_rEI_abs_error"),
                    }
                )
        out = pl.DataFrame(rows)
        if out.is_empty():
            return out
        return out.sort(["label_name", "separation_score"], descending=[False, True]).with_columns(
            pl.col("separation_score").rank(method="ordinal", descending=True).over("label_name").alias("rank")
        )

    def identify_non_electricity_lookalikes(self, node: pl.DataFrame, screening: pl.DataFrame) -> pl.DataFrame:
        """Find non-electricity nodes similar to electricity on top structural metrics."""
        top_metrics = (
            screening.filter((pl.col("label_name") == "electricity_like") & (pl.col("recommended_for_proxy")))
            .sort("rank")
            .head(5)["metric"]
            .to_list()
            if not screening.is_empty()
            else []
        )
        top_metrics = [m for m in top_metrics if m in node.columns and not m.startswith("mean_rEI_abs_error")]
        if not top_metrics:
            top_metrics = [m for m in ["cumulative_emissions_share", "cumulative_output_share", "jump_frequency"] if m in node.columns]
        if not top_metrics:
            return pl.DataFrame()
        scored = node
        score_exprs = []
        for metric in top_metrics:
            elec_median = _as_float(node.filter(pl.col("electricity_like"))[metric].median())
            std = _as_float(node[metric].std(), 1.0) or 1.0
            score_exprs.append(
                pl.when(pl.col(metric).is_not_null())
                .then(1.0 / (1.0 + ((pl.col(metric) - elec_median).abs() / std)))
                .otherwise(0.0)
                .alias(f"_sim_{metric}")
            )
            scored = scored.with_columns(score_exprs[-1])
        sim_cols = [f"_sim_{m}" for m in top_metrics]
        return (
            scored.filter(~pl.col("electricity_like"))
            .with_columns(pl.mean_horizontal([pl.col(c) for c in sim_cols]).alias("similarity_score"))
            .sort("similarity_score", descending=True)
            .head(50)
            .select(
                "country_sector",
                "Country",
                "Sector",
                pl.col("ecosystem").alias("ecosystem"),
                "similarity_score",
                pl.lit(";".join(top_metrics)).alias("top_matching_metrics"),
                pl.col("cumulative_emissions_share").alias("emissions_share"),
                pl.col("cumulative_output_share").alias("output_share"),
                "jump_frequency",
                pl.col("share_frontier_gap_worsens_emissions_error").fill_null(0).alias("needs_dampening_score"),
                pl.when(pl.col("similarity_score") >= 0.7)
                .then(pl.lit("plausible non-electricity structural lookalike"))
                .otherwise(pl.lit("weaker lookalike"))
                .alias("interpretation"),
            )
        )

    def build_candidate_proxy_table(
        self,
        node: pl.DataFrame,
        screening: pl.DataFrame,
        lookalikes: pl.DataFrame,
    ) -> pl.DataFrame:
        """Build generalized transition-inertia proxy shortlist."""
        proxy_defs = [
            ("systemic_scale_proxy", "systemic_scale", "rank-normalized cumulative emissions and output share", ["cumulative_emissions_share", "cumulative_output_share"]),
            ("emissions_centrality_proxy", "emissions_centrality", "cumulative emissions share and best emissions rank", ["cumulative_emissions_share", "best_emissions_rank"]),
            ("output_centrality_proxy", "output_centrality", "cumulative output share and best output rank", ["cumulative_output_share", "best_output_rank"]),
            ("volatility_jump_proxy", "volatility_jump_regime", "jump frequency plus rEI/output/emissions volatility", ["jump_frequency", "rEI_volatility", "X_volatility", "emissions_volatility"]),
            ("brown_lockin_proxy", "brown_lock_in", "brown centrality and emissions intensity level", ["mean_brown_centrality", "mean_EI_observed"]),
            ("supplier_constraint_proxy", "supplier_constraint", "supplier concentration and production feasibility pressure", ["mean_supplier_weight_concentration", "mean_production_feasibility_ratio"]),
            ("model_error_dampening_need_proxy", "model_error_signature", "frontier-gap error deterioration and aggregate contribution", ["share_frontier_gap_worsens_emissions_error", "mean_contribution_to_aggregate_error_difference"]),
            ("composite_transition_inertia_proxy", "composite", "systemic scale + volatility jump + brown lock-in + model-error caution screen", ["cumulative_emissions_share", "cumulative_output_share", "jump_frequency", "mean_brown_centrality", "share_frontier_gap_worsens_emissions_error"]),
        ]
        rows = []
        lookalike_count = lookalikes.filter(pl.col("similarity_score") >= 0.7).height if not lookalikes.is_empty() else 0
        for name, family, formula, required in proxy_defs:
            available = [m for m in required if m in node.columns]
            availability = "available" if len(available) == len(required) else ("partial" if available else "unavailable")
            electricity_percentile = self._proxy_electricity_percentile(node, available)
            dampening_rows = (
                screening.filter((pl.col("label_name") == "needs_dampening_node") & pl.col("metric").is_in(available))
                if not screening.is_empty() and available
                else pl.DataFrame()
            )
            dampening = "strong" if (not dampening_rows.is_empty() and dampening_rows["separation_score"].max() >= 0.8) else "weak_or_unknown"
            rows.append(
                {
                    "candidate_proxy": name,
                    "metric_family": family,
                    "formula_description": formula,
                    "required_metrics": ";".join(required),
                    "availability_status": availability,
                    "economic_interpretation": self._proxy_interpretation(name),
                    "electricity_percentile": electricity_percentile,
                    "high_emissions_node_relationship": self._proxy_label_relationship(screening, "high_emissions_node", available),
                    "dampening_need_relationship": dampening,
                    "non_electricity_lookalike_count": lookalike_count,
                    "risks": self._proxy_risk(name),
                    "recommended_for_phase22": availability != "unavailable" and electricity_percentile >= 0.6,
                }
            )
        return pl.DataFrame(rows)

    def build_recommendation(
        self,
        screening: pl.DataFrame,
        lookalikes: pl.DataFrame,
        proxies: pl.DataFrame,
    ) -> pl.DataFrame:
        """Apply Phase 21 decision rules."""
        elec = screening.filter(pl.col("label_name") == "electricity_like") if not screening.is_empty() else pl.DataFrame()
        strong_structural = elec.filter((pl.col("separation_score") >= 0.8) & (pl.col("metric_family") != "model_error_signature"))
        strong_families = strong_structural["metric_family"].unique().to_list() if not strong_structural.is_empty() else []
        lookalike_count = lookalikes.filter(pl.col("similarity_score") >= 0.7).height if not lookalikes.is_empty() else 0
        model_error = elec.filter((pl.col("separation_score") >= 0.8) & (pl.col("metric_family") == "model_error_signature"))
        if len(strong_families) >= 2 and lookalike_count > 0:
            recommendation = "build_composite_transition_inertia_proxy"
            interpretation = "Multiple structural metric families distinguish electricity and also identify non-electricity lookalikes."
        elif len(strong_families) == 1:
            family = strong_families[0]
            recommendation = {
                "systemic_scale": "test_systemic_scale_dampener",
                "volatility_jump_regime": "test_volatility_jump_dampener",
                "brown_lock_in": "test_brown_lockin_dampener",
                "supplier_constraint": "test_input_universality_dampener",
            }.get(family, "test_single_metric_dampener")
            interpretation = f"One structural family dominates the screen: {family}."
        elif not model_error.is_empty():
            recommendation = "keep_electricity_specific_diagnostic_rule_only"
            interpretation = "The strongest signal is model-error based, so avoid over-theorizing a structural dampener."
        elif elec.is_empty() or elec["separation_score"].max() < 0.4:
            recommendation = "insufficient_signature"
            interpretation = "Available metrics do not clearly distinguish electricity-like nodes."
        else:
            recommendation = "inconclusive"
            interpretation = "Evidence is mixed."
        return pl.DataFrame(
            [
                {
                    "recommendation": recommendation,
                    "evidence": f"strong_structural_families={strong_families}; lookalike_count={lookalike_count}",
                    "interpretation": interpretation,
                    "recommended_phase22": "Test transition-inertia proxy candidates diagnostically; do not enable scenarios.",
                    "scenario_readiness": "premature",
                }
            ]
        )

    def build_markdown_report(
        self,
        *,
        inventory: pl.DataFrame,
        label_summary: pl.DataFrame,
        contrast: pl.DataFrame,
        screening: pl.DataFrame,
        lookalikes: pl.DataFrame,
        proxies: pl.DataFrame,
        recommendation: pl.DataFrame,
        node_year_rows: int,
        node_rows: int,
    ) -> str:
        action = recommendation["recommendation"].item(0) if not recommendation.is_empty() else "inconclusive"
        lines = [
            "# ABM v4 Phase 21 Structural-Signature Diagnostics",
            "",
            f"Node-year rows: {node_year_rows}; node rows: {node_rows}.",
            f"Recommendation: `{action}`.",
            "",
            "## Recommendation",
            self._markdown_table(recommendation),
            "",
            "## Metric Inventory",
            self._markdown_table(inventory.select([c for c in inventory.columns if c != "columns"]).head(40)),
            "",
            "## Label Summary",
            self._markdown_table(label_summary),
            "",
            "## Electricity Contrast",
            self._markdown_table(contrast.sort("standardized_difference", descending=True).head(40)),
            "",
            "## Metric Screening",
            self._markdown_table(screening.sort("separation_score", descending=True).head(60) if not screening.is_empty() else screening),
            "",
            "## Non-Electricity Lookalikes",
            self._markdown_table(lookalikes.head(40)),
            "",
            "## Candidate Proxies",
            self._markdown_table(proxies),
            "",
            "Scenarios remain premature.",
        ]
        return "\n".join(lines) + "\n"

    def _phase16_panel(self) -> pl.DataFrame:
        if not self.paths.transition_rule_sign_failure_panel_path.exists():
            return pl.DataFrame()
        frame = pl.read_parquet(self.paths.transition_rule_sign_failure_panel_path)
        if "frontier_gap" not in frame.columns and "ei_gap" in frame.columns:
            frame = frame.rename({"ei_gap": "frontier_gap"})
        keep = [
            c
            for c in [
                "country_sector",
                "year",
                "frontier_gap",
                "readiness",
                "rEI_abs_error_readiness",
                "rEI_abs_error_frontier_gap",
                "delta_abs_error",
                "frontier_gap_improves_abs_error",
                "frontier_gap_worsens_sign",
                "frontier_gap_improves_magnitude_but_worsens_sign",
                "emissions_error_readiness",
                "emissions_error_frontier_gap",
                "contribution_to_aggregate_error_difference",
                "dampening_amount",
            ]
            if c in frame.columns
        ]
        return frame.select(keep)

    def _production_panel(self) -> pl.DataFrame:
        if not self.paths.production_feasibility_panel_path.exists():
            return pl.DataFrame()
        frame = pl.read_parquet(self.paths.production_feasibility_panel_path)
        keep = [c for c in ["country_sector", "year", "input_feasibility", "production_feasibility_ratio"] if c in frame.columns]
        return frame.select(keep)

    def _capability_panel(self) -> pl.DataFrame:
        frames = []
        if self.paths.capability_update_panel_path.exists():
            frame = pl.read_parquet(self.paths.capability_update_panel_path)
            keep = [c for c in ["country_sector", "year", "cap_model", "gcap_model"] if c in frame.columns]
            if keep:
                frames.append(frame.select(keep))
        if self.paths.capability_exposure_panel_path.exists():
            frame = pl.read_parquet(self.paths.capability_exposure_panel_path)
            keep = [c for c in ["country_sector", "year", "exposure_cap", "exposure_gcap"] if c in frame.columns]
            if keep:
                frames.append(frame.select(keep))
        if not frames:
            return pl.DataFrame()
        out = frames[0]
        for frame in frames[1:]:
            out = out.join(frame, on=["country_sector", "year"], how="outer_coalesce")
        return out

    def _supplier_features(self) -> pl.DataFrame:
        frames = []
        if self.paths.supplier_updated_weights_path.exists():
            frame = pl.read_parquet(self.paths.supplier_updated_weights_path)
            if {"buyer_country_sector", "supplier_country_sector", "weight"}.issubset(frame.columns):
                frames.append(
                    frame.group_by("buyer_country_sector")
                    .agg(
                        pl.col("supplier_country_sector").n_unique().alias("supplier_count"),
                        (pl.col("weight") ** 2).sum().alias("supplier_weight_concentration"),
                    )
                    .rename({"buyer_country_sector": "country_sector"})
                )
        if self.paths.supplier_opportunity_sets_path.exists():
            frame = pl.read_parquet(self.paths.supplier_opportunity_sets_path)
            if "buyer_country_sector" in frame.columns:
                supplier_col = "candidate_supplier_country_sector" if "candidate_supplier_country_sector" in frame.columns else frame.columns[-1]
                frames.append(
                    frame.group_by("buyer_country_sector")
                    .agg(pl.col(supplier_col).n_unique().alias("supplier_opportunity_count"))
                    .rename({"buyer_country_sector": "country_sector"})
                )
        if not frames:
            return pl.DataFrame()
        out = frames[0]
        for frame in frames[1:]:
            out = out.join(frame, on="country_sector", how="outer_coalesce")
        return out

    def _jump_features(self) -> pl.DataFrame:
        if not self.paths.raw_eora_electricity_breakpoint_audit_path.exists():
            return pl.DataFrame()
        frame = pl.read_csv(self.paths.raw_eora_electricity_breakpoint_audit_path)
        if frame.is_empty() or "jump_flag" not in frame.columns:
            return pl.DataFrame()
        return (
            frame.group_by("country_sector", "year")
            .agg(pl.col("jump_flag").max().alias("jump_flag"), pl.col("jump_flag").sum().alias("jump_count_recent"))
        )

    def _join_optional(self, left: pl.DataFrame, right: pl.DataFrame, on: list[str]) -> pl.DataFrame:
        if right.is_empty() or not set(on).issubset(right.columns):
            return left
        columns_to_add = [c for c in right.columns if c not in left.columns or c in on]
        return left.join(right.select(columns_to_add), on=on, how="left")

    def _screenable_metrics(self, frame: pl.DataFrame) -> list[str]:
        exclusions = {"country_sector", "Country", "Sector", "ecosystem", "ecosystem_id", "ecosystem_label", "year"}
        metrics = []
        for column, dtype in frame.schema.items():
            if column in exclusions or dtype == pl.Boolean or dtype == pl.Utf8:
                continue
            if dtype in {pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt32, pl.UInt64}:
                metrics.append(column)
        return metrics

    def _metric_families(self, columns: list[str], path: Path) -> list[str]:
        families = set()
        text = " ".join(columns).lower() + " " + path.name.lower()
        mapping = {
            "production_scale": ["x_observed", "output", "production"],
            "emissions_scale": ["emissions"],
            "emissions_intensity": ["ei", "intensity"],
            "transition_dynamics": ["rei", "transition"],
            "volatility_jump": ["jump", "volatility"],
            "frontier_gap": ["frontier", "gap"],
            "network_position": ["centrality", "network", "pagerank"],
            "supplier_structure": ["supplier"],
            "buyer_structure": ["buyer"],
            "capability": ["cap_model", "general_capability", "capability"],
            "green_capability": ["gcap", "green_capability"],
            "ecosystem": ["ecosystem"],
            "production_feasibility": ["feasibility"],
            "model_error_signature": ["error", "wrong_sign", "dampening"],
            "electricity_specific_diagnostics": ["electricity"],
        }
        for family, needles in mapping.items():
            if any(needle in text for needle in needles):
                families.add(family)
        return sorted(families) if families else ["unclassified"]

    def _candidate_metric_columns(self, columns: list[str]) -> list[str]:
        return [
            c
            for c in columns
            if any(token in c.lower() for token in ["observed", "emissions", "ei", "cap", "green", "centrality", "feasibility", "error", "jump", "supplier", "frontier", "readiness"])
        ]

    def _metric_family_for_metric(self, metric: str) -> str:
        metric_l = metric.lower()
        if "emissions" in metric_l and "error" not in metric_l:
            return "emissions_scale"
        if "output" in metric_l or "x_observed" in metric_l or "x_growth" in metric_l:
            return "production_scale"
        if "jump" in metric_l or "volatility" in metric_l:
            return "volatility_jump_regime"
        if "brown" in metric_l:
            return "brown_lock_in"
        if "supplier" in metric_l or "input" in metric_l:
            return "supplier_constraint"
        if "cap" in metric_l:
            return "capability_constraint"
        if "frontier" in metric_l or "error" in metric_l or "dampening" in metric_l:
            return "model_error_signature"
        return "systemic_scale"

    def _candidate_mechanism(self, metric: str) -> str:
        family = self._metric_family_for_metric(metric)
        return {
            "emissions_scale": "emissions_centrality",
            "production_scale": "systemic_scale",
            "volatility_jump_regime": "volatility_jump_regime",
            "brown_lock_in": "brown_lock_in",
            "supplier_constraint": "supplier_constraint",
            "capability_constraint": "capability_constraint",
            "model_error_signature": "model_error_signature",
        }.get(family, "unclear")

    def _proxy_electricity_percentile(self, node: pl.DataFrame, metrics: list[str]) -> float:
        if not metrics or node.filter(pl.col("electricity_like")).is_empty():
            return float("nan")
        values = []
        for metric in metrics:
            elec = _as_float(node.filter(pl.col("electricity_like"))[metric].median())
            values.append(self._percentile_of_value(node[metric].drop_nulls(), elec))
        return sum(values) / len(values)

    def _proxy_label_relationship(self, screening: pl.DataFrame, label: str, metrics: list[str]) -> str:
        if screening.is_empty() or not metrics:
            return "unknown"
        rows = screening.filter((pl.col("label_name") == label) & pl.col("metric").is_in(metrics))
        if rows.is_empty():
            return "weak_or_unknown"
        return "strong" if rows["separation_score"].max() >= 0.8 else "weak_or_unknown"

    def _proxy_interpretation(self, proxy: str) -> str:
        return {
            "systemic_scale_proxy": "Large nodes convert modest EI errors into aggregate emissions errors.",
            "emissions_centrality_proxy": "High-emissions nodes may need slower structural transition closure.",
            "output_centrality_proxy": "High-output nodes amplify production-forced EI errors.",
            "volatility_jump_proxy": "Jump-prone nodes reflect structural breaks and accounting shifts.",
            "brown_lockin_proxy": "Brown centrality can proxy fossil and emissions lock-in.",
            "supplier_constraint_proxy": "Supplier concentration and feasibility pressure may slow transition.",
            "model_error_dampening_need_proxy": "Direct diagnostic proxy for where frontier closure over-adjusts.",
            "composite_transition_inertia_proxy": "Combines systemic scale, volatility, lock-in, and error caution.",
        }.get(proxy, "Candidate transition-inertia proxy.")

    def _proxy_risk(self, proxy: str) -> str:
        if "model_error" in proxy:
            return "High overfitting risk because it uses model-error signatures."
        if "composite" in proxy:
            return "Moderate overfitting risk; must be tested with transparent weights."
        return "Lower complexity, but may miss multi-mechanism inertia."

    def _electricity_expr(self) -> pl.Expr:
        return pl.col("Sector").cast(pl.Utf8).str.to_lowercase().str.contains(
            "electricity|gas and water|utilities|power|water|gas"
        )

    def _share(self, frame: pl.DataFrame, expr: pl.Expr) -> float:
        if frame.is_empty():
            return 0.0
        return _as_float(frame.select(expr.mean()).item())

    def _percentile_of_value(self, series: pl.Series, value: float) -> float:
        clean = series.drop_nulls()
        if clean.is_empty() or math.isnan(value):
            return float("nan")
        return _as_float((clean <= value).sum() / clean.len())

    def _quantile_or_zero(self, frame: pl.DataFrame, column: str, quantile: float) -> float:
        if column not in frame.columns:
            return 0.0
        values = frame[column].drop_nulls()
        return 0.0 if values.is_empty() else _as_float(values.quantile(quantile), 0.0)

    def _rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.paths.project_root))
        except ValueError:
            return str(path)

    def _markdown_table(self, frame: pl.DataFrame) -> str:
        if frame.is_empty():
            return "_No rows available._"
        columns = frame.columns
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
        for row in frame.to_dicts():
            lines.append("| " + " | ".join(self._format_markdown_value(row.get(column)) for column in columns) + " |")
        return "\n".join(lines)

    def _format_markdown_value(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        if value is None:
            return ""
        return str(value).replace("|", "/")


class EssentialInputDependenceDiagnostics:
    """Diagnose whether transition inertia is grounded in IO essential-input dependence."""

    def __init__(self, paths: ABMV4Paths, start_year: int = 1995, end_year: int = 2016) -> None:
        self.paths = paths
        self.start_year = start_year
        self.end_year = end_year

    def run(self) -> EssentialInputDependenceDiagnosticResult:
        """Build all Phase 22 diagnostic artifacts in memory."""
        panel = self.build_supplier_buyer_dependence_panel()
        if panel.is_empty():
            raise FileNotFoundError(
                "No compact supplier-buyer dependence panel could be built. Run supplier opportunity/rewiring "
                "outputs first, or provide data/abm_v4/interim/historical_supplier_edges.parquet."
            )
        node_metrics = self.compute_essential_input_metrics(panel)
        contrast = self.contrast_electricity_vs_non_electricity(node_metrics)
        symptom = self.compare_dependence_to_symptom_metrics(node_metrics)
        screening = self.screen_dependence_metrics_against_labels(node_metrics)
        lookalikes = self.identify_structural_dependence_lookalikes(node_metrics)
        proxies = self.build_candidate_proxy_table(node_metrics, screening, lookalikes, symptom)
        recommendation = self.build_recommendation(node_metrics, screening, lookalikes, proxies, symptom)
        markdown = self.build_markdown_report(
            panel=panel,
            node_metrics=node_metrics,
            contrast=contrast,
            symptom=symptom,
            screening=screening,
            lookalikes=lookalikes,
            proxies=proxies,
            recommendation=recommendation,
        )
        return EssentialInputDependenceDiagnosticResult(
            supplier_buyer_panel=panel,
            node_metrics=node_metrics,
            electricity_contrast=contrast,
            symptom_comparison=symptom,
            metric_screening=screening,
            lookalikes=lookalikes,
            candidate_proxies=proxies,
            recommendation=recommendation,
            markdown=markdown,
        )

    def write_outputs(self, result: EssentialInputDependenceDiagnosticResult) -> None:
        """Write Phase 22 outputs. The caller controls whether this is invoked."""
        self.paths.validation.mkdir(parents=True, exist_ok=True)
        result.supplier_buyer_panel.write_parquet(self.paths.essential_input_supplier_buyer_panel_path)
        result.node_metrics.write_csv(self.paths.essential_input_node_metrics_path)
        result.electricity_contrast.write_csv(self.paths.electricity_dependence_signature_contrast_path)
        result.symptom_comparison.write_csv(self.paths.dependence_vs_symptom_metric_comparison_path)
        result.metric_screening.write_csv(self.paths.dependence_metric_screening_path)
        result.lookalikes.write_csv(self.paths.essential_input_non_electricity_lookalikes_path)
        result.candidate_proxies.write_csv(self.paths.candidate_structural_dependence_proxies_path)
        result.recommendation.write_csv(self.paths.essential_input_dependence_recommendation_path)
        self.paths.essential_input_dependence_report_path.write_text(result.markdown, encoding="utf-8")

    def load_state_panel(self) -> pl.DataFrame:
        """Load compact state metadata and observed scale fields."""
        state_path = self.paths.state_panel_path(self.start_year, self.end_year)
        if not state_path.exists():
            raise FileNotFoundError(f"Missing ABM v4 state panel: {state_path}. Run --build-state first.")
        state = pl.read_parquet(state_path)
        if "Year" in state.columns and "year" not in state.columns:
            state = state.rename({"Year": "year"})
        if "EI_observed" not in state.columns and "EI" in state.columns:
            state = state.rename({"EI": "EI_observed"})
        keep = [
            c
            for c in [
                "country_sector",
                "Country",
                "Sector",
                "year",
                "ecosystem_label",
                "X_observed",
                "emissions_observed",
                "EI_observed",
                "brown_centrality",
                "network_green_exposure",
            ]
            if c in state.columns
        ]
        return state.select(keep)

    def load_compact_supplier_edges(self) -> pl.DataFrame:
        """Load compact supplier relationships without touching raw-T full edge rebuilds."""
        frames: list[pl.DataFrame] = []
        if self.paths.supplier_updated_weights_path.exists():
            weights = pl.read_parquet(self.paths.supplier_updated_weights_path)
            if {"buyer_country_sector", "supplier_country_sector"}.issubset(weights.columns):
                weight_col = "updated_weight" if "updated_weight" in weights.columns else "weight"
                keep = [
                    c
                    for c in [
                        "buyer_country_sector",
                        "supplier_country_sector",
                        "supplier_type",
                        "candidate_sources",
                        "initial_weight",
                        "updated_weight",
                        "choice_probability",
                    ]
                    if c in weights.columns
                ]
                frame = weights.select(keep).rename({weight_col: "supplier_weight"} if weight_col in keep else {})
                if "supplier_weight" not in frame.columns:
                    frame = frame.with_columns(pl.lit(1.0).alias("supplier_weight"))
                frame = frame.with_columns(
                    pl.lit(None).cast(pl.Int64).alias("year"),
                    pl.lit("supplier_updated_weights").alias("observed_or_candidate_source"),
                )
                frames.append(frame)
        if self.paths.historical_supplier_edges_path.exists():
            hist = pl.read_parquet(self.paths.historical_supplier_edges_path)
            if {"buyer_country_sector", "supplier_country_sector"}.issubset(hist.columns):
                keep = [
                    c
                    for c in [
                        "year",
                        "buyer_country_sector",
                        "supplier_country_sector",
                        "supplier_country",
                        "supplier_sector",
                        "buyer_country",
                        "buyer_sector",
                        "supplier_ecosystem_label",
                        "buyer_ecosystem_label",
                        "transaction_value",
                        "historical_share",
                        "historical_tie_strength",
                    ]
                    if c in hist.columns
                ]
                frame = hist.select(keep)
                if "historical_share" in frame.columns:
                    frame = frame.rename({"historical_share": "supplier_weight"})
                elif "transaction_value" in frame.columns:
                    totals = frame.group_by(["buyer_country_sector", "year"]).agg(
                        pl.col("transaction_value").sum().alias("_buyer_total")
                    )
                    frame = frame.join(totals, on=["buyer_country_sector", "year"], how="left").with_columns(
                        (pl.col("transaction_value") / pl.col("_buyer_total")).fill_nan(0).alias("supplier_weight")
                    ).drop("_buyer_total")
                else:
                    frame = frame.with_columns(pl.lit(1.0).alias("supplier_weight"))
                frame = frame.with_columns(pl.lit("historical_supplier_edges").alias("observed_or_candidate_source"))
                frames.append(frame)
        if not frames:
            return pl.DataFrame()
        common = sorted(set().union(*(set(frame.columns) for frame in frames)))
        aligned = []
        for frame in frames:
            aligned.append(
                frame.with_columns(
                    [pl.lit(None).alias(column) for column in common if column not in frame.columns]
                ).select(common)
            )
        return pl.concat(aligned, how="vertical_relaxed")

    def build_supplier_buyer_dependence_panel(self) -> pl.DataFrame:
        """Build supplier-buyer dependence panel from compact supplier and state data."""
        edges = self.load_compact_supplier_edges()
        if edges.is_empty():
            return edges
        state = self.load_state_panel()
        latest_state = (
            state.sort("year")
            .group_by("country_sector")
            .agg(
                pl.col("Country").drop_nulls().last().alias("Country"),
                pl.col("Sector").drop_nulls().last().alias("Sector"),
                pl.col("ecosystem_label").drop_nulls().last().alias("ecosystem"),
                pl.col("X_observed").mean().alias("mean_output"),
                pl.col("emissions_observed").mean().alias("mean_emissions"),
            )
        )
        supplier_meta = latest_state.rename(
            {
                "country_sector": "supplier_country_sector",
                "Country": "supplier_country",
                "Sector": "supplier_sector",
                "ecosystem": "supplier_ecosystem",
                "mean_output": "supplier_output",
                "mean_emissions": "supplier_emissions",
            }
        )
        buyer_meta = latest_state.rename(
            {
                "country_sector": "buyer_country_sector",
                "Country": "buyer_country",
                "Sector": "buyer_sector",
                "ecosystem": "buyer_ecosystem",
                "mean_output": "buyer_output",
                "mean_emissions": "buyer_emissions",
            }
        )
        panel = edges.join(supplier_meta, on="supplier_country_sector", how="left", suffix="_state")
        panel = panel.join(buyer_meta, on="buyer_country_sector", how="left", suffix="_state")
        for original, fallback in [
            ("supplier_country", "supplier_country_state"),
            ("supplier_sector", "supplier_sector_state"),
            ("supplier_ecosystem_label", "supplier_ecosystem"),
            ("buyer_country", "buyer_country_state"),
            ("buyer_sector", "buyer_sector_state"),
            ("buyer_ecosystem_label", "buyer_ecosystem"),
        ]:
            if original in panel.columns and fallback in panel.columns:
                panel = panel.with_columns(pl.coalesce(pl.col(original), pl.col(fallback)).alias(original)).drop(fallback)
            elif fallback in panel.columns and original not in panel.columns:
                panel = panel.rename({fallback: original})
        if "supplier_weight" not in panel.columns:
            panel = panel.with_columns(pl.lit(1.0).alias("supplier_weight"))
        if "transaction_value" not in panel.columns:
            panel = panel.with_columns(pl.lit(None).cast(pl.Float64).alias("transaction_value"))
        panel = panel.with_columns(
            pl.col("supplier_weight").fill_null(0).clip(0, None).alias("supplier_share_in_buyer_inputs"),
            pl.col("buyer_output").alias("buyer_total_input_proxy"),
            pl.when(pl.col("transaction_value").is_not_null())
            .then(pl.col("transaction_value"))
            .otherwise(pl.col("supplier_weight"))
            .alias("transaction_value"),
            pl.lit("compact supplier diagnostics; raw-T not loaded").alias("notes"),
        )
        required = [
            "supplier_country_sector",
            "buyer_country_sector",
            "year",
            "supplier_country",
            "supplier_sector",
            "buyer_country",
            "buyer_sector",
            "supplier_ecosystem_label",
            "buyer_ecosystem_label",
            "observed_or_candidate_source",
            "transaction_value",
            "supplier_weight",
            "buyer_total_input_proxy",
            "supplier_share_in_buyer_inputs",
            "buyer_output",
            "buyer_emissions",
            "supplier_output",
            "supplier_emissions",
            "notes",
        ]
        for column in required:
            if column not in panel.columns:
                panel = panel.with_columns(pl.lit(None).alias(column))
        return panel.select(required).rename(
            {
                "supplier_ecosystem_label": "supplier_ecosystem",
                "buyer_ecosystem_label": "buyer_ecosystem",
            }
        )

    def compute_essential_input_metrics(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Aggregate supplier-buyer dependence metrics to supplier nodes."""
        all_buyers = max(panel["buyer_country_sector"].n_unique(), 1)
        all_buyer_sectors = max(panel["buyer_sector"].drop_nulls().n_unique(), 1)
        all_buyer_ecosystems = max(panel["buyer_ecosystem"].drop_nulls().n_unique(), 1)
        total_buyer_output = panel.select("buyer_country_sector", "buyer_output").unique()["buyer_output"].sum() or 1.0
        total_buyer_emissions = panel.select("buyer_country_sector", "buyer_emissions").unique()["buyer_emissions"].sum() or 1.0
        latest = self._node_metadata()
        by_supplier_buyer = (
            panel.group_by("supplier_country_sector", "buyer_country_sector")
            .agg(
                pl.col("supplier_country").drop_nulls().first().alias("Country"),
                pl.col("supplier_sector").drop_nulls().first().alias("Sector"),
                pl.col("supplier_ecosystem").drop_nulls().first().alias("ecosystem"),
                pl.col("buyer_country").drop_nulls().first().alias("buyer_country"),
                pl.col("buyer_sector").drop_nulls().first().alias("buyer_sector"),
                pl.col("buyer_ecosystem").drop_nulls().first().alias("buyer_ecosystem"),
                pl.col("supplier_share_in_buyer_inputs").mean().alias("supplier_share_in_buyer_inputs"),
                pl.col("buyer_output").max().alias("buyer_output"),
                pl.col("buyer_emissions").max().alias("buyer_emissions"),
                pl.col("supplier_output").max().alias("supplier_output"),
                pl.col("supplier_emissions").max().alias("supplier_emissions"),
                pl.col("year").drop_nulls().n_unique().alias("years_active"),
                pl.col("observed_or_candidate_source").str.contains("historical").max().alias("has_historical_source"),
            )
            .with_columns(
                (
                    pl.col("supplier_share_in_buyer_inputs")
                    / (pl.col("supplier_share_in_buyer_inputs").sum().over("supplier_country_sector") + 1e-12)
                )
                .fill_nan(0)
                .alias("_buyer_weight")
            )
        )
        rows = (
            by_supplier_buyer.group_by("supplier_country_sector")
            .agg(
                pl.col("Country").drop_nulls().first().alias("Country"),
                pl.col("Sector").drop_nulls().first().alias("Sector"),
                pl.col("ecosystem").drop_nulls().first().alias("ecosystem"),
                pl.col("buyer_country_sector").n_unique().alias("buyer_count"),
                pl.col("buyer_country").n_unique().alias("buyer_country_count"),
                pl.col("buyer_sector").n_unique().alias("buyer_sector_count"),
                pl.col("buyer_ecosystem").n_unique().alias("buyer_ecosystem_count"),
                (pl.col("_buyer_weight") ** 2).sum().alias("buyer_hhi"),
                pl.col("supplier_share_in_buyer_inputs").mean().alias("mean_supplier_share_in_buyer_inputs"),
                pl.col("supplier_share_in_buyer_inputs").median().alias("median_supplier_share_in_buyer_inputs"),
                pl.col("supplier_share_in_buyer_inputs").quantile(0.95).alias("p95_supplier_share_in_buyer_inputs"),
                pl.col("supplier_share_in_buyer_inputs").max().alias("max_supplier_share_in_buyer_inputs"),
                (pl.col("supplier_share_in_buyer_inputs") >= 0.01).sum().alias("buyers_dependency_above_1pct"),
                (pl.col("supplier_share_in_buyer_inputs") >= 0.05).sum().alias("buyers_dependency_above_5pct"),
                (pl.col("supplier_share_in_buyer_inputs") >= 0.10).sum().alias("buyers_dependency_above_10pct"),
                (pl.col("buyer_output") * pl.col("supplier_share_in_buyer_inputs")).sum().alias("downstream_output_exposure"),
                (pl.col("buyer_emissions") * pl.col("supplier_share_in_buyer_inputs")).sum().alias("downstream_emissions_exposure"),
                pl.col("buyer_output").sum().alias("_buyer_output_sum"),
                pl.col("buyer_emissions").sum().alias("_buyer_emissions_sum"),
                pl.col("years_active").mean().alias("mean_years_active_per_buyer"),
                (pl.col("years_active") >= 5).mean().alias("share_buyers_persistent_5yr_or_more"),
                pl.col("supplier_output").max().alias("supplier_output"),
                pl.col("supplier_emissions").max().alias("supplier_emissions"),
            )
            .rename({"supplier_country_sector": "country_sector"})
        )
        rows = rows.with_columns(
            (1.0 / pl.col("buyer_hhi")).fill_nan(0).alias("buyer_diversity_inverse_hhi"),
            (pl.col("buyer_count") / all_buyers).alias("share_of_all_buyers"),
            (pl.col("buyer_sector_count") / all_buyer_sectors).alias("buyer_sector_coverage"),
            (pl.col("buyer_ecosystem_count") / all_buyer_ecosystems).alias("buyer_ecosystem_coverage"),
            (pl.col("buyer_ecosystem_count") > 1).cast(pl.Float64).alias("cross_ecosystem_buyer_share"),
            (pl.col("_buyer_output_sum") / total_buyer_output).alias("buyer_output_share_total"),
            (pl.col("_buyer_emissions_sum") / total_buyer_emissions).alias("buyer_emissions_share_total"),
            pl.col("mean_years_active_per_buyer").fill_null(0).alias("relationship_stability_metric"),
        ).drop(["_buyer_output_sum", "_buyer_emissions_sum"])
        entropy = by_supplier_buyer.with_columns(
            (pl.col("supplier_share_in_buyer_inputs") / pl.col("supplier_share_in_buyer_inputs").sum().over("supplier_country_sector")).fill_nan(0).alias("_p")
        ).with_columns(
            pl.when(pl.col("_p") > 0).then(-pl.col("_p") * pl.col("_p").log()).otherwise(0.0).alias("_entropy_part")
        ).group_by("supplier_country_sector").agg(pl.col("_entropy_part").sum().alias("buyer_entropy")).rename({"supplier_country_sector": "country_sector"})
        rows = rows.join(entropy, on="country_sector", how="left")
        opportunity = self._opportunity_scarcity()
        rows = self._join_optional(rows, opportunity, ["country_sector"])
        rows = self._join_optional(rows, latest, ["country_sector"])
        for column in ["electricity_like", "cumulative_emissions_share", "jump_frequency", "needs_dampening_node", "aggregate_sensitive_node"]:
            if column not in rows.columns:
                rows = rows.with_columns(pl.lit(False if column.endswith("_node") or column == "electricity_like" else 0.0).alias(column))
        rows = rows.with_columns(
            self._normalized_score_expr(["buyer_count", "buyer_sector_coverage", "buyer_ecosystem_coverage"]).alias("essential_input_score_diagnostic"),
            self._normalized_score_expr(["mean_supplier_share_in_buyer_inputs", "opportunity_scarcity_metric", "relationship_stability_metric"]).alias("low_substitutability_score_diagnostic"),
            self._normalized_score_expr(["downstream_output_exposure", "downstream_emissions_exposure", "buyer_output_share_total", "buyer_emissions_share_total"]).alias("systemic_dependence_score_diagnostic"),
        ).with_columns(
            self._normalized_score_expr(
                ["essential_input_score_diagnostic", "low_substitutability_score_diagnostic", "systemic_dependence_score_diagnostic"]
            ).alias("structural_dependence_score_diagnostic"),
            pl.lit("compact updated weights plus historical edges where available").alias("notes"),
        )
        ordered = [
            "country_sector", "Country", "Sector", "electricity_like", "buyer_count", "buyer_country_count",
            "buyer_sector_count", "buyer_ecosystem_count", "buyer_entropy", "buyer_hhi",
            "buyer_diversity_inverse_hhi", "share_of_all_buyers", "buyer_sector_coverage",
            "buyer_ecosystem_coverage", "cross_ecosystem_buyer_share",
            "mean_supplier_share_in_buyer_inputs", "median_supplier_share_in_buyer_inputs",
            "p95_supplier_share_in_buyer_inputs", "max_supplier_share_in_buyer_inputs",
            "buyers_dependency_above_1pct", "buyers_dependency_above_5pct", "buyers_dependency_above_10pct",
            "downstream_output_exposure", "downstream_emissions_exposure", "buyer_output_share_total",
            "buyer_emissions_share_total", "relationship_stability_metric", "opportunity_scarcity_metric",
            "systemic_dependence_score_diagnostic", "essential_input_score_diagnostic",
            "low_substitutability_score_diagnostic", "structural_dependence_score_diagnostic", "notes",
        ]
        for column in ordered:
            if column not in rows.columns:
                rows = rows.with_columns(pl.lit(None).alias(column))
        return rows.select(ordered + [c for c in rows.columns if c not in ordered])

    def contrast_electricity_vs_non_electricity(self, node: pl.DataFrame) -> pl.DataFrame:
        """Contrast electricity-like nodes against others on dependence metrics."""
        rows = []
        for metric in self._dependence_metrics(node):
            frame = node.select(metric, "electricity_like").drop_nulls()
            if frame.height < 3 or frame["electricity_like"].n_unique() < 2:
                continue
            elec = frame.filter(pl.col("electricity_like"))[metric]
            other = frame.filter(~pl.col("electricity_like"))[metric]
            std = _as_float(frame[metric].std(), 0.0)
            elec_mean = _as_float(elec.mean())
            other_mean = _as_float(other.mean())
            sd = 0.0 if std == 0 else (elec_mean - other_mean) / std
            rows.append(
                {
                    "metric": metric,
                    "metric_family": self._dependence_family(metric),
                    "electricity_mean": elec_mean,
                    "non_electricity_mean": other_mean,
                    "electricity_median": _as_float(elec.median()),
                    "non_electricity_median": _as_float(other.median()),
                    "electricity_p25": _as_float(elec.quantile(0.25)),
                    "electricity_p75": _as_float(elec.quantile(0.75)),
                    "non_electricity_p25": _as_float(other.quantile(0.25)),
                    "non_electricity_p75": _as_float(other.quantile(0.75)),
                    "standardized_difference": sd,
                    "electricity_median_percentile_in_all_nodes": self._percentile_of_value(frame[metric], _as_float(elec.median())),
                    "interpretation": "strong structural dependence contrast" if abs(sd) >= 0.5 else "weak/moderate contrast",
                    "theoretical_relevance": self._dependence_family(metric),
                }
            )
        return pl.DataFrame(rows).sort("standardized_difference", descending=True) if rows else pl.DataFrame()

    def compare_dependence_to_symptom_metrics(self, node: pl.DataFrame) -> pl.DataFrame:
        """Compare dependence scores with Phase 21 symptom metrics."""
        dependence = [
            "essential_input_score_diagnostic",
            "structural_dependence_score_diagnostic",
            "systemic_dependence_score_diagnostic",
        ]
        symptoms = [
            "mean_log_EI_observed",
            "cumulative_emissions_share",
            "jump_frequency",
            "share_frontier_gap_worsens_emissions_error",
            "mean_contribution_to_aggregate_error_difference",
        ]
        rows = []
        for dep in dependence:
            if dep not in node.columns:
                continue
            dep_threshold = self._quantile_or_zero(node, dep, 0.90)
            for symptom in symptoms:
                if symptom not in node.columns:
                    continue
                frame = node.select(dep, symptom).drop_nulls()
                if frame.height < 3:
                    continue
                symptom_threshold = self._quantile_or_zero(frame, symptom, 0.90)
                corr = _as_float(frame.select(pl.corr(dep, symptom)).item())
                overlap = _as_float(frame.select(((pl.col(dep) >= dep_threshold) & (pl.col(symptom) >= symptom_threshold)).mean()).item())
                rows.append(
                    {
                        "dependence_metric": dep,
                        "symptom_metric": symptom,
                        "correlation": corr,
                        "overlap_top_decile_share": overlap / 0.10 if overlap else 0.0,
                        "interpretation": "possibly redundant with symptom metric" if abs(corr) >= 0.7 else "adds distinct structural information",
                    }
                )
        return pl.DataFrame(rows)

    def screen_dependence_metrics_against_labels(self, node: pl.DataFrame) -> pl.DataFrame:
        """Screen dependence metrics against Phase 21 labels."""
        labels = ["electricity_like", "high_emissions_node", "jump_prone_node", "aggregate_sensitive_node", "needs_dampening_node"]
        rows = []
        for label in labels:
            if label not in node.columns:
                continue
            for metric in self._dependence_metrics(node):
                frame = node.select(label, metric).drop_nulls()
                if frame.height < 3 or frame[label].n_unique() < 2:
                    continue
                pos = frame.filter(pl.col(label))[metric]
                neg = frame.filter(~pl.col(label))[metric]
                if pos.is_empty() or neg.is_empty():
                    continue
                std = _as_float(frame[metric].std(), 0.0)
                sd = 0.0 if std == 0 else (_as_float(pos.mean()) - _as_float(neg.mean())) / std
                corr = _as_float(frame.select(pl.corr(pl.col(metric), pl.col(label).cast(pl.Float64))).item())
                score = abs(sd) + abs(corr)
                structural = self._dependence_family(metric) not in {"weak_or_unclear"}
                rows.append(
                    {
                        "label_name": label,
                        "metric": metric,
                        "metric_family": self._dependence_family(metric),
                        "usable_rows": frame.height,
                        "separation_score": score,
                        "abs_correlation_or_auc": abs(corr),
                        "standardized_difference": sd,
                        "interpretation": "strong structural signal" if score >= 0.8 and structural else "screening signal",
                        "recommended_for_phase23": score >= 0.8 and structural and label in {"electricity_like", "aggregate_sensitive_node", "needs_dampening_node"},
                    }
                )
        out = pl.DataFrame(rows)
        if out.is_empty():
            return out
        return out.sort(["label_name", "separation_score"], descending=[False, True]).with_columns(
            pl.col("separation_score").rank(method="ordinal", descending=True).over("label_name").alias("rank")
        )

    def identify_structural_dependence_lookalikes(self, node: pl.DataFrame) -> pl.DataFrame:
        """Find non-electricity nodes with high structural dependence scores."""
        if "structural_dependence_score_diagnostic" not in node.columns:
            return pl.DataFrame()
        return (
            node.filter(~pl.col("electricity_like"))
            .sort("structural_dependence_score_diagnostic", descending=True)
            .head(50)
            .select(
                "country_sector",
                "Country",
                "Sector",
                pl.col("ecosystem").alias("ecosystem") if "ecosystem" in node.columns else pl.lit(None).alias("ecosystem"),
                pl.col("structural_dependence_score_diagnostic").alias("structural_dependence_score"),
                pl.col("essential_input_score_diagnostic").alias("essential_input_score"),
                pl.col("low_substitutability_score_diagnostic").alias("low_substitutability_score"),
                pl.col("systemic_dependence_score_diagnostic").alias("systemic_dependence_score"),
                "buyer_count",
                "buyer_sector_coverage",
                "buyer_ecosystem_coverage",
                "downstream_output_exposure",
                "downstream_emissions_exposure",
                pl.col("cumulative_emissions_share").fill_null(0).alias("cumulative_emissions_share"),
                pl.col("jump_frequency").fill_null(0).alias("jump_frequency"),
                pl.col("share_frontier_gap_worsens_emissions_error").fill_null(0).alias("needs_dampening_score"),
                pl.when(pl.col("structural_dependence_score_diagnostic") >= 0.7)
                .then(pl.lit("plausible structural-dependence lookalike"))
                .otherwise(pl.lit("weaker lookalike"))
                .alias("interpretation"),
            )
        )

    def build_candidate_proxy_table(
        self,
        node: pl.DataFrame,
        screening: pl.DataFrame,
        lookalikes: pl.DataFrame,
        symptom: pl.DataFrame,
    ) -> pl.DataFrame:
        """Build candidate structural-dependence proxy table."""
        proxy_defs = [
            ("essential_input_score", "buyer reach + sector/ecosystem coverage", ["buyer_count", "buyer_sector_coverage", "buyer_ecosystem_coverage"]),
            ("input_universality_score", "share of all buyers plus sector and ecosystem coverage", ["share_of_all_buyers", "buyer_sector_coverage", "buyer_ecosystem_coverage"]),
            ("buyer_dependence_score", "mean and tail supplier share in buyer inputs", ["mean_supplier_share_in_buyer_inputs", "p95_supplier_share_in_buyer_inputs", "buyers_dependency_above_5pct"]),
            ("low_substitutability_score", "buyer dependence plus opportunity scarcity and relationship stability", ["low_substitutability_score_diagnostic", "opportunity_scarcity_metric", "relationship_stability_metric"]),
            ("systemic_dependence_score", "downstream output and emissions exposure", ["systemic_dependence_score_diagnostic", "downstream_output_exposure", "downstream_emissions_exposure"]),
            ("structural_dependence_score", "essential input + low substitutability + systemic propagation", ["structural_dependence_score_diagnostic"]),
            ("structural_dependence_plus_brown_lockin", "structural dependence plus brown centrality", ["structural_dependence_score_diagnostic", "mean_brown_centrality"]),
            ("structural_dependence_plus_volatility", "structural dependence plus jump/volatility regime", ["structural_dependence_score_diagnostic", "jump_frequency"]),
        ]
        lookalike_count = lookalikes.filter(pl.col("structural_dependence_score") >= 0.7).height if not lookalikes.is_empty() else 0
        top_sectors = ";".join(lookalikes.head(8)["Sector"].to_list()) if not lookalikes.is_empty() and "Sector" in lookalikes.columns else ""
        rows = []
        for name, formula, required in proxy_defs:
            available = [metric for metric in required if metric in node.columns]
            relationship = self._proxy_relationship(screening, "needs_dampening_node", available)
            aggregate = self._proxy_relationship(screening, "aggregate_sensitive_node", available)
            redundancy = self._max_symptom_correlation(symptom, available)
            rows.append(
                {
                    "candidate_proxy": name,
                    "formula_description": formula,
                    "required_metrics": ";".join(required),
                    "availability_status": "available" if len(available) == len(required) else ("partial" if available else "unavailable"),
                    "theoretical_interpretation": self._proxy_theory(name),
                    "electricity_percentile": self._proxy_electricity_percentile(node, available),
                    "relation_to_needs_dampening": relationship,
                    "relation_to_aggregate_sensitivity": aggregate,
                    "non_electricity_lookalike_count": lookalike_count,
                    "top_lookalike_sectors": top_sectors,
                    "risks": self._proxy_risks(name, redundancy),
                    "recommended_for_phase23": bool(available) and relationship != "weak_or_unknown" and redundancy < 0.85,
                }
            )
        return pl.DataFrame(rows)

    def build_recommendation(
        self,
        node: pl.DataFrame,
        screening: pl.DataFrame,
        lookalikes: pl.DataFrame,
        proxies: pl.DataFrame,
        symptom: pl.DataFrame,
    ) -> pl.DataFrame:
        """Apply Phase 22 decision rules."""
        if node.is_empty() or proxies.is_empty():
            recommendation = "dependence_metrics_insufficient"
            interpretation = "Dependence metrics could not be computed robustly from compact inputs."
        else:
            elec = screening.filter(pl.col("label_name") == "electricity_like") if not screening.is_empty() else pl.DataFrame()
            damp = screening.filter(pl.col("label_name") == "needs_dampening_node") if not screening.is_empty() else pl.DataFrame()
            strong_elec = "separation_score" in elec.columns and not elec.filter(pl.col("separation_score") >= 0.8).is_empty()
            strong_damp = "separation_score" in damp.columns and not damp.filter(pl.col("separation_score") >= 0.8).is_empty()
            lookalike_count = lookalikes.filter(pl.col("structural_dependence_score") >= 0.7).height if not lookalikes.is_empty() else 0
            max_corr = _as_float(symptom["correlation"].abs().max()) if not symptom.is_empty() else 0.0
            if max_corr >= 0.9 and not strong_damp:
                recommendation = "dependence_metrics_insufficient"
                interpretation = "Dependence metrics are too redundant with symptom metrics and do not explain dampening need."
            elif strong_elec and strong_damp and lookalike_count > 0:
                recommendation = "build_structural_dependence_dampener"
                interpretation = "Dependence metrics distinguish electricity, relate to dampening need, and identify non-electricity lookalikes."
            elif strong_elec and lookalike_count == 0:
                recommendation = "keep_electricity_specific_diagnostic_only"
                interpretation = "Dependence metrics identify electricity but do not generalize to plausible non-electricity nodes."
            elif strong_elec:
                recommendation = "build_essential_input_dampener"
                interpretation = "Dependence metrics identify electricity-like essential input structure, but dampening evidence is weaker."
            else:
                recommendation = "inconclusive"
                interpretation = "Dependence metrics are available but do not clearly support a rule yet."
        evidence = (
            f"node_rows={node.height}; lookalikes={lookalikes.height if not lookalikes.is_empty() else 0}; "
            f"recommended_proxies={proxies.filter(pl.col('recommended_for_phase23')).height if not proxies.is_empty() else 0}"
        )
        return pl.DataFrame(
            [
                {
                    "recommendation": recommendation,
                    "evidence": evidence,
                    "interpretation": interpretation,
                    "recommended_phase23": "Build only a diagnostic structural-dependence dampener candidate if evidence is strong; keep scenarios blocked.",
                    "scenario_readiness": "premature",
                }
            ]
        )

    def build_markdown_report(
        self,
        *,
        panel: pl.DataFrame,
        node_metrics: pl.DataFrame,
        contrast: pl.DataFrame,
        symptom: pl.DataFrame,
        screening: pl.DataFrame,
        lookalikes: pl.DataFrame,
        proxies: pl.DataFrame,
        recommendation: pl.DataFrame,
    ) -> str:
        lines = [
            "# ABM v4 Phase 22 Essential-Input Dependence Diagnostics",
            "",
            f"Supplier-buyer rows: {panel.height}; supplier nodes: {node_metrics.height}.",
            f"Recommendation: `{recommendation['recommendation'].item(0) if not recommendation.is_empty() else 'inconclusive'}`.",
            "",
            "## Recommendation",
            self._markdown_table(recommendation),
            "",
            "## Electricity Dependence Contrast",
            self._markdown_table(contrast.head(40)),
            "",
            "## Dependence Versus Symptom Metrics",
            self._markdown_table(symptom.sort("correlation", descending=True).head(40) if not symptom.is_empty() else symptom),
            "",
            "## Dependence Metric Screening",
            self._markdown_table(screening.sort("separation_score", descending=True).head(60) if not screening.is_empty() else screening),
            "",
            "## Non-Electricity Dependence Lookalikes",
            self._markdown_table(lookalikes.head(40)),
            "",
            "## Candidate Structural-Dependence Proxies",
            self._markdown_table(proxies),
            "",
            "Scenarios remain premature.",
        ]
        return "\n".join(lines) + "\n"

    def _node_metadata(self) -> pl.DataFrame:
        if self.paths.structural_signature_node_panel_path.exists():
            return pl.read_parquet(self.paths.structural_signature_node_panel_path)
        state = self.load_state_panel()
        total_emissions = state["emissions_observed"].sum() or 1.0
        total_output = state["X_observed"].sum() or 1.0
        return state.group_by("country_sector").agg(
            pl.col("Country").drop_nulls().last().alias("Country"),
            pl.col("Sector").drop_nulls().last().alias("Sector"),
            pl.col("ecosystem_label").drop_nulls().last().alias("ecosystem"),
            pl.col("Sector").drop_nulls().last().str.to_lowercase().str.contains("electricity|gas and water|utilities|power|water|gas").alias("electricity_like"),
            (pl.col("emissions_observed").sum() / total_emissions).alias("cumulative_emissions_share"),
            (pl.col("X_observed").sum() / total_output).alias("cumulative_output_share"),
            pl.lit(False).alias("needs_dampening_node"),
            pl.lit(False).alias("aggregate_sensitive_node"),
            pl.lit(0.0).alias("jump_frequency"),
        )

    def _opportunity_scarcity(self) -> pl.DataFrame:
        if not self.paths.supplier_opportunity_sets_path.exists():
            return pl.DataFrame()
        frame = pl.read_parquet(self.paths.supplier_opportunity_sets_path)
        if "supplier_country_sector" not in frame.columns or "buyer_country_sector" not in frame.columns:
            return pl.DataFrame()
        by_buyer = frame.group_by("buyer_country_sector").agg(
            pl.col("supplier_country_sector").n_unique().alias("_opportunity_count"),
            pl.col("supplier_type").n_unique().alias("_supplier_type_count") if "supplier_type" in frame.columns else pl.lit(None).alias("_supplier_type_count"),
            pl.col("is_historical_candidate").mean().alias("_historical_share") if "is_historical_candidate" in frame.columns else pl.lit(None).alias("_historical_share"),
            pl.col("is_same_sector_candidate").mean().alias("_same_sector_share") if "is_same_sector_candidate" in frame.columns else pl.lit(None).alias("_same_sector_share"),
            pl.col("is_ecosystem_candidate").mean().alias("_ecosystem_share") if "is_ecosystem_candidate" in frame.columns else pl.lit(None).alias("_ecosystem_share"),
        )
        joined = frame.select("supplier_country_sector", "buyer_country_sector").unique().join(by_buyer, on="buyer_country_sector", how="left")
        return joined.group_by("supplier_country_sector").agg(
            pl.col("_opportunity_count").mean().alias("mean_buyer_supplier_opportunity_count"),
            pl.col("_opportunity_count").median().alias("median_buyer_supplier_opportunity_count"),
            (1.0 / pl.col("_opportunity_count")).mean().alias("opportunity_scarcity_metric"),
            (1.0 / pl.col("_supplier_type_count")).mean().alias("supplier_type_concentration"),
            pl.col("_historical_share").mean().alias("share_historical_supplier_relationships"),
            pl.col("_same_sector_share").mean().alias("share_same_sector_alternatives"),
            pl.col("_ecosystem_share").mean().alias("share_ecosystem_alternatives"),
        ).rename({"supplier_country_sector": "country_sector"})

    def _dependence_metrics(self, frame: pl.DataFrame) -> list[str]:
        names = [
            "buyer_count", "buyer_country_count", "buyer_sector_count", "buyer_ecosystem_count", "buyer_entropy",
            "buyer_hhi", "buyer_diversity_inverse_hhi", "share_of_all_buyers", "buyer_sector_coverage",
            "buyer_ecosystem_coverage", "cross_ecosystem_buyer_share", "mean_supplier_share_in_buyer_inputs",
            "median_supplier_share_in_buyer_inputs", "p95_supplier_share_in_buyer_inputs", "max_supplier_share_in_buyer_inputs",
            "buyers_dependency_above_1pct", "buyers_dependency_above_5pct", "buyers_dependency_above_10pct",
            "downstream_output_exposure", "downstream_emissions_exposure", "buyer_output_share_total",
            "buyer_emissions_share_total", "relationship_stability_metric", "opportunity_scarcity_metric",
            "systemic_dependence_score_diagnostic", "essential_input_score_diagnostic",
            "low_substitutability_score_diagnostic", "structural_dependence_score_diagnostic",
        ]
        return [name for name in names if name in frame.columns]

    def _dependence_family(self, metric: str) -> str:
        metric_l = metric.lower()
        if "essential_input" in metric_l or "buyer_count" in metric_l or "buyer_country_count" in metric_l or "entropy" in metric_l or "hhi" in metric_l:
            return "essential_input"
        if "coverage" in metric_l or "share_of_all" in metric_l or "cross_ecosystem" in metric_l or "buyer_sector_count" in metric_l or "buyer_ecosystem_count" in metric_l:
            return "input_universality"
        if "supplier_share" in metric_l or "dependency" in metric_l:
            return "buyer_dependence"
        if "scarcity" in metric_l or "stability" in metric_l or "substitutability" in metric_l:
            return "low_substitutability"
        if "downstream" in metric_l or "systemic" in metric_l or "buyer_output_share" in metric_l or "buyer_emissions_share" in metric_l:
            return "systemic_propagation"
        if "structural_dependence" in metric_l:
            return "systemic_propagation"
        return "weak_or_unclear"

    def _normalized_score_expr(self, columns: list[str]) -> pl.Expr:
        exprs = []
        for column in columns:
            exprs.append((pl.col(column).fill_null(0) - pl.col(column).fill_null(0).min()) / (pl.col(column).fill_null(0).max() - pl.col(column).fill_null(0).min() + 1e-12))
        return pl.mean_horizontal(exprs) if exprs else pl.lit(0.0)

    def _proxy_electricity_percentile(self, node: pl.DataFrame, metrics: list[str]) -> float:
        if not metrics or "electricity_like" not in node.columns or node.filter(pl.col("electricity_like")).is_empty():
            return float("nan")
        values = []
        for metric in metrics:
            if metric not in node.columns:
                continue
            elec = _as_float(node.filter(pl.col("electricity_like"))[metric].median())
            values.append(self._percentile_of_value(node[metric].drop_nulls(), elec))
        return sum(values) / len(values) if values else float("nan")

    def _proxy_relationship(self, screening: pl.DataFrame, label: str, metrics: list[str]) -> str:
        if screening.is_empty() or not metrics:
            return "weak_or_unknown"
        rows = screening.filter((pl.col("label_name") == label) & pl.col("metric").is_in(metrics))
        if rows.is_empty():
            return "weak_or_unknown"
        return "strong" if rows["separation_score"].max() >= 0.8 else "moderate_or_weak"

    def _max_symptom_correlation(self, symptom: pl.DataFrame, metrics: list[str]) -> float:
        if symptom.is_empty():
            return 0.0
        aliases = {
            "essential_input_score": "essential_input_score_diagnostic",
            "structural_dependence_score": "structural_dependence_score_diagnostic",
            "systemic_dependence_score": "systemic_dependence_score_diagnostic",
        }
        rows = symptom.filter(pl.col("dependence_metric").is_in([aliases.get(m, m) for m in metrics]))
        return _as_float(rows["correlation"].abs().max()) if not rows.is_empty() else 0.0

    def _proxy_theory(self, proxy: str) -> str:
        return {
            "essential_input_score": "Captures broad buyer reach and input universality.",
            "input_universality_score": "Captures use across many sectors and ecosystems.",
            "buyer_dependence_score": "Captures how strongly buyers depend on the supplier.",
            "low_substitutability_score": "Captures dependence, scarce alternatives, and persistent relationships.",
            "systemic_dependence_score": "Captures downstream propagation potential.",
            "structural_dependence_score": "Combines essential input, low substitutability, and propagation potential.",
            "structural_dependence_plus_brown_lockin": "Adds fossil/brown lock-in to structural dependence.",
            "structural_dependence_plus_volatility": "Adds structural-break volatility to structural dependence.",
        }.get(proxy, "Structural-dependence proxy.")

    def _proxy_risks(self, proxy: str, redundancy: float) -> str:
        risk = "High symptom redundancy risk. " if redundancy >= 0.85 else ""
        if "plus" in proxy:
            risk += "Composite may mix mechanism with symptom unless tested carefully."
        elif proxy in {"structural_dependence_score", "systemic_dependence_score"}:
            risk += "Moderate complexity; inspect lookalikes before use."
        else:
            risk += "Single mechanism may be too narrow."
        return risk

    def _join_optional(self, left: pl.DataFrame, right: pl.DataFrame, on: list[str]) -> pl.DataFrame:
        if right.is_empty() or not set(on).issubset(right.columns):
            return left
        columns_to_add = [c for c in right.columns if c not in left.columns or c in on]
        return left.join(right.select(columns_to_add), on=on, how="left")

    def _percentile_of_value(self, series: pl.Series, value: float) -> float:
        clean = series.drop_nulls()
        if clean.is_empty() or math.isnan(value):
            return float("nan")
        return _as_float((clean <= value).sum() / clean.len())

    def _quantile_or_zero(self, frame: pl.DataFrame, column: str, quantile: float) -> float:
        if column not in frame.columns:
            return 0.0
        values = frame[column].drop_nulls()
        return 0.0 if values.is_empty() else _as_float(values.quantile(quantile), 0.0)

    def _markdown_table(self, frame: pl.DataFrame) -> str:
        if frame.is_empty():
            return "_No rows available._"
        columns = frame.columns
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
        for row in frame.to_dicts():
            lines.append("| " + " | ".join(self._format_markdown_value(row.get(column)) for column in columns) + " |")
        return "\n".join(lines)

    def _format_markdown_value(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        if value is None:
            return ""
        return str(value).replace("|", "/")


class EssentialInputDampenerTester:
    """Test diagnostic essential-input dampeners against historical transition validation."""

    train_start_year = 1995
    train_end_year = 2010
    validation_start_year = 2011
    validation_end_year = 2016

    def __init__(self, paths: ABMV4Paths) -> None:
        self.paths = paths

    def run(self) -> EssentialInputDampenerTestResult:
        """Build Phase 23 candidate-test outputs in memory."""
        panel = self.build_evaluation_panel()
        scores = self.normalize_eid_scores()
        grid = self.build_candidate_grid()
        predictions, residual_panel = self.evaluate_candidates(panel, scores, grid)
        validation = self.build_validation_results(predictions, grid)
        by_sector = self.summarize_group(predictions, "Sector", self._best_candidate_ids(validation), "sector")
        by_country = self.summarize_group(predictions, "Country", self._best_candidate_ids(validation), "country")
        by_electricity = self.summarize_group(predictions.with_columns(pl.col("electricity_like").cast(pl.Utf8).alias("electricity_group")), "electricity_group", self._best_candidate_ids(validation), "electricity")
        china = self.summarize_china_electricity(predictions, self._best_candidate_ids(validation))
        by_decile = self.summarize_group(predictions, "EID_decile", self._best_candidate_ids(validation), "EID_decile")
        residual_summary = self.summarize_residuals(residual_panel, predictions)
        mechanism = self.build_mechanism_decomposition(validation, predictions)
        v5 = self.build_abm_v5_implications(validation, mechanism)
        recommendation = self.build_recommendation(validation, mechanism)
        markdown = self.build_markdown_report(validation, mechanism, recommendation, v5)
        return EssentialInputDampenerTestResult(
            candidate_grid=grid,
            scores=scores,
            residual_panel=residual_panel,
            residual_summary=residual_summary,
            validation_results=validation,
            by_sector=by_sector,
            by_country=by_country,
            by_electricity=by_electricity,
            china_electricity=china,
            by_eid_decile=by_decile,
            mechanism_decomposition=mechanism,
            abm_v5_implications=v5,
            recommendation=recommendation,
            markdown=markdown,
        )

    def write_outputs(self, result: EssentialInputDampenerTestResult) -> None:
        """Write Phase 23 outputs. The caller controls whether this is invoked."""
        self.paths.validation.mkdir(parents=True, exist_ok=True)
        result.candidate_grid.write_csv(self.paths.essential_input_dampener_candidate_grid_path)
        result.scores.write_csv(self.paths.essential_input_dampener_scores_path)
        result.residual_panel.write_parquet(self.paths.essential_input_historical_residual_panel_path)
        result.residual_summary.write_csv(self.paths.essential_input_historical_residual_summary_path)
        result.validation_results.write_csv(self.paths.essential_input_dampener_validation_results_path)
        result.by_sector.write_csv(self.paths.essential_input_dampener_by_sector_path)
        result.by_country.write_csv(self.paths.essential_input_dampener_by_country_path)
        result.by_electricity.write_csv(self.paths.essential_input_dampener_by_electricity_path)
        result.china_electricity.write_csv(self.paths.essential_input_dampener_china_electricity_path)
        result.by_eid_decile.write_csv(self.paths.essential_input_dampener_by_EID_decile_path)
        result.mechanism_decomposition.write_csv(self.paths.essential_input_dampener_mechanism_decomposition_path)
        result.abm_v5_implications.write_csv(self.paths.essential_input_dampener_abm_v5_implications_path)
        result.recommendation.write_csv(self.paths.essential_input_dampener_recommendation_path)
        self.paths.essential_input_dampener_report_path.write_text(result.markdown, encoding="utf-8")

    def build_evaluation_panel(self) -> pl.DataFrame:
        """Load transition comparison rows and add sector-year background."""
        if not self.paths.transition_rule_sign_failure_panel_path.exists():
            raise FileNotFoundError(
                f"Missing transition panel: {self.paths.transition_rule_sign_failure_panel_path}. Run Phase 16 first."
            )
        panel = pl.read_parquet(self.paths.transition_rule_sign_failure_panel_path)
        required = {"country_sector", "year", "Country", "Sector", "observed_rEI", "simulated_rEI_frontier_gap", "simulated_rEI_readiness", "X_observed", "emissions_observed"}
        missing = required - set(panel.columns)
        if missing:
            raise ValueError(f"Transition panel is missing required columns: {sorted(missing)}")
        alpha = panel.group_by(["Sector", "year"]).agg(pl.col("observed_rEI").mean().alias("alpha_sector_year"))
        return panel.join(alpha, on=["Sector", "year"], how="left").with_columns(
            pl.when(pl.col("Sector").str.to_lowercase().str.contains("electricity|gas and water|utilities|power|water|gas"))
            .then(True)
            .otherwise(False)
            .alias("electricity_like"),
            pl.when((pl.col("Country").str.to_lowercase().is_in(["chn", "china"])) & (pl.col("Sector").str.to_lowercase().str.contains("electricity|gas and water|utilities|power|water|gas")))
            .then(True)
            .otherwise(False)
            .alias("china_electricity"),
            pl.when(pl.col("year") <= self.train_end_year).then(pl.lit("train")).otherwise(pl.lit("validation")).alias("train_or_validation"),
        )

    def normalize_eid_scores(self) -> pl.DataFrame:
        """Normalize Phase 22 EID scores with p05-p95 robust scaling."""
        if not self.paths.essential_input_node_metrics_path.exists():
            raise FileNotFoundError(
                f"Missing Phase 22 node metrics: {self.paths.essential_input_node_metrics_path}. Run --diagnose-essential-input-dependence first."
            )
        metrics = pl.read_csv(self.paths.essential_input_node_metrics_path)
        score_names = [
            "essential_input_score_diagnostic",
            "low_substitutability_score_diagnostic",
            "systemic_dependence_score_diagnostic",
            "structural_dependence_score_diagnostic",
        ]
        if {"structural_dependence_score_diagnostic", "mean_brown_centrality"} <= set(metrics.columns):
            metrics = metrics.with_columns(
                ((pl.col("structural_dependence_score_diagnostic").fill_null(0) + pl.col("mean_brown_centrality").fill_null(0)) / 2).alias("structural_dependence_plus_brown_lockin")
            )
            score_names.append("structural_dependence_plus_brown_lockin")
        if {"structural_dependence_score_diagnostic", "jump_frequency"} <= set(metrics.columns):
            metrics = metrics.with_columns(
                ((pl.col("structural_dependence_score_diagnostic").fill_null(0) + pl.col("jump_frequency").fill_null(0)) / 2).alias("structural_dependence_plus_volatility")
            )
            score_names.append("structural_dependence_plus_volatility")
        rows = []
        for score in score_names:
            if score not in metrics.columns:
                continue
            values = metrics[score].drop_nulls()
            if values.is_empty():
                continue
            p05 = _as_float(values.quantile(0.05))
            p95 = _as_float(values.quantile(0.95))
            if p95 == p05:
                continue
            frame = metrics.select(
                "country_sector",
                "Country",
                "Sector",
                "electricity_like",
                pl.lit(score).alias("EID_score_name"),
                pl.col(score).alias("EID_raw"),
            ).with_columns(
                pl.lit(p05).alias("p05"),
                pl.lit(p95).alias("p95"),
                pl.col("EID_raw").is_null().alias("missing_flag"),
                ((pl.col("EID_raw") - p05) / (p95 - p05)).clip(0, 1).alias("EID_norm"),
                pl.lit("p05-p95 robust scaling").alias("notes"),
            ).with_columns(
                pl.when(pl.col("EID_norm") >= pl.col("EID_norm").quantile(0.90).over("EID_score_name"))
                .then(True)
                .otherwise(False)
                .alias("high_EID_decile")
            )
            rows.append(frame)
        return pl.concat(rows, how="vertical_relaxed") if rows else pl.DataFrame()

    def compute_d_eid(self, eid_norm: pl.Expr, lambda_eid: float, d_min: float) -> pl.Expr:
        """Return bounded structural dampener expression."""
        return (1.0 - lambda_eid * eid_norm).clip(d_min, 1.0)

    def build_candidate_grid(self) -> pl.DataFrame:
        """Create transparent Phase 23 candidate grid."""
        rows: list[dict[str, Any]] = []
        cid = 1
        def add(**kwargs: Any) -> None:
            nonlocal cid
            row = {
                "candidate_id": f"c{cid:04d}",
                "variant_name": kwargs.get("variant_name"),
                "EID_score_name": kwargs.get("EID_score_name"),
                "lambda_EID": kwargs.get("lambda_EID"),
                "d_min": kwargs.get("d_min"),
                "uses_EID_dampener": kwargs.get("uses_EID_dampener", False),
                "uses_historical_residual": kwargs.get("uses_historical_residual", False),
                "residual_level": kwargs.get("residual_level"),
                "shrinkage_k": kwargs.get("shrinkage_k"),
                "p_min": kwargs.get("p_min"),
                "p_max": kwargs.get("p_max"),
                "train_start_year": self.train_start_year,
                "train_end_year": self.train_end_year,
                "validation_start_year": self.validation_start_year,
                "validation_end_year": self.validation_end_year,
                "notes": kwargs.get("notes", ""),
            }
            rows.append(row)
            cid += 1
        add(variant_name="frontier_gap_readiness_baseline", notes="existing previous base rule")
        add(variant_name="historical_frontier_gap_only_baseline", notes="existing calibrated-historical rule")
        add(variant_name="electricity_dampened_frontier_gap_0_75_reference", notes="diagnostic reference from Phase 20; not integrated simulation rule")
        score_names = [
            "essential_input_score_diagnostic",
            "low_substitutability_score_diagnostic",
            "systemic_dependence_score_diagnostic",
            "structural_dependence_score_diagnostic",
            "structural_dependence_plus_brown_lockin",
            "structural_dependence_plus_volatility",
        ]
        for score in score_names:
            for lambda_eid in [0.25, 0.50, 0.75, 1.00]:
                for d_min in [0.25, 0.50, 0.75]:
                    add(variant_name="essential_input_dampener_only", EID_score_name=score, lambda_EID=lambda_eid, d_min=d_min, uses_EID_dampener=True)
        for level in ["country_sector", "sector", "country"]:
            for k in [5, 10, 20]:
                for bounds in [(0.75, 1.25), (0.50, 1.50)]:
                    add(variant_name="historical_residual_only", uses_historical_residual=True, residual_level=level, shrinkage_k=k, p_min=bounds[0], p_max=bounds[1])
        for score in score_names[:4]:
            for lambda_eid in [0.50, 0.75]:
                for d_min in [0.50, 0.75]:
                    for k in [10, 20]:
                        for bounds in [(0.75, 1.25), (0.50, 1.50)]:
                            add(variant_name="essential_input_dampener_plus_historical_residual", EID_score_name=score, lambda_EID=lambda_eid, d_min=d_min, uses_EID_dampener=True, uses_historical_residual=True, residual_level="country_sector", shrinkage_k=k, p_min=bounds[0], p_max=bounds[1])
        return pl.DataFrame(rows)

    def evaluate_candidates(self, panel: pl.DataFrame, scores: pl.DataFrame, grid: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Evaluate all candidate predictions row-wise."""
        available_scores = set(scores["EID_score_name"].unique().to_list()) if not scores.is_empty() else set()
        default_score_name = (
            "essential_input_score_diagnostic"
            if "essential_input_score_diagnostic" in available_scores
            else (sorted(available_scores)[0] if available_scores else None)
        )
        default_scores = (
            scores.filter(pl.col("EID_score_name") == default_score_name)
            .select("country_sector", "EID_norm", "high_EID_decile")
            if default_score_name
            else pl.DataFrame()
        )
        pred_frames = []
        residual_frames = []
        for row in grid.to_dicts():
            variant = row["variant_name"]
            if row.get("EID_score_name") and row["EID_score_name"] not in available_scores:
                continue
            base = panel
            if row.get("EID_score_name"):
                s = scores.filter(pl.col("EID_score_name") == row["EID_score_name"]).select("country_sector", "EID_score_name", "EID_norm", "high_EID_decile")
                base = base.join(s, on="country_sector", how="left")
            elif not default_scores.is_empty():
                base = base.join(default_scores, on="country_sector", how="left").with_columns(
                    pl.lit(None).alias("EID_score_name")
                )
            else:
                base = base.with_columns(pl.lit(None).alias("EID_score_name"), pl.lit(0.0).alias("EID_norm"), pl.lit(False).alias("high_EID_decile"))
            if variant == "frontier_gap_readiness_baseline":
                pred = base.with_columns(pl.col("simulated_rEI_readiness").alias("predicted_rEI"), pl.lit(1.0).alias("D_EID"), pl.lit(1.0).alias("P_hist"))
            elif variant == "historical_frontier_gap_only_baseline":
                pred = base.with_columns(pl.col("simulated_rEI_frontier_gap").alias("predicted_rEI"), pl.lit(1.0).alias("D_EID"), pl.lit(1.0).alias("P_hist"))
            else:
                lambda_eid = _as_float(row.get("lambda_EID"), 0.0)
                d_min = _as_float(row.get("d_min"), 1.0)
                pred = base.with_columns(
                    self.compute_d_eid(pl.col("EID_norm").fill_null(0), lambda_eid, d_min).alias("D_EID")
                ).with_columns(
                    (pl.col("alpha_sector_year") + pl.col("D_EID") * (pl.col("simulated_rEI_frontier_gap") - pl.col("alpha_sector_year"))).alias("predicted_without_residual")
                )
                if row.get("uses_historical_residual"):
                    residual = self.calibrate_historical_residual(pred, row)
                    residual_frames.append(residual)
                    pred = pred.join(residual.select("country_sector", "P_hist").unique(), on="country_sector", how="left").with_columns(pl.col("P_hist").fill_null(1.0))
                else:
                    pred = pred.with_columns(pl.lit(1.0).alias("P_hist"))
                if variant == "historical_residual_only":
                    pred = pred.with_columns((pl.col("alpha_sector_year") + pl.col("P_hist") * (pl.col("simulated_rEI_frontier_gap") - pl.col("alpha_sector_year"))).alias("predicted_rEI"))
                elif variant == "electricity_dampened_frontier_gap_0_75_reference":
                    pred = pred.with_columns(
                        pl.when(pl.col("electricity_like"))
                        .then(pl.col("alpha_sector_year") + 0.75 * (pl.col("simulated_rEI_frontier_gap") - pl.col("alpha_sector_year")))
                        .otherwise(pl.col("simulated_rEI_frontier_gap"))
                        .alias("predicted_rEI")
                    )
                else:
                    pred = pred.with_columns((pl.col("alpha_sector_year") + pl.col("D_EID") * pl.col("P_hist") * (pl.col("simulated_rEI_frontier_gap") - pl.col("alpha_sector_year"))).alias("predicted_rEI"))
            pred_frames.append(self._prepare_prediction_frame(pred, row))
        residual_panel = pl.concat(residual_frames, how="vertical_relaxed") if residual_frames else pl.DataFrame()
        return pl.concat(pred_frames, how="vertical_relaxed"), residual_panel

    def calibrate_historical_residual(self, pred: pl.DataFrame, candidate: dict[str, Any]) -> pl.DataFrame:
        """Calibrate bounded historical residual multipliers on train rows."""
        level = candidate.get("residual_level") or "country_sector"
        k = _as_float(candidate.get("shrinkage_k"), 10.0)
        p_min = _as_float(candidate.get("p_min"), 0.75)
        p_max = _as_float(candidate.get("p_max"), 1.25)
        base = pred.with_columns(
            pl.when(pl.col("predicted_without_residual").is_not_null())
            .then(pl.col("predicted_without_residual"))
            .otherwise(pl.col("simulated_rEI_frontier_gap"))
            .alias("predicted_without_residual")
        )
        train = base.filter(pl.col("year") <= self.train_end_year).with_columns(
            (pl.col("observed_rEI") - pl.col("predicted_without_residual")).alias("residual")
        )
        if level == "sector":
            theta = train.group_by("Sector").agg(pl.col("residual").mean().alias("theta_shrunk"))
            out = base.join(theta, on="Sector", how="left")
        elif level == "country":
            theta = train.group_by("Country").agg(pl.col("residual").mean().alias("theta_shrunk"))
            out = base.join(theta, on="Country", how="left")
        else:
            raw = train.group_by(["country_sector", "Sector"]).agg(pl.col("residual").mean().alias("theta_raw"), pl.len().alias("n_i"))
            sector = train.group_by("Sector").agg(pl.col("residual").mean().alias("theta_sector_mean"))
            theta = raw.join(sector, on="Sector", how="left").with_columns(
                (pl.col("n_i") / (pl.col("n_i") + k)).alias("_w")
            ).with_columns(
                (pl.col("_w") * pl.col("theta_raw") + (1 - pl.col("_w")) * pl.col("theta_sector_mean")).alias("theta_shrunk")
            )
            out = base.join(theta.select("country_sector", "theta_raw", "theta_sector_mean", "theta_shrunk"), on="country_sector", how="left")
        for column in ["theta_raw", "theta_sector_mean"]:
            if column not in out.columns:
                out = out.with_columns(pl.lit(None).cast(pl.Float64).alias(column))
        return out.with_columns(
            pl.col("theta_shrunk").fill_null(0).alias("theta_shrunk"),
            (1 + pl.col("theta_shrunk").fill_null(0)).clip(p_min, p_max).alias("P_hist"),
            pl.lit(candidate["candidate_id"]).alias("candidate_id"),
            pl.lit(level).alias("residual_level"),
            pl.lit(k).alias("shrinkage_k"),
            pl.lit(p_min).alias("p_min"),
            pl.lit(p_max).alias("p_max"),
            (pl.col("observed_rEI") - pl.col("predicted_without_residual")).alias("residual"),
            pl.lit("bounded historical residual; calibration only").alias("notes"),
        ).select(
            "candidate_id", "country_sector", "Country", "Sector", "year", "train_or_validation", "observed_rEI",
            "predicted_without_residual", "residual", "theta_raw", "theta_sector_mean", "theta_shrunk", "P_hist",
            "residual_level", "shrinkage_k", "p_min", "p_max", "notes",
        )

    def _prepare_prediction_frame(self, pred: pl.DataFrame, candidate: dict[str, Any]) -> pl.DataFrame:
        out = pred.with_columns(
            pl.lit(candidate["candidate_id"]).alias("candidate_id"),
            pl.lit(candidate["variant_name"]).alias("variant_name"),
            pl.lit(candidate.get("EID_score_name")).alias("EID_score_name"),
            pl.lit(candidate.get("lambda_EID")).alias("lambda_EID"),
            pl.lit(candidate.get("d_min")).alias("d_min"),
            pl.lit(candidate.get("residual_level")).alias("residual_level"),
            pl.lit(candidate.get("shrinkage_k")).alias("shrinkage_k"),
            pl.lit(candidate.get("p_min")).alias("p_min"),
            pl.lit(candidate.get("p_max")).alias("p_max"),
            (pl.col("predicted_rEI") - pl.col("observed_rEI")).alias("rEI_error"),
            (pl.col("predicted_rEI") - pl.col("observed_rEI")).abs().alias("rEI_abs_error"),
            (pl.col("predicted_rEI").sign() != pl.col("observed_rEI").sign()).cast(pl.Float64).alias("wrong_sign"),
            (pl.col("X_observed") * pl.col("EI_observed") * (pl.col("predicted_rEI") - pl.col("observed_rEI")).abs()).alias("emissions_error_abs"),
            pl.when(pl.col("emissions_decile").is_in(["d9", "d10"])).then(True).otherwise(False).alias("high_emissions_node"),
            pl.when(pl.col("high_EID_decile").is_null()).then(False).otherwise(pl.col("high_EID_decile")).alias("high_EID_decile"),
            pl.when(pl.col("EID_norm").is_null()).then(pl.lit("missing")).otherwise((pl.col("EID_norm") * 10).floor().clip(0, 9).cast(pl.Int64).cast(pl.Utf8)).alias("EID_decile"),
        )
        keep = [
            "candidate_id", "variant_name", "EID_score_name", "lambda_EID", "d_min", "residual_level", "shrinkage_k", "p_min", "p_max",
            "country_sector", "year", "Country", "Sector", "train_or_validation", "observed_rEI", "predicted_rEI",
            "rEI_error", "rEI_abs_error", "wrong_sign", "X_observed", "EI_observed", "emissions_observed", "emissions_error_abs",
            "electricity_like", "china_electricity", "high_emissions_node", "EID_norm", "EID_decile", "high_EID_decile", "D_EID", "P_hist",
            "simulated_rEI_frontier_gap", "simulated_rEI_readiness",
        ]
        for column in keep:
            if column not in out.columns:
                out = out.with_columns(pl.lit(None).alias(column))
        return out.select(keep)

    def build_validation_results(self, predictions: pl.DataFrame, grid: pl.DataFrame) -> pl.DataFrame:
        rows = []
        for (cid, split), frame in predictions.group_by(["candidate_id", "train_or_validation"]):
            meta = grid.filter(pl.col("candidate_id") == cid).to_dicts()[0]
            rows.append(self._metric_row(frame, meta, split))
        return pl.DataFrame(rows)

    def _metric_row(self, frame: pl.DataFrame, meta: dict[str, Any], split: str) -> dict[str, Any]:
        latest = frame["year"].max()
        yearly = frame.group_by("year").agg(pl.col("emissions_error_abs").sum().alias("err"), pl.col("emissions_observed").sum().alias("obs"))
        elec = frame.filter(pl.col("electricity_like"))
        china = frame.filter(pl.col("china_electricity"))
        high = frame.filter(pl.col("high_emissions_node"))
        high_eid = frame.filter(pl.col("high_EID_decile"))
        return {
            **{k: meta.get(k) for k in ["candidate_id", "variant_name", "EID_score_name", "lambda_EID", "d_min", "residual_level", "shrinkage_k", "p_min", "p_max"]},
            "train_or_validation": split,
            "all_node_unweighted_rEI_MAE": _as_float(frame["rEI_abs_error"].mean()),
            "all_node_output_weighted_rEI_MAE": self._weighted_mean(frame, "rEI_abs_error", "X_observed"),
            "all_node_emissions_weighted_rEI_MAE": self._weighted_mean(frame, "rEI_abs_error", "emissions_observed"),
            "all_node_wrong_sign_share": _as_float(frame["wrong_sign"].mean()),
            "all_node_output_weighted_wrong_sign_share": self._weighted_mean(frame, "wrong_sign", "X_observed"),
            "all_node_emissions_weighted_wrong_sign_share": self._weighted_mean(frame, "wrong_sign", "emissions_observed"),
            "validation_bias": _as_float(frame["rEI_error"].mean()),
            "validation_correlation": _as_float(frame.select(pl.corr("predicted_rEI", "observed_rEI")).item()),
            "latest_year_aggregate_emissions_pct_error": self._pct_error(yearly.filter(pl.col("year") == latest)["err"].sum(), yearly.filter(pl.col("year") == latest)["obs"].sum()),
            "mean_yearly_aggregate_emissions_pct_error": _as_float((yearly["err"] / yearly["obs"]).mean()),
            "total_emissions_absolute_error": _as_float(frame["emissions_error_abs"].sum()),
            "mean_emissions_absolute_error": _as_float(frame["emissions_error_abs"].mean()),
            "electricity_rEI_MAE": _as_float(elec["rEI_abs_error"].mean()) if not elec.is_empty() else None,
            "electricity_emissions_weighted_rEI_MAE": self._weighted_mean(elec, "rEI_abs_error", "emissions_observed"),
            "electricity_wrong_sign_share": _as_float(elec["wrong_sign"].mean()) if not elec.is_empty() else None,
            "electricity_aggregate_emissions_error": _as_float(elec["emissions_error_abs"].sum()) if not elec.is_empty() else None,
            "china_electricity_rEI_MAE": _as_float(china["rEI_abs_error"].mean()) if not china.is_empty() else None,
            "china_electricity_emissions_error": _as_float(china["emissions_error_abs"].sum()) if not china.is_empty() else None,
            "china_electricity_wrong_sign_share": _as_float(china["wrong_sign"].mean()) if not china.is_empty() else None,
            "high_emissions_node_rEI_MAE": _as_float(high["rEI_abs_error"].mean()) if not high.is_empty() else None,
            "high_emissions_node_emissions_error": _as_float(high["emissions_error_abs"].sum()) if not high.is_empty() else None,
            "high_EID_node_rEI_MAE": _as_float(high_eid["rEI_abs_error"].mean()) if not high_eid.is_empty() else None,
            "high_EID_node_emissions_error": _as_float(high_eid["emissions_error_abs"].sum()) if not high_eid.is_empty() else None,
            "notes": "",
        }

    def summarize_group(self, predictions: pl.DataFrame, group_col: str, candidate_ids: list[str], label: str) -> pl.DataFrame:
        if group_col not in predictions.columns:
            return pl.DataFrame()
        return predictions.filter(pl.col("candidate_id").is_in(candidate_ids) & (pl.col("train_or_validation") == "validation")).group_by("candidate_id", group_col).agg(
            pl.len().alias("rows"),
            (pl.col("emissions_observed").sum() / predictions["emissions_observed"].sum()).alias("observed_emissions_share"),
            pl.col("rEI_abs_error").mean().alias("rEI_MAE"),
            (pl.col("rEI_abs_error") * pl.col("emissions_observed")).sum().truediv(pl.col("emissions_observed").sum()).alias("emissions_weighted_rEI_MAE"),
            pl.col("wrong_sign").mean().alias("wrong_sign_share"),
            pl.col("emissions_error_abs").sum().alias("emissions_error"),
            pl.lit(label).alias("interpretation"),
        ).rename({group_col: "group"})

    def summarize_china_electricity(self, predictions: pl.DataFrame, candidate_ids: list[str]) -> pl.DataFrame:
        return predictions.filter(pl.col("candidate_id").is_in(candidate_ids) & pl.col("china_electricity")).select(
            "candidate_id", "year", "Country", "Sector", "observed_rEI", "predicted_rEI", "rEI_error", "emissions_error_abs", "D_EID", "P_hist"
        )

    def summarize_residuals(self, residual_panel: pl.DataFrame, predictions: pl.DataFrame) -> pl.DataFrame:
        if residual_panel.is_empty():
            return pl.DataFrame()
        return residual_panel.group_by("residual_level", "shrinkage_k", "p_min", "p_max").agg(
            pl.col("P_hist").mean().alias("mean_P_hist"),
            pl.col("P_hist").median().alias("median_P_hist"),
            pl.col("P_hist").quantile(0.05).alias("p05_P_hist"),
            pl.col("P_hist").quantile(0.95).alias("p95_P_hist"),
            (pl.col("P_hist") <= pl.col("p_min")).mean().alias("share_at_lower_bound"),
            (pl.col("P_hist") >= pl.col("p_max")).mean().alias("share_at_upper_bound"),
            pl.lit(None).alias("electricity_mean_P_hist"),
            pl.lit(None).alias("china_electricity_P_hist"),
            pl.lit(None).alias("high_EID_mean_P_hist"),
            pl.lit("bounded calibration-only residual").alias("interpretation"),
        )

    def build_mechanism_decomposition(self, validation: pl.DataFrame, predictions: pl.DataFrame) -> pl.DataFrame:
        val = validation.filter(pl.col("train_or_validation") == "validation")
        baseline = val.filter(pl.col("variant_name") == "historical_frontier_gap_only_baseline")
        if baseline.is_empty():
            return pl.DataFrame()
        base = _as_float(baseline["all_node_emissions_weighted_rEI_MAE"].min())
        best_eid = self._best_row(val.filter(pl.col("variant_name") == "essential_input_dampener_only"))
        best_res = self._best_row(val.filter(pl.col("variant_name") == "historical_residual_only"))
        best_combo = self._best_row(val.filter(pl.col("variant_name") == "essential_input_dampener_plus_historical_residual"))
        eid_gain = base - _as_float(best_eid.get("all_node_emissions_weighted_rEI_MAE"), base)
        res_gain = base - _as_float(best_res.get("all_node_emissions_weighted_rEI_MAE"), base)
        combo_gain = base - _as_float(best_combo.get("all_node_emissions_weighted_rEI_MAE"), base)
        denom = max(abs(combo_gain), 1e-12)
        residual_dominates = res_gain > max(eid_gain, 0) and res_gain >= 0.6 * max(combo_gain, 0)
        return pl.DataFrame([{
            "candidate_id": best_combo.get("candidate_id"),
            "EID_score_name": best_combo.get("EID_score_name"),
            "mean_D_EID": self._candidate_mean(predictions, best_combo.get("candidate_id"), "D_EID"),
            "median_D_EID": self._candidate_median(predictions, best_combo.get("candidate_id"), "D_EID"),
            "electricity_mean_D_EID": self._candidate_mean(predictions.filter(pl.col("electricity_like")), best_combo.get("candidate_id"), "D_EID"),
            "china_electricity_D_EID": self._candidate_mean(predictions.filter(pl.col("china_electricity")), best_combo.get("candidate_id"), "D_EID"),
            "high_EID_mean_D_EID": self._candidate_mean(predictions.filter(pl.col("high_EID_decile")), best_combo.get("candidate_id"), "D_EID"),
            "mean_P_hist": self._candidate_mean(predictions, best_combo.get("candidate_id"), "P_hist"),
            "electricity_mean_P_hist": self._candidate_mean(predictions.filter(pl.col("electricity_like")), best_combo.get("candidate_id"), "P_hist"),
            "china_electricity_P_hist": self._candidate_mean(predictions.filter(pl.col("china_electricity")), best_combo.get("candidate_id"), "P_hist"),
            "validation_gain_structural_only": eid_gain,
            "validation_gain_residual_only": res_gain,
            "validation_gain_combined": combo_gain,
            "structural_share_of_combined_gain": eid_gain / denom,
            "residual_share_of_combined_gain": res_gain / denom,
            "residual_dominates_flag": residual_dominates,
            "interpretation": "residual dominates mechanism" if residual_dominates else "structural contribution remains visible",
        }])

    def build_abm_v5_implications(self, validation: pl.DataFrame, mechanism: pl.DataFrame) -> pl.DataFrame:
        val = validation.filter(pl.col("train_or_validation") == "validation")
        base = self._best_row(val.filter(pl.col("variant_name") == "historical_frontier_gap_only_baseline"))
        eid = self._best_row(val.filter(pl.col("variant_name") == "essential_input_dampener_only"))
        improves_high = _as_float(eid.get("high_EID_node_rEI_MAE"), 999) < _as_float(base.get("high_EID_node_rEI_MAE"), 0)
        return pl.DataFrame([{
            "finding": "EID-only high-EID validation improvement" if improves_high else "EID-only high-EID validation not yet improved",
            "evidence": f"best_EID_candidate={eid.get('candidate_id')}; high_EID_MAE={eid.get('high_EID_node_rEI_MAE')}",
            "implication_for_abm_v5": "candidate structural role for future agent ontology" if improves_high else "do not promote to agent type yet",
            "agent_type_candidate": "essential_input_agent" if improves_high else "not_supported",
            "confidence_level": "moderate" if improves_high else "low",
            "notes": "ABM v4 parameter test only; no ABM v5 implementation.",
        }])

    def build_recommendation(self, validation: pl.DataFrame, mechanism: pl.DataFrame) -> pl.DataFrame:
        val = validation.filter(pl.col("train_or_validation") == "validation")
        base = self._best_row(val.filter(pl.col("variant_name") == "historical_frontier_gap_only_baseline"))
        eid = self._best_row(val.filter(pl.col("variant_name") == "essential_input_dampener_only"))
        material = self.material_worsening_flags(base, eid)
        residual_dominates = bool(mechanism["residual_dominates_flag"].item(0)) if not mechanism.is_empty() else False
        if residual_dominates:
            rec = "residual_dominates_mechanism"
            interp = "Historical residual explains most combined validation gain; do not treat EID as validated mechanism yet."
        elif self._improves(eid, base, "electricity_rEI_MAE") and self._improves(eid, base, "high_EID_node_rEI_MAE") and not any(material.values()):
            rec = "proceed_to_multiyear_candidate_integration"
            interp = "EID-only improves electricity/high-EID metrics without material all-node worsening."
        elif self._improves(eid, base, "electricity_rEI_MAE"):
            rec = "keep_EID_dampener_as_diagnostic_only"
            interp = "EID-only helps electricity but does not yet clear full validation thresholds."
        else:
            rec = "structural_signal_too_weak"
            interp = "Essential-input dependence is structurally meaningful but not validated as dampener."
        return pl.DataFrame([{
            "recommendation": rec,
            "evidence": f"best_EID={eid.get('candidate_id')}; material_worsening={material}; residual_dominates={residual_dominates}",
            "interpretation": interp,
            "recommended_phase24": "Integrate only a diagnostic candidate in the multi-year loop if thresholds pass; otherwise inspect external policy/energy variables.",
            "abm_v5_implication": "possible essential_input_agent candidate if high-EID improvement holds out of sample",
            "scenario_readiness": "premature",
        }])

    def material_worsening_flags(self, baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, bool]:
        return {
            "all_node_rEI_MAE": self._relative_worse(candidate, baseline, "all_node_unweighted_rEI_MAE", 0.02),
            "emissions_weighted_rEI_MAE": self._relative_worse(candidate, baseline, "all_node_emissions_weighted_rEI_MAE", 0.02),
            "wrong_sign_share": (_as_float(candidate.get("all_node_wrong_sign_share")) - _as_float(baseline.get("all_node_wrong_sign_share"))) > 0.02,
            "aggregate_emissions_pct_error": (_as_float(candidate.get("mean_yearly_aggregate_emissions_pct_error")) - _as_float(baseline.get("mean_yearly_aggregate_emissions_pct_error"))) > 0.02,
        }

    def build_markdown_report(self, validation: pl.DataFrame, mechanism: pl.DataFrame, recommendation: pl.DataFrame, v5: pl.DataFrame) -> str:
        return "\n".join([
            "# ABM v4 Phase 23 Essential-Input Dampener Test",
            "",
            f"Recommendation: `{recommendation['recommendation'].item(0) if not recommendation.is_empty() else 'inconclusive'}`.",
            "",
            "## Recommendation",
            self._markdown_table(recommendation),
            "",
            "## Best Validation Rows",
            self._markdown_table(validation.filter(pl.col("train_or_validation") == "validation").sort("all_node_emissions_weighted_rEI_MAE").head(20)),
            "",
            "## Mechanism Decomposition",
            self._markdown_table(mechanism),
            "",
            "## ABM v5 Implications",
            self._markdown_table(v5),
            "",
            "Scenarios remain premature.",
        ]) + "\n"

    def _best_candidate_ids(self, validation: pl.DataFrame) -> list[str]:
        val = validation.filter(pl.col("train_or_validation") == "validation")
        ids = val.sort("all_node_emissions_weighted_rEI_MAE").head(5)["candidate_id"].to_list()
        for name in ["historical_frontier_gap_only_baseline", "frontier_gap_readiness_baseline"]:
            rows = val.filter(pl.col("variant_name") == name)
            if not rows.is_empty():
                ids.append(rows["candidate_id"].item(0))
        return sorted(set(ids))

    def _best_row(self, frame: pl.DataFrame) -> dict[str, Any]:
        if frame.is_empty():
            return {}
        return frame.sort("all_node_emissions_weighted_rEI_MAE").to_dicts()[0]

    def _weighted_mean(self, frame: pl.DataFrame, value: str, weight: str) -> float | None:
        if frame.is_empty() or value not in frame.columns or weight not in frame.columns:
            return None
        denom = _as_float(frame[weight].sum())
        return None if denom == 0 else _as_float((frame[value] * frame[weight]).sum() / denom)

    def _pct_error(self, error: Any, observed: Any) -> float:
        obs = _as_float(observed)
        return 0.0 if obs == 0 else _as_float(error) / obs

    def _candidate_mean(self, frame: pl.DataFrame, candidate_id: Any, column: str) -> float | None:
        if candidate_id is None or frame.is_empty() or column not in frame.columns:
            return None
        rows = frame.filter(pl.col("candidate_id") == candidate_id)
        return None if rows.is_empty() else _as_float(rows[column].mean())

    def _candidate_median(self, frame: pl.DataFrame, candidate_id: Any, column: str) -> float | None:
        if candidate_id is None or frame.is_empty() or column not in frame.columns:
            return None
        rows = frame.filter(pl.col("candidate_id") == candidate_id)
        return None if rows.is_empty() else _as_float(rows[column].median())

    def _improves(self, candidate: dict[str, Any], baseline: dict[str, Any], metric: str) -> bool:
        return _as_float(candidate.get(metric), 999.0) < _as_float(baseline.get(metric), 0.0)

    def _relative_worse(self, candidate: dict[str, Any], baseline: dict[str, Any], metric: str, threshold: float) -> bool:
        base = _as_float(baseline.get(metric))
        cand = _as_float(candidate.get(metric))
        return base != 0 and (cand - base) / abs(base) > threshold

    def _markdown_table(self, frame: pl.DataFrame) -> str:
        if frame.is_empty():
            return "_No rows available._"
        columns = frame.columns
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
        for row in frame.to_dicts():
            lines.append("| " + " | ".join(self._format_markdown_value(row.get(column)) for column in columns) + " |")
        return "\n".join(lines)

    def _format_markdown_value(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        if value is None:
            return ""
        return str(value).replace("|", "/")


@dataclass(frozen=True)
class EssentialInputFailureModeResult:
    """In-memory outputs for the Phase 24 EID failure-mode audit."""

    heterogeneity_panel: pl.DataFrame
    subtype_composition: pl.DataFrame
    performance_by_subtype: pl.DataFrame
    failure_modes: pl.DataFrame
    pseudo_agent_audit: pl.DataFrame
    abm_v5_agent_type_candidates: pl.DataFrame
    recommendation: pl.DataFrame
    markdown: str


class EssentialInputFailureModeDiagnostics:
    """Audit heterogeneity and failure modes in high-EID dampener diagnostics."""

    high_eid_scores = [
        "structural_dependence_score_diagnostic",
        "essential_input_score_diagnostic",
        "low_substitutability_score_diagnostic",
        "structural_dependence_plus_brown_lockin",
    ]

    def __init__(self, paths: ABMV4Paths) -> None:
        self.paths = paths

    def run(self) -> EssentialInputFailureModeResult:
        """Build Phase 24 diagnostic outputs in memory."""
        heterogeneity = self.build_high_EID_node_panel()
        composition = self.build_subtype_composition(heterogeneity)
        performance = self.evaluate_dampener_performance_by_subtype(heterogeneity)
        failure_modes = self.identify_failure_modes(heterogeneity, performance)
        pseudo = self.identify_accounting_or_pseudo_agent_nodes(heterogeneity, failure_modes)
        v5 = self.identify_abm_v5_agent_type_candidates(composition, performance)
        recommendation = self.build_recommendation(composition, performance, failure_modes, pseudo, v5)
        markdown = self.build_markdown_report(composition, performance, failure_modes, pseudo, v5, recommendation)
        return EssentialInputFailureModeResult(
            heterogeneity_panel=heterogeneity,
            subtype_composition=composition,
            performance_by_subtype=performance,
            failure_modes=failure_modes,
            pseudo_agent_audit=pseudo,
            abm_v5_agent_type_candidates=v5,
            recommendation=recommendation,
            markdown=markdown,
        )

    def write_outputs(self, result: EssentialInputFailureModeResult) -> None:
        """Write Phase 24 outputs. The caller controls whether this is invoked."""
        self.paths.validation.mkdir(parents=True, exist_ok=True)
        result.heterogeneity_panel.write_csv(self.paths.eid_high_node_heterogeneity_panel_path)
        result.subtype_composition.write_csv(self.paths.eid_subtype_composition_path)
        result.performance_by_subtype.write_csv(self.paths.eid_dampener_performance_by_subtype_path)
        result.failure_modes.write_csv(self.paths.eid_dampener_failure_modes_path)
        result.pseudo_agent_audit.write_csv(self.paths.eid_pseudo_agent_audit_path)
        result.abm_v5_agent_type_candidates.write_csv(self.paths.eid_abm_v5_agent_type_candidates_path)
        result.recommendation.write_csv(self.paths.eid_failure_mode_recommendation_path)
        self.paths.eid_failure_mode_report_path.write_text(result.markdown, encoding="utf-8")

    def load_phase22_metrics(self) -> pl.DataFrame:
        """Load Phase 22 node metrics and structural node labels."""
        if not self.paths.essential_input_node_metrics_path.exists():
            raise FileNotFoundError(
                f"Missing Phase 22 essential-input node metrics: {self.paths.essential_input_node_metrics_path}. "
                "Run --diagnose-essential-input-dependence first."
            )
        metrics = pl.read_csv(self.paths.essential_input_node_metrics_path)
        if "country_sector" not in metrics.columns:
            raise ValueError("Phase 22 essential-input node metrics must include country_sector.")
        if self.paths.structural_signature_node_panel_path.exists():
            structural = pl.read_parquet(self.paths.structural_signature_node_panel_path)
            keep = [
                column
                for column in [
                    "country_sector",
                    "ecosystem",
                    "cumulative_emissions_share",
                    "cumulative_output_share",
                    "high_emissions_node",
                    "jump_prone_node",
                    "aggregate_sensitive_node",
                    "needs_dampening_node",
                    "mean_brown_centrality",
                ]
                if column in structural.columns
            ]
            metrics = metrics.join(structural.select(keep), on="country_sector", how="left", suffix="_struct")
        return metrics

    def load_phase23_results(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Load Phase 23 normalized scores and validation results."""
        missing = [
            path
            for path in [
                self.paths.essential_input_dampener_scores_path,
                self.paths.essential_input_dampener_validation_results_path,
            ]
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing Phase 23 dampener outputs: "
                + ", ".join(str(path) for path in missing)
                + ". Run --test-essential-input-dampener first."
            )
        return (
            pl.read_csv(self.paths.essential_input_dampener_scores_path),
            pl.read_csv(self.paths.essential_input_dampener_validation_results_path),
        )

    def build_high_EID_node_panel(self) -> pl.DataFrame:
        """Build one row per high-EID country-sector and high-EID definition."""
        metrics = self.load_phase22_metrics()
        scores, _ = self.load_phase23_results()
        if scores.is_empty():
            raise ValueError("Phase 23 EID score table is empty; cannot build high-EID heterogeneity panel.")
        definitions = [score for score in self.high_eid_scores if score in set(scores["EID_score_name"].unique().to_list())]
        if not definitions:
            raise ValueError("No requested high-EID score definitions are available in Phase 23 score table.")
        frames = []
        for score in definitions:
            high = scores.filter((pl.col("EID_score_name") == score) & pl.col("high_EID_decile")).select(
                "country_sector",
                "Country",
                "Sector",
                "electricity_like",
                pl.lit(score).alias("high_EID_definition"),
                pl.lit(True).alias("high_EID_flag"),
                "EID_score_name",
                "EID_norm",
            )
            frames.append(high)
        panel = pl.concat(frames, how="vertical_relaxed").join(
            metrics,
            on=["country_sector", "Country", "Sector"],
            how="left",
            suffix="_metric",
        )
        for column in [
            "essential_input_score_diagnostic",
            "low_substitutability_score_diagnostic",
            "systemic_dependence_score_diagnostic",
            "structural_dependence_score_diagnostic",
            "mean_brown_centrality",
            "cumulative_emissions_share",
            "cumulative_output_share",
        ]:
            if column not in panel.columns:
                panel = panel.with_columns(pl.lit(None).cast(pl.Float64).alias(column))
        for column in ["high_emissions_node", "jump_prone_node", "aggregate_sensitive_node", "needs_dampening_node"]:
            if column not in panel.columns:
                panel = panel.with_columns(pl.lit(False).alias(column))
        if "ecosystem" not in panel.columns:
            panel = panel.with_columns(pl.lit(None).alias("ecosystem"))
        return panel.with_columns(
            pl.struct("Country", "Sector", "country_sector").map_elements(
                lambda row: self.classify_high_EID_subtype(row["Country"], row["Sector"], row["country_sector"]),
                return_dtype=pl.Utf8,
            ).alias("candidate_subtype"),
            pl.struct("Country", "Sector", "country_sector").map_elements(
                lambda row: self._pseudo_agent_reason(row["Country"], row["Sector"], row["country_sector"]) is not None,
                return_dtype=pl.Boolean,
            ).alias("pseudo_agent_flag"),
            pl.lit("top-decile Phase 23 normalized EID score").alias("notes"),
        ).rename(
            {
                "essential_input_score_diagnostic": "essential_input_score",
                "low_substitutability_score_diagnostic": "low_substitutability_score",
                "systemic_dependence_score_diagnostic": "systemic_dependence_score",
                "structural_dependence_score_diagnostic": "structural_dependence_score",
                "mean_brown_centrality": "brown_centrality",
                "cumulative_emissions_share": "emissions_share",
                "cumulative_output_share": "output_share",
            }
        ).select(
            "country_sector",
            "Country",
            "Sector",
            "ecosystem",
            "high_EID_definition",
            "high_EID_flag",
            "EID_score_name",
            "EID_norm",
            "essential_input_score",
            "low_substitutability_score",
            "systemic_dependence_score",
            "structural_dependence_score",
            "brown_centrality",
            "emissions_share",
            "output_share",
            "electricity_like",
            "high_emissions_node",
            "jump_prone_node",
            "aggregate_sensitive_node",
            "needs_dampening_node",
            "candidate_subtype",
            "pseudo_agent_flag",
            "notes",
        )

    def classify_high_EID_subtype(self, country: Any, sector: Any, country_sector: Any = "") -> str:
        """Classify a high-EID node into a diagnostic economic subtype."""
        text = f"{country} {sector} {country_sector}".lower()
        if self._pseudo_agent_reason(country, sector, country_sector):
            return "accounting_or_pseudo_agent"
        if any(token in text for token in ["electricity", "gas and water", "utilities", "power", "water supply"]):
            return "infrastructure_energy"
        if any(token in text for token in ["transport", "logistics", "post", "telecommunication", "communication", "wholesale trade", "retail trade"]):
            return "transport_logistics_infrastructure"
        if any(token in text for token in ["petroleum", "chemical", "mining", "quarrying", "metal", "mineral", "coal", "oil", "gas extraction"]):
            return "heavy_industry_materials"
        if any(token in text for token in ["machinery", "equipment", "manufacturing", "electrical"]):
            return "manufacturing_system_core"
        if any(token in text for token in ["construction", "real estate"]):
            return "construction_real_estate_foundational"
        if any(token in text for token in ["finacial", "financial", "finance", "business activities", "business services", "knowledge"]):
            return "knowledge_finance_business_services"
        if any(token in text for token in ["public administration", "education", "health", "social", "households", "other services"]):
            return "public_social_services"
        return "ordinary_or_unclear"

    def build_subtype_composition(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Summarize high-EID node composition by subtype."""
        if panel.is_empty():
            return pl.DataFrame()
        unique_nodes = panel.unique(["country_sector", "candidate_subtype"])
        total_nodes = max(unique_nodes.height, 1)
        total_emissions = _as_float(unique_nodes["emissions_share"].fill_null(0).sum())
        total_output = _as_float(unique_nodes["output_share"].fill_null(0).sum())
        return unique_nodes.group_by("candidate_subtype").agg(
            pl.len().alias("nodes"),
            (pl.len() / total_nodes).alias("share_of_high_EID_nodes"),
            (pl.col("emissions_share").fill_null(0).sum() / max(total_emissions, 1e-12)).alias("observed_emissions_share"),
            (pl.col("output_share").fill_null(0).sum() / max(total_output, 1e-12)).alias("observed_output_share"),
            pl.col("electricity_like").mean().alias("electricity_like_share"),
            pl.col("high_emissions_node").mean().alias("high_emissions_share"),
            pl.col("aggregate_sensitive_node").mean().alias("aggregate_sensitive_share"),
            pl.col("needs_dampening_node").mean().alias("needs_dampening_share"),
            pl.col("pseudo_agent_flag").mean().alias("pseudo_agent_share"),
            pl.col("EID_norm").mean().alias("mean_EID_norm"),
            pl.col("structural_dependence_score").mean().alias("mean_structural_dependence_score"),
            pl.col("low_substitutability_score").mean().alias("mean_low_substitutability_score"),
            pl.col("systemic_dependence_score").mean().alias("mean_systemic_dependence_score"),
        ).with_columns(
            pl.when(pl.col("pseudo_agent_share") > 0.25)
            .then(pl.lit("high-EID subtype includes accounting or aggregate categories"))
            .when(pl.col("electricity_like_share") > 0.25)
            .then(pl.lit("high-EID subtype contains electricity-like infrastructure"))
            .otherwise(pl.lit("mixed high-EID subtype"))
            .alias("interpretation")
        ).sort("nodes", descending=True)

    def evaluate_dampener_performance_by_subtype(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Compare baseline and best EID-only candidate errors by subtype."""
        if not self.paths.transition_rule_sign_failure_panel_path.exists():
            raise FileNotFoundError(
                f"Missing transition panel: {self.paths.transition_rule_sign_failure_panel_path}. Run Phase 16 first."
            )
        _, validation = self.load_phase23_results()
        best_eid = self._best_candidate(validation, "essential_input_dampener_only")
        if not best_eid:
            raise ValueError("Phase 23 validation results do not contain an EID-only candidate.")
        transition = pl.read_parquet(self.paths.transition_rule_sign_failure_panel_path)
        required = {"country_sector", "year", "Country", "Sector", "observed_rEI", "simulated_rEI_frontier_gap", "simulated_rEI_readiness", "X_observed", "EI_observed", "emissions_observed"}
        missing = required - set(transition.columns)
        if missing:
            raise ValueError(f"Transition panel is missing required columns: {sorted(missing)}")
        alpha = transition.group_by(["Sector", "year"]).agg(pl.col("observed_rEI").mean().alias("alpha_sector_year"))
        scores = pl.read_csv(self.paths.essential_input_dampener_scores_path)
        score_name = best_eid.get("EID_score_name")
        score = scores.filter(pl.col("EID_score_name") == score_name).select("country_sector", "EID_norm")
        subtype = panel.unique("country_sector").select("country_sector", "candidate_subtype", "emissions_share", "output_share")
        base = transition.join(alpha, on=["Sector", "year"], how="left").join(score, on="country_sector", how="left").join(subtype, on="country_sector", how="inner")
        d_eid = (1 - _as_float(best_eid.get("lambda_EID")) * pl.col("EID_norm").fill_null(0)).clip(_as_float(best_eid.get("d_min"), 1), 1)
        rows = []
        variants = [
            ("frontier_gap_readiness_baseline", "c0001", pl.col("simulated_rEI_readiness")),
            ("historical_frontier_gap_only_baseline", "c0002", pl.col("simulated_rEI_frontier_gap")),
            ("essential_input_dampener_only", str(best_eid.get("candidate_id")), pl.col("alpha_sector_year") + d_eid * (pl.col("simulated_rEI_frontier_gap") - pl.col("alpha_sector_year"))),
        ]
        for variant_name, candidate_id, expr in variants:
            frame = base.with_columns(expr.alias("predicted_rEI")).with_columns(
                (pl.col("predicted_rEI") - pl.col("observed_rEI")).abs().alias("rEI_abs_error"),
                (pl.col("predicted_rEI").sign() != pl.col("observed_rEI").sign()).cast(pl.Float64).alias("wrong_sign"),
                (pl.col("X_observed") * pl.col("EI_observed") * (pl.col("predicted_rEI") - pl.col("observed_rEI")).abs()).alias("emissions_error_abs"),
            )
            summary = frame.group_by("candidate_subtype").agg(
                pl.len().alias("rows"),
                pl.col("emissions_observed").sum().alias("_obs_emissions"),
                pl.col("X_observed").sum().alias("_obs_output"),
                pl.col("rEI_abs_error").mean().alias("rEI_MAE"),
                (pl.col("rEI_abs_error") * pl.col("emissions_observed")).sum().truediv(pl.col("emissions_observed").sum()).alias("emissions_weighted_rEI_MAE"),
                pl.col("wrong_sign").mean().alias("wrong_sign_share"),
                pl.col("emissions_error_abs").sum().alias("emissions_error"),
            ).with_columns(
                pl.lit(variant_name).alias("variant_name"),
                pl.lit(candidate_id).alias("candidate_id"),
            )
            rows.append(summary)
        out = pl.concat(rows, how="vertical_relaxed")
        total_emissions = _as_float(out.filter(pl.col("variant_name") == "historical_frontier_gap_only_baseline")["_obs_emissions"].sum())
        total_output = _as_float(out.filter(pl.col("variant_name") == "historical_frontier_gap_only_baseline")["_obs_output"].sum())
        out = out.with_columns(
            (pl.col("_obs_emissions") / max(total_emissions, 1e-12)).alias("observed_emissions_share"),
            (pl.col("_obs_output") / max(total_output, 1e-12)).alias("observed_output_share"),
        )
        baseline = out.filter(pl.col("variant_name") == "historical_frontier_gap_only_baseline").select(
            "candidate_subtype",
            pl.col("emissions_weighted_rEI_MAE").alias("_hist_mae"),
            pl.col("emissions_error").alias("_hist_error"),
        )
        readiness = out.filter(pl.col("variant_name") == "frontier_gap_readiness_baseline").select(
            "candidate_subtype",
            pl.col("emissions_weighted_rEI_MAE").alias("_ready_mae"),
        )
        return out.join(baseline, on="candidate_subtype", how="left").join(readiness, on="candidate_subtype", how="left").with_columns(
            (pl.col("_hist_mae") - pl.col("emissions_weighted_rEI_MAE")).alias("improvement_vs_historical_frontier_gap"),
            (pl.col("_ready_mae") - pl.col("emissions_weighted_rEI_MAE")).alias("improvement_vs_frontier_gap_readiness"),
            (pl.col("emissions_error") - pl.col("_hist_error")).alias("aggregate_error_contribution"),
            ((pl.col("emissions_weighted_rEI_MAE") - pl.col("_hist_mae")) / pl.col("_hist_mae").abs() > 0.02).fill_null(False).alias("material_worsening_flag"),
            pl.when(pl.col("variant_name") == "essential_input_dampener_only")
            .then(pl.lit("best Phase 23 EID-only candidate evaluated on high-EID subtype"))
            .otherwise(pl.lit("baseline comparison"))
            .alias("interpretation"),
        ).select(
            "candidate_subtype",
            "variant_name",
            "candidate_id",
            "rows",
            "observed_emissions_share",
            "observed_output_share",
            "rEI_MAE",
            "emissions_weighted_rEI_MAE",
            "wrong_sign_share",
            "emissions_error",
            "aggregate_error_contribution",
            "improvement_vs_historical_frontier_gap",
            "improvement_vs_frontier_gap_readiness",
            "material_worsening_flag",
            "interpretation",
        ).sort(["candidate_subtype", "variant_name"])

    def identify_failure_modes(self, panel: pl.DataFrame, performance: pl.DataFrame) -> pl.DataFrame:
        """Classify node-level EID dampener failure modes."""
        if performance.is_empty():
            return pl.DataFrame()
        if not self.paths.transition_rule_sign_failure_panel_path.exists():
            raise FileNotFoundError(
                f"Missing transition panel: {self.paths.transition_rule_sign_failure_panel_path}. Run Phase 16 first."
            )
        _, validation = self.load_phase23_results()
        best_eid = self._best_candidate(validation, "essential_input_dampener_only")
        transition = pl.read_parquet(self.paths.transition_rule_sign_failure_panel_path)
        alpha = transition.group_by(["Sector", "year"]).agg(pl.col("observed_rEI").mean().alias("alpha_sector_year"))
        scores = pl.read_csv(self.paths.essential_input_dampener_scores_path)
        score = scores.filter(pl.col("EID_score_name") == best_eid.get("EID_score_name")).select("country_sector", "EID_norm")
        nodes = panel.unique("country_sector").select(
            "country_sector",
            "Country",
            "Sector",
            "candidate_subtype",
            "EID_norm",
            "emissions_share",
            "output_share",
            "pseudo_agent_flag",
        )
        d_eid = (1 - _as_float(best_eid.get("lambda_EID")) * pl.col("EID_norm").fill_null(0)).clip(_as_float(best_eid.get("d_min"), 1), 1)
        node_errors = transition.join(alpha, on=["Sector", "year"], how="left").join(score, on="country_sector", how="left").join(nodes.drop("EID_norm"), on="country_sector", how="inner").with_columns(
            d_eid.alias("D_EID")
        ).with_columns(
            pl.col("simulated_rEI_frontier_gap").alias("_baseline_pred"),
            (pl.col("alpha_sector_year") + pl.col("D_EID") * (pl.col("simulated_rEI_frontier_gap") - pl.col("alpha_sector_year"))).alias("_eid_pred"),
        ).with_columns(
            (pl.col("_baseline_pred") - pl.col("observed_rEI")).abs().alias("_baseline_abs_error"),
            (pl.col("_eid_pred") - pl.col("observed_rEI")).abs().alias("_eid_abs_error"),
        ).group_by("country_sector", "Country", "Sector", "candidate_subtype", "EID_norm", "emissions_share", "output_share", "pseudo_agent_flag").agg(
            pl.col("_baseline_abs_error").mean().alias("baseline_error"),
            pl.col("_eid_abs_error").mean().alias("EID_candidate_error"),
        ).with_columns(
            (pl.col("EID_candidate_error") - pl.col("baseline_error")).alias("error_change")
        )
        return node_errors.with_columns(
            pl.when(pl.col("pseudo_agent_flag"))
            .then(pl.lit("pseudo_agent_accounting_issue"))
            .when(pl.col("error_change") < -0.01)
            .then(pl.lit("helped_by_EID"))
            .when(pl.col("error_change") > 0.01)
            .then(pl.lit("harmed_by_EID"))
            .when((pl.col("emissions_share").fill_null(0) < 1e-5) & (pl.col("EID_norm").fill_null(0) > 0.9))
            .then(pl.lit("high_EID_but_low_emissions_relevance"))
            .when(pl.col("candidate_subtype").is_in(["knowledge_finance_business_services", "public_social_services"]))
            .then(pl.lit("high_EID_but_not_physical_transition_sector"))
            .otherwise(pl.lit("no_material_change"))
            .alias("failure_mode")
        ).with_columns(
            (pl.lit("baseline_error=") + pl.col("baseline_error").round(6).cast(pl.Utf8) + pl.lit("; EID_error=") + pl.col("EID_candidate_error").round(6).cast(pl.Utf8)).alias("evidence"),
            pl.when(pl.col("failure_mode") == "helped_by_EID")
            .then(pl.lit("EID dampening reduces transition error for this high-EID node."))
            .when(pl.col("failure_mode") == "harmed_by_EID")
            .then(pl.lit("EID dampening worsens transition error for this high-EID node."))
            .when(pl.col("failure_mode") == "pseudo_agent_accounting_issue")
            .then(pl.lit("High-EID node appears to be an accounting or aggregate category."))
            .otherwise(pl.lit("No clear EID-specific failure mode."))
            .alias("interpretation")
        ).select(
            "country_sector",
            "Country",
            "Sector",
            "candidate_subtype",
            "EID_norm",
            "emissions_share",
            "output_share",
            "baseline_error",
            "EID_candidate_error",
            "error_change",
            "failure_mode",
            "evidence",
            "interpretation",
        ).sort("error_change")

    def identify_accounting_or_pseudo_agent_nodes(self, panel: pl.DataFrame, failure_modes: pl.DataFrame) -> pl.DataFrame:
        """Flag high-EID accounting or pseudo-agent nodes."""
        nodes = panel.unique("country_sector").select(
            "country_sector",
            "Country",
            "Sector",
            "EID_norm",
            "emissions_share",
            "output_share",
        )
        if not failure_modes.is_empty():
            nodes = nodes.join(failure_modes.select("country_sector", "failure_mode"), on="country_sector", how="left")
        return nodes.with_columns(
            pl.struct("Country", "Sector", "country_sector").map_elements(
                lambda row: self._pseudo_agent_reason(row["Country"], row["Sector"], row["country_sector"]),
                return_dtype=pl.Utf8,
            ).alias("pseudo_agent_reason")
        ).with_columns(
            pl.col("pseudo_agent_reason").is_not_null().alias("pseudo_agent_flag"),
            pl.when(pl.col("pseudo_agent_reason").is_not_null())
            .then(pl.lit("exclude_from_agent_type_training"))
            .otherwise(pl.lit("keep_as_regular_node"))
            .alias("recommended_treatment"),
            pl.when(pl.col("failure_mode").is_null()).then(pl.lit("not evaluated")).otherwise(pl.col("failure_mode")).alias("dampener_performance"),
        ).select(
            "country_sector",
            "Country",
            "Sector",
            "pseudo_agent_flag",
            "pseudo_agent_reason",
            "EID_norm",
            "emissions_share",
            "output_share",
            "dampener_performance",
            "recommended_treatment",
        ).sort("pseudo_agent_flag", descending=True)

    def identify_abm_v5_agent_type_candidates(self, composition: pl.DataFrame, performance: pl.DataFrame) -> pl.DataFrame:
        """Summarize plausible ABM v5 agent-type candidates without implementing them."""
        rows = []
        for subtype in [
            "infrastructure_energy",
            "transport_logistics_infrastructure",
            "heavy_industry_materials",
            "knowledge_finance_business_services",
            "accounting_or_pseudo_agent",
        ]:
            comp = composition.filter(pl.col("candidate_subtype") == subtype)
            perf = performance.filter((pl.col("candidate_subtype") == subtype) & (pl.col("variant_name") == "essential_input_dampener_only"))
            nodes = _as_int(comp["nodes"].item(0)) if not comp.is_empty() else 0
            improvement = _as_float(perf["improvement_vs_historical_frontier_gap"].item(0)) if not perf.is_empty() else 0.0
            pseudo_share = _as_float(comp["pseudo_agent_share"].item(0)) if not comp.is_empty() else 0.0
            if subtype == "accounting_or_pseudo_agent":
                agent = "accounting_node_not_agent"
                confidence = "not_supported"
                recommended = False
                interpretation = "Accounting categories should not become behavioural agent types."
            elif nodes > 0 and improvement > 0 and pseudo_share < 0.25:
                agent = {
                    "infrastructure_energy": "energy_infrastructure_agent",
                    "transport_logistics_infrastructure": "transport_logistics_infrastructure_agent",
                    "heavy_industry_materials": "heavy_industry_materials_agent",
                    "knowledge_finance_business_services": "systemic_service_agent",
                }.get(subtype, "essential_input_infrastructure_agent")
                confidence = "moderate" if improvement > 0.01 else "low"
                recommended = subtype != "knowledge_finance_business_services"
                interpretation = "Subtype has coherent high-EID structure and positive EID diagnostic performance."
            else:
                agent = "not_supported"
                confidence = "low" if nodes > 0 else "not_supported"
                recommended = False
                interpretation = "Subtype evidence is weak, mixed, or absent."
            rows.append(
                {
                    "proposed_agent_type": agent,
                    "subtype_source": subtype,
                    "supporting_nodes": nodes,
                    "representative_nodes": "",
                    "evidence": f"EID improvement={improvement:.6g}; pseudo_share={pseudo_share:.3g}",
                    "dampener_performance": "improves" if improvement > 0 else "does_not_improve",
                    "theoretical_interpretation": interpretation,
                    "confidence_level": confidence,
                    "recommended_for_abm_v5_design": recommended,
                    "notes": "Diagnostic ontology evidence only; no ABM v5 implementation.",
                }
            )
        return pl.DataFrame(rows)

    def build_recommendation(
        self,
        composition: pl.DataFrame,
        performance: pl.DataFrame,
        failure_modes: pl.DataFrame,
        pseudo: pl.DataFrame,
        v5: pl.DataFrame,
    ) -> pl.DataFrame:
        """Apply Phase 24 decision rules."""
        eid_perf = performance.filter(pl.col("variant_name") == "essential_input_dampener_only")
        improved = set(eid_perf.filter(pl.col("improvement_vs_historical_frontier_gap") > 0)["candidate_subtype"].to_list())
        physical = {
            "infrastructure_energy",
            "transport_logistics_infrastructure",
            "heavy_industry_materials",
            "manufacturing_system_core",
            "construction_real_estate_foundational",
        }
        pseudo_share = _as_float(pseudo["pseudo_agent_flag"].mean()) if not pseudo.is_empty() else 0.0
        harmful_count = _as_int(failure_modes.filter(pl.col("failure_mode") == "harmed_by_EID").height) if not failure_modes.is_empty() else 0
        if pseudo_share > 0.25:
            rec = "exclude_pseudo_agents_before_retesting"
            interp = "Pseudo-agent/accounting categories materially contaminate the high-EID validation set."
        elif "infrastructure_energy" in improved and not (improved & (physical - {"infrastructure_energy"})):
            rec = "split_EID_into_subtype_specific_diagnostics"
            interp = "EID dampening helps infrastructure energy more clearly than other high-EID subtypes."
        elif improved & physical:
            rec = "integrate_EID_candidate_in_multiyear_loop_for_audit"
            interp = "Several coherent physical high-EID subtypes benefit from the diagnostic dampener."
        elif harmful_count > 0 and not improved:
            rec = "abandon_EID_dampener_for_v4"
            interp = "No coherent subtype benefits enough and some high-EID nodes are harmed."
        else:
            rec = "keep_EID_dampener_diagnostic_only"
            interp = "High-EID heterogeneity remains too broad for a final active rule."
        return pl.DataFrame(
            [
                {
                    "recommendation": rec,
                    "evidence": f"improved_subtypes={sorted(improved)}; pseudo_share={pseudo_share:.3g}; harmed_nodes={harmful_count}",
                    "interpretation": interp,
                    "recommended_phase25": "Split high-EID diagnostics by physical infrastructure, heavy industry, systemic services, and pseudo-agent categories before any rule integration.",
                    "abm_v5_implication": "Use subtype evidence as ontology material only; do not implement agent types yet.",
                    "scenario_readiness": "premature",
                }
            ]
        )

    def build_markdown_report(
        self,
        composition: pl.DataFrame,
        performance: pl.DataFrame,
        failure_modes: pl.DataFrame,
        pseudo: pl.DataFrame,
        v5: pl.DataFrame,
        recommendation: pl.DataFrame,
    ) -> str:
        """Build the Phase 24 markdown report."""
        return "\n".join(
            [
                "# ABM v4 Phase 24 EID Failure-Mode and Heterogeneity Audit",
                "",
                f"Recommendation: `{recommendation['recommendation'].item(0) if not recommendation.is_empty() else 'inconclusive'}`.",
                "",
                "## Subtype Composition",
                self._markdown_table(composition),
                "",
                "## Dampener Performance by Subtype",
                self._markdown_table(performance),
                "",
                "## Failure Modes",
                self._markdown_table(failure_modes.head(30)),
                "",
                "## Pseudo-Agent Audit",
                self._markdown_table(pseudo.filter(pl.col("pseudo_agent_flag")).head(30) if not pseudo.is_empty() else pseudo),
                "",
                "## ABM v5 Agent-Type Candidate Audit",
                self._markdown_table(v5),
                "",
                "## Recommendation",
                self._markdown_table(recommendation),
                "",
                "Scenarios remain premature.",
            ]
        ) + "\n"

    def _best_candidate(self, validation: pl.DataFrame, variant_name: str) -> dict[str, Any]:
        rows = validation.filter((pl.col("variant_name") == variant_name) & (pl.col("train_or_validation") == "validation"))
        if rows.is_empty():
            return {}
        return rows.sort("all_node_emissions_weighted_rEI_MAE").to_dicts()[0]

    def _pseudo_agent_reason(self, country: Any, sector: Any, country_sector: Any = "") -> str | None:
        text = f"{country} {sector} {country_sector}".lower()
        if "re-export" in text or "re-import" in text or "reexport" in text or "reimport" in text:
            return "re_export_re_import"
        if "total" in text:
            return "TOTAL"
        if " rest of world" in f" {text}" or "row |" in text or text.startswith("row "):
            return "ROW"
        if "unclassified" in text or "not elsewhere" in text or "nec" in text:
            return "unclassified"
        if "aggregate" in text:
            return "aggregate_category"
        return None

    def _markdown_table(self, frame: pl.DataFrame) -> str:
        if frame.is_empty():
            return "_No rows available._"
        columns = frame.columns
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
        for row in frame.to_dicts():
            lines.append("| " + " | ".join(self._format_markdown_value(row.get(column)) for column in columns) + " |")
        return "\n".join(lines)

    def _format_markdown_value(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        if value is None:
            return ""
        return str(value).replace("|", "/")


@dataclass(frozen=True)
class MultiYearEIDDiagnosticValidationResult:
    """In-memory outputs for Phase 25 EID diagnostic multi-year validation."""

    error_panel: pl.DataFrame
    error_summary: pl.DataFrame
    by_sector: pl.DataFrame
    by_country: pl.DataFrame
    by_electricity: pl.DataFrame
    china_electricity: pl.DataFrame
    by_eid_decile: pl.DataFrame
    by_subtype: pl.DataFrame
    pseudo_agent_sensitivity: pl.DataFrame
    comparison: pl.DataFrame
    mechanism_audit: pl.DataFrame
    abm_v5_implications: pl.DataFrame
    recommendation: pl.DataFrame
    markdown: str


class MultiYearEIDDiagnosticValidator:
    """Validate the Phase 25 EID diagnostic multi-year candidate against baselines."""

    variant_specs = (
        ("frontier_gap_readiness", "base_multiyear_state_panel_path"),
        ("historical_frontier_gap_only", "base_multiyear_state_panel_historical_frontier_gap_path"),
        ("historical_frontier_gap_EID_diagnostic", "base_multiyear_state_panel_EID_diagnostic_path"),
    )

    def __init__(self, paths: ABMV4Paths) -> None:
        self.paths = paths

    def run(self) -> MultiYearEIDDiagnosticValidationResult:
        """Build Phase 25 validation outputs in memory."""
        panel = self.build_error_panel()
        comparison = self.build_comparison(panel)
        error_summary = self.build_error_summary(panel)
        by_sector = self.group_summary(panel, "Sector")
        by_country = self.group_summary(panel, "Country")
        by_electricity = self.group_summary(
            panel.with_columns(pl.col("electricity_like").cast(pl.Utf8).alias("electricity_group")),
            "electricity_group",
        )
        china = self.china_electricity_summary(panel)
        by_eid_decile = self.group_summary(panel, "EID_decile")
        by_subtype = self.subtype_summary(panel)
        pseudo = self.pseudo_agent_sensitivity(panel)
        mechanism = self.mechanism_audit(panel)
        v5 = self.abm_v5_implications(by_subtype)
        recommendation = self.build_recommendation(comparison, by_subtype, pseudo)
        markdown = self.build_markdown_report(comparison, by_subtype, pseudo, mechanism, v5, recommendation)
        return MultiYearEIDDiagnosticValidationResult(
            error_panel=panel,
            error_summary=error_summary,
            by_sector=by_sector,
            by_country=by_country,
            by_electricity=by_electricity,
            china_electricity=china,
            by_eid_decile=by_eid_decile,
            by_subtype=by_subtype,
            pseudo_agent_sensitivity=pseudo,
            comparison=comparison,
            mechanism_audit=mechanism,
            abm_v5_implications=v5,
            recommendation=recommendation,
            markdown=markdown,
        )

    def write_outputs(self, result: MultiYearEIDDiagnosticValidationResult) -> None:
        """Write Phase 25 outputs. The caller controls whether this is invoked."""
        self.paths.validation.mkdir(parents=True, exist_ok=True)
        result.error_panel.write_parquet(self.paths.multiyear_EID_diagnostic_error_panel_path)
        result.error_summary.write_csv(self.paths.multiyear_EID_diagnostic_error_summary_path)
        result.by_sector.write_csv(self.paths.multiyear_EID_diagnostic_by_sector_path)
        result.by_country.write_csv(self.paths.multiyear_EID_diagnostic_by_country_path)
        result.by_electricity.write_csv(self.paths.multiyear_EID_diagnostic_by_electricity_path)
        result.china_electricity.write_csv(self.paths.multiyear_EID_diagnostic_china_electricity_path)
        result.by_eid_decile.write_csv(self.paths.multiyear_EID_diagnostic_by_EID_decile_path)
        result.by_subtype.write_csv(self.paths.multiyear_EID_diagnostic_by_subtype_path)
        result.pseudo_agent_sensitivity.write_csv(self.paths.multiyear_EID_diagnostic_pseudo_agent_sensitivity_path)
        result.comparison.write_csv(self.paths.multiyear_EID_diagnostic_comparison_path)
        result.mechanism_audit.write_csv(self.paths.multiyear_EID_diagnostic_mechanism_audit_path)
        result.abm_v5_implications.write_csv(self.paths.multiyear_EID_diagnostic_abm_v5_implications_path)
        result.recommendation.write_csv(self.paths.multiyear_EID_diagnostic_recommendation_path)
        self.paths.multiyear_EID_diagnostic_report_path.write_text(result.markdown, encoding="utf-8")

    def load_variant_panels(self) -> dict[str, pl.DataFrame]:
        """Load the three required multi-year state panels."""
        panels: dict[str, pl.DataFrame] = {}
        missing: list[Path] = []
        for variant, attr in self.variant_specs:
            path = getattr(self.paths, attr)
            if not path.exists():
                missing.append(path)
                continue
            panels[variant] = pl.read_parquet(path).with_columns(pl.lit(variant).alias("model_variant"))
        if missing:
            raise FileNotFoundError(
                "Missing required multi-year outputs for Phase 25: "
                + ", ".join(str(path) for path in missing)
                + ". Run the default base, historical frontier-gap base, and "
                "`--run-multiyear-EID-diagnostic` first."
            )
        return panels

    def load_eid_metadata(self) -> pl.DataFrame:
        """Load Phase 23/24 high-EID subtype and pseudo-agent metadata."""
        if not self.paths.eid_high_node_heterogeneity_panel_path.exists():
            raise FileNotFoundError(
                f"Missing Phase 24 heterogeneity panel: {self.paths.eid_high_node_heterogeneity_panel_path}. "
                "Run --diagnose-eid-failure-modes first."
            )
        meta = pl.read_csv(self.paths.eid_high_node_heterogeneity_panel_path)
        if meta.is_empty():
            return pl.DataFrame()
        preferred = meta.filter(pl.col("high_EID_definition") == "structural_dependence_plus_brown_lockin")
        if preferred.is_empty():
            preferred = meta
        return preferred.sort("EID_norm", descending=True).unique("country_sector").select(
            "country_sector",
            "candidate_subtype",
            "EID_norm",
            "pseudo_agent_flag",
            "electricity_like",
            "emissions_share",
            "output_share",
        )

    def build_error_panel(self) -> pl.DataFrame:
        """Build long variant-node-year error panel."""
        meta = self.load_eid_metadata()
        frames = []
        for variant, frame in self.load_variant_panels().items():
            required = {
                "country_sector",
                "year",
                "Country",
                "Sector",
                "EI_sim",
                "EI_observed",
                "emissions_sim",
                "emissions_observed",
                "X_observed",
            }
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"{variant} panel is missing required columns: {sorted(missing)}")
            out = frame.sort(["country_sector", "year"]).with_columns(
                (pl.col("EI_sim") - pl.col("EI_observed")).alias("EI_error"),
                (pl.col("EI_sim") - pl.col("EI_observed")).abs().alias("EI_abs_error"),
                pl.when((pl.col("EI_sim") > 0) & (pl.col("EI_observed") > 0))
                .then(pl.col("EI_sim").log() - pl.col("EI_observed").log())
                .otherwise(None)
                .alias("log_EI_error"),
                (pl.col("emissions_sim") - pl.col("emissions_observed")).alias("emissions_error"),
                (pl.col("emissions_sim") - pl.col("emissions_observed")).abs().alias("emissions_abs_error"),
                pl.col("EI_observed").shift(-1).over("country_sector").alias("_EI_observed_next"),
                pl.col("EI_sim").shift(-1).over("country_sector").alias("_EI_sim_next"),
                pl.col("year").shift(-1).over("country_sector").alias("_year_next"),
            ).with_columns(
                pl.when((pl.col("_year_next") == pl.col("year") + 1) & (pl.col("EI_observed") > 0) & (pl.col("_EI_observed_next") > 0))
                .then(pl.col("EI_observed").log() - pl.col("_EI_observed_next").log())
                .otherwise(None)
                .alias("observed_rEI"),
                pl.when((pl.col("_year_next") == pl.col("year") + 1) & (pl.col("EI_sim") > 0) & (pl.col("_EI_sim_next") > 0))
                .then(pl.col("EI_sim").log() - pl.col("_EI_sim_next").log())
                .otherwise(None)
                .alias("simulated_rEI"),
            ).with_columns(
                (pl.col("simulated_rEI") - pl.col("observed_rEI")).alias("rEI_error"),
                (pl.col("simulated_rEI") - pl.col("observed_rEI")).abs().alias("rEI_abs_error"),
                (pl.col("simulated_rEI").sign() != pl.col("observed_rEI").sign()).cast(pl.Float64).alias("wrong_sign"),
                pl.lit(variant).alias("model_variant"),
            ).drop("_EI_observed_next", "_EI_sim_next", "_year_next")
            frames.append(out)
        panel = pl.concat(frames, how="diagonal_relaxed")
        if not meta.is_empty():
            panel = panel.join(meta, on="country_sector", how="left", suffix="_phase24")
        return self._add_metadata_defaults(panel)

    def _add_metadata_defaults(self, panel: pl.DataFrame) -> pl.DataFrame:
        for column in [
            "candidate_subtype",
            "EID_norm",
            "pseudo_agent_flag",
            "electricity_like",
            "emissions_share",
            "output_share",
        ]:
            suffixed = f"{column}_phase24"
            if suffixed in panel.columns:
                if column in panel.columns:
                    panel = panel.with_columns(pl.coalesce([pl.col(column), pl.col(suffixed)]).alias(column))
                else:
                    panel = panel.rename({suffixed: column})
        if "candidate_subtype" not in panel.columns:
            panel = panel.with_columns(pl.lit("ordinary_or_unclear").alias("candidate_subtype"))
        if "EID_norm" not in panel.columns:
            panel = panel.with_columns(pl.lit(None, dtype=pl.Float64).alias("EID_norm"))
        if "pseudo_agent_flag" not in panel.columns:
            panel = panel.with_columns(pl.lit(False).alias("pseudo_agent_flag"))
        if "electricity_like" not in panel.columns:
            panel = panel.with_columns(
                pl.col("Sector").str.to_lowercase().str.contains("electricity|gas and water|utilities|power|water|gas").alias("electricity_like")
            )
        return panel.with_columns(
            pl.col("candidate_subtype").fill_null("ordinary_or_unclear"),
            pl.col("pseudo_agent_flag").fill_null(False),
            pl.col("electricity_like").fill_null(False),
            pl.when(pl.col("EID_norm").is_null()).then(pl.lit("missing")).otherwise((pl.col("EID_norm") * 10).floor().clip(0, 9).cast(pl.Int64).cast(pl.Utf8)).alias("EID_decile"),
            (
                pl.col("Country").str.to_lowercase().is_in(["chn", "china"])
                & pl.col("Sector").str.to_lowercase().str.contains("electricity|gas and water|utilities|power|water|gas")
            ).alias("china_electricity"),
        )

    def build_comparison(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Build one-row validation comparison per model variant."""
        rows = [self._metric_row(frame, self._group_key_to_string(variant)) for variant, frame in panel.group_by("model_variant")]
        return pl.DataFrame(rows).sort("model_variant")

    def _metric_row(self, frame: pl.DataFrame, variant: str) -> dict[str, Any]:
        yearly = frame.group_by("year").agg(
            pl.col("emissions_observed").sum().alias("obs"),
            pl.col("emissions_sim").sum().alias("sim"),
            pl.col("emissions_abs_error").sum().alias("abs_error"),
        ).with_columns(
            ((pl.col("sim") - pl.col("obs")).abs() / pl.col("obs")).alias("pct_error")
        )
        latest_year = yearly["year"].max()
        elec = frame.filter(pl.col("electricity_like"))
        china = frame.filter(pl.col("china_electricity"))
        high = frame.filter(pl.col("EID_norm").fill_null(0) >= 0.9)
        return {
            "model_variant": variant,
            "rows": frame.height,
            "unweighted_rEI_MAE": _as_float(frame["rEI_abs_error"].mean()),
            "output_weighted_rEI_MAE": self._weighted_mean(frame, "rEI_abs_error", "X_observed"),
            "emissions_weighted_rEI_MAE": self._weighted_mean(frame, "rEI_abs_error", "emissions_observed"),
            "wrong_sign_share": _as_float(frame["wrong_sign"].mean()),
            "output_weighted_wrong_sign_share": self._weighted_mean(frame, "wrong_sign", "X_observed"),
            "emissions_weighted_wrong_sign_share": self._weighted_mean(frame, "wrong_sign", "emissions_observed"),
            "validation_bias": _as_float(frame["rEI_error"].mean()),
            "validation_correlation": _as_float(frame.select(pl.corr("simulated_rEI", "observed_rEI")).item()),
            "latest_year_aggregate_emissions_pct_error": _as_float(yearly.filter(pl.col("year") == latest_year)["pct_error"].sum()),
            "mean_yearly_aggregate_emissions_pct_error": _as_float(yearly["pct_error"].mean()),
            "total_emissions_absolute_error": _as_float(frame["emissions_abs_error"].sum()),
            "mean_emissions_absolute_error": _as_float(frame["emissions_abs_error"].mean()),
            "electricity_rEI_MAE": _as_float(elec["rEI_abs_error"].mean()) if not elec.is_empty() else None,
            "electricity_emissions_weighted_rEI_MAE": self._weighted_mean(elec, "rEI_abs_error", "emissions_observed"),
            "electricity_wrong_sign_share": _as_float(elec["wrong_sign"].mean()) if not elec.is_empty() else None,
            "electricity_aggregate_emissions_error": _as_float(elec["emissions_abs_error"].sum()) if not elec.is_empty() else None,
            "china_electricity_rEI_MAE": _as_float(china["rEI_abs_error"].mean()) if not china.is_empty() else None,
            "china_electricity_emissions_error": _as_float(china["emissions_abs_error"].sum()) if not china.is_empty() else None,
            "china_electricity_wrong_sign_share": _as_float(china["wrong_sign"].mean()) if not china.is_empty() else None,
            "high_EID_node_rEI_MAE": _as_float(high["rEI_abs_error"].mean()) if not high.is_empty() else None,
            "high_EID_node_emissions_weighted_rEI_MAE": self._weighted_mean(high, "rEI_abs_error", "emissions_observed"),
            "high_EID_node_emissions_error": _as_float(high["emissions_abs_error"].sum()) if not high.is_empty() else None,
            "high_EID_node_wrong_sign_share": _as_float(high["wrong_sign"].mean()) if not high.is_empty() else None,
        }

    def build_error_summary(self, panel: pl.DataFrame) -> pl.DataFrame:
        return panel.group_by("model_variant", "year").agg(
            pl.col("emissions_observed").sum().alias("total_emissions_observed"),
            pl.col("emissions_sim").sum().alias("total_emissions_sim"),
            pl.col("emissions_abs_error").sum().alias("total_emissions_absolute_error"),
            pl.col("rEI_abs_error").mean().alias("rEI_MAE"),
            pl.col("wrong_sign").mean().alias("wrong_sign_share"),
        ).with_columns(
            ((pl.col("total_emissions_sim") - pl.col("total_emissions_observed")).abs() / pl.col("total_emissions_observed")).alias("aggregate_emissions_pct_error")
        ).sort(["model_variant", "year"])

    def group_summary(self, panel: pl.DataFrame, group_col: str) -> pl.DataFrame:
        if group_col not in panel.columns:
            return pl.DataFrame()
        base = panel.group_by("model_variant", group_col).agg(
            pl.len().alias("rows"),
            pl.col("country_sector").n_unique().alias("nodes"),
            pl.col("emissions_observed").sum().alias("_obs_emissions"),
            pl.col("X_observed").sum().alias("_obs_output"),
            pl.col("rEI_abs_error").mean().alias("rEI_MAE"),
            (pl.col("rEI_abs_error") * pl.col("emissions_observed")).sum().truediv(pl.col("emissions_observed").sum()).alias("emissions_weighted_rEI_MAE"),
            pl.col("wrong_sign").mean().alias("wrong_sign_share"),
            pl.col("emissions_abs_error").sum().alias("emissions_error"),
        ).rename({group_col: "group"})
        total_emissions = _as_float(panel.filter(pl.col("model_variant") == "historical_frontier_gap_only")["emissions_observed"].sum())
        return base.with_columns(
            (pl.col("_obs_emissions") / max(total_emissions, 1e-12)).alias("observed_emissions_share"),
            pl.lit("diagnostic group validation").alias("interpretation"),
        ).drop("_obs_emissions", "_obs_output")

    def subtype_summary(self, panel: pl.DataFrame) -> pl.DataFrame:
        out = self.group_summary(panel, "candidate_subtype")
        if out.is_empty():
            return out
        hist = out.filter(pl.col("model_variant") == "historical_frontier_gap_only").select("group", pl.col("emissions_weighted_rEI_MAE").alias("_hist_mae"))
        ready = out.filter(pl.col("model_variant") == "frontier_gap_readiness").select("group", pl.col("emissions_weighted_rEI_MAE").alias("_ready_mae"))
        return out.join(hist, on="group", how="left").join(ready, on="group", how="left").with_columns(
            (pl.col("_ready_mae") - pl.col("emissions_weighted_rEI_MAE")).alias("improvement_vs_frontier_gap_readiness"),
            (pl.col("_hist_mae") - pl.col("emissions_weighted_rEI_MAE")).alias("improvement_vs_historical_frontier_gap_only"),
            ((pl.col("emissions_weighted_rEI_MAE") - pl.col("_hist_mae")) / pl.col("_hist_mae").abs() > 0.02).fill_null(False).alias("material_worsening_flag"),
        ).rename({"group": "candidate_subtype"}).drop("_hist_mae", "_ready_mae")

    def china_electricity_summary(self, panel: pl.DataFrame) -> pl.DataFrame:
        return panel.filter(pl.col("china_electricity")).select(
            "model_variant", "country_sector", "year", "Country", "Sector", "EI_observed", "EI_sim", "emissions_observed", "emissions_sim", "observed_rEI", "simulated_rEI", "rEI_error", "emissions_error"
        )

    def pseudo_agent_sensitivity(self, panel: pl.DataFrame) -> pl.DataFrame:
        rows = []
        for include in [True, False]:
            frame = panel if include else panel.filter(~pl.col("pseudo_agent_flag"))
            for variant, group in frame.group_by("model_variant"):
                variant_name = self._group_key_to_string(variant)
                row = self._metric_row(group, variant_name)
                rows.append(
                    {
                        "model_variant": variant_name,
                        "pseudo_agent_scope": "including_pseudo_agents" if include else "excluding_pseudo_agents",
                        "rows": group.height,
                        "emissions_weighted_rEI_MAE": row["emissions_weighted_rEI_MAE"],
                        "mean_yearly_aggregate_emissions_pct_error": row["mean_yearly_aggregate_emissions_pct_error"],
                        "high_EID_node_rEI_MAE": row["high_EID_node_rEI_MAE"],
                        "interpretation": "pseudo-agent sensitivity diagnostic",
                    }
                )
        return pl.DataFrame(rows)

    def mechanism_audit(self, panel: pl.DataFrame) -> pl.DataFrame:
        eid = panel.filter(pl.col("model_variant") == "historical_frontier_gap_EID_diagnostic")
        hist = panel.filter(pl.col("model_variant") == "historical_frontier_gap_only").select(
            "country_sector",
            "year",
            pl.col("rEI_error").alias("_hist_rEI_error"),
            pl.col("emissions_abs_error").alias("_hist_emissions_abs_error"),
            pl.col("rEI_used").alias("_hist_rEI_used") if "rEI_used" in panel.columns else pl.lit(None).alias("_hist_rEI_used"),
        )
        joined = eid.join(hist, on=["country_sector", "year"], how="left")
        return joined.group_by("country_sector", "Country", "Sector", "candidate_subtype").agg(
            pl.col("EID_norm").mean().alias("EID_norm"),
            pl.col("D_EID").mean().alias("D_EID") if "D_EID" in joined.columns else pl.lit(None).alias("D_EID"),
            pl.col("EID_missing_flag").mean().alias("EID_missing_flag") if "EID_missing_flag" in joined.columns else pl.lit(None).alias("EID_missing_flag"),
            pl.col("EID_fallback_flag").mean().alias("EID_fallback_flag") if "EID_fallback_flag" in joined.columns else pl.lit(None).alias("EID_fallback_flag"),
            pl.col("electricity_like").max().alias("electricity_like"),
            pl.col("pseudo_agent_flag").max().alias("pseudo_agent_flag"),
            pl.col("emissions_share").mean().alias("observed_emissions_share") if "emissions_share" in joined.columns else pl.lit(None).alias("observed_emissions_share"),
            pl.col("output_share").mean().alias("observed_output_share") if "output_share" in joined.columns else pl.lit(None).alias("observed_output_share"),
            pl.col("ei_gap").mean().alias("mean_frontier_gap") if "ei_gap" in joined.columns else pl.lit(None).alias("mean_frontier_gap"),
            pl.col("_hist_rEI_used").mean().alias("mean_gap_closure_without_EID"),
            pl.col("rEI_used").mean().alias("mean_gap_closure_with_EID") if "rEI_used" in joined.columns else pl.lit(None).alias("mean_gap_closure_with_EID"),
            (pl.col("_hist_rEI_used") - pl.col("rEI_used")).mean().alias("mean_gap_closure_reduction") if "rEI_used" in joined.columns else pl.lit(None).alias("mean_gap_closure_reduction"),
            (pl.col("emissions_abs_error") - pl.col("_hist_emissions_abs_error")).mean().alias("emissions_error_change_vs_historical_frontier_gap"),
            (pl.col("rEI_error") - pl.col("_hist_rEI_error")).mean().alias("rEI_error_change_vs_historical_frontier_gap"),
        ).with_columns(
            pl.when(pl.col("EID_fallback_flag") > 0)
            .then(pl.lit("EID score missing for at least part of the run; fallback D_EID=1 used."))
            .when(pl.col("mean_gap_closure_reduction") > 0)
            .then(pl.lit("EID dampener reduces frontier-gap closure."))
            .otherwise(pl.lit("No material EID dampening detected."))
            .alias("interpretation")
        )

    def abm_v5_implications(self, subtype: pl.DataFrame) -> pl.DataFrame:
        rows = []
        eid = subtype.filter(pl.col("model_variant") == "historical_frontier_gap_EID_diagnostic")
        for group, agent in [
            ("infrastructure_energy", "energy_infrastructure_agent"),
            ("transport_logistics_infrastructure", "transport_logistics_infrastructure_agent"),
            ("heavy_industry_materials", "heavy_industry_materials_agent"),
            ("knowledge_finance_business_services", "systemic_service_agent"),
            ("accounting_or_pseudo_agent", "accounting_node_not_agent"),
        ]:
            row = eid.filter(pl.col("candidate_subtype") == group)
            improvement = _as_float(row["improvement_vs_historical_frontier_gap_only"].item(0)) if not row.is_empty() else 0.0
            supported = improvement > 0 and group != "accounting_or_pseudo_agent"
            rows.append(
                {
                    "finding": f"{group} EID multi-year evidence",
                    "evidence": f"improvement_vs_historical={improvement:.6g}",
                    "implication_for_abm_v5": "candidate ontology evidence" if supported else "not a behavioural agent candidate",
                    "agent_type_candidate": agent if supported or group == "accounting_or_pseudo_agent" else "not_supported",
                    "confidence_level": "moderate" if improvement > 0.01 and supported else ("low" if supported else "not_supported"),
                    "notes": "No ABM v5 code implemented.",
                }
            )
        return pl.DataFrame(rows)

    def build_recommendation(self, comparison: pl.DataFrame, subtype: pl.DataFrame, pseudo: pl.DataFrame) -> pl.DataFrame:
        base = self._comparison_row(comparison, "historical_frontier_gap_only")
        eid = self._comparison_row(comparison, "historical_frontier_gap_EID_diagnostic")
        material = self.material_worsening_flags(base, eid)
        eid_sub = (
            subtype.filter(pl.col("model_variant") == "historical_frontier_gap_EID_diagnostic")
            if not subtype.is_empty() and "model_variant" in subtype.columns
            else pl.DataFrame()
        )
        physical = {"infrastructure_energy", "transport_logistics_infrastructure", "heavy_industry_materials", "manufacturing_system_core", "construction_real_estate_foundational"}
        if eid_sub.is_empty() or "candidate_subtype" not in eid_sub.columns:
            improved_physical: set[str] = set()
            worsened: set[str] = set()
        else:
            improved_physical = set(eid_sub.filter((pl.col("candidate_subtype").is_in(list(physical))) & (pl.col("improvement_vs_historical_frontier_gap_only") > 0))["candidate_subtype"].to_list())
            worsened = set(eid_sub.filter(pl.col("material_worsening_flag"))["candidate_subtype"].to_list())
        include = (
            pseudo.filter(pl.col("pseudo_agent_scope") == "including_pseudo_agents")
            if not pseudo.is_empty() and "pseudo_agent_scope" in pseudo.columns
            else pl.DataFrame()
        )
        exclude = (
            pseudo.filter(pl.col("pseudo_agent_scope") == "excluding_pseudo_agents")
            if not pseudo.is_empty() and "pseudo_agent_scope" in pseudo.columns
            else pl.DataFrame()
        )
        pseudo_dependence = False
        if not include.is_empty() and not exclude.is_empty():
            inc_eid = self._pseudo_metric(include, "historical_frontier_gap_EID_diagnostic")
            exc_eid = self._pseudo_metric(exclude, "historical_frontier_gap_EID_diagnostic")
            inc_hist = self._pseudo_metric(include, "historical_frontier_gap_only")
            exc_hist = self._pseudo_metric(exclude, "historical_frontier_gap_only")
            pseudo_dependence = (inc_hist - inc_eid) > 0 and (exc_hist - exc_eid) <= 0
        if any(material.values()):
            rec = "reject_EID_for_v4"
            interp = "EID diagnostic materially worsens at least one all-node validation threshold."
        elif pseudo_dependence:
            rec = "exclude_pseudo_agents_and_retest"
            interp = "EID improvement depends on pseudo-agent/accounting nodes."
        elif self._improves(eid, base, "mean_yearly_aggregate_emissions_pct_error") and self._improves(eid, base, "electricity_rEI_MAE") and self._improves(eid, base, "china_electricity_rEI_MAE") and improved_physical:
            rec = "promote_EID_to_provisional_base_candidate"
            interp = "EID improves aggregate, electricity, China electricity, and physical high-EID subtype metrics without material all-node worsening."
        elif improved_physical and worsened:
            rec = "split_EID_by_subtype_before_integration"
            interp = "EID helps some physical subtypes but materially worsens others."
        elif improved_physical:
            rec = "keep_EID_diagnostic_only"
            interp = "EID has useful subtype evidence but is not robust enough for base selection."
        else:
            rec = "inconclusive"
            interp = "EID multi-year evidence is weak or mixed."
        return pl.DataFrame(
            [
                {
                    "recommendation": rec,
                    "evidence": f"material_worsening={material}; improved_physical={sorted(improved_physical)}; worsened_subtypes={sorted(worsened)}; pseudo_dependence={pseudo_dependence}",
                    "interpretation": interp,
                    "recommended_phase26": "Keep scenarios blocked; if promoted, compare provisional base candidates under explicit validation-objective selection.",
                    "abm_v5_implication": "Use multi-year subtype evidence as ontology input only; no agent types implemented.",
                    "scenario_readiness": "premature",
                }
            ]
        )

    def material_worsening_flags(self, baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, bool]:
        return {
            "all_node_rEI_MAE": self._relative_worse(candidate, baseline, "unweighted_rEI_MAE", 0.02),
            "emissions_weighted_rEI_MAE": self._relative_worse(candidate, baseline, "emissions_weighted_rEI_MAE", 0.02),
            "wrong_sign_share": (_as_float(candidate.get("wrong_sign_share")) - _as_float(baseline.get("wrong_sign_share"))) > 0.02,
            "aggregate_emissions_pct_error": (_as_float(candidate.get("mean_yearly_aggregate_emissions_pct_error")) - _as_float(baseline.get("mean_yearly_aggregate_emissions_pct_error"))) > 0.02,
        }

    def build_markdown_report(self, comparison: pl.DataFrame, subtype: pl.DataFrame, pseudo: pl.DataFrame, mechanism: pl.DataFrame, v5: pl.DataFrame, recommendation: pl.DataFrame) -> str:
        return "\n".join(
            [
                "# ABM v4 Phase 25 EID Diagnostic Multi-Year Integration Audit",
                "",
                f"Recommendation: `{recommendation['recommendation'].item(0) if not recommendation.is_empty() else 'inconclusive'}`.",
                "",
                "This is a historical diagnostic run only. It is not a scenario and does not activate EID as the default rule.",
                "",
                "## Model Comparison",
                self._markdown_table(comparison),
                "",
                "## Subtype Validation",
                self._markdown_table(subtype),
                "",
                "## Pseudo-Agent Sensitivity",
                self._markdown_table(pseudo),
                "",
                "## Mechanism Audit",
                self._markdown_table(mechanism.head(30)),
                "",
                "## ABM v5 Implications",
                self._markdown_table(v5),
                "",
                "## Recommendation",
                self._markdown_table(recommendation),
                "",
                "Scenarios remain premature.",
            ]
        ) + "\n"

    def _weighted_mean(self, frame: pl.DataFrame, value: str, weight: str) -> float | None:
        if frame.is_empty() or value not in frame.columns or weight not in frame.columns:
            return None
        denom = _as_float(frame[weight].sum())
        return None if denom == 0 else _as_float((frame[value] * frame[weight]).sum() / denom)

    def _comparison_row(self, comparison: pl.DataFrame, variant: str) -> dict[str, Any]:
        rows = comparison.filter(pl.col("model_variant") == variant)
        return rows.to_dicts()[0] if not rows.is_empty() else {}

    def _group_key_to_string(self, key: Any) -> str:
        if isinstance(key, tuple):
            return str(key[0]) if key else ""
        return str(key)

    def _pseudo_metric(self, frame: pl.DataFrame, variant: str) -> float:
        rows = frame.filter(pl.col("model_variant") == variant)
        return _as_float(rows["emissions_weighted_rEI_MAE"].item(0)) if not rows.is_empty() else float("nan")

    def _improves(self, candidate: dict[str, Any], baseline: dict[str, Any], metric: str) -> bool:
        return _as_float(candidate.get(metric), 999.0) < _as_float(baseline.get(metric), 0.0)

    def _relative_worse(self, candidate: dict[str, Any], baseline: dict[str, Any], metric: str, threshold: float) -> bool:
        base = _as_float(baseline.get(metric))
        cand = _as_float(candidate.get(metric))
        return base != 0 and (cand - base) / abs(base) > threshold

    def _markdown_table(self, frame: pl.DataFrame) -> str:
        if frame.is_empty():
            return "_No rows available._"
        columns = frame.columns
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
        for row in frame.to_dicts():
            lines.append("| " + " | ".join(self._format_markdown_value(row.get(column)) for column in columns) + " |")
        return "\n".join(lines)

    def _format_markdown_value(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        if value is None:
            return ""
        return str(value).replace("|", "/")


@dataclass(frozen=True)
class AdaptiveEIDCalibrationResult:
    """In-memory outputs for Phase 26 adaptive EID calibration diagnostics."""

    parameter_grid: pl.DataFrame
    windows: pl.DataFrame
    calibration_results: pl.DataFrame
    validation_panel: pl.DataFrame
    model_comparison: pl.DataFrame
    parameter_stability: pl.DataFrame
    by_subtype: pl.DataFrame
    pseudo_agent_sensitivity: pl.DataFrame
    hypothesis_tests: pl.DataFrame
    recommendation: pl.DataFrame
    markdown: str


@dataclass(frozen=True)
class QEnergyMixAuditResult:
    """In-memory outputs for Phase 27 Q energy-mix diagnostics."""

    source_inventory: pl.DataFrame
    row_mapping: pl.DataFrame
    energy_mix_panel: pl.DataFrame
    quality_audit: pl.DataFrame
    quality_by_year: pl.DataFrame
    quality_by_sector: pl.DataFrame
    quality_by_country: pl.DataFrame
    aggregate_plausibility: pl.DataFrame
    china_electricity_audit: pl.DataFrame
    transition_error_panel: pl.DataFrame
    predictor_screening: pl.DataFrame
    by_subtype: pl.DataFrame
    hypothesis_tests: pl.DataFrame
    recommendation: pl.DataFrame
    markdown: str


class AdaptiveEIDCalibrationDiagnostics:
    """Walk-forward diagnostic screen for discrete EID dampener parameters."""

    objectives = (
        "transition_accuracy",
        "emissions_weighted_transition",
        "aggregate_emissions_fit",
        "balanced_policy_objective",
        "electricity_high_EID_objective",
    )

    def __init__(self, paths: ABMV4Paths) -> None:
        self.paths = paths

    def run(self) -> AdaptiveEIDCalibrationResult:
        grid = self.build_parameter_grid()
        windows = self.build_rolling_windows()
        base = self.build_base_panel()
        calibration_results, validation_panel = self.calibrate_and_validate(base, grid, windows)
        comparison = self.build_model_comparison(validation_panel)
        stability = self.build_parameter_stability(calibration_results)
        by_subtype = self.build_group_diagnostics(validation_panel, "candidate_subtype")
        pseudo = self.build_pseudo_agent_sensitivity(validation_panel)
        hypotheses = self.build_hypothesis_tests(calibration_results, comparison, stability, by_subtype)
        recommendation = self.build_recommendation(comparison, stability, hypotheses, by_subtype, calibration_results)
        markdown = self.build_markdown_report(comparison, stability, hypotheses, recommendation)
        return AdaptiveEIDCalibrationResult(
            parameter_grid=grid,
            windows=windows,
            calibration_results=calibration_results,
            validation_panel=validation_panel,
            model_comparison=comparison,
            parameter_stability=stability,
            by_subtype=by_subtype,
            pseudo_agent_sensitivity=pseudo,
            hypothesis_tests=hypotheses,
            recommendation=recommendation,
            markdown=markdown,
        )

    def write_outputs(self, result: AdaptiveEIDCalibrationResult) -> None:
        self.paths.validation.mkdir(parents=True, exist_ok=True)
        result.parameter_grid.write_csv(self.paths.adaptive_EID_parameter_grid_path)
        result.windows.write_csv(self.paths.adaptive_EID_calibration_windows_path)
        result.calibration_results.write_csv(self.paths.adaptive_EID_calibration_results_path)
        result.validation_panel.write_parquet(self.paths.adaptive_EID_validation_panel_path)
        result.model_comparison.write_csv(self.paths.adaptive_EID_model_comparison_path)
        result.parameter_stability.write_csv(self.paths.adaptive_EID_parameter_stability_path)
        result.by_subtype.write_csv(self.paths.adaptive_EID_by_subtype_path)
        result.pseudo_agent_sensitivity.write_csv(self.paths.adaptive_EID_pseudo_agent_sensitivity_path)
        result.hypothesis_tests.write_csv(self.paths.adaptive_EID_hypothesis_tests_path)
        result.recommendation.write_csv(self.paths.adaptive_EID_recommendation_path)
        self.paths.adaptive_EID_report_path.write_text(result.markdown, encoding="utf-8")

    def build_parameter_grid(self) -> pl.DataFrame:
        rows = []
        pid = 1
        for lambda_eid in [0.0, 0.25, 0.50, 0.75, 1.00]:
            for d_min in [0.25, 0.50, 0.75, 1.00]:
                rows.append(
                    {
                        "parameter_id": f"p{pid:03d}",
                        "lambda_EID": lambda_eid,
                        "d_min": d_min,
                        "notes": "discrete diagnostic grid; not continuous optimization",
                    }
                )
                pid += 1
        return pl.DataFrame(rows)

    def build_rolling_windows(self) -> pl.DataFrame:
        rows: list[dict[str, Any]] = []
        for start in [1995, 1998, 2001, 2004, 2007, 2010]:
            cal_end = start + 4
            val_start = cal_end + 1
            val_end = min(val_start + 2, 2016)
            if val_start <= 2016:
                rows.append(self._window_row("rolling_5yr_cal_3yr_validation", start, cal_end, val_start, val_end))
        for start in range(1995, 2014, 2):
            cal_end = start + 2
            val_start = cal_end + 1
            val_end = min(val_start + 1, 2016)
            if val_start <= 2016:
                rows.append(self._window_row("rolling_3yr_cal_2yr_validation", start, cal_end, val_start, val_end))
        return pl.DataFrame(rows).with_row_index("window_index").with_columns(
            (pl.lit("w") + (pl.col("window_index") + 1).cast(pl.Utf8).str.zfill(2)).alias("window_id")
        ).drop("window_index").select(
            "window_id",
            "design_name",
            "calibration_start_year",
            "calibration_end_year",
            "validation_start_year",
            "validation_end_year",
            "calibration_years",
            "validation_years",
            "notes",
        )

    def _window_row(self, design: str, cal_start: int, cal_end: int, val_start: int, val_end: int) -> dict[str, Any]:
        return {
            "design_name": design,
            "calibration_start_year": cal_start,
            "calibration_end_year": cal_end,
            "validation_start_year": val_start,
            "validation_end_year": val_end,
            "calibration_years": cal_end - cal_start + 1,
            "validation_years": val_end - val_start + 1,
            "notes": "walk-forward; validation years are not used for selection",
        }

    def build_base_panel(self) -> pl.DataFrame:
        if not self.paths.base_multiyear_state_panel_historical_frontier_gap_path.exists():
            raise FileNotFoundError(
                f"Missing historical frontier-gap state panel: {self.paths.base_multiyear_state_panel_historical_frontier_gap_path}. "
                "Run Phase 15 first."
            )
        if not self.paths.essential_input_dampener_scores_path.exists():
            raise FileNotFoundError(
                f"Missing Phase 23 EID scores: {self.paths.essential_input_dampener_scores_path}. "
                "Run --test-essential-input-dampener first."
            )
        if not self.paths.eid_high_node_heterogeneity_panel_path.exists():
            raise FileNotFoundError(
                f"Missing Phase 24 high-EID panel: {self.paths.eid_high_node_heterogeneity_panel_path}. "
                "Run --diagnose-eid-failure-modes first."
            )
        sim = pl.read_parquet(self.paths.base_multiyear_state_panel_historical_frontier_gap_path)
        scores = pl.read_csv(self.paths.essential_input_dampener_scores_path).filter(
            pl.col("EID_score_name") == "structural_dependence_plus_brown_lockin"
        ).select("country_sector", "EID_norm")
        meta = pl.read_csv(self.paths.eid_high_node_heterogeneity_panel_path).filter(
            pl.col("high_EID_definition") == "structural_dependence_plus_brown_lockin"
        ).unique("country_sector").select(
            "country_sector",
            "candidate_subtype",
            "pseudo_agent_flag",
            "electricity_like",
        )
        panel = sim.sort(["country_sector", "year"]).with_columns(
            pl.col("EI_observed").shift(-1).over("country_sector").alias("_EI_observed_next"),
            pl.col("year").shift(-1).over("country_sector").alias("_year_next"),
        ).with_columns(
            pl.when((pl.col("_year_next") == pl.col("year") + 1) & (pl.col("EI_observed") > 0) & (pl.col("_EI_observed_next") > 0))
            .then(pl.col("EI_observed").log() - pl.col("_EI_observed_next").log())
            .otherwise(None)
            .alias("observed_rEI")
        )
        if "sector_background_trend" not in panel.columns:
            panel = panel.with_columns(pl.lit(0.0).alias("sector_background_trend"))
        if "rEI_used" not in panel.columns:
            panel = panel.with_columns(pl.lit(0.0).alias("rEI_used"))
        out = panel.join(scores, on="country_sector", how="left").join(meta, on="country_sector", how="left", suffix="_meta")
        return out.with_columns(
            (pl.col("rEI_used") - pl.col("sector_background_trend").fill_null(0.0)).alias("_gap_closure_without_EID"),
            pl.col("EID_norm").is_null().alias("EID_fallback_flag"),
            pl.col("EID_norm").fill_null(0.0),
            pl.col("candidate_subtype").fill_null("ordinary_or_unclear"),
            pl.col("pseudo_agent_flag").fill_null(False),
            pl.when(pl.col("electricity_like").is_null())
            .then(pl.col("Sector").str.to_lowercase().str.contains("electricity|gas and water|utilities|power|water|gas"))
            .otherwise(pl.col("electricity_like"))
            .alias("electricity_like"),
            (
                pl.col("Country").str.to_lowercase().is_in(["chn", "china"])
                & pl.col("Sector").str.to_lowercase().str.contains("electricity|gas and water|utilities|power|water|gas")
            ).alias("china_electricity"),
            (pl.col("EID_norm").fill_null(0.0) >= 0.9).alias("high_EID_flag"),
        ).drop("_EI_observed_next", "_year_next")

    def calibrate_and_validate(self, base: pl.DataFrame, grid: pl.DataFrame, windows: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        result_rows = []
        selected_frames = []
        for window in windows.to_dicts():
            cal = base.filter(pl.col("year").is_between(window["calibration_start_year"], window["calibration_end_year"]))
            val = base.filter(pl.col("year").is_between(window["validation_start_year"], window["validation_end_year"]))
            if cal.is_empty() or val.is_empty():
                continue
            scored = []
            for param in grid.to_dicts():
                cal_pred = self.apply_parameter(cal, param)
                metrics = self.compute_metrics(cal_pred)
                scored.append({**window, **param, **{f"cal_{k}": v for k, v in metrics.items()}})
            scored_frame = pl.DataFrame(scored)
            for objective in self.objectives:
                objective_scored = self.add_objective_score(scored_frame, objective)
                best = objective_scored.sort("calibration_score").to_dicts()[0]
                val_pred = self.apply_parameter(val, best).with_columns(
                    pl.lit(window["window_id"]).alias("window_id"),
                    pl.lit(window["design_name"]).alias("design_name"),
                    pl.lit(objective).alias("objective_name"),
                    pl.lit(best["lambda_EID"]).alias("lambda_EID"),
                    pl.lit(best["d_min"]).alias("d_min"),
                    pl.lit("validation").alias("validation_or_calibration"),
                )
                cal_selected = self.apply_parameter(cal, best).with_columns(
                    pl.lit(window["window_id"]).alias("window_id"),
                    pl.lit(window["design_name"]).alias("design_name"),
                    pl.lit(objective).alias("objective_name"),
                    pl.lit(best["lambda_EID"]).alias("lambda_EID"),
                    pl.lit(best["d_min"]).alias("d_min"),
                    pl.lit("calibration").alias("validation_or_calibration"),
                )
                val_metrics = self.compute_metrics(val_pred)
                result_rows.append(
                    {
                        "design_name": window["design_name"],
                        "window_id": window["window_id"],
                        "objective_name": objective,
                        "parameter_id": best["parameter_id"],
                        "lambda_EID": best["lambda_EID"],
                        "d_min": best["d_min"],
                        "calibration_score": best["calibration_score"],
                        "selected_flag": True,
                        "validation_score": self.objective_value(val_metrics, objective),
                        "validation_all_node_rEI_MAE": val_metrics["all_node_unweighted_rEI_MAE"],
                        "validation_emissions_weighted_rEI_MAE": val_metrics["emissions_weighted_rEI_MAE"],
                        "validation_wrong_sign_share": val_metrics["wrong_sign_share"],
                        "validation_mean_yearly_aggregate_emissions_pct_error": val_metrics["mean_yearly_aggregate_emissions_pct_error"],
                        "validation_electricity_emissions_error": val_metrics["electricity_emissions_error"],
                        "validation_china_electricity_emissions_error": val_metrics["china_electricity_emissions_error"],
                        "validation_high_EID_emissions_error": val_metrics["high_EID_node_emissions_error"],
                        "notes": "selected on calibration years only; evaluated on future years",
                    }
                )
                selected_frames.extend([cal_selected, val_pred])
        return pl.DataFrame(result_rows), pl.concat(selected_frames, how="diagonal_relaxed")

    def apply_parameter(self, frame: pl.DataFrame, param: dict[str, Any]) -> pl.DataFrame:
        lambda_eid = _as_float(param.get("lambda_EID"))
        d_min = _as_float(param.get("d_min"), 1.0)
        return frame.with_columns(
            (1.0 - lambda_eid * pl.col("EID_norm").fill_null(0.0)).clip(d_min, 1.0).alias("D_EID")
        ).with_columns(
            (
                pl.col("sector_background_trend").fill_null(0.0)
                + pl.col("D_EID") * pl.col("_gap_closure_without_EID").fill_null(0.0)
            ).alias("predicted_rEI")
        ).with_columns(
            (pl.col("predicted_rEI") - pl.col("observed_rEI")).alias("rEI_error"),
            (pl.col("predicted_rEI").sign() != pl.col("observed_rEI").sign()).cast(pl.Float64).alias("wrong_sign"),
            (pl.col("emissions_observed") * (-pl.col("predicted_rEI").fill_null(0.0)).exp()).alias("emissions_predicted"),
        ).with_columns(
            pl.col("rEI_error").abs().alias("rEI_abs_error"),
        ).with_columns(
            (pl.col("emissions_observed") * pl.col("rEI_abs_error").fill_null(0.0)).alias("emissions_error"),
        )

    def compute_metrics(self, frame: pl.DataFrame) -> dict[str, float]:
        valid = frame.filter(pl.col("observed_rEI").is_not_null() & pl.col("predicted_rEI").is_not_null())
        yearly = valid.group_by("year").agg(
            pl.col("emissions_observed").sum().alias("obs"),
            pl.col("emissions_predicted").sum().alias("pred"),
        ).with_columns(((pl.col("pred") - pl.col("obs")).abs() / pl.col("obs")).alias("pct"))
        elec = valid.filter(pl.col("electricity_like"))
        china = valid.filter(pl.col("china_electricity"))
        high = valid.filter(pl.col("high_EID_flag"))
        return {
            "all_node_unweighted_rEI_MAE": _as_float(valid["rEI_abs_error"].mean()),
            "output_weighted_rEI_MAE": self._weighted_mean(valid, "rEI_abs_error", "X_observed"),
            "emissions_weighted_rEI_MAE": self._weighted_mean(valid, "rEI_abs_error", "emissions_observed"),
            "wrong_sign_share": _as_float(valid["wrong_sign"].mean()),
            "emissions_weighted_wrong_sign_share": self._weighted_mean(valid, "wrong_sign", "emissions_observed"),
            "validation_bias": _as_float(valid["rEI_error"].mean()),
            "validation_correlation": _as_float(valid.select(pl.corr("predicted_rEI", "observed_rEI")).item()) if valid.height > 1 else 0.0,
            "latest_year_aggregate_emissions_pct_error": _as_float(yearly.sort("year").tail(1)["pct"].sum()) if not yearly.is_empty() else 0.0,
            "mean_yearly_aggregate_emissions_pct_error": _as_float(yearly["pct"].mean()) if not yearly.is_empty() else 0.0,
            "electricity_rEI_MAE": _as_float(elec["rEI_abs_error"].mean()) if not elec.is_empty() else 0.0,
            "electricity_emissions_weighted_rEI_MAE": self._weighted_mean(elec, "rEI_abs_error", "emissions_observed"),
            "electricity_emissions_error": _as_float(elec["emissions_error"].sum()) if not elec.is_empty() else 0.0,
            "china_electricity_rEI_MAE": _as_float(china["rEI_abs_error"].mean()) if not china.is_empty() else 0.0,
            "china_electricity_emissions_error": _as_float(china["emissions_error"].sum()) if not china.is_empty() else 0.0,
            "high_EID_node_rEI_MAE": _as_float(high["rEI_abs_error"].mean()) if not high.is_empty() else 0.0,
            "high_EID_node_emissions_error": _as_float(high["emissions_error"].sum()) if not high.is_empty() else 0.0,
        }

    def add_objective_score(self, frame: pl.DataFrame, objective: str) -> pl.DataFrame:
        if objective == "transition_accuracy":
            return frame.with_columns(pl.col("cal_all_node_unweighted_rEI_MAE").alias("calibration_score"))
        if objective == "emissions_weighted_transition":
            return frame.with_columns(pl.col("cal_emissions_weighted_rEI_MAE").alias("calibration_score"))
        if objective == "aggregate_emissions_fit":
            return frame.with_columns(pl.col("cal_mean_yearly_aggregate_emissions_pct_error").alias("calibration_score"))
        columns = [
            "cal_all_node_unweighted_rEI_MAE",
            "cal_emissions_weighted_rEI_MAE",
            "cal_wrong_sign_share",
            "cal_mean_yearly_aggregate_emissions_pct_error",
            "cal_electricity_emissions_error",
            "cal_high_EID_node_emissions_error",
        ]
        if objective == "electricity_high_EID_objective":
            columns = [
                "cal_electricity_emissions_weighted_rEI_MAE",
                "cal_china_electricity_emissions_error",
                "cal_high_EID_node_emissions_error",
                "cal_mean_yearly_aggregate_emissions_pct_error",
            ]
        out = frame
        parts = []
        for column in columns:
            if column not in out.columns:
                continue
            min_value = _as_float(out[column].min())
            max_value = _as_float(out[column].max())
            if max_value == min_value:
                parts.append(pl.lit(0.0))
            else:
                parts.append((pl.col(column) - min_value) / (max_value - min_value))
        return out.with_columns((sum(parts) / max(len(parts), 1)).alias("calibration_score"))

    def objective_value(self, metrics: dict[str, float], objective: str) -> float:
        key = {
            "transition_accuracy": "all_node_unweighted_rEI_MAE",
            "emissions_weighted_transition": "emissions_weighted_rEI_MAE",
            "aggregate_emissions_fit": "mean_yearly_aggregate_emissions_pct_error",
        }.get(objective)
        if key:
            return _as_float(metrics[key])
        if objective == "electricity_high_EID_objective":
            values = [
                metrics["electricity_emissions_weighted_rEI_MAE"],
                metrics["china_electricity_emissions_error"],
                metrics["high_EID_node_emissions_error"],
                metrics["mean_yearly_aggregate_emissions_pct_error"],
            ]
        else:
            values = [
                metrics["all_node_unweighted_rEI_MAE"],
                metrics["emissions_weighted_rEI_MAE"],
                metrics["wrong_sign_share"],
                metrics["mean_yearly_aggregate_emissions_pct_error"],
                metrics["electricity_emissions_error"],
                metrics["high_EID_node_emissions_error"],
            ]
        return sum(_as_float(value) for value in values) / len(values)

    def build_model_comparison(self, validation_panel: pl.DataFrame) -> pl.DataFrame:
        rows = []
        valid = validation_panel.filter(pl.col("validation_or_calibration") == "validation")
        for (design, objective), frame in valid.group_by(["design_name", "objective_name"]):
            metrics = self.compute_metrics(frame)
            rows.append({"model_variant": "adaptive_EID", "design_name": design, "objective_name": objective, "rows": frame.height, **metrics, "notes": "walk-forward selected grid parameters"})
        if self.paths.multiyear_EID_diagnostic_comparison_path.exists():
            existing = pl.read_csv(self.paths.multiyear_EID_diagnostic_comparison_path)
            for row in existing.to_dicts():
                rows.append(
                    {
                        "model_variant": row.get("model_variant"),
                        "design_name": "existing_multiyear",
                        "objective_name": "not_applicable",
                        "rows": row.get("rows"),
                        "all_node_unweighted_rEI_MAE": row.get("unweighted_rEI_MAE"),
                        "output_weighted_rEI_MAE": row.get("output_weighted_rEI_MAE"),
                        "emissions_weighted_rEI_MAE": row.get("emissions_weighted_rEI_MAE"),
                        "wrong_sign_share": row.get("wrong_sign_share"),
                        "emissions_weighted_wrong_sign_share": row.get("emissions_weighted_wrong_sign_share"),
                        "validation_bias": row.get("validation_bias"),
                        "validation_correlation": row.get("validation_correlation"),
                        "latest_year_aggregate_emissions_pct_error": row.get("latest_year_aggregate_emissions_pct_error"),
                        "mean_yearly_aggregate_emissions_pct_error": row.get("mean_yearly_aggregate_emissions_pct_error"),
                        "electricity_rEI_MAE": row.get("electricity_rEI_MAE"),
                        "electricity_emissions_weighted_rEI_MAE": row.get("electricity_emissions_weighted_rEI_MAE"),
                        "electricity_emissions_error": row.get("electricity_aggregate_emissions_error"),
                        "china_electricity_rEI_MAE": row.get("china_electricity_rEI_MAE"),
                        "china_electricity_emissions_error": row.get("china_electricity_emissions_error"),
                        "high_EID_node_rEI_MAE": row.get("high_EID_node_rEI_MAE"),
                        "high_EID_node_emissions_error": row.get("high_EID_node_emissions_error"),
                        "notes": "existing Phase 25/baseline comparison",
                    }
                )
        return pl.DataFrame(rows)

    def build_parameter_stability(self, calibration_results: pl.DataFrame) -> pl.DataFrame:
        rows = []
        for (design, objective), frame in calibration_results.group_by(["design_name", "objective_name"]):
            ordered = frame.sort("window_id")
            lambdas = ordered["lambda_EID"].to_list()
            dmins = ordered["d_min"].to_list()
            rows.append(
                {
                    "design_name": design,
                    "objective_name": objective,
                    "selected_lambda_sequence": ",".join(str(v) for v in lambdas),
                    "selected_d_min_sequence": ",".join(str(v) for v in dmins),
                    "lambda_changes_count": self._change_count(lambdas),
                    "d_min_changes_count": self._change_count(dmins),
                    "modal_lambda": self._mode(lambdas),
                    "modal_d_min": self._mode(dmins),
                    "stable_parameter_flag": self._change_count(lambdas) == 0 and self._change_count(dmins) == 0,
                    "interpretation": self._stability_interpretation(lambdas, dmins),
                }
            )
        return pl.DataFrame(rows)

    def build_group_diagnostics(self, panel: pl.DataFrame, group_col: str) -> pl.DataFrame:
        valid = panel.filter(pl.col("validation_or_calibration") == "validation")
        if group_col not in valid.columns:
            return pl.DataFrame()
        out = valid.group_by("design_name", "objective_name", group_col).agg(
            pl.len().alias("rows"),
            pl.col("country_sector").n_unique().alias("nodes"),
            pl.col("emissions_observed").sum().alias("_obs"),
            pl.col("rEI_abs_error").mean().alias("rEI_MAE"),
            (pl.col("rEI_abs_error") * pl.col("emissions_observed")).sum().truediv(pl.col("emissions_observed").sum()).alias("emissions_weighted_rEI_MAE"),
            pl.col("wrong_sign").mean().alias("wrong_sign_share"),
            pl.col("emissions_error").sum().alias("emissions_error"),
        ).rename({group_col: "group"})
        return out.with_columns(
            (pl.col("_obs") / max(_as_float(valid["emissions_observed"].sum()), 1e-12)).alias("observed_emissions_share"),
            pl.lit(None, dtype=pl.Float64).alias("improvement_vs_historical_frontier_gap"),
            pl.lit(None, dtype=pl.Float64).alias("improvement_vs_fixed_EID"),
            pl.lit("adaptive EID group diagnostic").alias("interpretation"),
        ).drop("_obs")

    def build_pseudo_agent_sensitivity(self, panel: pl.DataFrame) -> pl.DataFrame:
        rows = []
        for include in [True, False]:
            frame = panel if include else panel.filter(~pl.col("pseudo_agent_flag"))
            for (design, objective), group in frame.filter(pl.col("validation_or_calibration") == "validation").group_by(["design_name", "objective_name"]):
                metrics = self.compute_metrics(group)
                rows.append({"design_name": design, "objective_name": objective, "pseudo_agent_scope": "including_pseudo_agents" if include else "excluding_pseudo_agents", **metrics})
        return pl.DataFrame(rows)

    def build_hypothesis_tests(self, calibration_results: pl.DataFrame, comparison: pl.DataFrame, stability: pl.DataFrame, subtype: pl.DataFrame) -> pl.DataFrame:
        best_adaptive = self._best_adaptive(comparison)
        hist = self._variant_row(comparison, "historical_frontier_gap_only")
        fixed = self._variant_row(comparison, "historical_frontier_gap_EID_diagnostic")
        cal_improves = calibration_results["calibration_score"].mean() < calibration_results["validation_score"].mean()
        adaptive_improves_hist = self._metric_improves(best_adaptive, hist, "emissions_weighted_rEI_MAE")
        adaptive_improves_fixed = self._metric_improves(best_adaptive, fixed, "emissions_weighted_rEI_MAE")
        variable = stability.filter(~pl.col("stable_parameter_flag")).height > 0
        weak_selected = calibration_results.filter((pl.col("lambda_EID") == 0) | (pl.col("d_min") == 1)).height / max(calibration_results.height, 1)
        subtype_improves = (
            "emissions_weighted_rEI_MAE" in subtype.columns
            and subtype.filter(pl.col("emissions_weighted_rEI_MAE").is_not_null()).height > 0
        )
        rows = [
            self._hypothesis("fixed_EID_too_rigid", "adaptive beats fixed EID", adaptive_improves_fixed, str(best_adaptive), "adaptive EID improves over fixed EID" if adaptive_improves_fixed else "fixed EID failure is not solved by adaptation"),
            self._hypothesis("EID_regime_dependent", "selected parameters vary across windows", variable, f"unstable rows={stability.filter(~pl.col('stable_parameter_flag')).height}", "parameter changes suggest regime dependence" if variable else "selected parameters are stable"),
            self._hypothesis("fixed_EID_too_strong", "weak parameters selected often", weak_selected > 0.5, f"weak/no-effect share={weak_selected:.3g}", "weak/no EID dominates selection" if weak_selected > 0.5 else "stronger EID not clearly rejected by selection frequency"),
            self._hypothesis("subtype_specific_EID", "subtype diagnostics contain improvement candidates", subtype_improves, f"subtype rows={subtype.height}", "subtype slices remain relevant diagnostics"),
            self._hypothesis("EID_diagnostic_only", "adaptive fails historical baseline", not adaptive_improves_hist, str(best_adaptive), "EID remains diagnostic only" if not adaptive_improves_hist else "adaptive EID improves enough to keep testing"),
            self._hypothesis("missing_policy_counterforce", "unstable parameters imply omitted regimes", variable, f"parameter stability rows={stability.height}", "time-varying parameters point to missing policy/energy variables" if variable else "little evidence from parameter instability"),
            self._hypothesis("adaptive_calibration_overfitting", "calibration improves but forward validation fails", cal_improves and not adaptive_improves_hist, f"cal_mean={calibration_results['calibration_score'].mean()}; val_mean={calibration_results['validation_score'].mean()}", "overfitting risk is high" if cal_improves and not adaptive_improves_hist else "no strong overfitting rejection"),
        ]
        return pl.DataFrame(rows)

    def build_recommendation(
        self,
        comparison: pl.DataFrame,
        stability: pl.DataFrame,
        hypotheses: pl.DataFrame,
        subtype: pl.DataFrame,
        calibration_results: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        best = self._best_adaptive(comparison)
        hist = self._variant_row(comparison, "historical_frontier_gap_only")
        fixed = self._variant_row(comparison, "historical_frontier_gap_EID_diagnostic")
        material = self.material_worsening_flags(hist, best)
        improves_hist = self._metric_improves(best, hist, "emissions_weighted_rEI_MAE") and self._metric_improves(best, hist, "mean_yearly_aggregate_emissions_pct_error")
        improves_fixed = self._metric_improves(best, fixed, "emissions_weighted_rEI_MAE")
        stable = bool(stability.filter(pl.col("stable_parameter_flag")).height == stability.height) if not stability.is_empty() else False
        variable = bool(stability.filter(~pl.col("stable_parameter_flag")).height > 0) if not stability.is_empty() else False
        weak_share = 0.0
        if calibration_results is not None and not calibration_results.is_empty():
            weak_share = _as_float(
                calibration_results.filter((pl.col("lambda_EID") == 0) | (pl.col("d_min") == 1)).height
                / max(calibration_results.height, 1)
            )
        if any(material.values()):
            rec = "reject_EID_for_v4_confirmed"
            interp = "Adaptive EID materially worsens at least one key validation threshold."
        elif improves_hist and stable:
            rec = "stable_EID_parameter_found"
            interp = "A stable adaptive EID parameter improves forward validation."
        elif improves_hist and variable:
            rec = "regime_dependent_EID_found"
            interp = "Adaptive EID improves forward validation but selected parameters vary across windows."
        elif improves_fixed and not improves_hist:
            rec = "keep_EID_diagnostic_only"
            interp = "Adaptive EID improves over fixed EID but not over the historical baseline."
        elif weak_share > 0.5:
            rec = "keep_EID_diagnostic_only"
            interp = "Most selections remove or nearly remove EID dampening."
        elif subtype.filter(pl.col("group").is_in(["infrastructure_energy", "heavy_industry_materials"])).height > 0:
            rec = "test_subtype_specific_EID_candidate"
            interp = "Subtype diagnostics remain interesting, but all-node evidence is insufficient."
        else:
            rec = "reject_EID_for_v4_confirmed"
            interp = "Adaptive EID does not improve forward validation."
        return pl.DataFrame([{"recommendation": rec, "evidence": f"best_adaptive={best}; material_worsening={material}; improves_hist={improves_hist}; improves_fixed={improves_fixed}", "interpretation": interp, "recommended_phase27": "Keep scenarios blocked; either confirm rejection or move to validation-objective selection without EID as a rule.", "abm_v5_implication": "EID remains ontology evidence only unless future external policy/energy variables support it.", "scenario_readiness": "premature"}])

    def material_worsening_flags(self, baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, bool]:
        return {
            "all_node_rEI_MAE": self._relative_worse(candidate, baseline, "all_node_unweighted_rEI_MAE", 0.02),
            "emissions_weighted_rEI_MAE": self._relative_worse(candidate, baseline, "emissions_weighted_rEI_MAE", 0.02),
            "wrong_sign_share": (_as_float(candidate.get("wrong_sign_share")) - _as_float(baseline.get("wrong_sign_share"))) > 0.02,
            "aggregate_emissions_pct_error": (_as_float(candidate.get("mean_yearly_aggregate_emissions_pct_error")) - _as_float(baseline.get("mean_yearly_aggregate_emissions_pct_error"))) > 0.02,
        }

    def build_markdown_report(self, comparison: pl.DataFrame, stability: pl.DataFrame, hypotheses: pl.DataFrame, recommendation: pl.DataFrame) -> str:
        return "\n".join([
            "# ABM v4 Phase 26 Adaptive EID Calibration Diagnostics",
            "",
            f"Recommendation: `{recommendation['recommendation'].item(0) if not recommendation.is_empty() else 'inconclusive'}`.",
            "",
            "## Model Comparison",
            self._markdown_table(comparison),
            "",
            "## Parameter Stability",
            self._markdown_table(stability),
            "",
            "## Hypothesis Tests",
            self._markdown_table(hypotheses),
            "",
            "## Recommendation",
            self._markdown_table(recommendation),
            "",
            "Scenarios remain premature.",
        ]) + "\n"

    def _weighted_mean(self, frame: pl.DataFrame, value: str, weight: str) -> float:
        if frame.is_empty() or value not in frame.columns or weight not in frame.columns:
            return 0.0
        denom = _as_float(frame[weight].sum())
        return 0.0 if denom == 0 else _as_float((frame[value] * frame[weight]).sum() / denom)

    def _change_count(self, values: list[Any]) -> int:
        return sum(1 for left, right in zip(values, values[1:]) if left != right)

    def _mode(self, values: list[Any]) -> Any:
        return max(set(values), key=values.count) if values else None

    def _stability_interpretation(self, lambdas: list[Any], dmins: list[Any]) -> str:
        if self._change_count(lambdas) == 0 and self._change_count(dmins) == 0:
            return "stable selected EID parameter"
        if any(value == 0 for value in lambdas) or any(value == 1 for value in dmins):
            return "selection often weakens or removes EID dampening"
        return "selected EID parameter varies across windows"

    def _hypothesis(self, name: str, test: str, passed: bool, evidence: str, interpretation: str) -> dict[str, Any]:
        return {"hypothesis": name, "test": test, "result": passed, "evidence": evidence, "interpretation": interpretation, "status": "supported" if passed else "not_supported"}

    def _best_adaptive(self, comparison: pl.DataFrame) -> dict[str, Any]:
        rows = comparison.filter(pl.col("model_variant") == "adaptive_EID")
        return rows.sort("emissions_weighted_rEI_MAE").to_dicts()[0] if not rows.is_empty() else {}

    def _variant_row(self, comparison: pl.DataFrame, variant: str) -> dict[str, Any]:
        rows = comparison.filter(pl.col("model_variant") == variant)
        return rows.to_dicts()[0] if not rows.is_empty() else {}

    def _metric_improves(self, candidate: dict[str, Any], baseline: dict[str, Any], metric: str) -> bool:
        return _as_float(candidate.get(metric), 999.0) < _as_float(baseline.get(metric), 0.0)

    def _relative_worse(self, candidate: dict[str, Any], baseline: dict[str, Any], metric: str, threshold: float) -> bool:
        base = _as_float(baseline.get(metric))
        cand = _as_float(candidate.get(metric))
        return base != 0 and (cand - base) / abs(base) > threshold

    def _markdown_table(self, frame: pl.DataFrame) -> str:
        if frame.is_empty():
            return "_No rows available._"
        lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join("---" for _ in frame.columns) + " |"]
        for row in frame.to_dicts():
            lines.append("| " + " | ".join(self._format_markdown_value(row.get(column)) for column in frame.columns) + " |")
        return "\n".join(lines)

    def _format_markdown_value(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        if value is None:
            return ""
        return str(value).replace("|", "/")


class QEnergyMixAudit:
    """Audit Eora Q energy-use rows as country-sector energy-mix diagnostics."""

    energy_components: tuple[dict[str, Any], ...] = (
        {"canonical": "natural_gas_TJ", "item": "Natural Gas", "patterns": ("natural gas",)},
        {"canonical": "coal_TJ", "item": "Coal", "patterns": ("coal",)},
        {"canonical": "petroleum_TJ", "item": "Petroleum", "patterns": ("petroleum",)},
        {"canonical": "nuclear_electricity_TJ", "item": "Nuclear Electricity", "patterns": ("nuclear electricity", "nuclear")},
        {"canonical": "hydroelectric_electricity_TJ", "item": "Hydroelectric Electricity", "patterns": ("hydroelectric electricity", "hydro")},
        {"canonical": "geothermal_electricity_TJ", "item": "Geothermal Electricity", "patterns": ("geothermal electricity", "geothermal")},
        {"canonical": "wind_electricity_TJ", "item": "Wind Electricity", "patterns": ("wind electricity", "wind")},
        {"canonical": "solar_tide_wave_electricity_TJ", "item": "Solar, Tide and Wave Electricity", "patterns": ("solar, tide and wave electricity", "solar tide wave", "solar")},
        {"canonical": "biomass_waste_electricity_TJ", "item": "Biomass and Waste Electricity", "patterns": ("biomass and waste electricity", "biomass", "waste electricity")},
    )
    energy_columns = tuple(component["canonical"] for component in energy_components)
    share_columns = (
        "coal_share",
        "gas_share",
        "petroleum_share",
        "fossil_share",
        "nuclear_share",
        "hydro_share",
        "wind_share",
        "solar_tide_wave_share",
        "biomass_waste_share",
        "renewable_electricity_share",
        "clean_electricity_share",
    )
    predictor_columns = (
        "coal_share",
        "gas_share",
        "petroleum_share",
        "fossil_share",
        "clean_electricity_share",
        "renewable_electricity_share",
        "energy_mix_hhi",
        "energy_mix_entropy",
        "fossil_to_clean_ratio",
        "coal_to_clean_ratio",
        "total_tracked_energy_per_output",
        "fossil_energy_per_output",
        "coal_energy_per_output",
        "clean_electricity_per_output",
    )

    def __init__(self, paths: ABMV4Paths, start_year: int = 1995, end_year: int = 2016) -> None:
        self.paths = paths
        self.start_year = start_year
        self.end_year = end_year

    def run(self) -> QEnergyMixAuditResult:
        inventory = self.discover_q_sources()
        row_mapping = self.build_row_mapping(inventory)
        panel = self.build_energy_mix_panel(row_mapping)
        quality = self.build_quality_audit(panel)
        quality_by_year = self.build_group_quality(panel, "year")
        quality_by_sector = self.build_group_quality(panel, "Sector")
        quality_by_country = self.build_group_quality(panel, "Country")
        plausibility = self.build_aggregate_plausibility(panel)
        transition_panel = self.build_transition_error_panel(panel)
        china = self.build_china_electricity_audit(panel, transition_panel)
        screening = self.build_predictor_screening(transition_panel)
        by_subtype = self.build_subtype_diagnostics(transition_panel)
        hypotheses = self.build_hypothesis_tests(quality, plausibility, china, screening, by_subtype)
        recommendation = self.build_recommendation(hypotheses, quality, screening)
        markdown = self.build_markdown_report(inventory, row_mapping, quality, china, screening, hypotheses, recommendation)
        return QEnergyMixAuditResult(
            source_inventory=inventory,
            row_mapping=row_mapping,
            energy_mix_panel=panel,
            quality_audit=quality,
            quality_by_year=quality_by_year,
            quality_by_sector=quality_by_sector,
            quality_by_country=quality_by_country,
            aggregate_plausibility=plausibility,
            china_electricity_audit=china,
            transition_error_panel=transition_panel,
            predictor_screening=screening,
            by_subtype=by_subtype,
            hypothesis_tests=hypotheses,
            recommendation=recommendation,
            markdown=markdown,
        )

    def write_outputs(self, result: QEnergyMixAuditResult) -> None:
        self.paths.validation.mkdir(parents=True, exist_ok=True)
        result.source_inventory.write_csv(self.paths.q_energy_source_inventory_path)
        result.row_mapping.write_csv(self.paths.q_energy_row_mapping_path)
        result.energy_mix_panel.write_parquet(self.paths.q_energy_mix_panel_path)
        result.quality_audit.write_csv(self.paths.q_energy_mix_quality_audit_path)
        result.quality_by_year.write_csv(self.paths.q_energy_mix_quality_by_year_path)
        result.quality_by_sector.write_csv(self.paths.q_energy_mix_quality_by_sector_path)
        result.quality_by_country.write_csv(self.paths.q_energy_mix_quality_by_country_path)
        result.aggregate_plausibility.write_csv(self.paths.q_energy_mix_aggregate_plausibility_path)
        result.china_electricity_audit.write_csv(self.paths.q_energy_mix_china_electricity_audit_path)
        result.transition_error_panel.write_parquet(self.paths.q_energy_mix_transition_error_panel_path)
        result.predictor_screening.write_csv(self.paths.q_energy_mix_predictor_screening_path)
        result.by_subtype.write_csv(self.paths.q_energy_mix_by_subtype_path)
        result.hypothesis_tests.write_csv(self.paths.q_energy_mix_hypothesis_tests_path)
        result.recommendation.write_csv(self.paths.q_energy_mix_recommendation_path)
        self.paths.q_energy_mix_report_path.write_text(result.markdown, encoding="utf-8")

    def discover_q_sources(self) -> pl.DataFrame:
        rows: list[dict[str, Any]] = []
        for year in range(self.start_year, self.end_year + 1):
            q_path = self.paths.data_root / "parquet" / str(year) / "Q.parquet"
            labels_path = self.paths.data_root / "raw" / str(year) / "labels_Q.txt"
            exists = q_path.exists()
            label_matches = self._read_energy_label_matches(labels_path)
            columns: list[str] = []
            row_count: int | None = None
            notes = ""
            if exists:
                try:
                    schema = pl.scan_parquet(q_path).collect_schema()
                    columns = list(schema.names())
                    row_count = pl.scan_parquet(q_path).select(pl.len()).collect().item()
                except Exception as exc:  # pragma: no cover - defensive for malformed local files
                    notes = f"schema inspection failed: {exc}"
            rows.append(
                {
                    "source_path": str(q_path),
                    "exists": exists,
                    "file_type": "parquet",
                    "years_detected": str(year) if exists else "",
                    "rows_if_known": row_count,
                    "columns": ";".join(columns[:25]),
                    "possible_Q_structure": "stressor_rows_by_country_sector_columns" if exists and columns else "missing",
                    "candidate_energy_rows_detected": len(label_matches),
                    "usable_for_country_sector_energy_mix": bool(exists and len(columns) > 0 and len(label_matches) >= 9),
                    "notes": notes or (f"labels={labels_path}" if labels_path.exists() else "labels_Q missing"),
                }
            )
        for labels_path in sorted((self.paths.data_root / "raw").glob("*/labels_Q.txt")):
            if not labels_path.parent.name.isdigit() or self.start_year <= int(labels_path.parent.name) <= self.end_year:
                continue
            rows.append(
                {
                    "source_path": str(labels_path),
                    "exists": True,
                    "file_type": "labels_Q.txt",
                    "years_detected": labels_path.parent.name,
                    "rows_if_known": None,
                    "columns": "family;item",
                    "possible_Q_structure": "labels_only",
                    "candidate_energy_rows_detected": len(self._read_energy_label_matches(labels_path)),
                    "usable_for_country_sector_energy_mix": False,
                    "notes": "label file outside requested audit window",
                }
            )
        return pl.DataFrame(rows)

    def build_row_mapping(self, inventory: pl.DataFrame | None = None) -> pl.DataFrame:
        source_paths = []
        if inventory is not None and "source_path" in inventory.columns:
            source_paths = [Path(path) for path in inventory["source_path"].to_list() if str(path).endswith("Q.parquet")]
        if not source_paths:
            source_paths = [self.paths.data_root / "parquet" / str(year) / "Q.parquet" for year in range(self.start_year, self.end_year + 1)]
        index_path = self.paths.data_root / "indices" / "index_q.csv"
        index_rows = self._read_q_index(index_path)
        rows = []
        for component in self.energy_components:
            match = self._match_component_in_index(component, index_rows)
            for source_path in source_paths:
                rows.append(
                    {
                        "canonical_energy_component": component["canonical"],
                        "matched_row_label": match.get("row_label", ""),
                        "matched_item_label": match.get("item", ""),
                        "matched_family_label": match.get("family", ""),
                        "source_path": str(source_path),
                        "match_status": "matched" if match else "missing",
                        "match_confidence": 1.0 if match else 0.0,
                        "row_index_zero_based": match.get("row_index_zero_based") if match else None,
                        "notes": "matched from data/indices/index_q.csv" if match else "no flexible label match found",
                    }
                )
        return pl.DataFrame(rows)

    def build_energy_mix_panel(self, row_mapping: pl.DataFrame | None = None) -> pl.DataFrame:
        mappings = row_mapping if row_mapping is not None else self.build_row_mapping()
        usable_paths = sorted({Path(path) for path in mappings["source_path"].to_list() if Path(path).exists() and str(path).endswith("Q.parquet")})
        if not usable_paths:
            raise FileNotFoundError(
                "No usable Eora Q parquet sources found. Expected files like data/parquet/1995/Q.parquet; "
                "run the parquet conversion pipeline or inspect Q source paths first."
            )
        state = self._load_state_metadata()
        frames = []
        for q_path in usable_paths:
            year = self._year_from_path(q_path)
            if year is None or year < self.start_year or year > self.end_year:
                continue
            frame = self._read_year_energy_frame(q_path, mappings)
            if not frame.is_empty():
                frames.append(frame.with_columns(pl.lit(year).alias("year")))
        if not frames:
            raise FileNotFoundError("Q sources were present but none contained matched energy rows.")
        panel = pl.concat(frames, how="diagonal_relaxed")
        panel = panel.join(state, on=["country_sector", "year"], how="left")
        panel = self._add_energy_mix_variables(panel)
        panel = self._add_subtype_metadata(panel)
        return panel

    def build_quality_audit(self, panel: pl.DataFrame) -> pl.DataFrame:
        expected = self._expected_node_years()
        rows = []
        for component in list(self.energy_columns) + [
            "fossil_energy_TJ",
            "clean_electricity_TJ",
            "renewable_electricity_TJ",
            "total_tracked_energy_TJ",
        ] + list(self.share_columns):
            if component not in panel.columns:
                continue
            values = panel[component]
            rows.append(
                {
                    "component": component,
                    "rows": panel.height,
                    "node_years_expected": expected,
                    "node_years_observed": values.drop_nulls().len(),
                    "coverage_share": values.drop_nulls().len() / max(expected, 1),
                    "missing_share": values.null_count() / max(panel.height, 1),
                    "zero_share": _as_float((values.fill_null(0) == 0).mean()),
                    "negative_count": _as_int((values.drop_nulls() < 0).sum()),
                    "nonfinite_count": self._nonfinite_count(values),
                    "p01": self._quantile(values, 0.01),
                    "p50": self._quantile(values, 0.50),
                    "p99": self._quantile(values, 0.99),
                    "max": _as_float(values.max()),
                    "jump_flag_count": self._jump_count(panel, component),
                    "jump_flag_share": self._jump_count(panel, component) / max(panel.height, 1),
                    "notes": "share variable" if component in self.share_columns else "TJ or intensity variable",
                }
            )
        invalid_share = self._invalid_share_count(panel)
        rows.append(
            {
                "component": "_share_validity",
                "rows": panel.height,
                "node_years_expected": expected,
                "node_years_observed": panel.height,
                "coverage_share": panel.height / max(expected, 1),
                "missing_share": 0.0,
                "zero_share": 0.0,
                "negative_count": 0,
                "nonfinite_count": invalid_share,
                "p01": None,
                "p50": None,
                "p99": None,
                "max": None,
                "jump_flag_count": 0,
                "jump_flag_share": 0.0,
                "notes": "nonfinite_count stores invalid share rows outside [0, 1] tolerance",
            }
        )
        return pl.DataFrame(rows)

    def build_group_quality(self, panel: pl.DataFrame, group_col: str) -> pl.DataFrame:
        if group_col not in panel.columns:
            return pl.DataFrame()
        return panel.group_by(group_col).agg(
            pl.len().alias("rows"),
            pl.col("country_sector").n_unique().alias("nodes"),
            pl.col("total_tracked_energy_TJ").is_not_null().mean().alias("coverage_share"),
            (pl.col("total_tracked_energy_TJ") < 0).sum().alias("negative_count"),
            pl.col("total_tracked_energy_TJ").sum().alias("total_tracked_energy_TJ"),
            pl.col("fossil_share").mean().alias("mean_fossil_share"),
            pl.col("coal_share").mean().alias("mean_coal_share"),
            pl.col("clean_electricity_share").mean().alias("mean_clean_electricity_share"),
        ).rename({group_col: "group"}).with_columns(
            pl.lit("Q energy mix group quality summary").alias("notes")
        )

    def build_aggregate_plausibility(self, panel: pl.DataFrame) -> pl.DataFrame:
        rows = []
        rows.extend(self._aggregate_plausibility_rows(panel, "global-year", None))
        rows.extend(self._aggregate_plausibility_rows(panel, "country-year", "Country"))
        rows.extend(self._aggregate_plausibility_rows(panel, "sector-year", "Sector"))
        electricity = panel.filter(pl.col("electricity_like"))
        rows.extend(self._aggregate_plausibility_rows(electricity, "electricity-like country-year", "Country"))
        return pl.DataFrame(rows) if rows else pl.DataFrame()

    def build_china_electricity_audit(self, panel: pl.DataFrame, transition_panel: pl.DataFrame | None = None) -> pl.DataFrame:
        china = panel.filter(self._china_electricity_expr())
        if transition_panel is not None and not transition_panel.is_empty():
            error_cols = [
                col for col in [
                    "model_error_frontier_gap_readiness",
                    "model_error_historical_frontier_gap_only",
                    "model_error_EID",
                ] if col in transition_panel.columns
            ]
            if error_cols:
                china = china.join(
                    transition_panel.select(["country_sector", "year", *error_cols]).unique(["country_sector", "year"]),
                    on=["country_sector", "year"],
                    how="left",
                )
        for col in ["model_error_frontier_gap_readiness", "model_error_historical_frontier_gap_only", "model_error_EID"]:
            if col not in china.columns:
                china = china.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
        return china.select(
            "country_sector",
            "year",
            "coal_TJ",
            "natural_gas_TJ",
            "petroleum_TJ",
            "fossil_energy_TJ",
            "clean_electricity_TJ",
            "renewable_electricity_TJ",
            "total_tracked_energy_TJ",
            "coal_share",
            "fossil_share",
            "clean_electricity_share",
            "renewable_electricity_share",
            "energy_mix_hhi",
            "energy_mix_entropy",
            "X_observed",
            "EI_observed",
            "observed_rEI",
            "model_error_frontier_gap_readiness",
            "model_error_historical_frontier_gap_only",
            "model_error_EID",
            "notes",
        ) if not china.is_empty() else pl.DataFrame()

    def build_transition_error_panel(self, panel: pl.DataFrame) -> pl.DataFrame:
        if not self.paths.transition_rule_sign_failure_panel_path.exists():
            return panel.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("rEI_error_frontier_gap_readiness"),
                pl.lit(None, dtype=pl.Float64).alias("rEI_error_historical_frontier_gap_only"),
                pl.lit("transition_rule_sign_failure_panel missing").alias("notes_transition"),
            )
        errors = pl.read_parquet(self.paths.transition_rule_sign_failure_panel_path)
        keep = [
            col for col in [
                "country_sector",
                "year",
                "observed_rEI",
                "simulated_rEI_readiness",
                "simulated_rEI_frontier_gap",
                "rEI_error_readiness",
                "rEI_error_frontier_gap",
                "rEI_abs_error_readiness",
                "rEI_abs_error_frontier_gap",
                "emissions_error_readiness",
                "emissions_error_frontier_gap",
                "emissions_decile",
            ] if col in errors.columns
        ]
        out = panel.join(errors.select(keep), on=["country_sector", "year"], how="left", suffix="_error")
        if "observed_rEI_error" in out.columns and "observed_rEI" in out.columns:
            out = out.with_columns(pl.coalesce(["observed_rEI_error", "observed_rEI"]).alias("observed_rEI")).drop("observed_rEI_error")
        rename_map = {
            "simulated_rEI_readiness": "predicted_rEI_frontier_gap_readiness",
            "simulated_rEI_frontier_gap": "predicted_rEI_historical_frontier_gap_only",
            "rEI_error_readiness": "rEI_error_frontier_gap_readiness",
            "rEI_error_frontier_gap": "rEI_error_historical_frontier_gap_only",
            "emissions_error_readiness": "emissions_error_frontier_gap_readiness",
            "emissions_error_frontier_gap": "emissions_error_historical_frontier_gap_only",
        }
        out = out.rename({old: new for old, new in rename_map.items() if old in out.columns})
        for col in rename_map.values():
            if col not in out.columns:
                out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
        if "rEI_abs_error_readiness" in out.columns and "rEI_abs_error_frontier_gap" in out.columns:
            out = out.with_columns((pl.col("rEI_abs_error_frontier_gap") - pl.col("rEI_abs_error_readiness")).alias("error_difference_hist_minus_readiness"))
        else:
            out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("error_difference_hist_minus_readiness"))
        if "emissions_decile" in out.columns:
            out = out.with_columns(
                pl.col("emissions_decile").cast(pl.Utf8).str.to_lowercase().str.contains("9|10|d9|d10").fill_null(False).alias("high_emissions_node")
            )
        else:
            out = out.with_columns(pl.lit(False).alias("high_emissions_node"))
        return out

    def build_predictor_screening(self, transition_panel: pl.DataFrame) -> pl.DataFrame:
        targets = [
            "rEI_abs_error_frontier_gap",
            "rEI_abs_error_readiness",
            "emissions_error_historical_frontier_gap_only",
            "emissions_error_frontier_gap_readiness",
            "error_difference_hist_minus_readiness",
        ]
        rows = []
        for target in targets:
            if target not in transition_panel.columns:
                continue
            for predictor in self.predictor_columns:
                if predictor not in transition_panel.columns:
                    continue
                frame = transition_panel.select(predictor, target).drop_nulls()
                if frame.height < 5:
                    continue
                try:
                    corr = _as_float(frame.select(pl.corr(predictor, target)).item())
                except Exception:
                    corr = 0.0
                with_decile = frame.with_columns(
                    pl.col(predictor).rank("average").alias("_rank")
                ).with_columns(
                    (((pl.col("_rank") - 1) / max(frame.height, 1) * 10).floor() + 1).clip(1, 10).alias("_decile")
                )
                top = _as_float(with_decile.filter(pl.col("_decile") == 10)[target].mean())
                bottom = _as_float(with_decile.filter(pl.col("_decile") == 1)[target].mean())
                rows.append(
                    {
                        "target": target,
                        "predictor": predictor,
                        "rows": frame.height,
                        "correlation": corr,
                        "abs_correlation": abs(corr),
                        "top_decile_target_mean": top,
                        "bottom_decile_target_mean": bottom,
                        "decile_gap": top - bottom,
                        "interpretation": "strong univariate relationship" if abs(corr) >= 0.2 or abs(top - bottom) > abs(bottom) * 0.25 else "weak or local relationship",
                        "recommended_for_phase28": abs(corr) >= 0.2 or abs(top - bottom) > abs(bottom) * 0.25,
                    }
                )
        return pl.DataFrame(rows).sort(["target", "abs_correlation"], descending=[False, True]) if rows else pl.DataFrame()

    def build_subtype_diagnostics(self, transition_panel: pl.DataFrame) -> pl.DataFrame:
        if "candidate_subtype" not in transition_panel.columns:
            return pl.DataFrame()
        total_rows = max(transition_panel.height, 1)
        return transition_panel.group_by("candidate_subtype").agg(
            pl.len().alias("rows"),
            pl.col("country_sector").n_unique().alias("nodes"),
            (pl.col("total_tracked_energy_TJ").is_not_null().sum() / total_rows).alias("coverage_share"),
            pl.col("fossil_share").mean().alias("mean_fossil_share"),
            pl.col("fossil_share").median().alias("median_fossil_share"),
            pl.col("coal_share").mean().alias("mean_coal_share"),
            pl.col("coal_share").median().alias("median_coal_share"),
            pl.col("clean_electricity_share").mean().alias("mean_clean_electricity_share"),
            pl.col("total_tracked_energy_per_output").mean().alias("mean_total_energy_per_output"),
            pl.col("fossil_energy_per_output").mean().alias("mean_fossil_energy_per_output"),
            pl.col("rEI_error_historical_frontier_gap_only").abs().mean().alias("mean_error_historical_frontier_gap_only"),
            pl.col("rEI_error_frontier_gap_readiness").abs().mean().alias("mean_error_frontier_gap_readiness"),
        ).with_columns(
            pl.when(pl.col("mean_fossil_share") > 0.5)
            .then(pl.lit("fossil-heavy subtype"))
            .otherwise(pl.lit("mixed or low tracked fossil subtype"))
            .alias("energy_mix_error_relationship"),
            pl.lit("diagnostic subtype summary; not a transition rule").alias("interpretation"),
        )

    def build_hypothesis_tests(
        self,
        quality: pl.DataFrame,
        plausibility: pl.DataFrame,
        china: pl.DataFrame,
        screening: pl.DataFrame,
        by_subtype: pl.DataFrame,
    ) -> pl.DataFrame:
        coverage = self._quality_value(quality, "total_tracked_energy_TJ", "coverage_share")
        negatives = self._quality_value(quality, "total_tracked_energy_TJ", "negative_count")
        invalid_shares = self._quality_value(quality, "_share_validity", "nonfinite_count")
        severe_flags = plausibility.filter(pl.col("plausibility_flag") != "ok").height if not plausibility.is_empty() and "plausibility_flag" in plausibility.columns else 0
        best_abs_corr = _as_float(screening["abs_correlation"].max()) if not screening.is_empty() and "abs_correlation" in screening.columns else 0.0
        recommended_predictors = screening.filter(pl.col("recommended_for_phase28")).height if not screening.is_empty() and "recommended_for_phase28" in screening.columns else 0
        china_high_fossil = _as_float(china["fossil_share"].mean()) > 0.5 if not china.is_empty() and "fossil_share" in china.columns else False
        physical = by_subtype.filter(pl.col("candidate_subtype").is_in(["infrastructure_energy", "heavy_industry_materials", "transport_logistics_infrastructure"])) if not by_subtype.is_empty() else pl.DataFrame()
        service = by_subtype.filter(pl.col("candidate_subtype").str.contains("service|finance|knowledge")) if not by_subtype.is_empty() and "candidate_subtype" in by_subtype.columns else pl.DataFrame()
        physical_relationship = _as_float(physical["mean_fossil_share"].mean()) > _as_float(service["mean_fossil_share"].mean()) if not physical.is_empty() and not service.is_empty() else not physical.is_empty()
        usable = coverage >= 0.8 and negatives == 0 and invalid_shares == 0 and severe_flags < 10
        sparse_or_noisy = coverage < 0.5 or negatives > 0 or invalid_shares > 0 or severe_flags >= 50
        rows = [
            self._hypothesis("H1_energy_mix_data_usable", "coverage, invalid values, and aggregate plausibility", usable, f"coverage={coverage:.3g}; negatives={negatives}; invalid_share_rows={invalid_shares}; severe_flags={severe_flags}", "Q energy mix is usable at country-sector-year level" if usable else "Q energy mix has quality caveats"),
            self._hypothesis("H2_energy_mix_explains_errors_better_than_EID", "univariate energy predictors relate to error targets", best_abs_corr >= 0.2, f"best_abs_correlation={best_abs_corr:.3g}; recommended_predictors={recommended_predictors}", "energy mix has stronger direct error signal than prior EID-only diagnostics" if best_abs_corr >= 0.2 else "energy mix signal is weak"),
            self._hypothesis("H3_china_electricity_fuel_mix_mechanism", "China electricity fossil dependence is high", china_high_fossil, f"mean_fossil_share={_as_float(china['fossil_share'].mean()) if not china.is_empty() else 0:.3g}", "China electricity has fuel-mix evidence consistent with missing mechanism" if china_high_fossil else "China electricity fuel mix does not clearly explain error"),
            self._hypothesis("H4_energy_mix_useful_for_physical_subtypes_only", "physical subtypes show stronger fossil mix than services", physical_relationship, f"physical_rows={physical.height}; service_rows={service.height}", "energy mix appears most relevant for physical/high-emissions subtypes" if physical_relationship else "energy mix is not subtype-specific"),
            self._hypothesis("H5_energy_mix_too_sparse_or_noisy", "poor coverage or severe invalidity", sparse_or_noisy, f"coverage={coverage:.3g}; severe_flags={severe_flags}", "Q energy mix is too sparse/noisy for model use" if sparse_or_noisy else "quality does not force rejection"),
            self._hypothesis("H6_energy_mix_may_resolve_frontier_tradeoff", "predicts historical-vs-readiness error difference", screening.filter((pl.col("target") == "error_difference_hist_minus_readiness") & (pl.col("recommended_for_phase28"))).height > 0 if not screening.is_empty() else False, "screened predictors for error_difference_hist_minus_readiness", "energy variables may explain where frontier-gap-only loses to readiness"),
        ]
        return pl.DataFrame(rows)

    def build_recommendation(self, hypotheses: pl.DataFrame, quality: pl.DataFrame, screening: pl.DataFrame) -> pl.DataFrame:
        supported = set(hypotheses.filter(pl.col("result"))["hypothesis"].to_list()) if not hypotheses.is_empty() else set()
        coverage = self._quality_value(quality, "total_tracked_energy_TJ", "coverage_share")
        has_prediction = not screening.is_empty() and screening.filter(pl.col("recommended_for_phase28")).height > 0
        if "H1_energy_mix_data_usable" in supported and has_prediction and "H6_energy_mix_may_resolve_frontier_tradeoff" in supported:
            rec = "test_energy_mix_augmented_transition_candidate"
            interp = "Q energy mix is usable and has transition-error signal."
            phase = "Test a diagnostic energy-mix augmented transition candidate without making it default."
        elif coverage >= 0.8 and not has_prediction:
            rec = "use_energy_mix_as_validation_stratifier_only"
            interp = "Q energy mix is usable but not predictive enough for a rule."
            phase = "Use energy mix to stratify final validation and close ABM v4 rule search."
        elif coverage >= 0.5:
            rec = "aggregate_only_energy_mix_usable"
            interp = "Q energy mix has partial country-sector quality; aggregate diagnostics are safer."
            phase = "Inspect mapping and aggregate plausibility before model use."
        elif "H5_energy_mix_too_sparse_or_noisy" in supported:
            rec = "reserve_energy_mix_for_abm_v5"
            interp = "Q energy mix is too sparse or noisy for ABM v4 country-sector rules."
            phase = "Reserve fuel-mix mechanisms for an ABM v5 data-design pass."
        else:
            rec = "inconclusive"
            interp = "Energy-mix evidence is mixed."
            phase = "Inspect Q mapping before any model candidate."
        return pl.DataFrame(
            [
                {
                    "recommendation": rec,
                    "evidence": f"coverage={coverage:.3g}; supported={sorted(supported)}; recommended_predictors={screening.filter(pl.col('recommended_for_phase28')).height if not screening.is_empty() else 0}",
                    "interpretation": interp,
                    "recommended_phase28": phase,
                    "abm_v5_implication": "Fuel and energy-use mix should remain a candidate ABM v5 mechanism if ABM v4 diagnostics are inconclusive.",
                    "scenario_readiness": "premature",
                }
            ]
        )

    def build_markdown_report(
        self,
        inventory: pl.DataFrame,
        row_mapping: pl.DataFrame,
        quality: pl.DataFrame,
        china: pl.DataFrame,
        screening: pl.DataFrame,
        hypotheses: pl.DataFrame,
        recommendation: pl.DataFrame,
    ) -> str:
        return "\n".join(
            [
                "# ABM v4 Phase 27 Q Energy-Mix Audit",
                "",
                f"Recommendation: `{recommendation['recommendation'].item(0) if not recommendation.is_empty() else 'inconclusive'}`.",
                "",
                "This is a validation-only diagnostic. It does not implement scenarios, ABM v5 agent types, or an active energy-mix transition rule.",
                "",
                "## Source Inventory",
                self._markdown_table(inventory.head(20)),
                "",
                "## Row Mapping",
                self._markdown_table(row_mapping.select([col for col in row_mapping.columns if col != 'source_path']).unique().head(20)),
                "",
                "## Quality Audit",
                self._markdown_table(quality.head(25)),
                "",
                "## China Electricity",
                self._markdown_table(china.head(25)) if not china.is_empty() else "_No China electricity rows were identified._",
                "",
                "## Predictor Screening",
                self._markdown_table(screening.head(25)) if not screening.is_empty() else "_No predictor screening rows were available._",
                "",
                "## Hypotheses",
                self._markdown_table(hypotheses),
                "",
                "## Recommendation",
                self._markdown_table(recommendation),
                "",
                "Scenarios remain premature.",
            ]
        ) + "\n"

    def _read_year_energy_frame(self, q_path: Path, mappings: pl.DataFrame) -> pl.DataFrame:
        year_mappings = mappings.filter((pl.col("source_path") == str(q_path)) & (pl.col("match_status") == "matched"))
        if year_mappings.is_empty():
            return pl.DataFrame()
        max_row = _as_int(year_mappings["row_index_zero_based"].max()) + 1
        q = pl.read_parquet(q_path, n_rows=max(max_row, 9))
        data: dict[str, Any] = {"country_sector": q.columns}
        for row in year_mappings.to_dicts():
            idx = _as_int(row["row_index_zero_based"])
            if idx >= q.height:
                continue
            data[row["canonical_energy_component"]] = [self._clean_numeric(value) for value in q.row(idx)]
        for column in self.energy_columns:
            data.setdefault(column, [None] * len(q.columns))
        return pl.DataFrame(data)

    def _add_energy_mix_variables(self, panel: pl.DataFrame) -> pl.DataFrame:
        out = panel.with_columns(
            sum(pl.col(col).fill_null(0.0) for col in ["natural_gas_TJ", "coal_TJ", "petroleum_TJ"]).alias("fossil_energy_TJ"),
            sum(pl.col(col).fill_null(0.0) for col in [
                "nuclear_electricity_TJ",
                "hydroelectric_electricity_TJ",
                "geothermal_electricity_TJ",
                "wind_electricity_TJ",
                "solar_tide_wave_electricity_TJ",
                "biomass_waste_electricity_TJ",
            ]).alias("clean_electricity_TJ"),
            sum(pl.col(col).fill_null(0.0) for col in [
                "hydroelectric_electricity_TJ",
                "geothermal_electricity_TJ",
                "wind_electricity_TJ",
                "solar_tide_wave_electricity_TJ",
                "biomass_waste_electricity_TJ",
            ]).alias("renewable_electricity_TJ"),
        ).with_columns(
            (pl.col("fossil_energy_TJ") + pl.col("clean_electricity_TJ")).alias("total_tracked_energy_TJ")
        )
        share_specs = {
            "coal_share": "coal_TJ",
            "gas_share": "natural_gas_TJ",
            "petroleum_share": "petroleum_TJ",
            "fossil_share": "fossil_energy_TJ",
            "nuclear_share": "nuclear_electricity_TJ",
            "hydro_share": "hydroelectric_electricity_TJ",
            "wind_share": "wind_electricity_TJ",
            "solar_tide_wave_share": "solar_tide_wave_electricity_TJ",
            "biomass_waste_share": "biomass_waste_electricity_TJ",
            "renewable_electricity_share": "renewable_electricity_TJ",
            "clean_electricity_share": "clean_electricity_TJ",
        }
        out = out.with_columns(
            [
                pl.when(pl.col("total_tracked_energy_TJ") > 0)
                .then(pl.col(component) / pl.col("total_tracked_energy_TJ"))
                .otherwise(None)
                .alias(share)
                for share, component in share_specs.items()
            ]
        )
        component_shares = [
            "gas_share",
            "coal_share",
            "petroleum_share",
            "nuclear_share",
            "hydro_share",
            "wind_share",
            "solar_tide_wave_share",
            "biomass_waste_share",
        ]
        out = out.with_columns(
            sum(pl.col(col).fill_null(0.0) ** 2 for col in component_shares).alias("energy_mix_hhi"),
            sum(
                pl.when(pl.col(col) > 0).then(-pl.col(col) * pl.col(col).log()).otherwise(0.0)
                for col in component_shares
            ).alias("energy_mix_entropy"),
            (pl.col("fossil_energy_TJ") / (pl.col("clean_electricity_TJ") + 1e-12)).alias("fossil_to_clean_ratio"),
            (pl.col("coal_TJ") / (pl.col("clean_electricity_TJ") + 1e-12)).alias("coal_to_clean_ratio"),
            pl.when(pl.col("X_observed") > 0).then(pl.col("total_tracked_energy_TJ") / pl.col("X_observed")).otherwise(None).alias("total_tracked_energy_per_output"),
            pl.when(pl.col("X_observed") > 0).then(pl.col("fossil_energy_TJ") / pl.col("X_observed")).otherwise(None).alias("fossil_energy_per_output"),
            pl.when(pl.col("X_observed") > 0).then(pl.col("coal_TJ") / pl.col("X_observed")).otherwise(None).alias("coal_energy_per_output"),
            pl.when(pl.col("X_observed") > 0).then(pl.col("clean_electricity_TJ") / pl.col("X_observed")).otherwise(None).alias("clean_electricity_per_output"),
        )
        return out.with_columns(
            self._electricity_like_expr().alias("electricity_like"),
            self._share_quality_expr().alias("mapping_quality_flag"),
            pl.lit("Q energy-use mix; not interpreted as generation mix").alias("notes"),
        )

    def _add_subtype_metadata(self, panel: pl.DataFrame) -> pl.DataFrame:
        out = panel
        if self.paths.eid_high_node_heterogeneity_panel_path.exists():
            meta = pl.read_csv(self.paths.eid_high_node_heterogeneity_panel_path).unique("country_sector")
            cols = [col for col in ["country_sector", "high_EID_flag", "candidate_subtype", "pseudo_agent_flag"] if col in meta.columns]
            out = out.join(meta.select(cols), on="country_sector", how="left")
        for col, default in [("high_EID_flag", False), ("candidate_subtype", "ordinary_or_unclear"), ("pseudo_agent_flag", False)]:
            if col not in out.columns:
                out = out.with_columns(pl.lit(default).alias(col))
            else:
                out = out.with_columns(pl.col(col).fill_null(default))
        return out

    def _load_state_metadata(self) -> pl.DataFrame:
        path = self.paths.state_panel_path(self.start_year, self.end_year)
        if not path.exists():
            path = self.paths.inputs / f"abm_v4_state_panel_{self.start_year}_{self.end_year}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing ABM v4 state panel: {path}. Run --build-state first.")
        state = pl.read_parquet(path)
        year_col = "Year" if "Year" in state.columns else "year"
        cols = [col for col in ["country_sector", year_col, "Country", "Sector", "X_observed", "EI", "EI_observed", "emissions_observed"] if col in state.columns]
        out = state.select(cols).rename({year_col: "year"})
        if "EI_observed" not in out.columns and "EI" in out.columns:
            out = out.rename({"EI": "EI_observed"})
        return out.sort(["country_sector", "year"]).with_columns(
            pl.col("EI_observed").shift(-1).over("country_sector").alias("_EI_next"),
            pl.col("year").shift(-1).over("country_sector").alias("_year_next"),
        ).with_columns(
            pl.when((pl.col("_year_next") == pl.col("year") + 1) & (pl.col("EI_observed") > 0) & (pl.col("_EI_next") > 0))
            .then(pl.col("EI_observed").log() - pl.col("_EI_next").log())
            .otherwise(None)
            .alias("observed_rEI")
        ).drop("_EI_next", "_year_next")

    def _read_q_index(self, index_path: Path) -> list[dict[str, Any]]:
        if not index_path.exists():
            labels = self.paths.data_root / "raw" / str(self.start_year) / "labels_Q.txt"
            out = []
            if labels.exists():
                for idx, line in enumerate(labels.read_text(encoding="utf-8", errors="ignore").splitlines()):
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        out.append({"row": idx + 1, "family": parts[0], "item": parts[1], "row_label": line})
            return out
        try:
            frame = pl.read_csv(index_path)
        except Exception:
            labels = self.paths.data_root / "raw" / str(self.start_year) / "labels_Q.txt"
            out = []
            if labels.exists():
                for idx, line in enumerate(labels.read_text(encoding="utf-8", errors="ignore").splitlines()):
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        out.append({"row": idx + 1, "family": parts[0], "item": parts[1], "row_label": line})
            return out
        rows = []
        for row in frame.to_dicts():
            rows.append(
                {
                    "row": _as_int(row.get("Row")),
                    "family": str(row.get("IndicatorName", "")),
                    "item": str(row.get("LineItems", "")),
                    "row_label": f"{row.get('IndicatorName', '')}\t{row.get('LineItems', '')}",
                }
            )
        return rows

    def _match_component_in_index(self, component: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
        for row in rows:
            family = str(row.get("family", "")).lower()
            item = str(row.get("item", "")).lower()
            if "energy usage" not in family:
                continue
            if any(pattern in item for pattern in component["patterns"]):
                return {
                    "row_index_zero_based": _as_int(row.get("row")) - 1,
                    "family": row.get("family", ""),
                    "item": row.get("item", ""),
                    "row_label": row.get("row_label", ""),
                }
        return {}

    def _read_energy_label_matches(self, labels_path: Path) -> list[str]:
        if not labels_path.exists():
            return []
        text = labels_path.read_text(encoding="utf-8", errors="ignore").lower()
        return [
            component["canonical"]
            for component in self.energy_components
            if "energy usage" in text and any(pattern in text for pattern in component["patterns"])
        ]

    def _aggregate_plausibility_rows(self, panel: pl.DataFrame, level: str, group_col: str | None) -> list[dict[str, Any]]:
        if panel.is_empty():
            return []
        group_cols = ["year"] if group_col is None else [group_col, "year"]
        grouped = panel.group_by(group_cols).agg(
            pl.col("fossil_energy_TJ").sum().alias("fossil_energy_TJ"),
            pl.col("coal_TJ").sum().alias("coal_TJ"),
            pl.col("natural_gas_TJ").sum().alias("gas_TJ"),
            pl.col("petroleum_TJ").sum().alias("petroleum_TJ"),
            pl.col("clean_electricity_TJ").sum().alias("clean_electricity_TJ"),
            pl.col("renewable_electricity_TJ").sum().alias("renewable_electricity_TJ"),
            pl.col("total_tracked_energy_TJ").sum().alias("total_tracked_energy_TJ"),
        ).with_columns(
            pl.when(pl.col("total_tracked_energy_TJ") > 0).then(pl.col("fossil_energy_TJ") / pl.col("total_tracked_energy_TJ")).otherwise(None).alias("fossil_share"),
            pl.when(pl.col("total_tracked_energy_TJ") > 0).then(pl.col("coal_TJ") / pl.col("total_tracked_energy_TJ")).otherwise(None).alias("coal_share"),
            pl.when(pl.col("total_tracked_energy_TJ") > 0).then(pl.col("clean_electricity_TJ") / pl.col("total_tracked_energy_TJ")).otherwise(None).alias("clean_electricity_share"),
            pl.when(pl.col("total_tracked_energy_TJ") > 0).then(pl.col("renewable_electricity_TJ") / pl.col("total_tracked_energy_TJ")).otherwise(None).alias("renewable_electricity_share"),
        )
        sort_cols = [group_col, "year"] if group_col else ["year"]
        grouped = grouped.sort(sort_cols).with_columns(
            (pl.col("total_tracked_energy_TJ") / pl.col("total_tracked_energy_TJ").shift(1).over(group_col) - 1 if group_col else pl.col("total_tracked_energy_TJ") / pl.col("total_tracked_energy_TJ").shift(1) - 1).alias("year_on_year_total_growth"),
            (pl.col("fossil_share") - (pl.col("fossil_share").shift(1).over(group_col) if group_col else pl.col("fossil_share").shift(1))).alias("year_on_year_fossil_share_change"),
            (pl.col("clean_electricity_share") - (pl.col("clean_electricity_share").shift(1).over(group_col) if group_col else pl.col("clean_electricity_share").shift(1))).alias("year_on_year_clean_share_change"),
        )
        rows = []
        for row in grouped.to_dicts():
            flag = "ok"
            if abs(_as_float(row.get("year_on_year_total_growth"))) > 5 or abs(_as_float(row.get("year_on_year_fossil_share_change"))) > 0.5:
                flag = "extreme_jump"
            group = "global" if group_col is None else str(row.get(group_col))
            rows.append({"aggregation_level": level, "group": group, **{k: v for k, v in row.items() if k != group_col}, "plausibility_flag": flag, "notes": "internal plausibility check only"})
        return rows

    def _quality_value(self, quality: pl.DataFrame, component: str, column: str) -> float:
        if quality.is_empty() or "component" not in quality.columns or column not in quality.columns:
            return 0.0
        row = quality.filter(pl.col("component") == component)
        return _as_float(row[column].item()) if not row.is_empty() else 0.0

    def _expected_node_years(self) -> int:
        path = self.paths.state_panel_path(self.start_year, self.end_year)
        if not path.exists():
            return 0
        return _as_int(pl.scan_parquet(path).select(pl.len()).collect().item())

    def _invalid_share_count(self, panel: pl.DataFrame) -> int:
        count = 0
        for col in self.share_columns:
            if col in panel.columns:
                series = panel[col].drop_nulls()
                count += sum(1 for value in series if value < -1e-9 or value > 1 + 1e-9)
        return count

    def _share_quality_expr(self) -> pl.Expr:
        invalid = None
        for col in self.share_columns:
            expr = (pl.col(col) < -1e-9) | (pl.col(col) > 1 + 1e-9)
            invalid = expr if invalid is None else invalid | expr
        return pl.when(invalid if invalid is not None else pl.lit(False)).then(pl.lit("invalid_share")).otherwise(pl.lit("ok"))

    def _jump_count(self, panel: pl.DataFrame, component: str) -> int:
        if component not in panel.columns:
            return 0
        jumps = panel.sort(["country_sector", "year"]).with_columns(
            (pl.col(component).log() - pl.col(component).shift(1).over("country_sector").log()).abs().alias("_abs_log_change")
        )
        threshold = self._quantile(jumps["_abs_log_change"], 0.95)
        return _as_int((jumps["_abs_log_change"] > threshold).sum()) if threshold > 0 else 0

    def _quantile(self, values: pl.Series, q: float) -> float:
        clean = values.drop_nulls()
        if clean.is_empty():
            return 0.0
        return _as_float(clean.quantile(q))

    def _nonfinite_count(self, values: pl.Series) -> int:
        return sum(1 for value in values if value is not None and not math.isfinite(float(value)))

    def _clean_numeric(self, value: Any) -> float | None:
        try:
            out = float(value)
            return out if math.isfinite(out) else None
        except (TypeError, ValueError):
            return None

    def _year_from_path(self, path: Path) -> int | None:
        for part in reversed(path.parts):
            if part.isdigit() and len(part) == 4:
                return int(part)
        return None

    def _electricity_like_expr(self) -> pl.Expr:
        return pl.col("Sector").fill_null("").str.to_lowercase().str.contains("electricity|gas and water|utilities|power")

    def _china_electricity_expr(self) -> pl.Expr:
        return (
            pl.col("Country").fill_null("").str.to_lowercase().is_in(["china", "chn", "people's republic of china"])
            & self._electricity_like_expr()
        )

    def _hypothesis(self, name: str, test: str, result: bool, evidence: str, interpretation: str) -> dict[str, Any]:
        return {"hypothesis": name, "test": test, "result": result, "evidence": evidence, "interpretation": interpretation, "status": "supported" if result else "not_supported"}

    def _markdown_table(self, frame: pl.DataFrame) -> str:
        if frame.is_empty():
            return "_No rows available._"
        lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join("---" for _ in frame.columns) + " |"]
        for row in frame.to_dicts():
            lines.append("| " + " | ".join(self._format_markdown_value(row.get(column)) for column in frame.columns) + " |")
        return "\n".join(lines)

    def _format_markdown_value(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        if value is None:
            return ""
        return str(value).replace("|", "/").replace("\n", " ")


@dataclass(frozen=True)
class ABMV4FinalConsolidationResult:
    """Final ABM v4 consolidation artifacts built from existing diagnostics."""

    input_availability: pl.DataFrame
    surviving_rule_comparison: pl.DataFrame
    validation_objective_matrix: pl.DataFrame
    rejected_mechanism_register: pl.DataFrame
    model_boundary_statement: str
    scenario_readiness_assessment: pl.DataFrame
    abm_v5_research_agenda: pl.DataFrame
    hypothesis_status: pl.DataFrame
    consolidation_report: str
    portfolio_summary: str


class ABMV4FinalConsolidator:
    """Consolidate ABM v4 evidence without running scenarios or new mechanisms."""

    def __init__(self, paths: ABMV4Paths) -> None:
        self.paths = paths

    def run(self) -> ABMV4FinalConsolidationResult:
        """Build final ABM v4 consolidation outputs in memory."""
        availability = self.build_input_availability_report()
        missing_required = availability.filter(
            (pl.col("required_or_optional") == "required") & (~pl.col("exists"))
        )
        if not missing_required.is_empty():
            missing = ", ".join(missing_required["path"].to_list())
            raise FileNotFoundError(f"Cannot finalize ABM v4; missing required surviving-rule inputs: {missing}")

        surviving = self.build_surviving_rule_comparison()
        rejected = self.build_rejected_mechanism_register()
        objective = self.build_validation_objective_matrix()
        boundary = self.build_model_boundary_statement()
        readiness = self.build_scenario_readiness_assessment()
        agenda = self.build_abm_v5_research_agenda()
        hypotheses = self.build_hypothesis_status_table()
        report = self.build_consolidation_report(surviving, rejected, objective, readiness, agenda, hypotheses)
        portfolio = self.build_portfolio_summary()
        return ABMV4FinalConsolidationResult(
            input_availability=availability,
            surviving_rule_comparison=surviving,
            validation_objective_matrix=objective,
            rejected_mechanism_register=rejected,
            model_boundary_statement=boundary,
            scenario_readiness_assessment=readiness,
            abm_v5_research_agenda=agenda,
            hypothesis_status=hypotheses,
            consolidation_report=report,
            portfolio_summary=portfolio,
        )

    def load_surviving_rule_results(self) -> dict[str, pl.DataFrame]:
        """Load required and available surviving-rule result tables."""
        return {
            name: self._read_table(path)
            for name, path, required in self._input_specs()
            if required and path.exists()
        }

    def load_rejected_branch_recommendations(self) -> dict[str, pl.DataFrame]:
        """Load available rejected or diagnostic branch recommendation tables."""
        return {
            name: self._read_table(path)
            for name, path, required in self._input_specs()
            if not required and path.exists() and path.suffix.lower() in {".csv", ".parquet"}
        }

    def build_input_availability_report(self) -> pl.DataFrame:
        """Report required and optional finalization inputs without assuming availability."""
        rows = []
        for name, path, required in self._input_specs():
            exists = path.exists()
            rows.append(
                {
                    "input_group": name.split(":", 1)[0],
                    "path": self._relative(path),
                    "required_or_optional": "required" if required else "optional",
                    "exists": exists,
                    "rows_if_readable": self._row_count(path) if exists else None,
                    "status": "available" if exists else ("missing_required" if required else "missing_optional"),
                    "notes": "" if exists else ("required to compare surviving rules" if required else "optional branch evidence not found"),
                }
            )
        return pl.DataFrame(rows)

    def build_surviving_rule_comparison(self) -> pl.DataFrame:
        """Compare the two final ABM v4 surviving rules."""
        q_status = self._recommendation_value(self.paths.q_energy_mix_recommendation_path)
        adaptive_status = self._recommendation_value(self.paths.adaptive_EID_recommendation_path)
        fixed_status = self._recommendation_value(self.paths.essential_input_dampener_recommendation_path)
        return pl.DataFrame(
            [
                {
                    "rule_name": "frontier_gap_readiness",
                    "rule_status": "survives",
                    "theoretical_role": "aggregate-safe historical baseline with readiness gating",
                    "aggregate_emissions_fit": "stronger aggregate safety than historical_frontier_gap_only in final interpretation",
                    "transition_mechanism_fit": "less clean as a transition mechanism because readiness also dampens high-emissions nodes",
                    "electricity_fit": "diagnostic weakness remains for electricity-like high-emissions nodes",
                    "china_electricity_fit": "incomplete; China electricity points to missing fuel-mix mechanism",
                    "high_emissions_node_fit": "readiness dampening helps aggregate fit but obscures mechanism interpretation",
                    "high_EID_node_fit": f"diagnostic only; fixed_EID={fixed_status}; adaptive_EID={adaptive_status}",
                    "strengths": "Best retained rule for aggregate emissions plausibility under historical production forcing.",
                    "weaknesses": "Does not fully explain electricity and high-emissions transition mechanisms.",
                    "retained_as": "aggregate_safe_baseline",
                    "scenario_use_status": "not_scenario_ready",
                },
                {
                    "rule_name": "historical_frontier_gap_only",
                    "rule_status": "survives",
                    "theoretical_role": "transition-mechanism benchmark using calibrated historical frontier-gap closure",
                    "aggregate_emissions_fit": "weaker aggregate emissions safety than frontier_gap_readiness",
                    "transition_mechanism_fit": "cleaner benchmark for frontier-gap transition interpretation",
                    "electricity_fit": "diagnostic weakness remains for electricity-like high-emissions nodes",
                    "china_electricity_fit": "incomplete; China electricity points to missing fuel-mix mechanism",
                    "high_emissions_node_fit": "reveals high-emissions tradeoff more directly",
                    "high_EID_node_fit": f"EID retained as ontology evidence only; Q_energy_mix={q_status}",
                    "strengths": "Best retained rule for interpreting the frontier-gap mechanism itself.",
                    "weaknesses": "Aggregate emissions errors show that mechanism fit alone is not enough.",
                    "retained_as": "transition_mechanism_benchmark",
                    "scenario_use_status": "not_scenario_ready",
                },
            ]
        )

    def build_validation_objective_matrix(self) -> pl.DataFrame:
        """Summarize what each evidence branch can validly support."""
        rows = [
            {
                "objective": "transition_mechanism_validity",
                "frontier_gap_readiness_assessment": "supported as baseline but mechanism is mixed with readiness dampening",
                "historical_frontier_gap_only_assessment": "best surviving transition-mechanism benchmark",
                "EID_assessment": "rejected as ABM v4 rule; retained as ontology evidence",
                "Q_energy_mix_assessment": "supports missing fuel-structure mechanism conceptually",
                "evidence": "Phase 16-27 diagnostics; weak Q predictor correlations but local signal.",
                "conclusion": "historical_frontier_gap_only remains cleaner for transition-mechanism interpretation",
            },
            {
                "objective": "aggregate_emissions_validity",
                "frontier_gap_readiness_assessment": "wins aggregate/emissions safety",
                "historical_frontier_gap_only_assessment": "useful benchmark but weaker aggregate safety",
                "EID_assessment": "fixed/adaptive variants do not rescue aggregate validation",
                "Q_energy_mix_assessment": "aggregate-only diagnostics are usable",
                "evidence": "multi-year validation, base comparison, EID recommendations, Q aggregate recommendation.",
                "conclusion": "frontier_gap_readiness wins aggregate/emissions safety",
            },
            {
                "objective": "electricity_high_emissions_validity",
                "frontier_gap_readiness_assessment": "partly dampens high-emissions nodes but does not explain electricity mechanism",
                "historical_frontier_gap_only_assessment": "exposes electricity transition problem directly",
                "EID_assessment": "diagnostic/ontology only",
                "Q_energy_mix_assessment": "China electricity evidence supports missing fuel-mix mechanism",
                "evidence": "electricity audits, China electricity diagnostics, Q energy-mix audit.",
                "conclusion": "electricity/high-emissions validity requires energy-system variables beyond ABM v4",
            },
            {
                "objective": "capability_source_validity",
                "frontier_gap_readiness_assessment": "usable only as historical diagnostic context",
                "historical_frontier_gap_only_assessment": "usable only as historical diagnostic context",
                "EID_assessment": "EID and structural dependence are ontology evidence, not final rules",
                "Q_energy_mix_assessment": "not a capability source",
                "evidence": "Phase 9C/9D and EID branch diagnostics when available.",
                "conclusion": "capability/source diagnostics remain useful but not scenario sufficient",
            },
            {
                "objective": "production_feasibility_validity",
                "frontier_gap_readiness_assessment": "depends on observed production forcing",
                "historical_frontier_gap_only_assessment": "depends on observed production forcing",
                "EID_assessment": "does not solve endogenous production dynamics",
                "Q_energy_mix_assessment": "does not solve endogenous production dynamics",
                "evidence": "ABM v4 production remains historically forced.",
                "conclusion": "production dynamics must be rebuilt before scenario use",
            },
            {
                "objective": "data_quality_validity",
                "frontier_gap_readiness_assessment": "survives with existing validated state and simulation data",
                "historical_frontier_gap_only_assessment": "survives with existing validated state and simulation data",
                "EID_assessment": "diagnostic data useful but insufficient for behavioural rule",
                "Q_energy_mix_assessment": "invalid shares, negative values, severe flags block node-level rule integration",
                "evidence": "Q quality audit, EID failure-mode and adaptive recommendations.",
                "conclusion": "Q energy mix is aggregate-only/ABM v5 evidence",
            },
            {
                "objective": "scenario_readiness",
                "frontier_gap_readiness_assessment": "not scenario-ready",
                "historical_frontier_gap_only_assessment": "not scenario-ready",
                "EID_assessment": "diagnostic/ontology only",
                "Q_energy_mix_assessment": "aggregate-only/ABM v5 evidence",
                "evidence": "no endogenous production dynamics, no policy/fuel mechanism, no validated node-level energy rule.",
                "conclusion": "scenario readiness remains premature",
            },
        ]
        return pl.DataFrame(rows)

    def build_rejected_mechanism_register(self) -> pl.DataFrame:
        """Register rejected and diagnostic-only ABM v4 mechanism branches."""
        rows = [
            self._rejected_row("legacy_raw_log emissions rule", "Phase 7-14", "simple historical rEI extrapolation", "rejected", "too implicit and less theoretically defensible than frontier-gap rules", "baseline foil", "none except historical comparison", "rejected_for_scenarios"),
            self._rejected_row("fixed EID dampener", "Phase 23", "essential-input dependence might slow transition", "rejected_for_abm_v4_rule", "did not clear validation thresholds and risked treating ontology as behaviour", "ontology evidence", "ABM v5 agent typing and dependence diagnostics", "not_scenario_ready"),
            self._rejected_row("adaptive EID dampener", "Phase 26", "adaptive EID might improve forward validation", "rejected_for_abm_v4_rule", "walk-forward diagnostics did not beat the historical frontier-gap benchmark robustly", "ontology evidence and overfitting warning", "ABM v5 design input only", "not_scenario_ready"),
            self._rejected_row("EID diagnostic multi-year mode", "Phase 25", "test EID in integrated multi-year validation", "diagnostic_only", "materially worsened or failed key all-node validation thresholds", "failure-mode evidence", "ontology and subtype diagnostics", "not_scenario_ready"),
            self._rejected_row("Q energy mix country-sector transition rule", "Phase 27", "fuel structure may explain electricity/high-emissions transitions", "rejected_for_abm_v4_node_level_rule", "negative values, invalid shares, severe aggregate flags, and weak node-level predictor power", "aggregate diagnostics and ABM v5 fuel-mechanism evidence", "cleaner energy-system data for ABM v5", "not_scenario_ready"),
            self._rejected_row("historical residual as scenario-facing rule", "Phase 23-26", "residuals explain missed historical transition variation", "rejected_for_scenarios", "historical residual is not observable ex ante and would leak validation information", "diagnostic benchmark", "feature discovery only", "rejected_for_scenarios"),
            self._rejected_row("electricity-specific transition patch", "Phase 20", "electricity nodes need a different transition regime", "diagnostic_only", "supported missing-mechanism diagnosis but not a validated general ABM v4 rule", "electricity mechanism evidence", "ABM v5 energy/policy regime module", "not_scenario_ready"),
            self._rejected_row("structural signature proxy rule", "Phase 21", "network signatures might proxy transition inertia", "diagnostic_only", "useful as classification evidence but insufficient as a scenario rule", "agent ontology evidence", "ABM v5 agent typing", "not_scenario_ready"),
        ]
        return pl.DataFrame(rows)

    def build_model_boundary_statement(self) -> str:
        """Render the final ABM v4 model-boundary statement."""
        return "\n".join(
            [
                "# ABM v4 Final Model Boundary Statement",
                "",
                "ABM v4 is a historical diagnostic framework. It is a country-sector production-network model used to test emissions-transition mechanisms under observed production forcing, identify validation trade-offs, and locate missing mechanisms.",
                "",
                "ABM v4 is not scenario-ready. It is not a policy-counterfactual simulator, not a fully agent-typed ABM, and not a complete energy-system transition model.",
                "",
                "The central boundary is historical production forcing: output paths are anchored to observed production, so the model does not yet generate endogenous scenario production dynamics.",
                "",
                "Two emissions-transition rules survive because they optimize different objectives. `frontier_gap_readiness` is retained as the aggregate-safe baseline. `historical_frontier_gap_only` is retained as the cleaner transition-mechanism benchmark.",
                "",
                "Electricity and high-emissions nodes expose missing fuel, investment, and policy mechanisms. China electricity is the clearest diagnostic case: fuel mix matters, but the Q energy-mix data are not clean enough for node-level ABM v4 rule integration.",
                "",
                "EID failed as a behavioural dampener for ABM v4. It remains useful ontology evidence because essential-input dependence helps describe node roles, but it is not validated as a final transition rule.",
                "",
                "Q energy mix is retained as aggregate validation evidence and ABM v5 motivation, not as a country-sector behavioural rule in ABM v4.",
                "",
            ]
        )

    def build_scenario_readiness_assessment(self) -> pl.DataFrame:
        """Assess remaining blockers to scenario use."""
        rows = [
            self._readiness_row("emissions_transition_rule", "blocked", "two rules survive for different validation objectives", "no single scenario-facing rule is validated", "model fuel/policy mechanisms and select objective explicitly"),
            self._readiness_row("production_dynamics", "blocked", "historical production forcing remains central", "no endogenous output dynamics", "build demand, capacity, and propagation dynamics"),
            self._readiness_row("supplier_substitution", "limited", "supplier diagnostics exist", "substitution is not validated for future counterfactuals", "estimate substitution and capacity constraints"),
            self._readiness_row("capability_dynamics", "limited", "capability diagnostics are inspectable", "capability evolution is not scenario calibrated", "model capability investment and learning"),
            self._readiness_row("electricity_energy_system", "blocked", "China electricity and Q audits identify missing fuel mechanism", "node-level energy data quality is insufficient", "use cleaner generation/fuel/capacity data"),
            self._readiness_row("policy_institutional_variables", "blocked", "errors likely depend on policy/investment regimes", "policy variables are absent", "add policy, investment, carbon pricing, subsidies, and regulation data"),
            self._readiness_row("data_quality", "blocked", "Q audit finds invalid shares, negative values, and severe flags", "country-sector Q quality blocks behavioural rules", "reconstruct or source cleaner energy-system data"),
            self._readiness_row("validation_metrics", "limited", "tradeoff diagnostics exist", "validation objective is not scenario-facing", "define scenario validation objective and thresholds"),
            self._readiness_row("interpretation_risk", "blocked", "network-only rules leave missing mechanisms", "scenario claims would overstate ABM v4", "present ABM v4 as diagnostic foundation"),
            self._readiness_row("overall_scenario_readiness", "not_scenario_ready", "multiple blocking dimensions remain", "ABM v4 is historical diagnostic, not forecasting model", "build ABM v5 energy, policy, agent ontology, and endogenous production modules"),
        ]
        return pl.DataFrame(rows)

    def build_abm_v5_research_agenda(self) -> pl.DataFrame:
        """Build ABM v5 research priorities implied by ABM v4 evidence."""
        return pl.DataFrame(
            [
                {
                    "research_priority": "energy/fuel structure",
                    "motivation_from_abm_v4": "Q energy mix supports a fuel-mix mechanism but is too noisy at node level.",
                    "required_data": "cleaner external energy mix, power generation, fuel use, and capacity data",
                    "candidate_mechanism": "fuel structure and clean-generation substitution",
                    "candidate_agent_type": "energy_infrastructure_agent",
                    "expected_validation_test": "electricity and high-emissions node errors fall without worsening aggregate emissions",
                    "priority_level": "high",
                },
                {
                    "research_priority": "policy/institutional regime",
                    "motivation_from_abm_v4": "electricity and China electricity errors likely require policy, investment, and regulatory variables.",
                    "required_data": "renewable policy indices, electricity investment, coal phaseout, ETS/carbon pricing, fossil subsidies",
                    "candidate_mechanism": "policy priority and transition acceleration/counterforce",
                    "candidate_agent_type": "policy_regime_agent",
                    "expected_validation_test": "policy-regime terms explain transition accelerations and stalls",
                    "priority_level": "high",
                },
                {
                    "research_priority": "capital-stock inertia",
                    "motivation_from_abm_v4": "infrastructure and heavy industry may transition slowly because of long-lived assets.",
                    "required_data": "capital stock, asset age, generation capacity, industrial plant data",
                    "candidate_mechanism": "stock turnover constraint",
                    "candidate_agent_type": "capital_stock_agent",
                    "expected_validation_test": "asset-age and capacity constraints reduce wrong-sign transition errors",
                    "priority_level": "high",
                },
                {
                    "research_priority": "explicit agent ontology",
                    "motivation_from_abm_v4": "ABM v4 country-sector nodes are too flat for scenario mechanisms.",
                    "required_data": "structural role, sector classification, energy variables, and policy variables",
                    "candidate_mechanism": "agent-type-specific transition and response rules",
                    "candidate_agent_type": "ordinary production, energy infrastructure, heavy industry/materials, transport/logistics, systemic services, accounting/non-agent nodes",
                    "expected_validation_test": "agent-type rules improve subgroup validation without hiding aggregate errors",
                    "priority_level": "high",
                },
                {
                    "research_priority": "endogenous production dynamics",
                    "motivation_from_abm_v4": "scenarios require production dynamics without historical forcing.",
                    "required_data": "demand scenarios, substitution elasticities, capacity constraints",
                    "candidate_mechanism": "dynamic Leontief/agent demand propagation",
                    "candidate_agent_type": "production_network_agent",
                    "expected_validation_test": "historical backtests reproduce output and emissions dynamics without observed output forcing",
                    "priority_level": "high",
                },
            ]
        )

    def build_hypothesis_status_table(self) -> pl.DataFrame:
        """Build the final status table for ABM v4 hypotheses."""
        rows = [
            self._hypothesis_row("frontier_gap_readiness_aggregate_safe", "supported", "retained as aggregate_safe_baseline", "best surviving aggregate-safety rule", "use as ABM v4 baseline"),
            self._hypothesis_row("historical_frontier_gap_transition_benchmark", "supported", "retained as transition_mechanism_benchmark", "cleaner transition mechanism interpretation", "use as benchmark"),
            self._hypothesis_row("EID_as_transition_dampener", "not_supported", "fixed and integrated EID diagnostics fail validation thresholds", "EID is not an ABM v4 behavioural dampener", "do not promote to rule"),
            self._hypothesis_row("EID_as_ontology_signal", "supported", "EID dependence and failure-mode diagnostics identify structural roles", "EID helps classify nodes", "retain for ABM v5 ontology"),
            self._hypothesis_row("adaptive_EID_as_ABM_v4_rule", "not_supported", "walk-forward adaptive EID does not robustly beat historical frontier-gap benchmark", "adaptive calibration does not rescue EID", "do not integrate"),
            self._hypothesis_row("Q_energy_mix_as_country_sector_rule", "not_supported", "invalid shares, negative values, severe flags, weak correlations", "country-sector Q quality blocks rule integration", "do not integrate into ABM v4"),
            self._hypothesis_row("Q_energy_mix_as_aggregate_diagnostic", "supported", "all intended Q rows found and aggregate recommendation is usable", "energy mix is useful for interpretation", "retain as aggregate diagnostic"),
            self._hypothesis_row("China_electricity_missing_fuel_mix_mechanism", "supported", "China electricity has high fossil/coal dependence and persistent modelling problem", "fuel mix is a missing mechanism", "prioritize in ABM v5"),
            self._hypothesis_row("ABM_v4_scenario_ready", "not_supported", "historical forcing, two-rule tradeoff, missing fuel/policy mechanisms", "ABM v4 is diagnostic, not forecasting", "keep scenarios blocked"),
            self._hypothesis_row("ABM_v5_needs_energy_policy_agent_ontology", "supported", "EID, electricity, and Q diagnostics all point beyond flat nodes", "ABM v5 needs explicit agent ontology", "design ABM v5 around energy, policy, capital, and production dynamics"),
        ]
        return pl.DataFrame(rows)

    def write_outputs(self, result: ABMV4FinalConsolidationResult) -> None:
        """Write final consolidation outputs under validation only."""
        self.paths.validation.mkdir(parents=True, exist_ok=True)
        result.input_availability.write_csv(self.paths.final_abm_v4_input_availability_path)
        result.surviving_rule_comparison.write_csv(self.paths.final_surviving_rule_comparison_path)
        result.validation_objective_matrix.write_csv(self.paths.final_validation_objective_matrix_path)
        result.rejected_mechanism_register.write_csv(self.paths.final_rejected_mechanism_register_path)
        self.paths.final_model_boundary_statement_path.write_text(result.model_boundary_statement, encoding="utf-8")
        result.scenario_readiness_assessment.write_csv(self.paths.final_scenario_readiness_assessment_path)
        result.abm_v5_research_agenda.write_csv(self.paths.final_abm_v5_research_agenda_path)
        result.hypothesis_status.write_csv(self.paths.final_abm_v4_hypothesis_status_path)
        self.paths.final_abm_v4_consolidation_report_path.write_text(result.consolidation_report, encoding="utf-8")
        self.paths.final_abm_v4_portfolio_summary_path.write_text(result.portfolio_summary, encoding="utf-8")

    def build_consolidation_report(
        self,
        surviving: pl.DataFrame,
        rejected: pl.DataFrame,
        objective: pl.DataFrame,
        readiness: pl.DataFrame,
        agenda: pl.DataFrame,
        hypotheses: pl.DataFrame,
    ) -> str:
        """Render the final Markdown consolidation report."""
        lines = [
            "# ABM v4 Final Consolidation Report",
            "",
            "## 1. Final Status",
            "",
            "ABM v4 closes as a historically validated diagnostic framework, not a scenario-ready forecasting model.",
            "",
            "## 2. Surviving Rules",
            "",
            self._markdown_table(surviving),
            "",
            "## 3. Why Two Rules Survive",
            "",
            "`frontier_gap_readiness` is retained because it is aggregate-safe. `historical_frontier_gap_only` is retained because it is the cleaner transition-mechanism benchmark. A single rule would hide the validation-objective trade-off.",
            "",
            "## 4. EID Branch",
            "",
            "EID taught us that structural dependence is real ontology evidence, but the fixed, integrated, and adaptive dampeners do not validate as ABM v4 behavioural rules.",
            "",
            "## 5. Why EID Is Rejected",
            "",
            "EID is rejected for ABM v4 transition-rule use because it does not robustly improve the main historical validation objectives and risks converting structural labels into unsupported behaviour.",
            "",
            "## 6. Q Energy-Mix Branch",
            "",
            "Q energy mix found the intended energy-use rows and supports the fuel-structure mechanism conceptually, especially for China electricity and high-emissions electricity-like nodes.",
            "",
            "## 7. Why Q Energy Mix Is Not Integrated",
            "",
            "Country-sector Q data contain invalid shares, negative values, severe aggregate plausibility flags, and weak node-level predictor power. It is retained for aggregate diagnostics and ABM v5 evidence.",
            "",
            "## 8. Model Boundary",
            "",
            self.build_model_boundary_statement(),
            "## 9. Scenario Readiness",
            "",
            self._markdown_table(readiness),
            "",
            "## 10. ABM v5 Focus",
            "",
            self._markdown_table(agenda),
            "",
            "## 11. Portfolio Narrative",
            "",
            "ABM v4 is a strong research artifact because it demonstrates disciplined mechanism testing, records failed branches explicitly, and turns validation failures into a clear ABM v5 design agenda.",
            "",
            "## Final Hypotheses",
            "",
            self._markdown_table(hypotheses),
            "",
            "## Rejected Mechanisms",
            "",
            self._markdown_table(rejected),
            "",
            "## Validation Objective Matrix",
            "",
            self._markdown_table(objective),
            "",
        ]
        return "\n".join(lines)

    def build_portfolio_summary(self) -> str:
        """Render a short non-technical ABM v4 portfolio summary."""
        return "\n".join(
            [
                "# ABM v4 Portfolio Summary",
                "",
                "ABM v4 tested whether production-network structure and frontier dynamics can explain green transition patterns across country-sector nodes.",
                "",
                "The project found a robust trade-off: one surviving rule gives safer aggregate emissions behavior, while another gives a cleaner transition-mechanism benchmark. Instead of forcing a false single winner, ABM v4 preserves both and documents exactly what each can claim.",
                "",
                "The model then stress-tested essential-input dependence and energy-mix explanations. EID proved useful as structural ontology evidence but not as a validated behavioural dampener. Q energy mix supported the idea that fuel structure matters, especially for China electricity, but the country-sector data quality was not strong enough for direct rule integration.",
                "",
                "The final conclusion is honest and useful: a network-only ABM is not sufficient for scenarios without explicit energy, policy, capital-stock, and endogenous production mechanisms. ABM v4 therefore closes as a rigorous diagnostic foundation for ABM v5.",
                "",
            ]
        )

    def _input_specs(self) -> list[tuple[str, Path, bool]]:
        return [
            ("surviving_rules:frontier_gap_state", self.paths.base_multiyear_state_panel_path, True),
            ("surviving_rules:historical_frontier_gap_state", self.paths.base_multiyear_state_panel_historical_frontier_gap_path, True),
            ("surviving_rules:frontier_gap_summary", self.paths.base_multiyear_summary_panel_path, True),
            ("surviving_rules:historical_frontier_gap_summary", self.paths.base_multiyear_summary_panel_historical_frontier_gap_path, True),
            ("surviving_rules:error_summary", self.paths.multiyear_error_summary_path, True),
            ("surviving_rules:model_comparison", self.paths.multiyear_base_model_comparison_csv_path, False),
            ("EID_branch:fixed_recommendation", self.paths.essential_input_dampener_recommendation_path, False),
            ("EID_branch:failure_mode_recommendation", self.paths.eid_failure_mode_recommendation_path, False),
            ("EID_branch:multiyear_recommendation", self.paths.multiyear_EID_diagnostic_recommendation_path, False),
            ("EID_branch:adaptive_recommendation", self.paths.adaptive_EID_recommendation_path, False),
            ("EID_branch:adaptive_hypotheses", self.paths.adaptive_EID_hypothesis_tests_path, False),
            ("Q_energy_branch:recommendation", self.paths.q_energy_mix_recommendation_path, False),
            ("Q_energy_branch:hypotheses", self.paths.q_energy_mix_hypothesis_tests_path, False),
            ("Q_energy_branch:quality_audit", self.paths.q_energy_mix_quality_audit_path, False),
            ("Q_energy_branch:china_electricity_audit", self.paths.q_energy_mix_china_electricity_audit_path, False),
            ("Q_energy_branch:predictor_screening", self.paths.q_energy_mix_predictor_screening_path, False),
            ("Q_energy_branch:report", self.paths.q_energy_mix_report_path, False),
            ("electricity_branch:data_recommendation", self.paths.electricity_data_audit_recommendation_path, False),
            ("electricity_branch:raw_eora_recommendation", self.paths.raw_eora_electricity_data_audit_recommendation_path, False),
            ("electricity_branch:regime_recommendation", self.paths.electricity_transition_regime_recommendation_path, False),
            ("capability_branch:io_capability_robustness", self.paths.io_capability_robustness_path, False),
            ("capability_branch:io_downstream_exposure", self.paths.io_downstream_exposure_audit_path, False),
            ("production_branch:production_feasibility", self.paths.production_feasibility_report_path, False),
            ("documentation:implementation_note", self.paths.project_root / "abm_v4_implementation_note.md", False),
        ]

    def _row_count(self, path: Path) -> int | None:
        try:
            suffix = path.suffix.lower()
            if suffix == ".parquet":
                return int(pl.scan_parquet(path).select(pl.len()).collect().item())
            if suffix == ".csv":
                return int(pl.scan_csv(path).select(pl.len()).collect().item())
            if suffix == ".md":
                return len(path.read_text(encoding="utf-8").splitlines())
        except Exception as exc:  # pragma: no cover - defensive report path
            return None
        return None

    def _read_table(self, path: Path) -> pl.DataFrame:
        if path.suffix.lower() == ".parquet":
            return pl.read_parquet(path)
        if path.suffix.lower() == ".csv":
            return pl.read_csv(path)
        return pl.DataFrame()

    def _recommendation_value(self, path: Path) -> str:
        if not path.exists() or path.suffix.lower() != ".csv":
            return "missing_optional"
        try:
            frame = pl.read_csv(path)
            if frame.is_empty():
                return "empty"
            row = frame.to_dicts()[0]
            for column in ("recommendation", "recommended_next_action", "recommended_model_variant"):
                if column in row and row[column] is not None:
                    return str(row[column])
        except Exception:
            return "unreadable"
        return "available"

    def _relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.paths.project_root))
        except ValueError:
            return str(path)

    def _markdown_table(self, frame: pl.DataFrame) -> str:
        if frame.is_empty():
            return "_No rows available._"
        lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join("---" for _ in frame.columns) + " |"]
        for row in frame.to_dicts():
            lines.append("| " + " | ".join(str(row.get(column, "")).replace("|", "/").replace("\n", " ") for column in frame.columns) + " |")
        return "\n".join(lines)

    def _rejected_row(
        self,
        mechanism: str,
        phase_tested: str,
        theoretical_motivation: str,
        test_result: str,
        reason: str,
        retained_value: str,
        future_use: str,
        scenario_status: str,
    ) -> dict[str, str]:
        return {
            "mechanism": mechanism,
            "phase_tested": phase_tested,
            "theoretical_motivation": theoretical_motivation,
            "test_result": test_result,
            "reason_rejected_or_limited": reason,
            "retained_value": retained_value,
            "future_use": future_use,
            "scenario_status": scenario_status,
        }

    def _readiness_row(
        self,
        dimension: str,
        status: str,
        evidence: str,
        blocking_issue: str,
        future_work: str,
    ) -> dict[str, str]:
        return {
            "readiness_dimension": dimension,
            "status": status,
            "evidence": evidence,
            "blocking_issue": blocking_issue,
            "required_future_work": future_work,
        }

    def _hypothesis_row(
        self,
        hypothesis: str,
        status: str,
        evidence: str,
        interpretation: str,
        implication: str,
    ) -> dict[str, str]:
        return {
            "hypothesis": hypothesis,
            "status": status,
            "evidence": evidence,
            "interpretation": interpretation,
            "implication": implication,
        }


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
        out = float(value)
        return out if math.isfinite(out) else default
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
