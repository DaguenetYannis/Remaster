from __future__ import annotations

import json
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
