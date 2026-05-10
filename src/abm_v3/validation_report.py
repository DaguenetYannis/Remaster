from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.abm_v3.paths import ABMV3Paths

LOGGER = logging.getLogger(__name__)

CORRECTED_INPUT_ORIENTATION = "transpose_row_fd_without_inventory"
BEHAVIOURAL_COEFFICIENT_MODE = "transpose_row_output_fd_without_inventory"


@dataclass
class ABMV3ValidationReportBuilder:
    """Consolidate ABM v3 diagnostics into a pre-simulation validation report."""

    paths: ABMV3Paths = field(default_factory=ABMV3Paths)

    def build(self, start_year: int = 1995, end_year: int = 2016) -> dict[str, Path]:
        """Read existing diagnostics and write consolidated validation report files."""
        frames = self._load_report_inputs(start_year, end_year)
        flags = self._build_flags(frames)
        by_year = self._build_by_year_table(frames["behavioural_summary"])
        top_errors = self._build_top_output_errors(frames["node_comparison"])
        summary = self._build_summary(start_year, end_year, frames, flags)
        markdown = self._build_markdown(start_year, end_year, frames, summary, flags, by_year, top_errors)

        self.paths.validation_report_dir.mkdir(parents=True, exist_ok=True)
        written_paths = {
            "summary": self.paths.validation_report_summary_path(start_year, end_year),
            "by_year": self.paths.validation_report_by_year_path(start_year, end_year),
            "flags": self.paths.validation_report_flags_path(start_year, end_year),
            "top_output_errors": self.paths.validation_report_top_output_errors_path(start_year, end_year),
            "markdown": self.paths.validation_report_markdown_path(start_year, end_year),
        }
        summary.to_csv(written_paths["summary"], index=False)
        by_year.to_csv(written_paths["by_year"], index=False)
        flags.to_csv(written_paths["flags"], index=False)
        top_errors.to_csv(written_paths["top_output_errors"], index=False)
        written_paths["markdown"].write_text(markdown, encoding="utf-8")
        LOGGER.info("Wrote ABM v3 validation report to %s", self.paths.validation_report_dir)
        return written_paths

    def _load_report_inputs(self, start_year: int, end_year: int) -> dict[str, pd.DataFrame | None]:
        input_paths = {
            "build_report": self.paths.corrected_input_panel_build_report_path(start_year, end_year),
            "orientation": self.paths.input_panel_orientation_comparison_path(start_year, end_year),
            "smoke_test": self.paths.corrected_real_data_smoke_test_path(start_year, end_year),
            "input_intensity": self.paths.corrected_input_intensity_summary_path(start_year, end_year),
            "negative_ei": self.paths.corrected_negative_ei_rows_path(start_year, end_year),
            "behavioural_summary": self.paths.behavioural_leontief_summary_range_path(start_year, end_year),
            "node_comparison": self.paths.behavioural_leontief_node_comparison_range_path(start_year, end_year),
            "ei_sample": self.paths.ei_transition_sample_report_path(start_year, end_year),
            "ei_sample_by_year": self.paths.ei_transition_sample_report_by_year_path(start_year, end_year),
            "ei_scores": self.paths.ei_transition_model_scores_path(start_year, end_year),
            "ei_expected_signs": self.paths.ei_transition_expected_signs_path(start_year, end_year),
            "ei_coefficients": self.paths.ei_transition_coefficients_path(start_year, end_year),
            "legacy_reproduction": self.paths.legacy_historical_reproduction_summary_path(),
            "legacy_rolling": self.paths.legacy_rolling_validation_results_path(),
        }
        return {name: self._read_optional_csv(path, name) for name, path in input_paths.items()}

    def _read_optional_csv(self, path: Path, label: str) -> pd.DataFrame | None:
        if not path.exists():
            LOGGER.warning("ABM v3 validation report input missing: %s (%s)", path, label)
            return None
        try:
            return pd.read_csv(path)
        except Exception as error:
            LOGGER.warning("ABM v3 validation report could not read %s (%s): %s", path, label, error)
            return None

    def _build_by_year_table(self, behavioural_summary: pd.DataFrame | None) -> pd.DataFrame:
        if behavioural_summary is None or behavioural_summary.empty:
            return pd.DataFrame()
        result = behavioural_summary.copy()
        if {"realized_output_total", "observed_output_total"}.issubset(result.columns):
            result["realized_observed_output_ratio"] = self._safe_ratio_series(
                result["realized_output_total"],
                result["observed_output_total"],
            )
        return result

    def _build_top_output_errors(self, node_comparison: pd.DataFrame | None) -> pd.DataFrame:
        columns = [
            "Year",
            "country_sector",
            "Country",
            "Country_detail",
            "Category",
            "Sector",
            "X_observed",
            "X_realized",
            "X_desired",
            "output_gap",
            "output_ratio",
            "absolute_error",
            "absolute_percentage_error",
            "share_of_year_absolute_error",
        ]
        if node_comparison is None or node_comparison.empty or "absolute_error" not in node_comparison.columns:
            return pd.DataFrame(columns=columns)

        result = node_comparison.copy()
        result["absolute_error"] = pd.to_numeric(result["absolute_error"], errors="coerce")
        if "Year" in result.columns:
            year_total_error = result.groupby("Year")["absolute_error"].transform("sum")
            result["share_of_year_absolute_error"] = self._safe_ratio_series(result["absolute_error"], year_total_error)
        else:
            result["share_of_year_absolute_error"] = np.nan
        result = result.sort_values("absolute_error", ascending=False, na_position="last").head(100)
        return result.reindex(columns=columns)

    def _build_flags(self, frames: dict[str, pd.DataFrame | None]) -> pd.DataFrame:
        flags: list[dict[str, str]] = []
        self._append_data_flags(flags, frames)
        self._append_orientation_flags(flags, frames)
        self._append_production_flags(flags, frames)
        self._append_ei_flags(flags, frames)
        self._append_remaining_optional_file_flags(flags, frames)
        self._append_legacy_flags(flags, frames)
        return pd.DataFrame(flags, columns=["area", "severity", "flag", "evidence", "recommended_action"])

    def _append_data_flags(self, flags: list[dict[str, str]], frames: dict[str, pd.DataFrame | None]) -> None:
        build_report = frames["build_report"]
        if build_report is None:
            self._add_flag(
                flags,
                "data",
                "blocking",
                "Corrected input panel build report is missing.",
                "Expected corrected orientation diagnostics were not found.",
                "Build the corrected input panel diagnostics before simulation readiness assessment.",
            )
            return

        years_built = self._count_built_years(build_report)
        negative_fd_count = self._numeric_sum(build_report, ["negative_Y_no_inventory_count", "negative_FD_no_inventory_entries"])
        missing_ei_count = self._numeric_sum(build_report, ["missing_EI_count", "missing_EI_rows"])
        missing_atlas_count = self._numeric_sum(build_report, ["missing_atlas_rows", "unmatched_merged_labels_count"])
        inventory_excluded = self._numeric_sum(build_report, ["inventory_excluded_column_count"])
        self._add_flag(
            flags,
            "data",
            "info",
            "Corrected input panel diagnostics found.",
            (
                f"years_built={years_built}; missing_EI_rows={missing_ei_count:g}; "
                f"missing_Atlas_or_unmatched_rows={missing_atlas_count:g}; "
                f"inventory_FD_columns_excluded={inventory_excluded:g}; "
                f"negative_corrected_FD_indicators={negative_fd_count:g}"
            ),
            "Use the corrected input panel as the ABM v3 validation target.",
        )

        negative_ei = frames["negative_ei"]
        negative_ei_rows = len(negative_ei) if negative_ei is not None else np.nan
        if negative_ei is None:
            self._missing_optional_flag(flags, "data", "negative EI row diagnostics")
        elif negative_ei_rows > 0:
            self._add_flag(
                flags,
                "data",
                "warning",
                "Negative EI rows are present.",
                f"negative_EI_rows={negative_ei_rows}",
                "Keep these rows visible in diagnostics; do not silently coerce EI signs.",
            )

        smoke_test = frames["smoke_test"]
        if smoke_test is None:
            self._missing_optional_flag(flags, "data", "corrected input panel smoke test")
        elif "passed" in smoke_test.columns and not bool(smoke_test["passed"].fillna(False).all()):
            failed = int((~smoke_test["passed"].fillna(False).astype(bool)).sum())
            self._add_flag(
                flags,
                "data",
                "warning",
                "Corrected input panel smoke tests did not all pass.",
                f"failed_checks={failed}",
                "Inspect failed smoke-test rows before using the panel for scenarios.",
            )

    def _append_orientation_flags(self, flags: list[dict[str, str]], frames: dict[str, pd.DataFrame | None]) -> None:
        orientation = frames["orientation"]
        if orientation is None:
            self._missing_optional_flag(flags, "orientation", "corrected vs old orientation comparison")
            return
        mean_correlation = self._numeric_mean(orientation, "correlation_old_X_corrected_X")
        mean_difference = self._numeric_mean(orientation, "mean_absolute_percentage_difference_X")
        median_difference = self._numeric_mean(orientation, "median_absolute_percentage_difference_X")
        max_difference = self._numeric_max(orientation, "mean_absolute_percentage_difference_X")
        severity = "warning" if np.isfinite(mean_difference) and mean_difference > 0.25 else "info"
        self._add_flag(
            flags,
            "orientation",
            severity,
            "Corrected orientation materially changes output definitions." if severity == "warning" else "Corrected orientation comparison is available.",
            (
                f"mean_correlation={self._format_number(mean_correlation)}; "
                f"mean_abs_pct_difference={self._format_number(mean_difference)}; "
                f"median_abs_pct_difference={self._format_number(median_difference)}; "
                f"max_abs_pct_difference={self._format_number(max_difference)}"
            ),
            "Keep corrected orientation as baseline if data diagnostics remain acceptable.",
        )

    def _append_production_flags(self, flags: list[dict[str, str]], frames: dict[str, pd.DataFrame | None]) -> None:
        summary = frames["behavioural_summary"]
        if summary is None or summary.empty:
            self._add_flag(
                flags,
                "production",
                "blocking",
                "Behavioural Leontief range summary is missing.",
                "No corrected behavioural production diagnostics were found.",
                "Run behavioural Leontief diagnostics against the corrected input panel.",
            )
            return

        mean_relative_error = self._numeric_mean(summary, "relative_error_total")
        mean_ratio = self._mean_output_ratio(summary)
        convergence_share = self._convergence_share(summary)
        mean_final_residual_share = self._numeric_mean(summary, "final_residual_share")

        if np.isfinite(mean_relative_error) and mean_relative_error > 0.25:
            self._add_flag(
                flags,
                "production",
                "warning",
                "Production total relative error is high.",
                f"mean_relative_error_total={mean_relative_error:.6g}",
                "Diagnose production propagation before first scenario runs; do not mask the gap with a calibration scalar.",
            )
        if np.isfinite(mean_ratio) and mean_ratio < 0.90:
            self._add_flag(
                flags,
                "production",
                "warning",
                "Systematic underproduction detected.",
                f"mean_realized_observed_output_ratio={mean_ratio:.6g}",
                "Report the underproduction explicitly; do not rescale X_realized to force observed totals.",
            )
        if np.isfinite(convergence_share) and convergence_share < 0.80:
            severity = "info" if np.isfinite(mean_final_residual_share) and mean_final_residual_share < 1e-6 else "warning"
            self._add_flag(
                flags,
                "production",
                severity,
                "Convergence share is below threshold.",
                (
                    f"convergence_share={convergence_share:.6g}; "
                    f"mean_final_residual_share={self._format_number(mean_final_residual_share)}"
                ),
                "Inspect residual shares and rounds before treating convergence failures as substantive.",
            )
        if not any(flag["area"] == "production" for flag in flags):
            self._add_flag(
                flags,
                "production",
                "info",
                "Behavioural Leontief production diagnostics are within readiness thresholds.",
                (
                    f"mean_relative_error_total={self._format_number(mean_relative_error)}; "
                    f"mean_realized_observed_output_ratio={self._format_number(mean_ratio)}; "
                    f"convergence_share={self._format_number(convergence_share)}"
                ),
                "Use these diagnostics as the ABM v3 production validation baseline.",
            )

        if frames["node_comparison"] is None:
            self._missing_optional_flag(flags, "production", "behavioural node comparison diagnostics")

    def _append_ei_flags(self, flags: list[dict[str, str]], frames: dict[str, pd.DataFrame | None]) -> None:
        sample = frames["ei_sample"]
        if sample is None or sample.empty:
            self._missing_optional_flag(flags, "ei_transition", "EI transition sample report")
            return

        first = sample.iloc[0]
        total_rows = self._numeric_value(first, "total_rows")
        included_rows = self._numeric_value(first, "included_rows")
        excluded_rows = self._numeric_value(first, "excluded_rows")
        included_share = self._numeric_value(first, "included_share")
        if not np.isfinite(included_share):
            included_share = included_rows / total_rows if total_rows > 0 else np.nan

        self._add_flag(
            flags,
            "ei_transition",
            "info",
            "EI transition diagnostics are summarized for context.",
            (
                f"total_rows={self._format_number(total_rows)}; "
                f"included_rows={self._format_number(included_rows)}; "
                f"excluded_rows={self._format_number(excluded_rows)}; "
                f"included_share={self._format_number(included_share)}"
            ),
            "Treat EI transition as diagnostic-only for ABM v3; defer EI redesign to ABM v4.",
        )
        if np.isfinite(included_share) and included_share < 0.50:
            self._add_flag(
                flags,
                "ei_transition",
                "warning",
                "EI transition included share is low.",
                f"included_share={included_share:.6g}",
                "Document EI transition limits; do not make EI transition blocking for ABM v3 production validation.",
            )

        scores = frames["ei_scores"]
        if scores is None:
            self._missing_optional_flag(flags, "ei_transition", "EI transition model scores")
        elif self._green_r2_not_better_than_economic(scores):
            self._add_flag(
                flags,
                "ei_transition",
                "warning",
                "Green transition EI model does not improve R2 over economic-only.",
                self._model_r2_evidence(scores),
                "Keep EI transition improvements postponed to ABM v4 unless priorities change.",
            )

    def _append_legacy_flags(self, flags: list[dict[str, str]], frames: dict[str, pd.DataFrame | None]) -> None:
        existing = [
            name
            for name in ["legacy_reproduction", "legacy_rolling"]
            if frames.get(name) is not None
        ]
        if existing:
            self._add_flag(
                flags,
                "legacy_validation",
                "info",
                "Legacy scalar validation files exist.",
                f"legacy_files_found={', '.join(existing)}",
                "Use legacy scalar results only as context; do not let them drive ABM v3 readiness.",
            )

    def _append_remaining_optional_file_flags(
        self,
        flags: list[dict[str, str]],
        frames: dict[str, pd.DataFrame | None],
    ) -> None:
        optional_inputs = {
            "input_intensity": ("data", "corrected input intensity summary"),
            "ei_sample_by_year": ("ei_transition", "EI transition sample report by year"),
            "ei_scores": ("ei_transition", "EI transition model scores"),
            "ei_expected_signs": ("ei_transition", "EI transition expected-sign diagnostics"),
            "ei_coefficients": ("ei_transition", "EI transition model coefficients"),
        }
        for key, (area, label) in optional_inputs.items():
            flag_text = f"Missing optional {label}."
            already_flagged = any(existing["flag"] == flag_text for existing in flags)
            if frames.get(key) is None and not already_flagged:
                self._missing_optional_flag(flags, area, label)

    def _build_summary(
        self,
        start_year: int,
        end_year: int,
        frames: dict[str, pd.DataFrame | None],
        flags: pd.DataFrame,
    ) -> pd.DataFrame:
        data_status = self._layer_status(flags, "data", ready_label="ready")
        production_status = self._production_status(flags, frames["behavioural_summary"])
        ei_status = self._ei_status(flags)
        overall_status = self._overall_status(data_status, production_status, ei_status)
        recommendation = self._overall_recommendation(overall_status)
        return pd.DataFrame(
            [
                {
                    "start_year": start_year,
                    "end_year": end_year,
                    "data_layer_status": data_status,
                    "production_layer_status": production_status,
                    "ei_transition_status": ei_status,
                    "overall_status": overall_status,
                    "overall_recommendation": recommendation,
                }
            ]
        )

    def _build_markdown(
        self,
        start_year: int,
        end_year: int,
        frames: dict[str, pd.DataFrame | None],
        summary: pd.DataFrame,
        flags: pd.DataFrame,
        by_year: pd.DataFrame,
        top_errors: pd.DataFrame,
    ) -> str:
        row = summary.iloc[0].to_dict()
        lines = [
            f"# ABM v3 Validation Report ({start_year}-{end_year})",
            "",
            "## Readiness",
            "",
            f"- Data layer: {row['data_layer_status']}",
            f"- Production layer: {row['production_layer_status']}",
            f"- EI transition: {row['ei_transition_status']}",
            f"- Overall: {row['overall_status']}",
            f"- Recommendation: {row['overall_recommendation']}",
            "",
            "## Data Validity",
            "",
            self._data_markdown(frames),
            "",
            "## Orientation",
            "",
            self._orientation_markdown(frames),
            "",
            "## Behavioural Leontief Production",
            "",
            self._production_markdown(frames),
            "",
            "## Top Output Errors",
            "",
            self._top_error_markdown(top_errors, frames["node_comparison"]),
            "",
            "## EI Transition",
            "",
            self._ei_markdown(frames),
            "",
            "## Readiness Flags",
            "",
            self._flags_markdown(flags),
            "",
            "## By-Year Production Snapshot",
            "",
            self._table_markdown(by_year.head(10)),
        ]
        return "\n".join(lines) + "\n"

    def _data_markdown(self, frames: dict[str, pd.DataFrame | None]) -> str:
        build_report = frames["build_report"]
        if build_report is None:
            return "Corrected input panel build diagnostics are missing. Data layer is blocked."
        years_built = self._count_built_years(build_report)
        row_count = self._numeric_sum(build_report, ["row_count", "merged_nodes", "raw_nodes"])
        negative_ei_rows = len(frames["negative_ei"]) if frames["negative_ei"] is not None else np.nan
        smoke_passed = self._smoke_passed(frames["smoke_test"])
        inventory_excluded = self._numeric_sum(build_report, ["inventory_excluded_column_count"])
        negative_fd = self._numeric_sum(build_report, ["negative_Y_no_inventory_count", "negative_FD_no_inventory_entries"])
        return (
            f"- Years built: {years_built}\n"
            f"- Total rows/nodes reported: {self._format_number(row_count)}\n"
            f"- Negative EI rows: {self._format_number(negative_ei_rows)}\n"
            f"- Corrected input smoke tests passed: {smoke_passed}\n"
            f"- Inventory FD columns excluded: {self._format_number(inventory_excluded)}\n"
            f"- Corrected FD negative-demand indicators: {self._format_number(negative_fd)}"
        )

    def _orientation_markdown(self, frames: dict[str, pd.DataFrame | None]) -> str:
        orientation = frames["orientation"]
        if orientation is None:
            return "Orientation comparison diagnostics are missing."
        return (
            f"- Mean corrected vs old output correlation: {self._format_number(self._numeric_mean(orientation, 'correlation_old_X_corrected_X'))}\n"
            f"- Mean output definition difference: {self._format_number(self._numeric_mean(orientation, 'mean_absolute_percentage_difference_X'))}\n"
            f"- Median output definition difference: {self._format_number(self._numeric_mean(orientation, 'median_absolute_percentage_difference_X'))}\n"
            f"- Max yearly mean difference: {self._format_number(self._numeric_max(orientation, 'mean_absolute_percentage_difference_X'))}\n"
            "- Recommendation: corrected orientation remains the ABM v3 baseline if diagnostics are acceptable."
        )

    def _production_markdown(self, frames: dict[str, pd.DataFrame | None]) -> str:
        summary = frames["behavioural_summary"]
        if summary is None or summary.empty:
            return "Behavioural Leontief range summary is missing. Production layer is blocked."
        ratio = self._mean_output_ratio(summary)
        underproduction = "yes" if np.isfinite(ratio) and ratio < 0.90 else "no"
        return (
            f"- Years summarized: {len(summary)}\n"
            f"- Relative error total, mean/min/max: {self._triple(summary, 'relative_error_total')}\n"
            f"- Realized vs observed correlation, mean/min/max: {self._triple(summary, 'correlation_realized_vs_observed')}\n"
            f"- Median absolute percentage error, mean/min/max: {self._triple(summary, 'median_absolute_percentage_error')}\n"
            f"- Mean observed output total: {self._format_number(self._numeric_mean(summary, 'observed_output_total'))}\n"
            f"- Mean realized output total: {self._format_number(self._numeric_mean(summary, 'realized_output_total'))}\n"
            f"- Mean realized/observed output ratio: {self._format_number(ratio)}\n"
            f"- Systematic underproduction detected: {underproduction}\n"
            f"- Convergence count/share: {int(self._convergence_count(summary))}/{self._format_number(self._convergence_share(summary))}\n"
            f"- Average final residual share: {self._format_number(self._numeric_mean(summary, 'final_residual_share'))}\n"
            f"- Average mean capacity stress: {self._format_number(self._numeric_mean(summary, 'mean_capacity_stress_over_rounds'))}\n"
            f"- Average capacity binding share: {self._format_number(self._numeric_mean(summary, 'mean_share_capacity_binding'))}\n"
            f"- Average capacity missing share: {self._format_number(self._numeric_mean(summary, 'mean_share_capacity_missing'))}\n"
            "- Note: no calibration scalar is applied or recommended here."
        )

    def _top_error_markdown(self, top_errors: pd.DataFrame, node_comparison: pd.DataFrame | None) -> str:
        if top_errors.empty:
            return "Node comparison diagnostics are missing or contain no absolute_error column."
        sector_table = self._aggregate_top_errors(node_comparison, "Sector")
        country_table = self._aggregate_top_errors(node_comparison, "Country")
        return (
            "Top 100 node-level output errors are written to the CSV report.\n\n"
            "Top sectors by total absolute error:\n\n"
            f"{self._table_markdown(sector_table)}\n\n"
            "Top countries by total absolute error:\n\n"
            f"{self._table_markdown(country_table)}"
        )

    def _ei_markdown(self, frames: dict[str, pd.DataFrame | None]) -> str:
        sample = frames["ei_sample"]
        scores = frames["ei_scores"]
        signs = frames["ei_expected_signs"]
        parts = []
        if sample is None or sample.empty:
            parts.append("EI transition sample diagnostics are missing.")
        else:
            row = sample.iloc[0]
            parts.append(
                "- Sample: "
                f"total_rows={self._format_number(self._numeric_value(row, 'total_rows'))}; "
                f"included_rows={self._format_number(self._numeric_value(row, 'included_rows'))}; "
                f"excluded_rows={self._format_number(self._numeric_value(row, 'excluded_rows'))}; "
                f"included_share={self._format_number(self._numeric_value(row, 'included_share'))}"
            )
        if scores is not None and not scores.empty:
            parts.append("- Model scores:\n\n" + self._table_markdown(self._model_score_table(scores)))
        if signs is not None and not signs.empty:
            match_count = self._expected_sign_match_count(signs, True)
            mismatch_count = self._expected_sign_match_count(signs, False)
            parts.append(f"- Expected signs: matches={match_count}; mismatches={mismatch_count}")
        exclusion_reasons = self._exclusion_reason_markdown(sample)
        if exclusion_reasons:
            parts.append(exclusion_reasons)
        parts.append("EI transition is diagnosed but not blocking ABM v3 production validation. Improvements are deferred to ABM v4.")
        return "\n".join(parts)

    def _flags_markdown(self, flags: pd.DataFrame) -> str:
        if flags.empty:
            return "No flags were generated."
        return self._table_markdown(flags)

    def _aggregate_top_errors(self, node_comparison: pd.DataFrame | None, column: str) -> pd.DataFrame:
        if node_comparison is None or node_comparison.empty or column not in node_comparison.columns:
            return pd.DataFrame(columns=[column, "total_absolute_error"])
        result = node_comparison.copy()
        result["absolute_error"] = pd.to_numeric(result.get("absolute_error"), errors="coerce")
        return (
            result.groupby(column, dropna=False)["absolute_error"]
            .sum()
            .reset_index(name="total_absolute_error")
            .sort_values("total_absolute_error", ascending=False)
            .head(10)
        )

    def _model_score_table(self, scores: pd.DataFrame) -> pd.DataFrame:
        model_column = "model_name" if "model_name" in scores.columns else "model"
        columns = [model_column, "rmse", "mae", "r2", "correlation_predicted_observed", "correlation"]
        return scores.reindex(columns=[column for column in columns if column in scores.columns])

    def _exclusion_reason_markdown(self, sample: pd.DataFrame | None) -> str:
        if sample is None or sample.empty:
            return ""
        reason_columns = [
            column
            for column in sample.columns
            if "exclusion" in str(column).lower() or "excluded_reason" in str(column).lower()
        ]
        if not reason_columns:
            return ""
        rows = []
        for column in reason_columns:
            value = sample[column].iloc[0]
            if pd.notna(value):
                rows.append({"exclusion_reason_metric": column, "value": value})
        if not rows:
            return ""
        return "- Main exclusion reasons if available:\n\n" + self._table_markdown(pd.DataFrame(rows))

    def _table_markdown(self, df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return "No rows available."
        display = df.copy()
        display = display.fillna("")
        columns = [str(column) for column in display.columns]
        rows = []
        rows.append("| " + " | ".join(columns) + " |")
        rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for _, row in display.iterrows():
            values = [str(row[column]).replace("|", "/") for column in display.columns]
            rows.append("| " + " | ".join(values) + " |")
        return "\n".join(rows)

    def _layer_status(self, flags: pd.DataFrame, area: str, ready_label: str) -> str:
        area_flags = flags.loc[flags["area"].eq(area)]
        if area_flags["severity"].eq("blocking").any():
            return "blocked"
        if area_flags["severity"].eq("warning").any():
            return "warning"
        return ready_label

    def _production_status(self, flags: pd.DataFrame, behavioural_summary: pd.DataFrame | None) -> str:
        area_flags = flags.loc[flags["area"].eq("production")]
        if area_flags["severity"].eq("blocking").any() or behavioural_summary is None:
            return "blocked"
        if area_flags["severity"].eq("warning").any():
            return "usable_with_warnings"
        return "ready"

    def _ei_status(self, flags: pd.DataFrame) -> str:
        area_flags = flags.loc[flags["area"].eq("ei_transition")]
        if area_flags["severity"].eq("blocking").any():
            return "blocked"
        if area_flags["severity"].eq("warning").any():
            return "usable_with_warnings"
        return "diagnostic_only"

    def _overall_status(self, data_status: str, production_status: str, ei_status: str) -> str:
        if data_status == "blocked" or production_status == "blocked":
            return "not_yet_simulation_ready"
        if production_status in {"ready", "usable_with_warnings"} and ei_status in {"diagnostic_only", "usable_with_warnings"}:
            return "production_ready_ei_pending"
        if production_status == "ready" and ei_status == "ready":
            return "production_ready_ei_pending"
        return "not_yet_simulation_ready"

    def _overall_recommendation(self, overall_status: str) -> str:
        if overall_status == "production_ready_ei_pending":
            return "Use ABM v3 production diagnostics as the readiness baseline; keep EI transition improvements deferred to ABM v4."
        if overall_status == "ready_for_first_scenarios":
            return "Proceed to first scenarios with current diagnostics attached."
        return "Resolve blocking data or production diagnostics before scenario simulation."

    def _green_r2_not_better_than_economic(self, scores: pd.DataFrame) -> bool:
        model_column = "model_name" if "model_name" in scores.columns else "model" if "model" in scores.columns else None
        if model_column is None or "r2" not in scores.columns:
            return False
        r2_by_model = pd.to_numeric(scores.set_index(model_column)["r2"], errors="coerce")
        if "green_transition" not in r2_by_model.index or "economic_only" not in r2_by_model.index:
            return False
        return bool(r2_by_model.loc["green_transition"] <= r2_by_model.loc["economic_only"])

    def _model_r2_evidence(self, scores: pd.DataFrame) -> str:
        model_column = "model_name" if "model_name" in scores.columns else "model" if "model" in scores.columns else None
        if model_column is None or "r2" not in scores.columns:
            return "R2 columns unavailable."
        subset = scores.loc[scores[model_column].isin(["economic_only", "green_transition"]), [model_column, "r2"]]
        return "; ".join(f"{row[model_column]}_r2={row['r2']}" for _, row in subset.iterrows())

    def _mean_output_ratio(self, summary: pd.DataFrame) -> float:
        if {"realized_output_total", "observed_output_total"}.issubset(summary.columns):
            ratios = self._safe_ratio_series(summary["realized_output_total"], summary["observed_output_total"])
            return float(ratios.mean(skipna=True))
        return np.nan

    def _convergence_count(self, summary: pd.DataFrame) -> int:
        if "converged" not in summary.columns:
            return 0
        return int(summary["converged"].fillna(False).astype(bool).sum())

    def _convergence_share(self, summary: pd.DataFrame) -> float:
        if "converged" not in summary.columns or len(summary) == 0:
            return np.nan
        return float(summary["converged"].fillna(False).astype(bool).mean())

    def _count_built_years(self, build_report: pd.DataFrame) -> int:
        if "status" in build_report.columns:
            return int(build_report["status"].astype(str).str.lower().eq("built").sum())
        if "Year" in build_report.columns:
            return int(build_report["Year"].nunique())
        if "year" in build_report.columns:
            return int(build_report["year"].nunique())
        return int(len(build_report))

    def _smoke_passed(self, smoke_test: pd.DataFrame | None) -> str:
        if smoke_test is None or "passed" not in smoke_test.columns:
            return "unknown"
        return "yes" if bool(smoke_test["passed"].fillna(False).astype(bool).all()) else "no"

    def _expected_sign_match_count(self, signs: pd.DataFrame, expected: bool) -> int:
        for column in ["matches_expected_sign", "expected_sign_match", "sign_match"]:
            if column in signs.columns:
                return int(signs[column].fillna(False).astype(bool).eq(expected).sum())
        return 0

    def _triple(self, df: pd.DataFrame, column: str) -> str:
        return (
            f"{self._format_number(self._numeric_mean(df, column))}/"
            f"{self._format_number(self._numeric_min(df, column))}/"
            f"{self._format_number(self._numeric_max(df, column))}"
        )

    def _numeric_mean(self, df: pd.DataFrame, column: str) -> float:
        if df is None or column not in df.columns:
            return np.nan
        return float(pd.to_numeric(df[column], errors="coerce").mean(skipna=True))

    def _numeric_min(self, df: pd.DataFrame, column: str) -> float:
        if df is None or column not in df.columns:
            return np.nan
        return float(pd.to_numeric(df[column], errors="coerce").min(skipna=True))

    def _numeric_max(self, df: pd.DataFrame, column: str) -> float:
        if df is None or column not in df.columns:
            return np.nan
        return float(pd.to_numeric(df[column], errors="coerce").max(skipna=True))

    def _numeric_sum(self, df: pd.DataFrame, columns: list[str]) -> float:
        values = []
        for column in columns:
            if column in df.columns:
                values.append(pd.to_numeric(df[column], errors="coerce").sum(skipna=True))
        return float(np.nansum(values)) if values else np.nan

    def _numeric_value(self, row: pd.Series, column: str) -> float:
        if column not in row.index:
            return np.nan
        return float(pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0])

    def _safe_ratio_series(self, numerator: Any, denominator: Any) -> pd.Series:
        numerator_numeric = pd.to_numeric(numerator, errors="coerce")
        denominator_numeric = pd.to_numeric(denominator, errors="coerce")
        ratio = numerator_numeric / denominator_numeric.where(denominator_numeric > 0)
        return ratio.replace([np.inf, -np.inf], np.nan)

    def _format_number(self, value: float) -> str:
        if not np.isfinite(value):
            return "unknown"
        return f"{value:.6g}"

    def _missing_optional_flag(self, flags: list[dict[str, str]], area: str, label: str) -> None:
        self._add_flag(
            flags,
            area,
            "warning",
            f"Missing optional {label}.",
            "Input file was not found or could not be read.",
            "Regenerate the diagnostic if this section needs a complete audit trail.",
        )

    def _add_flag(
        self,
        flags: list[dict[str, str]],
        area: str,
        severity: str,
        flag: str,
        evidence: str,
        recommended_action: str,
    ) -> None:
        flags.append(
            {
                "area": area,
                "severity": severity,
                "flag": flag,
                "evidence": evidence,
                "recommended_action": recommended_action,
            }
        )
