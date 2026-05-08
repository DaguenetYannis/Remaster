from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.abm_v3.paths import ABMV3Paths

UNDERPRODUCTION_THRESHOLD = 0.95

IDENTIFICATION_COLUMNS = [
    "Year",
    "country_sector",
    "Country",
    "Country_detail",
    "Category",
    "Sector",
    "validation_mode",
    "train_start_year",
    "train_end_year",
    "validation_year",
    "sigma",
    "ei_mode",
]

PRODUCTION_STAGE_COLUMNS = [
    "planned_output",
    "D",
    "K",
    "desired_output",
    "input_availability",
    "adjusted_input_availability",
    "substitution_gain",
    "effective_input_intensity",
    "input_intensity_source",
    "input_feasible_output",
    "input_stress_ratio",
    "input_stress_factor",
    "input_constraint_penalty",
    "input_constraint_binding",
    "input_feasibility_missing",
    "realized_output",
]

NODE_REPORT_COLUMNS = [
    *IDENTIFICATION_COLUMNS,
    "X_observed_validation",
    "X_simulated",
    "output_gap",
    "output_ratio",
    *PRODUCTION_STAGE_COLUMNS,
    "planned_to_observed_ratio",
    "demand_to_observed_ratio",
    "capacity_to_observed_ratio",
    "desired_to_observed_ratio",
    "input_feasible_to_observed_ratio",
    "realized_to_observed_ratio",
    "planning_below_observed",
    "demand_below_observed",
    "capacity_below_observed",
    "desired_below_observed",
    "input_below_desired",
    "dominant_bottleneck",
]


def safe_divide(numerator: object, denominator: object) -> pd.Series:
    """Divide numeric values while treating missing or non-positive denominators as undefined."""

    numerator_series = pd.Series(numerator, dtype="float64")
    denominator_series = pd.Series(denominator, dtype="float64")
    numerator_aligned, denominator_aligned = numerator_series.align(denominator_series)
    result = numerator_aligned / denominator_aligned
    invalid_denominator = denominator_aligned.isna() | (denominator_aligned <= 0)
    result = result.mask(invalid_denominator)
    return result.replace([np.inf, -np.inf], np.nan)


@dataclass
class ProductionBottleneckReporter:
    """Build inspectable reports that locate production-stage output losses."""

    paths: ABMV3Paths
    underproduction_threshold: float = UNDERPRODUCTION_THRESHOLD

    def build_node_report(
        self,
        predicted: pd.DataFrame,
        observed: pd.DataFrame,
        validation_year: int,
        split_metadata: dict | None = None,
    ) -> pd.DataFrame:
        """Compare predicted production stages against observed validation output."""

        split_metadata = split_metadata or {}
        predicted_prepared = self._prepare_predicted(predicted, validation_year)
        observed_prepared = self._prepare_observed(observed, validation_year)
        merged = predicted_prepared.merge(
            observed_prepared,
            on=["country_sector", "Year"],
            how="left",
            suffixes=("_predicted", "_observed"),
        )
        report = pd.DataFrame(index=merged.index)
        report["Year"] = merged["Year"]
        report["country_sector"] = merged["country_sector"]

        for column in ["Country", "Country_detail", "Category", "Sector"]:
            report[column] = self._first_available_column(
                merged,
                [f"{column}_observed", f"{column}_predicted", column],
            )
        for column, value in split_metadata.items():
            report[column] = value
        report["validation_year"] = report.get("validation_year", validation_year)

        report["X_observed_validation"] = merged["X_observed_validation"]
        report["X_simulated"] = merged["X_simulated"]
        report["output_gap"] = report["X_simulated"] - report["X_observed_validation"]
        report["output_ratio"] = safe_divide(report["X_simulated"], report["X_observed_validation"])

        for column in PRODUCTION_STAGE_COLUMNS:
            report[column] = merged[column] if column in merged.columns else np.nan
        if "input_intensity_source" in report.columns:
            report["input_intensity_source"] = report["input_intensity_source"].fillna("missing")

        observed_output = report["X_observed_validation"]
        report["planned_to_observed_ratio"] = safe_divide(report["planned_output"], observed_output)
        report["demand_to_observed_ratio"] = safe_divide(report["D"], observed_output)
        report["capacity_to_observed_ratio"] = safe_divide(report["K"], observed_output)
        report["desired_to_observed_ratio"] = safe_divide(report["desired_output"], observed_output)
        report["input_feasible_to_observed_ratio"] = safe_divide(report["input_feasible_output"], observed_output)
        report["realized_to_observed_ratio"] = safe_divide(report["realized_output"], observed_output)

        report["planning_below_observed"] = self._below(report["planned_output"], observed_output)
        report["demand_below_observed"] = self._below(report["D"], observed_output)
        report["capacity_below_observed"] = self._below(report["K"], observed_output)
        report["desired_below_observed"] = self._below(report["desired_output"], observed_output)
        report["input_below_desired"] = self._below(report["input_feasible_output"], report["desired_output"])
        report["input_constraint_binding"] = self._boolean_column(report["input_constraint_binding"])
        report["input_feasibility_missing"] = self._boolean_column(report["input_feasibility_missing"])
        report["dominant_bottleneck"] = report.apply(self._classify_dominant_bottleneck, axis=1)

        for column in NODE_REPORT_COLUMNS:
            if column not in report.columns:
                report[column] = np.nan
        return report[NODE_REPORT_COLUMNS]

    def build_aggregate_report(
        self,
        node_report: pd.DataFrame,
        group_cols: list[str],
    ) -> pd.DataFrame:
        """Aggregate node diagnostics to year, sector-year, or country-year level."""

        if node_report.empty:
            return pd.DataFrame(columns=group_cols)

        rows = []
        for group_key, group in node_report.groupby(group_cols, dropna=False):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            row = dict(zip(group_cols, group_key, strict=False))
            observed_output = group["X_observed_validation"]
            simulated_output = group["X_simulated"]

            row["node_count"] = int(len(group))
            row["matched_node_count"] = int((observed_output.notna() & simulated_output.notna()).sum())
            row["zero_observed_output_count"] = int((observed_output == 0).sum())
            row["missing_observed_output_count"] = int(observed_output.isna().sum())
            row["observed_output_total"] = self._sum_numeric(observed_output)
            row["simulated_output_total"] = self._sum_numeric(simulated_output)
            row["planned_output_total"] = self._sum_numeric(group["planned_output"])
            row["demand_total"] = self._sum_numeric(group["D"])
            row["capacity_total"] = self._sum_numeric(group["K"])
            row["desired_output_total"] = self._sum_numeric(group["desired_output"])
            row["input_feasible_output_total"] = self._sum_numeric(group["input_feasible_output"])
            row["realized_output_total"] = self._sum_numeric(group["realized_output"])
            row["output_gap_total"] = row["simulated_output_total"] - row["observed_output_total"]
            row["output_ratio_total"] = self._safe_scalar_divide(row["simulated_output_total"], row["observed_output_total"])
            row["absolute_output_error_total"] = self._sum_numeric((simulated_output - observed_output).abs())
            row["mean_output_ratio"] = float(group["output_ratio"].mean())
            row["median_output_ratio"] = float(group["output_ratio"].median())
            row["planned_to_observed_total_ratio"] = self._safe_scalar_divide(row["planned_output_total"], row["observed_output_total"])
            row["demand_to_observed_total_ratio"] = self._safe_scalar_divide(row["demand_total"], row["observed_output_total"])
            row["capacity_to_observed_total_ratio"] = self._safe_scalar_divide(row["capacity_total"], row["observed_output_total"])
            row["desired_to_observed_total_ratio"] = self._safe_scalar_divide(row["desired_output_total"], row["observed_output_total"])
            row["input_feasible_to_observed_total_ratio"] = self._safe_scalar_divide(
                row["input_feasible_output_total"],
                row["observed_output_total"],
            )
            row["realized_to_observed_total_ratio"] = self._safe_scalar_divide(
                row["realized_output_total"],
                row["observed_output_total"],
            )

            for source_column, output_column in [
                ("planning_below_observed", "share_planning_below_observed"),
                ("demand_below_observed", "share_demand_below_observed"),
                ("capacity_below_observed", "share_capacity_below_observed"),
                ("desired_below_observed", "share_desired_below_observed"),
                ("input_below_desired", "share_input_below_desired"),
                ("input_constraint_binding", "share_input_constraint_binding"),
                ("input_feasibility_missing", "share_input_feasibility_missing"),
            ]:
                row[output_column] = self._share(group[source_column])

            for bottleneck, output_column in [
                ("planning", "share_bottleneck_planning"),
                ("demand", "share_bottleneck_demand"),
                ("capacity", "share_bottleneck_capacity"),
                ("input", "share_bottleneck_input"),
                ("missing_input_feasibility", "share_bottleneck_missing_input_feasibility"),
                ("no_underproduction", "share_no_underproduction"),
                ("unknown", "share_bottleneck_unknown"),
            ]:
                row[output_column] = float((group["dominant_bottleneck"] == bottleneck).mean())

            row["mean_input_stress_factor"] = float(group["input_stress_factor"].mean())
            row["median_input_stress_factor"] = float(group["input_stress_factor"].median())
            row["total_input_constraint_penalty"] = self._sum_numeric(group["input_constraint_penalty"])
            row["total_substitution_gain"] = self._sum_numeric(group["substitution_gain"])
            row["mean_effective_input_intensity"] = float(group["effective_input_intensity"].mean())

            source = group["input_intensity_source"].fillna("missing")
            for value, output_column in [
                ("node", "share_input_intensity_source_node"),
                ("country_category", "share_input_intensity_source_country_category"),
                ("country_ecosystem", "share_input_intensity_source_country_ecosystem"),
                ("sector", "share_input_intensity_source_sector"),
                ("global", "share_input_intensity_source_global"),
                ("missing", "share_input_intensity_source_missing"),
            ]:
                row[output_column] = float((source == value).mean())
            rows.append(row)
        return pd.DataFrame(rows)

    def write_reports(
        self,
        node_reports: list[pd.DataFrame],
        prefix: str = "rolling_validation",
    ) -> dict[str, Path]:
        """Write node and aggregate bottleneck diagnostics under data/abm_v3/diagnostics."""

        diagnostics_dir = self.paths.abm_v3_output_root / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        node_report = pd.concat(node_reports, ignore_index=True) if node_reports else pd.DataFrame(columns=NODE_REPORT_COLUMNS)

        if prefix == "rolling_validation":
            filenames = {
                "nodes": "production_bottleneck_nodes.csv",
                "by_year": "production_bottleneck_by_year.csv",
                "by_sector_year": "production_bottleneck_by_sector_year.csv",
                "by_country_year": "production_bottleneck_by_country_year.csv",
            }
        else:
            filenames = {
                "nodes": f"production_bottleneck_{prefix}_nodes.csv",
                "by_year": f"production_bottleneck_{prefix}_by_year.csv",
                "by_sector_year": f"production_bottleneck_{prefix}_by_sector_year.csv",
                "by_country_year": f"production_bottleneck_{prefix}_by_country_year.csv",
            }

        by_year = self.build_aggregate_report(node_report, ["Year"])
        by_sector_year = self.build_aggregate_report(node_report, ["Year", "Sector"])
        by_country_year = self.build_aggregate_report(node_report, ["Year", "Country"])
        reports = {
            "nodes": node_report,
            "by_year": by_year,
            "by_sector_year": by_sector_year,
            "by_country_year": by_country_year,
        }
        paths = {}
        for key, dataframe in reports.items():
            path = diagnostics_dir / filenames[key]
            dataframe.to_csv(path, index=False)
            paths[key] = path
        return paths

    def write_recursive_by_year_report(
        self,
        simulated: pd.DataFrame,
        observed: pd.DataFrame,
        split_metadata: dict | None = None,
    ) -> Path:
        """Write a year-level bottleneck summary for recursive historical reproduction."""

        node_reports = []
        for year in sorted(set(simulated["Year"]).intersection(set(observed["Year"]))):
            if split_metadata and year < split_metadata.get("start_year", year):
                continue
            predicted_year = simulated[simulated["Year"] == year].copy()
            observed_year = observed[observed["Year"] == year].copy()
            metadata = dict(split_metadata or {})
            metadata["validation_year"] = year
            node_reports.append(self.build_node_report(predicted_year, observed_year, year, metadata))
        node_report = pd.concat(node_reports, ignore_index=True) if node_reports else pd.DataFrame(columns=NODE_REPORT_COLUMNS)
        by_year = self.build_aggregate_report(node_report, ["Year"])
        diagnostics_dir = self.paths.abm_v3_output_root / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        path = diagnostics_dir / "production_bottleneck_recursive_by_year.csv"
        by_year.to_csv(path, index=False)
        return path

    def _prepare_predicted(self, predicted: pd.DataFrame, validation_year: int) -> pd.DataFrame:
        prepared = predicted.copy()
        if "Year" not in prepared.columns:
            prepared["Year"] = validation_year
        if "X_simulated" not in prepared.columns:
            prepared["X_simulated"] = prepared["X"] if "X" in prepared.columns else np.nan
        if "realized_output" not in prepared.columns and "X_simulated" in prepared.columns:
            prepared["realized_output"] = prepared["X_simulated"]
        for column in PRODUCTION_STAGE_COLUMNS:
            if column not in prepared.columns:
                prepared[column] = np.nan
        return prepared

    def _prepare_observed(self, observed: pd.DataFrame, validation_year: int) -> pd.DataFrame:
        prepared = observed.copy()
        if "Year" not in prepared.columns:
            prepared["Year"] = validation_year
        if "X_observed_validation" not in prepared.columns:
            if "X_observed" in prepared.columns:
                prepared["X_observed_validation"] = prepared["X_observed"]
            elif "X" in prepared.columns:
                prepared["X_observed_validation"] = prepared["X"]
            else:
                prepared["X_observed_validation"] = np.nan
        keep_columns = [
            column
            for column in ["country_sector", "Year", "X_observed_validation", "Country", "Country_detail", "Category", "Sector"]
            if column in prepared.columns
        ]
        return prepared[keep_columns]

    def _classify_dominant_bottleneck(self, row: pd.Series) -> str:
        observed = row["X_observed_validation"]
        simulated = row["X_simulated"]
        threshold = self.underproduction_threshold
        if pd.isna(observed) or observed <= 0 or pd.isna(simulated):
            return "unknown"
        if simulated >= threshold * observed:
            return "no_underproduction"
        if bool(row.get("input_feasibility_missing", False)):
            return "missing_input_feasibility"
        if self._below_threshold(row.get("planned_output"), observed):
            return "planning"
        if self._below_threshold(row.get("D"), observed):
            return "demand"
        if self._below_threshold(row.get("K"), observed):
            return "capacity"
        desired = row.get("desired_output")
        input_feasible = row.get("input_feasible_output")
        if pd.notna(desired) and desired > 0 and self._below_threshold(input_feasible, desired):
            return "input"
        return "unknown"

    def _below(self, numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        result = numerator.astype(float) < denominator.astype(float)
        result = result & denominator.notna() & numerator.notna()
        return result.map(bool).astype(object)

    def _boolean_column(self, series: pd.Series) -> pd.Series:
        return series.fillna(False).map(bool).astype(object)

    def _below_threshold(self, value: object, reference: object) -> bool:
        if pd.isna(value) or pd.isna(reference):
            return False
        return float(value) < self.underproduction_threshold * float(reference)

    def _safe_scalar_divide(self, numerator: float, denominator: float) -> float:
        if pd.isna(numerator) or pd.isna(denominator) or denominator <= 0:
            return float("nan")
        result = numerator / denominator
        if np.isinf(result):
            return float("nan")
        return float(result)

    def _sum_numeric(self, series: pd.Series) -> float:
        result = series.astype(float).sum(min_count=1)
        if pd.isna(result):
            return float("nan")
        return float(result)

    def _share(self, series: pd.Series) -> float:
        if len(series) == 0:
            return float("nan")
        return float(series.fillna(False).map(bool).mean())

    def _first_available_column(self, dataframe: pd.DataFrame, candidates: list[str]) -> pd.Series:
        for column in candidates:
            if column in dataframe.columns:
                return dataframe[column]
        return pd.Series(np.nan, index=dataframe.index)
