from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.abm_v3.capability import add_capability_features
from src.abm_v3.config import ABMV3Config
from src.abm_v3.outputs import ABMV3OutputWriter
from src.abm_v3.paths import ABMV3Paths

LOGGER = logging.getLogger(__name__)


ACCOUNTING_COLUMNS = [
    "intermediate_output",
    "intermediate_demand",
    "final_demand_total",
    "X_observed",
    "D_proxy_observed",
    "M_observed",
    "X",
    "D",
    "M",
    "available_inputs",
]


@dataclass
class ABMV3InputPanelBuilder:
    """Build the canonical ABM-ready historical input panel.

    The panel is a durable data product. It starts from raw Eora accounting
    matrices, preserves every ``labels_T`` country-sector node, and left joins
    Eora-Atlas metrics without interpreting missing Atlas, EI, or green-ness
    values as zero.
    """

    paths: ABMV3Paths
    config: ABMV3Config = field(default_factory=ABMV3Config)
    negative_tolerance: float = 1e-9

    def output_path(self, start_year: int, end_year: int) -> Path:
        return self.paths.abm_v3_historical_panel_file(start_year, end_year)

    def build(self, start_year: int = 1995, end_year: int = 2016, overwrite: bool = False) -> pd.DataFrame:
        output_path = self.output_path(start_year, end_year)
        if output_path.exists() and not overwrite:
            LOGGER.info("ABM v3 input panel already exists at %s", output_path)
            return pd.read_parquet(output_path)

        metrics_panel = self._load_metrics_panel()
        if not metrics_panel.empty:
            metrics_panel = self._canonicalize_metrics_panel(metrics_panel)
            if "Year" in metrics_panel.columns:
                metrics_panel = metrics_panel[
                    (metrics_panel["Year"] >= start_year) & (metrics_panel["Year"] <= end_year)
                ].copy()
        year_panels: list[pd.DataFrame] = []
        report_rows: list[dict[str, object]] = []
        all_raw_labels: set[tuple[int, str]] = set()
        all_merged_labels: set[tuple[int, str]] = set()

        if not metrics_panel.empty and {"Year", "country_sector"}.issubset(metrics_panel.columns):
            all_merged_labels = set(zip(metrics_panel["Year"].astype(int), metrics_panel["country_sector"].astype(str)))

        for year in range(start_year, end_year + 1):
            try:
                accounting_panel, accounting_report = self.build_year_accounting_panel(year)
            except FileNotFoundError as error:
                LOGGER.warning("Skipping %s because a raw matrix or label file is missing: %s", year, error)
                report_rows.append(self._missing_year_report(year, str(error)))
                continue
            except ValueError as error:
                LOGGER.warning("Skipping %s because labels are invalid: %s", year, error)
                report_rows.append(self._failed_year_report(year, str(error)))
                continue

            all_raw_labels.update(zip(accounting_panel["Year"].astype(int), accounting_panel["country_sector"].astype(str)))
            merged_panel = self.merge_with_metrics(accounting_panel, metrics_panel)
            merged_panel, negative_report = self.handle_negative_accounting_values(merged_panel)
            merged_panel = self.add_input_intensity_features(merged_panel)
            merged_panel = self.add_required_aliases(merged_panel)
            year_panels.append(merged_panel)
            report_rows.append(
                {
                    "year": year,
                    "status": "built",
                    "raw_nodes": len(accounting_panel),
                    "merged_nodes": len(merged_panel),
                    "missing_matrix_files": "",
                    "missing_metrics_rows": int(merged_panel["_metrics_matched"].isna().sum()) if "_metrics_matched" in merged_panel.columns else len(merged_panel),
                    "missing_atlas_rows": self._missing_atlas_count(merged_panel),
                    "missing_EI_rows": int(merged_panel["EI"].isna().sum()) if "EI" in merged_panel.columns else len(merged_panel),
                    "missing_g_in_rows": int(merged_panel["g_in"].isna().sum()) if "g_in" in merged_panel.columns else len(merged_panel),
                    "missing_g_out_rows": int(merged_panel["g_out"].isna().sum()) if "g_out" in merged_panel.columns else len(merged_panel),
                    "zero_X_share": float(merged_panel["X"].eq(0).mean()) if "X" in merged_panel.columns else np.nan,
                    "zero_D_share": float(merged_panel["D"].eq(0).mean()) if "D" in merged_panel.columns else np.nan,
                    "zero_M_share": float(merged_panel["M"].eq(0).mean()) if "M" in merged_panel.columns else np.nan,
                    **accounting_report,
                    **negative_report,
                }
            )

        panel = pd.concat(year_panels, ignore_index=True) if year_panels else pd.DataFrame()
        if "_metrics_matched" in panel.columns:
            panel = panel.drop(columns=["_metrics_matched"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(output_path, index=False)
        self.write_build_report(pd.DataFrame(report_rows), start_year, end_year)
        self.write_unmatched_labels(all_raw_labels, all_merged_labels, start_year, end_year)
        self.write_column_dictionary()
        self.write_negative_ei_rows(panel, start_year, end_year)
        self.write_input_intensity_summary(panel, start_year, end_year)
        LOGGER.info("Built ABM v3 input panel with %s rows at %s", len(panel), output_path)
        return panel

    def build_year_accounting_panel(self, year: int) -> tuple[pd.DataFrame, dict[str, int]]:
        matrix_dir = self.paths.parquet_root / str(year)
        t_path = matrix_dir / "T.parquet"
        fd_path = matrix_dir / "FD.parquet"
        labels_path = self.paths.raw_root / str(year) / "labels_T.txt"
        missing = [str(path) for path in [t_path, fd_path, labels_path] if not path.exists()]
        if missing:
            raise FileNotFoundError("; ".join(missing))

        labels = self.load_labels_T(year)
        label_parts = self.split_country_sector_labels(labels)
        t_matrix = pd.read_parquet(t_path)
        fd_matrix = pd.read_parquet(fd_path)
        if len(labels) != len(t_matrix.index) or len(labels) != len(t_matrix.columns) or len(labels) != len(fd_matrix.index):
            raise ValueError(
                f"Label count {len(labels)} does not match T/FD dimensions for year {year}: "
                f"T={t_matrix.shape}, FD={fd_matrix.shape}"
            )

        t_matrix = t_matrix.copy()
        fd_matrix = fd_matrix.copy()
        t_matrix.index = labels
        t_matrix.columns = labels
        fd_matrix.index = labels

        intermediate_output = t_matrix.sum(axis=0)
        intermediate_demand = t_matrix.sum(axis=1)
        final_demand_total = fd_matrix.sum(axis=1)
        result = label_parts.copy()
        result["Year"] = year
        result["intermediate_output"] = intermediate_output.to_numpy(dtype=float)
        result["intermediate_demand"] = intermediate_demand.to_numpy(dtype=float)
        result["final_demand_total"] = final_demand_total.to_numpy(dtype=float)
        result["X_observed"] = result["intermediate_output"] + result["final_demand_total"]
        result["D_proxy_observed"] = result["intermediate_demand"] + result["final_demand_total"]
        result["M_observed"] = result["intermediate_output"]
        result["X"] = result["X_observed"]
        result["D"] = result["D_proxy_observed"]
        result["M"] = result["M_observed"]
        result["available_inputs"] = result["M_observed"]
        capacity_margin = float(self.config.calibration.capacity_margin)
        inventory_days = float(self.config.calibration.inventory_days)
        result["K"] = capacity_margin * result["X_observed"]
        result["I"] = inventory_days * result["M_observed"] / 365.0
        result["capacity_margin_used"] = capacity_margin
        result["inventory_days_used"] = inventory_days

        report = {
            "negative_intermediate_output_count": int((result["intermediate_output"] < 0).sum()),
            "negative_intermediate_demand_count": int((result["intermediate_demand"] < 0).sum()),
            "negative_final_demand_total_count": int((result["final_demand_total"] < 0).sum()),
            "negative_X_count": int((result["X"] < 0).sum()),
            "negative_D_count": int((result["D"] < 0).sum()),
            "negative_M_count": int((result["M"] < 0).sum()),
        }
        return result, report

    def load_labels_T(self, year: int) -> list[str]:
        labels_path = self.paths.raw_root / str(year) / "labels_T.txt"
        if not labels_path.exists():
            raise FileNotFoundError(str(labels_path))
        labels = []
        for line in labels_path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = [part.strip() for part in line.split("\t") if part.strip()]
            labels.append(" | ".join(parts))
        return labels

    def split_country_sector_labels(self, labels: list[str]) -> pd.DataFrame:
        rows = []
        invalid = []
        for label in labels:
            parts = [part.strip() for part in label.split("|")]
            if len(parts) != 4:
                invalid.append(label)
                continue
            rows.append(
                {
                    "country_sector": label,
                    "Country": parts[0],
                    "Country_detail": parts[1],
                    "Category": parts[2],
                    "Sector": parts[3],
                }
            )
        if invalid:
            raise ValueError(f"{len(invalid)} labels do not split into four parts. First invalid: {invalid[0]}")
        return pd.DataFrame(rows)

    def merge_with_metrics(self, accounting_panel: pd.DataFrame, merged_metrics_panel: pd.DataFrame) -> pd.DataFrame:
        if merged_metrics_panel.empty:
            result = accounting_panel.copy()
            result["_metrics_matched"] = False
            return result
        metrics = self._canonicalize_metrics_panel(merged_metrics_panel)
        drop_from_metrics = [
            column
            for column in ["Country", "Country_detail", "Category", "Sector"]
            if column in metrics.columns
        ]
        metrics = metrics.drop(columns=drop_from_metrics)
        metrics["_metrics_matched"] = True
        return accounting_panel.merge(metrics, on=["country_sector", "Year"], how="left")

    def add_required_aliases(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        alias_map = {
            "emissions_intensity": "EI",
            "g_base": "g_local",
            "g_in_network": "g_in",
            "g_out_network": "g_out",
        }
        for source, target in alias_map.items():
            if target not in result.columns and source in result.columns:
                result[target] = pd.to_numeric(result[source], errors="coerce")
        if "g_network" not in result.columns and {"g_in", "g_out"}.issubset(result.columns):
            result["g_network"] = 0.5 * (result["g_in"].astype(float) + result["g_out"].astype(float))
        if "green_capability" not in result.columns:
            try:
                result = add_capability_features(result, self.config.capability)
            except ValueError as error:
                LOGGER.warning("green_capability not created during input build: %s", error)
                result["green_capability"] = np.nan
        if "general_complexity" not in result.columns:
            if "capability_export_weighted_pci" in result.columns:
                result["general_complexity"] = pd.to_numeric(result["capability_export_weighted_pci"], errors="coerce")
            else:
                result["general_complexity"] = np.nan
        if "EI" in result.columns:
            result["emissions_observed"] = result["X_observed"].astype(float) * result["EI"].astype(float)
            negative_ei_mask = pd.to_numeric(result["EI"], errors="coerce") < 0
            result.loc[negative_ei_mask, "emissions_observed"] = np.nan
        else:
            result["EI"] = np.nan
            result["emissions_observed"] = np.nan
        for column in ["g_local", "g_in", "g_out", "g_network"]:
            if column not in result.columns:
                result[column] = np.nan
        return result

    def add_input_intensity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add observed and fallback monetary input-intensity columns."""

        result = df.copy()
        minimum = float(self.config.production_feasibility.minimum_input_intensity)
        result["observed_input_intensity"] = self._safe_input_ratio(
            result["M_observed"],
            result["X_observed"],
        )
        node_valid = self._valid_input_intensity(result["observed_input_intensity"], minimum) & (
            pd.to_numeric(result["M_observed"], errors="coerce") > 0
        )

        result["country_category_input_intensity"] = self._group_input_intensity(
            result,
            ["Country", "Category", "Year"],
            minimum,
        )
        result["country_ecosystem_input_intensity"] = self._group_input_intensity(
            result,
            ["Country", "Year"],
            minimum,
        )
        result["sector_input_intensity"] = self._group_input_intensity(
            result,
            ["Sector", "Year"],
            minimum,
        )
        result["global_input_intensity"] = self._group_input_intensity(
            result,
            ["Year"],
            minimum,
        )

        result["effective_input_intensity"] = np.nan
        result["input_intensity_source"] = "missing"
        fallback_order = [
            ("node", "observed_input_intensity", node_valid),
            (
                "country_category",
                "country_category_input_intensity",
                self._valid_input_intensity(result["country_category_input_intensity"], minimum),
            ),
            (
                "country_ecosystem",
                "country_ecosystem_input_intensity",
                self._valid_input_intensity(result["country_ecosystem_input_intensity"], minimum),
            ),
            (
                "sector",
                "sector_input_intensity",
                self._valid_input_intensity(result["sector_input_intensity"], minimum),
            ),
            (
                "global",
                "global_input_intensity",
                self._valid_input_intensity(result["global_input_intensity"], minimum),
            ),
        ]
        unset = result["effective_input_intensity"].isna()
        for source, column, valid_mask in fallback_order:
            use_mask = unset & valid_mask
            result.loc[use_mask, "effective_input_intensity"] = result.loc[use_mask, column]
            result.loc[use_mask, "input_intensity_source"] = source
            unset = result["effective_input_intensity"].isna()
        return result

    def _safe_input_ratio(self, numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        numerator_numeric = pd.to_numeric(numerator, errors="coerce")
        denominator_numeric = pd.to_numeric(denominator, errors="coerce")
        ratio = numerator_numeric / denominator_numeric.where(denominator_numeric > 0)
        return ratio.replace([np.inf, -np.inf], np.nan)

    def _valid_input_intensity(self, series: pd.Series, minimum: float) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.notna() & np.isfinite(numeric) & (numeric > minimum)

    def _group_input_intensity(
        self,
        df: pd.DataFrame,
        group_cols: list[str],
        minimum: float,
    ) -> pd.Series:
        grouped = df.groupby(group_cols, dropna=False)
        group_m = grouped["M_observed"].transform("sum")
        group_x = grouped["X_observed"].transform("sum")
        ratio = self._safe_input_ratio(group_m, group_x)
        return ratio.where(self._valid_input_intensity(ratio, minimum))

    def handle_negative_accounting_values(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
        result = df.copy()
        tiny_negative_clipped_count = 0
        meaningful_negative_nan_count = 0
        for column in ACCOUNTING_COLUMNS:
            if column not in result.columns:
                continue
            numeric = pd.to_numeric(result[column], errors="coerce")
            tiny_mask = (numeric < 0) & (numeric >= -self.negative_tolerance)
            meaningful_mask = numeric < -self.negative_tolerance
            tiny_negative_clipped_count += int(tiny_mask.sum())
            meaningful_negative_nan_count += int(meaningful_mask.sum())
            numeric = numeric.mask(tiny_mask, 0.0)
            numeric = numeric.mask(meaningful_mask, np.nan)
            result[column] = numeric
        return result, {
            "tiny_negative_clipped_count": tiny_negative_clipped_count,
            "meaningful_negative_nan_count": meaningful_negative_nan_count,
        }

    def write_build_report(self, report: pd.DataFrame, start_year: int, end_year: int) -> Path:
        return ABMV3OutputWriter(self.paths).write_dataframe(
            report,
            "diagnostics",
            f"abm_v3_input_panel_build_report_{start_year}_{end_year}.csv",
        )

    def write_unmatched_labels(
        self,
        raw_labels: set[tuple[int, str]],
        merged_labels: set[tuple[int, str]],
        start_year: int,
        end_year: int,
    ) -> None:
        raw_not_merged = sorted(raw_labels - merged_labels)
        merged_not_raw = sorted(merged_labels - raw_labels)
        writer = ABMV3OutputWriter(self.paths)
        writer.write_dataframe(
            pd.DataFrame(raw_not_merged, columns=["Year", "country_sector"]),
            "diagnostics",
            f"unmatched_raw_eora_labels_{start_year}_{end_year}.csv",
        )
        writer.write_dataframe(
            pd.DataFrame(merged_not_raw, columns=["Year", "country_sector"]),
            "diagnostics",
            f"unmatched_merged_panel_labels_{start_year}_{end_year}.csv",
        )

    def write_column_dictionary(self) -> Path:
        rows = [
            ("country_sector", "Stable full Eora country-sector node label.", "labels_T", "observed", "Primary ABM key."),
            ("intermediate_output", "Intermediate sales/output from T.sum(axis=0).", "T.parquet", "observed", ""),
            ("intermediate_demand", "Intermediate demand from T.sum(axis=1).", "T.parquet", "observed", ""),
            ("final_demand_total", "Final demand total from FD.sum(axis=1).", "FD.parquet", "observed", ""),
            ("X_observed", "Observed production scale.", "T + FD", "observed", "intermediate_output + final_demand_total."),
            ("D_proxy_observed", "Historical demand proxy.", "T + FD", "proxy", "intermediate_demand + final_demand_total."),
            ("M_observed", "Intermediate output/input requirement proxy.", "T.parquet", "proxy", "Equals intermediate_output."),
            ("X", "Model-ready output alias.", "X_observed", "observed", ""),
            ("D", "Model-ready historical demand proxy alias.", "D_proxy_observed", "proxy", "Not true behavioural demand."),
            ("M", "Model-ready intermediate proxy alias.", "M_observed", "proxy", ""),
            ("available_inputs", "Historical input availability proxy.", "M_observed", "proxy", "Not an observed input stock."),
            ("observed_input_intensity", "Observed monetary input intensity.", "M_observed / X_observed", "observed", "Invalid for feasibility when zero or missing."),
            ("country_category_input_intensity", "Country-category fallback input intensity.", "grouped accounting", "proxy", "sum(M_observed) / sum(X_observed)."),
            ("country_ecosystem_input_intensity", "Country ecosystem fallback input intensity.", "grouped accounting", "proxy", "Country-Year grouped ratio."),
            ("sector_input_intensity", "Global sector fallback input intensity.", "grouped accounting", "proxy", "Sector-Year grouped ratio."),
            ("global_input_intensity", "Global fallback input intensity.", "grouped accounting", "proxy", "Year grouped ratio."),
            ("effective_input_intensity", "Input intensity used for scalar feasibility.", "fallback hierarchy", "proxy", "Source recorded in input_intensity_source."),
            ("input_intensity_source", "Fallback source used for effective input intensity.", "fallback hierarchy", "proxy", "node/country_category/country_ecosystem/sector/global/missing."),
            ("K", "Capacity proxy.", "X_observed", "proxy", "capacity_margin * X_observed."),
            ("I", "Inventory proxy.", "M_observed", "proxy", "inventory_days * M_observed / 365."),
            ("EI", "Emissions intensity.", "metrics panel", "observed", "Missing values preserved."),
            ("g_local", "Local green-ness.", "metrics panel", "observed/proxy", "Alias from g_base when needed."),
            ("g_in", "Incoming network green-ness.", "metrics panel", "observed/proxy", "Alias from g_in_network when needed."),
            ("g_out", "Outgoing network green-ness.", "metrics panel", "observed/proxy", "Alias from g_out_network when needed."),
            ("g_network", "Average network green-ness.", "g_in/g_out", "proxy", "0.5 * (g_in + g_out)."),
            ("green_capability", "Atlas green capability proxy.", "Atlas capability", "proxy", "Missing Atlas values preserved."),
            ("general_complexity", "Atlas general complexity proxy.", "Atlas capability", "proxy", "From capability_export_weighted_pci when needed."),
            ("emissions_observed", "Observed emissions identity.", "X_observed * EI", "observed/proxy", "NaN if EI is missing."),
        ]
        dictionary = pd.DataFrame(rows, columns=["column", "description", "source", "observed_or_proxy", "notes"])
        return ABMV3OutputWriter(self.paths).write_dataframe(
            dictionary,
            "diagnostics",
            "abm_v3_input_panel_column_dictionary.csv",
        )

    def _load_metrics_panel(self) -> pd.DataFrame:
        path = self.paths.eora_atlas_merged_file
        if not path.exists():
            LOGGER.warning("Merged Eora-Atlas metrics panel missing at %s; building accounting-only panel.", path)
            return pd.DataFrame()
        return pd.read_parquet(path)

    def _canonicalize_metrics_panel(self, metrics: pd.DataFrame) -> pd.DataFrame:
        result = metrics.copy()
        if "country_sector" not in result.columns and {"Country", "Country_detail", "Category", "Sector"}.issubset(result.columns):
            result["country_sector"] = (
                result["Country"].astype(str)
                + " | "
                + result["Country_detail"].astype(str)
                + " | "
                + result["Category"].astype(str)
                + " | "
                + result["Sector"].astype(str)
            )
        if "country_sector" not in result.columns:
            raise ValueError("Metrics panel lacks country_sector and cannot construct it from label components.")
        return result

    def _missing_year_report(self, year: int, missing: str) -> dict[str, object]:
        return {
            "year": year,
            "status": "missing_raw_files",
            "raw_nodes": 0,
            "merged_nodes": 0,
            "missing_matrix_files": missing,
        }

    def _failed_year_report(self, year: int, message: str) -> dict[str, object]:
        return {
            "year": year,
            "status": f"failed: {message}",
            "raw_nodes": 0,
            "merged_nodes": 0,
            "missing_matrix_files": "",
        }

    def _missing_atlas_count(self, df: pd.DataFrame) -> int:
        atlas_columns = ["green_capability", "general_complexity", "green_capability_export_share"]
        available = [column for column in atlas_columns if column in df.columns]
        if not available:
            return len(df)
        return int(df[available].isna().all(axis=1).sum())

    def write_negative_ei_rows(self, panel: pd.DataFrame, start_year: int, end_year: int) -> Path:
        columns = ["Year", "country_sector", "Country", "Sector", "X_observed", "EI", "emissions_observed"]
        if "EI" not in panel.columns:
            rows = pd.DataFrame(columns=columns)
        else:
            negative_mask = pd.to_numeric(panel["EI"], errors="coerce") < 0
            rows = panel.loc[negative_mask, columns].copy() if negative_mask.any() else pd.DataFrame(columns=columns)
        return ABMV3OutputWriter(self.paths).write_dataframe(
            rows,
            "diagnostics",
            f"negative_ei_rows_{start_year}_{end_year}.csv",
        )

    def write_input_intensity_summary(self, panel: pd.DataFrame, start_year: int, end_year: int) -> Path:
        rows = []
        if panel.empty:
            summary = pd.DataFrame(columns=self._input_intensity_summary_columns())
        else:
            for year, year_panel in panel.groupby("Year", dropna=False):
                rows.append(self._input_intensity_summary_row(year_panel, int(year)))
            summary = pd.DataFrame(rows, columns=self._input_intensity_summary_columns())
        return ABMV3OutputWriter(self.paths).write_dataframe(
            summary,
            "diagnostics",
            f"input_intensity_summary_{start_year}_{end_year}.csv",
        )

    def _input_intensity_summary_row(self, df: pd.DataFrame, year: int) -> dict[str, object]:
        observed = pd.to_numeric(df["observed_input_intensity"], errors="coerce")
        effective = pd.to_numeric(df["effective_input_intensity"], errors="coerce")
        valid_observed = self._valid_input_intensity(
            observed,
            float(self.config.production_feasibility.minimum_input_intensity),
        ) & (pd.to_numeric(df["M_observed"], errors="coerce") > 0)
        zero_m_positive_x = (
            pd.to_numeric(df["M_observed"], errors="coerce").eq(0)
            & (pd.to_numeric(df["X_observed"], errors="coerce") > 0)
        )
        source = df["input_intensity_source"]
        node_count = len(df)
        return {
            "Year": year,
            "node_count": node_count,
            "valid_node_ratio_count": int(valid_observed.sum()),
            "invalid_node_ratio_count": int((~valid_observed & observed.notna()).sum()),
            "missing_node_ratio_count": int(observed.isna().sum()),
            "zero_M_positive_X_count": int(zero_m_positive_x.sum()),
            "median_observed_input_intensity": float(observed.median(skipna=True)),
            "p05_observed_input_intensity": float(observed.quantile(0.05)),
            "p95_observed_input_intensity": float(observed.quantile(0.95)),
            "p99_observed_input_intensity": float(observed.quantile(0.99)),
            "max_observed_input_intensity": float(observed.max(skipna=True)),
            "median_effective_input_intensity": float(effective.median(skipna=True)),
            "share_a_above_1": float((effective > 1).mean()),
            "share_a_above_2": float((effective > 2).mean()),
            "share_a_above_5": float((effective > 5).mean()),
            "node_ratio_used_share": float(source.eq("node").mean()),
            "country_category_fallback_share": float(source.eq("country_category").mean()),
            "country_ecosystem_fallback_share": float(source.eq("country_ecosystem").mean()),
            "sector_fallback_share": float(source.eq("sector").mean()),
            "global_fallback_share": float(source.eq("global").mean()),
            "missing_feasibility_share": float(source.eq("missing").mean()),
        }

    def _input_intensity_summary_columns(self) -> list[str]:
        return [
            "Year",
            "node_count",
            "valid_node_ratio_count",
            "invalid_node_ratio_count",
            "missing_node_ratio_count",
            "zero_M_positive_X_count",
            "median_observed_input_intensity",
            "p05_observed_input_intensity",
            "p95_observed_input_intensity",
            "p99_observed_input_intensity",
            "max_observed_input_intensity",
            "median_effective_input_intensity",
            "share_a_above_1",
            "share_a_above_2",
            "share_a_above_5",
            "node_ratio_used_share",
            "country_category_fallback_share",
            "country_ecosystem_fallback_share",
            "sector_fallback_share",
            "global_fallback_share",
            "missing_feasibility_share",
        ]
