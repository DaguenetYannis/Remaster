from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.abm_v3.diagnostics.collapse import detect_bad_transition
from src.abm_v3.diagnostics.decomposition import emissions_decomposition
from src.abm_v3.outputs import ABMV3OutputWriter
from src.abm_v3.paths import ABMV3Paths


def _safe_log(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return np.log(numeric.where(numeric > 0))


def _linear_regression_table(
    df: pd.DataFrame,
    target: str,
    variables: list[str],
    sector_fixed_effects: bool = True,
) -> pd.DataFrame:
    """Fit an OLS diagnostic regression and return coefficient table."""

    work = df.copy()
    if "log_EI_lag1" not in work.columns and "EI" in work.columns:
        work["log_EI_lag1"] = _safe_log(work["EI"])
    columns = [target, *variables]
    if sector_fixed_effects and "Sector" in work.columns:
        columns.append("Sector")
    work = work[columns].dropna()
    if work.empty:
        return pd.DataFrame(columns=["term", "coefficient"])

    design = work[variables].astype(float).copy()
    if sector_fixed_effects and "Sector" in work.columns:
        sector_dummies = pd.get_dummies(work["Sector"], prefix="Sector", drop_first=True, dtype=float)
        design = pd.concat([design.reset_index(drop=True), sector_dummies.reset_index(drop=True)], axis=1)
    design.insert(0, "intercept", 1.0)
    y = work[target].to_numpy(dtype=float)
    x = design.to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return pd.DataFrame({"term": list(design.columns), "coefficient": beta})


@dataclass
class HypothesisReportGenerator:
    """Generate ABM v3 tables linked to the theoretical priors document."""

    paths: ABMV3Paths

    def output_writer(self) -> ABMV3OutputWriter:
        return ABMV3OutputWriter(self.paths)

    def green_capability_ei_reduction(self, panel: pd.DataFrame, write: bool = True) -> pd.DataFrame:
        variables = [
            column
            for column in ["green_capability", "log_EI_lag1", "general_complexity"]
            if column in panel.columns or column == "log_EI_lag1"
        ]
        table = _linear_regression_table(panel, "delta_log_EI_next", variables)
        if write:
            self.output_writer().write_dataframe(
                table,
                "diagnostics",
                "green_capability_ei_reduction_regression.csv",
            )
            self._plot_binned_relationship(
                panel,
                "green_capability",
                "delta_log_EI_next",
                "green_capability_ei_reduction_binned.png",
            )
        return table

    def network_greenness_ei_reduction(self, panel: pd.DataFrame, write: bool = True) -> pd.DataFrame:
        rows = []
        for greenness_col in ["g_in", "g_out", "g_network"]:
            if greenness_col not in panel.columns:
                continue
            variables = [
                column
                for column in [greenness_col, "log_EI_lag1", "green_capability", "general_complexity"]
                if column in panel.columns or column == "log_EI_lag1"
            ]
            table = _linear_regression_table(panel, "delta_log_EI_next", variables)
            table.insert(0, "greenness_variable", greenness_col)
            rows.append(table)
        result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if write:
            self.output_writer().write_dataframe(
                result,
                "diagnostics",
                "network_greenness_ei_reduction_regression.csv",
            )
        return result

    def local_vs_network_greenness_yearly(self, panel: pd.DataFrame, write: bool = True) -> pd.DataFrame:
        result = panel.copy()
        if "g_network" not in result.columns and {"g_in", "g_out"}.issubset(result.columns):
            result["g_network"] = 0.5 * (result["g_in"].astype(float) + result["g_out"].astype(float))
        columns = [column for column in ["g_local", "g_in", "g_out", "g_network"] if column in result.columns]
        yearly = result.groupby("Year", as_index=False)[columns].mean()
        if write:
            self.output_writer().write_dataframe(
                yearly,
                "diagnostics",
                "local_vs_network_greenness_yearly.csv",
            )
            self._plot_lines(yearly, "Year", columns, "local_vs_network_greenness_yearly.png")
        return yearly

    def emissions_decomposition_yearly(self, panel: pd.DataFrame, write: bool = True) -> pd.DataFrame:
        if not {"country_sector", "Year", "X", "EI"}.issubset(panel.columns):
            result = pd.DataFrame(columns=["Year", "ei_effect", "output_effect", "interaction_effect"])
            if write:
                self.output_writer().write_dataframe(result, "diagnostics", "emissions_decomposition_yearly.csv")
            return result
        rows = []
        years = sorted(pd.to_numeric(panel["Year"], errors="coerce").dropna().astype(int).unique())
        for previous_year, current_year in zip(years[:-1], years[1:]):
            previous = panel[panel["Year"] == previous_year]
            current = panel[panel["Year"] == current_year]
            _node_table, summary = emissions_decomposition(previous, current)
            rows.append({"Year": current_year, **summary})
        result = pd.DataFrame(rows)
        if write:
            self.output_writer().write_dataframe(result, "diagnostics", "emissions_decomposition_yearly.csv")
            self._plot_lines(
                result,
                "Year",
                ["ei_effect", "output_effect", "interaction_effect"],
                "emissions_decomposition_yearly.png",
            )
        return result

    def bad_transition_report(self, panel: pd.DataFrame, write: bool = True) -> pd.DataFrame:
        if not {"Year", "X", "EI"}.issubset(panel.columns):
            result = pd.DataFrame(columns=["Year", "bad_transition", "output_loss_fraction"])
            if write:
                self.output_writer().write_dataframe(result, "diagnostics", "bad_transition_report.csv")
            return result
        rows = []
        years = sorted(pd.to_numeric(panel["Year"], errors="coerce").dropna().astype(int).unique())
        for previous_year, current_year in zip(years[:-1], years[1:]):
            previous = panel[panel["Year"] == previous_year]
            current = panel[panel["Year"] == current_year]
            rows.append({"Year": current_year, **detect_bad_transition(previous, current)})
        result = pd.DataFrame(rows)
        if write:
            self.output_writer().write_dataframe(result, "diagnostics", "bad_transition_report.csv")
        return result

    def substitution_resilience_report(self, sigma_results: pd.DataFrame, write: bool = True) -> pd.DataFrame:
        if write:
            self.output_writer().write_dataframe(sigma_results, "diagnostics", "sigma_grid_results.csv")
            self._plot_lines(
                sigma_results,
                "sigma",
                ["output_validation_loss"],
                "sigma_vs_output_validation_loss.png",
            )
        return sigma_results.copy()

    def write_all(self, panel: pd.DataFrame, sigma_results: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
        reports = {
            "green_capability": self.green_capability_ei_reduction(panel),
            "network_greenness": self.network_greenness_ei_reduction(panel),
            "local_vs_network": self.local_vs_network_greenness_yearly(panel),
            "emissions_decomposition": self.emissions_decomposition_yearly(panel),
            "bad_transition": self.bad_transition_report(panel),
        }
        if sigma_results is not None:
            reports["substitution_resilience"] = self.substitution_resilience_report(sigma_results)
        return reports

    def _plot_binned_relationship(self, df: pd.DataFrame, x_col: str, y_col: str, filename: str) -> None:
        if x_col not in df.columns or y_col not in df.columns:
            return
        import matplotlib.pyplot as plt

        work = df[[x_col, y_col]].dropna().copy()
        if work.empty:
            return
        work["bin"] = pd.qcut(work[x_col], q=min(10, len(work)), duplicates="drop")
        binned = work.groupby("bin", observed=False).agg(x=(x_col, "mean"), y=(y_col, "mean")).reset_index()
        self._plot_dir().mkdir(parents=True, exist_ok=True)
        plt.figure()
        plt.plot(binned["x"], binned["y"], marker="o")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.tight_layout()
        plt.savefig(self._plot_dir() / filename)
        plt.close()

    def _plot_lines(self, df: pd.DataFrame, x_col: str, y_cols: list[str], filename: str) -> None:
        available = [column for column in y_cols if column in df.columns]
        if x_col not in df.columns or not available or df.empty:
            return
        import matplotlib.pyplot as plt

        self._plot_dir().mkdir(parents=True, exist_ok=True)
        plt.figure()
        for column in available:
            plt.plot(df[x_col], df[column], marker="o", label=column)
        plt.xlabel(x_col)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self._plot_dir() / filename)
        plt.close()

    def _plot_dir(self) -> Path:
        return self.paths.abm_v3_output_root / "plots"
