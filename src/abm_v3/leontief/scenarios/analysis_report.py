from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.abm_v3.leontief.scenarios import plots
from src.abm_v3.leontief.scenarios.registry import list_behavioural_scenarios
from src.abm_v3.paths import ABMV3Paths

DEFAULT_MODE = "transpose_row_output_fd_without_inventory"
DEFAULT_ORIENTATION = "transpose_row_fd_without_inventory"

SCENARIO_INTERPRETATIONS = {
    "low_ei_node_demand_expansion_10": "Demand shock to currently low-emissions-intensity nodes.",
    "green_capability_node_demand_expansion_10": "Demand shock to nodes with high green productive capability.",
    "clean_and_capable_node_demand_expansion_10": "Demand shock to nodes that are both low-EI and high green-capability.",
    "transition_pivot_node_demand_expansion_10": "Demand shock to high-EI but high green-capability nodes.",
    "high_ei_node_capacity_bottleneck_10": "Exogenous capacity stress test on high-EI nodes.",
}


@dataclass
class BehaviouralScenarioAnalysisReportBuilder:
    """Build tables, Markdown, and plots from existing behavioural scenario CSV outputs."""

    paths: ABMV3Paths
    mode: str = DEFAULT_MODE
    input_panel_orientation: str = DEFAULT_ORIENTATION
    audience: str = "both"
    color_mode: str = "default"
    make_plots: bool = True

    def build(self, start_year: int = 1995, end_year: int = 2016) -> dict[str, Path]:
        """Consolidate yearly behavioural scenario outputs into an analysis report."""
        summary_raw = self._load_discovered_csvs(self.paths.behavioural_leontief_scenario_diagnostics_dir, "summary_*.csv", "summary")
        selected_raw = self._load_discovered_csvs(self.paths.behavioural_leontief_scenario_selected_nodes_dir, "selected_nodes_*.csv", "selected_nodes")
        aggregate_raw = self._load_discovered_csvs(self.paths.behavioural_leontief_scenario_diagnostics_dir, "aggregate_*.csv", "aggregate")
        if summary_raw.empty:
            raise FileNotFoundError(f"No scenario summary CSV files found in {self.paths.behavioural_leontief_scenario_diagnostics_dir}")
        if selected_raw.empty:
            raise FileNotFoundError(f"No selected-node CSV files found in {self.paths.behavioural_leontief_scenario_selected_nodes_dir}")

        summary_raw = self._filter_years(summary_raw, start_year, end_year)
        selected_raw = self._filter_years(selected_raw, start_year, end_year)
        aggregate_raw = self._filter_years(aggregate_raw, start_year, end_year)
        by_year = self._build_by_year(summary_raw)
        summary = self._build_summary(by_year)
        selector_overlap = self._build_selector_overlap(selected_raw)
        sector_effects = self._build_effects(aggregate_raw, "Sector")
        country_effects = self._build_effects(aggregate_raw, "Country")
        flags = self._build_flags(start_year, end_year, by_year, selected_raw, summary)
        markdown = self._build_markdown(start_year, end_year, summary, selector_overlap, sector_effects, country_effects, flags)

        self._ensure_output_dirs()
        written = {
            "summary": self.paths.behavioural_scenario_analysis_summary_path(start_year, end_year),
            "by_year": self.paths.behavioural_scenario_analysis_by_year_path(start_year, end_year),
            "selector_overlap": self.paths.behavioural_scenario_analysis_selector_overlap_path(start_year, end_year),
            "sector_effects": self.paths.behavioural_scenario_analysis_sector_effects_path(start_year, end_year),
            "country_effects": self.paths.behavioural_scenario_analysis_country_effects_path(start_year, end_year),
            "flags": self.paths.behavioural_scenario_analysis_flags_path(start_year, end_year),
            "markdown": self.paths.behavioural_scenario_analysis_markdown_path(start_year, end_year),
        }
        summary.to_csv(written["summary"], index=False)
        by_year.to_csv(written["by_year"], index=False)
        selector_overlap.to_csv(written["selector_overlap"], index=False)
        sector_effects.to_csv(written["sector_effects"], index=False)
        country_effects.to_csv(written["country_effects"], index=False)
        flags.to_csv(written["flags"], index=False)
        written["markdown"].write_text(markdown, encoding="utf-8")
        if self.make_plots:
            written.update(self._write_plots(start_year, end_year, summary, by_year, selector_overlap, sector_effects, country_effects))
        return written

    def _load_discovered_csvs(self, directory: Path, pattern: str, file_kind: str) -> pd.DataFrame:
        frames = []
        for path in sorted(directory.glob(pattern)):
            try:
                frame = pd.read_csv(path)
            except Exception as error:
                raise ValueError(f"Could not read {file_kind} file {path}: {error}") from error
            if "scenario_name" not in frame.columns:
                frame["scenario_name"] = self._scenario_from_file(path.name, file_kind)
            if "Year" not in frame.columns:
                year = self._year_from_file(path.name)
                if year is not None:
                    frame["Year"] = year
            frame["_source_file"] = str(path)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _build_by_year(self, summary_raw: pd.DataFrame) -> pd.DataFrame:
        df = summary_raw.copy()
        numeric_columns = [
            "Year", "selected_node_count", "total_node_count", "pct_delta_realized_output_total",
            "delta_realized_output_total", "baseline_realized_output_total", "scenario_realized_output_total",
            "pct_delta_desired_output_total", "baseline_final_residual_share", "scenario_final_residual_share",
        ]
        for column in numeric_columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
        if "pct_delta_desired_output_total" not in df.columns and {"delta_desired_output_total", "baseline_desired_output_total"}.issubset(df.columns):
            df["pct_delta_desired_output_total"] = self._safe_ratio(df["delta_desired_output_total"], df["baseline_desired_output_total"])
        if "total_node_count" in df.columns:
            df["selected_node_share"] = self._safe_ratio(df["selected_node_count"], df["total_node_count"])
        else:
            df["selected_node_share"] = np.nan
        df["is_capacity_bottleneck"] = df["scenario_name"].astype(str).str.contains("capacity_bottleneck")
        df["is_demand_expansion"] = ~df["is_capacity_bottleneck"]
        df["output_effect_sign"] = np.where(df["pct_delta_realized_output_total"] > 0, "positive", np.where(df["pct_delta_realized_output_total"] < 0, "negative", "near_zero"))
        df["scenario_rank_by_pct_delta_output_within_year"] = df.groupby("Year")["pct_delta_realized_output_total"].rank(ascending=False, method="min")
        return df

    def _build_summary(self, by_year: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for scenario_name, group in by_year.groupby("scenario_name", dropna=False):
            row = {
                "scenario_name": scenario_name,
                "scenario_type": "capacity_bottleneck" if "capacity_bottleneck" in str(scenario_name) else "demand_expansion",
                "selector_interpretation": SCENARIO_INTERPRETATIONS.get(str(scenario_name), ""),
                "years": int(group["Year"].nunique()),
                "mean_selected_node_count": self._mean(group, "selected_node_count"),
                "min_selected_node_count": self._min(group, "selected_node_count"),
                "max_selected_node_count": self._max(group, "selected_node_count"),
                "mean_pct_delta_realized_output_total": self._mean(group, "pct_delta_realized_output_total"),
                "min_pct_delta_realized_output_total": self._min(group, "pct_delta_realized_output_total"),
                "max_pct_delta_realized_output_total": self._max(group, "pct_delta_realized_output_total"),
                "mean_delta_realized_output_total": self._mean(group, "delta_realized_output_total"),
                "mean_baseline_realized_output_total": self._mean(group, "baseline_realized_output_total"),
                "mean_scenario_realized_output_total": self._mean(group, "scenario_realized_output_total"),
                "mean_pct_delta_desired_output_total": self._mean(group, "pct_delta_desired_output_total"),
                "baseline_converged_years": int(group.get("baseline_converged", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                "scenario_converged_years": int(group.get("scenario_converged", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                "mean_baseline_final_residual_share": self._mean(group, "baseline_final_residual_share"),
                "mean_scenario_final_residual_share": self._mean(group, "scenario_final_residual_share"),
                "interpretation_short": self._short_interpretation(str(scenario_name), group),
            }
            rows.append(row)
        return pd.DataFrame(rows).sort_values("mean_pct_delta_realized_output_total", ascending=False)

    def _build_selector_overlap(self, selected_raw: pd.DataFrame) -> pd.DataFrame:
        required_bool = ["is_low_EI", "is_high_EI", "is_high_green_capability_export_share", "is_clean_and_capable", "is_transition_pivot"]
        rows = []
        for scenario_name, group in selected_raw.groupby("scenario_name", dropna=False):
            row = {"scenario_name": scenario_name, "rows": len(group), "years": int(group["Year"].nunique()), "mean_nodes_per_year": float(group.groupby("Year")["country_sector"].nunique().mean())}
            for column in required_bool:
                values = group[column].fillna(False).astype(bool) if column in group.columns else pd.Series([False] * len(group))
                key = column.removeprefix("is_")
                if key == "high_green_capability_export_share":
                    key = "high_green_capability"
                row[f"{key}_count"] = int(values.sum())
                row[f"{key}_share"] = float(values.mean()) if len(values) else np.nan
            row["mean_EI"] = self._mean(group, "EI")
            row["median_EI"] = self._median(group, "EI")
            row["mean_green_capability_metric_value"] = self._mean(group, "green_capability_metric_value")
            row["median_green_capability_metric_value"] = self._median(group, "green_capability_metric_value")
            row["capability_metric_used_values"] = ", ".join(sorted(group.get("green_capability_metric_used", pd.Series(dtype=str)).dropna().astype(str).unique()))
            rows.append(row)
        return pd.DataFrame(rows)

    def _build_effects(self, aggregate_raw: pd.DataFrame, level: str) -> pd.DataFrame:
        if aggregate_raw.empty:
            return pd.DataFrame()
        df = aggregate_raw.loc[aggregate_raw["aggregation_level"].eq(level)].copy()
        if df.empty:
            return pd.DataFrame()
        df["delta_X_realized_sum"] = pd.to_numeric(df["delta_X_realized_sum"], errors="coerce")
        grouped = df.groupby(["scenario_name", "aggregation_key"], dropna=False).agg(
            mean_delta_X_realized_sum=("delta_X_realized_sum", "mean"),
            mean_pct_delta_X_realized_sum=("pct_delta_X_realized_sum", "mean"),
            total_delta_X_realized_sum=("delta_X_realized_sum", "sum"),
            mean_selected_node_count=("selected_node_count", "mean"),
            years_present=("Year", "nunique"),
        ).reset_index().rename(columns={"aggregation_key": level})
        grouped["rank_within_scenario_by_abs_effect"] = grouped.groupby("scenario_name")["total_delta_X_realized_sum"].transform(lambda s: s.abs().rank(ascending=False, method="min"))
        return grouped.sort_values(["scenario_name", "rank_within_scenario_by_abs_effect"])

    def _build_flags(self, start_year: int, end_year: int, by_year: pd.DataFrame, selected_raw: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
        flags = []
        expected_years = set(range(start_year, end_year + 1))
        for scenario_name in list_behavioural_scenarios():
            observed_years = set(pd.to_numeric(by_year.loc[by_year["scenario_name"].eq(scenario_name), "Year"], errors="coerce").dropna().astype(int))
            missing = sorted(expected_years - observed_years)
            if missing:
                self._flag(flags, "completeness", "warning", "Missing scenario-year files.", f"scenario={scenario_name}; missing_years={missing}", "Regenerate missing yearly scenario runs before final interpretation.")
            if len(observed_years) < len(expected_years):
                self._flag(flags, "completeness", "warning", "Scenario has fewer than expected years.", f"scenario={scenario_name}; years={len(observed_years)}; expected={len(expected_years)}", "Treat range summaries as incomplete.")
        zero_selected = by_year.loc[pd.to_numeric(by_year.get("selected_node_count"), errors="coerce").fillna(0).eq(0)]
        for _, row in zero_selected.iterrows():
            self._flag(flags, "selection", "blocking", "Scenario-year has zero selected nodes.", f"scenario={row['scenario_name']}; year={row['Year']}", "Inspect selector diagnostics.")
        for _, row in summary.iterrows():
            effect = float(row["mean_pct_delta_realized_output_total"])
            if row["scenario_type"] == "demand_expansion" and effect < 0:
                self._flag(flags, "effect", "warning", "Demand expansion scenario has negative mean output effect.", f"scenario={row['scenario_name']}; mean_effect={effect:.6g}", "Inspect selected nodes and propagation diagnostics.")
            if row["scenario_type"] == "capacity_bottleneck" and effect > 0.01:
                self._flag(flags, "effect", "warning", "Capacity bottleneck scenario has large positive output effect.", f"scenario={row['scenario_name']}; mean_effect={effect:.6g}", "Inspect capacity-shock diagnostics.")
            if row["scenario_type"] == "capacity_bottleneck" and abs(effect) < 1e-4:
                self._flag(flags, "effect", "info", "Capacity bottleneck scenario has near-zero effect.", f"scenario={row['scenario_name']}; mean_effect={effect:.6g}", "Interpret as evidence that ABM v3 capacity is weakly binding.")
        if "baseline_final_residual_share" not in by_year.columns or "scenario_final_residual_share" not in by_year.columns:
            self._flag(flags, "convergence", "warning", "Missing residual fields in scenario summaries.", "Residual share columns unavailable.", "Regenerate scenario summaries if convergence interpretation is needed.")
        else:
            bad = by_year.loc[(~by_year.get("baseline_converged", True).astype(bool) | ~by_year.get("scenario_converged", True).astype(bool))]
            for _, row in bad.iterrows():
                self._flag(flags, "convergence", "warning", "Strict convergence failure.", f"scenario={row['scenario_name']}; year={row['Year']}; baseline_residual={row.get('baseline_final_residual_share')}; scenario_residual={row.get('scenario_final_residual_share')}", "Inspect residual shares before using the scenario-year.")
        missing_selected_columns = [c for c in ["is_low_EI", "is_high_EI", "is_high_green_capability_export_share", "is_clean_and_capable", "is_transition_pivot"] if c not in selected_raw.columns]
        if missing_selected_columns:
            self._flag(flags, "selection", "warning", "Missing selected-node diagnostic columns.", f"missing={missing_selected_columns}", "Regenerate selected-node diagnostics for full selector interpretation.")
        self._flag(flags, "interpretation", "info", "No EI transition interpretation.", "ABM v3 scenarios are production-network perturbations only.", "Do not interpret scenario effects as emissions reductions or endogenous green transition.")
        return pd.DataFrame(flags, columns=["area", "severity", "flag", "evidence", "recommended_action"])

    def _build_markdown(self, start_year: int, end_year: int, summary: pd.DataFrame, selector: pd.DataFrame, sector: pd.DataFrame, country: pd.DataFrame, flags: pd.DataFrame) -> str:
        return "\n".join([
            f"# ABM v3 Behavioural Scenario Analysis Report ({start_year}-{end_year})",
            "",
            "## Scope",
            f"This report consolidates 5 scenarios across {end_year - start_year + 1} years as single-year comparative production-network perturbation experiments. They are not forecasts, do not include EI transition, do not include adaptive capacity, and do not optimize policy.",
            "",
            "## Scenario Ranking",
            self._table(summary[["scenario_name", "scenario_type", "years", "mean_selected_node_count", "mean_pct_delta_realized_output_total", "interpretation_short"]]),
            "",
            "Broad low-EI demand expansion will often have the largest effect when it selects many nodes. Capability-focused and transition-pivot scenarios are narrower diagnostics. A weak high-EI capacity bottleneck effect means ABM v3 capacity is mostly a feasibility guardrail.",
            "",
            "## Selector Logic",
            "Low EI means current production is relatively low-carbon intensive. High green capability is a productive capability proxy, not direct environmental performance. Clean-and-capable is their overlap; transition-pivot means high EI but high capability.",
            self._table(selector),
            "",
            "## Production Propagation",
            "This report uses summary and aggregate files. Selected vs non-selected node propagation is not computed unless node-comparison files are loaded in a later analysis pass.",
            "",
            "## Sector Effects",
            self._top_effect_tables(sector, "Sector"),
            "",
            "## Country Effects",
            self._top_effect_tables(country, "Country"),
            "",
            "## Capacity Bottleneck Caveat",
            "If the high-EI capacity bottleneck has weak output effect, this indicates capacity is weakly binding in ABM v3. Adaptive capacity is deferred to ABM v4.",
            "",
            "## Readiness for Visual Portfolio",
            "Good portfolio candidates are the scenario output effect comparison, selector overlap chart, and top sector effects for green capability and transition-pivot scenarios.",
            "",
            "## Caveats",
            "These are not forecasts, do not measure emissions reduction, do not prove green transition, defer EI transition, and treat capacity bottlenecks as exogenous stress tests.",
            "",
            "## Flags",
            self._table(flags),
        ]) + "\n"

    def _write_plots(self, start_year: int, end_year: int, summary: pd.DataFrame, by_year: pd.DataFrame, selector: pd.DataFrame, sector: pd.DataFrame, country: pd.DataFrame) -> dict[str, Path]:
        written = {}
        audiences = ["portfolio", "research"] if self.audience == "both" else [self.audience]
        for audience in audiences:
            ext = "png" if audience == "portfolio" else "svg"
            plot_dir = self.paths.behavioural_leontief_scenario_plot_dir
            items = [
                ("plot_output_effect_" + audience, plot_dir / f"scenario_output_effect_global_{start_year}_{end_year}_{audience}.{ext}", plots.plot_scenario_output_effect, (summary,)),
                ("plot_output_trajectory_" + audience, plot_dir / f"scenario_output_trajectory_global_{start_year}_{end_year}_{audience}.{ext}", plots.plot_scenario_output_trajectory, (by_year,)),
                ("plot_selector_overlap_" + audience, plot_dir / f"scenario_selector_overlap_{start_year}_{end_year}_{audience}.{ext}", plots.plot_selector_overlap, (selector,)),
            ]
            for key, path, fn, args in items:
                fig = fn(*args, audience=audience, color_mode=self.color_mode, output_path=path)
                written[key] = path
                import matplotlib.pyplot as plt
                plt.close(fig)
        for scenario_key, scenario_name in [
            ("green_capability", "green_capability_node_demand_expansion_10"),
            ("transition_pivot", "transition_pivot_node_demand_expansion_10"),
            ("low_ei", "low_ei_node_demand_expansion_10"),
        ]:
            for level_name, frame, fn in [("sector", sector, plots.plot_top_sector_effects), ("country", country, plots.plot_top_country_effects)]:
                path = self.paths.behavioural_leontief_scenario_plot_dir / f"scenario_{level_name}_effects_{scenario_key}_{start_year}_{end_year}_portfolio.png"
                fig = fn(frame, scenario_name, audience="portfolio", color_mode=self.color_mode, output_path=path)
                written[f"plot_{level_name}_{scenario_key}_portfolio"] = path
                import matplotlib.pyplot as plt
                plt.close(fig)
        return written

    def _ensure_output_dirs(self) -> None:
        for directory in [self.paths.behavioural_leontief_scenario_analysis_tables_dir, self.paths.behavioural_leontief_scenario_analysis_markdown_dir, self.paths.behavioural_leontief_scenario_analysis_diagnostics_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def _filter_years(self, df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
        if df.empty or "Year" not in df.columns:
            return df
        years = pd.to_numeric(df["Year"], errors="coerce")
        return df.loc[years.between(start_year, end_year)].copy()

    def _scenario_from_file(self, filename: str, file_kind: str) -> str:
        prefix = f"{file_kind}_"
        name = filename.removeprefix(prefix).removesuffix(".csv")
        match = re.search(r"_(?:19|20)\d{2}(?:_|$)", name)
        return name[: match.start()] if match else name

    def _year_from_file(self, filename: str) -> int | None:
        match = re.search(r"(?:19|20)\d{2}", filename)
        return int(match.group(0)) if match else None

    def _short_interpretation(self, scenario_name: str, group: pd.DataFrame) -> str:
        effect = self._mean(group, "pct_delta_realized_output_total")
        return f"{SCENARIO_INTERPRETATIONS.get(scenario_name, 'Scenario perturbation')} Mean output effect={effect:.6g}."

    def _top_effect_tables(self, df: pd.DataFrame, key: str) -> str:
        if df.empty:
            return "No aggregate rows available."
        blocks = []
        for scenario_name, group in df.groupby("scenario_name"):
            top = group.sort_values("rank_within_scenario_by_abs_effect").head(10)
            blocks.append(f"### {scenario_name}\n\n{self._table(top[[key, 'total_delta_X_realized_sum', 'mean_pct_delta_X_realized_sum', 'years_present']])}")
        return "\n\n".join(blocks)

    def _table(self, df: pd.DataFrame) -> str:
        if df.empty:
            return "No rows available."
        display = df.fillna("").copy()
        rows = ["| " + " | ".join(map(str, display.columns)) + " |", "| " + " | ".join(["---"] * len(display.columns)) + " |"]
        for _, row in display.iterrows():
            rows.append("| " + " | ".join(str(row[c]).replace("|", "/") for c in display.columns) + " |")
        return "\n".join(rows)

    def _flag(self, flags: list[dict[str, str]], area: str, severity: str, flag: str, evidence: str, action: str) -> None:
        flags.append({"area": area, "severity": severity, "flag": flag, "evidence": evidence, "recommended_action": action})

    def _safe_ratio(self, numerator, denominator) -> pd.Series:
        den = pd.to_numeric(denominator, errors="coerce")
        return (pd.to_numeric(numerator, errors="coerce") / den.where(den != 0)).replace([np.inf, -np.inf], np.nan)

    def _mean(self, df: pd.DataFrame, column: str) -> float:
        return float(pd.to_numeric(df[column], errors="coerce").mean()) if column in df.columns else np.nan

    def _median(self, df: pd.DataFrame, column: str) -> float:
        return float(pd.to_numeric(df[column], errors="coerce").median()) if column in df.columns else np.nan

    def _min(self, df: pd.DataFrame, column: str) -> float:
        return float(pd.to_numeric(df[column], errors="coerce").min()) if column in df.columns else np.nan

    def _max(self, df: pd.DataFrame, column: str) -> float:
        return float(pd.to_numeric(df[column], errors="coerce").max()) if column in df.columns else np.nan
