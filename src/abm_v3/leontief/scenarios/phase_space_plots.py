from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.abm_v3.leontief.scenarios.plots import clean_display_label
from src.abm_v3.paths import ABMV3Paths


STATE_COLUMNS = [
    "country_sector",
    "Year",
    "Country",
    "Sector",
    "green_capability_export_share",
    "g_local",
    "g_in_network",
    "g_out_network",
    "log_X_observed",
    "X_observed",
    "emissions_observed",
    "EI",
    "is_top25_by_output_over_period",
    "is_top25_by_emissions_over_period",
]
AXIS_VARIABLES = [
    "green_capability_export_share",
    "g_local",
    "g_in_network",
    "g_out_network",
    "log_total_realized_output",
    "total_realized_output",
    "total_emissions_observed",
    "mean_EI",
]
DEFAULT_COLORS = ["#2f6f8f", "#6a994e", "#bc6c25", "#5f6c7b", "#7b2cbf", "#8d6e63", "#3a5a40"]
COLORBLIND_COLORS = ["#0072B2", "#009E73", "#E69F00", "#CC79A7", "#999999", "#56B4E9", "#D55E00"]
MARKERS = ["o", "s", "^", "D", "P", "X", "v"]

TITLE_REGISTRY = {
    "global_green_local": {
        "portfolio": "Scenarios differ in whether they shift the system toward greener productive positions",
        "research": "Scenario trajectories in green capability and local green-ness, 1995-2016",
    },
    "global_output_local": {
        "portfolio": "Scenario output gains do not automatically imply greener production states",
        "research": "Scenario trajectories in production scale and local green-ness, 1995-2016",
    },
    "global_incoming": {
        "portfolio": "Upstream green exposure changes unevenly across scenario perturbations",
        "research": "Scenario trajectories in green capability and incoming network green-ness, 1995-2016",
    },
    "global_outgoing": {
        "portfolio": "Downstream green embedding remains harder to shift than local green-ness",
        "research": "Scenario trajectories in green capability and outgoing network green-ness, 1995-2016",
    },
    "endpoint": {
        "portfolio": "Scenario endpoints separate green repositioning from production expansion",
        "research": "Final-year scenario endpoints in phase-space coordinates",
    },
    "delta": {
        "portfolio": "Most scenario differences are output responses before they are structural green transitions",
        "research": "Scenario endpoint deltas relative to the selected reference",
    },
    "time_series": {
        "portfolio": "Scenario response paths bend differently across output and green-state variables",
        "research": "Scenario time-series comparison for output-weighted phase-space variables",
    },
    "sector": {
        "portfolio": "Sector pathways reveal capability-constrained transition responses",
        "research": "Sector-level scenario trajectories in phase-space coordinates",
    },
    "node": {
        "portfolio": "Major country-sector nodes do not share a single scenario transition path",
        "research": "Selected country-sector scenario trajectories in phase-space coordinates",
    },
    "capacity": {
        "portfolio": "Capacity stress barely propagates under the current ABM v3 capacity proxy",
        "research": "Capacity-bottleneck scenario response in phase-space diagnostics",
    },
}

CAPTIONS = {
    "global": "Scenario trajectories use historical state coordinates joined to behavioural Leontief scenario output responses. They are production-network perturbation diagnostics, not forecasts.",
    "endpoint": "Endpoints compare final-year output-weighted scenario positions. If no true baseline scenario exists, deltas use the historical endpoint reference.",
    "delta": "Deltas distinguish output response from green-readiness movement. Green-state changes mainly reflect output reweighting across historical node states.",
    "sector": "Filtered sector trajectories show response channels. Selection is documented in the manifest and CSV outputs.",
    "node": "Filtered node trajectories show heterogeneous response among major country-sector nodes. Display labels are shortened only in plots.",
    "diagnostic": "These figures are diagnostic summaries of behavioural Leontief perturbations, not causal green-transition rules.",
}


@dataclass(frozen=True)
class ScenarioPhaseSpaceSources:
    """Authoritative source paths used by the scenario phase-space layer."""

    scenario_summary: Path
    scenario_by_year: Path
    scenario_sector_effects: Path
    scenario_country_effects: Path
    scenario_node_comparison_dir: Path
    historical_state_panel: Path


@dataclass
class ScenarioPhaseSpacePlotBuilder:
    """Build scenario phase-space summaries and plots from existing scenario outputs."""

    paths: ABMV3Paths
    state_panel: Path | None = None
    scenario_names: list[str] | None = None
    reference_scenario: str = "historical_or_baseline"
    title_mode: str = "interpretive"
    top_sector_n: int = 8
    top_node_n: int = 10
    research_top_node_n: int = 25
    mark_years: tuple[int, ...] = (1995, 2000, 2008, 2016)
    write_diagnostics: bool = True
    make_plots: bool = True
    color_mode: str = "default"

    def build(self, start_year: int = 1995, end_year: int = 2016) -> dict[str, Path]:
        """Build scenario phase-space tables, diagnostics, plots, and README."""
        sources = discover_scenario_phase_space_sources(self.paths, start_year, end_year, self.state_panel)
        self._active_start_year = start_year
        self._active_end_year = end_year
        diagnostics: list[dict[str, object]] = []
        summary = _read_csv(sources.scenario_summary, diagnostics, "scenario-level summary")
        by_year = _read_csv(sources.scenario_by_year, diagnostics, "scenario-year summary")
        sector_effects = _read_csv(sources.scenario_sector_effects, diagnostics, "scenario-sector effects")
        state_panel = pd.read_parquet(sources.historical_state_panel, columns=[c for c in STATE_COLUMNS if c in _parquet_columns(sources.historical_state_panel)])
        _record(diagnostics, "source", "historical_state_panel", "Loaded historical phase-space state variables.", str(sources.historical_state_panel))
        for required in ["green_capability_export_share", "g_local", "g_in_network", "g_out_network", "X_observed"]:
            if required not in state_panel.columns:
                _record(diagnostics, "missing_variable", required, "Expected phase-space state variable is missing.", str(sources.historical_state_panel))

        scenario_names = self._active_scenarios(by_year)
        self.scenario_names = scenario_names
        reference = resolve_reference_scenario(scenario_names, self.reference_scenario)
        node_panel = build_scenario_node_panel(sources.scenario_node_comparison_dir, state_panel, scenario_names, start_year, end_year, diagnostics)
        time_series = build_scenario_time_series(node_panel, by_year)
        endpoint = build_endpoint_summary(time_series, end_year, reference)
        delta = build_delta_summary(endpoint, reference)
        sector_summary = build_sector_summary(node_panel, sector_effects, self.top_sector_n)
        node_summary = build_node_summary(node_panel, self.top_node_n, self.research_top_node_n)

        self.paths.scenario_phase_space_dir.mkdir(parents=True, exist_ok=True)
        self.paths.scenario_phase_space_plot_dir.mkdir(parents=True, exist_ok=True)
        written = self._write_tables(start_year, end_year, endpoint, delta, time_series, sector_summary, node_summary, diagnostics)

        manifest_rows: list[dict[str, object]] = []
        if self.make_plots:
            manifest_rows.extend(self._write_plots(start_year, end_year, time_series, endpoint, delta, sector_summary, node_summary, reference, sources))
        manifest = pd.DataFrame(manifest_rows, columns=manifest_columns())
        manifest.to_csv(self.paths.scenario_phase_space_manifest_path(start_year, end_year), index=False)
        recommendations = build_figure_recommendations(manifest)
        recommendations.to_csv(self.paths.scenario_phase_space_figure_recommendations_path(start_year, end_year), index=False)
        readme = build_readme(start_year, end_year, sources, reference, diagnostics, manifest)
        self.paths.scenario_phase_space_readme_path(start_year, end_year).write_text(readme, encoding="utf-8")

        written.update(
            {
                "manifest": self.paths.scenario_phase_space_manifest_path(start_year, end_year),
                "figure_recommendations": self.paths.scenario_phase_space_figure_recommendations_path(start_year, end_year),
                "readme": self.paths.scenario_phase_space_readme_path(start_year, end_year),
            }
        )
        return written

    def _active_scenarios(self, by_year: pd.DataFrame) -> list[str]:
        if self.scenario_names:
            return list(self.scenario_names)
        if "scenario_name" not in by_year.columns:
            return []
        return sorted(by_year["scenario_name"].dropna().astype(str).unique())

    def _write_tables(
        self,
        start_year: int,
        end_year: int,
        endpoint: pd.DataFrame,
        delta: pd.DataFrame,
        time_series: pd.DataFrame,
        sector_summary: pd.DataFrame,
        node_summary: pd.DataFrame,
        diagnostics: list[dict[str, object]],
    ) -> dict[str, Path]:
        paths = {
            "endpoint_summary": self.paths.scenario_phase_space_endpoint_summary_path(start_year, end_year),
            "delta_summary": self.paths.scenario_phase_space_delta_summary_path(start_year, end_year),
            "time_series": self.paths.scenario_phase_space_time_series_path(start_year, end_year),
            "sector_summary": self.paths.scenario_phase_space_sector_summary_path(start_year, end_year),
            "node_summary": self.paths.scenario_phase_space_node_summary_path(start_year, end_year),
            "diagnostics": self.paths.scenario_phase_space_diagnostics_path(start_year, end_year),
        }
        endpoint.to_csv(paths["endpoint_summary"], index=False)
        delta.to_csv(paths["delta_summary"], index=False)
        time_series.to_csv(paths["time_series"], index=False)
        sector_summary.to_csv(paths["sector_summary"], index=False)
        node_summary.to_csv(paths["node_summary"], index=False)
        pd.DataFrame(diagnostics).to_csv(paths["diagnostics"], index=False)
        return paths

    def _write_plots(
        self,
        start_year: int,
        end_year: int,
        time_series: pd.DataFrame,
        endpoint: pd.DataFrame,
        delta: pd.DataFrame,
        sector_summary: pd.DataFrame,
        node_summary: pd.DataFrame,
        reference: str,
        sources: ScenarioPhaseSpaceSources,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        plot_specs = [
            ("global_green_local", "green_capability_export_share", "g_local", "thesis-core"),
            ("global_output_local", "log_total_realized_output", "g_local", "thesis-core"),
            ("global_incoming", "green_capability_export_share", "g_in_network", "research-support"),
            ("global_outgoing", "green_capability_export_share", "g_out_network", "research-support"),
        ]
        for key, x_col, y_col, tier in plot_specs:
            for audience, ext in [("portfolio", "png"), ("research", "svg")]:
                path = self.paths.scenario_phase_space_plot_dir / f"scenario_phase_space_{key}_{start_year}_{end_year}_{audience}.{ext}"
                fig = plot_global_scenario_overlay(time_series, x_col, y_col, _title(key, audience), path, self.color_mode, self.mark_years)
                plt.close(fig)
                rows.append(self._manifest_row(path, key, "global_overlay", ext, audience, tier, x_col, y_col, "", reference, sources, "all scenarios"))

        endpoint_specs = [
            ("endpoint_green_local", "green_capability_export_share", "g_local"),
            ("endpoint_output_local", "log_total_realized_output", "g_local"),
        ]
        for key, x_col, y_col in endpoint_specs:
            for audience, ext in [("portfolio", "png"), ("research", "svg")]:
                path = self.paths.scenario_phase_space_plot_dir / f"scenario_phase_space_{key}_{start_year}_{end_year}_{audience}.{ext}"
                fig = plot_endpoint_scatter(endpoint, x_col, y_col, _title("endpoint", audience), path, self.color_mode)
                plt.close(fig)
                rows.append(self._manifest_row(path, key, "endpoint_scatter", ext, audience, "thesis-core", x_col, y_col, "", reference, sources, "final-year endpoint"))

        for audience, ext in [("portfolio", "png"), ("research", "svg")]:
            path = self.paths.scenario_phase_space_plot_dir / f"scenario_phase_space_delta_summary_{start_year}_{end_year}_{audience}.{ext}"
            fig = plot_delta_bar(delta, _title("delta", audience), path, self.color_mode)
            plt.close(fig)
            rows.append(self._manifest_row(path, "delta_summary", "delta_bar", ext, audience, "thesis-core", "scenario", "delta_value", "", reference, sources, "endpoint delta"))

            heatmap_path = self.paths.scenario_phase_space_plot_dir / f"scenario_phase_space_scorecard_{start_year}_{end_year}_{audience}.{ext}"
            fig = plot_scorecard_heatmap(delta, _title("delta", audience), heatmap_path, self.color_mode)
            plt.close(fig)
            rows.append(self._manifest_row(heatmap_path, "scorecard", "scorecard_heatmap", ext, audience, "research-support", "scenario", "normalized delta", "", reference, sources, "normalized endpoint deltas"))

            ts_path = self.paths.scenario_phase_space_plot_dir / f"scenario_phase_space_time_series_{start_year}_{end_year}_{audience}.{ext}"
            fig = plot_time_series(time_series, ["total_realized_output", "g_local", "g_in_network", "g_out_network"], _title("time_series", audience), ts_path, self.color_mode)
            plt.close(fig)
            rows.append(self._manifest_row(ts_path, "time_series", "time_series", ext, audience, "research-support", "Year", "state variables", "", reference, sources, "scenario-year values"))

        if not sector_summary.empty:
            sector_plot = sector_summary.loc[sector_summary["selection_rule"].eq("top_movement_relevance")].copy()
            path = self.paths.scenario_phase_space_plot_dir / f"scenario_phase_space_sector_delta_ranking_{start_year}_{end_year}_portfolio.png"
            fig = plot_sector_delta_ranking(sector_plot, _title("sector", "portfolio"), path, self.color_mode)
            plt.close(fig)
            rows.append(self._manifest_row(path, "sector_delta_ranking", "sector_delta_ranking", "png", "portfolio", "research-support", "sector", "movement_score", "", reference, sources, f"top {self.top_sector_n} sectors by movement relevance"))
        if not node_summary.empty:
            path = self.paths.scenario_phase_space_plot_dir / f"scenario_phase_space_node_delta_ranking_{start_year}_{end_year}_portfolio.png"
            fig = plot_node_delta_ranking(node_summary.loc[node_summary["selection_rule"].str.contains("top10", na=False)], _title("node", "portfolio"), path, self.color_mode)
            plt.close(fig)
            rows.append(self._manifest_row(path, "node_delta_ranking", "node_delta_ranking", "png", "portfolio", "research-support", "country_sector", "delta_X_realized_total", "", reference, sources, "top 10 output/emissions nodes"))
        return rows

    def _manifest_row(
        self,
        path: Path,
        figure_name: str,
        family: str,
        output_format: str,
        audience: str,
        tier: str,
        axis_x: str,
        axis_y: str,
        axis_z: str,
        reference: str,
        sources: ScenarioPhaseSpaceSources,
        selection_rule: str,
    ) -> dict[str, object]:
        return {
            "figure_path": str(path),
            "figure_name": figure_name,
            "figure_family": family,
            "output_format": output_format,
            "portfolio_or_research": audience,
            "recommendation_status": "recommended-draft" if tier == "thesis-core" else "supporting",
            "title": _title(figure_name, audience),
            "caption": _caption_for_family(family),
            "interpretation_message": _interpretation_for_family(family),
            "axis_x": axis_x,
            "axis_y": axis_y,
            "axis_z": axis_z,
            "scenario_names": ",".join(self.scenario_names or []),
            "reference_scenario": reference,
            "start_year": getattr(self, "_active_start_year", ""),
            "end_year": getattr(self, "_active_end_year", ""),
            "mark_years": ",".join(str(year) for year in self.mark_years),
            "selection_rule": selection_rule,
            "data_source_files": "; ".join(str(path) for path in sources.__dict__.values()),
            "limitations": "Scenario response is a behavioural Leontief production-network perturbation layer, not a full endogenous green-transition simulation.",
            "figure_tier": tier,
        }


def discover_scenario_phase_space_sources(
    paths: ABMV3Paths,
    start_year: int,
    end_year: int,
    state_panel: Path | None = None,
) -> ScenarioPhaseSpaceSources:
    """Locate source files used by scenario phase-space plots."""
    return ScenarioPhaseSpaceSources(
        scenario_summary=paths.behavioural_scenario_analysis_summary_path(start_year, end_year),
        scenario_by_year=paths.behavioural_scenario_analysis_by_year_path(start_year, end_year),
        scenario_sector_effects=paths.behavioural_scenario_analysis_sector_effects_path(start_year, end_year),
        scenario_country_effects=paths.behavioural_scenario_analysis_country_effects_path(start_year, end_year),
        scenario_node_comparison_dir=paths.behavioural_leontief_scenario_diagnostics_dir,
        historical_state_panel=state_panel or paths.scenario_phase_space_dir.parent / "phase_space" / f"abm_v3_phase_space_state_panel_{start_year}_{end_year}.parquet",
    )


def resolve_reference_scenario(scenario_names: list[str], requested: str) -> str:
    """Resolve the scenario reference without inventing a baseline."""
    if requested and requested not in {"historical_or_baseline", "auto"}:
        return requested if requested in scenario_names else "no_baseline_available"
    for scenario_name in scenario_names:
        lowered = scenario_name.lower()
        if "historical" in lowered or "baseline" in lowered:
            return scenario_name
    return "historical_endpoint_reference"


def build_scenario_node_panel(
    node_comparison_dir: Path,
    state_panel: pd.DataFrame,
    scenario_names: list[str],
    start_year: int,
    end_year: int,
    diagnostics: list[dict[str, object]],
) -> pd.DataFrame:
    """Join node-level scenario responses to historical phase-space state variables."""
    frames = []
    for path in sorted(Path(node_comparison_dir).glob("node_comparison_*.csv")):
        try:
            preview = pd.read_csv(path, usecols=["Year", "scenario_name"], nrows=1)
        except Exception as error:  # noqa: BLE001
            _record(diagnostics, "read_warning", "node_comparison", f"Could not inspect node comparison file: {error}", str(path))
            continue
        if preview.empty:
            continue
        year = int(preview["Year"].iloc[0])
        scenario = str(preview["scenario_name"].iloc[0])
        if year < start_year or year > end_year or (scenario_names and scenario not in scenario_names):
            continue
        columns = [
            "Year",
            "scenario_name",
            "selector_name",
            "country_sector",
            "Country",
            "Sector",
            "X_realized_baseline",
            "X_realized_scenario",
            "delta_X_realized",
            "pct_delta_X_realized",
            "is_selected_node",
        ]
        available = [column for column in columns if column in pd.read_csv(path, nrows=0).columns]
        frames.append(pd.read_csv(path, usecols=available))
    if not frames:
        _record(diagnostics, "missing_source", "node_comparison", "No node-comparison scenario files were available.", str(node_comparison_dir))
        return pd.DataFrame()
    scenario_nodes = pd.concat(frames, ignore_index=True)
    state_columns = [column for column in STATE_COLUMNS if column in state_panel.columns]
    joined = scenario_nodes.merge(
        state_panel[state_columns],
        on=["country_sector", "Year"],
        how="left",
        suffixes=("_scenario_file", ""),
    )
    missing_state = int(joined["g_local"].isna().sum()) if "g_local" in joined.columns else len(joined)
    if missing_state:
        _record(diagnostics, "join_warning", "historical_state_panel", "Scenario nodes missing joined phase-space state variables.", missing_state)
    return joined


def build_scenario_time_series(node_panel: pd.DataFrame, by_year: pd.DataFrame) -> pd.DataFrame:
    """Build output-weighted scenario-year phase-space summaries."""
    if node_panel.empty:
        return pd.DataFrame()
    rows = []
    for (scenario, year), group in node_panel.groupby(["scenario_name", "Year"], dropna=False):
        weight = _positive_weight(group, "X_realized_scenario", "X_observed")
        total_output = _sum(group, "X_realized_scenario")
        row = {
            "scenario_name": scenario,
            "Year": int(year),
            "green_capability_export_share": _weighted(group, "green_capability_export_share", weight),
            "g_local": _weighted(group, "g_local", weight),
            "g_in_network": _weighted(group, "g_in_network", weight),
            "g_out_network": _weighted(group, "g_out_network", weight),
            "total_realized_output": total_output,
            "log_total_realized_output": np.log1p(total_output) if np.isfinite(total_output) and total_output >= 0 else np.nan,
            "total_delta_X_realized": _sum(group, "delta_X_realized"),
            "total_emissions_observed": _sum(group, "emissions_observed"),
            "mean_EI": _weighted(group, "EI", weight),
            "selected_node_count": int(group.get("is_selected_node", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        }
        rows.append(row)
    output = pd.DataFrame(rows)
    if not by_year.empty and {"scenario_name", "Year"}.issubset(by_year.columns):
        columns = [
            column
            for column in ["scenario_name", "Year", "pct_delta_realized_output_total", "delta_realized_output_total"]
            if column in by_year.columns
        ]
        output = output.merge(by_year[columns], on=["scenario_name", "Year"], how="left")
    return output.sort_values(["scenario_name", "Year"])


def build_endpoint_summary(time_series: pd.DataFrame, end_year: int, reference: str) -> pd.DataFrame:
    """Build final-year scenario endpoints and reference metadata."""
    if time_series.empty:
        return pd.DataFrame()
    endpoint = time_series.loc[pd.to_numeric(time_series["Year"], errors="coerce").eq(end_year)].copy()
    endpoint["reference_scenario"] = reference
    return endpoint.sort_values("scenario_name")


def build_delta_summary(endpoint: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Build long-form endpoint deltas from a scenario or historical endpoint reference."""
    if endpoint.empty:
        return pd.DataFrame()
    if reference in set(endpoint["scenario_name"].astype(str)):
        reference_row = endpoint.loc[endpoint["scenario_name"].astype(str).eq(reference)].iloc[0]
        reference_label = reference
    else:
        numeric = endpoint[AXIS_VARIABLES].apply(pd.to_numeric, errors="coerce")
        reference_row = numeric.mean()
        reference_label = reference
    rows = []
    for _, row in endpoint.iterrows():
        for variable in AXIS_VARIABLES:
            if variable not in endpoint.columns:
                continue
            value = pd.to_numeric(pd.Series([row.get(variable)]), errors="coerce").iloc[0]
            reference_value = pd.to_numeric(pd.Series([reference_row.get(variable)]), errors="coerce").iloc[0]
            rows.append(
                {
                    "scenario_name": row["scenario_name"],
                    "reference_scenario": reference_label,
                    "variable": variable,
                    "value": value,
                    "reference_value": reference_value,
                    "delta": value - reference_value if np.isfinite(value) and np.isfinite(reference_value) else np.nan,
                    "pct_delta": (value - reference_value) / reference_value if np.isfinite(value) and np.isfinite(reference_value) and reference_value != 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_sector_summary(node_panel: pd.DataFrame, sector_effects: pd.DataFrame, top_sector_n: int) -> pd.DataFrame:
    """Summarize scenario response by sector and select portfolio sectors."""
    if node_panel.empty or "Sector" not in node_panel.columns:
        return pd.DataFrame()
    agg_spec = {
        "total_delta_X_realized": ("delta_X_realized", "sum"),
        "mean_pct_delta_X_realized": ("pct_delta_X_realized", "mean"),
    }
    for source, output in [
        ("g_local", "mean_g_local"),
        ("g_in_network", "mean_g_in_network"),
        ("g_out_network", "mean_g_out_network"),
        ("green_capability_export_share", "mean_green_capability_export_share"),
    ]:
        if source in node_panel.columns:
            agg_spec[output] = (source, "mean")
    grouped = node_panel.groupby(["scenario_name", "Sector"], dropna=False).agg(**agg_spec).reset_index()
    for column in ["mean_g_local", "mean_g_in_network", "mean_g_out_network", "mean_green_capability_export_share"]:
        if column not in grouped.columns:
            grouped[column] = np.nan
    grouped["movement_score"] = grouped[["total_delta_X_realized", "mean_g_local", "mean_g_in_network", "mean_g_out_network"]].abs().sum(axis=1)
    grouped["selection_rule"] = "research_all"
    top = grouped.sort_values("movement_score", ascending=False).head(top_sector_n).index
    grouped.loc[top, "selection_rule"] = "top_movement_relevance"
    return grouped


def build_node_summary(node_panel: pd.DataFrame, top_node_n: int, research_top_node_n: int) -> pd.DataFrame:
    """Summarize selected top-output and top-emissions country-sector nodes."""
    if node_panel.empty:
        return pd.DataFrame()
    summaries = []
    for rule, value_column, limit in [
        ("top10_output", "X_observed", top_node_n),
        ("top10_emissions", "emissions_observed", top_node_n),
        ("top25_output", "X_observed", research_top_node_n),
        ("top25_emissions", "emissions_observed", research_top_node_n),
    ]:
        if value_column not in node_panel.columns:
            continue
        top_nodes = node_panel.groupby("country_sector")[value_column].sum(min_count=1).sort_values(ascending=False).head(limit).index
        subset = node_panel.loc[node_panel["country_sector"].isin(top_nodes)]
        agg_spec = {
            "Country": ("Country", "first"),
            "Sector": ("Sector", "first"),
            "delta_X_realized_total": ("delta_X_realized", "sum"),
            "mean_pct_delta_X_realized": ("pct_delta_X_realized", "mean"),
        }
        for source, output in [
            ("g_local", "mean_g_local"),
            ("g_in_network", "mean_g_in_network"),
            ("g_out_network", "mean_g_out_network"),
            ("green_capability_export_share", "mean_green_capability_export_share"),
        ]:
            if source in subset.columns:
                agg_spec[output] = (source, "mean")
        grouped = subset.groupby(["scenario_name", "country_sector"], dropna=False).agg(**agg_spec).reset_index()
        for column in ["mean_g_local", "mean_g_in_network", "mean_g_out_network", "mean_green_capability_export_share"]:
            if column not in grouped.columns:
                grouped[column] = np.nan
        grouped["selection_rule"] = rule
        summaries.append(grouped)
    return pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()


def plot_global_scenario_overlay(df: pd.DataFrame, x_col: str, y_col: str, title: str, output_path: Path, color_mode: str, mark_years: tuple[int, ...]):
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = _palette(color_mode, max(1, df["scenario_name"].nunique()))
    for index, (scenario, group) in enumerate(df.groupby("scenario_name", sort=True)):
        group = group.sort_values("Year")
        ax.plot(group[x_col], group[y_col], color=colors[index], marker=MARKERS[index % len(MARKERS)], linewidth=1.7, label=clean_display_label(scenario), markevery=_mark_indices(group, mark_years))
        anchors = group.loc[group["Year"].isin(mark_years)]
        ax.scatter(anchors[x_col], anchors[y_col], s=28, color=colors[index], edgecolor="#111111", linewidth=0.4)
    ax.set_xlabel(_axis_label(x_col))
    ax.set_ylabel(_axis_label(y_col))
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, output_path)
    return fig


def plot_endpoint_scatter(endpoint: pd.DataFrame, x_col: str, y_col: str, title: str, output_path: Path, color_mode: str):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = _palette(color_mode, max(1, len(endpoint)))
    for index, (_, row) in enumerate(endpoint.iterrows()):
        ax.scatter(row[x_col], row[y_col], color=colors[index], marker=MARKERS[index % len(MARKERS)], s=70)
        ax.text(row[x_col], row[y_col], " " + _short_label(row["scenario_name"]), fontsize=8, va="center")
    ax.set_xlabel(_axis_label(x_col))
    ax.set_ylabel(_axis_label(y_col))
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, output_path)
    return fig


def plot_delta_bar(delta: pd.DataFrame, title: str, output_path: Path, color_mode: str):
    key = delta.loc[delta["variable"].isin(["g_local", "g_in_network", "g_out_network", "total_realized_output"])].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    if key.empty:
        ax.set_title(title)
        _save(fig, output_path)
        return fig
    key["label"] = key["scenario_name"].map(_short_label) + " | " + key["variable"]
    key = key.sort_values("delta")
    ax.barh(key["label"], key["delta"], color=_palette(color_mode, 1)[0])
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Delta from reference")
    fig.tight_layout()
    _save(fig, output_path)
    return fig


def plot_scorecard_heatmap(delta: pd.DataFrame, title: str, output_path: Path, color_mode: str):
    variables = ["green_capability_export_share", "g_local", "g_in_network", "g_out_network", "total_realized_output", "total_emissions_observed", "mean_EI"]
    matrix = delta.loc[delta["variable"].isin(variables)].pivot_table(index="scenario_name", columns="variable", values="pct_delta", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    values = matrix.fillna(0.0).to_numpy(dtype=float)
    image = ax.imshow(values, aspect="auto", cmap="PuOr" if color_mode == "default" else "cividis")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels([_axis_label(col) for col in matrix.columns], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels([clean_display_label(value) for value in matrix.index])
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="Percent delta from reference")
    fig.tight_layout()
    _save(fig, output_path)
    return fig


def plot_time_series(df: pd.DataFrame, variables: list[str], title: str, output_path: Path, color_mode: str):
    fig, axes = plt.subplots(len(variables), 1, figsize=(10, 2.4 * len(variables)), sharex=True)
    if len(variables) == 1:
        axes = [axes]
    colors = _palette(color_mode, max(1, df["scenario_name"].nunique()))
    for axis, variable in zip(axes, variables):
        for index, (scenario, group) in enumerate(df.groupby("scenario_name", sort=True)):
            axis.plot(group["Year"], group[variable], color=colors[index], linewidth=1.5, label=clean_display_label(scenario))
        axis.set_ylabel(_axis_label(variable))
        axis.grid(alpha=0.2)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Year")
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    _save(fig, output_path)
    return fig


def plot_sector_delta_ranking(df: pd.DataFrame, title: str, output_path: Path, color_mode: str):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    data = df.sort_values("movement_score", ascending=True).tail(12)
    ax.barh(data["Sector"].map(clean_display_label), data["movement_score"], color=_palette(color_mode, 1)[0])
    ax.set_xlabel("Movement relevance score")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, output_path)
    return fig


def plot_node_delta_ranking(df: pd.DataFrame, title: str, output_path: Path, color_mode: str):
    fig, ax = plt.subplots(figsize=(9, 6))
    data = df.sort_values("delta_X_realized_total", ascending=True).tail(12)
    ax.barh(data["country_sector"].map(_short_node_label), data["delta_X_realized_total"], color=_palette(color_mode, 1)[0])
    ax.set_xlabel("Total delta realized output")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, output_path)
    return fig


def build_figure_recommendations(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return pd.DataFrame(columns=["plot_file", "figure_tier", "recommendation_status", "reason", "suggested_use", "title", "caption_note", "known_limitation"])
    rows = []
    for _, row in manifest.iterrows():
        tier = row.get("figure_tier", "diagnostic")
        rows.append(
            {
                "plot_file": row["figure_path"],
                "figure_tier": tier,
                "recommendation_status": row["recommendation_status"],
                "reason": "Directly supports scenario phase-space comparison." if tier == "thesis-core" else "Supports scenario interpretation or diagnostics.",
                "suggested_use": "main thesis scenario figure candidate" if tier == "thesis-core" else "research support or appendix",
                "title": row["title"],
                "caption_note": row["caption"],
                "known_limitation": row["limitations"],
            }
        )
    return pd.DataFrame(rows)


def build_readme(
    start_year: int,
    end_year: int,
    sources: ScenarioPhaseSpaceSources,
    reference: str,
    diagnostics: list[dict[str, object]],
    manifest: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            f"# ABM v3 Scenario Phase-Space Plots ({start_year}-{end_year})",
            "",
            "These figures compare behavioural Leontief scenario perturbation responses in the visual language of the historical phase-space layer.",
            "",
            "They are not forecasts, welfare results, or full endogenous green-transition simulations.",
            "",
            f"Reference scenario: `{reference}`.",
            "",
            "If `historical_endpoint_reference` is used, no true baseline scenario was found in the available scenario names. Deltas are measured against the average final-year historical endpoint represented by the scenario endpoint table.",
            "",
            "Green-state coordinates are historical node-state variables joined to scenario output responses. Scenario movement in these coordinates should be read as output-reweighted green-readiness response, not as endogenous changes in green capability or emissions intensity.",
            "",
            "Source files:",
            f"- Scenario summary: `{sources.scenario_summary}`",
            f"- Scenario-year summary: `{sources.scenario_by_year}`",
            f"- Sector effects: `{sources.scenario_sector_effects}`",
            f"- Node comparison directory: `{sources.scenario_node_comparison_dir}`",
            f"- Historical state panel: `{sources.historical_state_panel}`",
            "",
            f"Figures written: {len(manifest)}.",
            f"Diagnostics recorded: {len(diagnostics)}.",
        ]
    )


def manifest_columns() -> list[str]:
    return [
        "figure_path",
        "figure_name",
        "figure_family",
        "output_format",
        "portfolio_or_research",
        "recommendation_status",
        "title",
        "caption",
        "interpretation_message",
        "axis_x",
        "axis_y",
        "axis_z",
        "scenario_names",
        "reference_scenario",
        "start_year",
        "end_year",
        "mark_years",
        "selection_rule",
        "data_source_files",
        "limitations",
        "figure_tier",
    ]


def _read_csv(path: Path, diagnostics: list[dict[str, object]], label: str) -> pd.DataFrame:
    if not path.exists():
        _record(diagnostics, "missing_source", label, "Source file is missing.", str(path))
        return pd.DataFrame()
    _record(diagnostics, "source", label, "Loaded source file.", str(path))
    return pd.read_csv(path)


def _parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).schema_arrow.names
    except Exception:  # noqa: BLE001
        return []


def _record(diagnostics: list[dict[str, object]], diagnostic_type: str, subject: str, message: str, value: object) -> None:
    diagnostics.append({"diagnostic_type": diagnostic_type, "subject": subject, "message": message, "value": value})


def _positive_weight(df: pd.DataFrame, preferred: str, fallback: str) -> pd.Series:
    if preferred in df.columns:
        weight = pd.to_numeric(df[preferred], errors="coerce")
    elif fallback in df.columns:
        weight = pd.to_numeric(df[fallback], errors="coerce")
    else:
        weight = pd.Series(1.0, index=df.index)
    return weight.where(weight.gt(0), 1.0).fillna(1.0)


def _weighted(df: pd.DataFrame, column: str, weight: pd.Series) -> float:
    if column not in df.columns:
        return np.nan
    values = pd.to_numeric(df[column], errors="coerce")
    valid = values.notna() & weight.notna() & weight.gt(0)
    if not valid.any():
        return float(values.mean(skipna=True))
    return float(np.average(values.loc[valid], weights=weight.loc[valid]))


def _sum(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return np.nan
    return float(pd.to_numeric(df[column], errors="coerce").sum(skipna=True))


def _palette(color_mode: str, count: int) -> list[str]:
    colors = COLORBLIND_COLORS if color_mode == "colorblind" else DEFAULT_COLORS
    return [colors[index % len(colors)] for index in range(count)]


def _mark_indices(group: pd.DataFrame, mark_years: tuple[int, ...]) -> list[int]:
    years = pd.to_numeric(group["Year"], errors="coerce").tolist()
    indices = [index for index, year in enumerate(years) if np.isfinite(year) and int(year) in set(mark_years)]
    if 0 not in indices and years:
        indices.insert(0, 0)
    if years and len(years) - 1 not in indices:
        indices.append(len(years) - 1)
    return sorted(set(indices))


def _save(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")


def _title(key: str, audience: str) -> str:
    if key in TITLE_REGISTRY:
        return TITLE_REGISTRY[key].get(audience, key)
    for candidate in ["endpoint", "delta", "time_series", "sector", "node", "capacity"]:
        if candidate in key and candidate in TITLE_REGISTRY:
            return TITLE_REGISTRY[candidate].get(audience, key)
    return key


def _caption_for_family(family: str) -> str:
    if "endpoint" in family:
        return CAPTIONS["endpoint"]
    if "delta" in family or "scorecard" in family:
        return CAPTIONS["delta"]
    if "sector" in family:
        return CAPTIONS["sector"]
    if "node" in family:
        return CAPTIONS["node"]
    return CAPTIONS["global"]


def _interpretation_for_family(family: str) -> str:
    if "global" in family:
        return "Compares scenario response as output-reweighted movement through historical green-state coordinates."
    if "endpoint" in family:
        return "Separates final-year green-readiness position from output expansion."
    if "delta" in family:
        return "Shows whether differences are output responses before structural green repositioning."
    if "sector" in family:
        return "Shows sectoral channels and capability-constrained response paths."
    if "node" in family:
        return "Shows heterogeneous response among major country-sector nodes."
    return "Scenario phase-space diagnostic."


def _axis_label(column: str) -> str:
    labels = {
        "green_capability_export_share": "Green capability",
        "g_local": "Local green-ness",
        "g_in_network": "Incoming network green-ness",
        "g_out_network": "Outgoing network green-ness",
        "log_total_realized_output": "Production scale, log output",
        "total_realized_output": "Total realized output",
        "total_emissions_observed": "Observed emissions",
        "mean_EI": "Mean emissions intensity",
    }
    return labels.get(column, clean_display_label(column))


def _short_label(value: object) -> str:
    text = clean_display_label(value)
    text = text.replace("_node_demand_expansion_10", "")
    text = text.replace("_", " ")
    return text[:34]


def _short_node_label(value: object) -> str:
    text = clean_display_label(value)
    parts = [part.strip() for part in re.split(r"\||-", text) if part.strip()]
    if len(parts) >= 2:
        return f"{parts[0][:3]}: {parts[-1][:24]}"
    return text[:30]
