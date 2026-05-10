from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCENARIO_LABELS = {
    "low_ei_node_demand_expansion_10": "Low EI demand",
    "green_capability_node_demand_expansion_10": "Green capability demand",
    "clean_and_capable_node_demand_expansion_10": "Clean and capable demand",
    "transition_pivot_node_demand_expansion_10": "Transition pivot demand",
    "high_ei_node_capacity_bottleneck_10": "High EI capacity bottleneck",
}

DEFAULT_COLORS = ["#2f6f8f", "#6a994e", "#bc6c25", "#7b2cbf", "#6c757d"]
COLORBLIND_COLORS = ["#0072B2", "#009E73", "#E69F00", "#CC79A7", "#999999"]


def plot_scenario_output_effect(
    summary_df: pd.DataFrame,
    audience: str = "portfolio",
    color_mode: str = "default",
    output_path: Path | None = None,
):
    """Compare mean percent realized-output effects by scenario."""
    _validate_plot_options(audience, color_mode)
    data = summary_df.copy()
    data = data.sort_values("mean_pct_delta_realized_output_total", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5 if audience == "portfolio" else 6))
    colors = _palette(color_mode, len(data))
    labels = [_label(value) for value in data["scenario_name"]]
    values = pd.to_numeric(data["mean_pct_delta_realized_output_total"], errors="coerce") * 100.0
    xerr = None
    if audience == "research" and {"min_pct_delta_realized_output_total", "max_pct_delta_realized_output_total"}.issubset(data.columns):
        lower = values - pd.to_numeric(data["min_pct_delta_realized_output_total"], errors="coerce") * 100.0
        upper = pd.to_numeric(data["max_pct_delta_realized_output_total"], errors="coerce") * 100.0 - values
        xerr = np.vstack([lower.clip(lower=0.0), upper.clip(lower=0.0)])
    ax.barh(labels, values, color=colors, xerr=xerr)
    ax.axvline(0, color="#333333", linewidth=0.8)
    for index, value in enumerate(values):
        ax.text(value, index, f" {value:.2f}%", va="center", fontsize=9)
    ax.set_xlabel("Mean change in realized output (%)")
    title = "Demand shocks dominate ABM v3 production-network responses"
    if audience == "research":
        title = "Mean, min, and max yearly realized-output effects by scenario"
    ax.set_title(title)
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig


def plot_scenario_output_trajectory(
    by_year_df: pd.DataFrame,
    audience: str = "portfolio",
    color_mode: str = "default",
    output_path: Path | None = None,
):
    """Plot yearly percent realized-output effects by scenario."""
    _validate_plot_options(audience, color_mode)
    data = by_year_df.copy()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    scenarios = list(data["scenario_name"].dropna().unique())
    colors = _palette(color_mode, len(scenarios))
    markers = ["o", "s", "^", "D", "x", "P"]
    for index, scenario_name in enumerate(scenarios):
        group = data.loc[data["scenario_name"].eq(scenario_name)].sort_values("Year")
        y = pd.to_numeric(group["pct_delta_realized_output_total"], errors="coerce") * 100.0
        ax.plot(group["Year"], y, marker=markers[index % len(markers)], color=colors[index], label=_label(scenario_name), linewidth=1.8)
        if audience == "portfolio" and len(group):
            ax.text(group["Year"].iloc[-1], y.iloc[-1], f" {_label(scenario_name)}", va="center", fontsize=8)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Change in realized output (%)")
    ax.set_title("Scenario output effects vary by year but remain comparative perturbations")
    if audience == "research":
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig


def plot_selector_overlap(
    selector_df: pd.DataFrame,
    audience: str = "portfolio",
    color_mode: str = "default",
    output_path: Path | None = None,
):
    """Show selected-node composition across EI and capability dimensions."""
    _validate_plot_options(audience, color_mode)
    data = selector_df.copy()
    metrics = ["low_EI_share", "high_EI_share", "high_green_capability_share", "clean_and_capable_share", "transition_pivot_share"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(data))
    width = 0.15
    colors = _palette(color_mode, len(metrics))
    for index, metric in enumerate(metrics):
        ax.bar(x + (index - 2) * width, pd.to_numeric(data[metric], errors="coerce"), width=width, label=metric.replace("_share", ""), color=colors[index])
    ax.set_xticks(x)
    ax.set_xticklabels([_label(value) for value in data["scenario_name"]], rotation=25, ha="right")
    ax.set_ylabel("Share of selected rows")
    ax.set_title("Low EI and green productive capability select different node sets")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig


def plot_top_sector_effects(
    sector_df: pd.DataFrame,
    scenario_name: str,
    top_n: int = 10,
    audience: str = "portfolio",
    color_mode: str = "default",
    output_path: Path | None = None,
):
    """Plot top sectors by absolute total realized-output effect."""
    return _plot_top_effects(sector_df, scenario_name, "Sector", top_n, audience, color_mode, output_path)


def plot_top_country_effects(
    country_df: pd.DataFrame,
    scenario_name: str,
    top_n: int = 10,
    audience: str = "portfolio",
    color_mode: str = "default",
    output_path: Path | None = None,
):
    """Plot top countries by absolute total realized-output effect."""
    return _plot_top_effects(country_df, scenario_name, "Country", top_n, audience, color_mode, output_path)


def plot_capacity_bottleneck_effect(
    summary_df: pd.DataFrame,
    audience: str = "research",
    color_mode: str = "default",
    output_path: Path | None = None,
):
    """Highlight capacity-bottleneck effect relative to demand shocks."""
    return plot_scenario_output_effect(summary_df, audience=audience, color_mode=color_mode, output_path=output_path)


def _plot_top_effects(
    df: pd.DataFrame,
    scenario_name: str,
    key_column: str,
    top_n: int,
    audience: str,
    color_mode: str,
    output_path: Path | None,
):
    _validate_plot_options(audience, color_mode)
    data = df.loc[df["scenario_name"].eq(scenario_name)].copy()
    data["abs_effect"] = pd.to_numeric(data["total_delta_X_realized_sum"], errors="coerce").abs()
    data = data.sort_values("abs_effect", ascending=False).head(top_n).sort_values("total_delta_X_realized_sum")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = _palette(color_mode, 1)[0]
    values = pd.to_numeric(data["total_delta_X_realized_sum"], errors="coerce")
    ax.barh(data[key_column].astype(str), values, color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Total change in realized output")
    ax.set_title(f"Top {key_column.lower()} output effects: {_label(scenario_name)}")
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig


def _validate_plot_options(audience: str, color_mode: str) -> None:
    if audience not in {"portfolio", "research"}:
        raise ValueError("audience must be 'portfolio' or 'research'.")
    if color_mode not in {"default", "colorblind"}:
        raise ValueError("color_mode must be 'default' or 'colorblind'.")


def _palette(color_mode: str, count: int) -> list[str]:
    colors = COLORBLIND_COLORS if color_mode == "colorblind" else DEFAULT_COLORS
    return [colors[index % len(colors)] for index in range(count)]


def _save_figure(fig, output_path: Path | None) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")


def _label(scenario_name: str) -> str:
    return SCENARIO_LABELS.get(str(scenario_name), str(scenario_name))
