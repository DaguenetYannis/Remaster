from __future__ import annotations

import re
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

DISPLAY_REPLACEMENTS = {
    "capabiliity": "capability",
    "Capabiliity": "Capability",
    "CAPABILIITY": "CAPABILITY",
    "Finacial": "Financial",
    "Finacial Intermediation and Business Activities": "Financial Intermediation and Business Activities",
    "Restraurants": "Restaurants",
    "Hotels and Restraurants": "Hotels and Restaurants",
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
    labels = [clean_display_label(value) for value in data["scenario_name"]]
    values = pd.to_numeric(data["mean_pct_delta_realized_output_total"], errors="coerce") * 100.0
    xerr = None
    if audience == "research" and {"min_pct_delta_realized_output_total", "max_pct_delta_realized_output_total"}.issubset(data.columns):
        lower = values - pd.to_numeric(data["min_pct_delta_realized_output_total"], errors="coerce") * 100.0
        upper = pd.to_numeric(data["max_pct_delta_realized_output_total"], errors="coerce") * 100.0 - values
        xerr = np.vstack([lower.clip(lower=0.0), upper.clip(lower=0.0)])
    ax.barh(labels, values, color=colors, xerr=xerr)
    ax.axvline(0, color="#333333", linewidth=0.8)
    for index, value in enumerate(values):
        ax.text(value, index, clean_display_label(f" {value:.2f}%"), va="center", fontsize=9)
    ax.set_xlabel(clean_display_label("Mean change in realized output (%)"))
    title = "Demand perturbations generate larger network responses than capacity stress"
    if audience == "research":
        title = "Mean, min, and max yearly realized-output effects by scenario"
    ax.set_title(clean_display_label(title))
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
        ax.plot(group["Year"], y, marker=markers[index % len(markers)], color=colors[index], label=clean_display_label(scenario_name), linewidth=1.8)
        if audience == "portfolio" and len(group):
            ax.text(group["Year"].iloc[-1], y.iloc[-1], clean_display_label(f" {clean_display_label(scenario_name)}"), va="center", fontsize=8)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel(clean_display_label("Year"))
    ax.set_ylabel(clean_display_label("Change in realized output (%)"))
    ax.set_title(clean_display_label("Scenario effects vary over time but remain comparative perturbations"))
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
        ax.bar(x + (index - 2) * width, pd.to_numeric(data[metric], errors="coerce"), width=width, label=clean_display_label(metric.replace("_share", "")), color=colors[index])
    ax.set_xticks(x)
    ax.set_xticklabels([clean_display_label(value) for value in data["scenario_name"]], rotation=25, ha="right")
    ax.set_ylabel(clean_display_label("Share of selected rows"))
    ax.set_title(clean_display_label("Low EI and green productive capability select different node sets"))
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig


def plot_selector_overlap_heatmap(
    selector_df: pd.DataFrame,
    audience: str = "portfolio",
    color_mode: str = "default",
    output_path: Path | None = None,
):
    """Portfolio selector-overlap heatmap for EI and capability dimensions."""
    _validate_plot_options(audience, color_mode)
    data = selector_df.copy()
    columns = [
        ("low_EI_share", "Low EI"),
        ("high_EI_share", "High EI"),
        ("high_green_capability_share", "High green capability"),
        ("clean_and_capable_share", "Clean and capable"),
        ("transition_pivot_share", "Transition pivot"),
    ]
    values = data[[column for column, _label_text in columns]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    cmap = "Blues" if color_mode == "default" else "viridis"
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_yticks(np.arange(len(data)))
    ax.set_yticklabels([clean_display_label(value) for value in data["scenario_name"]])
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels([clean_display_label(label) for _column, label in columns], rotation=25, ha="right")
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            text = "" if not np.isfinite(value) else f"{value:.0%}"
            ax.text(column_index, row_index, clean_display_label(text), ha="center", va="center", color="#111111", fontsize=9)
    ax.set_title(clean_display_label("Low EI and green capability identify different node groups"))
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label=clean_display_label("Share of selected rows"))
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
    """Highlight near-zero capacity-bottleneck effect relative to demand perturbations."""
    _validate_plot_options(audience, color_mode)
    data = summary_df.copy().sort_values("mean_pct_delta_realized_output_total", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    values = pd.to_numeric(data["mean_pct_delta_realized_output_total"], errors="coerce") * 100.0
    labels = [clean_display_label(value) for value in data["scenario_name"]]
    colors = ["#6c757d" if "capacity bottleneck" in label.lower() else "#2f6f8f" for label in labels]
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel(clean_display_label("Mean change in realized output (%)"))
    ax.set_title(clean_display_label("High-EI capacity stress barely propagates under the ABM v3 capacity proxy"))
    for index, value in enumerate(values):
        ax.text(value, index, clean_display_label(f" {value:.3f}%"), va="center", fontsize=8)
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig


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
    values = pd.to_numeric(data["total_delta_X_realized_sum"], errors="coerce") / 1_000_000_000.0
    ax.barh(data[key_column].astype(str).map(clean_display_label), values, color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel(clean_display_label("Total change in realized output, billions"))
    ax.set_title(clean_display_label(f"Top {key_column.lower()} output effects: {clean_display_label(scenario_name)}"))
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


def clean_display_label(value: object) -> str:
    """Return display-only cleaned labels for plots and Markdown."""
    text = SCENARIO_LABELS.get(str(value), str(value))
    for old, new in DISPLAY_REPLACEMENTS.items():
        text = text.replace(old, new)
    text = re.sub(r"\b(bottleneck|bottelneck|botleneck|bottle neck|bottel neck)\b", "bottleneck", text, flags=re.IGNORECASE)
    return text


def display_label_cleaner(value: object) -> str:
    """Alias for display-only label cleaning used by report tests and callers."""
    return clean_display_label(value)
