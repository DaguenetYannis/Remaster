from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PHASE_SPACE_PANEL_PATH = Path("data/abm_v3/phase_space/abm_v3_phase_space_state_panel_1995_2016.parquet")
PHASE_SPACE_PLOT_DIR = Path("outputs/plots/abm_v3/phase_space")
VECTOR_FIELD_DIR = Path("data/abm_v3/phase_space/vector_fields")
SELECTED_NODE_DIR = Path("data/abm_v3/phase_space")

DISPLAY_REPLACEMENTS = {
    "Finacial": "Financial",
    "Restraurants": "Restaurants",
}
DEFAULT_COLORS = ["#2f6f8f", "#6a994e", "#bc6c25", "#5f6c7b", "#7b2cbf", "#8d6e63", "#3a5a40"]
COLORBLIND_COLORS = ["#0072B2", "#009E73", "#E69F00", "#CC79A7", "#999999", "#56B4E9", "#D55E00"]
LINE_STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "P", "X", "v"]
DEFAULT_MARK_YEARS = (1995, 2000, 2008, 2016)

TITLE_REGISTRY = {
    "global_green_readiness_xy": {
        "portfolio": "Green capability and local green-ness rise together, but not smoothly",
        "research": "Output-weighted global trajectory in green capability and local green-ness, 1995-2016",
    },
    "global_green_readiness_incoming_3d": {
        "portfolio": "Incoming green exposure is more volatile than local green-ness",
        "research": "Global trajectory in green capability, local green-ness, and incoming network green-ness, 1995-2016",
    },
    "global_green_readiness_outgoing_3d": {
        "portfolio": "The system becomes locally greener faster than it becomes network-greener",
        "research": "Global trajectory in green capability, local green-ness, and outgoing network green-ness, 1995-2016",
    },
    "global_production_safe_xy": {
        "portfolio": "Local green-ness improves while production scale expands",
        "research": "Output-weighted trajectory in log production scale and local green-ness, 1995-2016",
    },
    "global_production_safe_3d": {
        "portfolio": "Greening occurs alongside rising production scale, not through collapse",
        "research": "Production-safe greening trajectory with directional network green-ness, 1995-2016",
    },
    "sector_green_readiness": {
        "portfolio": "Sector trajectories reveal distinct capability-constrained transition paths",
        "research": "Sector-level trajectories in green capability, local green-ness, and network green-ness, 1995-2016",
    },
    "nodes_top_output_green_readiness": {
        "portfolio": "Large production nodes do not share a single green transition path",
        "research": "Top output country-sector trajectories in green-readiness space, 1995-2016",
    },
    "nodes_top_emissions_green_readiness": {
        "portfolio": "The main carbon nodes do not converge uniformly toward green-readiness",
        "research": "Top emissions country-sector trajectories in green-readiness space, 1995-2016",
    },
    "vector_field_green_readiness": {
        "portfolio": "Historical movement fields show uneven motion across the green-readiness space",
        "research": "Binned year-to-year movement field in green capability and local green-ness, 1995-2016",
    },
}

CAPTION_REGISTRY = {
    "global": "Output-weighted country-sector trajectory. Local green-ness is a node-level proxy; incoming and outgoing network green-ness are directional network proxies, not the missing canonical network-green-exposure variable.",
    "production_safe": "This plot checks whether apparent greening occurs alongside production resilience rather than output collapse.",
    "sector_node": "Filtered trajectory view. Labels show selected sectors or country-sector nodes only; full trajectories remain available in research outputs.",
    "vector_field": "Arrows show average year-to-year movement in binned state space. These are historical diagnostics, not forecasts or causal transition rules.",
}


@dataclass(frozen=True)
class PhaseSpaceCubeSpec:
    """Axis and interpretation metadata for one available ABM v3 phase space."""

    name: str
    slug: str
    x: str
    y: str
    z: str
    x_label: str
    y_label: str
    z_label: str
    title_portfolio: str
    title_research: str
    interpretation_note: str
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...]
    caveats: str


PHASE_SPACE_CUBE_SPECS: dict[str, PhaseSpaceCubeSpec] = {
    "green_readiness_incoming": PhaseSpaceCubeSpec(
        name="Green Transition Readiness Cube - incoming network version",
        slug="green_readiness_incoming",
        x="green_capability_export_share",
        y="g_local",
        z="g_in_network",
        x_label="Green capability",
        y_label="Local green-ness (green up)",
        z_label="Incoming network green-ness",
        title_portfolio="Historical trajectories reveal uneven movement toward greener network states",
        title_research="Historical trajectories in green capability, local green-ness, and incoming network green-ness, 1995-2016",
        interpretation_note="Uses available incoming directional network green-ness, not the missing aggregate network_green_exposure concept.",
        required_columns=("country_sector", "Year", "green_capability_export_share", "g_local", "g_in_network"),
        optional_columns=("X_observed", "trajectory_weight_output", "emissions_observed", "Sector", "Country"),
        caveats="Historical state-space diagnostic, not a scenario forecast or causal transition estimate.",
    ),
    "green_readiness_outgoing": PhaseSpaceCubeSpec(
        name="Green Transition Readiness Cube - outgoing network version",
        slug="green_readiness_outgoing",
        x="green_capability_export_share",
        y="g_local",
        z="g_out_network",
        x_label="Green capability",
        y_label="Local green-ness (green up)",
        z_label="Outgoing network green-ness",
        title_portfolio="Historical trajectories reveal uneven movement toward greener network states",
        title_research="Historical trajectories in green capability, local green-ness, and outgoing network green-ness, 1995-2016",
        interpretation_note="Uses available outgoing directional network green-ness, not the missing aggregate network_green_exposure concept.",
        required_columns=("country_sector", "Year", "green_capability_export_share", "g_local", "g_out_network"),
        optional_columns=("X_observed", "trajectory_weight_output", "emissions_observed", "Sector", "Country"),
        caveats="Historical state-space diagnostic, not a scenario forecast or causal transition estimate.",
    ),
    "production_safe_incoming": PhaseSpaceCubeSpec(
        name="Production-Safe Greening Cube - incoming network version",
        slug="production_safe_incoming",
        x="log_X_observed",
        y="g_local",
        z="g_in_network",
        x_label="Production scale, log output",
        y_label="Local green-ness (green up)",
        z_label="Incoming network green-ness",
        title_portfolio="Production scale and local green-ness follow uneven paths",
        title_research="Historical trajectories in production scale, local green-ness, and incoming network green-ness, 1995-2016",
        interpretation_note="Places greening next to production scale to avoid confusing green transition with output collapse.",
        required_columns=("country_sector", "Year", "log_X_observed", "g_local", "g_in_network"),
        optional_columns=("X_observed", "trajectory_weight_output", "emissions_observed", "Sector", "Country"),
        caveats="Production-safe greening is diagnostic here; it is not a full endogenous transition simulation.",
    ),
    "production_safe_outgoing": PhaseSpaceCubeSpec(
        name="Production-Safe Greening Cube - outgoing network version",
        slug="production_safe_outgoing",
        x="log_X_observed",
        y="g_local",
        z="g_out_network",
        x_label="Production scale, log output",
        y_label="Local green-ness (green up)",
        z_label="Outgoing network green-ness",
        title_portfolio="Production scale and local green-ness follow uneven paths",
        title_research="Historical trajectories in production scale, local green-ness, and outgoing network green-ness, 1995-2016",
        interpretation_note="Places greening next to production scale to avoid confusing green transition with output collapse.",
        required_columns=("country_sector", "Year", "log_X_observed", "g_local", "g_out_network"),
        optional_columns=("X_observed", "trajectory_weight_output", "emissions_observed", "Sector", "Country"),
        caveats="Production-safe greening is diagnostic here; it is not a full endogenous transition simulation.",
    ),
}


def available_cube_specs() -> dict[str, PhaseSpaceCubeSpec]:
    """Return the implemented available-axis cube specs."""
    return dict(PHASE_SPACE_CUBE_SPECS)


def weighted_mean(df: pd.DataFrame, value_col: str, weight_col: str | None = None) -> float:
    """Compute a weighted mean, falling back to an unweighted mean when weights are unusable."""
    value = pd.to_numeric(df.get(value_col), errors="coerce")
    if value.empty:
        return np.nan
    if weight_col is None or weight_col not in df.columns:
        return float(value.mean(skipna=True))
    weight = pd.to_numeric(df[weight_col], errors="coerce")
    valid = value.notna() & weight.notna() & weight.gt(0)
    if not valid.any():
        return float(value.mean(skipna=True))
    return float(np.average(value.loc[valid], weights=weight.loc[valid]))


def clean_display_label(value: object) -> str:
    """Clean plot labels without changing source data."""
    text = str(value)
    for old, new in DISPLAY_REPLACEMENTS.items():
        text = text.replace(old, new)
    return text


def display_label_cleaner(value: object) -> str:
    """Alias for shared display-cleaning tests and callers."""
    return clean_display_label(value)


def plot_3d_global_trajectory(
    df: pd.DataFrame,
    cube_spec: PhaseSpaceCubeSpec,
    audience: str = "portfolio",
    color_mode: str = "default",
    output_path: Path | None = None,
    title_mode: str = "theory",
    mark_years: tuple[int, ...] = DEFAULT_MARK_YEARS,
):
    """Plot one output-weighted global trajectory through a 3D phase space."""
    _validate_plot_options(audience, color_mode)
    data = df.copy(deep=True)
    trajectory = aggregate_by_year(data, cube_spec)
    fig = plt.figure(figsize=(8, 6) if audience == "portfolio" else (9, 7))
    ax = fig.add_subplot(111, projection="3d")
    color = _palette(color_mode, 1)[0]
    _plot_3d_line(ax, trajectory, cube_spec, color=color, label="Global", marker="o", linewidth=2.4, mark_years=mark_years)
    _annotate_3d_anchor_years(ax, trajectory, cube_spec, mark_years)
    _format_3d_axes(ax, cube_spec, audience, title=_plot_title("global", cube_spec, "3d", audience, title_mode))
    _save_figure(fig, output_path)
    return fig


def plot_3d_sector_trajectories(
    df: pd.DataFrame,
    cube_spec: PhaseSpaceCubeSpec,
    audience: str = "research",
    color_mode: str = "default",
    output_path: Path | None = None,
    top_n: int = 8,
    title_mode: str = "theory",
    mark_years: tuple[int, ...] = DEFAULT_MARK_YEARS,
    selection_rule: str = "movement",
):
    """Plot output-weighted sector trajectories through a 3D phase space."""
    _validate_plot_options(audience, color_mode)
    data = df.copy(deep=True)
    if "Sector" not in data.columns:
        raise ValueError("Sector trajectory plot requires a Sector column.")
    sector_data = aggregate_by_group_year(data, ["Sector"], cube_spec)
    if audience == "portfolio":
        if selection_rule == "movement":
            selected = set(select_sectors_for_display(data, cube_spec, top_n).astype(str))
        else:
            sector_totals = _group_totals(data, "Sector", "emissions_observed" if selection_rule == "emissions" else _weight_column(data))
            selected = set(sector_totals.head(top_n).index.astype(str))
        sector_data = sector_data.loc[sector_data["Sector"].astype(str).isin(selected)].copy()
    sectors = sorted(sector_data["Sector"].dropna().astype(str).unique())
    fig = plt.figure(figsize=(9, 6.5) if audience == "portfolio" else (10, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    colors = _palette(color_mode, max(1, len(sectors)))
    alpha = 0.9 if audience == "portfolio" else 0.45
    for index, sector in enumerate(sectors):
        group = sector_data.loc[sector_data["Sector"].astype(str).eq(sector)].sort_values("Year")
        _plot_3d_line(
            ax,
            group,
            cube_spec,
            color=colors[index],
            label=clean_display_label(sector),
            marker=MARKERS[index % len(MARKERS)],
            linestyle=LINE_STYLES[index % len(LINE_STYLES)],
            alpha=alpha,
            linewidth=1.8,
            mark_years=mark_years,
        )
        if audience == "portfolio" and not group.empty:
            last = group.iloc[-1]
            ax.text(last[cube_spec.x], last[cube_spec.y], last[cube_spec.z], _short_sector_label(sector), fontsize=7)
    _format_3d_axes(ax, cube_spec, audience, title=_plot_title("sector", cube_spec, "3d", audience, title_mode))
    if audience == "portfolio" and len(sectors) <= 8:
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    _save_figure(fig, output_path)
    return fig


def plot_3d_node_trajectories(
    df: pd.DataFrame,
    cube_spec: PhaseSpaceCubeSpec,
    node_filter: str | Callable[[pd.DataFrame], pd.Series] = "top_output",
    audience: str = "research",
    color_mode: str = "default",
    output_path: Path | None = None,
    top_n: int = 25,
    title_mode: str = "theory",
    mark_years: tuple[int, ...] = DEFAULT_MARK_YEARS,
):
    """Plot selected country-sector node trajectories without plotting the full node universe."""
    _validate_plot_options(audience, color_mode)
    data = select_node_trajectories(df.copy(deep=True), node_filter=node_filter, top_n=top_n)
    nodes = sorted(data["country_sector"].dropna().astype(str).unique())
    fig = plt.figure(figsize=(10, 7) if audience == "portfolio" else (11, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = _palette(color_mode, max(1, len(nodes)))
    alpha = 0.85 if audience == "portfolio" else 0.55
    for index, node in enumerate(nodes):
        group = data.loc[data["country_sector"].astype(str).eq(node)].sort_values("Year")
        _plot_3d_line(
            ax,
            group,
            cube_spec,
            color=colors[index],
            label=_short_node_label(node),
            marker=MARKERS[index % len(MARKERS)],
            linestyle=LINE_STYLES[index % len(LINE_STYLES)],
            alpha=alpha,
            linewidth=1.3,
            mark_years=mark_years,
        )
        if audience == "portfolio" and not group.empty and index < 10:
            last = group.iloc[-1]
            ax.text(last[cube_spec.x], last[cube_spec.y], last[cube_spec.z], _short_node_label(node), fontsize=6)
    unit = "nodes_top_emissions" if "emissions" in str(node_filter) else "nodes_top_output"
    _format_3d_axes(ax, cube_spec, audience, title=_plot_title(unit, cube_spec, "3d", audience, title_mode))
    _save_figure(fig, output_path)
    return fig


def plot_2d_projection_trajectory(
    df: pd.DataFrame,
    cube_spec: PhaseSpaceCubeSpec,
    projection: str,
    unit: str,
    audience: str = "portfolio",
    color_mode: str = "default",
    output_path: Path | None = None,
    top_n: int = 25,
    title_mode: str = "theory",
    mark_years: tuple[int, ...] = DEFAULT_MARK_YEARS,
):
    """Plot a 2D projection for global, sector, or selected-node trajectories."""
    _validate_plot_options(audience, color_mode)
    x_col, y_col, x_label, y_label = _projection_columns(cube_spec, projection)
    data = df.copy(deep=True)
    fig, ax = plt.subplots(figsize=(8, 5.5) if audience == "portfolio" else (9, 6.5))
    if unit == "global":
        trajectory = aggregate_by_year(data, cube_spec)
        _plot_2d_line(ax, trajectory, x_col, y_col, color=_palette(color_mode, 1)[0], label="Global", marker="o", mark_years=mark_years)
        _annotate_2d_anchor_years(ax, trajectory, x_col, y_col, mark_years)
    elif unit == "sector":
        trajectory = aggregate_by_group_year(data, ["Sector"], cube_spec)
        groups = sorted(trajectory["Sector"].dropna().astype(str).unique())
        colors = _palette(color_mode, max(1, len(groups)))
        for index, sector in enumerate(groups[: (8 if audience == "portfolio" else len(groups))]):
            group = trajectory.loc[trajectory["Sector"].astype(str).eq(sector)].sort_values("Year")
            _plot_2d_line(ax, group, x_col, y_col, color=colors[index], label=clean_display_label(sector), marker=MARKERS[index % len(MARKERS)], alpha=0.8 if audience == "portfolio" else 0.45, mark_years=mark_years)
    else:
        filter_name = "top_emissions" if "emissions" in unit else "top_output"
        trajectory = select_node_trajectories(data, filter_name, top_n)
        groups = sorted(trajectory["country_sector"].dropna().astype(str).unique())
        colors = _palette(color_mode, max(1, len(groups)))
        for index, node in enumerate(groups):
            group = trajectory.loc[trajectory["country_sector"].astype(str).eq(node)].sort_values("Year")
            _plot_2d_line(ax, group, x_col, y_col, color=colors[index], label=_short_node_label(node), marker=MARKERS[index % len(MARKERS)], alpha=0.75 if audience == "portfolio" else 0.45, mark_years=mark_years)
    ax.set_xlabel(clean_display_label(x_label))
    ax.set_ylabel(clean_display_label(y_label))
    ax.set_title(clean_display_label(_plot_title(unit, cube_spec, projection, audience, title_mode)))
    ax.grid(alpha=0.25)
    if audience == "portfolio" and unit in {"global", "sector"}:
        ax.legend(fontsize=7)
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig


def plot_vector_field(
    df: pd.DataFrame,
    cube_spec: PhaseSpaceCubeSpec,
    projection: str = "x_y",
    bins: int = 10,
    weight_variable: str = "X_observed",
    audience: str = "research",
    color_mode: str = "default",
    output_path: Path | None = None,
    min_count: int = 30,
    title_mode: str = "theory",
):
    """Plot an average historical movement vector field in a 2D phase-space projection."""
    _validate_plot_options(audience, color_mode)
    vector_table = compute_vector_field_table(
        df.copy(deep=True),
        cube_spec,
        projection=projection,
        bins=bins,
        weight_variable=weight_variable,
        min_count=min_count,
    )
    x_col, y_col, x_label, y_label = _projection_columns(cube_spec, projection)
    fig, ax = plt.subplots(figsize=(8, 5.8) if audience == "portfolio" else (9, 6.5))
    if not vector_table.empty:
        ax.quiver(
            vector_table["x_center"],
            vector_table["y_center"],
            vector_table["delta_x_mean"],
            vector_table["delta_y_mean"],
            angles="xy",
            scale_units="xy",
            scale=1,
            color=_palette(color_mode, 1)[0],
            width=0.004,
            alpha=0.85,
        )
        ax.scatter(vector_table["x_center"], vector_table["y_center"], s=12, color="#333333", alpha=0.55)
    ax.set_xlabel(clean_display_label(x_label))
    ax.set_ylabel(clean_display_label(y_label))
    ax.set_title(clean_display_label(_plot_title("vector_field", cube_spec, projection, audience, title_mode)))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig


def compute_vector_field_table(
    df: pd.DataFrame,
    cube_spec: PhaseSpaceCubeSpec,
    projection: str = "x_y",
    bins: int = 10,
    weight_variable: str = "X_observed",
    min_count: int = 30,
) -> pd.DataFrame:
    """Compute binned average movement vectors for a 2D projection."""
    x_col, y_col, _x_label, _y_label = _projection_columns(cube_spec, projection)
    data = df.copy(deep=True).sort_values(["country_sector", "Year"])
    for column in [x_col, y_col, weight_variable]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data["x_current"] = data[x_col]
    data["y_current"] = data[y_col]
    data["x_next"] = _next_or_shift(data, x_col)
    data["y_next"] = _next_or_shift(data, y_col)
    data["delta_x"] = data["x_next"] - data["x_current"]
    data["delta_y"] = data["y_next"] - data["y_current"]
    valid = data[["x_current", "y_current", "delta_x", "delta_y"]].notna().all(axis=1)
    data = data.loc[valid].copy()
    if data.empty:
        return _empty_vector_table()
    data["weight"] = pd.to_numeric(data[weight_variable], errors="coerce") if weight_variable in data.columns else 1.0
    data["weight"] = data["weight"].where(data["weight"].gt(0), 1.0).fillna(1.0)
    data["x_bin"] = pd.cut(data["x_current"], bins=bins, duplicates="drop")
    data["y_bin"] = pd.cut(data["y_current"], bins=bins, duplicates="drop")
    rows = []
    for (x_bin, y_bin), group in data.groupby(["x_bin", "y_bin"], observed=True):
        if len(group) < min_count:
            continue
        rows.append(
            {
                "cube_slug": cube_spec.slug,
                "projection": projection,
                "x_variable": x_col,
                "y_variable": y_col,
                "x_bin": str(x_bin),
                "y_bin": str(y_bin),
                "x_center": float(group["x_current"].mean()),
                "y_center": float(group["y_current"].mean()),
                "delta_x_mean": _weighted_average(group["delta_x"], group["weight"]),
                "delta_y_mean": _weighted_average(group["delta_y"], group["weight"]),
                "observation_count": int(len(group)),
                "weight_sum": float(group["weight"].sum()),
            }
        )
    return pd.DataFrame(rows, columns=_empty_vector_table().columns)


def write_selected_node_tables(df: pd.DataFrame, output_dir: Path, start_year: int, end_year: int, top_n: int = 25) -> dict[str, Path]:
    """Write top-output and top-emissions selected-node tables for trajectory plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = {
        "top_output": output_dir / f"selected_nodes_top{top_n}_output_{start_year}_{end_year}.csv",
        "top_emissions": output_dir / f"selected_nodes_top{top_n}_emissions_{start_year}_{end_year}.csv",
    }
    _selected_node_summary(df, "X_observed", top_n).to_csv(written["top_output"], index=False)
    _selected_node_summary(df, "emissions_observed", top_n).to_csv(written["top_emissions"], index=False)
    return written


def build_movement_summary(df: pd.DataFrame, start_year: int, end_year: int, top_n: int = 25) -> pd.DataFrame:
    """Build trajectory movement diagnostics for global, sector, and selected-node units."""
    rows: list[dict[str, object]] = []
    for cube_spec in PHASE_SPACE_CUBE_SPECS.values():
        if not set(cube_spec.required_columns).issubset(df.columns):
            continue
        global_trajectory = aggregate_by_year(df, cube_spec)
        rows.append(_movement_row(global_trajectory, "global", "global", cube_spec, start_year, end_year))
        if "Sector" in df.columns:
            sector_trajectory = aggregate_by_group_year(df, ["Sector"], cube_spec)
            for sector, group in sector_trajectory.groupby("Sector", sort=False):
                rows.append(_movement_row(group, "sector", str(sector), cube_spec, start_year, end_year))
        if cube_spec.slug in {"green_readiness_incoming", "green_readiness_outgoing"}:
            for filter_name in ["top_output", "top_emissions"]:
                selected = select_node_trajectories(df, filter_name, top_n)
                for node, group in selected.groupby("country_sector", sort=False):
                    rows.append(_movement_row(group, "country_sector", str(node), cube_spec, start_year, end_year))
    return pd.DataFrame(rows)


def select_sectors_for_display(df: pd.DataFrame, cube_spec: PhaseSpaceCubeSpec, top_n: int = 8) -> pd.Index:
    """Select sectors by movement relevance for portfolio readability."""
    if "Sector" not in df.columns:
        return pd.Index([])
    sector_trajectory = aggregate_by_group_year(df.copy(deep=True), ["Sector"], cube_spec)
    scores = []
    for sector, group in sector_trajectory.groupby("Sector", sort=False):
        row = _movement_row(group, "sector", str(sector), cube_spec, int(group["Year"].min()), int(group["Year"].max()))
        movement_score = row["path_length"]
        if not np.isfinite(movement_score):
            movement_score = abs(row["x_delta"]) + abs(row["y_delta"]) + abs(row["z_delta"])
        scores.append({"Sector": sector, "movement_score": movement_score})
    if not scores:
        return pd.Index([])
    score_frame = pd.DataFrame(scores).sort_values("movement_score", ascending=False)
    return pd.Index(score_frame.head(top_n)["Sector"].astype(str))


def select_node_trajectories(
    df: pd.DataFrame,
    node_filter: str | Callable[[pd.DataFrame], pd.Series] = "top_output",
    top_n: int = 25,
) -> pd.DataFrame:
    """Filter node trajectories to top-output or top-emissions nodes."""
    data = df.copy(deep=True)
    if callable(node_filter):
        mask = node_filter(data)
        return data.loc[mask].copy()
    if node_filter in {"top_output", "top25_output"} and "is_top25_by_output_over_period" in data.columns and top_n == 25:
        flagged = data.loc[data["is_top25_by_output_over_period"].fillna(False).astype(bool)].copy()
        if flagged["country_sector"].nunique() >= top_n:
            return flagged
    if node_filter in {"top_emissions", "top25_emissions"} and "is_top25_by_emissions_over_period" in data.columns and top_n == 25:
        flagged = data.loc[data["is_top25_by_emissions_over_period"].fillna(False).astype(bool)].copy()
        if flagged["country_sector"].nunique() >= top_n:
            return flagged
    value_column = "emissions_observed" if "emissions" in str(node_filter) else "X_observed"
    if value_column not in data.columns:
        return data.iloc[0:0].copy()
    top_nodes = _group_totals(data, "country_sector", value_column).head(top_n).index.astype(str)
    return data.loc[data["country_sector"].astype(str).isin(set(top_nodes))].copy()


def compare_green_readiness_vector_fields(
    incoming_table: pd.DataFrame,
    outgoing_table: pd.DataFrame,
    projection: str,
    incoming_vector_file: Path | str,
    outgoing_vector_file: Path | str,
) -> pd.DataFrame:
    """Compare incoming and outgoing vector fields on common bins."""
    key_columns = ["x_bin", "y_bin"]
    if incoming_table.empty or outgoing_table.empty:
        common = pd.DataFrame()
    else:
        common = incoming_table.merge(outgoing_table, on=key_columns, suffixes=("_incoming", "_outgoing"))
    same_projection_explanation = (
        "Incoming and outgoing versions share the same XY axes; differences only appear in z-aware plots or YZ/XZ projections."
        if projection == "x_y"
        else "This projection includes the network green-ness axis, so incoming and outgoing fields can differ."
    )
    row = {
        "projection": projection,
        "incoming_vector_file": str(incoming_vector_file),
        "outgoing_vector_file": str(outgoing_vector_file),
        "n_common_bins": int(len(common)),
        "mean_abs_delta_dx": _mean_abs_difference(common, "delta_x_mean_incoming", "delta_x_mean_outgoing"),
        "mean_abs_delta_dy": _mean_abs_difference(common, "delta_y_mean_incoming", "delta_y_mean_outgoing"),
        "correlation_dx": _correlation(common, "delta_x_mean_incoming", "delta_x_mean_outgoing"),
        "correlation_dy": _correlation(common, "delta_y_mean_incoming", "delta_y_mean_outgoing"),
        "same_projection_explanation": same_projection_explanation,
        "interpretation_warning": "Treat vector fields as historical binned diagnostics, not forecasts or causal transition rules.",
    }
    return pd.DataFrame([row])


@dataclass
class PhaseSpacePlotBuilder:
    """Write the first ABM v3 phase-space plotting layer from an existing state panel."""

    state_panel: Path | str = PHASE_SPACE_PANEL_PATH
    output_dir: Path | str = PHASE_SPACE_PLOT_DIR
    audience: str = "both"
    color_mode: str = "default"
    plot_3d: bool = True
    plot_2d: bool = True
    plot_vector_fields: bool = True
    top_n: int = 25
    top_sector_n: int = 8
    top_node_n: int = 10
    research_top_node_n: int = 25
    title_mode: str = "theory"
    mark_years: tuple[int, ...] = DEFAULT_MARK_YEARS
    validate_vector_fields: bool = True
    write_movement_diagnostics: bool = True
    include_global: bool = True
    include_sector: bool = True
    include_node: bool = True
    strict: bool = False

    def build(self, start_year: int = 1995, end_year: int = 2016) -> dict[str, Path]:
        """Read the existing state panel and write plots, vector tables, selected nodes, and documentation."""
        state_panel_path = Path(self.state_panel)
        if not state_panel_path.exists():
            raise FileNotFoundError(f"Phase-space state panel not found: {state_panel_path}")
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        data = pd.read_parquet(state_panel_path)
        data = self._filter_years(data, start_year, end_year)
        audiences = ["portfolio", "research"] if self.audience == "both" else [self.audience]
        manifest_rows: list[dict[str, object]] = []
        if self.include_node:
            write_selected_node_tables(data, SELECTED_NODE_DIR, start_year, end_year, top_n=self.research_top_node_n)
        movement_path = SELECTED_NODE_DIR / f"phase_space_movement_summary_{start_year}_{end_year}.csv"
        if self.write_movement_diagnostics:
            movement_path.parent.mkdir(parents=True, exist_ok=True)
            build_movement_summary(data, start_year, end_year, top_n=self.research_top_node_n).to_csv(movement_path, index=False)
        for cube_spec in PHASE_SPACE_CUBE_SPECS.values():
            missing = [column for column in cube_spec.required_columns if column not in data.columns]
            if missing:
                if self.strict:
                    raise ValueError(f"Missing required columns for {cube_spec.slug}: {missing}")
                manifest_rows.append(
                    self._manifest_row(
                        "",
                        "skipped",
                        cube_spec,
                        "",
                        "",
                        "",
                        f"Missing required columns: {missing}",
                        "skipped",
                    )
                )
                continue
            for audience in audiences:
                self._write_cube_plots(data, cube_spec, audience, start_year, end_year, output_path, manifest_rows)
        if self.validate_vector_fields:
            comparison_path = self._write_vector_field_comparison(start_year, end_year)
        else:
            comparison_path = None
        manifest_path = output_path / f"phase_space_plot_manifest_{start_year}_{end_year}.csv"
        readme_path = output_path / f"phase_space_plot_readme_{start_year}_{end_year}.md"
        recommendations_path = output_path / f"phase_space_figure_recommendations_{start_year}_{end_year}.csv"
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
        build_figure_recommendations(pd.DataFrame(manifest_rows)).to_csv(recommendations_path, index=False)
        readme_path.write_text(build_phase_space_plot_readme(start_year, end_year, manifest_rows), encoding="utf-8")
        written = {"manifest": manifest_path, "readme": readme_path, "recommendations": recommendations_path}
        if self.write_movement_diagnostics:
            written["movement_summary"] = movement_path
        if comparison_path is not None:
            written["vector_field_comparison"] = comparison_path
        return written

    def _filter_years(self, data: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
        filtered = data.copy(deep=True)
        filtered["Year"] = pd.to_numeric(filtered["Year"], errors="coerce")
        return filtered.loc[filtered["Year"].between(start_year, end_year)].copy()

    def _write_cube_plots(
        self,
        data: pd.DataFrame,
        cube_spec: PhaseSpaceCubeSpec,
        audience: str,
        start_year: int,
        end_year: int,
        output_path: Path,
        manifest_rows: list[dict[str, object]],
    ) -> None:
        ext = "png" if audience == "portfolio" else "svg"
        node_top_n = self.top_node_n if audience == "portfolio" else self.research_top_node_n
        if self.plot_3d and self.include_global:
            path = output_path / f"phase_space_global_{cube_spec.slug}_3d_{start_year}_{end_year}_{audience}.{ext}"
            fig = plot_3d_global_trajectory(data, cube_spec, audience=audience, color_mode=self.color_mode, output_path=path, title_mode=self.title_mode, mark_years=self.mark_years)
            plt.close(fig)
            manifest_rows.append(self._manifest_row(path, "3d_trajectory", cube_spec, "global", audience, "trajectory_weight_output", "all country-sector nodes", "written", projection="3d"))
        if self.plot_3d and self.include_sector and cube_spec.slug in {"green_readiness_incoming", "green_readiness_outgoing"}:
            path = output_path / f"phase_space_sector_{cube_spec.slug}_3d_{start_year}_{end_year}_{audience}.{ext}"
            fig = plot_3d_sector_trajectories(data, cube_spec, audience=audience, color_mode=self.color_mode, output_path=path, top_n=self.top_sector_n, title_mode=self.title_mode, mark_years=self.mark_years)
            plt.close(fig)
            manifest_rows.append(self._manifest_row(path, "3d_trajectory", cube_spec, "sector", audience, "trajectory_weight_output", f"top {self.top_sector_n} sectors by movement relevance" if audience == "portfolio" else "all sectors with transparency", "written", projection="3d"))
        if self.plot_3d and self.include_node and cube_spec.slug == "green_readiness_incoming":
            for filter_name, metric_label in [("top_output", "output"), ("top_emissions", "emissions")]:
                label = f"top{node_top_n}_{metric_label}"
                path = output_path / f"phase_space_nodes_{label}_{cube_spec.slug}_3d_{start_year}_{end_year}_{audience}.{ext}"
                fig = plot_3d_node_trajectories(data, cube_spec, node_filter=filter_name, audience=audience, color_mode=self.color_mode, output_path=path, top_n=node_top_n, title_mode=self.title_mode, mark_years=self.mark_years)
                plt.close(fig)
                manifest_rows.append(self._manifest_row(path, "3d_trajectory", cube_spec, label, audience, "trajectory_weight_output", f"top {node_top_n} nodes by {metric_label}", "written", projection="3d"))
        if self.plot_2d and cube_spec.slug in {"green_readiness_incoming", "production_safe_incoming"}:
            path = output_path / f"phase_space_global_{cube_spec.slug}_xy_{start_year}_{end_year}_{audience}.{ext}"
            fig = plot_2d_projection_trajectory(data, cube_spec, "x_y", "global", audience=audience, color_mode=self.color_mode, output_path=path, top_n=node_top_n, title_mode=self.title_mode, mark_years=self.mark_years)
            plt.close(fig)
            manifest_rows.append(self._manifest_row(path, "2d_projection", cube_spec, "global", audience, "trajectory_weight_output", "x_y projection", "written", projection="x_y"))
        if self.plot_vector_fields and cube_spec.slug in {"green_readiness_incoming", "green_readiness_outgoing"}:
            projections = ["x_y"] if audience == "portfolio" else ["x_y", "x_z", "y_z"]
            for projection in projections:
                suffix = projection.replace("_", "")
                path = output_path / f"phase_space_vector_field_{cube_spec.slug}_{suffix}_{start_year}_{end_year}_{audience}.{ext}"
                vector_table = compute_vector_field_table(data, cube_spec, projection=projection, bins=10, weight_variable=_weight_column(data), min_count=30)
                vector_path = VECTOR_FIELD_DIR / f"vector_field_{cube_spec.slug}_{projection}_{start_year}_{end_year}.csv"
                vector_path.parent.mkdir(parents=True, exist_ok=True)
                vector_table.to_csv(vector_path, index=False)
                fig = plot_vector_field(data, cube_spec, projection=projection, bins=10, weight_variable=_weight_column(data), audience=audience, color_mode=self.color_mode, output_path=path, min_count=30, title_mode=self.title_mode)
                plt.close(fig)
                manifest_rows.append(self._manifest_row(path, "vector_field", cube_spec, "country_sector-year movement", audience, _weight_column(data), f"binned {projection} movement; table={vector_path}", "written", projection=projection))

    def _write_vector_field_comparison(self, start_year: int, end_year: int) -> Path | None:
        incoming_path = VECTOR_FIELD_DIR / f"vector_field_green_readiness_incoming_x_y_{start_year}_{end_year}.csv"
        outgoing_path = VECTOR_FIELD_DIR / f"vector_field_green_readiness_outgoing_x_y_{start_year}_{end_year}.csv"
        if not incoming_path.exists() or not outgoing_path.exists():
            return None
        incoming = pd.read_csv(incoming_path)
        outgoing = pd.read_csv(outgoing_path)
        comparison = compare_green_readiness_vector_fields(incoming, outgoing, "x_y", incoming_path, outgoing_path)
        output_path = VECTOR_FIELD_DIR / f"vector_field_comparison_green_readiness_xy_{start_year}_{end_year}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(output_path, index=False)
        return output_path

    def _manifest_row(
        self,
        plot_file: Path | str,
        plot_family: str,
        cube_spec: PhaseSpaceCubeSpec,
        unit: str,
        audience: str,
        weight_variable: str,
        filter_description: str,
        status: str,
        projection: str = "",
    ) -> dict[str, object]:
        title_key = _title_key(unit, cube_spec, projection, plot_family)
        title = _registry_title(title_key, audience, cube_spec, title_mode=self.title_mode)
        caption_note = _caption_note(unit, cube_spec, plot_family)
        tier, recommendation_status, known_limitation = _recommendation_for_row(plot_family, unit, cube_spec, projection)
        return {
            "plot_file": str(plot_file),
            "plot_family": plot_family,
            "cube_name": cube_spec.name,
            "unit": unit,
            "audience": audience,
            "color_mode": self.color_mode,
            "x_variable": cube_spec.x,
            "y_variable": cube_spec.y,
            "z_variable": cube_spec.z,
            "weight_variable": weight_variable,
            "filter_description": filter_description,
            "interpretation_note": cube_spec.interpretation_note,
            "caveats": cube_spec.caveats,
            "title": title,
            "caption_note": caption_note,
            "figure_tier": tier,
            "recommendation_status": recommendation_status,
            "interpretation_message": _interpretation_message(unit, cube_spec, plot_family),
            "display_transform": "display labels cleaned only; source data unchanged",
            "selection_rule": filter_description,
            "known_limitation": known_limitation,
            "status": status,
        }


def build_phase_space_plot_readme(start_year: int, end_year: int, manifest_rows: list[dict[str, object]]) -> str:
    """Build a short Markdown note for generated phase-space plot outputs."""
    written_count = sum(1 for row in manifest_rows if row.get("status") == "written")
    skipped_count = sum(1 for row in manifest_rows if row.get("status") == "skipped")
    return "\n".join(
        [
            f"# ABM v3 Phase-Space Plot Readme ({start_year}-{end_year})",
            "",
            "## What These Plots Are",
            "These plots are historical state-space diagnostics. They are not forecasts, not causal transition rules, and not scenario overlays.",
            "",
            "They show whether the country-sector production network moves through higher-green states in an uneven, path-dependent, capability-constrained way.",
            "",
            "## Interpretive Hierarchy",
            "- Global 2D plots: main thesis story.",
            "- Outgoing 3D plots: relational green-ness story.",
            "- Production-safe plots: anti-collapse check.",
            "- Sector plots: heterogeneity and capability constraints.",
            "- Node plots: concentration and persistence.",
            "- Vector fields: diagnostic movement geometry.",
            "",
            "## Title Logic",
            "Titles are theory-led but cautious. Portfolio titles are finding-based draft candidates. Research titles are variable-specific and more technical.",
            "",
            "## Caveats",
            "`g_in_network` and `g_out_network` are available directional network green-ness proxies. They are not the missing canonical aggregate target `network_green_exposure`.",
            "",
            "`brown_centrality` and `capability_ecosystem_exposure` remain future target axes and are not used in this first plotting layer.",
            "",
            "Sector and node plots are filtered views. Vector fields are historical binned summaries.",
            "",
            "Portfolio plots are draft candidates for later refinement; they should not be treated as final portfolio claims.",
            "",
            f"Plots written: {written_count}. Skipped requests: {skipped_count}.",
            "",
            "The implemented cubes are green-readiness incoming/outgoing and production-safe greening incoming/outgoing.",
        ]
    )


def build_figure_recommendations(manifest: pd.DataFrame) -> pd.DataFrame:
    """Create a figure recommendation manifest from plot manifest rows."""
    rows = []
    for _, row in manifest.iterrows():
        if str(row.get("status", "")) != "written":
            continue
        tier = row.get("figure_tier", "diagnostic")
        status = row.get("recommendation_status", "diagnostic")
        reason = _recommendation_reason(row)
        rows.append(
            {
                "plot_file": row.get("plot_file", ""),
                "figure_tier": tier,
                "recommendation_status": status,
                "reason": reason,
                "suggested_use": _suggested_use(str(tier)),
                "title": row.get("title", ""),
                "caption_note": row.get("caption_note", ""),
                "known_limitation": row.get("known_limitation", ""),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "plot_file",
            "figure_tier",
            "recommendation_status",
            "reason",
            "suggested_use",
            "title",
            "caption_note",
            "known_limitation",
        ],
    )


def aggregate_by_year(df: pd.DataFrame, cube_spec: PhaseSpaceCubeSpec) -> pd.DataFrame:
    """Aggregate country-sector rows to one output-weighted point per year."""
    weight_col = _weight_column(df)
    rows = []
    for year, group in df.groupby("Year", dropna=False):
        rows.append(
            {
                "Year": int(year),
                cube_spec.x: weighted_mean(group, cube_spec.x, weight_col),
                cube_spec.y: weighted_mean(group, cube_spec.y, weight_col),
                cube_spec.z: weighted_mean(group, cube_spec.z, weight_col),
            }
        )
    return pd.DataFrame(rows).sort_values("Year")


def aggregate_by_group_year(df: pd.DataFrame, group_cols: list[str], cube_spec: PhaseSpaceCubeSpec) -> pd.DataFrame:
    """Aggregate country-sector rows to output-weighted group-year points."""
    weight_col = _weight_column(df)
    rows = []
    for keys, group in df.groupby(group_cols + ["Year"], dropna=False):
        keys_tuple = keys if isinstance(keys, tuple) else (keys,)
        row = {column: keys_tuple[index] for index, column in enumerate(group_cols)}
        row["Year"] = int(keys_tuple[-1])
        row[cube_spec.x] = weighted_mean(group, cube_spec.x, weight_col)
        row[cube_spec.y] = weighted_mean(group, cube_spec.y, weight_col)
        row[cube_spec.z] = weighted_mean(group, cube_spec.z, weight_col)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols + ["Year"])


def parse_mark_years(value: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Parse comma-separated anchor years."""
    if isinstance(value, tuple):
        return tuple(int(year) for year in value)
    if isinstance(value, list):
        return tuple(int(year) for year in value)
    return tuple(int(token.strip()) for token in str(value).split(",") if token.strip())


def _movement_row(
    trajectory: pd.DataFrame,
    unit_type: str,
    unit_id: str,
    cube_spec: PhaseSpaceCubeSpec,
    start_year: int,
    end_year: int,
) -> dict[str, object]:
    """Compute movement geometry for one trajectory."""
    group = trajectory.sort_values("Year").copy()
    coords = group[[cube_spec.x, cube_spec.y, cube_spec.z]].apply(pd.to_numeric, errors="coerce")
    valid = coords.notna().all(axis=1)
    group = group.loc[valid].copy()
    coords = coords.loc[valid]
    if coords.empty:
        return _empty_movement_row(unit_type, unit_id, cube_spec, start_year, end_year)
    first = coords.iloc[0]
    last = coords.iloc[-1]
    deltas = coords.diff().dropna()
    net = last - first
    net_displacement = float(np.linalg.norm(net.to_numpy(dtype=float)))
    path_length = float(np.linalg.norm(deltas.to_numpy(dtype=float), axis=1).sum()) if not deltas.empty else 0.0
    ratio = float(net_displacement / path_length) if path_length > 0 else np.nan
    row = {
        "unit_type": unit_type,
        "unit_id": unit_id,
        "cube_slug": cube_spec.slug,
        "start_year": int(group["Year"].iloc[0]) if "Year" in group.columns else start_year,
        "end_year": int(group["Year"].iloc[-1]) if "Year" in group.columns else end_year,
        "x_start": float(first[cube_spec.x]),
        "x_end": float(last[cube_spec.x]),
        "x_delta": float(net[cube_spec.x]),
        "y_start": float(first[cube_spec.y]),
        "y_end": float(last[cube_spec.y]),
        "y_delta": float(net[cube_spec.y]),
        "z_start": float(first[cube_spec.z]),
        "z_end": float(last[cube_spec.z]),
        "z_delta": float(net[cube_spec.z]),
        "net_displacement": net_displacement,
        "path_length": path_length,
        "displacement_to_path_ratio": ratio,
        "turning_intensity": float(1.0 - ratio) if np.isfinite(ratio) else np.nan,
        "monotonicity_x": _monotonicity(coords[cube_spec.x]),
        "monotonicity_y": _monotonicity(coords[cube_spec.y]),
        "monotonicity_z": _monotonicity(coords[cube_spec.z]),
    }
    row["interpretation_flag"] = _movement_flag(row, cube_spec)
    return row


def _empty_movement_row(unit_type: str, unit_id: str, cube_spec: PhaseSpaceCubeSpec, start_year: int, end_year: int) -> dict[str, object]:
    return {
        "unit_type": unit_type,
        "unit_id": unit_id,
        "cube_slug": cube_spec.slug,
        "start_year": start_year,
        "end_year": end_year,
        "x_start": np.nan,
        "x_end": np.nan,
        "x_delta": np.nan,
        "y_start": np.nan,
        "y_end": np.nan,
        "y_delta": np.nan,
        "z_start": np.nan,
        "z_end": np.nan,
        "z_delta": np.nan,
        "net_displacement": np.nan,
        "path_length": np.nan,
        "displacement_to_path_ratio": np.nan,
        "turning_intensity": np.nan,
        "monotonicity_x": np.nan,
        "monotonicity_y": np.nan,
        "monotonicity_z": np.nan,
        "interpretation_flag": "weak_net_movement",
    }


def _monotonicity(values: pd.Series) -> float:
    deltas = pd.to_numeric(values, errors="coerce").diff().dropna()
    if deltas.empty:
        return np.nan
    net_delta = float(values.iloc[-1] - values.iloc[0])
    if abs(net_delta) < 1e-12:
        return np.nan
    if net_delta > 0:
        return float(deltas.gt(0).mean())
    return float(deltas.lt(0).mean())


def _movement_flag(row: dict[str, object], cube_spec: PhaseSpaceCubeSpec) -> str:
    y_delta = float(row.get("y_delta", np.nan))
    z_delta = float(row.get("z_delta", np.nan))
    x_delta = float(row.get("x_delta", np.nan))
    ratio = float(row.get("displacement_to_path_ratio", np.nan))
    if not np.isfinite(row.get("net_displacement", np.nan)) or float(row["net_displacement"]) < 1e-9:
        return "weak_net_movement"
    if np.isfinite(ratio) and ratio < 0.55 and y_delta > 0:
        return "winding_greening"
    if y_delta > 0 and z_delta <= 0 and "green_readiness" in cube_spec.slug:
        return "local_greening_without_network_greening"
    if y_delta > 0 and x_delta >= 0 and "production_safe" in cube_spec.slug:
        return "production_safe_greening"
    if abs(z_delta) > abs(y_delta) * 1.5 and "green_readiness" in cube_spec.slug:
        return "volatile_network_embedding"
    if y_delta > 0:
        return "directional_greening"
    return "weak_net_movement"


def _plot_title(unit: str, cube_spec: PhaseSpaceCubeSpec, projection: str, audience: str, title_mode: str) -> str:
    if title_mode == "technical":
        return cube_spec.title_research if audience != "portfolio" else cube_spec.title_portfolio
    return _registry_title(_title_key(unit, cube_spec, projection, ""), audience, cube_spec, title_mode)


def _registry_title(title_key: str, audience: str, cube_spec: PhaseSpaceCubeSpec, title_mode: str) -> str:
    if title_mode == "technical":
        return cube_spec.title_research if audience != "portfolio" else cube_spec.title_portfolio
    entry = TITLE_REGISTRY.get(title_key)
    if entry is None:
        return cube_spec.title_portfolio if audience == "portfolio" else cube_spec.title_research
    return entry.get(audience, entry.get("research", cube_spec.title_research))


def _title_key(unit: str, cube_spec: PhaseSpaceCubeSpec, projection: str, plot_family: str) -> str:
    if plot_family == "vector_field" or unit == "vector_field":
        return "vector_field_green_readiness"
    if unit == "sector":
        return "sector_green_readiness"
    if "top_emissions" in unit:
        return "nodes_top_emissions_green_readiness"
    if "top_output" in unit:
        return "nodes_top_output_green_readiness"
    if "green_readiness" in cube_spec.slug and projection == "x_y":
        return "global_green_readiness_xy"
    if cube_spec.slug == "green_readiness_incoming":
        return "global_green_readiness_incoming_3d"
    if cube_spec.slug == "green_readiness_outgoing":
        return "global_green_readiness_outgoing_3d"
    if "production_safe" in cube_spec.slug and projection == "x_y":
        return "global_production_safe_xy"
    if "production_safe" in cube_spec.slug:
        return "global_production_safe_3d"
    return ""


def _caption_note(unit: str, cube_spec: PhaseSpaceCubeSpec, plot_family: str) -> str:
    if plot_family == "vector_field":
        return CAPTION_REGISTRY["vector_field"]
    if "production_safe" in cube_spec.slug:
        return CAPTION_REGISTRY["production_safe"]
    if unit in {"sector"} or "top" in unit:
        return CAPTION_REGISTRY["sector_node"]
    return CAPTION_REGISTRY["global"]


def _recommendation_for_row(plot_family: str, unit: str, cube_spec: PhaseSpaceCubeSpec, projection: str) -> tuple[str, str, str]:
    if plot_family == "vector_field":
        return "diagnostic", "diagnostic-until-validated", "Vector fields are binned historical summaries; XY incoming and outgoing can be similar because z is not used."
    if unit == "global" and plot_family == "2d_projection" and cube_spec.slug in {"green_readiness_incoming", "production_safe_incoming"}:
        return "thesis-core", "recommended-draft", "Use with directional-proxy caveat."
    if unit == "global" and cube_spec.slug == "green_readiness_outgoing":
        return "thesis-core", "recommended-draft", "Outgoing network green-ness is directional, not aggregate network_green_exposure."
    if unit == "global":
        return "research-support", "supporting", "Use as contextual 3D trajectory."
    if unit == "sector" or "top" in unit:
        return "research-support", "supporting", "Filtered view; not all sectors or nodes are displayed."
    return "diagnostic", "diagnostic", "Interpret cautiously."


def _interpretation_message(unit: str, cube_spec: PhaseSpaceCubeSpec, plot_family: str) -> str:
    if plot_family == "vector_field":
        return "Average historical movement in binned state space; diagnostic only."
    if "production_safe" in cube_spec.slug:
        return "Checks whether greening coexists with production resilience."
    if unit == "sector":
        return "Shows heterogeneous sector corridors in available green-readiness space."
    if "top" in unit:
        return "Shows concentration and persistence among selected country-sector nodes."
    return "Shows output-weighted movement through available green-readiness space."


def _recommendation_reason(row: pd.Series) -> str:
    if row.get("figure_tier") == "thesis-core":
        return "Directly supports the main phase-space interpretation."
    if row.get("figure_tier") == "research-support":
        return "Useful for documenting heterogeneity, directional network proxies, or production-safe checks."
    return "Use mainly as a diagnostic until movement geometry is validated."


def _suggested_use(tier: str) -> str:
    if tier == "thesis-core":
        return "main thesis figure candidate"
    if tier == "research-support":
        return "methods or appendix support"
    return "diagnostic review only"


def _mean_abs_difference(df: pd.DataFrame, left: str, right: str) -> float:
    if df.empty or left not in df.columns or right not in df.columns:
        return np.nan
    return float((pd.to_numeric(df[left], errors="coerce") - pd.to_numeric(df[right], errors="coerce")).abs().mean())


def _correlation(df: pd.DataFrame, left: str, right: str) -> float:
    if df.empty or left not in df.columns or right not in df.columns or len(df) < 2:
        return np.nan
    return float(pd.to_numeric(df[left], errors="coerce").corr(pd.to_numeric(df[right], errors="coerce")))


def _validate_plot_options(audience: str, color_mode: str) -> None:
    if audience not in {"portfolio", "research", "diagnostic"}:
        raise ValueError("audience must be 'portfolio', 'research', or 'diagnostic'.")
    if color_mode not in {"default", "colorblind"}:
        raise ValueError("color_mode must be 'default' or 'colorblind'.")


def _palette(color_mode: str, count: int) -> list[str]:
    colors = COLORBLIND_COLORS if color_mode == "colorblind" else DEFAULT_COLORS
    return [colors[index % len(colors)] for index in range(count)]


def _save_figure(fig, output_path: Path | None) -> None:
    if output_path is None:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")


def _weight_column(df: pd.DataFrame) -> str | None:
    if "trajectory_weight_output" in df.columns:
        return "trajectory_weight_output"
    if "X_observed" in df.columns:
        return "X_observed"
    return None


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return float(values.mean(skipna=True))
    return float(np.average(values.loc[valid], weights=weights.loc[valid]))


def _next_or_shift(data: pd.DataFrame, column: str) -> pd.Series:
    next_column = f"{column}_next"
    if next_column in data.columns:
        return pd.to_numeric(data[next_column], errors="coerce")
    return data.groupby("country_sector", sort=False)[column].shift(-1)


def _projection_columns(cube_spec: PhaseSpaceCubeSpec, projection: str) -> tuple[str, str, str, str]:
    if projection == "x_y":
        return cube_spec.x, cube_spec.y, cube_spec.x_label, cube_spec.y_label
    if projection == "x_z":
        return cube_spec.x, cube_spec.z, cube_spec.x_label, cube_spec.z_label
    if projection == "y_z":
        return cube_spec.y, cube_spec.z, cube_spec.y_label, cube_spec.z_label
    raise ValueError("projection must be one of: x_y, x_z, y_z.")


def _title(cube_spec: PhaseSpaceCubeSpec, audience: str) -> str:
    return cube_spec.title_portfolio if audience == "portfolio" else cube_spec.title_research


def _plot_3d_line(
    ax,
    trajectory: pd.DataFrame,
    cube_spec: PhaseSpaceCubeSpec,
    color: str,
    label: str,
    marker: str,
    linestyle: str = "-",
    alpha: float = 0.9,
    linewidth: float = 1.8,
    mark_years: tuple[int, ...] = DEFAULT_MARK_YEARS,
) -> None:
    if trajectory.empty:
        return
    ax.plot(
        trajectory[cube_spec.x],
        trajectory[cube_spec.y],
        trajectory[cube_spec.z],
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
        markevery=_anchor_year_mark_indices(trajectory, mark_years),
    )


def _format_3d_axes(ax, cube_spec: PhaseSpaceCubeSpec, audience: str, title: str | None = None) -> None:
    ax.set_xlabel(clean_display_label(cube_spec.x_label), labelpad=8)
    ax.set_ylabel(clean_display_label(cube_spec.y_label), labelpad=8)
    ax.set_zlabel(clean_display_label(cube_spec.z_label), labelpad=8)
    ax.set_title(clean_display_label(title or _title(cube_spec, audience)), pad=18)
    ax.view_init(elev=24, azim=-52)
    ax.grid(True, alpha=0.25)


def _annotate_3d_anchor_years(
    ax,
    trajectory: pd.DataFrame,
    cube_spec: PhaseSpaceCubeSpec,
    mark_years: tuple[int, ...] = DEFAULT_MARK_YEARS,
) -> None:
    if trajectory.empty:
        return
    years = set(mark_years)
    selected = trajectory.loc[pd.to_numeric(trajectory["Year"], errors="coerce").astype("Int64").isin(years)]
    if selected.empty:
        selected = trajectory.iloc[[0, -1]]
    ax.scatter(selected[cube_spec.x], selected[cube_spec.y], selected[cube_spec.z], s=42, color="#111111", depthshade=False)
    for _, row in selected.iterrows():
        ax.text(row[cube_spec.x], row[cube_spec.y], row[cube_spec.z], str(int(row["Year"])), fontsize=8)


def _plot_2d_line(
    ax,
    trajectory: pd.DataFrame,
    x_col: str,
    y_col: str,
    color: str,
    label: str,
    marker: str,
    alpha: float = 0.9,
    mark_years: tuple[int, ...] = DEFAULT_MARK_YEARS,
) -> None:
    if trajectory.empty:
        return
    ax.plot(
        trajectory[x_col],
        trajectory[y_col],
        color=color,
        marker=marker,
        linewidth=1.8,
        alpha=alpha,
        label=label,
        markevery=_anchor_year_mark_indices(trajectory, mark_years),
    )


def _annotate_2d_anchor_years(
    ax,
    trajectory: pd.DataFrame,
    x_col: str,
    y_col: str,
    mark_years: tuple[int, ...] = DEFAULT_MARK_YEARS,
) -> None:
    if trajectory.empty:
        return
    years = set(mark_years)
    selected = trajectory.loc[pd.to_numeric(trajectory["Year"], errors="coerce").astype("Int64").isin(years)]
    if selected.empty:
        selected = trajectory.iloc[[0, -1]]
    ax.scatter(selected[x_col], selected[y_col], s=42, color="#111111", zorder=5)
    for _, row in selected.iterrows():
        ax.text(row[x_col], row[y_col], f" {int(row['Year'])}", fontsize=8)


def _anchor_year_mark_indices(trajectory: pd.DataFrame, mark_years: tuple[int, ...] = DEFAULT_MARK_YEARS) -> list[int]:
    years = pd.to_numeric(trajectory["Year"], errors="coerce").tolist()
    anchor_years = set(mark_years)
    indices = [index for index, year in enumerate(years) if np.isfinite(year) and int(year) in anchor_years]
    if 0 not in indices and years:
        indices.insert(0, 0)
    if years and len(years) - 1 not in indices:
        indices.append(len(years) - 1)
    return sorted(set(indices))


def _group_totals(df: pd.DataFrame, group_col: str, value_col: str | None) -> pd.Series:
    if value_col is None or value_col not in df.columns:
        return pd.Series(dtype=float)
    data = df[[group_col, value_col]].copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    return data.groupby(group_col)[value_col].sum(min_count=1).sort_values(ascending=False)


def _short_sector_label(value: object) -> str:
    text = clean_display_label(value)
    words = [word for word in re.split(r"\W+", text) if word]
    if len(words) <= 2:
        return text[:22]
    return " ".join(words[:3])[:22]


def _short_node_label(value: object) -> str:
    text = clean_display_label(value)
    parts = [part.strip() for part in re.split(r"\||-", text) if part.strip()]
    if len(parts) >= 2:
        country = parts[0][:3]
        sector = _short_sector_label(parts[-1])
        return f"{country}: {sector}"
    return text[:28]


def _selected_node_summary(df: pd.DataFrame, value_col: str, top_n: int) -> pd.DataFrame:
    if value_col not in df.columns:
        return pd.DataFrame(
            columns=[
                "country_sector",
                "Country",
                "Sector",
                "total_X_observed",
                "total_emissions_observed",
                "mean_EI",
                "mean_g_local",
                "mean_green_capability_export_share",
                "mean_g_in_network",
                "mean_g_out_network",
            ]
        )
    top_nodes = _group_totals(df, "country_sector", value_col).head(top_n).index.astype(str)
    selected = df.loc[df["country_sector"].astype(str).isin(set(top_nodes))].copy()
    rows = []
    axis_columns = ["green_capability_export_share", "log_X_observed", "g_local", "g_in_network", "g_out_network"]
    for node, group in selected.groupby("country_sector", sort=False):
        group = group.sort_values("Year")
        row = {
            "country_sector": node,
            "Country": group["Country"].dropna().iloc[0] if "Country" in group.columns and group["Country"].notna().any() else "",
            "Sector": group["Sector"].dropna().iloc[0] if "Sector" in group.columns and group["Sector"].notna().any() else "",
            "total_X_observed": _sum_column(group, "X_observed"),
            "total_emissions_observed": _sum_column(group, "emissions_observed"),
            "mean_EI": _mean_column(group, "EI"),
            "mean_g_local": _mean_column(group, "g_local"),
            "mean_green_capability_export_share": _mean_column(group, "green_capability_export_share"),
            "mean_g_in_network": _mean_column(group, "g_in_network"),
            "mean_g_out_network": _mean_column(group, "g_out_network"),
        }
        for column in axis_columns:
            if column in group.columns and not group[column].dropna().empty:
                row[f"{column}_start"] = group[column].iloc[0]
                row[f"{column}_end"] = group[column].iloc[-1]
        rows.append(row)
    return pd.DataFrame(rows)


def _sum_column(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return np.nan
    return float(pd.to_numeric(df[column], errors="coerce").sum(skipna=True))


def _mean_column(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return np.nan
    return float(pd.to_numeric(df[column], errors="coerce").mean(skipna=True))


def _empty_vector_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "cube_slug",
            "projection",
            "x_variable",
            "y_variable",
            "x_bin",
            "y_bin",
            "x_center",
            "y_center",
            "delta_x_mean",
            "delta_y_mean",
            "observation_count",
            "weight_sum",
        ]
    )
