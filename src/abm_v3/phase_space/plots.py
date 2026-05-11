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
):
    """Plot one output-weighted global trajectory through a 3D phase space."""
    _validate_plot_options(audience, color_mode)
    data = df.copy(deep=True)
    trajectory = aggregate_by_year(data, cube_spec)
    fig = plt.figure(figsize=(8, 6) if audience == "portfolio" else (9, 7))
    ax = fig.add_subplot(111, projection="3d")
    color = _palette(color_mode, 1)[0]
    _plot_3d_line(ax, trajectory, cube_spec, color=color, label="Global", marker="o", linewidth=2.4)
    _annotate_3d_start_end(ax, trajectory, cube_spec)
    _format_3d_axes(ax, cube_spec, audience)
    _save_figure(fig, output_path)
    return fig


def plot_3d_sector_trajectories(
    df: pd.DataFrame,
    cube_spec: PhaseSpaceCubeSpec,
    audience: str = "research",
    color_mode: str = "default",
    output_path: Path | None = None,
    top_n: int = 8,
):
    """Plot output-weighted sector trajectories through a 3D phase space."""
    _validate_plot_options(audience, color_mode)
    data = df.copy(deep=True)
    if "Sector" not in data.columns:
        raise ValueError("Sector trajectory plot requires a Sector column.")
    sector_data = aggregate_by_group_year(data, ["Sector"], cube_spec)
    if audience == "portfolio":
        sector_totals = _group_totals(data, "Sector", _weight_column(data))
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
        )
        if audience == "portfolio" and not group.empty:
            last = group.iloc[-1]
            ax.text(last[cube_spec.x], last[cube_spec.y], last[cube_spec.z], _short_sector_label(sector), fontsize=7)
    _format_3d_axes(ax, cube_spec, audience)
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
        )
        if audience == "portfolio" and not group.empty and index < 10:
            last = group.iloc[-1]
            ax.text(last[cube_spec.x], last[cube_spec.y], last[cube_spec.z], _short_node_label(node), fontsize=6)
    _format_3d_axes(ax, cube_spec, audience)
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
):
    """Plot a 2D projection for global, sector, or selected-node trajectories."""
    _validate_plot_options(audience, color_mode)
    x_col, y_col, x_label, y_label = _projection_columns(cube_spec, projection)
    data = df.copy(deep=True)
    fig, ax = plt.subplots(figsize=(8, 5.5) if audience == "portfolio" else (9, 6.5))
    if unit == "global":
        trajectory = aggregate_by_year(data, cube_spec)
        _plot_2d_line(ax, trajectory, x_col, y_col, color=_palette(color_mode, 1)[0], label="Global", marker="o")
        _annotate_2d_start_end(ax, trajectory, x_col, y_col)
    elif unit == "sector":
        trajectory = aggregate_by_group_year(data, ["Sector"], cube_spec)
        groups = sorted(trajectory["Sector"].dropna().astype(str).unique())
        colors = _palette(color_mode, max(1, len(groups)))
        for index, sector in enumerate(groups[: (8 if audience == "portfolio" else len(groups))]):
            group = trajectory.loc[trajectory["Sector"].astype(str).eq(sector)].sort_values("Year")
            _plot_2d_line(ax, group, x_col, y_col, color=colors[index], label=clean_display_label(sector), marker=MARKERS[index % len(MARKERS)], alpha=0.8 if audience == "portfolio" else 0.45)
    else:
        filter_name = "top_emissions" if "emissions" in unit else "top_output"
        trajectory = select_node_trajectories(data, filter_name, top_n)
        groups = sorted(trajectory["country_sector"].dropna().astype(str).unique())
        colors = _palette(color_mode, max(1, len(groups)))
        for index, node in enumerate(groups):
            group = trajectory.loc[trajectory["country_sector"].astype(str).eq(node)].sort_values("Year")
            _plot_2d_line(ax, group, x_col, y_col, color=colors[index], label=_short_node_label(node), marker=MARKERS[index % len(MARKERS)], alpha=0.75 if audience == "portfolio" else 0.45)
    ax.set_xlabel(clean_display_label(x_label))
    ax.set_ylabel(clean_display_label(y_label))
    ax.set_title(clean_display_label(_title(cube_spec, audience)))
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
    ax.set_title(clean_display_label(f"Average historical movement field: {cube_spec.name}"))
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
        return data.loc[data["is_top25_by_output_over_period"].fillna(False).astype(bool)].copy()
    if node_filter in {"top_emissions", "top25_emissions"} and "is_top25_by_emissions_over_period" in data.columns and top_n == 25:
        return data.loc[data["is_top25_by_emissions_over_period"].fillna(False).astype(bool)].copy()
    value_column = "emissions_observed" if "emissions" in str(node_filter) else "X_observed"
    if value_column not in data.columns:
        return data.iloc[0:0].copy()
    top_nodes = _group_totals(data, "country_sector", value_column).head(top_n).index.astype(str)
    return data.loc[data["country_sector"].astype(str).isin(set(top_nodes))].copy()


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
            write_selected_node_tables(data, SELECTED_NODE_DIR, start_year, end_year, top_n=self.top_n)
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
        manifest_path = output_path / f"phase_space_plot_manifest_{start_year}_{end_year}.csv"
        readme_path = output_path / f"phase_space_plot_readme_{start_year}_{end_year}.md"
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
        readme_path.write_text(build_phase_space_plot_readme(start_year, end_year, manifest_rows), encoding="utf-8")
        return {"manifest": manifest_path, "readme": readme_path}

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
        if self.plot_3d and self.include_global:
            path = output_path / f"phase_space_global_{cube_spec.slug}_3d_{start_year}_{end_year}_{audience}.{ext}"
            fig = plot_3d_global_trajectory(data, cube_spec, audience=audience, color_mode=self.color_mode, output_path=path)
            plt.close(fig)
            manifest_rows.append(self._manifest_row(path, "3d_trajectory", cube_spec, "global", audience, "trajectory_weight_output", "all country-sector nodes", "written"))
        if self.plot_3d and self.include_sector and cube_spec.slug in {"green_readiness_incoming", "green_readiness_outgoing"}:
            path = output_path / f"phase_space_sector_{cube_spec.slug}_3d_{start_year}_{end_year}_{audience}.{ext}"
            fig = plot_3d_sector_trajectories(data, cube_spec, audience=audience, color_mode=self.color_mode, output_path=path)
            plt.close(fig)
            manifest_rows.append(self._manifest_row(path, "3d_trajectory", cube_spec, "sector", audience, "trajectory_weight_output", "sectors aggregated by year", "written"))
        if self.plot_3d and self.include_node and cube_spec.slug == "green_readiness_incoming":
            for filter_name, label in [("top_output", "top25_output"), ("top_emissions", "top25_emissions")]:
                path = output_path / f"phase_space_nodes_{label}_{cube_spec.slug}_3d_{start_year}_{end_year}_{audience}.{ext}"
                fig = plot_3d_node_trajectories(data, cube_spec, node_filter=filter_name, audience=audience, color_mode=self.color_mode, output_path=path, top_n=self.top_n)
                plt.close(fig)
                manifest_rows.append(self._manifest_row(path, "3d_trajectory", cube_spec, label, audience, "trajectory_weight_output", f"top {self.top_n} nodes by {'emissions' if 'emissions' in label else 'output'}", "written"))
        if self.plot_2d and cube_spec.slug in {"green_readiness_incoming", "production_safe_incoming"}:
            path = output_path / f"phase_space_global_{cube_spec.slug}_xy_{start_year}_{end_year}_{audience}.{ext}"
            fig = plot_2d_projection_trajectory(data, cube_spec, "x_y", "global", audience=audience, color_mode=self.color_mode, output_path=path, top_n=self.top_n)
            plt.close(fig)
            manifest_rows.append(self._manifest_row(path, "2d_projection", cube_spec, "global", audience, "trajectory_weight_output", "x_y projection", "written"))
        if self.plot_vector_fields and cube_spec.slug in {"green_readiness_incoming", "green_readiness_outgoing"}:
            path = output_path / f"phase_space_vector_field_{cube_spec.slug}_xy_{start_year}_{end_year}_{audience}.{ext}"
            vector_table = compute_vector_field_table(data, cube_spec, projection="x_y", bins=10, weight_variable=_weight_column(data), min_count=30)
            vector_path = VECTOR_FIELD_DIR / f"vector_field_{cube_spec.slug}_x_y_{start_year}_{end_year}.csv"
            vector_path.parent.mkdir(parents=True, exist_ok=True)
            vector_table.to_csv(vector_path, index=False)
            fig = plot_vector_field(data, cube_spec, projection="x_y", bins=10, weight_variable=_weight_column(data), audience=audience, color_mode=self.color_mode, output_path=path, min_count=30)
            plt.close(fig)
            manifest_rows.append(self._manifest_row(path, "vector_field", cube_spec, "country_sector-year movement", audience, _weight_column(data), f"binned x_y movement; table={vector_path}", "written"))

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
    ) -> dict[str, object]:
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
            "These plots are historical state-space diagnostics, not scenario forecasts.",
            "",
            "`g_in_network` and `g_out_network` are available directional network green-ness proxies. They are not the missing canonical aggregate target `network_green_exposure`.",
            "",
            "`brown_centrality` and `capability_ecosystem_exposure` remain future target axes and are not used in this first plotting layer.",
            "",
            "Portfolio plots are draft candidates for later refinement; they should not be treated as final portfolio claims.",
            "",
            f"Plots written: {written_count}. Skipped requests: {skipped_count}.",
            "",
            "The implemented cubes are green-readiness incoming/outgoing and production-safe greening incoming/outgoing.",
        ]
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
        markevery=_five_year_mark_indices(trajectory),
    )


def _format_3d_axes(ax, cube_spec: PhaseSpaceCubeSpec, audience: str) -> None:
    ax.set_xlabel(clean_display_label(cube_spec.x_label), labelpad=8)
    ax.set_ylabel(clean_display_label(cube_spec.y_label), labelpad=8)
    ax.set_zlabel(clean_display_label(cube_spec.z_label), labelpad=8)
    ax.set_title(clean_display_label(_title(cube_spec, audience)), pad=18)
    ax.view_init(elev=24, azim=-52)
    ax.grid(True, alpha=0.25)


def _annotate_3d_start_end(ax, trajectory: pd.DataFrame, cube_spec: PhaseSpaceCubeSpec) -> None:
    if trajectory.empty:
        return
    for _, row in trajectory.iloc[[0, -1]].iterrows():
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
        markevery=_five_year_mark_indices(trajectory),
    )


def _annotate_2d_start_end(ax, trajectory: pd.DataFrame, x_col: str, y_col: str) -> None:
    if trajectory.empty:
        return
    for _, row in trajectory.iloc[[0, -1]].iterrows():
        ax.text(row[x_col], row[y_col], f" {int(row['Year'])}", fontsize=8)


def _five_year_mark_indices(trajectory: pd.DataFrame) -> list[int]:
    years = pd.to_numeric(trajectory["Year"], errors="coerce").tolist()
    indices = [index for index, year in enumerate(years) if np.isfinite(year) and int(year) % 5 == 0]
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
