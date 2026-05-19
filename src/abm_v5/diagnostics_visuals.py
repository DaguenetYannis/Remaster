from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.abm_v5.paths import ABMV5Paths
from src.abm_v5.phase_space import PHASE_SPACE_OUTPUT_FILENAME


PHASE2_FOOTER = "ABM_v5 Phase 2 diagnostic exploration"
PLOT_FILENAMES = (
    "local_greenness_vs_network_green_exposure.png",
    "brown_centrality_vs_network_green_exposure.png",
    "supplier_lock_in_vs_green_capability.png",
    "emissions_intensity_gap_vs_green_capability.png",
    "brown_centrality_vs_green_capability.png",
    "distribution_supplier_count.png",
    "distribution_buyer_count.png",
    "distribution_supplier_lock_in.png",
    "distribution_brown_centrality.png",
    "selected_node_trajectories_ei_gap_vs_network_green_exposure.png",
)


@dataclass(frozen=True)
class DiagnosticVisualBuildResult:
    """Result metadata for post-Phase-2 diagnostic visual artifacts."""

    output_plot_dir: Path
    output_table_dir: Path
    n_plots_created: int
    n_tables_created: int
    phase_space_input_path: Path

    def validate(self) -> None:
        """Validate diagnostic artifact metadata."""
        if not self.output_plot_dir:
            raise ValueError("output_plot_dir must not be empty.")
        if not self.output_table_dir:
            raise ValueError("output_table_dir must not be empty.")
        if not self.phase_space_input_path:
            raise ValueError("phase_space_input_path must not be empty.")
        if self.n_plots_created <= 0:
            raise ValueError("n_plots_created must be positive.")
        if self.n_tables_created <= 0:
            raise ValueError("n_tables_created must be positive.")


def load_phase2_diagnostic_inputs(project_root: Path) -> dict[str, Any]:
    """Load post-Phase-2 diagnostic inputs without interpreting regimes."""
    import polars as pl

    paths = ABMV5Paths.from_project_root(project_root)
    phase_space_path = paths.phase_space / PHASE_SPACE_OUTPUT_FILENAME
    coverage_summary_path = paths.validation / "supplier_candidate_coverage_summary.json"
    if not phase_space_path.exists():
        raise FileNotFoundError(f"Phase-space panel missing: {phase_space_path}. Run Phase 2.6 first.")
    return {
        "phase_space": pl.read_parquet(phase_space_path),
        "phase_space_path": phase_space_path,
        "supplier_candidate_coverage_summary_path": coverage_summary_path,
    }


def _mean_expr(column: str):
    import polars as pl

    return pl.col(column).cast(pl.Float64, strict=False).mean().alias(f"mean_{column}")


def build_top_emitters_table(df):
    """Build top 25 nodes by mean emissions over 1995-2016."""
    import polars as pl

    frame = df if isinstance(df, pl.DataFrame) else pl.DataFrame(df)
    columns = [
        "emissions",
        "output",
        "emissions_intensity",
        "local_greenness",
        "network_green_exposure",
        "brown_centrality",
        "green_capability",
        "supplier_lock_in",
    ]
    table = (
        frame.group_by(["country_sector", "country", "sector"])
        .agg([_mean_expr(column) for column in columns])
        .sort("mean_emissions", descending=True, nulls_last=True)
        .head(25)
        .with_row_index("rank", offset=1)
    )
    return table


def build_top_brown_central_nodes_table(df):
    """Build top 25 nodes by mean brown centrality over 1995-2016."""
    import polars as pl

    frame = df if isinstance(df, pl.DataFrame) else pl.DataFrame(df)
    columns = [
        "brown_centrality",
        "output",
        "network_green_exposure",
        "local_greenness",
        "green_capability",
        "supplier_lock_in",
    ]
    return (
        frame.group_by(["country_sector", "country", "sector"])
        .agg([_mean_expr(column) for column in columns])
        .sort("mean_brown_centrality", descending=True, nulls_last=True)
        .head(25)
        .with_row_index("rank", offset=1)
    )


def build_supplier_candidate_coverage_table(summary_json_path: Path):
    """Flatten supplier candidate yearly coverage JSON into a table."""
    import polars as pl

    payload = json.loads(summary_json_path.read_text(encoding="utf-8"))
    rows = payload.get("yearly_coverage", payload if isinstance(payload, list) else [])
    required = [
        "year",
        "raw_positive_edges",
        "retained_historical_candidate_rows",
        "fallback_candidate_rows",
        "total_candidate_rows",
        "retained_edge_share",
        "retained_transaction_value_coverage",
        "mean_buyer_input_coverage",
        "share_buyer_years_reaching_coverage_target",
        "max_candidates_per_buyer_year",
        "buyers_with_coverage_target_unmet",
        "buyers_with_fallback_candidates",
    ]
    return pl.DataFrame([{column: row.get(column) for column in required} for row in rows]).select(required)


def select_illustrative_nodes(df, max_nodes: int = 12):
    """Select illustrative nodes from high emissions, brown centrality, capability, and lock-in."""
    import polars as pl

    frame = df if isinstance(df, pl.DataFrame) else pl.DataFrame(df)
    grouped = frame.group_by(["country_sector", "country", "sector"]).agg(
        _mean_expr("emissions"),
        _mean_expr("brown_centrality"),
        _mean_expr("green_capability"),
        _mean_expr("supplier_lock_in"),
    )
    selections: dict[str, dict[str, str]] = {}
    rules = (
        ("mean_emissions", "top mean emissions"),
        ("mean_brown_centrality", "top mean brown centrality"),
        ("mean_green_capability", "top mean green capability"),
        ("mean_supplier_lock_in", "top mean supplier lock-in"),
    )
    for column, reason in rules:
        for row in grouped.sort(column, descending=True, nulls_last=True).head(3).to_dicts():
            key = row["country_sector"]
            if key in selections:
                selections[key]["selection_reason"] += f"; {reason}"
            else:
                selections[key] = {
                    "country_sector": key,
                    "country": row.get("country"),
                    "sector": row.get("sector"),
                    "selection_reason": reason,
                }
    return pl.DataFrame(list(selections.values())[:max_nodes])


def _plot_ready(df, required: tuple[str, ...]):
    import polars as pl

    frame = df if isinstance(df, pl.DataFrame) else pl.DataFrame(df)
    return frame.drop_nulls(list(required))


def _sizes(values: list[float]) -> list[float]:
    logs = [math.log(max(value or 0.0, 0.0) + 1.0) for value in values]
    if not logs:
        return []
    low, high = min(logs), max(logs)
    if low == high:
        return [35.0 for _ in logs]
    return [15.0 + 85.0 * (value - low) / (high - low) for value in logs]


def _annotate_outliers(ax, rows: list[dict[str, Any]], x: str, y: str) -> None:
    ranked = sorted(
        rows,
        key=lambda row: (row.get("output") or 0.0, abs(row.get(x) or 0.0) + abs(row.get(y) or 0.0)),
        reverse=True,
    )[:12]
    for row in ranked:
        label = _short_node_label(row.get("country_sector", ""))
        ax.annotate(label, (row[x], row[y]), fontsize=6, alpha=0.75)


def _short_node_label(value: object) -> str:
    text = str(value or "").strip()
    if "\t" in text:
        return text.split("\t", 1)[0].strip()
    if "|" in text:
        return text.split("|", 1)[0].strip()
    return text[:24]


def _scatter_plot(
    df,
    output_path: Path,
    x: str,
    y: str,
    color: str,
    title: str,
    quadrant_labels: tuple[str, str, str, str],
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = _plot_ready(df, (x, y, color, "output"))
    rows = frame.to_dicts()
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        frame[x].to_list(),
        frame[y].to_list(),
        s=_sizes(frame["output"].to_list()),
        c=frame[color].to_list(),
        cmap="viridis",
        alpha=0.65,
        linewidths=0,
    )
    x_median = frame[x].median()
    y_median = frame[y].median()
    ax.axvline(x_median, color="0.35", linewidth=0.8, linestyle="--")
    ax.axhline(y_median, color="0.35", linewidth=0.8, linestyle="--")
    ax.set_title(f"{title}\n{PHASE2_FOOTER}")
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(y.replace("_", " "))
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(color.replace("_", " "))
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.text(xmin, ymax, quadrant_labels[0], va="top", ha="left", fontsize=8)
    ax.text(xmax, ymax, quadrant_labels[1], va="top", ha="right", fontsize=8)
    ax.text(xmin, ymin, quadrant_labels[2], va="bottom", ha="left", fontsize=8)
    ax.text(xmax, ymin, quadrant_labels[3], va="bottom", ha="right", fontsize=8)
    _annotate_outliers(ax, rows, x, y)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_local_vs_network_greenness(df, output_path: Path) -> Path:
    return _scatter_plot(
        df,
        output_path,
        "local_greenness",
        "network_green_exposure",
        "brown_centrality",
        "Local vs network-embedded green-ness",
        ("locally brown, network green", "locally green, network green", "locally brown, network brown", "locally green, network brown"),
    )


def plot_brown_centrality_vs_network_green_exposure(df, output_path: Path) -> Path:
    return _scatter_plot(
        df,
        output_path,
        "brown_centrality",
        "network_green_exposure",
        "supplier_lock_in",
        "Carbon centrality vs network green exposure",
        ("less central, greener network", "brown central, greener network", "less central, browner network", "brown central, browner network"),
    )


def plot_supplier_lock_in_vs_green_capability(df, output_path: Path) -> Path:
    return _scatter_plot(
        df,
        output_path,
        "supplier_lock_in",
        "green_capability",
        "emissions_intensity_gap",
        "Constraint vs potential: supplier lock-in and green capability",
        ("flexible but capable", "capable but constrained", "adaptable", "trapped"),
    )


def plot_emissions_intensity_gap_vs_green_capability(df, output_path: Path) -> Path:
    return _scatter_plot(
        df,
        output_path,
        "emissions_intensity_gap",
        "green_capability",
        "network_green_exposure",
        "Dirty but capable? Emissions-intensity gap vs green capability",
        ("cleaner gap, capable", "dirty gap, capable", "cleaner gap, weak", "dirty gap, weak"),
    )


def plot_brown_centrality_vs_green_capability(df, output_path: Path) -> Path:
    return _scatter_plot(
        df,
        output_path,
        "brown_centrality",
        "green_capability",
        "supplier_lock_in",
        "Strategic transition terrain: brown centrality vs green capability",
        ("less brown, capable", "brown central, capable", "less brown, weak", "brown central, weak"),
    )


def plot_distribution(df, variable: str, output_path: Path, bins: int = 40) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = _plot_ready(df, (variable,))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(frame[variable].to_list(), bins=bins, color="#4c78a8", alpha=0.85)
    ax.set_title(f"Distribution of {variable.replace('_', ' ')}\n{PHASE2_FOOTER}")
    ax.set_xlabel(variable.replace("_", " "))
    ax.set_ylabel("count")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_selected_node_trajectories(df, selected_nodes, output_path: Path, metadata_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import polars as pl

    frame = df if isinstance(df, pl.DataFrame) else pl.DataFrame(df)
    selected = selected_nodes if isinstance(selected_nodes, pl.DataFrame) else pl.DataFrame(selected_nodes)
    selected = selected.head(12)
    fig, ax = plt.subplots(figsize=(9, 6))
    metadata_rows: list[dict[str, Any]] = []
    for row in selected.to_dicts():
        node = row["country_sector"]
        node_frame = (
            frame.filter(pl.col("country_sector") == node)
            .drop_nulls(["emissions_intensity_gap", "network_green_exposure"])
            .sort("year")
        )
        if node_frame.is_empty():
            continue
        ax.plot(
            node_frame["emissions_intensity_gap"].to_list(),
            node_frame["network_green_exposure"].to_list(),
            marker="o",
            linewidth=1.1,
            markersize=3,
            label=_short_node_label(node),
        )
        ax.scatter(node_frame["emissions_intensity_gap"][0], node_frame["network_green_exposure"][0], marker="s", s=30)
        ax.scatter(node_frame["emissions_intensity_gap"][-1], node_frame["network_green_exposure"][-1], marker="^", s=35)
        metadata_rows.append(
            {
                "country_sector": node,
                "country": row.get("country"),
                "sector": row.get("sector"),
                "selection_reason": row.get("selection_reason"),
                "first_year_available": int(node_frame["year"][0]),
                "last_year_available": int(node_frame["year"][-1]),
            }
        )
    ax.set_title(f"Selected country-sector trajectories in observed phase space\n{PHASE2_FOOTER}")
    ax.set_xlabel("emissions intensity gap")
    ax.set_ylabel("network green exposure")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    pl.DataFrame(metadata_rows).write_csv(metadata_path)
    return output_path


def build_phase2_diagnostic_visuals(project_root: Path) -> DiagnosticVisualBuildResult:
    """Build all post-Phase-2 exploratory diagnostic tables and PNG plots."""
    paths = ABMV5Paths.from_project_root(project_root)
    paths.validate_project_root()
    paths.ensure_directories()
    plot_dir = paths.plots_diagnostics
    table_dir = paths.diagnostics
    inputs = load_phase2_diagnostic_inputs(project_root)
    phase_space = inputs["phase_space"]
    coverage_path = inputs["supplier_candidate_coverage_summary_path"]

    tables = {
        "top_emitters_avg_1995_2016.csv": build_top_emitters_table(phase_space),
        "top_brown_central_nodes_avg_1995_2016.csv": build_top_brown_central_nodes_table(phase_space),
        "illustrative_nodes_selection.csv": select_illustrative_nodes(phase_space),
    }
    if coverage_path.exists():
        tables["supplier_candidate_coverage_by_year.csv"] = build_supplier_candidate_coverage_table(coverage_path)
    else:
        import polars as pl

        tables["supplier_candidate_coverage_by_year.csv"] = pl.DataFrame()

    for filename, table in tables.items():
        table.write_csv(table_dir / filename)

    selected_nodes = tables["illustrative_nodes_selection.csv"]
    plot_paths = [
        plot_local_vs_network_greenness(phase_space, plot_dir / PLOT_FILENAMES[0]),
        plot_brown_centrality_vs_network_green_exposure(phase_space, plot_dir / PLOT_FILENAMES[1]),
        plot_supplier_lock_in_vs_green_capability(phase_space, plot_dir / PLOT_FILENAMES[2]),
        plot_emissions_intensity_gap_vs_green_capability(phase_space, plot_dir / PLOT_FILENAMES[3]),
        plot_brown_centrality_vs_green_capability(phase_space, plot_dir / PLOT_FILENAMES[4]),
        plot_distribution(phase_space, "supplier_count", plot_dir / PLOT_FILENAMES[5]),
        plot_distribution(phase_space, "buyer_count", plot_dir / PLOT_FILENAMES[6]),
        plot_distribution(phase_space, "supplier_lock_in", plot_dir / PLOT_FILENAMES[7]),
        plot_distribution(phase_space, "brown_centrality", plot_dir / PLOT_FILENAMES[8]),
        plot_selected_node_trajectories(
            phase_space,
            selected_nodes,
            plot_dir / PLOT_FILENAMES[9],
            table_dir / "selected_node_trajectories_metadata.csv",
        ),
    ]
    result = DiagnosticVisualBuildResult(
        output_plot_dir=plot_dir,
        output_table_dir=table_dir,
        n_plots_created=sum(path.exists() for path in plot_paths),
        n_tables_created=sum((table_dir / filename).exists() for filename in tables)
        + int((table_dir / "selected_node_trajectories_metadata.csv").exists()),
        phase_space_input_path=inputs["phase_space_path"],
    )
    result.validate()
    return result
