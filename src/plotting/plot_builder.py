import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import numpy as np
import plotly.graph_objects as go

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-path", default="data/metrics")
    parser.add_argument("--output-path", default="outputs/plots")
    parser.add_argument("--year", type=int, required=False)
    parser.add_argument("--centrality-col", default="out_strength")
    parser.add_argument(
    "--plots",
    nargs="+",
    default=["all"],
    choices=[
        "all",
        "ei",
        "et",
        "efficiency",
        "network",
        "heatmap",
        "country-network",
        "country-sankey",
        "sector-sankey",
        "time",
        "phase-space",
        "phase-trajectories",
    ],
)
    return parser.parse_args()


# =========================
# LOADERS
# =========================

def load_year_metrics(metrics_path, year):
    year_path = metrics_path / str(year)

    ei = pd.read_parquet(year_path / f"ei_{year}.parquet")
    centrality = pd.read_parquet(year_path / f"centrality_{year}.parquet")

    return ei, centrality


def load_year_metrics_with_et(metrics_path, year):
    year_path = metrics_path / str(year)

    et = pd.read_parquet(year_path / f"et_{year}.parquet")
    centrality = pd.read_parquet(year_path / f"centrality_{year}.parquet")

    return et, centrality


def load_efficiency(metrics_path, year):
    year_path = metrics_path / str(year)
    return pd.read_parquet(year_path / f"efficiency_{year}.parquet")

# =========================
# HELPERS
# =========================

def extract_country(label):
    parts = str(label).split(" | ")
    return parts[0].strip()

def extract_sector(label):
    parts = str(label).split(" | ")
    return parts[-1].strip()

def extract_country_sector(label):
    country = extract_country(label)
    sector = extract_sector(label)
    return f"{country} — {sector}"


def prepare_edges_from_matrix(matrix, value_col="embedded_emissions"):
    matrix = matrix.copy()
    matrix.index.name = "source_sector"
    matrix.columns.name = "target_sector"

    return (
        matrix.stack()
        .reset_index(name=value_col)
        .query(f"{value_col} > 0")
    )


# =========================
# SINGLE-YEAR PLOTS
# =========================

def plot_ei_vs_centrality(ei, centrality, year, centrality_col, output_path):
    df = ei.join(centrality, how="inner")

    df = df.replace([float("inf"), float("-inf")], pd.NA).dropna(
        subset=["emissions_intensity", centrality_col]
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["emissions_intensity"], df[centrality_col], alpha=0.5, s=12)

    ax.set_title(f"EI vs {centrality_col} — {year}")
    ax.set_xlabel("Emissions intensity")
    ax.set_ylabel(centrality_col)

    fig.savefig(output_path / f"ei_vs_{centrality_col}_{year}.png", dpi=300)
    plt.close(fig)

    print(f"[OK] Saved EI plot for {year}")


def plot_et_embodied_vs_centrality(et, centrality, year, output_path):
    out_embodied = et.sum(axis=1).rename("out_embodied")
    in_embodied = et.sum(axis=0).rename("in_embodied")

    df = centrality.join(out_embodied).join(in_embodied)
    df = df.replace([float("inf"), float("-inf")], pd.NA).dropna()

    # Out plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["out_strength"], df["out_embodied"], alpha=0.5, s=12)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_title(f"Carbon transmission vs centrality — {year}")
    ax.set_xlabel("Out-strength")
    ax.set_ylabel("Out embodied emissions")

    fig.savefig(output_path / f"out_embodied_vs_out_strength_{year}.png", dpi=300)
    plt.close(fig)

    # In plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["in_strength"], df["in_embodied"], alpha=0.5, s=12)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_title(f"Carbon dependency vs centrality — {year}")
    ax.set_xlabel("In-strength")
    ax.set_ylabel("In embodied emissions")

    fig.savefig(output_path / f"in_embodied_vs_in_strength_{year}.png", dpi=300)
    plt.close(fig)

    print(f"[OK] Saved ET plots for {year}")


def plot_efficiency_vs_centrality(efficiency, year, output_path):
    df = efficiency.replace([float("inf"), float("-inf")], pd.NA).dropna()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["out_strength"], df["out_efficiency"], alpha=0.5, s=12)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_title(f"Carbon efficiency vs centrality — {year}")
    ax.set_xlabel("Out-strength")
    ax.set_ylabel("Out efficiency")

    fig.savefig(output_path / f"out_efficiency_vs_out_strength_{year}.png", dpi=300)
    plt.close(fig)

    print(f"[OK] Saved efficiency plot for {year}")

def build_reduced_network(
    et,
    centrality,
    top_n=150,
    centrality_col="pagerank",
    edge_quantile=0.99,
    top_k_edges_per_node=5,
):
    top_nodes = (
        centrality
        .sort_values(centrality_col, ascending=False)
        .head(top_n)
        .index
    )

    et_reduced = et.loc[
        et.index.intersection(top_nodes),
        et.columns.intersection(top_nodes),
    ].copy()

    edges = prepare_edges_from_matrix(et_reduced)

    threshold = edges["embedded_emissions"].quantile(edge_quantile)
    edges = edges[edges["embedded_emissions"] >= threshold]

    edges = (
        edges
        .sort_values("embedded_emissions", ascending=False)
        .groupby("source_sector")
        .head(top_k_edges_per_node)
    )

    return nx.from_pandas_edgelist(
        edges,
        source="source_sector",
        target="target_sector",
        edge_attr="embedded_emissions",
        create_using=nx.DiGraph(),
    )

def plot_network_view(et, centrality, year, output_path, top_n=150, centrality_col="pagerank"):
    df = centrality.copy()
    df = df.replace([float("inf"), float("-inf")], pd.NA).dropna()

    if centrality_col not in df.columns:
        raise ValueError(f"{centrality_col} not found in centrality columns: {list(df.columns)}")

    G = build_reduced_network(
        et=et,
        centrality=df,
        top_n=top_n,
        centrality_col=centrality_col,
    )

    pos = nx.kamada_kawai_layout(G)

    node_sizes = []
    for node in G.nodes:
        value = df.loc[node, centrality_col] if node in df.index else 0
        node_sizes.append(100 + 5000 * value)

    edge_weights = [
        G[u][v]["embedded_emissions"]
        for u, v in G.edges
    ]

    max_edge_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [
        0.2 + 3 * (w / max_edge_weight)
        for w in edge_weights
    ]

    fig, ax = plt.subplots(figsize=(12, 10))

    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        alpha=0.25,
        arrows=False,
        ax=ax,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        alpha=0.8,
        ax=ax,
    )

    ax.set_title(f"Embodied carbon network — top {top_n} nodes by {centrality_col} — {year}")
    ax.axis("off")

    fig.savefig(output_path / f"network_top_{top_n}_{centrality_col}_{year}.png", dpi=300)
    plt.close(fig)

    print(f"[OK] Saved network plot for {year}")

def plot_et_heatmap(et, year, output_path, top_n=150):
    out_embodied = et.sum(axis=1)
    top_nodes = out_embodied.sort_values(ascending=False).head(top_n).index

    et_reduced = et.loc[
        et.index.intersection(top_nodes),
        et.columns.intersection(top_nodes),
    ]

    log_et = np.log10(et_reduced.clip(lower=1e-10)) # Avoid NaNs

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(log_et, aspect="auto")

    ax.set_title(f"Log ET heatmap — top {top_n} emitting sectors — {year}")
    ax.set_xlabel("Target sector")
    ax.set_ylabel("Source sector")

    fig.colorbar(im, ax=ax, label="log10(embedded emissions)")
    fig.savefig(output_path / f"et_heatmap_log_top_{top_n}_{year}.png", dpi=300)
    plt.close(fig)

    print(f"[OK] Saved ET heatmap for {year}")

def plot_country_flow_network(et, year, output_path, top_n_edges=30):
    edges = prepare_edges_from_matrix(et)

    edges["source_country"] = edges["source_sector"].map(extract_country)
    edges["target_country"] = edges["target_sector"].map(extract_country)

    country_edges = (
        edges
        .groupby(["source_country", "target_country"], as_index=False)["embedded_emissions"]
        .sum()
    )

    country_edges = country_edges[
        country_edges["source_country"] != country_edges["target_country"]
    ]

    country_edges = (
        country_edges
        .sort_values("embedded_emissions", ascending=False)
        .head(top_n_edges)
    )

    G = nx.from_pandas_edgelist(
        country_edges,
        source="source_country",
        target="target_country",
        edge_attr="embedded_emissions",
        create_using=nx.DiGraph(),
    )

    pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)

    weights = [G[u][v]["embedded_emissions"] for u, v in G.edges]
    log_weights = np.log10(weights)

    max_weight = log_weights.max() if len(log_weights) > 0 else 1
    widths = [0.5 + 4 * (w / max_weight) for w in log_weights]

    node_strength = country_edges.groupby("source_country")["embedded_emissions"].sum()
    max_node_strength = node_strength.max() if len(node_strength) > 0 else 1

    node_sizes = [
        300 + 2000 * (node_strength.get(node, 0) / max_node_strength)
        for node in G.nodes
    ]

    fig, ax = plt.subplots(figsize=(12, 10))

    nx.draw_networkx_edges(
        G,
        pos,
        width=widths,
        alpha=0.35,
        arrows=True,
        arrowsize=10,
        ax=ax,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        alpha=0.9,
        ax=ax,
    )

    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    ax.set_title(f"Country-level embodied carbon flow network — {year}")
    ax.axis("off")

    fig.savefig(output_path / f"country_flow_network_{year}.png", dpi=300)
    plt.close(fig)

    print(f"[OK] Saved country flow network for {year}")

def plot_sankey_top_flows(et, year, output_path, top_n_edges=25):
    edges = prepare_edges_from_matrix(et)

    edges["source_country"] = edges["source_sector"].map(extract_country)
    edges["target_country"] = edges["target_sector"].map(extract_country)

    top_edges = (
        edges
        .groupby(["source_country", "target_country"], as_index=False)["embedded_emissions"]
        .sum()
    )

    top_edges = top_edges[
        top_edges["source_country"] != top_edges["target_country"]
    ]

    top_edges = (
        top_edges
        .sort_values("embedded_emissions", ascending=False)
        .head(top_n_edges)
    )

    labels = pd.Index(
        pd.concat([top_edges["source_country"], top_edges["target_country"]]).unique()
    )

    label_to_id = {label: i for i, label in enumerate(labels)}

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    label=list(labels),
                    pad=18,
                    thickness=18,
                ),
                link=dict(
                    source=top_edges["source_country"].map(label_to_id),
                    target=top_edges["target_country"].map(label_to_id),
                    value=top_edges["embedded_emissions"],
                ),
            )
        ]
    )

    fig.update_layout(
        title_text=f"Top international embodied carbon flows — {year}",
        font_size=11,
        width=1200,
        height=800,
    )

    fig.write_html(output_path / f"sankey_country_flows_{year}.html")

    print(f"[OK] Saved Sankey for {year}")


def plot_sankey_sector_country_flows(et, year, output_path, top_n_edges=50):
    edges = prepare_edges_from_matrix(et)

    edges["source_country"] = edges["source_sector"].map(extract_country)
    edges["target_country"] = edges["target_sector"].map(extract_country)

    edges["source_sector_clean"] = edges["source_sector"].map(extract_sector)
    edges["target_sector_clean"] = edges["target_sector"].map(extract_sector)

    edges["source_node"] = edges["source_sector"].map(extract_country_sector)
    edges["target_node"] = edges["target_sector"].map(extract_country_sector)

    sector_country_edges = (
        edges
        .groupby(
            [
                "source_node",
                "target_node",
                "source_country",
                "target_country",
                "source_sector_clean",
                "target_sector_clean",
            ],
            as_index=False,
        )["embedded_emissions"]
        .sum()
    )

    sector_country_edges = sector_country_edges[
        sector_country_edges["source_node"] != sector_country_edges["target_node"]
    ]

    sector_country_edges = (
        sector_country_edges
        .sort_values("embedded_emissions", ascending=False)
        .head(top_n_edges)
    )

    labels = pd.Index(
        pd.concat(
            [
                sector_country_edges["source_node"],
                sector_country_edges["target_node"],
            ]
        ).unique()
    )

    label_to_id = {label: i for i, label in enumerate(labels)}

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    label=list(labels),
                    pad=18,
                    thickness=14,
                ),
                link=dict(
                    source=sector_country_edges["source_node"].map(label_to_id),
                    target=sector_country_edges["target_node"].map(label_to_id),
                    value=sector_country_edges["embedded_emissions"],
                    customdata=sector_country_edges[
                        [
                            "source_country",
                            "target_country",
                            "source_sector_clean",
                            "target_sector_clean",
                        ]
                    ],
                    hovertemplate=(
                        "Source country: %{customdata[0]}<br>"
                        "Target country: %{customdata[1]}<br>"
                        "Source sector: %{customdata[2]}<br>"
                        "Target sector: %{customdata[3]}<br>"
                        "Embodied emissions: %{value}<extra></extra>"
                    ),
                ),
            )
        ]
    )

    fig.update_layout(
        title_text=f"Top country-sector embodied carbon flows — {year}",
        font_size=10,
        width=1400,
        height=900,
    )

    fig.write_html(output_path / f"sankey_sector_country_flows_{year}.html")

    print(f"[OK] Saved sector-country Sankey for {year}")

# =========================
# TIME EVOLUTION
# =========================

def herfindahl_index(series):
    values = series.replace([np.inf, -np.inf], np.nan).dropna()
    values = values[values > 0]

    total = values.sum()
    if total == 0:
        return np.nan

    shares = values / total
    return (shares ** 2).sum()


def load_all_years(metrics_path):
    rows = []

    for year_folder in sorted(metrics_path.iterdir()):
        if not year_folder.is_dir():
            continue

        year = int(year_folder.name)

        ei_path = year_folder / f"ei_{year}.parquet"
        centrality_path = year_folder / f"centrality_{year}.parquet"
        greenness_path = year_folder / f"greenness_{year}.parquet"
        efficiency_path = year_folder / f"efficiency_{year}.parquet"

        required_paths = [ei_path, centrality_path, greenness_path]

        if not all(path.exists() for path in required_paths):
            print(f"[SKIP] Missing required metrics for {year}")
            continue

        ei = pd.read_parquet(ei_path)
        centrality = pd.read_parquet(centrality_path)
        greenness = pd.read_parquet(greenness_path)

        df = (
            ei
            .join(centrality, how="inner")
            .join(greenness, how="inner")
        )

        if efficiency_path.exists():
            efficiency = pd.read_parquet(efficiency_path)
            efficiency_cols = [
                col for col in ["out_embodied", "in_embodied", "out_efficiency", "in_efficiency"]
                if col in efficiency.columns
            ]
            df = df.join(efficiency[efficiency_cols], how="left")

        df = df.replace([np.inf, -np.inf], np.nan)

        required_cols = [
            "emissions_intensity",
            "out_strength",
            "in_strength",
            "pagerank",
            "g_base",
            "g_out_network",
            "g_in_network",
        ]

        df = df.dropna(subset=[col for col in required_cols if col in df.columns])

        row = {
            "year": year,
            "n_sectors": len(df),

            # Local ecological state
            "mean_ei": df["emissions_intensity"].mean(),
            "median_ei": df["emissions_intensity"].median(),
            "mean_g_base": df["g_base"].mean(),
            "median_g_base": df["g_base"].median(),

            # Network-embedded green-ness
            "mean_g_out_network": df["g_out_network"].mean(),
            "median_g_out_network": df["g_out_network"].median(),
            "mean_g_in_network": df["g_in_network"].mean(),
            "median_g_in_network": df["g_in_network"].median(),

            # Centrality / structure
            "mean_out_strength": df["out_strength"].mean(),
            "mean_in_strength": df["in_strength"].mean(),
            "pagerank_hhi": herfindahl_index(df["pagerank"]),
            "out_strength_hhi": herfindahl_index(df["out_strength"]),
            "in_strength_hhi": herfindahl_index(df["in_strength"]),

            # Core Remaster relationships
            "corr_ei_out_strength": df["emissions_intensity"].corr(df["out_strength"]),
            "corr_ei_pagerank": df["emissions_intensity"].corr(df["pagerank"]),
            "corr_g_base_pagerank": df["g_base"].corr(df["pagerank"]),
            "corr_g_out_network_pagerank": df["g_out_network"].corr(df["pagerank"]),
            "corr_g_in_network_pagerank": df["g_in_network"].corr(df["pagerank"]),
            "corr_g_base_g_out_network": df["g_base"].corr(df["g_out_network"]),
            "corr_g_base_g_in_network": df["g_base"].corr(df["g_in_network"]),
        }

        if "out_embodied" in df.columns:
            row["total_out_embodied"] = df["out_embodied"].sum()
            row["out_embodied_hhi"] = herfindahl_index(df["out_embodied"])

        if "in_embodied" in df.columns:
            row["total_in_embodied"] = df["in_embodied"].sum()
            row["in_embodied_hhi"] = herfindahl_index(df["in_embodied"])

        if "out_efficiency" in df.columns:
            row["mean_out_efficiency"] = df["out_efficiency"].mean()
            row["median_out_efficiency"] = df["out_efficiency"].median()

        if "in_efficiency" in df.columns:
            row["mean_in_efficiency"] = df["in_efficiency"].mean()
            row["median_in_efficiency"] = df["in_efficiency"].median()

        rows.append(row)

    return pd.DataFrame(rows).sort_values("year")


def plot_time_evolution(summary, output_path):
    def save_line_plot(columns, title, ylabel, filename):
        fig, ax = plt.subplots(figsize=(9, 5))

        for col in columns:
            if col in summary.columns:
                ax.plot(summary["year"], summary[col], marker="o", label=col)

        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.legend()

        fig.tight_layout()
        fig.savefig(output_path / filename, dpi=300)
        plt.close(fig)

        print(f"[OK] Saved {filename}")

    save_line_plot(
        columns=["mean_g_base", "mean_g_out_network", "mean_g_in_network"],
        title="Local and network-embedded green-ness over time",
        ylabel="Green-ness",
        filename="greenness_system_trajectory.png",
    )

    save_line_plot(
        columns=["mean_ei", "median_ei"],
        title="Emissions intensity over time",
        ylabel="Emissions intensity",
        filename="emissions_intensity_over_time.png",
    )

    save_line_plot(
        columns=[
            "corr_g_base_pagerank",
            "corr_g_out_network_pagerank",
            "corr_g_in_network_pagerank",
        ],
        title="Correlation between green-ness and PageRank over time",
        ylabel="Correlation",
        filename="greenness_pagerank_correlation_over_time.png",
    )

    save_line_plot(
        columns=[
            "pagerank_hhi",
            "out_strength_hhi",
            "in_strength_hhi",
        ],
        title="Network concentration over time",
        ylabel="Herfindahl-Hirschman index",
        filename="network_concentration_over_time.png",
    )

    if "out_embodied_hhi" in summary.columns or "in_embodied_hhi" in summary.columns:
        save_line_plot(
            columns=["out_embodied_hhi", "in_embodied_hhi"],
            title="Embodied carbon flow concentration over time",
            ylabel="Herfindahl-Hirschman index",
            filename="embodied_carbon_concentration_over_time.png",
        )

    if "total_out_embodied" in summary.columns:
        save_line_plot(
            columns=["total_out_embodied"],
            title="Total embodied carbon transmission over time",
            ylabel="Total embodied emissions",
            filename="total_embodied_carbon_over_time.png",
        )

# =========================
# SYSTEMS
# ========================= 

def load_phase_space_data(metrics_path, years):
    dfs = []

    for year in years:
        year_path = metrics_path / str(year)

        ei_path = year_path / f"ei_{year}.parquet"
        centrality_path = year_path / f"centrality_{year}.parquet"
        greenness_path = year_path / f"greenness_{year}.parquet"

        if not all(p.exists() for p in [ei_path, centrality_path, greenness_path]):
            print(f"[SKIP] Missing data for {year}")
            continue

        ei = pd.read_parquet(ei_path)
        centrality = pd.read_parquet(centrality_path)
        greenness = pd.read_parquet(greenness_path)

        df = ei.join(centrality).join(greenness)

        df["year"] = year

        # Phase-space coordinates
        df["x"] = np.log1p(df["out_strength"])
        df["y"] = df["g_out_network"]

        # Color (diagnostic)
        df["color"] = np.log1p(df["emissions_intensity"])

        df = df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["x", "y"]
        )

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

def compute_phase_thresholds(df):
    x_threshold = df["x"].quantile(0.75)
    y_threshold = df["y"].median()
    return x_threshold, y_threshold

def plot_phase_space_panel(df, years, output_path):
    x_thr, y_thr = compute_phase_thresholds(df)

    n = len(years)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    axes = axes.flatten()

    for i, year in enumerate(years):
        ax = axes[i]

        sub = df[df["year"] == year]

        sc = ax.scatter(
            sub["x"],
            sub["y"],
            c=sub["color"],
            s=5,
            alpha=0.4,
        )

        # Threshold lines
        ax.axvline(x_thr, linestyle="--")
        ax.axhline(y_thr, linestyle="--")

        ax.set_title(f"{year}")
        ax.set_xlabel("log(1 + out-strength)")
        ax.set_ylabel("Network green-ness")

        # Quadrant labels
        ax.text(0.05, 0.95, "Green-periphery", transform=ax.transAxes, fontsize=8)
        ax.text(0.65, 0.95, "Green-core", transform=ax.transAxes, fontsize=8)
        ax.text(0.05, 0.05, "Brown-periphery", transform=ax.transAxes, fontsize=8)
        ax.text(0.65, 0.05, "Brown-core", transform=ax.transAxes, fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(output_path / "phase_space_panel.png", dpi=300)
    plt.close(fig)

    print("[OK] Saved phase-space panel")

def classify_regimes(df, x_thr, y_thr):
    conditions = [
        (df["x"] < x_thr) & (df["y"] < y_thr),
        (df["x"] >= x_thr) & (df["y"] < y_thr),
        (df["x"] < x_thr) & (df["y"] >= y_thr),
        (df["x"] >= x_thr) & (df["y"] >= y_thr),
    ]

    labels = [
        "brown_periphery",
        "brown_core",
        "green_periphery",
        "green_core",
    ]

    df["regime"] = np.select(conditions, labels, default="unknown")
    return df

def plot_regime_shares(df, output_path):
    x_thr, y_thr = compute_phase_thresholds(df)
    df = classify_regimes(df, x_thr, y_thr)

    shares = (
        df.groupby(["year", "regime"])
        .size()
        .unstack(fill_value=0)
    )

    shares = shares.div(shares.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))

    for col in shares.columns:
        ax.plot(shares.index, shares[col], marker="o", label=col)

    ax.set_title("Regime shares over time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Share")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path / "phase_space_regime_shares.png", dpi=300)
    plt.close(fig)

    print("[OK] Saved regime shares plot")


def select_trajectory_nodes(
    metrics_path,
    base_year=1990,
    top_n_pagerank=10,
    top_n_out_strength=10,
    country_codes=None,
):
    if country_codes is None:
        country_codes = ["BRA", "KOR", "BGD"]

    year_path = metrics_path / str(base_year)
    centrality = pd.read_parquet(year_path / f"centrality_{base_year}.parquet")

    top_pagerank_nodes = (
        centrality.sort_values("pagerank", ascending=False)
        .head(top_n_pagerank)
        .index
        .tolist()
    )

    top_out_strength_nodes = (
        centrality.sort_values("out_strength", ascending=False)
        .head(top_n_out_strength)
        .index
        .tolist()
    )

    country_nodes = [
        node for node in centrality.index
        if extract_country(node) in country_codes
    ]

    selected_nodes = sorted(
        set(top_pagerank_nodes + top_out_strength_nodes + country_nodes)
    )

    return selected_nodes

def load_phase_space_trajectory_data(metrics_path, years, selected_nodes):
    dfs = []

    for year in years:
        year_path = metrics_path / str(year)

        ei_path = year_path / f"ei_{year}.parquet"
        centrality_path = year_path / f"centrality_{year}.parquet"
        greenness_path = year_path / f"greenness_{year}.parquet"

        if not all(p.exists() for p in [ei_path, centrality_path, greenness_path]):
            print(f"[SKIP] Missing data for {year}")
            continue

        ei = pd.read_parquet(ei_path)
        centrality = pd.read_parquet(centrality_path)
        greenness = pd.read_parquet(greenness_path)

        df = ei.join(centrality, how="inner").join(greenness, how="inner")

        df = df.loc[df.index.intersection(selected_nodes)].copy()

        df["node"] = df.index
        df["country"] = df["node"].map(extract_country)
        df["sector"] = df["node"].map(extract_sector)
        df["year"] = year

        df["x"] = np.log1p(df["out_strength"])
        df["y"] = df["g_out_network"]

        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y"])

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

def plot_selected_node_trajectories(df, output_path, max_nodes=25):
    nodes = df["node"].drop_duplicates().head(max_nodes)

    fig, ax = plt.subplots(figsize=(10, 7))

    for node in nodes:
        sub = df[df["node"] == node].sort_values("year")

        if len(sub) < 2:
            continue

        label = extract_country_sector(node)

        ax.plot(
            sub["x"],
            sub["y"],
            marker="o",
            linewidth=1,
            alpha=0.7,
            label=label,
        )

        first = sub.iloc[0]
        last = sub.iloc[-1]

        ax.text(first["x"], first["y"], str(int(first["year"])), fontsize=7)
        ax.text(last["x"], last["y"], str(int(last["year"])), fontsize=7)

    ax.set_title("Selected country-sector trajectories in phase-space")
    ax.set_xlabel("log(1 + out-strength)")
    ax.set_ylabel("Outgoing network green-ness")

    ax.legend(fontsize=6, loc="best", ncol=1)

    fig.tight_layout()
    fig.savefig(output_path / "phase_space_selected_node_trajectories.png", dpi=300)
    plt.close(fig)

    print("[OK] Saved selected node trajectories")

def load_phase_space_all_years(metrics_path, years):
    return load_phase_space_data(metrics_path, years)

def compute_regime_centroids(df):
    x_thr, y_thr = compute_phase_thresholds(df)

    df = classify_regimes(df.copy(), x_thr, y_thr)

    centroids = (
        df.groupby(["year", "regime"], as_index=False)
        .agg(
            x=("x", "mean"),
            y=("y", "mean"),
            n=("regime", "size"),
        )
    )

    return centroids

def plot_regime_centroid_trajectories(centroids, output_path):
    fig, ax = plt.subplots(figsize=(10, 7))

    for regime in sorted(centroids["regime"].unique()):
        sub = centroids[centroids["regime"] == regime].sort_values("year")

        if len(sub) < 2:
            continue

        ax.plot(
            sub["x"],
            sub["y"],
            marker="o",
            linewidth=2,
            label=regime,
        )

        first = sub.iloc[0]
        last = sub.iloc[-1]

        ax.text(first["x"], first["y"], str(int(first["year"])), fontsize=8)
        ax.text(last["x"], last["y"], str(int(last["year"])), fontsize=8)

    ax.set_title("Regime centroid trajectories in phase-space")
    ax.set_xlabel("log(1 + out-strength)")
    ax.set_ylabel("Outgoing network green-ness")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path / "phase_space_regime_centroid_trajectories.png", dpi=300)
    plt.close(fig)

    print("[OK] Saved regime centroid trajectories")


# =========================
# MAIN
# =========================

def main():
    args = parse_args()

    metrics_path = Path(args.metrics_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    selected_plots = set(args.plots)
    run_all = "all" in selected_plots

    if args.year is None:
        summary = load_all_years(metrics_path)

        summary.to_csv(output_path / "metrics_time_summary.csv", index=False)
        print("[OK] Saved summary CSV")

        if run_all or "time" in selected_plots:
            plot_time_evolution(summary, output_path)

        if run_all or "phase-space" in selected_plots:
            years = [1990, 1995, 2000, 2005, 2010, 2016]

            df = load_phase_space_data(metrics_path, years)

            plot_phase_space_panel(df, years, output_path)
            plot_regime_shares(df, output_path)

        if run_all or "phase-trajectories" in selected_plots:
            years = list(range(1990, 2017))

            selected_nodes = select_trajectory_nodes(
                metrics_path=metrics_path,
                base_year=1990,
                top_n_pagerank=10,
                top_n_out_strength=10,
                country_codes=["BRA", "KOR", "BGD"],
            )

            trajectory_df = load_phase_space_trajectory_data(
                metrics_path=metrics_path,
                years=years,
                selected_nodes=selected_nodes,
            )

            plot_selected_node_trajectories(
                df=trajectory_df,
                output_path=output_path,
                max_nodes=25,
            )

            phase_df = load_phase_space_all_years(
                metrics_path=metrics_path,
                years=years,
            )

            centroids = compute_regime_centroids(phase_df)

            plot_regime_centroid_trajectories(
                centroids=centroids,
                output_path=output_path,
            )

        return

    year = args.year

    if run_all or "ei" in selected_plots:
        ei, centrality = load_year_metrics(metrics_path, year)
        plot_ei_vs_centrality(
            ei, centrality, year, args.centrality_col, output_path
        )

    if (
        run_all
        or "et" in selected_plots
        or "network" in selected_plots
        or "heatmap" in selected_plots
        or "country-network" in selected_plots
        or "country-sankey" in selected_plots
        or "sector-sankey" in selected_plots
    ):
        et, centrality = load_year_metrics_with_et(metrics_path, year)

    if run_all or "et" in selected_plots:
        plot_et_embodied_vs_centrality(et, centrality, year, output_path)

    if run_all or "efficiency" in selected_plots:
        efficiency = load_efficiency(metrics_path, year)
        plot_efficiency_vs_centrality(efficiency, year, output_path)

    if run_all or "network" in selected_plots:
        plot_network_view(
            et=et,
            centrality=centrality,
            year=year,
            output_path=output_path,
            top_n=150,
            centrality_col="pagerank",
        )

    if run_all or "heatmap" in selected_plots:
        plot_et_heatmap(et, year, output_path, top_n=150)

    if run_all or "country-network" in selected_plots:
        plot_country_flow_network(
            et=et,
            year=year,
            output_path=output_path,
            top_n_edges=100,
        )

    if run_all or "country-sankey" in selected_plots:
        plot_sankey_top_flows(
            et=et,
            year=year,
            output_path=output_path,
            top_n_edges=50,
        )

    if run_all or "sector-sankey" in selected_plots:
        plot_sankey_sector_country_flows(
            et=et,
            year=year,
            output_path=output_path,
            top_n_edges=50,
        )


if __name__ == "__main__":
    main()