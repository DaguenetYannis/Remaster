from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DYNAMIC_PANEL_PATH = Path("data/final/eora_atlas_dynamic_panel.parquet")
CLUSTERS_PATH = Path("data/final/country_sector_trajectory_clusters.parquet")
OUTPUT_DIR = Path("outputs/plots/trajectory_clusters")

EXCLUDED_CLUSTERS = [5]

CLUSTER_COLORS = {
    0: "tab:blue",
    1: "tab:orange",
    2: "tab:green",
    3: "tab:red",
    4: "tab:purple",
    5: "tab:brown",
}


class TrajectoryClusterPlotter:
    def __init__(
        self,
        dynamic_panel_path: Path,
        clusters_path: Path,
        output_dir: Path,
        excluded_clusters: list[int] | None = None,
    ) -> None:
        self.dynamic_panel_path = dynamic_panel_path
        self.clusters_path = clusters_path
        self.output_dir = output_dir
        self.excluded_clusters = excluded_clusters or []
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.df = self.load_data()

    def load_data(self) -> pd.DataFrame:
        dynamic = pd.read_parquet(self.dynamic_panel_path)
        clusters = pd.read_parquet(self.clusters_path)

        df = dynamic.merge(
            clusters[["Country", "Sector", "trajectory_cluster"]],
            on=["Country", "Sector"],
            how="inner",
            validate="many_to_one",
        )

        df = df[df["Year"].between(1995, 2016)].copy()

        if self.excluded_clusters:
            df = df[~df["trajectory_cluster"].isin(self.excluded_clusters)].copy()

        return df

    def save_plot(self, filename: str) -> None:
        path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"[INFO] Saved plot: {path}")

    def get_cluster_color(self, cluster: int) -> str:
        return CLUSTER_COLORS.get(int(cluster), "tab:gray")

    def apply_quantile_limits(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        lower: float = 0.01,
        upper: float = 0.99,
    ) -> None:
        plt.xlim(df[x_col].quantile(lower), df[x_col].quantile(upper))
        plt.ylim(df[y_col].quantile(lower), df[y_col].quantile(upper))

    def add_velocity_columns(self) -> None:
        self.df = self.df.sort_values(["Country", "Sector", "Year"]).copy()

        for col in [
            "emissions_intensity",
            "g_out_network",
            "g_in_network",
            "green_capability_share",
        ]:
            self.df[f"d_{col}"] = self.df.groupby(["Country", "Sector"])[col].diff()

        self.df["speed_E_gin"] = np.sqrt(
            self.df["d_emissions_intensity"].pow(2)
            + self.df["d_g_in_network"].pow(2)
        )

        self.df["speed_E_gout"] = np.sqrt(
            self.df["d_emissions_intensity"].pow(2)
            + self.df["d_g_out_network"].pow(2)
        )

    def plot_cluster_phase_space_2d(
        self,
        x_col: str,
        y_col: str,
        filename: str,
        alpha: float = 0.35,
        quantile_limits: bool = True,
    ) -> None:
        df = self.df.dropna(subset=[x_col, y_col, "trajectory_cluster"]).copy()

        plt.figure(figsize=(9, 7))

        for cluster, group in df.groupby("trajectory_cluster"):
            plt.scatter(
                group[x_col],
                group[y_col],
                s=10,
                alpha=alpha,
                color=self.get_cluster_color(cluster),
                label=f"Cluster {cluster}",
            )

        if quantile_limits:
            self.apply_quantile_limits(df, x_col, y_col)

        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"Phase-space projection by trajectory cluster\n{x_col} vs {y_col}")
        plt.legend(fontsize=8)

        self.save_plot(filename)

    def plot_cluster_trajectories_2d(
        self,
        x_col: str,
        y_col: str,
        filename: str,
        max_trajectories_per_cluster: int = 40,
        quantile_limits: bool = True,
    ) -> None:
        df = self.df.dropna(subset=[x_col, y_col, "trajectory_cluster"]).copy()

        plt.figure(figsize=(10, 8))

        for cluster, cluster_df in df.groupby("trajectory_cluster"):
            keys = (
                cluster_df[["Country", "Sector"]]
                .drop_duplicates()
                .sample(
                    n=min(
                        max_trajectories_per_cluster,
                        cluster_df[["Country", "Sector"]].drop_duplicates().shape[0],
                    ),
                    random_state=42,
                )
            )

            sampled = cluster_df.merge(keys, on=["Country", "Sector"], how="inner")

            for _, group in sampled.groupby(["Country", "Sector"]):
                group = group.sort_values("Year")

                plt.plot(
                    group[x_col],
                    group[y_col],
                    linewidth=0.8,
                    alpha=0.18,
                    color=self.get_cluster_color(cluster),
                )

            centroid = (
                cluster_df
                .groupby("Year", as_index=False)[[x_col, y_col]]
                .mean()
                .sort_values("Year")
            )

            plt.plot(
                centroid[x_col],
                centroid[y_col],
                linewidth=3,
                marker="o",
                color=self.get_cluster_color(cluster),
                label=f"Cluster {cluster} centroid",
            )

        if quantile_limits:
            self.apply_quantile_limits(df, x_col, y_col)

        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"Trajectory corridors by cluster\n{x_col} vs {y_col}")
        plt.legend(fontsize=8)

        self.save_plot(filename)

    def plot_cluster_time_profiles(self, column: str, filename: str) -> None:
        plot_df = (
            self.df
            .groupby(["Year", "trajectory_cluster"], as_index=False)[column]
            .mean()
            .sort_values(["trajectory_cluster", "Year"])
        )

        plt.figure(figsize=(10, 6))

        for cluster, group in plot_df.groupby("trajectory_cluster"):
            plt.plot(
                group["Year"],
                group[column],
                marker="o",
                linewidth=2,
                color=self.get_cluster_color(cluster),
                label=f"Cluster {cluster}",
            )

        plt.xlabel("Year")
        plt.ylabel(column)
        plt.title(f"Cluster mean profile over time: {column}")
        plt.legend(fontsize=8)

        self.save_plot(filename)

    def plot_cluster_composition_by_sector(self, filename: str) -> None:
        composition = (
            self.df[["Country", "Sector", "trajectory_cluster"]]
            .drop_duplicates()
            .groupby(["trajectory_cluster", "Sector"])
            .size()
            .reset_index(name="n")
        )

        composition["cluster_total"] = (
            composition
            .groupby("trajectory_cluster")["n"]
            .transform("sum")
        )

        composition["share"] = composition["n"] / composition["cluster_total"]

        top = (
            composition
            .sort_values(["trajectory_cluster", "n"], ascending=[True, False])
            .groupby("trajectory_cluster")
            .head(6)
        )

        top["label"] = (
            "C" + top["trajectory_cluster"].astype(str)
            + " | "
            + top["Sector"]
        )

        top["color"] = top["trajectory_cluster"].map(self.get_cluster_color)

        plt.figure(figsize=(11, 8))
        plt.barh(top["label"], top["share"], color=top["color"])
        plt.xlabel("Share of country-sector trajectories within cluster")
        plt.title("Top sectors within each trajectory cluster")

        self.save_plot(filename)

    def plot_vector_field(
        self,
        x_col: str,
        y_col: str,
        dx_col: str,
        dy_col: str,
        filename: str,
        max_points_per_cluster: int = 1000,
        normalize_arrows: bool = True,
        quantile_limits: bool = True,
    ) -> None:
        df = self.df.dropna(
            subset=[x_col, y_col, dx_col, dy_col, "trajectory_cluster"]
        ).copy()

        plt.figure(figsize=(10, 8))

        for cluster, group in df.groupby("trajectory_cluster"):
            sample = group.sample(
                n=min(max_points_per_cluster, len(group)),
                random_state=42,
            ).copy()

            dx = sample[dx_col].to_numpy()
            dy = sample[dy_col].to_numpy()

            if normalize_arrows:
                norm = np.sqrt(dx ** 2 + dy ** 2)
                norm[norm == 0] = np.nan
                dx = dx / norm
                dy = dy / norm

            plt.quiver(
                sample[x_col],
                sample[y_col],
                dx,
                dy,
                angles="xy",
                scale_units="xy",
                scale=25 if normalize_arrows else 1,
                alpha=0.35,
                color=self.get_cluster_color(cluster),
                label=f"Cluster {cluster}",
            )

        if quantile_limits:
            self.apply_quantile_limits(df, x_col, y_col)

        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"Normalized vector field by trajectory cluster\n{x_col} vs {y_col}")
        plt.legend(fontsize=8)

        self.save_plot(filename)

    def plot_speed_profile(self, speed_col: str, filename: str) -> None:
        plot_df = (
            self.df
            .groupby(["Year", "trajectory_cluster"], as_index=False)[speed_col]
            .mean()
            .sort_values(["trajectory_cluster", "Year"])
        )

        plt.figure(figsize=(10, 6))

        for cluster, group in plot_df.groupby("trajectory_cluster"):
            plt.plot(
                group["Year"],
                group[speed_col],
                marker="o",
                linewidth=2,
                color=self.get_cluster_color(cluster),
                label=f"Cluster {cluster}",
            )

        plt.xlabel("Year")
        plt.ylabel(speed_col)
        plt.title(f"Mean trajectory speed over time: {speed_col}")
        plt.legend(fontsize=8)

        self.save_plot(filename)

    def run_default_plots(self) -> None:
        self.add_velocity_columns()

        self.plot_cluster_phase_space_2d(
            x_col="emissions_intensity",
            y_col="g_out_network",
            filename="clusters_phase_E_gout_clean.png",
        )

        self.plot_cluster_phase_space_2d(
            x_col="emissions_intensity",
            y_col="g_in_network",
            filename="clusters_phase_E_gin_clean.png",
        )

        self.plot_cluster_phase_space_2d(
            x_col="g_out_network",
            y_col="g_in_network",
            filename="clusters_phase_gout_gin_clean.png",
        )

        self.plot_cluster_trajectories_2d(
            x_col="emissions_intensity",
            y_col="g_out_network",
            filename="cluster_trajectories_E_gout_clean.png",
        )

        self.plot_cluster_trajectories_2d(
            x_col="emissions_intensity",
            y_col="g_in_network",
            filename="cluster_trajectories_E_gin_clean.png",
        )

        for column in [
            "emissions_intensity",
            "g_out_network",
            "g_in_network",
            "green_capability_share",
        ]:
            self.plot_cluster_time_profiles(
                column=column,
                filename=f"cluster_time_profile_{column}_clean.png",
            )

        self.plot_vector_field(
            x_col="emissions_intensity",
            y_col="g_in_network",
            dx_col="d_emissions_intensity",
            dy_col="d_g_in_network",
            filename="vector_field_E_gin_normalized_clean.png",
        )

        self.plot_vector_field(
            x_col="emissions_intensity",
            y_col="g_out_network",
            dx_col="d_emissions_intensity",
            dy_col="d_g_out_network",
            filename="vector_field_E_gout_normalized_clean.png",
        )

        self.plot_speed_profile(
            speed_col="speed_E_gin",
            filename="speed_profile_E_gin_clean.png",
        )

        self.plot_speed_profile(
            speed_col="speed_E_gout",
            filename="speed_profile_E_gout_clean.png",
        )

        self.plot_cluster_composition_by_sector(
            filename="cluster_sector_composition_clean.png"
        )


def main() -> None:
    plotter = TrajectoryClusterPlotter(
        dynamic_panel_path=DYNAMIC_PANEL_PATH,
        clusters_path=CLUSTERS_PATH,
        output_dir=OUTPUT_DIR,
        excluded_clusters=EXCLUDED_CLUSTERS,
    )

    plotter.run_default_plots()


if __name__ == "__main__":
    main()