from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class ABMPlotter:
    def __init__(self, output_dir: Path = Path("outputs/abm/plots")) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_all(
        self,
        aggregate_results: pd.DataFrame,
        node_results: pd.DataFrame,
    ) -> None:
        self.plot_scenario_total_emissions(aggregate_results)
        self.plot_scenario_total_output(aggregate_results)
        self.plot_scenario_mean_ei(aggregate_results)
        self.plot_greeness_trajectories(aggregate_results)
        self.plot_phase_space(node_results)
        self.plot_regime_shares(node_results)
        self.plot_regime_centroids(node_results)
        self.plot_selected_node_trajectories(node_results)
        self.plot_distributions(node_results)
        self.plot_top_embodied_carbon_flows(node_results)

    def _save(self, filename: str) -> None:
        path = self.output_dir / filename
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close()

    def _plot_scenario_line(
        self,
        df: pd.DataFrame,
        y_col: str,
        title: str,
        ylabel: str,
        filename: str,
    ) -> None:
        self._require_columns(df, ["step", "scenario", y_col])

        plt.figure(figsize=(9, 5))

        for scenario, group in df.groupby("scenario"):
            plt.plot(group["step"], group[y_col], marker="o", label=scenario)

        plt.xlabel("Simulation step")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(alpha=0.3)
        self._save(filename)

    def plot_scenario_total_emissions(self, aggregate_results: pd.DataFrame) -> None:
        self._plot_scenario_line(
            aggregate_results,
            y_col="total_emissions",
            title="Scenario comparison: total emissions",
            ylabel="Total emissions",
            filename="01_scenario_total_emissions.png",
        )

    def plot_scenario_total_output(self, aggregate_results: pd.DataFrame) -> None:
        self._plot_scenario_line(
            aggregate_results,
            y_col="total_output",
            title="Scenario comparison: total output",
            ylabel="Total output",
            filename="02_scenario_total_output.png",
        )

    def plot_scenario_mean_ei(self, aggregate_results: pd.DataFrame) -> None:
        self._plot_scenario_line(
            aggregate_results,
            y_col="mean_ei",
            title="Scenario comparison: mean emissions intensity",
            ylabel="Mean emissions intensity",
            filename="03_scenario_mean_ei.png",
        )

    def plot_greeness_trajectories(self, aggregate_results: pd.DataFrame) -> None:
        self._require_columns(
            aggregate_results,
            ["step", "scenario", "mean_g_local", "mean_g_in", "mean_g_out"],
        )

        green_cols = ["mean_g_local", "mean_g_in", "mean_g_out"]

        for scenario, group in aggregate_results.groupby("scenario"):
            plt.figure(figsize=(9, 5))

            for col in green_cols:
                plt.plot(group["step"], group[col], marker="o", label=col)

            plt.xlabel("Simulation step")
            plt.ylabel("Mean green-ness")
            plt.title(f"Local, incoming, and outgoing green-ness: {scenario}")
            plt.legend()
            plt.grid(alpha=0.3)
            self._save(f"04_greeness_trajectories_{scenario}.png")

    def plot_phase_space(self, node_results: pd.DataFrame) -> None:
        self._require_columns(node_results, ["step", "scenario", "out_strength", "g_out"])

        for scenario, group in node_results.groupby("scenario"):
            final_step = group["step"].max()
            data = group[group["step"] == final_step].copy()
            data["log_out_strength"] = np.log1p(data["out_strength"])

            plt.figure(figsize=(7, 6))
            plt.scatter(data["log_out_strength"], data["g_out"], alpha=0.35, s=12)
            plt.xlabel("log(1 + out_strength)")
            plt.ylabel("Outgoing network green-ness")
            plt.title(f"Phase space: structural position × green-ness ({scenario})")
            plt.grid(alpha=0.3)
            self._save(f"05_phase_space_{scenario}.png")

    def plot_regime_shares(self, node_results: pd.DataFrame) -> None:
        self._require_columns(node_results, ["step", "scenario", "regime"])

        shares = (
            node_results
            .groupby(["scenario", "step", "regime"])
            .size()
            .reset_index(name="n")
        )

        shares["share"] = shares["n"] / shares.groupby(["scenario", "step"])["n"].transform("sum")

        for scenario, group in shares.groupby("scenario"):
            pivot = group.pivot(index="step", columns="regime", values="share").fillna(0)

            plt.figure(figsize=(9, 5))

            for regime in pivot.columns:
                plt.plot(pivot.index, pivot[regime], marker="o", label=regime)

            plt.xlabel("Simulation step")
            plt.ylabel("Share of nodes")
            plt.title(f"Regime shares over time: {scenario}")
            plt.legend()
            plt.grid(alpha=0.3)
            self._save(f"06_regime_shares_{scenario}.png")

    def plot_regime_centroids(self, node_results: pd.DataFrame) -> None:
        self._require_columns(
            node_results,
            ["step", "scenario", "regime", "out_strength", "g_out"],
        )

        data = node_results.copy()
        data["log_out_strength"] = np.log1p(data["out_strength"])

        centroids = (
            data
            .groupby(["scenario", "step", "regime"], as_index=False)
            .agg(
                mean_log_out_strength=("log_out_strength", "mean"),
                mean_g_out=("g_out", "mean"),
            )
        )

        for scenario, group in centroids.groupby("scenario"):
            plt.figure(figsize=(7, 6))

            for regime, regime_group in group.groupby("regime"):
                plt.plot(
                    regime_group["mean_log_out_strength"],
                    regime_group["mean_g_out"],
                    marker="o",
                    label=regime,
                )

            plt.xlabel("Mean log(1 + out_strength)")
            plt.ylabel("Mean outgoing green-ness")
            plt.title(f"Regime centroid trajectories: {scenario}")
            plt.legend()
            plt.grid(alpha=0.3)
            self._save(f"07_regime_centroids_{scenario}.png")

    def plot_selected_node_trajectories(
        self,
        node_results: pd.DataFrame,
        max_nodes: int = 12,
    ) -> None:
        self._require_columns(
            node_results,
            ["step", "scenario", "country_sector", "EI", "g_out"],
        )

        for scenario, group in node_results.groupby("scenario"):
            final_step = group["step"].max()
            final_data = group[group["step"] == final_step]

            selected_nodes = (
                final_data
                .sort_values("EI", ascending=False)
                .head(max_nodes)["country_sector"]
                .tolist()
            )

            data = group[group["country_sector"].isin(selected_nodes)]

            plt.figure(figsize=(10, 6))

            for node, node_group in data.groupby("country_sector"):
                plt.plot(node_group["step"], node_group["EI"], label=node)

            plt.xlabel("Simulation step")
            plt.ylabel("Emissions intensity")
            plt.title(f"Selected node EI trajectories: {scenario}")
            plt.legend(fontsize=7, loc="best")
            plt.grid(alpha=0.3)
            self._save(f"08_selected_node_ei_trajectories_{scenario}.png")

            plt.figure(figsize=(10, 6))

            for node, node_group in data.groupby("country_sector"):
                plt.plot(node_group["step"], node_group["g_out"], label=node)

            plt.xlabel("Simulation step")
            plt.ylabel("Outgoing network green-ness")
            plt.title(f"Selected node g_out trajectories: {scenario}")
            plt.legend(fontsize=7, loc="best")
            plt.grid(alpha=0.3)
            self._save(f"08_selected_node_gout_trajectories_{scenario}.png")

    def plot_distributions(self, node_results: pd.DataFrame) -> None:
        self._require_columns(node_results, ["step", "scenario", "EI", "g_out", "out_strength"])

        metrics = ["EI", "g_out", "out_strength"]

        for scenario, group in node_results.groupby("scenario"):
            final_step = group["step"].max()
            data = group[group["step"] == final_step]

            for metric in metrics:
                plt.figure(figsize=(8, 5))
                plt.hist(data[metric].dropna(), bins=50)
                plt.xlabel(metric)
                plt.ylabel("Node count")
                plt.title(f"Distribution of {metric} at final step: {scenario}")
                plt.grid(alpha=0.3)
                self._save(f"09_distribution_{metric}_{scenario}.png")

    def plot_top_embodied_carbon_flows(
        self,
        node_results: pd.DataFrame,
        top_n: int = 25,
    ) -> None:
        """
        Placeholder-compatible plot.

        This expects edge-level simulated output if available:
        source, target, step, scenario, embodied_carbon_flow.

        If node_results does not contain these columns, the function skips silently.
        """
        required = ["source", "target", "step", "scenario", "embodied_carbon_flow"]

        if not all(col in node_results.columns for col in required):
            return

        for scenario, group in node_results.groupby("scenario"):
            final_step = group["step"].max()

            data = (
                group[group["step"] == final_step]
                .sort_values("embodied_carbon_flow", ascending=False)
                .head(top_n)
                .copy()
            )

            data["edge"] = data["source"] + " → " + data["target"]

            plt.figure(figsize=(10, 8))
            plt.barh(data["edge"], data["embodied_carbon_flow"])
            plt.xlabel("Embodied carbon flow")
            plt.ylabel("Edge")
            plt.title(f"Top simulated embodied-carbon flows: {scenario}")
            plt.gca().invert_yaxis()
            plt.grid(axis="x", alpha=0.3)
            self._save(f"10_top_embodied_carbon_flows_{scenario}.png")

    @staticmethod
    def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
        missing = [col for col in columns if col not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")