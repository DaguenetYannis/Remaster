from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class TransitionPlotter:
    scenario_name: str
    tables: Dict[str, pd.DataFrame]
    output_dir: Path

    def run(self) -> None:
        self._prepare_output_dir()

        self.plot_regime_transition_matrix()
        self.plot_regime_change_share()
        self.plot_vector_field()
        self.plot_green_gap_summary()
        self.plot_centroid_trajectories()

    def _prepare_output_dir(self) -> None:
        (self.output_dir / "plots").mkdir(parents=True, exist_ok=True)

    def _plot_path(self, name: str) -> Path:
        return self.output_dir / "plots" / f"{self.scenario_name}_{name}.png"

    def plot_regime_transition_matrix(self) -> None:
        df = self.tables["regime_transition"].copy()

        matrix = df.pivot(
            index="regime",
            columns="regime_next",
            values="share",
        ).fillna(0)

        fig, ax = plt.subplots(figsize=(7, 6))

        image = ax.imshow(matrix.values)

        ax.set_title(f"Regime transition matrix — {self.scenario_name}")
        ax.set_xlabel("Next regime")
        ax.set_ylabel("Initial regime")

        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, rotation=45, ha="right")

        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index)

        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = matrix.iloc[row_index, col_index]
                ax.text(
                    col_index,
                    row_index,
                    f"{value:.1%}",
                    ha="center",
                    va="center",
                )

        fig.colorbar(image, ax=ax, label="Share")
        fig.tight_layout()
        fig.savefig(self._plot_path("regime_transition_matrix"), dpi=200)
        plt.close(fig)

    def plot_regime_change_share(self) -> None:
        df = self.tables["regime_change"].copy()

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(df["sim_year"], df["share"], marker="o")

        ax.set_title(f"Regime-change share — {self.scenario_name}")
        ax.set_xlabel("Simulated year")
        ax.set_ylabel("Share of nodes changing regime")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self._plot_path("regime_change_share"), dpi=200)
        plt.close(fig)

    def plot_vector_field(self) -> None:
        df = self.tables["vector_field"].copy()

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(
            df["x"],
            df["y"],
            s=df["count"].clip(lower=5, upper=300),
            alpha=0.45,
        )

        ax.quiver(
            df["x"],
            df["y"],
            df["dx"],
            df["dy"],
            angles="xy",
            scale_units="xy",
            scale=1,
            alpha=0.8,
        )

        ax.set_title(f"Empirical transition field — {self.scenario_name}")
        ax.set_xlabel("log10(1 + out-strength)")
        ax.set_ylabel("Outgoing network green-ness")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self._plot_path("vector_field"), dpi=200)
        plt.close(fig)

    def plot_green_gap_summary(self) -> None:
        df = self.tables["green_gap"].copy()

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(df["sim_year"], df["50%"], marker="o", label="Median")
        ax.fill_between(
            df["sim_year"],
            df["10%"],
            df["90%"],
            alpha=0.25,
            label="10th–90th percentile",
        )

        ax.axhline(0, linestyle="--", linewidth=1)

        ax.set_title(f"Network green-ness gap — {self.scenario_name}")
        ax.set_xlabel("Simulated year")
        ax.set_ylabel("g_out_network - g_base")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self._plot_path("green_gap_summary"), dpi=200)
        plt.close(fig)

    def plot_centroid_trajectories(self) -> None:
        df = self.tables["centroids"].copy()

        fig, ax = plt.subplots(figsize=(8, 6))

        for regime, group in df.groupby("regime"):
            group = group.sort_values("sim_year")

            ax.plot(
                group["x"],
                group["y"],
                marker="o",
                label=regime,
            )

            for _, row in group.iterrows():
                ax.text(
                    row["x"],
                    row["y"],
                    str(int(row["sim_year"])),
                    fontsize=8,
                    alpha=0.7,
                )

        ax.set_title(f"Regime centroid trajectories — {self.scenario_name}")
        ax.set_xlabel("Mean log10(1 + out-strength)")
        ax.set_ylabel("Mean outgoing network green-ness")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self._plot_path("centroid_trajectories"), dpi=200)
        plt.close(fig)