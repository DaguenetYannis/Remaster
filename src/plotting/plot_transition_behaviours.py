from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT_PATH = Path("data/final/transition_dynamics.parquet")
DEFAULT_OUTPUT_DIR = Path("outputs/plots/transition_behaviours")


@dataclass(frozen=True)
class TransitionBehaviourPlotConfig:
    input_path: Path = DEFAULT_INPUT_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    year_col: str = "Year"
    country_col: str = "Country"
    sector_col: str = "Sector"


class TransitionBehaviourPlotter:
    def __init__(self, config: TransitionBehaviourPlotConfig) -> None:
        self.config = config

    def run(self) -> None:
        df = self._load()
        df = self._prepare(df)

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        self.plot_transition_response_space(df)
        self.plot_green_precedence_response(df)
        self.plot_capability_readiness_response(df)
        self.plot_behaviour_quadrants(df)
        self.plot_selected_country_trajectories(df)

    def _load(self) -> pd.DataFrame:
        if not self.config.input_path.exists():
            raise FileNotFoundError(f"Missing input file: {self.config.input_path}")

        logging.info("Loading transition dynamics: %s", self.config.input_path)
        return pd.read_parquet(self.config.input_path)

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        numeric_cols = [
            "emissions_intensity",
            "delta_ei",
            "delta_network_green",
            "delta_green_capability_share",
            "green_capability_readiness",
            "capability_ecosystem_exposure",
            "network_green_exposure",
            "upstream_ei_exposure",
            "out_strength",
            "pagerank",
        ]

        for col in numeric_cols:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        out["log_out_strength"] = np.log1p(out.get("out_strength", 0))
        out["ei_reduction"] = -out["delta_ei"]

        return out.replace([np.inf, -np.inf], np.nan)

    def plot_transition_response_space(self, df: pd.DataFrame) -> None:
        required = ["emissions_intensity", "upstream_ei_exposure", "ei_reduction"]
        self._require(df, required)

        sample = df.dropna(subset=required).sample(
            n=min(30_000, len(df.dropna(subset=required))),
            random_state=42,
        )

        plt.figure(figsize=(9, 6))
        plt.scatter(
            sample["emissions_intensity"],
            sample["upstream_ei_exposure"],
            s=8,
            alpha=0.25,
            c=sample["ei_reduction"],
        )
        plt.xscale("symlog")
        plt.yscale("symlog")
        plt.colorbar(label="EI reduction: -ΔEI")
        plt.xlabel("Initial emissions intensity")
        plt.ylabel("Upstream EI exposure")
        plt.title("Behaviour space: initial state, network exposure, and EI reduction")
        self._savefig("transition_response_space.png")

    def plot_green_precedence_response(self, df: pd.DataFrame) -> None:
        required = ["capability_ecosystem_exposure", "ei_reduction"]
        self._require(df, required)

        plot_df = self._bin_response(
            df=df,
            x_col="capability_ecosystem_exposure",
            y_col="ei_reduction",
            bins=20,
        )

        plt.figure(figsize=(9, 6))
        plt.plot(plot_df["x_mid"], plot_df["y_mean"], marker="o")
        plt.axhline(0, linewidth=1)
        plt.xlabel("Green precedence / capability ecosystem exposure")
        plt.ylabel("Mean EI reduction: -ΔEI")
        plt.title("Do green-precedence environments precede EI reductions?")
        self._savefig("green_precedence_response.png")

    def plot_capability_readiness_response(self, df: pd.DataFrame) -> None:
        required = ["green_capability_readiness", "ei_reduction"]
        self._require(df, required)

        plot_df = self._bin_response(
            df=df,
            x_col="green_capability_readiness",
            y_col="ei_reduction",
            bins=20,
        )

        plt.figure(figsize=(9, 6))
        plt.plot(plot_df["x_mid"], plot_df["y_mean"], marker="o")
        plt.axhline(0, linewidth=1)
        plt.xlabel("Green capability readiness C(i,t)")
        plt.ylabel("Mean EI reduction: -ΔEI")
        plt.title("Capability readiness and subsequent emissions-intensity reduction")
        self._savefig("capability_readiness_response.png")

    def plot_behaviour_quadrants(self, df: pd.DataFrame) -> None:
        required = ["delta_network_green", "delta_ei"]
        self._require(df, required)

        plot_df = df.dropna(subset=required).copy()

        plot_df["behaviour_type"] = np.select(
            [
                (plot_df["delta_network_green"] > 0) & (plot_df["delta_ei"] < 0),
                (plot_df["delta_network_green"] > 0) & (plot_df["delta_ei"] >= 0),
                (plot_df["delta_network_green"] <= 0) & (plot_df["delta_ei"] < 0),
                (plot_df["delta_network_green"] <= 0) & (plot_df["delta_ei"] >= 0),
            ],
            [
                "full_green_upgrade",
                "network_upgrade_only",
                "local_EI_upgrade_only",
                "brown_or_stagnant",
            ],
            default="unknown",
        )

        shares = (
            plot_df.groupby([self.config.year_col, "behaviour_type"])
            .size()
            .reset_index(name="count")
        )
        shares["share"] = shares["count"] / shares.groupby(self.config.year_col)["count"].transform("sum")

        pivot = shares.pivot(
            index=self.config.year_col,
            columns="behaviour_type",
            values="share",
        ).fillna(0)

        plt.figure(figsize=(10, 6))
        for col in pivot.columns:
            plt.plot(pivot.index, pivot[col], marker="o", label=col)

        plt.xlabel("Year")
        plt.ylabel("Share of country-sector transitions")
        plt.title("Behaviour typologies over time")
        plt.legend(loc="best", fontsize=8)
        self._savefig("behaviour_typologies_over_time.png")

    def plot_selected_country_trajectories(self, df: pd.DataFrame) -> None:
        required = ["log_out_strength", "network_green_exposure"]
        self._require(df, required)

        selected = ["BRA", "KOR", "BGD"]
        plot_df = df[df[self.config.country_col].isin(selected)].dropna(subset=required)

        if plot_df.empty:
            logging.warning("No selected country trajectories found.")
            return

        agg = (
            plot_df.groupby([self.config.country_col, self.config.year_col], as_index=False)
            .agg(
                log_out_strength=("log_out_strength", "mean"),
                network_green_exposure=("network_green_exposure", "mean"),
            )
        )

        plt.figure(figsize=(9, 6))

        for country, group in agg.groupby(self.config.country_col):
            group = group.sort_values(self.config.year_col)
            plt.plot(
                group["log_out_strength"],
                group["network_green_exposure"],
                marker="o",
                label=country,
            )

        plt.xlabel("Mean log(1 + out-strength)")
        plt.ylabel("Mean network green exposure")
        plt.title("Country trajectories in behaviour space")
        plt.legend()
        self._savefig("selected_country_trajectories.png")

    def _bin_response(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        bins: int,
    ) -> pd.DataFrame:
        temp = df[[x_col, y_col]].dropna().copy()
        temp["bin"] = pd.qcut(temp[x_col], q=bins, duplicates="drop")

        out = (
            temp.groupby("bin", observed=True)
            .agg(
                x_mid=(x_col, "mean"),
                y_mean=(y_col, "mean"),
                y_median=(y_col, "median"),
                n=(y_col, "size"),
            )
            .reset_index(drop=True)
        )

        return out

    def _savefig(self, filename: str) -> None:
        path = self.config.output_dir / filename
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()
        logging.info("Saved plot: %s", path)

    @staticmethod
    def _require(df: pd.DataFrame, columns: list[str]) -> None:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot transition behaviour patterns from transition dynamics."
    )
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    args = parse_args()

    config = TransitionBehaviourPlotConfig(
        input_path=Path(args.input_path),
        output_dir=Path(args.output_dir),
    )

    plotter = TransitionBehaviourPlotter(config)
    plotter.run()


if __name__ == "__main__":
    main()