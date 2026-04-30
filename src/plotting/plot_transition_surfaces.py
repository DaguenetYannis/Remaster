from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT_PATH = Path("data/final/transition_dynamics.parquet")
DEFAULT_OUTPUT_DIR = Path("outputs/plots/transition_surfaces")


@dataclass(frozen=True)
class TransitionSurfaceConfig:
    input_path: Path = DEFAULT_INPUT_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    bins: int = 10
    grid_size: int = 120
    smoothing_power: float = 2.0
    min_distance: float = 1e-12


class TransitionSurfacePlotter:
    def __init__(self, config: TransitionSurfaceConfig) -> None:
        self.config = config

    def run(self) -> None:
        df = self._load()
        df = self._prepare(df)

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        self.plot_global_surface(df)
        self.plot_global_interpolated_surface(df)
        self.plot_surfaces_by_initial_ei_quantile(df)
        self.plot_interpolated_surfaces_by_initial_ei_quantile(df)

    def _load(self) -> pd.DataFrame:
        if not self.config.input_path.exists():
            raise FileNotFoundError(f"Missing input file: {self.config.input_path}")

        logging.info("Loading transition dynamics: %s", self.config.input_path)
        return pd.read_parquet(self.config.input_path)

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        required = [
            "emissions_intensity",
            "delta_ei",
            "green_capability_readiness",
            "capability_ecosystem_exposure",
        ]

        self._require(df, required)

        out = df.copy()

        for col in required:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        out["ei_reduction"] = -out["delta_ei"]

        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.dropna(
            subset=[
                "emissions_intensity",
                "green_capability_readiness",
                "capability_ecosystem_exposure",
                "ei_reduction",
            ]
        )

        out["initial_ei_quantile"] = pd.qcut(
            out["emissions_intensity"],
            q=4,
            labels=["Q1 low EI", "Q2", "Q3", "Q4 high EI"],
            duplicates="drop",
        )

        return out

    def plot_global_surface(self, df: pd.DataFrame) -> None:
        surface = self._build_surface(df)

        self._plot_surface(
            surface=surface,
            title="Transition surface: EI reduction by capability readiness and ecosystem exposure",
            filename="transition_surface_global.png",
        )

    def plot_global_interpolated_surface(self, df: pd.DataFrame) -> None:
        surface = self._build_surface(df)

        self._plot_interpolated_surface(
            surface=surface,
            title="Interpolated transition surface: EI reduction by capability readiness and ecosystem exposure",
            filename="transition_surface_global_interpolated.png",
        )

    def plot_surfaces_by_initial_ei_quantile(self, df: pd.DataFrame) -> None:
        for quantile, group in df.groupby("initial_ei_quantile", observed=True):
            if len(group) < 100:
                logging.warning("Skipping %s: too few observations", quantile)
                continue

            surface = self._build_surface(group)
            safe_name = str(quantile).replace(" ", "_").lower()

            self._plot_surface(
                surface=surface,
                title=f"Transition surface: EI reduction | {quantile}",
                filename=f"transition_surface_{safe_name}.png",
            )

    def plot_interpolated_surfaces_by_initial_ei_quantile(self, df: pd.DataFrame) -> None:
        for quantile, group in df.groupby("initial_ei_quantile", observed=True):
            if len(group) < 100:
                logging.warning("Skipping interpolated %s: too few observations", quantile)
                continue

            surface = self._build_surface(group)
            safe_name = str(quantile).replace(" ", "_").lower()

            self._plot_interpolated_surface(
                surface=surface,
                title=f"Interpolated transition surface: EI reduction | {quantile}",
                filename=f"transition_surface_{safe_name}_interpolated.png",
            )

    def _build_surface(self, df: pd.DataFrame) -> pd.DataFrame:
        temp = df[
            [
                "green_capability_readiness",
                "capability_ecosystem_exposure",
                "ei_reduction",
            ]
        ].copy()

        temp["c_bin"] = pd.qcut(
            temp["green_capability_readiness"],
            q=self.config.bins,
            duplicates="drop",
        )

        temp["ce_bin"] = pd.qcut(
            temp["capability_ecosystem_exposure"],
            q=self.config.bins,
            duplicates="drop",
        )

        surface = (
            temp.groupby(["c_bin", "ce_bin"], observed=True)
            .agg(
                c_mean=("green_capability_readiness", "mean"),
                ce_mean=("capability_ecosystem_exposure", "mean"),
                ei_reduction_mean=("ei_reduction", "mean"),
                n=("ei_reduction", "size"),
            )
            .reset_index()
        )

        return surface

    def _plot_surface(
        self,
        surface: pd.DataFrame,
        title: str,
        filename: str,
    ) -> None:
        plt.figure(figsize=(9, 7))

        scatter = plt.scatter(
            surface["c_mean"],
            surface["ce_mean"],
            c=surface["ei_reduction_mean"],
            s=np.sqrt(surface["n"].clip(lower=1)) * 12,
            alpha=0.9,
        )

        plt.colorbar(scatter, label="Mean EI reduction: -ΔEI")
        plt.xlabel("Capability readiness C(i,t)")
        plt.ylabel("Capability ecosystem exposure CE(i,t)")
        plt.title(title)

        self._savefig(filename)

    def _plot_interpolated_surface(
        self,
        surface: pd.DataFrame,
        title: str,
        filename: str,
    ) -> None:
        clean = surface.dropna(
            subset=["c_mean", "ce_mean", "ei_reduction_mean"]
        ).copy()

        if len(clean) < 4:
            logging.warning("Skipping interpolated plot %s: too few surface points", filename)
            return

        x = clean["c_mean"].to_numpy()
        y = clean["ce_mean"].to_numpy()
        z = clean["ei_reduction_mean"].to_numpy()

        grid_x, grid_y, grid_z = self._inverse_distance_weighted_grid(x, y, z)

        plt.figure(figsize=(9, 7))

        image = plt.imshow(
            grid_z,
            origin="lower",
            aspect="auto",
            extent=[
                grid_x.min(),
                grid_x.max(),
                grid_y.min(),
                grid_y.max(),
            ],
        )

        plt.scatter(
            x,
            y,
            c=z,
            s=np.sqrt(clean["n"].clip(lower=1)) * 8,
            edgecolors="black",
            linewidths=0.3,
            alpha=0.75,
        )

        plt.colorbar(image, label="Interpolated mean EI reduction: -ΔEI")
        plt.xlabel("Capability readiness C(i,t)")
        plt.ylabel("Capability ecosystem exposure CE(i,t)")
        plt.title(title)

        self._savefig(filename)

    def _inverse_distance_weighted_grid(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        grid_x = np.linspace(x.min(), x.max(), self.config.grid_size)
        grid_y = np.linspace(y.min(), y.max(), self.config.grid_size)

        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)

        flat_x = mesh_x.ravel()
        flat_y = mesh_y.ravel()

        distances = np.sqrt(
            (flat_x[:, None] - x[None, :]) ** 2
            + (flat_y[:, None] - y[None, :]) ** 2
        )

        distances = np.maximum(distances, self.config.min_distance)
        weights = 1 / (distances ** self.config.smoothing_power)

        interpolated = (weights @ z) / weights.sum(axis=1)
        grid_z = interpolated.reshape(mesh_x.shape)

        return grid_x, grid_y, grid_z

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
        description="Plot transition response surfaces."
    )
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=120)
    parser.add_argument("--smoothing-power", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    args = parse_args()

    config = TransitionSurfaceConfig(
        input_path=Path(args.input_path),
        output_dir=Path(args.output_dir),
        bins=args.bins,
        grid_size=args.grid_size,
        smoothing_power=args.smoothing_power,
    )

    plotter = TransitionSurfacePlotter(config)
    plotter.run()


if __name__ == "__main__":
    main()