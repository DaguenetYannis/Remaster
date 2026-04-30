from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor


DEFAULT_INPUT_PATH = Path("data/final/transition_dynamics.parquet")
DEFAULT_OUTPUT_DIR = Path("outputs/plots/transition_vector_fields")


@dataclass(frozen=True)
class VectorFieldConfig:
    input_path: Path = DEFAULT_INPUT_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    bins: int = 8
    min_observations_per_cell: int = 20
    knn_neighbors: int = 200
    grid_size: int = 40


class TransitionVectorFieldPlotter:
    def __init__(self, config: VectorFieldConfig) -> None:
        self.config = config

    def run(self) -> None:
        df = self._load()
        df = self._prepare(df)

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        self.plot_vector_field(df, "global", "Transition vector field: global")
        self.plot_knn_vector_field(df, "global", "KNN transition vector field: global")
        self.plot_pca_transition_axis(df, "global", "PCA transition axis: global")

        for quantile, group in df.groupby("initial_ei_quantile", observed=True):
            safe_name = str(quantile).replace(" ", "_").lower()

            self.plot_vector_field(
                group,
                safe_name,
                f"Transition vector field: {quantile}",
            )

            self.plot_knn_vector_field(
                group,
                safe_name,
                f"KNN transition vector field: {quantile}",
            )

            self.plot_pca_transition_axis(
                group,
                safe_name,
                f"PCA transition axis: {quantile}",
            )

        self._save_vector_field_tables(df)
        self._save_pca_tables(df)

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
            "green_capability_readiness_next",
            "capability_ecosystem_exposure",
            "capability_ecosystem_exposure_next",
        ]
        self._require(df, required)

        out = df.copy()

        for col in required:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        out["delta_c"] = (
            out["green_capability_readiness_next"]
            - out["green_capability_readiness"]
        )

        out["delta_ce"] = (
            out["capability_ecosystem_exposure_next"]
            - out["capability_ecosystem_exposure"]
        )

        out["ei_reduction"] = -out["delta_ei"]

        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.dropna(
            subset=[
                "emissions_intensity",
                "green_capability_readiness",
                "capability_ecosystem_exposure",
                "delta_c",
                "delta_ce",
                "delta_ei",
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

    def _build_vector_field(self, df: pd.DataFrame) -> pd.DataFrame:
        temp = df.copy()

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

        field = (
            temp.groupby(["c_bin", "ce_bin"], observed=True)
            .agg(
                c_mean=("green_capability_readiness", "mean"),
                ce_mean=("capability_ecosystem_exposure", "mean"),
                delta_c_mean=("delta_c", "mean"),
                delta_ce_mean=("delta_ce", "mean"),
                delta_ei_mean=("delta_ei", "mean"),
                ei_reduction_mean=("ei_reduction", "mean"),
                n=("delta_ei", "size"),
            )
            .reset_index()
        )

        field = field[field["n"] >= self.config.min_observations_per_cell].copy()

        return field

    def plot_vector_field(self, df: pd.DataFrame, safe_name: str, title: str) -> None:
        field = self._build_vector_field(df)

        if field.empty:
            logging.warning("Skipping %s: no cells with enough observations", safe_name)
            return

        plt.figure(figsize=(9, 7))

        scatter = plt.scatter(
            field["c_mean"],
            field["ce_mean"],
            c=field["ei_reduction_mean"],
            s=np.sqrt(field["n"]) * 18,
            alpha=0.75,
        )

        plt.quiver(
            field["c_mean"],
            field["ce_mean"],
            field["delta_c_mean"],
            field["delta_ce_mean"],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.003,
        )

        plt.colorbar(scatter, label="Mean EI reduction: -ΔEI")
        plt.axhline(0, linewidth=0.8)
        plt.xlabel("Capability readiness C(i,t)")
        plt.ylabel("Capability ecosystem exposure CE(i,t)")
        plt.title(title)

        self._savefig(f"transition_vector_field_{safe_name}.png")

    def plot_knn_vector_field(self, df: pd.DataFrame, safe_name: str, title: str) -> None:
        if len(df) < 10:
            logging.warning("Skipping KNN %s: too few observations", safe_name)
            return

        x = df[
            [
                "green_capability_readiness",
                "capability_ecosystem_exposure",
            ]
        ].to_numpy()

        y_delta_c = df["delta_c"].to_numpy()
        y_delta_ce = df["delta_ce"].to_numpy()
        y_ei_reduction = df["ei_reduction"].to_numpy()

        n_neighbors = min(self.config.knn_neighbors, len(df))

        model_delta_c = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights="distance",
        ).fit(x, y_delta_c)

        model_delta_ce = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights="distance",
        ).fit(x, y_delta_ce)

        model_ei = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights="distance",
        ).fit(x, y_ei_reduction)

        c_grid = np.linspace(
            df["green_capability_readiness"].min(),
            df["green_capability_readiness"].max(),
            self.config.grid_size,
        )

        ce_grid = np.linspace(
            df["capability_ecosystem_exposure"].min(),
            df["capability_ecosystem_exposure"].max(),
            self.config.grid_size,
        )

        grid_c, grid_ce = np.meshgrid(c_grid, ce_grid)
        grid_points = np.column_stack([grid_c.ravel(), grid_ce.ravel()])

        delta_c_smooth = model_delta_c.predict(grid_points).reshape(grid_c.shape)
        delta_ce_smooth = model_delta_ce.predict(grid_points).reshape(grid_ce.shape)
        ei_smooth = model_ei.predict(grid_points).reshape(grid_c.shape)

        step = max(1, self.config.grid_size // 15)

        plt.figure(figsize=(9, 7))

        image = plt.imshow(
            ei_smooth,
            origin="lower",
            aspect="auto",
            extent=[
                c_grid.min(),
                c_grid.max(),
                ce_grid.min(),
                ce_grid.max(),
            ],
        )

        plt.quiver(
            grid_c[::step, ::step],
            grid_ce[::step, ::step],
            delta_c_smooth[::step, ::step],
            delta_ce_smooth[::step, ::step],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.003,
        )

        plt.scatter(
            df["green_capability_readiness"],
            df["capability_ecosystem_exposure"],
            s=4,
            alpha=0.15,
        )

        plt.colorbar(image, label="KNN-smoothed EI reduction: -ΔEI")
        plt.axhline(0, linewidth=0.8)
        plt.xlabel("Capability readiness C(i,t)")
        plt.ylabel("Capability ecosystem exposure CE(i,t)")
        plt.title(title)

        self._savefig(f"transition_vector_field_knn_{safe_name}.png")

    def plot_pca_transition_axis(self, df: pd.DataFrame, safe_name: str, title: str) -> None:
        if len(df) < 10:
            logging.warning("Skipping PCA %s: too few observations", safe_name)
            return

        coords = df[
            [
                "green_capability_readiness",
                "capability_ecosystem_exposure",
            ]
        ].to_numpy()

        deltas = df[["delta_c", "delta_ce"]].to_numpy()

        pca = PCA(n_components=1)
        z = pca.fit_transform(coords).ravel()

        direction = pca.components_[0]
        delta_z = deltas @ direction

        temp = pd.DataFrame(
            {
                "z": z,
                "delta_z": delta_z,
                "ei_reduction": df["ei_reduction"].to_numpy(),
            }
        ).dropna()

        n_bins = min(20, temp["z"].nunique())

        if n_bins < 2:
            logging.warning("Skipping PCA %s: not enough z variation", safe_name)
            return

        temp["z_bin"] = pd.qcut(temp["z"], q=n_bins, duplicates="drop")

        summary = (
            temp.groupby("z_bin", observed=True)
            .agg(
                z_mean=("z", "mean"),
                delta_z_mean=("delta_z", "mean"),
                ei_reduction_mean=("ei_reduction", "mean"),
                n=("z", "size"),
            )
            .reset_index(drop=True)
        )

        plt.figure(figsize=(9, 6))

        scatter = plt.scatter(
            summary["z_mean"],
            summary["delta_z_mean"],
            c=summary["ei_reduction_mean"],
            s=np.sqrt(summary["n"]) * 25,
            alpha=0.8,
        )

        plt.axhline(0, linewidth=0.8)
        plt.colorbar(scatter, label="Mean EI reduction: -ΔEI")
        plt.xlabel("Latent capability-ecosystem axis z")
        plt.ylabel("Mean movement along axis Δz")
        plt.title(title)

        self._savefig(f"transition_pca_axis_{safe_name}.png")

    def _save_vector_field_tables(self, df: pd.DataFrame) -> None:
        global_field = self._build_vector_field(df)
        global_field["ei_regime"] = "global"

        fields = [global_field]

        for quantile, group in df.groupby("initial_ei_quantile", observed=True):
            field = self._build_vector_field(group)
            field["ei_regime"] = str(quantile)
            fields.append(field)

        out = pd.concat(fields, ignore_index=True)

        csv_path = self.config.output_dir / "transition_vector_field_cells.csv"
        parquet_path = self.config.output_dir / "transition_vector_field_cells.parquet"

        out.to_csv(csv_path, index=False)
        out.to_parquet(parquet_path, index=False)

        logging.info("Saved vector field table: %s", csv_path)
        logging.info("Saved vector field table: %s", parquet_path)

    def _save_pca_tables(self, df: pd.DataFrame) -> None:
        rows = []

        regimes: list[tuple[str, pd.DataFrame]] = [("global", df)]
        regimes.extend(
            [
                (str(quantile), group)
                for quantile, group in df.groupby("initial_ei_quantile", observed=True)
            ]
        )

        for regime, group in regimes:
            if len(group) < 10:
                continue

            coords = group[
                [
                    "green_capability_readiness",
                    "capability_ecosystem_exposure",
                ]
            ].to_numpy()

            deltas = group[["delta_c", "delta_ce"]].to_numpy()

            pca = PCA(n_components=1)
            z = pca.fit_transform(coords).ravel()
            direction = pca.components_[0]
            delta_z = deltas @ direction

            temp = pd.DataFrame(
                {
                    "ei_regime": regime,
                    "z": z,
                    "delta_z": delta_z,
                    "ei_reduction": group["ei_reduction"].to_numpy(),
                }
            )

            n_bins = min(20, temp["z"].nunique())

            if n_bins < 2:
                continue

            temp["z_bin"] = pd.qcut(temp["z"], q=n_bins, duplicates="drop")

            summary = (
                temp.groupby(["ei_regime", "z_bin"], observed=True)
                .agg(
                    z_mean=("z", "mean"),
                    delta_z_mean=("delta_z", "mean"),
                    ei_reduction_mean=("ei_reduction", "mean"),
                    n=("z", "size"),
                )
                .reset_index()
            )

            rows.append(summary)

        if not rows:
            logging.warning("No PCA tables created")
            return

        out = pd.concat(rows, ignore_index=True)

        csv_path = self.config.output_dir / "transition_pca_axis_cells.csv"
        parquet_path = self.config.output_dir / "transition_pca_axis_cells.parquet"

        out.to_csv(csv_path, index=False)
        out.to_parquet(parquet_path, index=False)

        logging.info("Saved PCA table: %s", csv_path)
        logging.info("Saved PCA table: %s", parquet_path)

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
        description="Plot transition vector fields in capability/ecosystem space."
    )
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--min-observations-per-cell", type=int, default=20)
    parser.add_argument("--knn-neighbors", type=int, default=200)
    parser.add_argument("--grid-size", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    args = parse_args()

    config = VectorFieldConfig(
        input_path=Path(args.input_path),
        output_dir=Path(args.output_dir),
        bins=args.bins,
        min_observations_per_cell=args.min_observations_per_cell,
        knn_neighbors=args.knn_neighbors,
        grid_size=args.grid_size,
    )

    plotter = TransitionVectorFieldPlotter(config)
    plotter.run()


if __name__ == "__main__":
    main()