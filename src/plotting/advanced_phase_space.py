from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np


INPUT_PATH = Path("data/final/eora_atlas_dynamic_panel.parquet")
OUTPUT_DIR = Path("outputs/plots/phase_space")


class AdvancedPhaseSpacePlotter:
    def __init__(self, input_path: Path, output_dir: Path) -> None:
        self.input_path = input_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = self.load_data()

    def load_data(self) -> pd.DataFrame:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Dynamic panel not found: {self.input_path}")

        df = pd.read_parquet(self.input_path)

        required_cols = [
            "Country",
            "Sector",
            "Year",
            "emissions_intensity",
            "g_out_network",
            "g_in_network",
            "green_capability_share",
        ]

        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return df

    def filter_data(
        self,
        year_start: int = 1995,
        year_end: int = 2016,
        sectors: list[str] | None = None,
        countries: list[str] | None = None,
        min_active_good_count: int | None = None,
    ) -> pd.DataFrame:
        df = self.df[
            (self.df["Year"] >= year_start)
            & (self.df["Year"] <= year_end)
        ].copy()

        if sectors is not None:
            df = df[df["Sector"].isin(sectors)]

        if countries is not None:
            df = df[df["Country"].isin(countries)]

        if min_active_good_count is not None and "active_good_count" in df.columns:
            df = df[df["active_good_count"].fillna(0) >= min_active_good_count]

        return df

    def save_plot(self, filename: str) -> None:
        path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"[INFO] Saved plot: {path}")

    def plot_3d_phase_space(
        self,
        x_col: str,
        y_col: str,
        z_col: str,
        color_col: str,
        filename: str,
        year_start: int = 1995,
        year_end: int = 2016,
        sectors: list[str] | None = None,
        countries: list[str] | None = None,
        min_active_good_count: int | None = None,
        max_trajectories: int | None = 250,
    ) -> None:
        required = ["Country", "Sector", "Year", x_col, y_col, z_col, color_col]
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = self.filter_data(
            year_start=year_start,
            year_end=year_end,
            sectors=sectors,
            countries=countries,
            min_active_good_count=min_active_good_count,
        )

        df = df.dropna(subset=[x_col, y_col, z_col, color_col])

        trajectory_keys = (
            df[["Country", "Sector"]]
            .drop_duplicates()
            .sort_values(["Country", "Sector"])
        )

        if max_trajectories is not None and len(trajectory_keys) > max_trajectories:
            trajectory_keys = trajectory_keys.sample(
                n=max_trajectories,
                random_state=42,
            )

        df = df.merge(trajectory_keys, on=["Country", "Sector"], how="inner")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        scatter = ax.scatter(
            df[x_col],
            df[y_col],
            df[z_col],
            c=df[color_col],
            alpha=0.45,
            s=12,
        )

        for _, group in df.groupby(["Country", "Sector"]):
            group = group.sort_values("Year")
            if len(group) < 2:
                continue

            ax.plot(
                group[x_col],
                group[y_col],
                group[z_col],
                linewidth=0.7,
                alpha=0.25,
            )

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        ax.set_title(
            f"3D phase space: {x_col}, {y_col}, {z_col}\n"
            f"Color: {color_col} | {year_start}-{year_end}"
        )

        colorbar = fig.colorbar(scatter, ax=ax, shrink=0.65, pad=0.1)
        colorbar.set_label(color_col)

        self.save_plot(filename)

    def plot_sector_phase_space(
        self,
        sector: str,
        x_col: str,
        y_col: str,
        z_col: str,
        color_col: str,
        filename: str,
        year_start: int = 1995,
        year_end: int = 2016,
        max_trajectories: int | None = 200,
    ) -> None:
        self.plot_3d_phase_space(
            x_col=x_col,
            y_col=y_col,
            z_col=z_col,
            color_col=color_col,
            filename=filename,
            year_start=year_start,
            year_end=year_end,
            sectors=[sector],
            max_trajectories=max_trajectories,
        )

    def run_default_phase_spaces(self) -> None:
        self.plot_3d_phase_space(
            x_col="emissions_intensity",
            y_col="g_out_network",
            z_col="green_capability_share",
            color_col="g_in_network",
            filename="phase_E_gout_C_colored_gin.png",
            min_active_good_count=1,
        )

        self.plot_3d_phase_space(
            x_col="emissions_intensity",
            y_col="g_in_network",
            z_col="green_capability_share",
            color_col="g_out_network",
            filename="phase_E_gin_C_colored_gout.png",
            min_active_good_count=1,
        )

        self.plot_3d_phase_space(
            x_col="emissions_intensity",
            y_col="g_out_network",
            z_col="g_in_network",
            color_col="green_capability_share",
            filename="phase_E_gout_gin_colored_C.png",
            min_active_good_count=1,
        )

        for sector in [
            "Electrical and Machinery",
            "Transport Equipment",
            "Metal Products",
            "Petroleum, Chemical and Non-Metallic Mineral Products",
        ]:
            safe_sector_name = (
                sector.lower()
                .replace(" ", "_")
                .replace(",", "")
                .replace("&", "and")
                .replace("-", "_")
            )

            self.plot_sector_phase_space(
                sector=sector,
                x_col="emissions_intensity",
                y_col="g_out_network",
                z_col="green_capability_share",
                color_col="g_in_network",
                filename=f"sector_{safe_sector_name}_phase_E_gout_C_colored_gin.png",
            )

    def add_scaled_columns(
        self,
        columns: list[str],
        method: str = "signed_log1p",
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ) -> None:
        """
        Add scaled versions of selected columns to self.df.

        Methods:
        - signed_log1p: sign(x) * log(1 + abs(x))
        - winsorized: clip values to lower/upper quantiles
        - winsorized_signed_log1p: clip, then signed log transform
        """
        df = self.df.copy()

        for col in columns:
            if col not in df.columns:
                print(f"[WARNING] Column not found, skipping scaling: {col}")
                continue

            x = pd.to_numeric(df[col], errors="coerce")

            if method == "signed_log1p":
                scaled = np.sign(x) * np.log1p(np.abs(x))

            elif method == "winsorized":
                lower = x.quantile(lower_quantile)
                upper = x.quantile(upper_quantile)
                scaled = x.clip(lower=lower, upper=upper)

            elif method == "winsorized_signed_log1p":
                lower = x.quantile(lower_quantile)
                upper = x.quantile(upper_quantile)
                clipped = x.clip(lower=lower, upper=upper)
                scaled = np.sign(clipped) * np.log1p(np.abs(clipped))

            else:
                raise ValueError(f"Unknown scaling method: {method}")

            scaled_col = f"{col}_scaled"
            df[scaled_col] = scaled

            print(
                f"[INFO] Added scaled column: {scaled_col} "
                f"using method={method}"
            )

        self.df = df

    def run_scaled_phase_spaces(
        self,
        scaling_method: str = "winsorized_signed_log1p",
    ) -> None:
        self.add_scaled_columns(
            columns=[
                "emissions_intensity",
                "g_out_network",
                "g_in_network",
            ],
            method=scaling_method,
        )

        self.plot_3d_phase_space(
            x_col="emissions_intensity_scaled",
            y_col="g_out_network_scaled",
            z_col="green_capability_share",
            color_col="g_in_network_scaled",
            filename=f"scaled_phase_E_gout_C_colored_gin_{scaling_method}.png",
            min_active_good_count=1,
        )

        self.plot_3d_phase_space(
            x_col="emissions_intensity_scaled",
            y_col="g_in_network_scaled",
            z_col="green_capability_share",
            color_col="g_out_network_scaled",
            filename=f"scaled_phase_E_gin_C_colored_gout_{scaling_method}.png",
            min_active_good_count=1,
        )

        self.plot_3d_phase_space(
            x_col="emissions_intensity_scaled",
            y_col="g_out_network_scaled",
            z_col="g_in_network_scaled",
            color_col="green_capability_share",
            filename=f"scaled_phase_E_gout_gin_colored_C_{scaling_method}.png",
            min_active_good_count=1,
        )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["raw", "scaled", "all"],
        default="raw",
        help="Which phase-space plots to generate.",
    )

    parser.add_argument(
        "--scaling-method",
        choices=[
            "signed_log1p",
            "winsorized",
            "winsorized_signed_log1p",
        ],
        default="winsorized_signed_log1p",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    plotter = AdvancedPhaseSpacePlotter(
        input_path=INPUT_PATH,
        output_dir=OUTPUT_DIR,
    )

    if args.mode in ["raw", "all"]:
        plotter.run_default_phase_spaces()

    if args.mode in ["scaled", "all"]:
        plotter.run_scaled_phase_spaces(
            scaling_method=args.scaling_method,
        )


if __name__ == "__main__":
    main()