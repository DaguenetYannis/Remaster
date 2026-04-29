from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_PATH = Path("data/final/eora_atlas_dynamic_panel.parquet")
OUTPUT_DIR = Path("outputs/plots/dynamics")


class DynamicPlotBuilder:
    def __init__(self, input_path: Path, output_dir: Path) -> None:
        self.input_path = input_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.df = self.load_data()

    def load_data(self) -> pd.DataFrame:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Dynamic panel not found: {self.input_path}")

        df = pd.read_parquet(self.input_path)

        required_cols = ["Country", "Sector", "Year"]
        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return df

    def save_current_plot(self, filename: str) -> None:
        path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"[INFO] Saved plot: {path}")

    def plot_global_mean_over_time(
        self,
        column: str,
        filename: str | None = None,
    ) -> None:
        if column not in self.df.columns:
            raise ValueError(f"Column not found: {column}")

        plot_df = (
            self.df
            .groupby("Year", as_index=False)[column]
            .mean()
            .sort_values("Year")
        )

        plt.figure(figsize=(9, 5))
        plt.plot(plot_df["Year"], plot_df[column], marker="o")
        plt.xlabel("Year")
        plt.ylabel(column)
        plt.title(f"Global mean of {column} over time")

        self.save_current_plot(filename or f"global_mean_{column}.png")

    def plot_sector_mean_over_time(
        self,
        column: str,
        sectors: list[str] | None = None,
        filename: str | None = None,
    ) -> None:
        if column not in self.df.columns:
            raise ValueError(f"Column not found: {column}")

        df = self.df.copy()

        if sectors is not None:
            df = df[df["Sector"].isin(sectors)]

        plot_df = (
            df
            .groupby(["Year", "Sector"], as_index=False)[column]
            .mean()
            .sort_values(["Sector", "Year"])
        )

        plt.figure(figsize=(10, 6))

        for sector, group in plot_df.groupby("Sector"):
            plt.plot(group["Year"], group[column], marker="o", label=sector)

        plt.xlabel("Year")
        plt.ylabel(column)
        plt.title(f"Sector mean of {column} over time")
        plt.legend(fontsize=8, loc="best")

        self.save_current_plot(filename or f"sector_mean_{column}.png")

    def plot_scatter_green_vs_emissions_change(
        self,
        year: int,
        filename: str | None = None,
    ) -> None:
        required_cols = [
            "Year",
            "green_capability_share_change",
            "emissions_intensity_change",
        ]

        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        plot_df = self.df[self.df["Year"] == year].copy()
        plot_df = plot_df.dropna(
            subset=[
                "green_capability_share_change",
                "emissions_intensity_change",
            ]
        )

        plt.figure(figsize=(8, 6))
        plt.scatter(
            plot_df["green_capability_share_change"],
            plot_df["emissions_intensity_change"],
            alpha=0.5,
        )

        plt.axhline(0, linewidth=1)
        plt.axvline(0, linewidth=1)

        plt.xlabel("Change in green capability share")
        plt.ylabel("Change in emissions intensity")
        plt.title(f"Green capability change vs emissions intensity change, {year}")

        self.save_current_plot(
            filename or f"scatter_green_vs_emissions_change_{year}.png"
        )

    def plot_transition_counts(self, filename: str | None = None) -> None:
        required_cols = [
            "Year",
            "gained_green_capability",
            "lost_green_capability",
        ]

        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        plot_df = (
            self.df
            .groupby("Year", as_index=False)
            .agg(
                gained_green_capability=("gained_green_capability", "sum"),
                lost_green_capability=("lost_green_capability", "sum"),
            )
            .sort_values("Year")
        )

        plt.figure(figsize=(9, 5))
        plt.plot(
            plot_df["Year"],
            plot_df["gained_green_capability"],
            marker="o",
            label="Gained green capability",
        )
        plt.plot(
            plot_df["Year"],
            plot_df["lost_green_capability"],
            marker="o",
            label="Lost green capability",
        )

        plt.xlabel("Year")
        plt.ylabel("Number of country-sector transitions")
        plt.title("Green capability transitions over time")
        plt.legend()

        self.save_current_plot(filename or "green_capability_transitions.png")

    def run_default_plots(self) -> None:
        self.plot_global_mean_over_time("emissions_intensity")
        self.plot_global_mean_over_time("g_base")
        self.plot_global_mean_over_time("g_out_network")
        self.plot_global_mean_over_time("green_capability_share")

        self.plot_sector_mean_over_time(
            column="green_capability_share",
            sectors=[
                "Agriculture",
                "Electrical and Machinery",
                "Transport Equipment",
                "Petroleum, Chemical and Non-Metallic Mineral Products",
                "Metal Products",
            ],
        )

        self.plot_transition_counts()

        for year in [2000, 2005, 2010, 2015]:
            self.plot_scatter_green_vs_emissions_change(year)


def main() -> None:
    plotter = DynamicPlotBuilder(
        input_path=INPUT_PATH,
        output_dir=OUTPUT_DIR,
    )

    plotter.run_default_plots()


if __name__ == "__main__":
    main()