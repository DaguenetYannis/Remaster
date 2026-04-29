from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


INPUT_PATH = Path("data/final/eora_atlas_dynamic_panel.parquet")
OUTPUT_PATH = Path("data/final/country_sector_trajectory_clusters.parquet")
SUMMARY_OUTPUT_PATH = Path("data/final/trajectory_cluster_summary.csv")


STATE_COLS = [
    "emissions_intensity",
    "g_out_network",
    "g_in_network",
    "green_capability_share",
]

KEY_COLS = ["Country", "Sector"]


class TrajectoryClusterBuilder:
    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        summary_output_path: Path,
        year_start: int = 1995,
        year_end: int = 2016,
        n_clusters: int = 6,
    ) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.summary_output_path = summary_output_path
        self.year_start = year_start
        self.year_end = year_end
        self.n_clusters = n_clusters

    def load_panel(self) -> pd.DataFrame:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input panel not found: {self.input_path}")

        df = pd.read_parquet(self.input_path)

        required_cols = KEY_COLS + ["Year"] + STATE_COLS
        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return df

    def filter_panel(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[
            (df["Year"] >= self.year_start)
            & (df["Year"] <= self.year_end)
        ].copy()

        df = df.dropna(subset=STATE_COLS)

        if "active_good_count" in df.columns:
            df = df[df["active_good_count"].fillna(0) >= 1]

        return df

    def build_trajectory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []

        for keys, group in df.groupby(KEY_COLS):
            country, sector = keys
            group = group.sort_values("Year")

            if group["Year"].min() > self.year_start:
                continue

            if group["Year"].max() < self.year_end:
                continue

            start = group[group["Year"] == self.year_start].iloc[0]
            end = group[group["Year"] == self.year_end].iloc[0]

            row = {
                "Country": country,
                "Sector": sector,
                "year_start": self.year_start,
                "year_end": self.year_end,
                "n_years_observed": group["Year"].nunique(),
            }

            squared_steps = []

            for col in STATE_COLS:
                start_col = f"{col}_start"
                end_col = f"{col}_end"
                delta_col = f"{col}_delta"

                row[start_col] = start[col]
                row[end_col] = end[col]
                row[delta_col] = end[col] - start[col]

                step_change = group[col].diff()
                squared_steps.append(step_change.pow(2))

            step_length = np.sqrt(sum(squared_steps))
            row["trajectory_path_length"] = step_length.sum(skipna=True)

            net_squared = sum(
                row[f"{col}_delta"] ** 2 for col in STATE_COLS
            )
            row["trajectory_net_displacement"] = np.sqrt(net_squared)

            if row["trajectory_path_length"] > 0:
                row["trajectory_directness"] = (
                    row["trajectory_net_displacement"]
                    / row["trajectory_path_length"]
                )
            else:
                row["trajectory_directness"] = 0.0

            row["emissions_reduced"] = row["emissions_intensity_delta"] < 0
            row["network_out_improved"] = row["g_out_network_delta"] > 0
            row["network_in_improved"] = row["g_in_network_delta"] > 0
            row["green_capability_improved"] = row["green_capability_share_delta"] > 0

            rows.append(row)

        features = pd.DataFrame(rows)

        if features.empty:
            raise ValueError("No complete country-sector trajectories were built.")

        return features

    def cluster_trajectories(self, features: pd.DataFrame) -> pd.DataFrame:
        feature_cols = [
            "emissions_intensity_start",
            "g_out_network_start",
            "g_in_network_start",
            "green_capability_share_start",
            "emissions_intensity_delta",
            "g_out_network_delta",
            "g_in_network_delta",
            "green_capability_share_delta",
            "trajectory_path_length",
            "trajectory_directness",
        ]

        missing = [col for col in feature_cols if col not in features.columns]
        if missing:
            raise ValueError(f"Missing clustering feature columns: {missing}")

        X = features[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=20,
        )

        clustered = features.copy()
        clustered["trajectory_cluster"] = model.fit_predict(X_scaled)

        return clustered

    def summarize_clusters(self, clustered: pd.DataFrame) -> pd.DataFrame:
        summary = (
            clustered
            .groupby("trajectory_cluster")
            .agg(
                n=("Country", "size"),
                mean_emissions_start=("emissions_intensity_start", "mean"),
                mean_emissions_delta=("emissions_intensity_delta", "mean"),
                mean_g_out_start=("g_out_network_start", "mean"),
                mean_g_out_delta=("g_out_network_delta", "mean"),
                mean_g_in_start=("g_in_network_start", "mean"),
                mean_g_in_delta=("g_in_network_delta", "mean"),
                mean_green_capability_start=("green_capability_share_start", "mean"),
                mean_green_capability_delta=("green_capability_share_delta", "mean"),
                mean_path_length=("trajectory_path_length", "mean"),
                mean_directness=("trajectory_directness", "mean"),
                share_emissions_reduced=("emissions_reduced", "mean"),
                share_network_out_improved=("network_out_improved", "mean"),
                share_network_in_improved=("network_in_improved", "mean"),
                share_green_capability_improved=("green_capability_improved", "mean"),
            )
            .reset_index()
            .sort_values("trajectory_cluster")
        )

        return summary

    def save_outputs(
        self,
        clustered: pd.DataFrame,
        summary: pd.DataFrame,
    ) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        clustered.to_parquet(self.output_path, index=False)
        summary.to_csv(self.summary_output_path, index=False)

        print("[INFO] Saved trajectory clusters to:", self.output_path)
        print("[INFO] Saved cluster summary to:", self.summary_output_path)

    def build(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        print("[INFO] Loading dynamic panel")
        df = self.load_panel()

        print("[INFO] Filtering panel")
        df = self.filter_panel(df)

        print("[INFO] Building trajectory features")
        features = self.build_trajectory_features(df)
        print("[INFO] Trajectories:", len(features))

        print("[INFO] Clustering trajectories")
        clustered = self.cluster_trajectories(features)

        print("[INFO] Summarizing clusters")
        summary = self.summarize_clusters(clustered)

        print(summary.to_string(index=False))

        self.save_outputs(clustered, summary)

        return clustered, summary


def main() -> None:
    builder = TrajectoryClusterBuilder(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        summary_output_path=SUMMARY_OUTPUT_PATH,
        year_start=1995,
        year_end=2016,
        n_clusters=6,
    )

    builder.build()


if __name__ == "__main__":
    main()