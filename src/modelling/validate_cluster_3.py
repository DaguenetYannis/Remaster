from pathlib import Path

import pandas as pd


DYNAMIC_PANEL_PATH = Path("data/final/eora_atlas_dynamic_panel.parquet")
CLUSTERS_PATH = Path("data/final/country_sector_trajectory_clusters.parquet")
OUTPUT_PATH = Path("data/final/cluster_3_validation.csv")


def main() -> None:
    dynamic = pd.read_parquet(DYNAMIC_PANEL_PATH)
    clusters = pd.read_parquet(CLUSTERS_PATH)

    df = dynamic.merge(
        clusters[["Country", "Sector", "trajectory_cluster"]],
        on=["Country", "Sector"],
        how="inner",
        validate="many_to_one",
    )

    df = df[
        (df["Year"].between(1995, 2016))
        & (df["trajectory_cluster"] == 3)
    ].copy()

    if df.empty:
        raise ValueError("Cluster 3 is empty.")

    columns = [
        "green_capability_share",
        "green_active_good_count",
        "active_good_count",
        "green_active_good_export_value",
        "active_good_export_value",
        "green_capability_export_share",
        "capability_mean_pci",
        "capability_export_weighted_pci",
        "emissions_intensity",
        "g_in_network",
        "g_out_network",
    ]

    columns = [col for col in columns if col in df.columns]

    summary = (
        df
        .groupby("Year")[columns]
        .agg(["mean", "median", "min", "max"])
    )

    summary.columns = [
        f"{variable}_{stat}"
        for variable, stat in summary.columns
    ]

    summary = summary.reset_index()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)

    print("[INFO] Cluster 3 observations:", len(df))
    print("[INFO] Cluster 3 country-sectors:", df[["Country", "Sector"]].drop_duplicates().shape[0])
    print("[INFO] Saved validation file to:", OUTPUT_PATH)

    print("\nTop Cluster 3 sectors:")
    print(
        df[["Country", "Sector"]]
        .drop_duplicates()
        .groupby("Sector")
        .size()
        .sort_values(ascending=False)
        .head(15)
    )

    print("\nCluster 3 yearly median profile:")
    print(
        df
        .groupby("Year")[[
            col for col in [
                "green_capability_share",
                "green_active_good_count",
                "active_good_count",
                "emissions_intensity",
                "g_in_network",
                "g_out_network",
            ]
            if col in df.columns
        ]]
        .median()
    )


if __name__ == "__main__":
    main()