from pathlib import Path
import pandas as pd
import numpy as np


INPUT_PATH = Path("data/final/eora_atlas_merged.parquet")
OUTPUT_PATH = Path("data/final/eora_atlas_dynamic_panel.parquet")

KEYS = ["Country", "Sector"]
TIME_COL = "Year"


def add_lag_and_change(
    df: pd.DataFrame,
    columns: list[str],
    group_cols: list[str],
    time_col: str,
) -> pd.DataFrame:
    df = df.sort_values(group_cols + [time_col]).copy()

    for col in columns:
        if col not in df.columns:
            print(f"[WARNING] Column not found, skipping: {col}")
            continue

        lag_col = f"{col}_lag"
        change_col = f"{col}_change"
        pct_change_col = f"{col}_pct_change"

        df[lag_col] = df.groupby(group_cols)[col].shift(1)
        df[change_col] = df[col] - df[lag_col]

        df[pct_change_col] = np.where(
            df[lag_col].notna() & (df[lag_col] != 0),
            df[change_col] / df[lag_col],
            np.nan,
        )

    return df


def add_green_capability_transitions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(KEYS + [TIME_COL]).copy()

    df["has_green_capability"] = df["green_active_good_count"].fillna(0) > 0

    df["has_green_capability_lag"] = (
        df.groupby(KEYS)["has_green_capability"]
        .shift(1)
    )

    df["gained_green_capability"] = (
        (df["has_green_capability"] == True)
        & (df["has_green_capability_lag"] == False)
    )

    df["lost_green_capability"] = (
        (df["has_green_capability"] == False)
        & (df["has_green_capability_lag"] == True)
    )

    return df


def main() -> None:
    print("[INFO] Loading merged panel")
    df = pd.read_parquet(INPUT_PATH)

    dynamic_cols = [
        "emissions_intensity",
        "g_base",
        "g_out_network",
        "g_in_network",
        "pagerank",
        "in_strength",
        "out_strength",
        "eigenvector_centrality",
        "reverse_eigenvector_centrality",
        "green_capability_share",
        "green_capability_export_share",
        "capability_mean_pci",
        "capability_export_weighted_pci",
    ]

    print("[INFO] Adding lags and changes")
    df = add_lag_and_change(
        df=df,
        columns=dynamic_cols,
        group_cols=KEYS,
        time_col=TIME_COL,
    )

    print("[INFO] Adding green capability transitions")
    df = add_green_capability_transitions(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print("[INFO] Saved dynamic panel to:", OUTPUT_PATH)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))


if __name__ == "__main__":
    main()