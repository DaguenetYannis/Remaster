from pathlib import Path

import numpy as np
import pandas as pd


TRANSITIONS_PATH = Path("data/abm/transitions_panel.parquet")
OUTPUT_DIR = Path("data/abm/diagnostics")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(TRANSITIONS_PATH)

    cols = [
        "emissions_intensity",
        "emissions_intensity_next",
        "delta_emissions_intensity",
        "g_base",
        "delta_g_base",
        "g_out_network",
        "delta_g_out_network",
        "g_in_network",
        "delta_g_in_network",
    ]

    existing_cols = [col for col in cols if col in df.columns]

    summary = (
        df[existing_cols]
        .replace([np.inf, -np.inf], np.nan)
        .describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
        .T
    )

    summary.to_csv(OUTPUT_DIR / "transition_variable_summary.csv")

    regime_summary = (
        df.groupby("regime_transition", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    regime_summary["share"] = regime_summary["count"] / regime_summary["count"].sum()
    regime_summary.to_csv(OUTPUT_DIR / "regime_transition_counts.csv", index=False)

    df_clean = df.copy()
    df_clean["log_emissions_intensity"] = np.log1p(df_clean["emissions_intensity"])
    df_clean["log_emissions_intensity_next"] = np.log1p(
        df_clean["emissions_intensity_next"]
    )
    df_clean["delta_log_emissions_intensity"] = (
        df_clean["log_emissions_intensity_next"]
        - df_clean["log_emissions_intensity"]
    )

    lower = df_clean["delta_log_emissions_intensity"].quantile(0.01)
    upper = df_clean["delta_log_emissions_intensity"].quantile(0.99)

    df_clean["delta_log_emissions_intensity_winsorized"] = (
        df_clean["delta_log_emissions_intensity"].clip(lower=lower, upper=upper)
    )

    df_clean.to_parquet(
        OUTPUT_DIR / "transitions_with_clean_targets.parquet",
        index=False,
    )

    print("\nSaved diagnostics to:", OUTPUT_DIR)
    print("\nTarget summary:")
    print(summary.loc[["delta_emissions_intensity"]])
    print("\nTop regime transitions:")
    print(regime_summary.head(20))


if __name__ == "__main__":
    main()