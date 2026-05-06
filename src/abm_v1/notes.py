import pandas as pd
from pathlib import Path

for file in Path("data/abm").glob("*.parquet"):
    df = pd.read_parquet(file)
    print(f"\n=== {file.name} ===")
    print(df.shape)
    print(df.columns.tolist())
    print(df.head())

for file in Path("data/abm/model_outputs").glob("*.csv"):
    print(f"\n=== {file.name} ===")
    print(pd.read_csv(file).head(20))

df = pd.read_parquet("data/abm/simulation_output.parquet")

print(df["regime"].value_counts())
print(df["emissions_intensity"].describe())

df = pd.read_parquet("data/abm/simulation_output_v2.parquet")

print("\nRegime counts by simulated year:")
print(pd.crosstab(df["sim_year"], df["regime"], normalize="index"))

print("\nRegime change share by simulated year:")
print(df.groupby("sim_year")["regime_changed"].mean())

print("\nEI summary:")
print(df.groupby("sim_year")["emissions_intensity"].describe())