import pandas as pd

year = 2000

et = pd.read_parquet(f"data/metrics/{year}/et_{year}.parquet")
metrics = pd.read_parquet(f"data/metrics/{year}/metrics_{year}.parquet")

print("ET shape:", et.shape)
print("Metrics length:", len(metrics))

print("\nFirst ET index:")
print(et.index[:5])

print("\nFirst metrics country_sector:")
print(metrics["country_sector"].head())