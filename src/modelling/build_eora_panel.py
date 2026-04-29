import pandas as pd

def build_eora_panel(metrics_dir):
    frames = []

    for year_dir in metrics_dir.iterdir():
        year = int(year_dir.name)

        ei = pd.read_parquet(year_dir / f"ei_{year}.parquet")
        centrality = pd.read_parquet(year_dir / f"centrality_{year}.parquet")
        greenness = pd.read_parquet(year_dir / f"greenness_{year}.parquet")
        efficiency = pd.read_parquet(year_dir / f"efficiency_{year}.parquet")

        df = pd.concat([ei, centrality, greenness, efficiency], axis=1)

        df["Year"] = year

        # split index
        parts = df.index.str.split("|", expand=True)
        df["Country"] = parts[0].str.strip()
        df["Sector"] = parts[3].str.strip()

        frames.append(df.reset_index(drop=True))

    return pd.concat(frames, ignore_index=True)