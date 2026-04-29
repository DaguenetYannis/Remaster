from pathlib import Path
import pandas as pd


METRICS_DIR = Path("data/metrics")
ATLAS_PATH = Path("data/atlas/processed/atlas_eora26_sector_capabilities_1995_2016.parquet")

EORA_PANEL_OUTPUT_PATH = Path("data/final/eora_metrics_panel.parquet")
MERGED_OUTPUT_PATH = Path("data/final/eora_atlas_merged.parquet")


def split_country_sector_index(index: pd.Index) -> pd.DataFrame:
    labels = pd.Series(index, name="country_sector").astype(str)
    parts = labels.str.split("|", expand=True)
    parts = parts.apply(lambda col: col.str.strip())

    if parts.shape[1] != 4:
        raise ValueError(
            "Unexpected Eora label format. Expected: "
            "'COUNTRY | COUNTRY_DETAIL | CATEGORY | SECTOR'"
        )

    parts.columns = ["Country", "Country_detail", "Category", "Sector"]
    return parts


def load_metric_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing metric file: {path}")

    df = pd.read_parquet(path)

    if isinstance(df, pd.Series):
        df = df.to_frame()

    return df


def build_year_panel(year: int, year_dir: Path) -> pd.DataFrame:
    print(f"[INFO] Building Eora panel for {year}")

    ei = load_metric_file(year_dir / f"ei_{year}.parquet")
    greenness = load_metric_file(year_dir / f"greenness_{year}.parquet")
    centrality = load_metric_file(year_dir / f"centrality_{year}.parquet")
    efficiency = load_metric_file(year_dir / f"efficiency_{year}.parquet")

    df = pd.concat(
        [
            ei,
            greenness,
            centrality,
            efficiency,
        ],
        axis=1,
    )

    # Remove duplicate columns caused by efficiency already containing centrality columns
    df = df.loc[:, ~df.columns.duplicated()]

    labels = split_country_sector_index(df.index)

    df = df.reset_index(drop=True)
    df = pd.concat([labels.reset_index(drop=True), df], axis=1)

    df["Year"] = year

    return df


def build_eora_panel(metrics_dir: Path) -> pd.DataFrame:
    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics directory not found: {metrics_dir}")

    year_dirs = [
        path for path in metrics_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]

    if not year_dirs:
        raise FileNotFoundError(f"No yearly metric folders found under {metrics_dir}")

    frames = []

    for year_dir in sorted(year_dirs, key=lambda p: int(p.name)):
        year = int(year_dir.name)
        frames.append(build_year_panel(year, year_dir))

    panel = pd.concat(frames, ignore_index=True)

    print("[INFO] Built Eora panel")
    print("Rows:", len(panel))
    print("Years:", panel["Year"].min(), "-", panel["Year"].max())
    print("Countries:", panel["Country"].nunique())
    print("Sectors:", panel["Sector"].nunique())

    return panel


def load_atlas_panel(atlas_path: Path) -> pd.DataFrame:
    if not atlas_path.exists():
        raise FileNotFoundError(f"Atlas panel not found: {atlas_path}")

    atlas = pd.read_parquet(atlas_path)

    atlas = atlas.rename(
        columns={
            "iso3Code": "Country",
            "year": "Year",
            "eora26_sector": "Sector",
        }
    )

    atlas["Country"] = atlas["Country"].astype(str).str.strip()
    atlas["Sector"] = atlas["Sector"].astype(str).str.strip()
    atlas["Year"] = atlas["Year"].astype(int)

    return atlas


def merge_eora_atlas(eora: pd.DataFrame, atlas: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Merging Eora LEFT join Atlas")

    eora = eora.copy()
    eora["Country"] = eora["Country"].astype(str).str.strip()
    eora["Sector"] = eora["Sector"].astype(str).str.strip()
    eora["Year"] = eora["Year"].astype(int)

    duplicate_eora = eora.duplicated(["Country", "Year", "Sector"]).sum()
    duplicate_atlas = atlas.duplicated(["Country", "Year", "Sector"]).sum()

    print("[INFO] Duplicate Eora keys:", duplicate_eora)
    print("[INFO] Duplicate Atlas keys:", duplicate_atlas)

    if duplicate_eora > 0:
        raise ValueError("Duplicate Eora country-year-sector keys detected.")

    if duplicate_atlas > 0:
        raise ValueError("Duplicate Atlas country-year-sector keys detected.")

    merged = eora.merge(
        atlas,
        on=["Country", "Year", "Sector"],
        how="left",
        validate="one_to_one",
    )

    print("[INFO] Merge complete")
    print("Rows:", len(merged))

    if "active_good_count" in merged.columns:
        print(
            "Missing Atlas share:",
            merged["active_good_count"].isna().mean(),
        )

    return merged


def main() -> None:
    print("[INFO] Building Eora metrics panel")
    eora = build_eora_panel(METRICS_DIR)

    EORA_PANEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    eora.to_parquet(EORA_PANEL_OUTPUT_PATH, index=False)
    print("[INFO] Saved Eora panel to:", EORA_PANEL_OUTPUT_PATH)

    print("[INFO] Loading Atlas panel")
    atlas = load_atlas_panel(ATLAS_PATH)

    merged = merge_eora_atlas(eora, atlas)

    MERGED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(MERGED_OUTPUT_PATH, index=False)

    print("[INFO] Saved merged panel to:", MERGED_OUTPUT_PATH)


if __name__ == "__main__":
    main()