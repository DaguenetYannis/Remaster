from pathlib import Path
import pandas as pd


EORA_PATH = Path("data/eora/processed/eora_metrics.parquet")
ATLAS_PATH = Path("data/atlas/processed/atlas_eora26_sector_capabilities_1995_2016.parquet")
OUTPUT_PATH = Path("data/final/eora_atlas_merged.parquet")


def main():
    print("[INFO] Loading datasets")

    eora = pd.read_parquet(EORA_PATH)
    atlas = pd.read_parquet(ATLAS_PATH)

    print("[INFO] Renaming Atlas keys")
    atlas = atlas.rename(columns={
        "iso3Code": "Country",
        "year": "Year",
        "eora26_sector": "Sector",
    })

    print("[INFO] Merging (Eora LEFT join Atlas)")
    merged = eora.merge(
        atlas,
        on=["Country", "Year", "Sector"],
        how="left",
        validate="one_to_one",
    )

    print("[INFO] Merge complete")
    print("Rows:", len(merged))
    print("Missing Atlas share:",
          merged["active_good_count"].isna().mean())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT_PATH, index=False)

    print("[INFO] Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()