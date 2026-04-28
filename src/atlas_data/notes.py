from pathlib import Path
import pandas as pd


def find_file(start_path: Path, filename: str) -> Path:
    matches = list(start_path.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"{filename} not found under {start_path}")
    if len(matches) > 1:
        print(f"[WARNING] Multiple matches for {filename}, using first one:")
        for m in matches:
            print(" -", m)
    return matches[0]


def detect_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    print(f"\n[INFO] Columns in {label}:")
    print(list(df.columns))

    for candidate in candidates:
        if candidate in df.columns:
            print(f"[INFO] Using {label} column: {candidate}")
            return candidate

    raise KeyError(
        f"Could not detect {label} column. "
        f"Tried {candidates}. Available columns: {list(df.columns)}"
    )


def load_data(base_dir: Path):
    atlas_file = find_file(base_dir, "atlas_eora26_sector_capabilities_1995_2016.parquet")
    eora_file = find_file(base_dir, "eora26_sector_list.csv")

    print(f"[INFO] Atlas file found at: {atlas_file}")
    print(f"[INFO] Eora file found at: {eora_file}")

    df_atlas = pd.read_parquet(atlas_file)
    df_eora = pd.read_csv(eora_file)

    return df_atlas, df_eora


def check_sector_alignment(df_atlas: pd.DataFrame, df_eora: pd.DataFrame):
    print("\n=== SECTOR ALIGNMENT CHECK ===")

    atlas_sector_col = detect_column(
        df_atlas,
        candidates=["eora26_sector", "Sector", "sector"],
        label="Atlas",
    )

    eora_sector_col = detect_column(
        df_eora,
        candidates=["Sector", "sector", "eora26_sector", "Eora26 sector", "eora_sector"],
        label="Eora",
    )

    atlas_sectors = set(df_atlas[atlas_sector_col].dropna().astype(str).str.strip().unique())
    eora_sectors = set(df_eora[eora_sector_col].dropna().astype(str).str.strip().unique())

    only_in_atlas = sorted(atlas_sectors - eora_sectors)
    only_in_eora = sorted(eora_sectors - atlas_sectors)

    print("\nSectors in Atlas but NOT in Eora:")
    if only_in_atlas:
        for s in only_in_atlas:
            print(" -", s)
    else:
        print(" None")

    print("\nSectors in Eora but NOT in Atlas:")
    if only_in_eora:
        for s in only_in_eora:
            print(" -", s)
    else:
        print(" None")

    print("\nCounts:")
    print("Atlas sectors:", len(atlas_sectors))
    print("Eora sectors:", len(eora_sectors))
    print("Overlap:", len(atlas_sectors & eora_sectors))

    if not only_in_atlas and not only_in_eora:
        print("\n[OK] Perfect sector alignment.")
    else:
        print("\n[WARNING] Sector mismatch detected.")


if __name__ == "__main__":
    BASE_DIR = Path("data")

    df_atlas, df_eora = load_data(BASE_DIR)
    check_sector_alignment(df_atlas, df_eora)