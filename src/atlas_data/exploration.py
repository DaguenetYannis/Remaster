from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


###
# CONFIG
###

METRICS_DIR = Path("data/metrics")
ATLAS_CONCORDANCE_DIR = Path("data/atlas/concordance")
OUTPUT_DIR = Path("data/atlas/processed")

DEFAULT_PREFILLED_CONCORDANCE_PATH = (
    ATLAS_CONCORDANCE_DIR / "hs92_to_eora26_prefilled.csv"
)


###
# HELPERS
###

def print_section(title: str) -> None:
    logging.info("=" * 80)
    logging.info(title)
    logging.info("=" * 80)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_any_year_ei(metrics_dir: Path, preferred_year: int = 1995) -> pd.Series:
    """
    Load one EI file from data/metrics/{year}/ei_{year}.parquet.
    """
    preferred_path = metrics_dir / str(preferred_year) / f"ei_{preferred_year}.parquet"

    if preferred_path.exists():
        df = pd.read_parquet(preferred_path)
    else:
        files = sorted(metrics_dir.glob("*/ei_*.parquet"))

        if not files:
            raise FileNotFoundError("No EI files found in data/metrics/{year}/")

        df = pd.read_parquet(files[0])

    if isinstance(df, pd.Series):
        return df

    if "emissions_intensity" in df.columns:
        return df["emissions_intensity"]

    return df.iloc[:, 0]


def split_country_sector(index: pd.Index) -> pd.DataFrame:
    """
    Expected format:
    'AFG | AFG | Industries | Agriculture'
    """
    df = pd.Series(index, name="country_sector").to_frame()

    parts = df["country_sector"].str.split("|", expand=True)
    parts = parts.apply(lambda col: col.str.strip())

    if parts.shape[1] != 4:
        raise ValueError(
            "Unexpected Eora label structure. Expected 4 fields separated by '|'."
        )

    parts.columns = [
        "country_code",
        "country_detail",
        "category",
        "sector",
    ]

    return parts


def normalize_text_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    return df


###
# EORA SECTOR EXPLORATION
###

def export_eora_sector_list(
    metrics_dir: Path,
    output_dir: Path,
    preferred_year: int = 1995,
) -> pd.DataFrame:
    print_section("Loading Eora sector labels")

    ei = load_any_year_ei(
        metrics_dir=metrics_dir,
        preferred_year=preferred_year,
    )

    labels_df = split_country_sector(ei.index)

    print_section("Basic Eora structure")

    logging.info("Total country-sector rows: %s", len(labels_df))
    logging.info("Unique countries: %s", labels_df["country_code"].nunique())
    logging.info("Unique categories: %s", labels_df["category"].nunique())
    logging.info("Unique sectors: %s", labels_df["sector"].nunique())

    print_section("Unique Eora sector list")

    sectors = sorted(labels_df["sector"].unique())

    for sector in sectors:
        print(sector)

    print_section("Saving Eora sector list")

    ensure_directory(output_dir)

    sector_output_path = output_dir / "eora26_sector_list.csv"
    labels_output_path = output_dir / "eora26_country_sector_labels.csv"

    pd.DataFrame({"sector": sectors}).to_csv(sector_output_path, index=False)
    labels_df.to_csv(labels_output_path, index=False)

    logging.info("Saved sector list -> %s", sector_output_path)
    logging.info("Saved country-sector labels -> %s", labels_output_path)

    return labels_df


###
# CONCORDANCE VALIDATION
###

def load_eora_sector_vocabulary(
    metrics_dir: Path,
    processed_dir: Path,
    preferred_year: int = 1995,
) -> set[str]:
    sector_list_path = processed_dir / "eora26_sector_list.csv"

    if sector_list_path.exists():
        sectors = pd.read_csv(sector_list_path)["sector"].dropna().astype(str)
        return set(sectors.str.strip())

    labels_df = export_eora_sector_list(
        metrics_dir=metrics_dir,
        output_dir=processed_dir,
        preferred_year=preferred_year,
    )

    return set(labels_df["sector"].dropna().astype(str).str.strip())


def validate_concordance(
    concordance_path: Path,
    metrics_dir: Path,
    processed_dir: Path,
    preferred_year: int = 1995,
) -> pd.DataFrame:
    if not concordance_path.exists():
        raise FileNotFoundError(f"Concordance file not found: {concordance_path}")

    print_section("Loading concordance")

    df = pd.read_csv(concordance_path, dtype=str).fillna("")
    df = normalize_text_columns(
        df,
        columns=[
            "productIdRaw",
            "code",
            "nameEn",
            "eora26_sector",
            "mapping_status",
            "mapping_method",
            "notes",
        ],
    )

    eora_sectors = load_eora_sector_vocabulary(
        metrics_dir=metrics_dir,
        processed_dir=processed_dir,
        preferred_year=preferred_year,
    )

    print_section("Concordance structure")

    logging.info("Rows: %s", len(df))
    logging.info("Unique products: %s", df["productIdRaw"].nunique())
    logging.info("Unique codes: %s", df["code"].nunique())

    if "is_hs4_code" in df.columns:
        logging.info("HS4 rows: %s", (df["is_hs4_code"].astype(str).str.lower() == "true").sum())

    if "is_service_like" in df.columns:
        logging.info(
            "Service-like rows: %s",
            (df["is_service_like"].astype(str).str.lower() == "true").sum(),
        )

    print_section("Mapping status counts")

    print(df["mapping_status"].value_counts(dropna=False))

    print_section("Mapping method counts")

    print(df["mapping_method"].value_counts(dropna=False))

    print_section("Mapped sector distribution")

    mapped = df[df["eora26_sector"] != ""].copy()

    print(
        mapped["eora26_sector"]
        .value_counts(dropna=False)
        .sort_index()
    )

    print_section("Unmapped rows")

    unmapped = df[df["eora26_sector"] == ""].copy()

    logging.info("Unmapped rows: %s", len(unmapped))

    if len(unmapped) > 0:
        print(
            unmapped[
                [
                    "productIdRaw",
                    "code",
                    "nameEn",
                    "mapping_status",
                    "mapping_method",
                    "notes",
                ]
            ].head(50).to_string(index=False)
        )

    print_section("Invalid Eora sector labels")

    invalid = df[
        (df["eora26_sector"] != "")
        & (~df["eora26_sector"].isin(eora_sectors))
    ].copy()

    logging.info("Invalid sector labels: %s", len(invalid))

    if len(invalid) > 0:
        print(
            invalid[
                [
                    "productIdRaw",
                    "code",
                    "nameEn",
                    "eora26_sector",
                    "mapping_status",
                    "mapping_method",
                ]
            ].to_string(index=False)
        )

    print_section("Potential duplicate product mappings")

    duplicate_products = df[
        df.duplicated(subset=["productIdRaw"], keep=False)
    ].copy()

    logging.info("Duplicate productIdRaw rows: %s", len(duplicate_products))

    if len(duplicate_products) > 0:
        print(
            duplicate_products[
                [
                    "productIdRaw",
                    "code",
                    "nameEn",
                    "eora26_sector",
                    "mapping_status",
                ]
            ].to_string(index=False)
        )

    print_section("Validation summary")

    summary = {
        "rows": len(df),
        "mapped_rows": len(mapped),
        "unmapped_rows": len(unmapped),
        "invalid_sector_rows": len(invalid),
        "duplicate_product_rows": len(duplicate_products),
        "mapped_share": len(mapped) / len(df) if len(df) else 0,
    }

    summary_df = pd.DataFrame([summary])

    print(summary_df.to_string(index=False))

    ensure_directory(processed_dir)

    summary_output_path = processed_dir / "concordance_validation_summary.csv"
    invalid_output_path = processed_dir / "concordance_invalid_sector_rows.csv"
    unmapped_output_path = processed_dir / "concordance_unmapped_rows.csv"

    summary_df.to_csv(summary_output_path, index=False)
    invalid.to_csv(invalid_output_path, index=False)
    unmapped.to_csv(unmapped_output_path, index=False)

    logging.info("Saved validation summary -> %s", summary_output_path)
    logging.info("Saved invalid sector rows -> %s", invalid_output_path)
    logging.info("Saved unmapped rows -> %s", unmapped_output_path)

    return df


###
# CLI
###

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=[
            "eora-sectors",
            "validate-concordance",
            "all",
        ],
        help="Exploration task to run.",
    )

    parser.add_argument(
        "--metrics-dir",
        default=str(METRICS_DIR),
    )

    parser.add_argument(
        "--processed-dir",
        default=str(OUTPUT_DIR),
    )

    parser.add_argument(
        "--preferred-year",
        type=int,
        default=1995,
    )

    parser.add_argument(
        "--concordance-path",
        default=str(DEFAULT_PREFILLED_CONCORDANCE_PATH),
    )

    return parser.parse_args()


###
# MAIN
###

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    args = parse_args()

    metrics_dir = Path(args.metrics_dir)
    processed_dir = Path(args.processed_dir)
    concordance_path = Path(args.concordance_path)

    if args.command in ["eora-sectors", "all"]:
        export_eora_sector_list(
            metrics_dir=metrics_dir,
            output_dir=processed_dir,
            preferred_year=args.preferred_year,
        )

    if args.command in ["validate-concordance", "all"]:
        validate_concordance(
            concordance_path=concordance_path,
            metrics_dir=metrics_dir,
            processed_dir=processed_dir,
            preferred_year=args.preferred_year,
        )


if __name__ == "__main__":
    main()