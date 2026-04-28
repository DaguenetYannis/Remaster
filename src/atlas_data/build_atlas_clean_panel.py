from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


###
# CONFIG
###

DEFAULT_RAW_DIR = Path("data/atlas/raw")
DEFAULT_PROCESSED_DIR = Path("data/atlas/processed")
DEFAULT_CONCORDANCE_DIR = Path("data/atlas/concordance")

PRODUCT_METADATA_FILENAME = "product_hs92_level4.parquet"


###
# HELPERS
###

def print_section(title: str) -> None:
    logging.info("=" * 80)
    logging.info(title)
    logging.info("=" * 80)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_hs4_code(value: str) -> bool:
    return str(value).isdigit() and len(str(value)) == 4


###
# BUILDERS
###

class AtlasCleanPanelBuilder:
    def __init__(
        self,
        raw_dir: Path,
        processed_dir: Path,
        concordance_dir: Path,
        year_start: int,
        year_end: int,
    ) -> None:
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.concordance_dir = concordance_dir
        self.year_start = year_start
        self.year_end = year_end

        self.metadata_dir = self.raw_dir / "metadata"
        self.country_product_year_dir = self.raw_dir / "country_product_year"

        ensure_directory(self.processed_dir)
        ensure_directory(self.concordance_dir)

    def build_all(self, force: bool = False) -> None:
        product_metadata = self.load_product_metadata()

        self.build_clean_panel(
            product_metadata=product_metadata,
            force=force,
        )

        self.build_concordance_skeleton(
            product_metadata=product_metadata,
            force=force,
        )

    def load_product_metadata(self) -> pd.DataFrame:
        path = self.metadata_dir / PRODUCT_METADATA_FILENAME

        if not path.exists():
            raise FileNotFoundError(f"Product metadata not found: {path}")

        df = pd.read_parquet(path)

        required_columns = [
            "productIdRaw",
            "productIdNumeric",
            "code",
            "nameEn",
            "greenProduct",
            "naturalResource",
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(
                f"Product metadata is missing required columns: {missing_columns}"
            )

        df = df.copy()
        df["is_hs4_code"] = df["code"].map(is_hs4_code)
        df["is_service_like"] = ~df["is_hs4_code"]

        return df

    def build_clean_panel(
        self,
        product_metadata: pd.DataFrame,
        force: bool = False,
    ) -> pd.DataFrame:
        output_path = (
            self.processed_dir
            / f"atlas_hs92_level4_clean_panel_{self.year_start}_{self.year_end}.parquet"
        )

        if output_path.exists() and not force:
            logging.info("Skipping clean panel; file already exists: %s", output_path)
            return pd.read_parquet(output_path)

        print_section("Building Atlas clean panel")

        frames: list[pd.DataFrame] = []

        metadata_cols = [
            "productIdRaw",
            "code",
            "nameEn",
            "greenProduct",
            "naturalResource",
            "is_hs4_code",
            "is_service_like",
        ]

        for year in range(self.year_start, self.year_end + 1):
            path = (
                self.country_product_year_dir
                / f"country_product_year_hs92_level4_{year}.parquet"
            )

            if not path.exists():
                logging.warning("Missing year file, skipping: %s", path)
                continue

            logging.info("Loading country-product-year file for %s", year)

            df_year = pd.read_parquet(path)

            df_year = df_year.merge(
                product_metadata[metadata_cols],
                on="productIdRaw",
                how="left",
                validate="many_to_one",
            )

            missing_metadata_share = df_year["code"].isna().mean()

            if missing_metadata_share > 0:
                logging.warning(
                    "Year %s has %.4f missing product metadata share",
                    year,
                    missing_metadata_share,
                )

            frames.append(df_year)

        if not frames:
            raise ValueError("No Atlas country-product-year files were loaded.")

        panel = pd.concat(frames, ignore_index=True)

        panel = self.clean_panel_types(panel)
        panel = self.add_capability_indicators(panel)

        panel = panel.sort_values(
            ["iso3Code", "year", "productIdNumeric"]
        ).reset_index(drop=True)

        panel.to_parquet(output_path, index=False)

        logging.info("Saved clean panel: %s rows -> %s", len(panel), output_path)

        return panel

    def clean_panel_types(self, panel: pd.DataFrame) -> pd.DataFrame:
        panel = panel.copy()

        numeric_cols = [
            "exportValue",
            "importValue",
            "exportRca",
            "globalMarketShare",
            "distance",
            "cog",
            "normalizedPci",
        ]

        for col in numeric_cols:
            if col in panel.columns:
                panel[col] = pd.to_numeric(panel[col], errors="coerce")

        panel["year"] = pd.to_numeric(panel["year"], errors="raise").astype(int)
        panel["countryId"] = pd.to_numeric(panel["countryId"], errors="raise").astype(int)
        panel["productIdNumeric"] = pd.to_numeric(
            panel["productIdNumeric"],
            errors="raise",
        ).astype(int)

        return panel

    def add_capability_indicators(self, panel: pd.DataFrame) -> pd.DataFrame:
        panel = panel.copy()

        panel["has_rca"] = panel["exportRca"] > 1
        panel["is_good_capability"] = panel["has_rca"] & panel["is_hs4_code"]
        panel["is_green_good_capability"] = (
            panel["is_good_capability"] & panel["greenProduct"]
        )

        return panel

    def build_concordance_skeleton(
        self,
        product_metadata: pd.DataFrame,
        force: bool = False,
    ) -> pd.DataFrame:
        output_path = self.concordance_dir / "hs92_to_eora26_manual.csv"

        if output_path.exists() and not force:
            logging.info(
                "Skipping concordance skeleton; file already exists: %s",
                output_path,
            )
            return pd.read_csv(output_path)

        print_section("Building HS92 to Eora26 concordance skeleton")

        df = product_metadata.copy()

        df = df[
            [
                "productIdRaw",
                "productIdNumeric",
                "code",
                "nameEn",
                "greenProduct",
                "naturalResource",
                "is_hs4_code",
                "is_service_like",
            ]
        ].copy()

        df["eora26_sector"] = ""
        df["eora26_sector_code"] = ""
        df["mapping_status"] = "unmapped"
        df["mapping_method"] = ""
        df["notes"] = ""

        df = df.sort_values(
            ["is_service_like", "code"],
            ascending=[False, True],
        ).reset_index(drop=True)

        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        logging.info("Saved concordance skeleton: %s rows -> %s", len(df), output_path)

        return df

    def summarize_clean_panel(self) -> None:
        output_path = (
            self.processed_dir
            / f"atlas_hs92_level4_clean_panel_{self.year_start}_{self.year_end}.parquet"
        )

        if not output_path.exists():
            logging.warning("Clean panel not found, cannot summarize: %s", output_path)
            return

        panel = pd.read_parquet(output_path)

        print_section("Clean panel summary")

        logging.info("Rows: %s", len(panel))
        logging.info("Years: %s - %s", panel["year"].min(), panel["year"].max())
        logging.info("Countries: %s", panel["iso3Code"].nunique())
        logging.info("Products: %s", panel["productIdRaw"].nunique())
        logging.info("HS4 products: %s", panel.loc[panel["is_hs4_code"], "productIdRaw"].nunique())
        logging.info(
            "Service-like products: %s",
            panel.loc[panel["is_service_like"], "productIdRaw"].nunique(),
        )

        logging.info("Missing RCA share: %.4f", panel["exportRca"].isna().mean())
        logging.info("Missing normalized PCI share: %.4f", panel["normalizedPci"].isna().mean())
        logging.info("RCA > 1 rows: %s", int(panel["has_rca"].sum()))
        logging.info("Good capability rows: %s", int(panel["is_good_capability"].sum()))
        logging.info(
            "Green good capability rows: %s",
            int(panel["is_green_good_capability"].sum()),
        )


###
# CLI
###

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-dir",
        default=str(DEFAULT_RAW_DIR),
        help="Directory containing raw Atlas parquet files.",
    )

    parser.add_argument(
        "--processed-dir",
        default=str(DEFAULT_PROCESSED_DIR),
        help="Directory where processed Atlas files will be saved.",
    )

    parser.add_argument(
        "--concordance-dir",
        default=str(DEFAULT_CONCORDANCE_DIR),
        help="Directory where concordance skeleton files will be saved.",
    )

    parser.add_argument(
        "--year-start",
        type=int,
        default=1995,
        help="Start year for clean Atlas panel.",
    )

    parser.add_argument(
        "--year-end",
        type=int,
        default=2016,
        help="Inclusive end year for clean Atlas panel.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing processed files.",
    )

    return parser.parse_args()


###
# MAIN
###

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    args = parse_args()

    builder = AtlasCleanPanelBuilder(
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
        concordance_dir=Path(args.concordance_dir),
        year_start=args.year_start,
        year_end=args.year_end,
    )

    builder.build_all(force=args.force)
    builder.summarize_clean_panel()

    logging.info("Atlas clean panel build completed")


if __name__ == "__main__":
    main()