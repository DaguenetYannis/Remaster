from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


###
# CONFIG
###

DEFAULT_CLEAN_PANEL_PATH = Path(
    "data/atlas/processed/atlas_hs92_level4_clean_panel_1995_2016.parquet"
)

DEFAULT_CONCORDANCE_PATH = Path(
    "data/atlas/concordance/hs92_to_eora26_prefilled.csv"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/atlas/processed/atlas_eora26_sector_capabilities_1995_2016.parquet"
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


def weighted_average(
    values: pd.Series,
    weights: pd.Series,
) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")

    mask = values.notna() & weights.notna() & (weights > 0)

    if not mask.any():
        return np.nan

    return float(np.average(values[mask], weights=weights[mask]))


def safe_share(numerator: float, denominator: float) -> float:
    if denominator == 0 or pd.isna(denominator):
        return 0.0

    return float(numerator / denominator)


###
# AGGREGATOR
###

class AtlasToEoraSectorAggregator:
    def __init__(
        self,
        clean_panel_path: Path,
        concordance_path: Path,
        output_path: Path,
    ) -> None:
        self.clean_panel_path = clean_panel_path
        self.concordance_path = concordance_path
        self.output_path = output_path

    def build(self, force: bool = False) -> pd.DataFrame:
        if self.output_path.exists() and not force:
            logging.info("Skipping aggregation; file already exists: %s", self.output_path)
            return pd.read_parquet(self.output_path)

        panel = self.load_clean_panel()
        concordance = self.load_concordance()

        merged = self.merge_concordance(
            panel=panel,
            concordance=concordance,
        )

        self.validate_merged_panel(merged)

        aggregated = self.aggregate_to_country_sector_year(merged)

        ensure_directory(self.output_path.parent)
        aggregated.to_parquet(self.output_path, index=False)

        logging.info(
            "Saved Atlas-to-Eora sector panel: %s rows -> %s",
            len(aggregated),
            self.output_path,
        )

        self.summarize_output(aggregated)

        return aggregated

    def load_clean_panel(self) -> pd.DataFrame:
        print_section("Loading Atlas clean panel")

        if not self.clean_panel_path.exists():
            raise FileNotFoundError(f"Clean panel not found: {self.clean_panel_path}")

        panel = pd.read_parquet(self.clean_panel_path)

        logging.info("Rows: %s", len(panel))
        logging.info("Years: %s - %s", panel["year"].min(), panel["year"].max())
        logging.info("Countries: %s", panel["iso3Code"].nunique())
        logging.info("Products: %s", panel["productIdRaw"].nunique())

        return panel

    def load_concordance(self) -> pd.DataFrame:
        print_section("Loading concordance")

        if not self.concordance_path.exists():
            raise FileNotFoundError(f"Concordance not found: {self.concordance_path}")

        concordance = pd.read_csv(self.concordance_path, dtype=str).fillna("")

        required_cols = [
            "productIdRaw",
            "eora26_sector",
            "mapping_status",
            "mapping_method",
        ]

        missing_cols = [col for col in required_cols if col not in concordance.columns]

        if missing_cols:
            raise ValueError(f"Concordance missing columns: {missing_cols}")

        concordance = concordance[
            [
                "productIdRaw",
                "eora26_sector",
                "mapping_status",
                "mapping_method",
            ]
        ].copy()

        concordance["productIdRaw"] = concordance["productIdRaw"].str.strip()
        concordance["eora26_sector"] = concordance["eora26_sector"].str.strip()

        logging.info("Rows: %s", len(concordance))
        logging.info("Mapped rows: %s", (concordance["eora26_sector"] != "").sum())

        return concordance

    def merge_concordance(
        self,
        panel: pd.DataFrame,
        concordance: pd.DataFrame,
    ) -> pd.DataFrame:
        print_section("Merging Atlas panel with Eora concordance")

        merged = panel.merge(
            concordance,
            on="productIdRaw",
            how="left",
            validate="many_to_one",
        )

        missing_mapping = merged["eora26_sector"].isna() | (merged["eora26_sector"] == "")

        if missing_mapping.any():
            missing_count = int(missing_mapping.sum())
            raise ValueError(f"Missing Eora sector mapping for {missing_count} rows.")

        logging.info("Merged rows: %s", len(merged))
        logging.info("Unique Eora sectors: %s", merged["eora26_sector"].nunique())

        return merged

    def validate_merged_panel(self, merged: pd.DataFrame) -> None:
        print_section("Validating merged panel")

        duplicate_count = merged.duplicated(
            subset=["iso3Code", "year", "productIdRaw"]
        ).sum()

        logging.info("Duplicate country-product-year rows: %s", duplicate_count)

        if duplicate_count > 0:
            raise ValueError("Duplicate country-product-year rows found after merge.")

        logging.info("Missing RCA share: %.4f", merged["exportRca"].isna().mean())
        logging.info(
            "Missing normalized PCI share: %.4f",
            merged["normalizedPci"].isna().mean(),
        )

        logging.info("Service-like rows: %s", int(merged["is_service_like"].sum()))
        logging.info("HS4 rows: %s", int(merged["is_hs4_code"].sum()))

    def aggregate_to_country_sector_year(self, merged: pd.DataFrame) -> pd.DataFrame:
        print_section("Aggregating to country-sector-year")

        df = merged.copy()

        for col in [
            "exportValue",
            "importValue",
            "exportRca",
            "normalizedPci",
            "greenProduct",
            "naturalResource",
            "is_hs4_code",
            "is_service_like",
            "has_rca",
            "is_good_capability",
            "is_green_good_capability",
        ]:
            if col not in df.columns:
                raise ValueError(f"Missing required column in merged panel: {col}")

        numeric_cols = [
            "exportValue",
            "importValue",
            "exportRca",
            "normalizedPci",
        ]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        bool_cols = [
            "greenProduct",
            "naturalResource",
            "is_hs4_code",
            "is_service_like",
            "has_rca",
            "is_good_capability",
            "is_green_good_capability",
        ]

        for col in bool_cols:
            df[col] = df[col].astype(bool)

        group_cols = [
            "iso3Code",
            "countryName",
            "year",
            "eora26_sector",
        ]

        rows = []

        for keys, group in df.groupby(group_cols, dropna=False):
            iso3, country_name, year, eora_sector = keys

            goods = group[group["is_hs4_code"]].copy()
            services = group[group["is_service_like"]].copy()

            active_goods = goods[goods["exportRca"] > 1].copy()
            active_green_goods = active_goods[active_goods["greenProduct"]].copy()

            total_export_value = group["exportValue"].sum(skipna=True)
            total_import_value = group["importValue"].sum(skipna=True)

            goods_export_value = goods["exportValue"].sum(skipna=True)
            services_export_value = services["exportValue"].sum(skipna=True)

            active_goods_export_value = active_goods["exportValue"].sum(skipna=True)
            active_green_goods_export_value = active_green_goods["exportValue"].sum(skipna=True)

            product_count = group["productIdRaw"].nunique()
            goods_product_count = goods["productIdRaw"].nunique()
            service_product_count = services["productIdRaw"].nunique()

            active_good_count = active_goods["productIdRaw"].nunique()
            active_green_good_count = active_green_goods["productIdRaw"].nunique()

            row = {
                "iso3Code": iso3,
                "countryName": country_name,
                "year": int(year),
                "eora26_sector": eora_sector,

                # Scale
                "atlas_export_value": total_export_value,
                "atlas_import_value": total_import_value,
                "atlas_goods_export_value": goods_export_value,
                "atlas_services_export_value": services_export_value,

                # Coverage / composition
                "atlas_product_count": product_count,
                "atlas_goods_product_count": goods_product_count,
                "atlas_service_product_count": service_product_count,
                "atlas_service_export_share": safe_share(
                    services_export_value,
                    total_export_value,
                ),

                # RCA-thresholded capability set
                "active_good_count": active_good_count,
                "active_good_export_value": active_goods_export_value,
                "active_good_share_of_goods_products": safe_share(
                    active_good_count,
                    goods_product_count,
                ),
                "active_good_share_of_goods_exports": safe_share(
                    active_goods_export_value,
                    goods_export_value,
                ),

                # Baseline complexity variables
                "capability_mean_pci": active_goods["normalizedPci"].mean(skipna=True),
                "capability_export_weighted_pci": weighted_average(
                    active_goods["normalizedPci"],
                    active_goods["exportValue"],
                ),

                # Green capability variables
                "green_active_good_count": active_green_good_count,
                "green_capability_share": safe_share(
                    active_green_good_count,
                    active_good_count,
                ),
                "green_active_good_export_value": active_green_goods_export_value,
                "green_capability_export_share": safe_share(
                    active_green_goods_export_value,
                    active_goods_export_value,
                ),

                # Diagnostics
                "missing_rca_share": group["exportRca"].isna().mean(),
                "missing_pci_share": group["normalizedPci"].isna().mean(),
            }

            rows.append(row)

        result = pd.DataFrame(rows)

        result = result.sort_values(
            ["iso3Code", "year", "eora26_sector"]
        ).reset_index(drop=True)

        logging.info("Aggregated rows: %s", len(result))
        logging.info("Countries: %s", result["iso3Code"].nunique())
        logging.info("Years: %s - %s", result["year"].min(), result["year"].max())
        logging.info("Eora sectors: %s", result["eora26_sector"].nunique())

        return result

    def summarize_output(self, df: pd.DataFrame) -> None:
        print_section("Output summary")

        logging.info("Rows: %s", len(df))
        logging.info("Countries: %s", df["iso3Code"].nunique())
        logging.info("Years: %s - %s", df["year"].min(), df["year"].max())
        logging.info("Sectors: %s", df["eora26_sector"].nunique())

        logging.info(
            "Rows with at least one active good: %s",
            int((df["active_good_count"] > 0).sum()),
        )

        logging.info(
            "Rows with green capabilities: %s",
            int((df["green_active_good_count"] > 0).sum()),
        )

        logging.info(
            "Mean green capability share: %.4f",
            df["green_capability_share"].mean(skipna=True),
        )


###
# CLI
###

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--clean-panel-path",
        default=str(DEFAULT_CLEAN_PANEL_PATH),
    )

    parser.add_argument(
        "--concordance-path",
        default=str(DEFAULT_CONCORDANCE_PATH),
    )

    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_PATH),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing aggregated panel.",
    )

    return parser.parse_args()


###
# MAIN
###

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    args = parse_args()

    aggregator = AtlasToEoraSectorAggregator(
        clean_panel_path=Path(args.clean_panel_path),
        concordance_path=Path(args.concordance_path),
        output_path=Path(args.output_path),
    )

    aggregator.build(force=args.force)


if __name__ == "__main__":
    main()