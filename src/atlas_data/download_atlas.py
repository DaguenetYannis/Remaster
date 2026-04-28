from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests


###
# CONSTANTS
###

ATLAS_GRAPHQL_URL = "https://atlas.hks.harvard.edu/api/graphql"


###
# ERRORS
###

class AtlasApiError(Exception):
    pass


###
# HELPERS
###

def parse_country_id(value: str | int) -> int:
    value = str(value)
    return int(value.replace("country-", ""))


def parse_product_id(value: str | int) -> int:
    value = str(value)
    return int(value.split("-")[-1])


###
# GRAPHQL CLIENT
###

class AtlasGraphQLClient:
    def __init__(
        self,
        url: str = ATLAS_GRAPHQL_URL,
        timeout: int = 60,
        max_retries: int = 3,
        sleep_seconds: float = 0.6,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds

    def query(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {"query": query}

        if variables is not None:
            payload["variables"] = variables

        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.url,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(payload),
                    timeout=self.timeout,
                )

                if response.status_code != 200:
                    raise AtlasApiError(
                        f"HTTP {response.status_code}: {response.text[:500]}"
                    )

                result = response.json()

                if "errors" in result:
                    raise AtlasApiError(json.dumps(result["errors"], indent=2))

                time.sleep(self.sleep_seconds)

                return result["data"]

            except Exception as exc:
                last_error = exc
                logging.warning(
                    "Atlas query failed on attempt %s/%s: %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                time.sleep(self.sleep_seconds * attempt)

        raise AtlasApiError(f"Atlas query failed after retries: {last_error}")


###
# DOWNLOADER
###

class AtlasDownloader:
    def __init__(
        self,
        output_dir: Path,
        product_class: str = "HS92",
        product_level: int = 4,
        client: AtlasGraphQLClient | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.product_class = product_class
        self.product_level = product_level
        self.client = client or AtlasGraphQLClient()

        self.metadata_dir = self.output_dir / "metadata"
        self.country_product_year_dir = self.output_dir / "country_product_year"
        self.country_product_year_shards_dir = (
            self.output_dir / "country_product_year_shards"
        )

        self._create_dirs()

    def _create_dirs(self) -> None:
        for path in [
            self.output_dir,
            self.metadata_dir,
            self.country_product_year_dir,
            self.country_product_year_shards_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def download_all(self, years: Iterable[int], force: bool = False) -> None:
        logging.info("Downloading Atlas metadata")

        countries = self.download_countries(force=force)
        self.download_products(force=force)

        for year in years:
            logging.info("Downloading Atlas country-product-year data for %s", year)

            self.download_country_product_year(
                year=year,
                countries=countries,
                force=force,
            )

    def download_countries(self, force: bool = False) -> pd.DataFrame:
        output_path = self.metadata_dir / "location_country.parquet"

        if output_path.exists() and not force:
            logging.info("Skipping countries; file already exists: %s", output_path)
            return pd.read_parquet(output_path)

        query = """
        {
          locationCountry {
            countryId
            iso3Code
            nameEn
          }
        }
        """

        data = self.client.query(query)
        df = pd.DataFrame(data["locationCountry"])

        df["countryIdRaw"] = df["countryId"]
        df["countryIdNumeric"] = df["countryId"].map(parse_country_id)

        df = df.sort_values("countryIdNumeric").reset_index(drop=True)
        df.to_parquet(output_path, index=False)

        logging.info("Saved countries: %s rows -> %s", len(df), output_path)

        return df

    def download_products(self, force: bool = False) -> pd.DataFrame:
        output_path = (
            self.metadata_dir
            / f"product_{self.product_class.lower()}_level{self.product_level}.parquet"
        )

        if output_path.exists() and not force:
            logging.info("Skipping products; file already exists: %s", output_path)
            return pd.read_parquet(output_path)

        if self.product_class != "HS92":
            raise ValueError(
                "This downloader currently supports product metadata only for HS92. "
                "Add productHs12 or productSitc queries before using another class."
            )

        query = """
        query ProductHs92($productLevel: Int!) {
          productHs92(productLevel: $productLevel) {
            productId
            code
            nameEn
            clusterId
            naturalResource
            greenProduct
          }
        }
        """

        data = self.client.query(
            query,
            variables={"productLevel": self.product_level},
        )

        df = pd.DataFrame(data["productHs92"])

        df["productIdRaw"] = df["productId"]
        df["productIdNumeric"] = df["productId"].map(parse_product_id)

        df = df.sort_values("code").reset_index(drop=True)
        df.to_parquet(output_path, index=False)

        logging.info("Saved products: %s rows -> %s", len(df), output_path)

        return df

    def download_country_product_year(
        self,
        year: int,
        countries: pd.DataFrame,
        force: bool = False,
    ) -> pd.DataFrame:
        output_path = (
            self.country_product_year_dir
            / f"country_product_year_{self.product_class.lower()}_level{self.product_level}_{year}.parquet"
        )

        if output_path.exists() and not force:
            logging.info("Skipping country-product-year %s; file already exists", year)
            return pd.read_parquet(output_path)

        year_shard_dir = self.country_product_year_shards_dir / str(year)
        year_shard_dir.mkdir(parents=True, exist_ok=True)

        frames: list[pd.DataFrame] = []

        for _, country in countries.iterrows():
            country_id = int(country["countryIdNumeric"])
            country_id_raw = country["countryIdRaw"]
            iso3_code = country["iso3Code"]
            country_name = country["nameEn"]

            shard_path = (
                year_shard_dir
                / f"country_product_year_{country_id}_{year}.parquet"
            )

            if shard_path.exists() and not force:
                df_country = pd.read_parquet(shard_path)
                frames.append(df_country)
                continue

            logging.info("Downloading country-product-year: %s %s", iso3_code, year)

            df_country = self._download_country_product_year_single(
                country_id=country_id,
                year=year,
            )

            if df_country.empty:
                logging.warning(
                    "No country-product-year rows for %s %s",
                    iso3_code,
                    year,
                )
                continue

            df_country["countryId"] = country_id
            df_country["countryIdRaw"] = country_id_raw
            df_country["iso3Code"] = iso3_code
            df_country["countryName"] = country_name

            df_country["productIdRaw"] = df_country["productId"]
            df_country["productIdNumeric"] = df_country["productId"].map(
                parse_product_id
            )

            df_country.to_parquet(shard_path, index=False)
            frames.append(df_country)

        if not frames:
            logging.warning("No country-product-year data found for %s", year)
            return pd.DataFrame()

        df_year = pd.concat(frames, ignore_index=True)

        df_year = df_year.sort_values(
            ["countryId", "productIdNumeric", "year"]
        ).reset_index(drop=True)

        df_year.to_parquet(output_path, index=False)

        logging.info(
            "Saved country-product-year %s: %s rows -> %s",
            year,
            len(df_year),
            output_path,
        )

        return df_year

    def _download_country_product_year_single(
        self,
        country_id: int,
        year: int,
    ) -> pd.DataFrame:
        query = """
        query CountryProductYear(
          $countryId: Int!,
          $productClass: ProductClass!,
          $productLevel: Int!,
          $yearMin: Int!,
          $yearMax: Int!
        ) {
          countryProductYear(
            countryId: $countryId,
            productClass: $productClass,
            productLevel: $productLevel,
            yearMin: $yearMin,
            yearMax: $yearMax
          ) {
            productId
            year
            exportValue
            importValue
            exportRca
            globalMarketShare
            distance
            cog
            normalizedPci
          }
        }
        """

        data = self.client.query(
            query,
            variables={
                "countryId": country_id,
                "productClass": self.product_class,
                "productLevel": self.product_level,
                "yearMin": year,
                "yearMax": year,
            },
        )

        return pd.DataFrame(data["countryProductYear"])


###
# CLI
###

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="data/atlas/raw",
        help="Directory where raw Atlas parquet files will be stored.",
    )

    parser.add_argument(
        "--year-start",
        type=int,
        default=1990,
    )

    parser.add_argument(
        "--year-end",
        type=int,
        default=2016,
        help="Inclusive end year.",
    )

    parser.add_argument(
        "--product-class",
        default="HS92",
        choices=["HS92", "HS12", "SITC"],
    )

    parser.add_argument(
        "--product-level",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even if cached parquet files already exist.",
    )

    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.6,
        help="Delay between API requests.",
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

    years = range(args.year_start, args.year_end + 1)

    client = AtlasGraphQLClient(
        sleep_seconds=args.sleep_seconds,
    )

    downloader = AtlasDownloader(
        output_dir=Path(args.output_dir),
        product_class=args.product_class,
        product_level=args.product_level,
        client=client,
    )

    downloader.download_all(
        years=years,
        force=args.force,
    )

    logging.info("Atlas download completed")


if __name__ == "__main__":
    main()