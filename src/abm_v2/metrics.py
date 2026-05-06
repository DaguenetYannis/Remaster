"""
Build ABM-ready metrics panels from existing Eora-derived outputs.

This module prepares the empirical state variables used by the ABM.
It should be run before any simulation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricsConfig:
    metrics_dir: Path = Path("data/metrics")
    eora_parquet_dir: Path = Path("data/parquet")
    output_dir: Path = Path("data/abm/metrics")
    inventory_days: float = 30.0
    capacity_margin: float = 1.10
    epsilon: float = 1e-12


class ABMMetricsBuilder:
    """
    Builds the country-sector-year panel required by the ABM.

    Required yearly inputs:
    - data/metrics/{year}/ei_{year}.parquet
    - data/metrics/{year}/greenness_{year}.parquet
    - data/parquet/{year}/T.parquet
    - data/parquet/{year}/FD.parquet

    Optional yearly inputs:
    - data/metrics/{year}/centrality_{year}.parquet
    """

    def __init__(self, config: MetricsConfig) -> None:
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def build_for_year(self, year: int) -> pd.DataFrame:
        ei = self._load_ei(year)
        greenness = self._load_greenness(year)
        production_variables = self._load_production_variables(year)
        centrality = self._load_optional_parquet(year, f"centrality_{year}.parquet")

        panel = ei.copy()

        for yearly_metrics in [greenness, production_variables, centrality]:
            if yearly_metrics is not None:
                panel = self._safe_merge(panel, yearly_metrics)

        panel = self._standardize_columns(panel, year)
        self._validate_metric_alignment(
            panel=panel,
            required_columns=["EI"],
            metric_name="EI",
            year=year,
        )
        self._validate_metric_alignment(
            panel=panel,
            required_columns=["g_local", "g_out", "g_in"],
            metric_name="greenness",
            year=year,
        )
        panel = self._derive_missing_metrics(panel)
        panel = self._validate_panel(panel)

        return panel

    def build_many_years(self, years: list[int]) -> pd.DataFrame:
        panels = [self.build_for_year(year) for year in years]
        full_panel = pd.concat(panels, ignore_index=True)

        output_path = self.config.output_dir / "abm_metrics_panel.parquet"
        full_panel.to_parquet(output_path, index=False)

        return full_panel

    def _year_dir(self, year: int) -> Path:
        return self.config.metrics_dir / str(year)

    def _matrix_year_dir(self, year: int) -> Path:
        return self.config.eora_parquet_dir / str(year)

    def _load_ei(self, year: int) -> pd.DataFrame:
        path = self._year_dir(year) / f"ei_{year}.parquet"

        if not path.exists():
            raise FileNotFoundError(f"Missing required EI file: {path}")

        df = pd.read_parquet(path)
        return self._ensure_country_sector_column(df)

    def _load_greenness(self, year: int) -> pd.DataFrame:
        path = self._year_dir(year) / f"greenness_{year}.parquet"

        if not path.exists():
            raise FileNotFoundError(f"Missing required greenness file: {path}")

        df = pd.read_parquet(path)
        return self._ensure_country_sector_column(df)

    def _load_optional_parquet(self, year: int, filename: str) -> pd.DataFrame | None:
        path = self._year_dir(year) / filename

        if not path.exists():
            LOGGER.warning("Skipping missing optional metrics file: %s", path)
            return None

        df = pd.read_parquet(path)
        return self._ensure_country_sector_column(df)

    def _load_production_variables(self, year: int) -> pd.DataFrame:
        t_path = self._matrix_year_dir(year) / "T.parquet"
        fd_path = self._matrix_year_dir(year) / "FD.parquet"

        if not t_path.exists():
            raise FileNotFoundError(f"Missing required T matrix: {t_path}")

        if not fd_path.exists():
            raise FileNotFoundError(f"Missing required FD matrix: {fd_path}")

        t_matrix = pd.read_parquet(t_path)
        fd_matrix = pd.read_parquet(fd_path)

        self._validate_t_fd_alignment(t_matrix=t_matrix, fd_matrix=fd_matrix, year=year)

        t_matrix.index = t_matrix.index.astype(str)
        t_matrix.columns = t_matrix.columns.astype(str)
        fd_matrix.index = fd_matrix.index.astype(str)

        t_matrix = t_matrix.loc[t_matrix.index, t_matrix.index]
        fd_matrix = fd_matrix.loc[t_matrix.index]

        fd_row_sum = fd_matrix.sum(axis=1)
        t_column_sum = t_matrix.sum(axis=0)
        t_row_sum = t_matrix.sum(axis=1)

        production_variables = pd.DataFrame(
            {
                "country_sector": t_matrix.index.astype(str),
                "X": t_column_sum.reindex(t_matrix.index).to_numpy()
                + fd_row_sum.to_numpy(),
                "D": t_row_sum.to_numpy() + fd_row_sum.to_numpy(),
                "M": t_column_sum.reindex(t_matrix.index).to_numpy(),
            }
        )

        self._validate_production_variables(
            production_variables=production_variables,
            year=year,
        )

        return production_variables

    @staticmethod
    def _validate_t_fd_alignment(
        t_matrix: pd.DataFrame,
        fd_matrix: pd.DataFrame,
        year: int,
    ) -> None:
        if t_matrix.shape[0] != t_matrix.shape[1]:
            raise ValueError(
                f"T matrix for {year} must be square. Shape: {t_matrix.shape}"
            )

        if not t_matrix.index.is_unique:
            raise ValueError(f"T matrix for {year} has duplicate row labels.")

        if not t_matrix.columns.is_unique:
            raise ValueError(f"T matrix for {year} has duplicate column labels.")

        index_labels = pd.Index(t_matrix.index.astype(str))
        column_labels = pd.Index(t_matrix.columns.astype(str))

        if not index_labels.equals(column_labels):
            if set(index_labels) != set(column_labels):
                raise ValueError(
                    f"T matrix for {year} has row/column labels that cannot be aligned."
                )

            LOGGER.warning(
                "T matrix for %s has matching row/column labels in different order; "
                "columns will be aligned to rows.",
                year,
            )

        fd_index_labels = pd.Index(fd_matrix.index.astype(str))

        if not fd_index_labels.equals(index_labels):
            if set(fd_index_labels) != set(index_labels):
                raise ValueError(
                    f"FD index for {year} cannot be aligned with T index."
                )

            LOGGER.warning(
                "FD matrix for %s has matching row labels in different order; "
                "rows will be aligned to T.",
                year,
            )

    @staticmethod
    def _validate_production_variables(
        production_variables: pd.DataFrame,
        year: int,
    ) -> None:
        required_columns = ["country_sector", "X", "D", "M"]
        missing = [
            column
            for column in required_columns
            if column not in production_variables.columns
        ]

        if missing:
            raise ValueError(
                f"Production variables for {year} are missing columns: {missing}"
            )

        if production_variables.empty:
            raise ValueError(f"Production variables for {year} are empty.")

        for column in ["X", "D", "M"]:
            numeric_values = pd.to_numeric(
                production_variables[column],
                errors="coerce",
            )

            if numeric_values.isna().any():
                missing_count = int(numeric_values.isna().sum())
                raise ValueError(
                    f"Production variable {column} for {year} contains "
                    f"{missing_count} missing/non-numeric values."
                )

            if not np.isfinite(numeric_values).all():
                raise ValueError(
                    f"Production variable {column} for {year} contains non-finite values."
                )

    @staticmethod
    def _ensure_country_sector_column(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "country_sector" in df.columns:
            return df

        if df.index.name == "country_sector":
            return df.reset_index()

        if df.index.name is None and not isinstance(df.index, pd.RangeIndex):
            return df.reset_index(names="country_sector")

        if "index" in df.columns:
            return df.rename(columns={"index": "country_sector"})

        raise ValueError(
            "Could not identify country-sector identifier. "
            "Expected 'country_sector', index named 'country_sector', "
            "an unnamed non-range index, or an 'index' column."
        )

    @staticmethod
    def _safe_merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        duplicate_columns = [
            col for col in right.columns
            if col in left.columns and col != "country_sector"
        ]

        right_clean = right.drop(columns=duplicate_columns)

        merged = left.merge(
            right_clean,
            on="country_sector",
            how="left",
            validate="one_to_one",
        )

        if merged.empty:
            raise ValueError("Merge produced an empty ABM metrics panel.")

        return merged

    def _standardize_columns(self, df: pd.DataFrame, year: int) -> pd.DataFrame:
        df = df.copy()
        df["Year"] = year

        rename_map = {
            "direct_co2_intensity": "EI",
            "emissions_intensity": "EI",
            "g_base": "g_local",
            "g_out_network": "g_out",
            "g_in_network": "g_in",
            "greeness": "g_local",
            "green-ness": "g_local",
            "g": "g_local",
            "network_greeness_in": "g_in",
            "network_greeness_out": "g_out",
            "incoming_greeness": "g_in",
            "outgoing_greeness": "g_out",
        }

        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        if "Country" not in df.columns or "Sector" not in df.columns:
            parsed = df["country_sector"].apply(self._parse_country_sector)
            parsed_df = pd.DataFrame(
                parsed.tolist(),
                columns=["Country_parsed", "Sector_parsed"],
            )

            if "Country" not in df.columns:
                df["Country"] = parsed_df["Country_parsed"]

            if "Sector" not in df.columns:
                df["Sector"] = parsed_df["Sector_parsed"]

        return df

    @staticmethod
    def _parse_country_sector(value: str) -> tuple[str, str]:
        parts = [part.strip() for part in str(value).split("|")]

        if len(parts) >= 4:
            return parts[0], parts[-1]

        if len(parts) >= 2:
            return parts[0], parts[-1]

        return "Unknown", str(value)

    @staticmethod
    def _validate_metric_alignment(
        panel: pd.DataFrame,
        required_columns: list[str],
        metric_name: str,
        year: int,
    ) -> None:
        missing_columns = [
            column for column in required_columns
            if column not in panel.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{metric_name} metrics for {year} are missing columns: "
                f"{missing_columns}"
            )

        missing_rows = panel[required_columns].isna().any(axis=1)

        if missing_rows.any():
            missing_count = int(missing_rows.sum())
            raise ValueError(
                f"{metric_name} metrics for {year} do not align with "
                f"{missing_count} country-sector rows."
            )

    def _derive_missing_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "EI" not in df.columns:
            raise ValueError("Missing EI column after standardization.")

        df["EI"] = self._clean_numeric(df["EI"])

        if "g_local" not in df.columns:
            df["g_local"] = 1 / (1 + df["EI"])

        if "g_in" not in df.columns:
            df["g_in"] = df["g_local"]

        if "g_out" not in df.columns:
            df["g_out"] = df["g_local"]

        if "NG" not in df.columns:
            df["NG"] = df[["g_in", "g_out"]].mean(axis=1)

        missing_production_columns = [
            column for column in ["X", "D", "M"]
            if column not in df.columns
        ]

        if missing_production_columns:
            raise ValueError(
                "Missing production variables after loading T and FD: "
                f"{missing_production_columns}"
            )

        df["X"] = self._clean_numeric(df["X"])
        df["D"] = self._clean_numeric(df["D"])
        df["M"] = self._clean_numeric(df["M"])

        if "inventory_base" not in df.columns:
            df["inventory_base"] = (
                self.config.inventory_days * df["M"] / 365
            ).round(12)

        if "capacity_base" not in df.columns:
            df["capacity_base"] = (self.config.capacity_margin * df["X"]).round(12)

        if "capability_readiness" not in df.columns:
            df["capability_readiness"] = df[["g_local", "NG"]].mean(axis=1)

        return df

    @staticmethod
    def _clean_numeric(series: pd.Series) -> pd.Series:
        clean = pd.to_numeric(series, errors="coerce")
        clean = clean.replace([np.inf, -np.inf], np.nan)
        return clean

    def _validate_panel(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        required_columns = [
            "Year",
            "country_sector",
            "Country",
            "Sector",
            "X",
            "D",
            "M",
            "EI",
            "g_local",
            "g_in",
            "g_out",
            "NG",
            "inventory_base",
            "capacity_base",
            "capability_readiness",
        ]

        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            raise ValueError(f"ABM metrics panel is missing columns: {missing}")

        numeric_columns = [
            "X",
            "D",
            "M",
            "EI",
            "g_local",
            "g_in",
            "g_out",
            "NG",
            "inventory_base",
            "capacity_base",
            "capability_readiness",
        ]

        for col in numeric_columns:
            df[col] = self._clean_numeric(df[col])

        production_missing = df[["X", "D", "M"]].isna().any(axis=1)

        if production_missing.any():
            missing_count = int(production_missing.sum())
            raise ValueError(
                "ABM metrics panel contains missing production variables "
                f"for {missing_count} rows."
            )

        for col in [c for c in numeric_columns if c not in ["X", "D", "M"]]:
            df[col] = df[col].fillna(0.0)

        return df[required_columns + [col for col in df.columns if col not in required_columns]]


def build_abm_metrics_panel(
    years: list[int],
    metrics_dir: Path = Path("data/metrics"),
    eora_parquet_dir: Path = Path("data/parquet"),
    output_dir: Path = Path("data/abm/metrics"),
) -> pd.DataFrame:
    config = MetricsConfig(
        metrics_dir=metrics_dir,
        eora_parquet_dir=eora_parquet_dir,
        output_dir=output_dir,
    )
    builder = ABMMetricsBuilder(config)
    return builder.build_many_years(years)
