from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ValidationResult:
    """Structured validation result used by loaders, features, and tests."""

    passed: bool
    name: str
    message: str
    details: dict[str, Any] | None = None


class DataContractValidator:
    """Validate ABM v3 data contracts without assuming real Eora files exist."""

    def validate_country_sector_key(self, df: pd.DataFrame) -> ValidationResult:
        if "country_sector" not in df.columns:
            return ValidationResult(
                False,
                "country_sector_key",
                "Missing required stable node key: country_sector.",
                {"columns": list(df.columns)},
            )
        missing_count = int(df["country_sector"].isna().sum())
        return ValidationResult(
            missing_count == 0,
            "country_sector_key",
            "country_sector key is present and non-missing."
            if missing_count == 0
            else "country_sector contains missing values.",
            {"missing_count": missing_count},
        )

    def validate_year_coverage(
        self,
        df: pd.DataFrame,
        start_year: int,
        end_year: int,
    ) -> ValidationResult:
        if "Year" not in df.columns:
            return ValidationResult(
                False,
                "year_coverage",
                "Missing required Year column.",
                {"columns": list(df.columns)},
            )
        observed_years = set(pd.to_numeric(df["Year"], errors="coerce").dropna().astype(int))
        expected_years = set(range(start_year, end_year + 1))
        missing_years = sorted(expected_years - observed_years)
        return ValidationResult(
            not missing_years,
            "year_coverage",
            "All requested years are present."
            if not missing_years
            else "Panel is missing requested years.",
            {"missing_years": missing_years, "observed_years": sorted(observed_years)},
        )

    def validate_required_columns(
        self,
        df: pd.DataFrame,
        required_columns: list[str],
    ) -> ValidationResult:
        missing = [column for column in required_columns if column not in df.columns]
        return ValidationResult(
            not missing,
            "required_columns",
            "All required columns are present."
            if not missing
            else f"Missing required columns: {missing}",
            {"missing_columns": missing, "available_columns": list(df.columns)},
        )

    def validate_no_duplicate_nodes(self, df: pd.DataFrame) -> ValidationResult:
        required = ["country_sector", "Year"]
        missing = [column for column in required if column not in df.columns]
        if missing:
            return ValidationResult(
                False,
                "duplicate_nodes",
                f"Cannot check duplicates because columns are missing: {missing}",
                {"missing_columns": missing},
            )
        duplicate_mask = df.duplicated(required)
        duplicate_count = int(duplicate_mask.sum())
        return ValidationResult(
            duplicate_count == 0,
            "duplicate_nodes",
            "No duplicate country_sector-Year rows found."
            if duplicate_count == 0
            else "Duplicate country_sector-Year rows found.",
            {"duplicate_count": duplicate_count},
        )

    def validate_et_alignment(
        self,
        et: pd.DataFrame,
        country_sector_index: pd.Index,
    ) -> ValidationResult:
        expected = pd.Index(country_sector_index.astype(str), name=country_sector_index.name)
        rows = pd.Index(et.index.astype(str))
        columns = pd.Index(et.columns.astype(str))
        rows_match = rows.equals(expected)
        columns_match = columns.equals(expected)
        return ValidationResult(
            rows_match and columns_match,
            "et_alignment",
            "ET rows and columns align with country_sector index."
            if rows_match and columns_match
            else "ET rows or columns are not aligned with country_sector index.",
            {"rows_match": rows_match, "columns_match": columns_match},
        )

    def validate_non_negative(
        self,
        df: pd.DataFrame,
        columns: list[str],
    ) -> ValidationResult:
        missing = [column for column in columns if column not in df.columns]
        negative_counts = {
            column: int((pd.to_numeric(df[column], errors="coerce") < 0).sum())
            for column in columns
            if column in df.columns
        }
        passed = not missing and all(count == 0 for count in negative_counts.values())
        return ValidationResult(
            passed,
            "non_negative",
            "Selected columns are present and non-negative."
            if passed
            else "Selected columns are missing or contain negative values.",
            {"missing_columns": missing, "negative_counts": negative_counts},
        )

    def validate_missingness_report(
        self,
        df: pd.DataFrame,
        columns: list[str],
    ) -> ValidationResult:
        missing_columns = [column for column in columns if column not in df.columns]
        missing_values = {
            column: int(df[column].isna().sum())
            for column in columns
            if column in df.columns
        }
        return ValidationResult(
            not missing_columns,
            "missingness_report",
            "Missingness report created."
            if not missing_columns
            else "Missingness report created with absent columns.",
            {"missing_columns": missing_columns, "missing_values": missing_values},
        )
