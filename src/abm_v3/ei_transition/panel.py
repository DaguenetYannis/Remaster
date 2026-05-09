from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.abm_v3.paths import ABMV3Paths


CORRECTED_INPUT_PANEL_ORIENTATION = "transpose_row_fd_without_inventory"
CORE_OUTPUT_COLUMNS = [
    "country_sector",
    "Year",
    "Country",
    "Country_detail",
    "Category",
    "Sector",
    "EI",
    "EI_next",
    "log_EI",
    "log_EI_next",
    "ei_reduction_next",
    "green_capability",
    "g_in",
    "g_out",
    "g_network",
    "general_complexity",
    "X",
    "X_observed",
    "sample_included",
    "exclusion_reason",
]


@dataclass(frozen=True)
class EITransitionPanelResult:
    """Cleaned transition panel and sample diagnostics."""

    panel: pd.DataFrame
    sample_report: pd.DataFrame
    sample_report_by_year: pd.DataFrame
    notes: str


class EITransitionPanelBuilder:
    """Build the historical EI transition panel from the corrected ABM-ready panel."""

    def __init__(
        self,
        paths: ABMV3Paths,
        input_panel_orientation: str = CORRECTED_INPUT_PANEL_ORIENTATION,
    ) -> None:
        self.paths = paths
        self.input_panel_orientation = input_panel_orientation

    def build(self, start_year: int = 1995, end_year: int = 2016) -> EITransitionPanelResult:
        """Load the corrected input panel and build t-to-t+1 EI transition rows."""
        input_path = self.paths.abm_v3_corrected_historical_panel_file(
            start_year,
            end_year,
            self.input_panel_orientation,
        )
        if not input_path.exists():
            raise FileNotFoundError(
                "Corrected ABM v3 input panel is required for EI transition estimation: "
                f"{input_path}"
            )
        raw_panel = pd.read_parquet(input_path)
        return self.build_from_dataframe(raw_panel, start_year, end_year, input_path=input_path)

    def build_from_dataframe(
        self,
        raw_panel: pd.DataFrame,
        start_year: int,
        end_year: int,
        input_path: Path | None = None,
    ) -> EITransitionPanelResult:
        """Build a transition panel from an already loaded dataframe."""
        source_columns = self._resolve_columns(raw_panel)
        panel = self._standardize_columns(raw_panel, source_columns)
        panel = panel.loc[(panel["Year"] >= start_year) & (panel["Year"] <= end_year)].copy()
        panel = panel.sort_values(["country_sector", "Year"]).reset_index(drop=True)
        panel["EI_next"] = panel.groupby("country_sector", sort=False)["EI"].shift(-1)
        panel["Year_next"] = panel.groupby("country_sector", sort=False)["Year"].shift(-1)
        panel.loc[panel["Year_next"] != panel["Year"] + 1, "EI_next"] = np.nan
        panel["log_EI"] = np.nan
        panel.loc[panel["EI"] > 0.0, "log_EI"] = np.log(panel.loc[panel["EI"] > 0.0, "EI"])
        panel["log_EI_next"] = np.nan
        panel.loc[panel["EI_next"] > 0.0, "log_EI_next"] = np.log(
            panel.loc[panel["EI_next"] > 0.0, "EI_next"]
        )
        panel["ei_reduction_next"] = panel["log_EI"] - panel["log_EI_next"]
        panel["exclusion_reason"] = panel.apply(self._exclusion_reason, axis=1)
        panel["sample_included"] = panel["exclusion_reason"] == ""
        output_panel = panel.reindex(columns=CORE_OUTPUT_COLUMNS).copy()
        notes = self._build_notes(source_columns, input_path)
        sample_report = self._build_sample_report(output_panel, start_year, end_year, notes)
        by_year = self._build_sample_report_by_year(output_panel)
        return EITransitionPanelResult(
            panel=output_panel,
            sample_report=sample_report,
            sample_report_by_year=by_year,
            notes=notes,
        )

    def _resolve_columns(self, raw_panel: pd.DataFrame) -> dict[str, str]:
        available = list(raw_panel.columns)
        required_exact = [
            "country_sector",
            "Year",
            "Country",
            "Country_detail",
            "Category",
            "Sector",
            "EI",
            "g_in",
            "g_out",
            "X",
            "X_observed",
        ]
        resolved = {column: column for column in required_exact if column in raw_panel.columns}
        resolved["green_capability"] = self._resolve_alias(
            raw_panel,
            "green_capability",
            ["green_capability", "green_capability_score", "atlas_green_capability"],
        )
        resolved["general_complexity"] = self._resolve_alias(
            raw_panel,
            "general_complexity",
            ["general_complexity", "complexity", "general_capability", "atlas_general_complexity"],
        )
        if "g_network" in raw_panel.columns:
            resolved["g_network"] = "g_network"
        missing = [column for column in required_exact if column not in resolved]
        if resolved["green_capability"] is None:
            missing.append("green_capability")
        if resolved["general_complexity"] is None:
            missing.append("general_complexity")
        if missing:
            raise ValueError(
                "Cannot build EI transition panel. Missing required columns: "
                f"{sorted(missing)}. Available columns: {available}"
            )
        return {key: value for key, value in resolved.items() if value is not None}

    def _resolve_alias(self, raw_panel: pd.DataFrame, canonical_name: str, aliases: list[str]) -> str | None:
        for alias in aliases:
            if alias in raw_panel.columns:
                return alias
        lowered_columns = {column.lower(): column for column in raw_panel.columns}
        if canonical_name in lowered_columns:
            return lowered_columns[canonical_name]
        tokens = canonical_name.lower().split("_")
        for column in raw_panel.columns:
            lowered = column.lower()
            if all(token in lowered for token in tokens):
                return column
        return None

    def _standardize_columns(self, raw_panel: pd.DataFrame, source_columns: dict[str, str]) -> pd.DataFrame:
        panel = pd.DataFrame()
        for canonical in [
            "country_sector",
            "Year",
            "Country",
            "Country_detail",
            "Category",
            "Sector",
            "EI",
            "green_capability",
            "g_in",
            "g_out",
            "general_complexity",
            "X",
            "X_observed",
        ]:
            panel[canonical] = raw_panel[source_columns[canonical]]
        if "g_network" in source_columns:
            panel["g_network"] = raw_panel[source_columns["g_network"]]
        else:
            panel["g_network"] = 0.5 * (
                pd.to_numeric(panel["g_in"], errors="coerce")
                + pd.to_numeric(panel["g_out"], errors="coerce")
            )
        numeric_columns = [
            "Year",
            "EI",
            "green_capability",
            "g_in",
            "g_out",
            "g_network",
            "general_complexity",
            "X",
            "X_observed",
        ]
        for column in numeric_columns:
            panel[column] = pd.to_numeric(panel[column], errors="coerce")
        panel["Year"] = panel["Year"].astype("Int64")
        return panel

    def _exclusion_reason(self, row: pd.Series) -> str:
        reasons = []
        if pd.isna(row["EI"]):
            reasons.append("missing_EI")
        elif float(row["EI"]) <= 0.0:
            reasons.append("non_positive_EI")
        if pd.isna(row["EI_next"]):
            reasons.append("missing_EI_next")
        elif float(row["EI_next"]) <= 0.0:
            reasons.append("non_positive_EI_next")
        for column in ["green_capability", "g_network", "general_complexity"]:
            if pd.isna(row[column]):
                reasons.append(f"missing_{column}")
        return ";".join(reasons)

    def _build_notes(self, source_columns: dict[str, str], input_path: Path | None) -> str:
        g_network_note = (
            "g_network_source=existing_column"
            if source_columns.get("g_network") == "g_network"
            else "g_network_source=computed_0.5_g_in_plus_g_out"
        )
        path_note = f"input_panel={input_path}" if input_path is not None else "input_panel=dataframe"
        return "; ".join(
            [
                path_note,
                f"green_capability_source={source_columns['green_capability']}",
                f"general_complexity_source={source_columns['general_complexity']}",
                g_network_note,
            ]
        )

    def _build_sample_report(
        self,
        panel: pd.DataFrame,
        start_year: int,
        end_year: int,
        notes: str,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "total_rows": len(panel),
                    "included_rows": int(panel["sample_included"].sum()),
                    "excluded_rows": int((~panel["sample_included"]).sum()),
                    "missing_EI_rows": self._reason_count(panel, "missing_EI"),
                    "non_positive_EI_rows": self._reason_count(panel, "non_positive_EI"),
                    "missing_EI_next_rows": self._reason_count(panel, "missing_EI_next"),
                    "non_positive_EI_next_rows": self._reason_count(panel, "non_positive_EI_next"),
                    "missing_green_capability_rows": self._reason_count(panel, "missing_green_capability"),
                    "missing_g_network_rows": self._reason_count(panel, "missing_g_network"),
                    "missing_general_complexity_rows": self._reason_count(panel, "missing_general_complexity"),
                    "first_year": start_year,
                    "last_year": end_year,
                    "number_of_nodes": int(panel["country_sector"].nunique()),
                    "number_of_sectors": int(panel["Sector"].nunique()),
                    "notes": notes,
                }
            ]
        )

    def _build_sample_report_by_year(self, panel: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for year, year_panel in panel.groupby("Year", dropna=False):
            rows.append(
                {
                    "Year": int(year) if pd.notna(year) else np.nan,
                    "total_rows": len(year_panel),
                    "included_rows": int(year_panel["sample_included"].sum()),
                    "excluded_rows": int((~year_panel["sample_included"]).sum()),
                    "missing_EI_rows": self._reason_count(year_panel, "missing_EI"),
                    "non_positive_EI_rows": self._reason_count(year_panel, "non_positive_EI"),
                    "missing_EI_next_rows": self._reason_count(year_panel, "missing_EI_next"),
                    "non_positive_EI_next_rows": self._reason_count(year_panel, "non_positive_EI_next"),
                    "missing_green_capability_rows": self._reason_count(year_panel, "missing_green_capability"),
                    "missing_g_network_rows": self._reason_count(year_panel, "missing_g_network"),
                    "missing_general_complexity_rows": self._reason_count(year_panel, "missing_general_complexity"),
                }
            )
        return pd.DataFrame(rows)

    def _reason_count(self, panel: pd.DataFrame, reason: str) -> int:
        return int(panel["exclusion_reason"].fillna("").str.contains(reason, regex=False).sum())
