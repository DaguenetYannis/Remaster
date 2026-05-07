from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.abm_v3.config import ABMV3Config
from src.abm_v3.data_contracts import DataContractValidator
from src.abm_v3.outputs import ABMV3OutputWriter
from src.abm_v3.paths import ABMV3Paths


@dataclass
class RealDataSmokeTester:
    """Check whether the merged Eora-Atlas panel can support ABM v3.

    The smoke test is intentionally diagnostic. Missing Atlas-derived values
    are reported, never silently converted to zero.
    """

    paths: ABMV3Paths
    config: ABMV3Config = field(default_factory=ABMV3Config)
    start_year: int = 1995
    end_year: int = 2016

    def run(
        self,
        df: pd.DataFrame | None = None,
        write_report: bool = True,
        panel_kind: str = "auto",
    ) -> pd.DataFrame:
        if df is None:
            path, using_input_panel = self._select_panel_path(panel_kind)
            rows = [self._check_file_exists(path)]
            rows.append(
                {
                    "check": "abm_ready_input_panel_selected",
                    "passed": using_input_panel,
                    "message": "Smoke test selected ABM-ready input panel."
                    if using_input_panel
                    else "Smoke test selected old merged metrics panel.",
                    "details": str({"path": str(path)}),
                }
            )
            if not path.exists():
                report = pd.DataFrame(rows)
                if write_report:
                    ABMV3OutputWriter(self.paths).write_dataframe(
                        report,
                        "diagnostics",
                        "real_data_smoke_test.csv",
                    )
                return report
            df = pd.read_parquet(path)
        else:
            rows = [{"check": "file_exists", "passed": True, "message": "Dataframe supplied directly.", "details": ""}]
            using_input_panel = self._looks_like_input_panel(df)

        validator = DataContractValidator()
        if using_input_panel:
            required_core = [
                "country_sector",
                "Country",
                "Sector",
                "Year",
                "X",
                "D",
                "M",
                "available_inputs",
                "EI",
                "g_in",
                "g_out",
                "observed_input_intensity",
                "effective_input_intensity",
                "input_intensity_source",
            ]
            rows.extend(self._validation_result_rows([
                validator.validate_required_columns(df, required_core),
                validator.validate_year_coverage(df, 1995, 2016),
                validator.validate_no_duplicate_nodes(df),
                validator.validate_non_negative(df, ["X", "D", "M", "available_inputs", "EI"]),
                validator.validate_country_sector_key(df),
            ]))
            rows.extend(self._input_panel_zero_share_rows(df))
            rows.extend(self._input_intensity_smoke_rows(df))
            rows.extend(self._constructibility_checks(df))
            rows.extend(self._missingness_rows(df))
            rows.append(self._valid_sector_check(df))
            report = pd.DataFrame(rows)
            if write_report:
                ABMV3OutputWriter(self.paths).write_dataframe(
                    report,
                    "diagnostics",
                    "real_data_smoke_test.csv",
                )
            return report

        rows.extend(self._validation_result_rows([
            validator.validate_required_columns(df, ["Country", "Sector", "Year"]),
            validator.validate_year_coverage(df, 1995, 2016),
        ]))
        canonical = self._canonical_view(df)
        rows.extend(self._validation_result_rows([
            validator.validate_required_columns(canonical, ["country_sector", "Country", "Sector", "Year"]),
            validator.validate_required_columns(canonical, ["X", "D", "EI", "g_in", "g_out"]),
            validator.validate_no_duplicate_nodes(canonical),
            validator.validate_non_negative(canonical, ["X", "D", "EI"]),
            validator.validate_country_sector_key(canonical),
        ]))
        rows.extend(self._constructibility_checks(df))
        rows.extend(self._missingness_rows(df))
        rows.append(self._valid_sector_check(df))

        report = pd.DataFrame(rows)
        if write_report:
            ABMV3OutputWriter(self.paths).write_dataframe(
                report,
                "diagnostics",
                "real_data_smoke_test.csv",
            )
        return report

    def _select_panel_path(self, panel_kind: str) -> tuple[Path, bool]:
        input_panel_path = self.paths.abm_v3_historical_panel_file(self.start_year, self.end_year)
        if panel_kind == "input_panel":
            return input_panel_path, True
        if panel_kind == "merged_panel":
            return self.paths.eora_atlas_merged_file, False
        if panel_kind != "auto":
            raise ValueError(f"Unsupported smoke-test panel kind: {panel_kind}")
        if input_panel_path.exists():
            return input_panel_path, True
        return self.paths.eora_atlas_merged_file, False

    def _looks_like_input_panel(self, df: pd.DataFrame) -> bool:
        return {"X", "D", "M", "available_inputs"}.issubset(df.columns)

    def _input_panel_zero_share_rows(self, df: pd.DataFrame) -> list[dict[str, object]]:
        rows = []
        for column in ["X", "D", "M"]:
            rows.append(
                {
                    "check": f"share_zero_{column}",
                    "passed": column in df.columns,
                    "message": "Zero share calculated." if column in df.columns else "Column absent.",
                    "details": str({"share": float(df[column].eq(0).mean()) if column in df.columns else None}),
                }
            )
        return rows

    def _input_intensity_smoke_rows(self, df: pd.DataFrame) -> list[dict[str, object]]:
        rows = []
        for column in ["observed_input_intensity", "effective_input_intensity"]:
            rows.append(
                {
                    "check": f"share_{column}_missing",
                    "passed": column in df.columns,
                    "message": "Missing share calculated." if column in df.columns else "Column absent.",
                    "details": str({"share": float(df[column].isna().mean()) if column in df.columns else None}),
                }
            )
        if "input_intensity_source" in df.columns:
            shares = df["input_intensity_source"].value_counts(normalize=True, dropna=False).to_dict()
        else:
            shares = {}
        rows.append(
            {
                "check": "input_intensity_source_shares",
                "passed": "input_intensity_source" in df.columns,
                "message": "Input intensity fallback shares calculated."
                if "input_intensity_source" in df.columns
                else "input_intensity_source absent.",
                "details": str(shares),
            }
        )
        return rows

    def _check_file_exists(self, path: Path) -> dict[str, object]:
        return {
            "check": "file_exists",
            "passed": path.exists(),
            "message": f"Checked merged panel path: {path}",
            "details": "",
        }

    def _validation_result_rows(self, results: list[object]) -> list[dict[str, object]]:
        return [
            {
                "check": result.name,
                "passed": result.passed,
                "message": result.message,
                "details": str(result.details),
            }
            for result in results
        ]

    def _canonical_view(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        if "country_sector" not in result.columns and {"Country", "Country_detail", "Category", "Sector"}.issubset(result.columns):
            result["country_sector"] = (
                result["Country"].astype(str)
                + " | "
                + result["Country_detail"].astype(str)
                + " | "
                + result["Category"].astype(str)
                + " | "
                + result["Sector"].astype(str)
            )
        rename_map = {
            "emissions_intensity": "EI",
            "g_in_network": "g_in",
            "g_out_network": "g_out",
            "g_base": "g_local",
        }
        for source, target in rename_map.items():
            if target not in result.columns and source in result.columns:
                result[target] = result[source]
        return result

    def _constructibility_checks(self, df: pd.DataFrame) -> list[dict[str, object]]:
        capability_sources = [
            "green_active_good_export_value",
            "active_good_export_value",
            "green_capability_export_share",
        ]
        can_make_green_capability = "green_capability" in df.columns or all(
            column in df.columns for column in capability_sources
        )
        can_make_complexity = "general_complexity" in df.columns or "capability_export_weighted_pci" in df.columns
        can_make_country_sector = "country_sector" in df.columns or {"Country", "Country_detail", "Category", "Sector"}.issubset(df.columns)
        return [
            {
                "check": "country_sector_constructible",
                "passed": can_make_country_sector,
                "message": "country_sector exists or can be constructed from Eora label components.",
                "details": "",
            },
            {
                "check": "green_capability_constructible",
                "passed": can_make_green_capability,
                "message": "green_capability exists or Atlas source columns are present.",
                "details": str({"source_columns": capability_sources}),
            },
            {
                "check": "general_complexity_constructible",
                "passed": can_make_complexity,
                "message": "general_complexity exists or capability_export_weighted_pci is present.",
                "details": str({"source_column": "capability_export_weighted_pci"}),
            },
        ]

    def _missingness_rows(self, df: pd.DataFrame) -> list[dict[str, object]]:
        columns = [
            "X",
            "D",
            "EI",
            "g_in",
            "g_out",
            "green_capability",
            "green_capability_export_share",
            "general_complexity",
            "capability_export_weighted_pci",
        ]
        rows = []
        for column in columns:
            rows.append(
                {
                    "check": f"missingness_{column}",
                    "passed": column in df.columns,
                    "message": "Missingness counted." if column in df.columns else "Column absent.",
                    "details": str({
                        "missing_count": int(df[column].isna().sum()) if column in df.columns else None,
                        "row_count": int(len(df)),
                    }),
                }
            )
        return rows

    def _valid_sector_check(self, df: pd.DataFrame) -> dict[str, object]:
        if "Sector" not in df.columns:
            return {"check": "valid_sector", "passed": False, "message": "Sector column absent.", "details": ""}
        missing_count = int(df["Sector"].isna().sum())
        return {
            "check": "valid_sector",
            "passed": missing_count == 0,
            "message": "All modelling rows have Sector." if missing_count == 0 else "Some rows have missing Sector.",
            "details": str({"missing_count": missing_count}),
        }
