from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.abm_v3.config import ABMV3Config
from src.abm_v3.input_panel_builder import ABMV3InputPanelBuilder
from src.abm_v3.paths import ABMV3Paths

LOGGER = logging.getLogger(__name__)

CURRENT_COLUMN_INPUT_PANEL_ORIENTATION = "current_column"
CORRECTED_INPUT_PANEL_ORIENTATION = "transpose_row_fd_without_inventory"
BUILD_CORRECTED_INPUT_PANEL_COMMAND = (
    "python -m src.abm_v3.runner build-corrected-input-panel "
    "--start-year 1995 --end-year 2016 --overwrite"
)


@dataclass
class ABMV3DataLoader:
    """Load ABM v3 inputs without performing heavy transformations."""

    paths: ABMV3Paths

    def load_merged_panel(self) -> pd.DataFrame:
        LOGGER.info("Loading merged Eora-Atlas panel from %s", self.paths.eora_atlas_merged_file)
        return pd.read_parquet(self.paths.eora_atlas_merged_file)

    def load_eora_metrics_panel(self) -> pd.DataFrame:
        LOGGER.info("Loading Eora-only metrics panel from %s", self.paths.eora_metrics_panel_file)
        return pd.read_parquet(self.paths.eora_metrics_panel_file)

    def load_et_matrix(self, year: int) -> pd.DataFrame:
        path = self.paths.et_file(year)
        LOGGER.info("Loading ET matrix for %s from %s", year, path)
        return pd.read_parquet(path)

    def load_metric(self, year: int, metric_name: str) -> pd.DataFrame:
        path = self.paths.metric_file(year, metric_name)
        LOGGER.info("Loading metric %s for %s from %s", metric_name, year, path)
        return pd.read_parquet(path)

    def load_historical_panel(self, start_year: int, end_year: int) -> pd.DataFrame:
        panel = self.load_merged_panel()
        if "Year" not in panel.columns:
            LOGGER.warning("Loaded panel has no Year column; returning unfiltered data.")
            return panel
        filtered = panel[(panel["Year"] >= start_year) & (panel["Year"] <= end_year)].copy()
        LOGGER.info(
            "Historical panel filtered from %s rows to %s rows for %s-%s.",
            len(panel),
            len(filtered),
            start_year,
            end_year,
        )
        return filtered

    def load_abm_ready_historical_panel(
        self,
        start_year: int,
        end_year: int,
        config: ABMV3Config | None = None,
    ) -> pd.DataFrame:
        """Load or build the canonical ABM-ready historical panel."""

        path = self.paths.abm_v3_historical_panel_file(start_year, end_year)
        if path.exists():
            LOGGER.info("Loading ABM-ready historical panel from %s", path)
            return pd.read_parquet(path)
        LOGGER.info("ABM-ready historical panel missing at %s; building it now.", path)
        builder = ABMV3InputPanelBuilder(self.paths, config or ABMV3Config())
        return builder.build(start_year=start_year, end_year=end_year, overwrite=False)

    def input_panel_path_for_orientation(
        self,
        start_year: int,
        end_year: int,
        input_panel_orientation: str | None,
    ) -> Path:
        """Return the panel path selected by an explicit orientation option."""
        orientation = input_panel_orientation or CURRENT_COLUMN_INPUT_PANEL_ORIENTATION
        if orientation == CURRENT_COLUMN_INPUT_PANEL_ORIENTATION:
            return self.paths.abm_v3_historical_panel_file(start_year, end_year)
        if orientation == CORRECTED_INPUT_PANEL_ORIENTATION:
            return self.paths.abm_v3_corrected_historical_panel_file(start_year, end_year, orientation)
        allowed = ", ".join([CURRENT_COLUMN_INPUT_PANEL_ORIENTATION, CORRECTED_INPUT_PANEL_ORIENTATION])
        raise ValueError(f"Unknown input panel orientation '{orientation}'. Allowed orientations: {allowed}")

    def load_input_panel_for_orientation(
        self,
        start_year: int,
        end_year: int,
        input_panel_orientation: str | None,
        config: ABMV3Config | None = None,
    ) -> pd.DataFrame:
        """Load the ABM-ready panel selected by explicit orientation.

        The current-column panel preserves the old behavior and may be built
        if missing. The corrected panel is experimental and must exist so runs
        cannot silently fall back to the old convention.
        """
        orientation = input_panel_orientation or CURRENT_COLUMN_INPUT_PANEL_ORIENTATION
        if orientation == CURRENT_COLUMN_INPUT_PANEL_ORIENTATION:
            return self.load_abm_ready_historical_panel(start_year, end_year, config)
        if orientation != CORRECTED_INPUT_PANEL_ORIENTATION:
            self.input_panel_path_for_orientation(start_year, end_year, orientation)

        path = self.input_panel_path_for_orientation(start_year, end_year, orientation)
        if not path.exists():
            raise FileNotFoundError(
                "Corrected ABM v3 input panel is missing for orientation "
                f"'{orientation}': {path}. Run: {BUILD_CORRECTED_INPUT_PANEL_COMMAND}"
            )
        LOGGER.info("Loading corrected ABM-ready historical panel from %s", path)
        return pd.read_parquet(path)
