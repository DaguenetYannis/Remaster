from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.abm_v3.config import ABMV3Config
from src.abm_v3.input_panel_builder import ABMV3InputPanelBuilder
from src.abm_v3.paths import ABMV3Paths

LOGGER = logging.getLogger(__name__)


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
