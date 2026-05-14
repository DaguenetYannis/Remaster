from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Any

import polars as pl

from src.abm_v4.config import EmissionsConfig
from src.abm_v4.paths import ABMV4Paths


ALLOWED_EMISSIONS_TRANSITION_MODES = {"frontier_gap_readiness", "legacy_raw_log"}


@dataclass(frozen=True)
class EmissionsDecomposition:
    """Aggregate emissions decomposition for one transition."""

    emissions_total_t: float
    emissions_total_t_plus_1: float
    delta_emissions_total: float
    emissions_intensity_effect: float
    production_scale_effect: float
    interaction_effect: float
    residual: float
    aggregate_output_loss_pct: float
    bad_transition_flag: bool


@dataclass(frozen=True)
class EmissionsCalibrationResult:
    """Artifacts from emissions-transition calibration diagnostics."""

    dataset: pl.DataFrame
    search_results: pl.DataFrame
    best_parameters: dict[str, Any]
    validation_summary: pl.DataFrame
    by_sector: pl.DataFrame
    by_capability_source: pl.DataFrame
    model_comparison: pl.DataFrame
    parameter_plausibility: pl.DataFrame
    markdown: str


def emissions_identity(output: float, emissions_intensity: float) -> float:
    """Return emissions from output and emissions intensity."""
    return output * emissions_intensity


def next_emissions_intensity(
    emissions_intensity: float,
    green_capability: float,
    network_green_exposure: float,
    general_capability: float,
    brown_centrality: float,
    config: EmissionsConfig,
) -> float:
    """Legacy raw-log EI update retained for comparison diagnostics."""
    reduction_rate = (
        config.beta_0
        + config.beta_log_ei * math.log(emissions_intensity)
        + config.beta_green_capability * green_capability
        + config.beta_network_green_exposure * network_green_exposure
        + config.beta_general_capability * general_capability
        - config.beta_brown_centrality * brown_centrality
    )
    return max(config.ei_min, emissions_intensity * math.exp(-reduction_rate))


def _safe_sum(series: pl.Series) -> float:
    """Return a float sum, using zero for an empty or all-null series."""
    value = series.sum()
    return 0.0 if value is None else float(value)


def _share_true(series: pl.Series) -> float:
    """Return the share of truthy values in a boolean series."""
    if len(series) == 0:
        return 0.0
    true_count = series.fill_null(False).sum()
    return float(true_count or 0) / len(series)


class EmissionsTransitionEngine:
    """Build one-step emissions-intensity transition diagnostics."""

    def __init__(
        self,
        paths: ABMV4Paths,
        start_year: int = 1995,
        end_year: int = 2016,
        config: EmissionsConfig | None = None,
        transition_mode: str | None = None,
    ) -> None:
        self.paths = paths
        self.start_year = start_year
        self.end_year = end_year
        self.config = config or EmissionsConfig()
        self.transition_mode = transition_mode or self.config.emissions_transition_mode
        if self.transition_mode not in ALLOWED_EMISSIONS_TRANSITION_MODES:
            raise ValueError(
                f"Unsupported emissions transition mode: {self.transition_mode}. "
                f"Allowed values: {sorted(ALLOWED_EMISSIONS_TRANSITION_MODES)}"
            )
        if self.config.ei_frontier_group != "sector_year":
            raise ValueError("Only ei_frontier_group='sector_year' is implemented.")

    def load_state_panel(self) -> pl.DataFrame:
        """Load the ABM v4 state panel."""
        state_path = self.paths.state_panel_path(self.start_year, self.end_year)
        if not state_path.exists():
            raise FileNotFoundError(f"ABM v4 state panel not found: {state_path}")
        return pl.read_parquet(state_path)

    def load_capability_update_panel(self) -> pl.DataFrame:
        """Load one-step capability updates."""
        if not self.paths.capability_update_panel_path.exists():
            raise FileNotFoundError(
                f"Capability update panel not found: {self.paths.capability_update_panel_path}"
            )
        return pl.read_parquet(self.paths.capability_update_panel_path)

    def load_capability_exposure_panel(self) -> pl.DataFrame:
        """Load capability exposures used in readiness and green exposure terms."""
        if not self.paths.capability_exposure_panel_path.exists():
            raise FileNotFoundError(
                f"Capability exposure panel not found: {self.paths.capability_exposure_panel_path}"
            )
        return pl.read_parquet(self.paths.capability_exposure_panel_path)

    def load_production_feasibility_panel(self) -> pl.DataFrame:
        """Load one-step production feasibility diagnostics."""
        if not self.paths.production_feasibility_panel_path.exists():
            raise FileNotFoundError(
                f"Production feasibility panel not found: {self.paths.production_feasibility_panel_path}"
            )
        return pl.read_parquet(self.paths.production_feasibility_panel_path)

    def load_supplier_weights(self) -> pl.DataFrame | None:
        """Load updated supplier weights if available."""
        if not self.paths.supplier_updated_weights_path.exists():
            return None
        return pl.read_parquet(self.paths.supplier_updated_weights_path)

    def prepare_latest_valid_state(
        self,
        state_panel: pl.DataFrame | None = None,
        year: int | None = None,
    ) -> pl.DataFrame:
        """Return one state row per node for the selected year, preserving invalid EI flags."""
        state_panel = self.load_state_panel() if state_panel is None else state_panel
        selected_year = year or max(state_panel["Year"].drop_nulls().to_list())
        select_exprs: list[pl.Expr | str] = [
            "country_sector",
            pl.col("Year").alias("year"),
            pl.col("Sector") if "Sector" in state_panel.columns else pl.lit(None).alias("Sector"),
            "X_observed",
            "EI",
            pl.coalesce([pl.col("log_EI"), pl.col("EI").log()]).alias("log_EI")
            if "log_EI" in state_panel.columns
            else pl.col("EI").log().alias("log_EI"),
            pl.col("brown_centrality")
            if "brown_centrality" in state_panel.columns
            else pl.lit(0.0).alias("brown_centrality"),
        ]
        return (
            state_panel.filter(pl.col("Year") == selected_year)
            .select(select_exprs)
            .unique(subset=["country_sector"])
            .with_columns(
                (
                    pl.col("EI").is_null()
                    | (pl.col("EI") <= 0)
                    | pl.col("X_observed").is_null()
                ).alias("invalid_EI_flag")
            )
            .with_columns(
                pl.when(pl.col("invalid_EI_flag"))
                .then(None)
                .otherwise(pl.coalesce([pl.col("log_EI"), pl.col("EI").log()]))
                .alias("log_EI")
            )
        )

    def compute_historical_rEI(
        self,
        state_panel: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Compute observed historical rEI = log(EI_t) - log(EI_t+1)."""
        state_panel = self.load_state_panel() if state_panel is None else state_panel
        sector_expr = (
            pl.col("Sector")
            if "Sector" in state_panel.columns
            else pl.lit(None, dtype=pl.Utf8).alias("Sector")
        )
        observed = (
            state_panel.select("country_sector", "Year", "EI", sector_expr)
            .sort(["country_sector", "Year"])
            .with_columns(
                pl.col("Year").shift(-1).over("country_sector").alias("_next_year"),
                pl.col("EI").shift(-1).over("country_sector").alias("_next_EI"),
            )
            .filter(
                (pl.col("_next_year") == pl.col("Year") + 1)
                & (pl.col("EI") > 0)
                & (pl.col("_next_EI") > 0)
            )
            .with_columns(
                (pl.col("EI").log() - pl.col("_next_EI").log()).alias("rEI_observed"),
                (pl.col("Year") + 1).alias("next_year"),
            )
            .select(
                "country_sector",
                "Sector",
                pl.col("Year").alias("year"),
                "next_year",
                "EI",
                pl.col("_next_EI").alias("EI_next_observed"),
                "rEI_observed",
            )
        )
        return observed

    def compute_sector_frontiers(
        self,
        latest_state: pl.DataFrame,
    ) -> pl.DataFrame:
        """Compute sector-year lower-carbon EI frontiers with global fallback."""
        valid = latest_state.filter(~pl.col("invalid_EI_flag") & (pl.col("EI") > 0))
        global_frontier = valid["EI"].quantile(self.config.ei_frontier_quantile)
        if global_frontier is None:
            global_frontier = self.config.ei_min
        frontier = (
            valid.group_by("Sector")
            .agg(
                pl.len().alias("valid_frontier_nodes"),
                pl.col("EI")
                .quantile(self.config.ei_frontier_quantile)
                .alias("sector_EI_frontier"),
            )
            .with_columns(
                (pl.col("valid_frontier_nodes") < self.config.min_frontier_nodes).alias(
                    "frontier_fallback_used"
                )
            )
            .with_columns(
                pl.when(pl.col("frontier_fallback_used"))
                .then(pl.lit(float(global_frontier)))
                .otherwise(pl.col("sector_EI_frontier"))
                .alias("EI_frontier")
            )
            .select(
                "Sector",
                "valid_frontier_nodes",
                "sector_EI_frontier",
                "frontier_fallback_used",
                "EI_frontier",
            )
        )
        return latest_state.join(frontier, on="Sector", how="left").with_columns(
            pl.when(pl.col("EI_frontier").is_null() & ~pl.col("invalid_EI_flag"))
            .then(pl.lit(float(global_frontier)))
            .otherwise(pl.col("EI_frontier"))
            .alias("EI_frontier"),
            (
                pl.col("frontier_fallback_used").fill_null(False)
                | (pl.col("valid_frontier_nodes").is_null() & ~pl.col("invalid_EI_flag"))
            ).alias("frontier_fallback_used"),
        )

    def compute_ei_gap(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Compute non-negative log distance to the sector frontier."""
        return panel.with_columns(
            pl.when(~pl.col("invalid_EI_flag") & (pl.col("EI_frontier") > 0))
            .then(pl.col("EI_frontier").log())
            .otherwise(None)
            .alias("log_EI_frontier")
        ).with_columns(
            pl.when(~pl.col("invalid_EI_flag") & pl.col("log_EI_frontier").is_not_null())
            .then(
                pl.max_horizontal(
                    pl.lit(0.0),
                    pl.col("log_EI") - pl.col("log_EI_frontier"),
                )
            )
            .otherwise(None)
            .alias("ei_gap")
        )

    def compute_sector_background_trend(
        self,
        historical_rEI: pl.DataFrame,
        sectors: pl.Series | list[str | None],
    ) -> pl.DataFrame:
        """Compute clipped median historical rEI by sector with fallback."""
        sector_values = list(dict.fromkeys([s for s in list(sectors) if s is not None]))
        global_median = historical_rEI["rEI_observed"].median() if historical_rEI.height else None
        fallback = (
            self.config.sector_background_fallback
            if global_median is None
            else float(global_median)
        )
        fallback = max(-0.03, min(0.05, fallback))
        if not self.config.use_sector_background_trend or historical_rEI.is_empty():
            return pl.DataFrame(
                {
                    "Sector": sector_values,
                    "valid_observations": [0 for _ in sector_values],
                    "median_rEI_observed": [None for _ in sector_values],
                    "background_trend_used": [fallback for _ in sector_values],
                    "sector_background_fallback_used": [True for _ in sector_values],
                }
            )

        background = (
            historical_rEI.group_by("Sector")
            .agg(
                pl.len().alias("valid_observations"),
                pl.col("rEI_observed").median().alias("median_rEI_observed"),
            )
            .with_columns(
                (pl.col("valid_observations") < self.config.min_frontier_nodes).alias(
                    "sector_background_fallback_used"
                )
            )
            .with_columns(
                pl.when(pl.col("sector_background_fallback_used"))
                .then(pl.lit(fallback))
                .otherwise(pl.col("median_rEI_observed").clip(-0.03, 0.05))
                .alias("background_trend_used")
            )
        )
        universe = pl.DataFrame({"Sector": sector_values})
        return (
            universe.join(background, on="Sector", how="left")
            .with_columns(
                pl.col("valid_observations").fill_null(0),
                pl.col("background_trend_used").fill_null(fallback),
                pl.col("sector_background_fallback_used").fill_null(True),
            )
            .select(
                "Sector",
                "valid_observations",
                "median_rEI_observed",
                "background_trend_used",
                "sector_background_fallback_used",
            )
        )

    def compute_supplier_lockin(
        self,
        latest_state: pl.DataFrame,
        supplier_weights: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Compute supplier lock-in as Herfindahl concentration of updated weights."""
        supplier_weights = self.load_supplier_weights() if supplier_weights is None else supplier_weights
        if supplier_weights is None or supplier_weights.is_empty():
            return latest_state.select("country_sector").with_columns(
                pl.lit(0.0).alias("supplier_lockin"),
                pl.lit(True).alias("supplier_lockin_fallback_used"),
            )
        lockin = (
            supplier_weights.group_by("buyer_country_sector")
            .agg((pl.col("updated_weight") ** 2).sum().alias("supplier_lockin"))
            .rename({"buyer_country_sector": "country_sector"})
            .with_columns(pl.col("supplier_lockin").clip(0.0, 1.0))
        )
        return (
            latest_state.select("country_sector")
            .join(lockin, on="country_sector", how="left")
            .with_columns(
                pl.col("supplier_lockin").is_null().alias("supplier_lockin_fallback_used"),
                pl.col("supplier_lockin").fill_null(0.0),
            )
        )

    def compute_transition_readiness(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Compute bounded readiness to close the EI frontier gap."""
        return (
            panel.with_columns(
                (
                    pl.lit(self.config.theta_intercept)
                    + pl.lit(self.config.theta_gcap) * pl.col("gcap_next").fill_null(0.0)
                    + pl.lit(self.config.theta_cap) * pl.col("cap_next").fill_null(0.0)
                    + pl.lit(self.config.theta_network_green)
                    * pl.col("network_green_exposure").fill_null(0.0)
                    + pl.lit(self.config.theta_ecosystem_exposure)
                    * pl.col("ecosystem_capability_exposure").fill_null(0.0)
                    - pl.lit(self.config.theta_brown_centrality)
                    * pl.col("brown_centrality").fill_null(0.0)
                    - pl.lit(self.config.theta_supplier_lockin)
                    * pl.col("supplier_lockin").fill_null(0.0)
                ).alias("readiness_linear")
            )
            .with_columns(
                (
                    pl.lit(self.config.rho_max)
                    / (pl.lit(1.0) + (-pl.col("readiness_linear")).exp())
                ).alias("readiness")
            )
        )

    def compute_gap_closure_potential(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Compute saturating readiness-gated EI gap closure potential."""
        return panel.with_columns(
            pl.when(pl.col("ei_gap").is_null())
            .then(None)
            .when(pl.col("ei_gap") <= 0)
            .then(0.0)
            .otherwise(
                pl.col("readiness")
                * pl.col("ei_gap")
                / (pl.col("ei_gap") + self.config.tau_gap)
            )
            .alias("gap_closure_potential")
        )

    def compute_frontier_gap_rEI(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Compute rEI from sector background trend plus readiness-gated gap closure."""
        return panel.with_columns(
            pl.when(pl.col("invalid_EI_flag"))
            .then(None)
            .otherwise(
                pl.col("sector_background_trend").fill_null(
                    self.config.sector_background_fallback
                )
                + pl.col("gap_closure_potential").fill_null(0.0)
            )
            .alias("rEI_frontier_gap")
        )

    def compute_legacy_raw_log_rEI(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Compute the previous raw-log behavioural rEI for comparison only."""
        return panel.with_columns(
            pl.when(~pl.col("invalid_EI_flag"))
            .then(
                pl.lit(self.config.beta_0)
                + pl.lit(self.config.beta_log_ei) * pl.col("log_EI")
                + pl.lit(self.config.beta_green_capability)
                * pl.col("gcap_next").fill_null(0.0)
                + pl.lit(self.config.beta_network_green_exposure)
                * pl.col("network_green_exposure").fill_null(0.0)
                + pl.lit(self.config.beta_general_capability)
                * pl.col("cap_next").fill_null(0.0)
                - pl.lit(self.config.beta_brown_centrality)
                * pl.col("brown_centrality").fill_null(0.0)
            )
            .otherwise(None)
            .alias("rEI_legacy_raw_log")
        ).with_columns(
            pl.when(~pl.col("invalid_EI_flag"))
            .then(pl.col("EI") * (-pl.col("rEI_legacy_raw_log")).exp())
            .otherwise(None)
            .alias("EI_next_legacy_raw_log")
        )

    def apply_rEI_bounds(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Apply optional rEI clipping while preserving raw rEI."""
        if not self.config.clip_rEI:
            return panel.with_columns(
                pl.col("rEI_raw").alias("rEI_used"),
                pl.lit(False).alias("rEI_clipped_low_flag"),
                pl.lit(False).alias("rEI_clipped_high_flag"),
            )
        return panel.with_columns(
            (
                pl.col("rEI_raw").is_not_null()
                & (pl.col("rEI_raw") < self.config.rEI_min)
            ).alias("rEI_clipped_low_flag"),
            (
                pl.col("rEI_raw").is_not_null()
                & (pl.col("rEI_raw") > self.config.rEI_max)
            ).alias("rEI_clipped_high_flag"),
            pl.col("rEI_raw")
            .clip(self.config.rEI_min, self.config.rEI_max)
            .alias("rEI_used"),
        )

    def compute_emissions_update(
        self,
        panel: pl.DataFrame,
    ) -> pl.DataFrame:
        """Compute EI_next and current/next emissions."""
        return (
            panel.with_columns(
                pl.when(~pl.col("invalid_EI_flag"))
                .then(pl.col("EI") * (-pl.col("rEI_used")).exp())
                .otherwise(None)
                .alias("_EI_next_raw")
            )
            .with_columns(
                pl.when(pl.col("_EI_next_raw").is_null())
                .then(None)
                .otherwise(
                    pl.max_horizontal(pl.lit(self.config.ei_min), pl.col("_EI_next_raw"))
                )
                .alias("EI_next")
            )
            .with_columns(
                (
                    pl.col("_EI_next_raw").is_not_null()
                    & (pl.col("_EI_next_raw") < self.config.ei_min)
                ).alias("EI_clipped_flag")
            )
        )

    def compute_emissions_decomposition(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Compute node-level emissions decomposition terms."""
        return (
            panel.with_columns(
                (pl.col("X_observed") * pl.col("EI")).alias("emissions_observed_current"),
                (pl.col("X_feasible") * pl.col("EI")).alias(
                    "emissions_feasible_current_EI"
                ),
                (pl.col("X_feasible") * pl.col("EI_next")).alias(
                    "emissions_feasible_next_EI"
                ),
                (pl.col("X_feasible") - pl.col("X_observed")).alias("_delta_X"),
                (pl.col("EI_next") - pl.col("EI")).alias("_delta_EI"),
            )
            .with_columns(
                (pl.col("EI") * pl.col("_delta_X")).alias("production_scale_effect"),
                (pl.col("X_observed") * pl.col("_delta_EI")).alias(
                    "emissions_intensity_effect"
                ),
                (pl.col("_delta_X") * pl.col("_delta_EI")).alias("interaction_effect"),
            )
            .with_columns(
                (
                    (pl.col("emissions_feasible_next_EI") - pl.col("emissions_observed_current"))
                    - (
                        pl.col("production_scale_effect")
                        + pl.col("emissions_intensity_effect")
                        + pl.col("interaction_effect")
                    )
                ).alias("decomposition_residual_node")
            )
        )

    def build_emissions_update_panel(
        self,
        year: int | None = None,
        transition_mode: str | None = None,
    ) -> pl.DataFrame:
        """Build one-step emissions update and node-level decomposition."""
        mode = transition_mode or self.transition_mode
        if mode not in ALLOWED_EMISSIONS_TRANSITION_MODES:
            raise ValueError(f"Unsupported emissions transition mode: {mode}")

        state_panel = self.load_state_panel()
        latest_state = self.prepare_latest_valid_state(state_panel=state_panel, year=year)
        historical_rEI = self.compute_historical_rEI(state_panel)
        frontiers = self.compute_ei_gap(self.compute_sector_frontiers(latest_state))
        background = self.compute_sector_background_trend(
            historical_rEI=historical_rEI,
            sectors=frontiers["Sector"],
        )
        supplier_lockin = self.compute_supplier_lockin(frontiers)
        capability = self.load_capability_update_panel().select(
            "country_sector",
            "cap_next",
            "gcap_next",
        )
        production = self.load_production_feasibility_panel().select(
            "country_sector",
            "X_feasible",
        )
        exposure_columns = ["country_sector"]
        exposure_panel = self.load_capability_exposure_panel()
        for column_name in (
            "network_green_exposure",
            "ecosystem_capability_exposure",
        ):
            if column_name in exposure_panel.columns:
                exposure_columns.append(column_name)
        exposure = exposure_panel.select(exposure_columns)
        if "network_green_exposure" not in exposure.columns:
            exposure = exposure.with_columns(pl.lit(None).alias("network_green_exposure"))
        if "ecosystem_capability_exposure" not in exposure.columns:
            exposure = exposure.with_columns(
                pl.lit(None).alias("ecosystem_capability_exposure")
            )

        panel = (
            frontiers.join(background, on="Sector", how="left")
            .join(supplier_lockin, on="country_sector", how="left")
            .join(capability, on="country_sector", how="left")
            .join(production, on="country_sector", how="left")
            .join(exposure, on="country_sector", how="left")
            .with_columns(
                (
                    pl.col("invalid_EI_flag")
                    | pl.col("X_feasible").is_null()
                ).alias("invalid_EI_flag"),
                pl.col("background_trend_used")
                .fill_null(self.config.sector_background_fallback)
                .alias("sector_background_trend"),
                pl.col("sector_background_fallback_used").fill_null(True),
            )
        )
        panel = self.compute_transition_readiness(panel)
        panel = self.compute_gap_closure_potential(panel)
        panel = self.compute_frontier_gap_rEI(panel)
        panel = self.compute_legacy_raw_log_rEI(panel)
        panel = panel.with_columns(
            pl.when(pl.lit(mode) == "legacy_raw_log")
            .then(pl.col("rEI_legacy_raw_log"))
            .otherwise(pl.col("rEI_frontier_gap"))
            .alias("rEI_raw")
        )
        panel = self.apply_rEI_bounds(panel)
        panel = self.compute_emissions_update(panel)
        panel = self.compute_emissions_decomposition(panel)
        return panel.select(
            "country_sector",
            "year",
            "Sector",
            "X_observed",
            "X_feasible",
            "EI",
            "log_EI",
            "EI_frontier",
            "log_EI_frontier",
            "ei_gap",
            "cap_next",
            "gcap_next",
            "network_green_exposure",
            "ecosystem_capability_exposure",
            "brown_centrality",
            "supplier_lockin",
            "readiness_linear",
            "readiness",
            "gap_closure_potential",
            "sector_background_trend",
            "rEI_raw",
            "rEI_used",
            pl.col("rEI_used").alias("rEI"),
            "rEI_clipped_low_flag",
            "rEI_clipped_high_flag",
            "EI_next",
            "invalid_EI_flag",
            "frontier_fallback_used",
            "sector_background_fallback_used",
            "supplier_lockin_fallback_used",
            "emissions_observed_current",
            "emissions_feasible_current_EI",
            "emissions_feasible_next_EI",
            "production_scale_effect",
            "emissions_intensity_effect",
            "interaction_effect",
            "decomposition_residual_node",
            "rEI_legacy_raw_log",
            "EI_next_legacy_raw_log",
            "EI_clipped_flag",
        )

    def build_historical_rEI_summary(
        self,
        historical_rEI: pl.DataFrame,
    ) -> pl.DataFrame:
        """Build a one-row summary of historical observed rEI."""
        if historical_rEI.is_empty():
            return pl.DataFrame(
                {
                    "count": [0],
                    "mean": [None],
                    "median": [None],
                    "p05": [None],
                    "p25": [None],
                    "p75": [None],
                    "p95": [None],
                    "min": [None],
                    "max": [None],
                    "share_positive": [0.0],
                    "share_negative": [0.0],
                    "share_zero_or_near_zero": [0.0],
                    "notes": ["No valid consecutive positive-EI observations."],
                }
            )
        rei = historical_rEI["rEI_observed"]
        return pl.DataFrame(
            {
                "count": [historical_rEI.height],
                "mean": [rei.mean()],
                "median": [rei.median()],
                "p05": [rei.quantile(0.05)],
                "p25": [rei.quantile(0.25)],
                "p75": [rei.quantile(0.75)],
                "p95": [rei.quantile(0.95)],
                "min": [rei.min()],
                "max": [rei.max()],
                "share_positive": [_share_true(rei > 0)],
                "share_negative": [_share_true(rei < 0)],
                "share_zero_or_near_zero": [_share_true(rei.abs() <= 1e-9)],
                "notes": [
                    "Observed historical rEI is log(EI_t) - log(EI_t+1) for consecutive positive-EI node years."
                ],
            }
        )

    def build_frontier_gap_report(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Build one-row diagnostics for sector frontier and EI gaps."""
        valid = panel.filter(~pl.col("invalid_EI_flag"))
        sectors = valid["Sector"].n_unique()
        fallback_sectors = (
            valid.filter(pl.col("frontier_fallback_used"))
            .select("Sector")
            .unique()
            .height
        )
        return pl.DataFrame(
            {
                "year": [panel["year"].max()],
                "sectors": [sectors],
                "sectors_with_valid_frontier": [sectors - fallback_sectors],
                "sectors_using_global_fallback": [fallback_sectors],
                "mean_ei_gap": [valid["ei_gap"].mean()],
                "median_ei_gap": [valid["ei_gap"].median()],
                "share_zero_gap": [_share_true(valid["ei_gap"].fill_null(0.0) <= 1e-12)],
                "max_ei_gap": [valid["ei_gap"].max()],
                "frontier_quantile": [self.config.ei_frontier_quantile],
                "notes": [
                    "Sector-year EI frontier uses positive EI values and global-year fallback for small sectors."
                ],
            }
        )

    def build_emissions_update_report(
        self,
        panel: pl.DataFrame,
        transition_mode: str | None = None,
    ) -> pl.DataFrame:
        """Build one-row emissions update diagnostics."""
        mode = transition_mode or self.transition_mode
        defaults: dict[str, pl.Expr] = {
            "frontier_fallback_used": pl.lit(False),
            "sector_background_fallback_used": pl.lit(False),
            "supplier_lockin_fallback_used": pl.lit(False),
            "ei_gap": pl.lit(None, dtype=pl.Float64),
            "readiness": pl.lit(None, dtype=pl.Float64),
            "gap_closure_potential": pl.lit(None, dtype=pl.Float64),
            "rEI_raw": pl.col("rEI") if "rEI" in panel.columns else pl.lit(None, dtype=pl.Float64),
            "rEI_used": pl.col("rEI") if "rEI" in panel.columns else pl.lit(None, dtype=pl.Float64),
            "rEI_clipped_low_flag": pl.lit(False),
            "rEI_clipped_high_flag": pl.lit(False),
            "Sector": pl.lit(None, dtype=pl.Utf8),
        }
        for column_name, default_expr in defaults.items():
            if column_name not in panel.columns:
                panel = panel.with_columns(default_expr.alias(column_name))
        valid = panel.filter(~pl.col("invalid_EI_flag"))
        node_count = panel.height
        valid_count = valid.height
        invalid_count = node_count - valid_count
        total_current = _safe_sum(valid["emissions_observed_current"])
        total_feasible_current = _safe_sum(valid["emissions_feasible_current_EI"])
        total_next = _safe_sum(valid["emissions_feasible_next_EI"])
        production_effect = _safe_sum(valid["production_scale_effect"])
        intensity_effect = _safe_sum(valid["emissions_intensity_effect"])
        interaction_effect = _safe_sum(valid["interaction_effect"])
        aggregate_delta = total_next - total_current
        decomposition_sum = production_effect + intensity_effect + interaction_effect
        residual = aggregate_delta - decomposition_sum
        bad_transition = (
            aggregate_delta < 0
            and production_effect < 0
            and abs(production_effect) > abs(intensity_effect)
        )
        return pl.DataFrame(
            {
                "year": [panel["year"].max()],
                "emissions_transition_mode": [mode],
                "frontier_quantile": [self.config.ei_frontier_quantile],
                "node_count": [node_count],
                "valid_EI_nodes": [valid_count],
                "invalid_EI_nodes": [invalid_count],
                "sectors_using_frontier_fallback": [
                    valid.filter(pl.col("frontier_fallback_used"))
                    .select("Sector")
                    .unique()
                    .height
                ],
                "sector_background_fallback_nodes": [
                    panel["sector_background_fallback_used"].fill_null(False).sum()
                ],
                "supplier_lockin_fallback_nodes": [
                    panel["supplier_lockin_fallback_used"].fill_null(False).sum()
                ],
                "mean_ei_gap": [valid["ei_gap"].mean()],
                "median_ei_gap": [valid["ei_gap"].median()],
                "share_zero_gap": [_share_true(valid["ei_gap"].fill_null(0.0) <= 1e-12)],
                "mean_readiness": [valid["readiness"].mean()],
                "median_readiness": [valid["readiness"].median()],
                "mean_gap_closure_potential": [valid["gap_closure_potential"].mean()],
                "median_gap_closure_potential": [valid["gap_closure_potential"].median()],
                "mean_rEI_raw": [valid["rEI_raw"].mean()],
                "median_rEI_raw": [valid["rEI_raw"].median()],
                "mean_rEI_used": [valid["rEI_used"].mean()],
                "median_rEI_used": [valid["rEI_used"].median()],
                "share_negative_rEI_raw": [_share_true(valid["rEI_raw"] < 0)],
                "share_negative_rEI_used": [_share_true(valid["rEI_used"] < 0)],
                "rEI_clipped_low_count": [panel["rEI_clipped_low_flag"].fill_null(False).sum()],
                "rEI_clipped_high_count": [panel["rEI_clipped_high_flag"].fill_null(False).sum()],
                "EI_clipped_count": [panel["EI_clipped_flag"].fill_null(False).sum()],
                "total_emissions_current": [total_current],
                "total_emissions_feasible_current_EI": [total_feasible_current],
                "total_emissions_next": [total_next],
                "aggregate_delta_emissions": [aggregate_delta],
                "aggregate_production_scale_effect": [production_effect],
                "aggregate_emissions_intensity_effect": [intensity_effect],
                "aggregate_interaction_effect": [interaction_effect],
                "decomposition_residual": [residual],
                "bad_transition_flag": [bad_transition],
                "notes": [
                    (
                        "One-step emissions diagnostic. Default frontier-gap mode closes "
                        "sector-frontier EI gaps through readiness-gated conservative placeholder parameters."
                    )
                ],
            }
        )

    def build_emissions_decomposition_base(self, report: pl.DataFrame) -> pl.DataFrame:
        """Build aggregate decomposition table for the one-step diagnostic."""
        return report.select(
            "year",
            "emissions_transition_mode",
            "total_emissions_current",
            "total_emissions_next",
            "aggregate_delta_emissions",
            "aggregate_production_scale_effect",
            "aggregate_emissions_intensity_effect",
            "aggregate_interaction_effect",
            "decomposition_residual",
            "bad_transition_flag",
        )

    def build_emissions_update(
        self,
        year: int | None = None,
        transition_mode: str | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Build emissions outputs and diagnostics without writing."""
        mode = transition_mode or self.transition_mode
        state_panel = self.load_state_panel()
        historical_rEI = self.compute_historical_rEI(state_panel)
        panel = self.build_emissions_update_panel(year=year, transition_mode=mode)
        report = self.build_emissions_update_report(panel, transition_mode=mode)
        decomposition = self.build_emissions_decomposition_base(report)
        historical_summary = self.build_historical_rEI_summary(historical_rEI)
        background = self.compute_sector_background_trend(
            historical_rEI=historical_rEI,
            sectors=panel["Sector"],
        ).rename({"sector_background_fallback_used": "fallback_used"})
        frontier_gap_report = self.build_frontier_gap_report(panel)
        return panel, report, decomposition, historical_summary, background, frontier_gap_report

    def build_transition_comparison(
        self,
        year: int | None = None,
    ) -> pl.DataFrame:
        """Compare default frontier-gap mode with legacy raw-log mode."""
        rows = []
        for mode in ("frontier_gap_readiness", "legacy_raw_log"):
            panel = self.build_emissions_update_panel(year=year, transition_mode=mode)
            report = self.build_emissions_update_report(panel, transition_mode=mode)
            report_row = report.to_dicts()[0]
            rows.append(
                {
                    "mode": mode,
                    "mean_rEI_used": report_row["mean_rEI_used"],
                    "median_rEI_used": report_row["median_rEI_used"],
                    "share_negative_rEI_used": report_row["share_negative_rEI_used"],
                    "total_emissions_next": report_row["total_emissions_next"],
                    "aggregate_delta_emissions": report_row["aggregate_delta_emissions"],
                    "aggregate_EI_effect": report_row["aggregate_emissions_intensity_effect"],
                    "decomposition_residual": report_row["decomposition_residual"],
                    "notes": (
                        "frontier_gap_readiness is the default; legacy_raw_log is retained only for comparison."
                    ),
                }
            )
        return pl.DataFrame(rows)

    def write_outputs(
        self,
        panel: pl.DataFrame,
        report: pl.DataFrame,
        decomposition: pl.DataFrame,
        historical_summary: pl.DataFrame | None = None,
        sector_background: pl.DataFrame | None = None,
        frontier_gap_report: pl.DataFrame | None = None,
    ) -> None:
        """Write emissions update outputs."""
        self.paths.interim.mkdir(parents=True, exist_ok=True)
        self.paths.diagnostics.mkdir(parents=True, exist_ok=True)
        panel.write_parquet(self.paths.emissions_update_panel_path)
        report.write_csv(self.paths.emissions_update_report_path)
        decomposition.write_csv(self.paths.emissions_decomposition_base_path)
        if historical_summary is not None:
            historical_summary.write_csv(self.paths.emissions_historical_rEI_summary_path)
        if sector_background is not None:
            sector_background.write_csv(self.paths.emissions_sector_background_trend_path)
        if frontier_gap_report is not None:
            frontier_gap_report.write_csv(self.paths.emissions_frontier_gap_report_path)

    def write_transition_comparison(self, comparison: pl.DataFrame) -> None:
        """Write the emissions transition mode comparison diagnostic."""
        self.paths.diagnostics.mkdir(parents=True, exist_ok=True)
        comparison.write_csv(self.paths.emissions_transition_comparison_path)


class EmissionsUpdater(EmissionsTransitionEngine):
    """Backward-compatible name for the emissions transition engine."""


class EmissionsTransitionCalibrator:
    """Historically disciplined parameter-search scaffold for rEI transitions."""

    PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
        "rho_max": (0.005, 0.15),
        "theta_intercept": (-3.0, 1.0),
        "theta_gcap": (0.0, 3.0),
        "theta_cap": (0.0, 2.0),
        "theta_network_green": (0.0, 2.0),
        "theta_ecosystem_exposure": (0.0, 2.0),
        "theta_brown_centrality": (0.0, 2.0),
        "theta_supplier_lockin": (0.0, 2.0),
        "tau_gap": (0.1, 5.0),
    }

    def __init__(
        self,
        paths: ABMV4Paths,
        *,
        start_year: int = 1995,
        end_year: int = 2016,
        config: EmissionsConfig | None = None,
        random_search_iterations: int = 200,
        seed: int = 42,
        train_end_year: int = 2011,
        validation_start_year: int = 2012,
    ) -> None:
        self.paths = paths
        self.start_year = start_year
        self.end_year = end_year
        self.config = config or EmissionsConfig()
        self.random_search_iterations = random_search_iterations
        self.seed = seed
        self.train_end_year = train_end_year
        self.validation_start_year = validation_start_year

    def load_state_panel(self) -> pl.DataFrame:
        """Load the ABM v4 state panel."""
        state_path = self.paths.state_panel_path(self.start_year, self.end_year)
        if not state_path.exists():
            canonical = self.paths.state_panel_path(1995, 2016)
            if canonical.exists():
                return pl.read_parquet(canonical)
            raise FileNotFoundError(f"ABM v4 state panel not found: {state_path}")
        return pl.read_parquet(state_path)

    def load_supplier_weights(self) -> pl.DataFrame | None:
        """Load supplier weights for supplier-lock-in diagnostics."""
        if not self.paths.supplier_updated_weights_path.exists():
            return None
        return pl.read_parquet(self.paths.supplier_updated_weights_path)

    def build_calibration_dataset(self, state_panel: pl.DataFrame | None = None) -> pl.DataFrame:
        """Build valid node-year observations for rEI calibration."""
        state = self.load_state_panel() if state_panel is None else state_panel
        state = state.filter(pl.col("Year").is_between(self.start_year, self.end_year))
        base = self._prepare_state_features(state)
        base = self._add_sector_frontiers(base)
        base = self._add_sector_background_trend(base)
        base = self._add_supplier_lockin(base)
        return (
            base.sort(["country_sector", "year"])
            .with_columns(
                pl.col("year").shift(-1).over("country_sector").alias("next_year"),
                pl.col("EI").shift(-1).over("country_sector").alias("EI_next_observed"),
            )
            .filter(
                (pl.col("next_year") == pl.col("year") + 1)
                & (pl.col("EI") > 0)
                & (pl.col("EI_next_observed") > 0)
            )
            .with_columns(
                (pl.col("EI").log() - pl.col("EI_next_observed").log()).alias("observed_rEI")
            )
            .filter(
                pl.all_horizontal(
                    pl.col("observed_rEI").is_not_null(),
                    pl.col("ei_gap").is_not_null(),
                    pl.col("cap_model").is_not_null(),
                    pl.col("gcap_model").is_not_null(),
                    pl.col("network_green_exposure").is_not_null(),
                    pl.col("ecosystem_capability_exposure").is_not_null(),
                    pl.col("brown_centrality").is_not_null(),
                    pl.col("supplier_lockin").is_not_null(),
                )
            )
            .select(
                "country_sector",
                "year",
                "next_year",
                "Sector",
                "Country",
                "ecosystem_id",
                "observed_rEI",
                "ei_gap",
                "cap_model",
                "gcap_model",
                "general_capability_source",
                "green_capability_source",
                "network_green_exposure",
                "ecosystem_capability_exposure",
                "brown_centrality",
                "supplier_lockin",
                "sector_background_trend",
                "log_EI",
            )
        )

    def split_train_validation(self, dataset: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Split by time so validation remains out-of-sample in years."""
        return (
            dataset.filter(pl.col("year") <= self.train_end_year),
            dataset.filter(pl.col("year") >= self.validation_start_year),
        )

    def sample_parameter_sets(self) -> list[dict[str, float]]:
        """Sample bounded parameters with fixed seed and sign constraints."""
        rng = random.Random(self.seed)
        sampled: list[dict[str, float]] = [self._default_parameter_set()]
        for _ in range(max(0, self.random_search_iterations - 1)):
            sampled.append(
                {
                    name: rng.uniform(lower, upper)
                    for name, (lower, upper) in self.PARAMETER_BOUNDS.items()
                }
            )
        return sampled

    def evaluate_parameter_set(
        self,
        params: dict[str, float],
        train: pl.DataFrame,
        validation: pl.DataFrame,
        *,
        model_name: str = "frontier_gap_readiness",
    ) -> dict[str, Any]:
        """Evaluate a parameter set against train and validation splits."""
        train_pred = self.predict_rEI(train, params, model_name=model_name)
        validation_pred = self.predict_rEI(validation, params, model_name=model_name)
        return {
            **params,
            "model_name": model_name,
            **self._prefix_metrics(self.compute_metrics(train_pred), "train"),
            **self._prefix_metrics(self.compute_metrics(validation_pred), "validation"),
        }

    def run_parameter_search(self, dataset: pl.DataFrame) -> pl.DataFrame:
        """Run random search and mark the lowest validation-MAE candidate."""
        train, validation = self.split_train_validation(dataset)
        rows = [
            self.evaluate_parameter_set(params, train, validation)
            for params in self.sample_parameter_sets()
        ]
        results = pl.DataFrame(rows).sort("validation_mae").with_row_index("_row")
        return results.with_columns((pl.col("_row") == 0).alias("selected")).drop("_row")

    def select_best_parameters(self, search_results: pl.DataFrame) -> dict[str, float]:
        """Return the selected validation-MAE-minimizing parameters."""
        row = search_results.sort("validation_mae").to_dicts()[0]
        return {name: float(row[name]) for name in self.PARAMETER_BOUNDS}

    def evaluate_baseline_models(
        self,
        dataset: pl.DataFrame,
        selected_params: dict[str, float],
    ) -> pl.DataFrame:
        """Compare full frontier-gap readiness with simpler baselines."""
        train, validation = self.split_train_validation(dataset)
        rows: list[dict[str, Any]] = []
        specs = [
            ("frontier_gap_readiness", "Selected full frontier-gap readiness specification."),
            ("sector_background_only", "Uses only median historical sector rEI."),
            ("frontier_gap_only", "Uses sector background plus ungated frontier gap closure."),
            ("readiness_without_capability", "Readiness excludes general and green capability terms."),
            ("legacy_raw_log", "Legacy raw-log rule retained only as a comparison."),
        ]
        for model_name, notes in specs:
            metrics = self.evaluate_parameter_set(
                selected_params,
                train,
                validation,
                model_name=model_name,
            )
            rows.append(
                {
                    "model_name": model_name,
                    "train_mae": metrics["train_mae"],
                    "validation_mae": metrics["validation_mae"],
                    "train_rmse": metrics["train_rmse"],
                    "validation_rmse": metrics["validation_rmse"],
                    "train_bias": metrics["train_bias"],
                    "validation_bias": metrics["validation_bias"],
                    "validation_wrong_sign_share": metrics["validation_wrong_sign_share"],
                    "validation_correlation": metrics["validation_correlation"],
                    "notes": notes,
                }
            )
        return pl.DataFrame(rows).sort("validation_mae")

    def validate_best_parameters(
        self,
        dataset: pl.DataFrame,
        selected_params: dict[str, float],
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Build selected-parameter summary and grouped validation diagnostics."""
        predictions = self.predict_rEI(
            dataset,
            selected_params,
            model_name="frontier_gap_readiness",
        )
        train, validation = self.split_train_validation(predictions)
        summary = pl.DataFrame(
            [
                {"split": "train", **self.compute_metrics(train)},
                {"split": "validation", **self.compute_metrics(validation)},
            ]
        )
        by_sector = self._group_prediction_metrics(validation, ["Sector"]).sort(
            "mae", descending=True
        )
        by_source = self._capability_source_prediction_metrics(validation)
        return summary, by_sector, by_source

    def build_parameter_plausibility_report(
        self,
        selected_params: dict[str, float],
        model_comparison: pl.DataFrame,
        validation_summary: pl.DataFrame,
    ) -> pl.DataFrame:
        """Flag boundary solutions and weak mechanism evidence."""
        rows: list[dict[str, Any]] = []
        for name, value in selected_params.items():
            lower, upper = self.PARAMETER_BOUNDS[name]
            width = upper - lower
            near_lower = value <= lower + 0.05 * width
            near_upper = value >= upper - 0.05 * width
            sign_valid = value >= 0 or name == "theta_intercept"
            notes: list[str] = []
            if near_lower:
                notes.append("near lower bound")
            if near_upper:
                notes.append("near upper bound")
            if name == "rho_max":
                notes.append("annual maximum EI reduction gate")
            if name == "tau_gap" and near_lower:
                notes.append("gap closure may be too sensitive")
            if name == "tau_gap" and near_upper:
                notes.append("gap closure may be too flat")
            rows.append(
                {
                    "parameter": name,
                    "value": value,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "near_lower_bound": near_lower,
                    "near_upper_bound": near_upper,
                    "theoretical_sign_valid": sign_valid,
                    "notes": "; ".join(notes),
                }
            )
        self._append_model_plausibility_rows(rows, model_comparison, validation_summary)
        return pl.DataFrame(rows)

    def run(self) -> EmissionsCalibrationResult:
        """Run dataset construction, search, comparisons, and reports."""
        dataset = self.build_calibration_dataset()
        search_results = self.run_parameter_search(dataset)
        selected_params = self.select_best_parameters(search_results)
        model_comparison = self.evaluate_baseline_models(dataset, selected_params)
        validation_summary, by_sector, by_source = self.validate_best_parameters(
            dataset,
            selected_params,
        )
        plausibility = self.build_parameter_plausibility_report(
            selected_params,
            model_comparison,
            validation_summary,
        )
        best_json = self._build_best_parameter_json(
            selected_params,
            search_results,
            model_comparison,
            plausibility,
            dataset,
        )
        markdown = self.build_markdown_report(
            dataset,
            validation_summary,
            model_comparison,
            by_sector,
            by_source,
            plausibility,
            selected_params,
        )
        return EmissionsCalibrationResult(
            dataset=dataset,
            search_results=search_results,
            best_parameters=best_json,
            validation_summary=validation_summary,
            by_sector=by_sector,
            by_capability_source=by_source,
            model_comparison=model_comparison,
            parameter_plausibility=plausibility,
            markdown=markdown,
        )

    def write_outputs(self, result: EmissionsCalibrationResult) -> None:
        """Write calibration outputs under data/abm_v4/validation."""
        self.paths.validation.mkdir(parents=True, exist_ok=True)
        result.dataset.write_parquet(self.paths.emissions_calibration_dataset_path)
        result.search_results.write_csv(self.paths.emissions_parameter_search_results_path)
        self.paths.emissions_best_parameters_path.write_text(
            json.dumps(result.best_parameters, indent=2),
            encoding="utf-8",
        )
        result.validation_summary.write_csv(self.paths.emissions_calibration_validation_summary_path)
        result.by_sector.write_csv(self.paths.emissions_calibration_by_sector_path)
        result.by_capability_source.write_csv(
            self.paths.emissions_calibration_by_capability_source_path
        )
        result.model_comparison.write_csv(self.paths.emissions_model_comparison_path)
        result.parameter_plausibility.write_csv(self.paths.emissions_parameter_plausibility_path)
        self.paths.emissions_calibration_report_path.write_text(result.markdown, encoding="utf-8")

    def predict_rEI(
        self,
        frame: pl.DataFrame,
        params: dict[str, float],
        *,
        model_name: str,
    ) -> pl.DataFrame:
        """Add simulated rEI and prediction errors for one model."""
        readiness_linear = (
            pl.lit(params["theta_intercept"])
            + pl.lit(params["theta_gcap"]) * pl.col("gcap_model").fill_null(0.0)
            + pl.lit(params["theta_cap"]) * pl.col("cap_model").fill_null(0.0)
            + pl.lit(params["theta_network_green"]) * pl.col("network_green_exposure").fill_null(0.0)
            + pl.lit(params["theta_ecosystem_exposure"])
            * pl.col("ecosystem_capability_exposure").fill_null(0.0)
            - pl.lit(params["theta_brown_centrality"]) * pl.col("brown_centrality").fill_null(0.0)
            - pl.lit(params["theta_supplier_lockin"]) * pl.col("supplier_lockin").fill_null(0.0)
        )
        no_cap_readiness_linear = (
            pl.lit(params["theta_intercept"])
            + pl.lit(params["theta_network_green"]) * pl.col("network_green_exposure").fill_null(0.0)
            + pl.lit(params["theta_ecosystem_exposure"])
            * pl.col("ecosystem_capability_exposure").fill_null(0.0)
            - pl.lit(params["theta_brown_centrality"]) * pl.col("brown_centrality").fill_null(0.0)
            - pl.lit(params["theta_supplier_lockin"]) * pl.col("supplier_lockin").fill_null(0.0)
        )
        gap_fraction = pl.col("ei_gap") / (pl.col("ei_gap") + params["tau_gap"])
        legacy = (
            pl.lit(self.config.beta_0)
            + pl.lit(self.config.beta_log_ei) * pl.col("log_EI").fill_null(0.0)
            + pl.lit(self.config.beta_green_capability) * pl.col("gcap_model").fill_null(0.0)
            + pl.lit(self.config.beta_network_green_exposure)
            * pl.col("network_green_exposure").fill_null(0.0)
            + pl.lit(self.config.beta_general_capability) * pl.col("cap_model").fill_null(0.0)
            - pl.lit(self.config.beta_brown_centrality) * pl.col("brown_centrality").fill_null(0.0)
        )
        predicted_expr = (
            pl.when(pl.lit(model_name) == "sector_background_only")
            .then(pl.col("sector_background_trend"))
            .when(pl.lit(model_name) == "frontier_gap_only")
            .then(pl.col("sector_background_trend") + pl.lit(params["rho_max"]) * gap_fraction)
            .when(pl.lit(model_name) == "readiness_without_capability")
            .then(
                pl.col("sector_background_trend")
                + (
                    pl.lit(params["rho_max"])
                    / (pl.lit(1.0) + (-no_cap_readiness_linear).exp())
                )
                * gap_fraction
            )
            .when(pl.lit(model_name) == "legacy_raw_log")
            .then(legacy)
            .otherwise(
                pl.col("sector_background_trend")
                + (
                    pl.lit(params["rho_max"])
                    / (pl.lit(1.0) + (-readiness_linear).exp())
                )
                * gap_fraction
            )
        )
        return frame.with_columns(predicted_expr.alias("simulated_rEI")).with_columns(
            (pl.col("simulated_rEI") - pl.col("observed_rEI")).alias("rEI_error"),
            (pl.col("simulated_rEI") - pl.col("observed_rEI")).abs().alias("rEI_abs_error"),
        )

    def compute_metrics(self, predictions: pl.DataFrame) -> dict[str, float]:
        """Compute loss and diagnostic metrics."""
        if predictions.is_empty():
            return {
                "mae": float("nan"),
                "rmse": float("nan"),
                "median_abs_error": float("nan"),
                "bias": float("nan"),
                "wrong_sign_share": float("nan"),
                "correlation": float("nan"),
                "sector_weighted_mae": float("nan"),
                "rows": 0.0,
            }
        metrics = predictions.select(
            pl.col("rEI_abs_error").mean().alias("mae"),
            (pl.col("rEI_error").pow(2).mean().sqrt()).alias("rmse"),
            pl.col("rEI_abs_error").median().alias("median_abs_error"),
            pl.col("rEI_error").mean().alias("bias"),
            (
                (
                    ((pl.col("simulated_rEI") > 0) & (pl.col("observed_rEI") < 0))
                    | ((pl.col("simulated_rEI") < 0) & (pl.col("observed_rEI") > 0))
                )
                .mean()
            ).alias("wrong_sign_share"),
            pl.corr("simulated_rEI", "observed_rEI").alias("correlation"),
            pl.len().alias("rows"),
        ).to_dicts()[0]
        sector_mae = (
            predictions.group_by("Sector")
            .agg(pl.col("rEI_abs_error").mean().alias("sector_mae"))
            ["sector_mae"]
            .mean()
        )
        return {
            "mae": _clean_float(metrics["mae"]),
            "rmse": _clean_float(metrics["rmse"]),
            "median_abs_error": _clean_float(metrics["median_abs_error"]),
            "bias": _clean_float(metrics["bias"]),
            "wrong_sign_share": _clean_float(metrics["wrong_sign_share"]),
            "correlation": _clean_float(metrics["correlation"]),
            "sector_weighted_mae": _clean_float(sector_mae),
            "rows": float(metrics["rows"]),
        }

    def build_markdown_report(
        self,
        dataset: pl.DataFrame,
        validation_summary: pl.DataFrame,
        model_comparison: pl.DataFrame,
        by_sector: pl.DataFrame,
        by_source: pl.DataFrame,
        plausibility: pl.DataFrame,
        selected_params: dict[str, float],
    ) -> str:
        """Render a calibration report with train/validation and baseline comparisons."""
        validation = validation_summary.filter(pl.col("split") == "validation").to_dicts()[0]
        lines = [
            "# ABM v4 Emissions-Transition Calibration Scaffold",
            "",
            "This is historically disciplined parameter selection, not structural estimation.",
            "",
            "## Sample",
            "",
            f"- Calibration rows: {dataset.height}",
            f"- Train years: {self.start_year}-{self.train_end_year}",
            f"- Validation years: {self.validation_start_year}-{self.end_year}",
            "",
            "## Selected Validation Metrics",
            "",
            f"- Validation MAE: {validation['mae']}",
            f"- Validation bias: {validation['bias']}",
            f"- Validation wrong-sign share: {validation['wrong_sign_share']}",
            f"- Validation correlation: {validation['correlation']}",
            "",
            "## Best Parameters",
            "",
            self._markdown_table(
                pl.DataFrame([selected_params]).transpose(
                    include_header=True,
                    header_name="parameter",
                    column_names=["value"],
                )
            ),
            "",
            "## Model Comparison",
            "",
            self._markdown_table(model_comparison),
            "",
            "## Parameter Plausibility",
            "",
            self._markdown_table(plausibility),
            "",
            "## Error by Sector",
            "",
            self._markdown_table(by_sector.head(10)),
            "",
            "## Error by Capability Source",
            "",
            self._markdown_table(by_source),
            "",
            "## Interpretation",
            "",
            "Use these parameters as candidates for a calibrated historical run only after checking "
            "whether the full frontier-gap readiness model materially improves over the baselines. "
            "Do not treat them as causal estimates or scenario policy parameters.",
        ]
        return "\n".join(lines) + "\n"

    def _prepare_state_features(self, state: pl.DataFrame) -> pl.DataFrame:
        cap_expr = (
            pl.col("general_capability_model")
            if "general_capability_model" in state.columns
            else pl.col("general_capability")
        )
        gcap_expr = (
            pl.col("green_capability_model")
            if "green_capability_model" in state.columns
            else pl.col("green_capability")
        )
        network_expr = (
            pl.col("network_green_exposure")
            if "network_green_exposure" in state.columns
            else pl.col("g_local_v4")
            if "g_local_v4" in state.columns
            else pl.lit(0.0)
        )
        brown_expr = (
            pl.col("brown_centrality") if "brown_centrality" in state.columns else pl.lit(0.0)
        )
        selected = state.select(
            "country_sector",
            pl.col("Year").alias("year"),
            "Country",
            "Sector",
            (
                pl.col("ecosystem_id")
                if "ecosystem_id" in state.columns
                else pl.lit("missing")
            ).alias("ecosystem_id"),
            "EI",
            pl.when(pl.col("EI") > 0).then(pl.col("EI").log()).otherwise(None).alias("log_EI"),
            cap_expr.cast(pl.Float64).alias("cap_model"),
            gcap_expr.cast(pl.Float64).alias("gcap_model"),
            (
                pl.col("general_capability_source")
                if "general_capability_source" in state.columns
                else pl.lit("unavailable")
            ).alias("general_capability_source"),
            (
                pl.col("green_capability_source")
                if "green_capability_source" in state.columns
                else pl.lit("unavailable")
            ).alias("green_capability_source"),
            network_expr.cast(pl.Float64).alias("network_green_exposure"),
            brown_expr.cast(pl.Float64).fill_null(0.0).alias("brown_centrality"),
        )
        ecosystem_exposure = selected.group_by(["year", "ecosystem_id"]).agg(
            pl.col("cap_model").mean().alias("ecosystem_capability_exposure")
        )
        return selected.join(ecosystem_exposure, on=["year", "ecosystem_id"], how="left")

    def _add_sector_frontiers(self, frame: pl.DataFrame) -> pl.DataFrame:
        valid = frame.filter(pl.col("EI") > 0)
        global_frontiers = valid.group_by("year").agg(
            pl.col("EI").quantile(self.config.ei_frontier_quantile).alias("global_EI_frontier")
        )
        sector_frontiers = (
            valid.group_by(["year", "Sector"])
            .agg(
                pl.len().alias("valid_frontier_nodes"),
                pl.col("EI").quantile(self.config.ei_frontier_quantile).alias("sector_EI_frontier"),
            )
            .join(global_frontiers, on="year", how="left")
            .with_columns(
                pl.when(pl.col("valid_frontier_nodes") < self.config.min_frontier_nodes)
                .then(pl.col("global_EI_frontier"))
                .otherwise(pl.col("sector_EI_frontier"))
                .alias("EI_frontier")
            )
            .select("year", "Sector", "EI_frontier")
        )
        return frame.join(sector_frontiers, on=["year", "Sector"], how="left").with_columns(
            pl.when((pl.col("EI") > 0) & (pl.col("EI_frontier") > 0))
            .then((pl.col("EI").log() - pl.col("EI_frontier").log()).clip(0.0, None))
            .otherwise(None)
            .alias("ei_gap")
        )

    def _add_sector_background_trend(self, frame: pl.DataFrame) -> pl.DataFrame:
        observed = (
            frame.sort(["country_sector", "year"])
            .with_columns(
                pl.col("year").shift(-1).over("country_sector").alias("_next_year"),
                pl.col("EI").shift(-1).over("country_sector").alias("_next_EI"),
            )
            .filter(
                (pl.col("_next_year") == pl.col("year") + 1)
                & (pl.col("year") <= self.train_end_year)
                & (pl.col("EI") > 0)
                & (pl.col("_next_EI") > 0)
            )
            .with_columns((pl.col("EI").log() - pl.col("_next_EI").log()).alias("_observed_rEI"))
        )
        global_median = observed["_observed_rEI"].median()
        fallback = self.config.sector_background_fallback if global_median is None else float(global_median)
        background = (
            observed.group_by("Sector")
            .agg(
                pl.len().alias("_background_observations"),
                pl.col("_observed_rEI").median().clip(-0.03, 0.05).alias("sector_background_trend"),
            )
            .with_columns(
                pl.when(pl.col("_background_observations") < self.config.min_frontier_nodes)
                .then(pl.lit(fallback))
                .otherwise(pl.col("sector_background_trend"))
                .alias("sector_background_trend")
            )
            .select("Sector", "sector_background_trend")
        )
        return frame.join(background, on="Sector", how="left").with_columns(
            pl.col("sector_background_trend").fill_null(fallback)
        )

    def _add_supplier_lockin(self, frame: pl.DataFrame) -> pl.DataFrame:
        weights = self.load_supplier_weights()
        if weights is None or weights.is_empty():
            return frame.with_columns(pl.lit(0.0).alias("supplier_lockin"))
        lockin = (
            weights.group_by("buyer_country_sector")
            .agg((pl.col("updated_weight") ** 2).sum().alias("supplier_lockin"))
            .rename({"buyer_country_sector": "country_sector"})
            .with_columns(pl.col("supplier_lockin").clip(0.0, 1.0))
        )
        return frame.join(lockin, on="country_sector", how="left").with_columns(
            pl.col("supplier_lockin").fill_null(0.0)
        )

    def _default_parameter_set(self) -> dict[str, float]:
        return {
            "rho_max": self.config.rho_max,
            "theta_intercept": self.config.theta_intercept,
            "theta_gcap": self.config.theta_gcap,
            "theta_cap": self.config.theta_cap,
            "theta_network_green": self.config.theta_network_green,
            "theta_ecosystem_exposure": self.config.theta_ecosystem_exposure,
            "theta_brown_centrality": self.config.theta_brown_centrality,
            "theta_supplier_lockin": self.config.theta_supplier_lockin,
            "tau_gap": self.config.tau_gap,
        }

    def _prefix_metrics(self, metrics: dict[str, float], prefix: str) -> dict[str, float]:
        return {f"{prefix}_{name}": value for name, value in metrics.items()}

    def _group_prediction_metrics(self, frame: pl.DataFrame, groups: list[str]) -> pl.DataFrame:
        if frame.is_empty():
            return pl.DataFrame()
        return frame.group_by(groups).agg(
            pl.len().alias("rows"),
            pl.col("rEI_abs_error").mean().alias("mae"),
            (pl.col("rEI_error").pow(2).mean().sqrt()).alias("rmse"),
            pl.col("rEI_error").mean().alias("bias"),
            (
                (
                    ((pl.col("simulated_rEI") > 0) & (pl.col("observed_rEI") < 0))
                    | ((pl.col("simulated_rEI") < 0) & (pl.col("observed_rEI") > 0))
                ).mean()
            ).alias("wrong_sign_share"),
        )

    def _capability_source_prediction_metrics(self, frame: pl.DataFrame) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        for column in ("general_capability_source", "green_capability_source"):
            if column not in frame.columns:
                continue
            summary = self._group_prediction_metrics(frame, [column]).rename(
                {column: "capability_source"}
            )
            frames.append(summary.with_columns(pl.lit(column).alias("capability_source_type")))
        return pl.concat(frames, how="diagonal") if frames else pl.DataFrame()

    def _append_model_plausibility_rows(
        self,
        rows: list[dict[str, Any]],
        model_comparison: pl.DataFrame,
        validation_summary: pl.DataFrame,
    ) -> None:
        full = model_comparison.filter(pl.col("model_name") == "frontier_gap_readiness")
        background = model_comparison.filter(pl.col("model_name") == "sector_background_only")
        no_cap = model_comparison.filter(pl.col("model_name") == "readiness_without_capability")
        if not full.is_empty() and not background.is_empty():
            improvement = background["validation_mae"].item() - full["validation_mae"].item()
            rows.append(
                {
                    "parameter": "model_improvement_vs_background",
                    "value": improvement,
                    "lower_bound": None,
                    "upper_bound": None,
                    "near_lower_bound": False,
                    "near_upper_bound": False,
                    "theoretical_sign_valid": True,
                    "notes": "weak mechanism validation" if improvement < 0.001 else "meaningful baseline improvement",
                }
            )
        if not full.is_empty() and not no_cap.is_empty():
            contribution = no_cap["validation_mae"].item() - full["validation_mae"].item()
            rows.append(
                {
                    "parameter": "capability_contribution",
                    "value": contribution,
                    "lower_bound": None,
                    "upper_bound": None,
                    "near_lower_bound": False,
                    "near_upper_bound": False,
                    "theoretical_sign_valid": True,
                    "notes": "capability terms weak" if contribution < 0.001 else "capability terms improve validation fit",
                }
            )
        validation = validation_summary.filter(pl.col("split") == "validation")
        if not validation.is_empty():
            rows.append(
                {
                    "parameter": "universal_positive_rEI_check",
                    "value": 1.0 - float(validation["wrong_sign_share"].item()),
                    "lower_bound": 0.0,
                    "upper_bound": 1.0,
                    "near_lower_bound": False,
                    "near_upper_bound": False,
                    "theoretical_sign_valid": True,
                    "notes": "review sign distribution; this is not a universal decarbonization proof",
                }
            )

    def _build_best_parameter_json(
        self,
        selected_params: dict[str, float],
        search_results: pl.DataFrame,
        model_comparison: pl.DataFrame,
        plausibility: pl.DataFrame,
        dataset: pl.DataFrame,
    ) -> dict[str, Any]:
        selected = search_results.filter(pl.col("selected")).to_dicts()[0]
        return {
            "selected_parameters": selected_params,
            "train_metrics": {
                key.removeprefix("train_"): selected[key]
                for key in selected
                if key.startswith("train_")
            },
            "validation_metrics": {
                key.removeprefix("validation_"): selected[key]
                for key in selected
                if key.startswith("validation_")
            },
            "baseline_comparison_summary": model_comparison.to_dicts(),
            "plausibility_flags": plausibility.to_dicts(),
            "run_metadata": {
                "start_year": self.start_year,
                "end_year": self.end_year,
                "train_end_year": self.train_end_year,
                "validation_start_year": self.validation_start_year,
                "random_search_iterations": self.random_search_iterations,
                "seed": self.seed,
                "calibration_rows": dataset.height,
            },
            "note": (
                "These are historically disciplined base parameters, not structural estimates "
                "and not scenario policy parameters. config.py is not overwritten."
            ),
        }

    def _markdown_table(self, frame: pl.DataFrame) -> str:
        if frame.is_empty():
            return "_No rows available._"
        columns = frame.columns
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for row in frame.to_dicts():
            lines.append(
                "| "
                + " | ".join(self._format_markdown_value(row.get(column)) for column in columns)
                + " |"
            )
        return "\n".join(lines)

    def _format_markdown_value(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        if value is None:
            return ""
        return str(value).replace("|", "/")


def _clean_float(value: Any) -> float:
    """Convert numeric values to plain floats, preserving missingness as NaN."""
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
