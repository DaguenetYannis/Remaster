from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from src.abm_v4.config import EmissionsConfig
from src.abm_v4.paths import ABMV4Paths


HISTORICAL_FRONTIER_GAP_ONLY_MODE = "historical_frontier_gap_only"
ALLOWED_EMISSIONS_TRANSITION_MODES = {
    "frontier_gap_readiness",
    "legacy_raw_log",
    HISTORICAL_FRONTIER_GAP_ONLY_MODE,
}
HISTORICAL_FRONTIER_GAP_DEFAULT_RHO = 0.03
HISTORICAL_FRONTIER_GAP_DEFAULT_TAU = 1.0


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


@dataclass(frozen=True)
class EmissionsHypothesisDiagnosticResult:
    """Artifacts from emissions-transition hypothesis diagnostics."""

    hypothesis_diagnosis: pl.DataFrame
    target_horizon_panel: pl.DataFrame
    target_horizon_summary: pl.DataFrame
    predictor_screening: pl.DataFrame
    sector_dominance: pl.DataFrame
    capability_source: pl.DataFrame
    readiness_threshold: pl.DataFrame
    frontier_specification: pl.DataFrame
    macro_shock: pl.DataFrame
    markdown: str


@dataclass(frozen=True)
class EmissionsTransitionVariantComparisonResult:
    """Artifacts from Phase 14 transition-rule variant comparison."""

    results: pl.DataFrame
    by_sector_family: pl.DataFrame
    by_capability_source: pl.DataFrame
    best_parameters: dict[str, Any]
    recommendation: pl.DataFrame
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


def load_historical_frontier_gap_parameters(
    parameter_file: Path | str | None = None,
) -> dict[str, Any]:
    """Load Phase 14 gap-only parameters, falling back to conservative defaults."""
    fallback = {
        "rho_gap": HISTORICAL_FRONTIER_GAP_DEFAULT_RHO,
        "tau_gap": HISTORICAL_FRONTIER_GAP_DEFAULT_TAU,
        "parameter_source": "fallback_conservative_defaults",
        "parameter_file": "" if parameter_file is None else str(parameter_file),
        "fallback_used": True,
    }
    if parameter_file is None:
        return fallback
    path = Path(parameter_file)
    if not path.exists():
        return fallback
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return fallback
    records = payload.get("parameter_records", {})
    preferred_keys = [
        "one_year_rEI|rolling_sector_p50",
        "one_year_rEI|sector_year_p50",
        "one_year_rEI|rolling_sector_p25",
        "one_year_rEI|sector_year_p25",
    ]
    for key in preferred_keys:
        params = records.get(key, {}).get("global_parameters", {})
        if "rho_max" in params and "tau_gap" in params:
            return {
                "rho_gap": float(params["rho_max"]),
                "tau_gap": float(params["tau_gap"]),
                "parameter_source": f"phase14_global_parameters:{key}",
                "parameter_file": str(path),
                "fallback_used": False,
            }
    return fallback


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

    def load_historical_frontier_gap_parameters(
        self,
        parameter_file: Path | str | None = None,
    ) -> dict[str, Any]:
        """Load Phase 14 frontier-gap-only parameters or report conservative fallback."""
        return load_historical_frontier_gap_parameters(parameter_file)

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

    def compute_rolling_sector_frontiers(
        self,
        state_panel: pl.DataFrame,
        year: int,
        *,
        quantile: float = 0.50,
    ) -> pl.DataFrame:
        """Compute sector frontiers using only EI observations up to the selected year."""
        valid = state_panel.filter((pl.col("Year") <= year) & (pl.col("EI") > 0))
        global_frontier = valid["EI"].quantile(quantile) if not valid.is_empty() else None
        if global_frontier is None:
            global_frontier = self.config.ei_min
        frontier = (
            valid.group_by("Sector")
            .agg(
                pl.len().alias("valid_frontier_nodes"),
                pl.col("EI").quantile(quantile).alias("sector_EI_frontier"),
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
            .select("Sector", "valid_frontier_nodes", "sector_EI_frontier", "frontier_fallback_used", "EI_frontier")
        )
        return frontier

    def compute_historical_frontier_gap_rEI(
        self,
        panel: pl.DataFrame,
        *,
        rho_gap: float,
        tau_gap: float,
    ) -> pl.DataFrame:
        """Compute sector-background plus frontier-gap-only rEI without readiness terms."""
        return panel.with_columns(
            pl.when(pl.col("invalid_EI_flag"))
            .then(None)
            .otherwise(
                pl.col("sector_background_trend").fill_null(
                    self.config.sector_background_fallback
                )
                + pl.lit(rho_gap)
                * pl.col("ei_gap").fill_null(0.0)
                / (pl.col("ei_gap").fill_null(0.0) + pl.lit(tau_gap))
            )
            .alias("rEI_historical_frontier_gap_only")
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
        parameter_file: Path | str | None = None,
    ) -> pl.DataFrame:
        """Build one-step emissions update and node-level decomposition."""
        mode = transition_mode or self.transition_mode
        if mode not in ALLOWED_EMISSIONS_TRANSITION_MODES:
            raise ValueError(f"Unsupported emissions transition mode: {mode}")

        state_panel = self.load_state_panel()
        latest_state = self.prepare_latest_valid_state(state_panel=state_panel, year=year)
        selected_year = int(latest_state["year"].max())
        historical_rEI = self.compute_historical_rEI(state_panel)
        if mode == HISTORICAL_FRONTIER_GAP_ONLY_MODE:
            rolling_frontier = self.compute_rolling_sector_frontiers(
                state_panel,
                selected_year,
                quantile=0.50,
            )
            frontiers = latest_state.join(rolling_frontier, on="Sector", how="left").with_columns(
                pl.when(pl.col("EI_frontier").is_null() & ~pl.col("invalid_EI_flag"))
                .then(pl.lit(float(self.config.ei_min)))
                .otherwise(pl.col("EI_frontier"))
                .alias("EI_frontier"),
                pl.col("frontier_fallback_used").fill_null(True),
            )
            frontiers = self.compute_ei_gap(frontiers)
        else:
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
        parameter_info = self.load_historical_frontier_gap_parameters(parameter_file)
        panel = self.compute_historical_frontier_gap_rEI(
            panel,
            rho_gap=float(parameter_info["rho_gap"]),
            tau_gap=float(parameter_info["tau_gap"]),
        )
        panel = self.compute_legacy_raw_log_rEI(panel)
        panel = panel.with_columns(
            pl.when(pl.lit(mode) == "legacy_raw_log")
            .then(pl.col("rEI_legacy_raw_log"))
            .when(pl.lit(mode) == HISTORICAL_FRONTIER_GAP_ONLY_MODE)
            .then(pl.col("rEI_historical_frontier_gap_only"))
            .otherwise(pl.col("rEI_frontier_gap"))
            .alias("rEI_raw")
        ).with_columns(
            pl.lit(float(parameter_info["rho_gap"])).alias("historical_rho_gap"),
            pl.lit(float(parameter_info["tau_gap"])).alias("historical_tau_gap"),
            pl.lit(str(parameter_info["parameter_source"])).alias("historical_parameter_source"),
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
            "rEI_historical_frontier_gap_only",
            "historical_rho_gap",
            "historical_tau_gap",
            "historical_parameter_source",
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
                "historical_rho_gap": [
                    valid["historical_rho_gap"].item(0)
                    if "historical_rho_gap" in valid.columns and not valid.is_empty()
                    else None
                ],
                "historical_tau_gap": [
                    valid["historical_tau_gap"].item(0)
                    if "historical_tau_gap" in valid.columns and not valid.is_empty()
                    else None
                ],
                "historical_parameter_source": [
                    valid["historical_parameter_source"].item(0)
                    if "historical_parameter_source" in valid.columns and not valid.is_empty()
                    else ""
                ],
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
                        "One-step emissions diagnostic. Historical frontier-gap-only mode uses "
                        "sector background plus rolling-sector p50 gap closure without readiness terms."
                        if mode == HISTORICAL_FRONTIER_GAP_ONLY_MODE
                        else "One-step emissions diagnostic. Default frontier-gap mode closes "
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
        parameter_file: Path | str | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Build emissions outputs and diagnostics without writing."""
        mode = transition_mode or self.transition_mode
        state_panel = self.load_state_panel()
        historical_rEI = self.compute_historical_rEI(state_panel)
        panel = self.build_emissions_update_panel(
            year=year,
            transition_mode=mode,
            parameter_file=parameter_file,
        )
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


class EmissionsTransitionVariantComparator:
    """Compare theory-structured emissions-transition variants without scenario outputs."""

    MODEL_VARIANTS: tuple[str, ...] = (
        "sector_background_only",
        "sector_background_plus_year_country_controls",
        "frontier_gap_only",
        "global_frontier_gap_readiness",
        "sector_family_frontier_gap_readiness",
        "gated_readiness_by_sector_signal",
        "readiness_without_capability",
        "capability_only_readiness",
    )
    FRONTIER_VARIANTS: tuple[str, ...] = (
        "sector_year_p25",
        "sector_year_p50",
        "rolling_sector_p25",
        "rolling_sector_p50",
    )
    DEFAULT_TARGETS: tuple[str, ...] = (
        "one_year_rEI",
        "smoothed_one_year_rEI",
        "three_year_rEI",
    )
    FAMILY_PARAMETER_NAMES: tuple[str, ...] = ("rho_max", "theta_intercept", "tau_gap")

    def __init__(
        self,
        paths: ABMV4Paths,
        *,
        start_year: int = 1995,
        end_year: int = 2016,
        config: EmissionsConfig | None = None,
        random_search_iterations: int = 100,
        seed: int = 42,
        train_end_year: int = 2011,
        validation_start_year: int = 2012,
        targets: list[str] | None = None,
        frontier_variants: list[str] | None = None,
        minimum_family_observations: int = 500,
    ) -> None:
        self.paths = paths
        self.start_year = start_year
        self.end_year = end_year
        self.config = config or EmissionsConfig()
        self.random_search_iterations = random_search_iterations
        self.seed = seed
        self.train_end_year = train_end_year
        self.validation_start_year = validation_start_year
        self.targets = targets
        self.frontier_variants = frontier_variants or list(self.FRONTIER_VARIANTS)
        self.minimum_family_observations = minimum_family_observations
        self._calibrator = EmissionsTransitionCalibrator(
            paths,
            start_year=start_year,
            end_year=end_year,
            config=self.config,
            random_search_iterations=random_search_iterations,
            seed=seed,
            train_end_year=train_end_year,
            validation_start_year=validation_start_year,
        )

    def run(self) -> EmissionsTransitionVariantComparisonResult:
        """Run the feasible Phase 14 comparison set in memory."""
        base = self._build_base_panel()
        target_panel = self._load_or_build_target_panel(base)
        selected_targets = self.targets or self.infer_recommended_targets(target_panel)
        dataset = self.build_variant_dataset(base, target_panel, selected_targets, self.frontier_variants)
        if dataset.is_empty():
            raise ValueError("No valid rows available for transition variant comparison.")

        improved_sectors = self.load_readiness_improved_sectors()
        rows: list[dict[str, Any]] = []
        grouped_frames: list[pl.DataFrame] = []
        source_frames: list[pl.DataFrame] = []
        parameter_records: dict[str, Any] = {}

        for target_name in selected_targets:
            for frontier_variant in self.frontier_variants:
                combo = dataset.filter(
                    (pl.col("target_name") == target_name)
                    & (pl.col("frontier_variant") == frontier_variant)
                )
                if combo.is_empty():
                    continue
                train, validation = self.split_train_validation(combo)
                baseline_pred = self.predict_with_parameters(
                    validation,
                    self._default_parameter_set(),
                    "sector_background_only",
                    improved_sectors=improved_sectors,
                )
                baseline_metrics = self.compute_metrics(baseline_pred)
                global_params = self.search_parameters(
                    train,
                    validation,
                    "global_frontier_gap_readiness",
                    improved_sectors=improved_sectors,
                )
                family_params, family_fallbacks = self.search_sector_family_parameters(
                    train,
                    validation,
                    global_params,
                    improved_sectors=improved_sectors,
                )
                parameter_records[f"{target_name}|{frontier_variant}"] = {
                    "global_parameters": global_params,
                    "sector_family_parameters": family_params,
                    "sector_family_fallbacks": family_fallbacks,
                }

                for model_variant in self.MODEL_VARIANTS:
                    params = self.search_parameters(
                        train,
                        validation,
                        model_variant,
                        improved_sectors=improved_sectors,
                    ) if model_variant in {
                        "frontier_gap_only",
                        "readiness_without_capability",
                        "capability_only_readiness",
                    } else global_params
                    train_pred = self._predict_model(
                        train,
                        train,
                        params,
                        model_variant,
                        family_params=family_params,
                        improved_sectors=improved_sectors,
                    )
                    validation_pred = self._predict_model(
                        train,
                        validation,
                        params,
                        model_variant,
                        family_params=family_params,
                        improved_sectors=improved_sectors,
                    )
                    train_metrics = self.compute_metrics(train_pred)
                    validation_metrics = self.compute_metrics(validation_pred)
                    rows.append(
                        self._result_row(
                            target_name,
                            frontier_variant,
                            model_variant,
                            train_metrics,
                            validation_metrics,
                            baseline_metrics,
                        )
                    )
                    grouped_frames.append(
                        self.group_metrics(
                            validation_pred,
                            ["sector_family"],
                            target_name,
                            frontier_variant,
                            model_variant,
                            baseline_metrics["mae"],
                        )
                    )
                    source_frames.append(
                        self.capability_source_metrics(
                            validation_pred,
                            target_name,
                            frontier_variant,
                            model_variant,
                            baseline_metrics["mae"],
                        )
                    )

        results = pl.DataFrame(rows).sort(["target_name", "frontier_variant", "validation_mae"])
        by_family = pl.concat([frame for frame in grouped_frames if not frame.is_empty()], how="diagonal")
        by_source = pl.concat([frame for frame in source_frames if not frame.is_empty()], how="diagonal")
        recommendation = self.build_recommendation(results)
        markdown = self.build_markdown_report(results, by_family, by_source, recommendation)
        return EmissionsTransitionVariantComparisonResult(
            results=results,
            by_sector_family=by_family,
            by_capability_source=by_source,
            best_parameters={
                "parameter_records": parameter_records,
                "metadata": {
                    "start_year": self.start_year,
                    "end_year": self.end_year,
                    "train_end_year": self.train_end_year,
                    "validation_start_year": self.validation_start_year,
                    "random_search_iterations": self.random_search_iterations,
                    "seed": self.seed,
                    "minimum_family_observations": self.minimum_family_observations,
                    "note": "Phase 14 diagnostics only; config.py is not overwritten.",
                },
            },
            recommendation=recommendation,
            markdown=markdown,
        )

    def write_outputs(self, result: EmissionsTransitionVariantComparisonResult) -> None:
        """Write Phase 14 comparison outputs only when explicitly called."""
        self.paths.validation.mkdir(parents=True, exist_ok=True)
        result.results.write_csv(self.paths.emissions_transition_variant_results_path)
        result.by_sector_family.write_csv(
            self.paths.emissions_transition_variant_by_sector_family_path
        )
        result.by_capability_source.write_csv(
            self.paths.emissions_transition_variant_by_capability_source_path
        )
        self.paths.emissions_transition_variant_best_parameters_path.write_text(
            json.dumps(result.best_parameters, indent=2),
            encoding="utf-8",
        )
        result.recommendation.write_csv(self.paths.emissions_transition_variant_recommendation_path)
        self.paths.emissions_transition_variant_report_path.write_text(
            result.markdown,
            encoding="utf-8",
        )

    def build_variant_dataset(
        self,
        base_panel: pl.DataFrame,
        target_panel: pl.DataFrame,
        targets: list[str],
        frontier_variants: list[str],
    ) -> pl.DataFrame:
        """Create target-frontier rows for model comparison."""
        targets_frame = target_panel.filter(pl.col("target_name").is_in(targets))
        frames: list[pl.DataFrame] = []
        for frontier_variant in frontier_variants:
            gaps = self.compute_frontier_gaps(base_panel, frontier_variant)
            frames.append(
                targets_frame.join(gaps, on=["country_sector", "year"], how="inner")
                .with_columns(
                    pl.lit(frontier_variant).alias("frontier_variant"),
                    pl.col("target").alias("observed_rEI"),
                    pl.col("sector_background_prediction").alias("sector_background_trend"),
                )
            )
        dataset = pl.concat(frames, how="diagonal") if frames else pl.DataFrame()
        return self.add_sector_family(dataset).filter(
            pl.all_horizontal(
                pl.col("observed_rEI").is_not_null(),
                pl.col("frontier_gap").is_not_null(),
                pl.col("sector_background_trend").is_not_null(),
            )
        )

    def add_sector_family(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Add broad sector-family labels using transparent sector-name rules."""
        if "Sector" not in frame.columns:
            return frame.with_columns(pl.lit("other").alias("sector_family"))
        sector = pl.col("Sector").cast(pl.Utf8).str.to_lowercase()
        return frame.with_columns(
            pl.when(sector.str.contains("agriculture|forestry|fishing|food|wood|paper"))
            .then(pl.lit("agriculture_biomass"))
            .when(sector.str.contains("mining|quarry|coal|petroleum|gas|fuel|coke"))
            .then(pl.lit("extractive_energy"))
            .when(sector.str.contains("chemical|rubber|plastic|mineral|metal|steel|recycling"))
            .then(pl.lit("materials_chemicals"))
            .when(sector.str.contains("textile|machinery|equipment|vehicle|manufactur|electrical"))
            .then(pl.lit("manufacturing"))
            .when(sector.str.contains("electricity|water|utility|infrastructure"))
            .then(pl.lit("utilities_infrastructure"))
            .when(sector.str.contains("construction|real estate"))
            .then(pl.lit("construction_real_estate"))
            .when(sector.str.contains("trade|transport|logistics|wholesale|retail|hotel|restaurant"))
            .then(pl.lit("trade_transport_logistics"))
            .when(sector.str.contains("finance|financial|business|education|research|communication|knowledge"))
            .then(pl.lit("services_knowledge_finance"))
            .when(sector.str.contains("public|health|social|household|government|community"))
            .then(pl.lit("public_social_household"))
            .otherwise(pl.lit("other"))
            .alias("sector_family")
        )

    def compute_frontier_gaps(self, base_panel: pl.DataFrame, frontier_variant: str) -> pl.DataFrame:
        """Compute sector-year or rolling-sector frontier gaps without future information."""
        if frontier_variant not in self.FRONTIER_VARIANTS:
            raise ValueError(f"Unsupported frontier variant: {frontier_variant}")
        quantile = 0.50 if frontier_variant.endswith("p50") else 0.25
        valid = base_panel.filter(pl.col("EI") > 0)
        if frontier_variant.startswith("rolling_sector"):
            rows: list[dict[str, Any]] = []
            for row in valid.select("year", "Sector").unique().sort(["Sector", "year"]).to_dicts():
                history = valid.filter(
                    (pl.col("Sector") == row["Sector"]) & (pl.col("year") <= row["year"])
                )
                rows.append(
                    {
                        "year": row["year"],
                        "Sector": row["Sector"],
                        "EI_frontier_variant": history["EI"].quantile(quantile),
                    }
                )
            frontier = pl.DataFrame(rows)
        else:
            frontier = valid.group_by(["year", "Sector"]).agg(
                pl.col("EI").quantile(quantile).alias("EI_frontier_variant")
            )
        return (
            base_panel.join(frontier, on=["year", "Sector"], how="left")
            .select(
                "country_sector",
                "year",
                pl.when((pl.col("EI") > 0) & (pl.col("EI_frontier_variant") > 0))
                .then((pl.col("EI").log() - pl.col("EI_frontier_variant").log()).clip(0.0, None))
                .otherwise(None)
                .alias("frontier_gap"),
            )
        )

    def infer_recommended_targets(self, target_panel: pl.DataFrame) -> list[str]:
        """Use Phase 13's clear target recommendation when available."""
        if self.paths.emissions_hypothesis_diagnosis_path.exists():
            diagnosis = pl.read_csv(self.paths.emissions_hypothesis_diagnosis_path)
            h1 = diagnosis.filter(pl.col("hypothesis_id") == "H1")
            if not h1.is_empty():
                interpretation = str(h1["interpretation"].item())
                for target in self.DEFAULT_TARGETS:
                    if target in interpretation and target in set(target_panel["target_name"].to_list()):
                        return [target]
        available = set(target_panel["target_name"].to_list())
        return [target for target in self.DEFAULT_TARGETS if target in available]

    def load_readiness_improved_sectors(self) -> set[str]:
        """Return sectors where Phase 13 found readiness beating sector background."""
        if not self.paths.emissions_sector_dominance_diagnostics_path.exists():
            return set()
        frame = pl.read_csv(self.paths.emissions_sector_dominance_diagnostics_path)
        if "readiness_improvement" not in frame.columns:
            return set()
        return set(
            frame.filter(pl.col("readiness_improvement") > 0)
            .select(pl.col("Sector").cast(pl.Utf8))
            ["Sector"]
            .to_list()
        )

    def split_train_validation(self, dataset: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Split observations using the Phase 14 temporal rule."""
        return (
            dataset.filter(pl.col("year") <= self.train_end_year),
            dataset.filter(pl.col("year") >= self.validation_start_year),
        )

    def search_parameters(
        self,
        train: pl.DataFrame,
        validation: pl.DataFrame,
        model_variant: str,
        *,
        improved_sectors: set[str],
    ) -> dict[str, float]:
        """Small bounded random search for one variant."""
        if model_variant in {
            "sector_background_only",
            "sector_background_plus_year_country_controls",
            "sector_family_frontier_gap_readiness",
        }:
            return self._default_parameter_set()
        best_params = self._default_parameter_set()
        best_mae = float("inf")
        rng = random.Random(self.seed + abs(hash(model_variant)) % 10000)
        samples = [best_params]
        for _ in range(max(0, self.random_search_iterations - 1)):
            samples.append(
                {
                    name: rng.uniform(lower, upper)
                    for name, (lower, upper) in self._calibrator.PARAMETER_BOUNDS.items()
                }
            )
        for params in samples:
            prediction = self.predict_with_parameters(
                validation,
                params,
                model_variant,
                improved_sectors=improved_sectors,
            )
            mae = self.compute_metrics(prediction)["mae"]
            if mae < best_mae:
                best_mae = mae
                best_params = params
        return best_params

    def search_sector_family_parameters(
        self,
        train: pl.DataFrame,
        validation: pl.DataFrame,
        global_params: dict[str, float],
        *,
        improved_sectors: set[str],
    ) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
        """Estimate lightweight family-specific parameters with fallback on small groups."""
        family_params: dict[str, dict[str, float]] = {}
        fallbacks: dict[str, str] = {}
        for family in train["sector_family"].unique().to_list():
            train_family = train.filter(pl.col("sector_family") == family)
            validation_family = validation.filter(pl.col("sector_family") == family)
            if train_family.height < self.minimum_family_observations or validation_family.is_empty():
                family_params[str(family)] = global_params
                fallbacks[str(family)] = "global_parameters_minimum_observations"
                continue
            best = global_params
            best_mae = float("inf")
            rng = random.Random(self.seed + abs(hash(str(family))) % 10000)
            for index in range(max(1, self.random_search_iterations)):
                candidate = dict(global_params)
                if index > 0:
                    for name in self.FAMILY_PARAMETER_NAMES:
                        lower, upper = self._calibrator.PARAMETER_BOUNDS[name]
                        candidate[name] = rng.uniform(lower, upper)
                pred = self.predict_with_parameters(
                    validation_family,
                    candidate,
                    "global_frontier_gap_readiness",
                    improved_sectors=improved_sectors,
                )
                mae = self.compute_metrics(pred)["mae"]
                if mae < best_mae:
                    best_mae = mae
                    best = candidate
            family_params[str(family)] = best
            fallbacks[str(family)] = "estimated"
        return family_params, fallbacks

    def predict_with_parameters(
        self,
        frame: pl.DataFrame,
        params: dict[str, float],
        model_variant: str,
        *,
        improved_sectors: set[str],
    ) -> pl.DataFrame:
        """Predict rEI for non-control variants."""
        gap_fraction = pl.col("frontier_gap") / (pl.col("frontier_gap") + params["tau_gap"])
        full_linear = self._readiness_linear(params, include_capability=True, include_network=True)
        no_cap_linear = self._readiness_linear(params, include_capability=False, include_network=True)
        cap_only_linear = self._readiness_linear(params, include_capability=True, include_network=False)
        full_readiness = pl.lit(params["rho_max"]) / (pl.lit(1.0) + (-full_linear).exp())
        no_cap_readiness = pl.lit(params["rho_max"]) / (pl.lit(1.0) + (-no_cap_linear).exp())
        cap_only_readiness = pl.lit(params["rho_max"]) / (pl.lit(1.0) + (-cap_only_linear).exp())
        improved_list = sorted(improved_sectors)
        prediction = (
            pl.when(pl.lit(model_variant) == "sector_background_only")
            .then(pl.col("sector_background_trend"))
            .when(pl.lit(model_variant) == "frontier_gap_only")
            .then(pl.col("sector_background_trend") + pl.lit(params["rho_max"]) * gap_fraction)
            .when(pl.lit(model_variant) == "readiness_without_capability")
            .then(pl.col("sector_background_trend") + no_cap_readiness * gap_fraction)
            .when(pl.lit(model_variant) == "capability_only_readiness")
            .then(pl.col("sector_background_trend") + cap_only_readiness * gap_fraction)
            .when(pl.lit(model_variant) == "gated_readiness_by_sector_signal")
            .then(
                pl.when(pl.col("Sector").cast(pl.Utf8).is_in(improved_list))
                .then(pl.col("sector_background_trend") + full_readiness * gap_fraction)
                .otherwise(
                    pl.col("sector_background_trend")
                    + pl.lit(params["rho_max"]) * gap_fraction
                )
            )
            .otherwise(pl.col("sector_background_trend") + full_readiness * gap_fraction)
        )
        return frame.with_columns(prediction.alias("simulated_rEI")).with_columns(
            (pl.col("simulated_rEI") - pl.col("observed_rEI")).alias("rEI_error"),
            (pl.col("simulated_rEI") - pl.col("observed_rEI")).abs().alias("rEI_abs_error"),
        )

    def compute_metrics(self, predictions: pl.DataFrame) -> dict[str, float]:
        """Compute Phase 14 required metrics."""
        return self._calibrator.compute_metrics(predictions)

    def group_metrics(
        self,
        predictions: pl.DataFrame,
        group_columns: list[str],
        target_name: str,
        frontier_variant: str,
        model_variant: str,
        baseline_mae: float,
    ) -> pl.DataFrame:
        """Build grouped validation metrics."""
        if predictions.is_empty():
            return pl.DataFrame()
        grouped = (
            predictions.group_by(group_columns)
            .agg(
                pl.len().alias("rows_validation"),
                pl.col("rEI_abs_error").mean().alias("validation_mae"),
                (pl.col("rEI_error").pow(2).mean().sqrt()).alias("validation_rmse"),
                pl.col("rEI_error").mean().alias("validation_bias"),
                (
                    ((pl.col("simulated_rEI") > 0) & (pl.col("observed_rEI") < 0))
                    | ((pl.col("simulated_rEI") < 0) & (pl.col("observed_rEI") > 0))
                )
                .mean()
                .alias("validation_wrong_sign_share"),
                pl.corr("simulated_rEI", "observed_rEI").alias("validation_correlation"),
                pl.col("rEI_abs_error").median().alias("validation_median_abs_error"),
            )
            .with_columns(
                pl.lit(target_name).alias("target_name"),
                pl.lit(frontier_variant).alias("frontier_variant"),
                pl.lit(model_variant).alias("model_variant"),
                (pl.lit(baseline_mae) - pl.col("validation_mae")).alias(
                    "improvement_over_sector_background"
                ),
                pl.when(pl.lit(baseline_mae) > 0)
                .then((pl.lit(baseline_mae) - pl.col("validation_mae")) / pl.lit(baseline_mae))
                .otherwise(None)
                .alias("improvement_over_sector_background_pct"),
            )
        )
        return grouped

    def capability_source_metrics(
        self,
        predictions: pl.DataFrame,
        target_name: str,
        frontier_variant: str,
        model_variant: str,
        baseline_mae: float,
    ) -> pl.DataFrame:
        """Build validation metrics by capability source and source type."""
        frames: list[pl.DataFrame] = []
        for column in ("general_capability_source", "green_capability_source"):
            if column not in predictions.columns:
                continue
            frames.append(
                self.group_metrics(
                    predictions,
                    [column],
                    target_name,
                    frontier_variant,
                    model_variant,
                    baseline_mae,
                )
                .rename({column: "capability_source"})
                .with_columns(pl.lit(column).alias("capability_source_type"))
            )
        return pl.concat(frames, how="diagonal") if frames else pl.DataFrame()

    def build_recommendation(self, results: pl.DataFrame) -> pl.DataFrame:
        """Apply Phase 14 decision rules to the comparison table."""
        if results.is_empty():
            return pl.DataFrame()
        best = results.sort("validation_mae").to_dicts()[0]
        family = self._best_result(results, "sector_family_frontier_gap_readiness")
        gated = self._best_result(results, "gated_readiness_by_sector_signal")
        background = self._best_result(results, "sector_background_only")
        p50_wins = self._frontier_family_wins(results, "p50") > self._frontier_family_wins(results, "p25")
        rolling_wins = self._frontier_prefix_wins(results, "rolling_sector") > self._frontier_prefix_wins(results, "sector_year")

        recommended = best
        interpretation = "Lowest validation MAE in the diagnostic comparison."
        next_action = "Use as diagnostic evidence only; do not create scenarios yet."
        if family and background:
            family_improvement = family["improvement_over_sector_background_pct"]
            family_sign_ok = family["validation_wrong_sign_share"] <= background["validation_wrong_sign_share"]
            if family_improvement is not None and family_improvement >= 0.05 and family_sign_ok:
                recommended = family
                interpretation = "Sector-family readiness clears the 5% MAE rule without worsening sign fit."
                next_action = "Develop a sector-family historical transition rule before scenario work."
        if gated and background and recommended.get("model_variant") != "sector_family_frontier_gap_readiness":
            if gated["improvement_over_sector_background"] > 0:
                recommended = gated
                interpretation = "Gated readiness improves over background while limiting readiness to historically signaled sectors."
                next_action = "Use gated readiness as the next diagnostic specification."
        readiness_winners = results.filter(
            pl.col("model_variant").is_in(
                [
                    "global_frontier_gap_readiness",
                    "sector_family_frontier_gap_readiness",
                    "gated_readiness_by_sector_signal",
                    "readiness_without_capability",
                    "capability_only_readiness",
                ]
            )
            & (pl.col("improvement_over_sector_background") > 0)
        )
        if readiness_winners.is_empty() and background:
            recommended = self._best_result(results, "frontier_gap_only") or background
            interpretation = (
                "Readiness variants do not beat sector background; keep a conservative "
                "sector-background plus frontier-gap historical rule."
            )
            next_action = "Defer readiness to selected sectors or scenarios after stronger evidence."
        parameter_plausibility = (
            "plausible diagnostic"
            if recommended.get("improvement_over_sector_background_pct", 0.0) >= 0
            else "weak historical mechanism evidence"
        )
        overfit_risk = (
            "medium" if recommended.get("model_variant") == "sector_family_frontier_gap_readiness" else "low"
        )
        frontier_note = []
        if p50_wins:
            frontier_note.append("p50 frontiers outperform p25 more often")
        if rolling_wins:
            frontier_note.append("rolling frontiers outperform sector-year frontiers more often")
        if frontier_note:
            interpretation = f"{interpretation} {'; '.join(frontier_note)}."
        return pl.DataFrame(
            [
                {
                    "recommended_model_variant": recommended["model_variant"],
                    "recommended_target": recommended["target_name"],
                    "recommended_frontier": recommended["frontier_variant"],
                    "validation_mae": recommended["validation_mae"],
                    "improvement_over_sector_background_pct": recommended[
                        "improvement_over_sector_background_pct"
                    ],
                    "wrong_sign_share": recommended["validation_wrong_sign_share"],
                    "correlation": recommended["validation_correlation"],
                    "parameter_plausibility": parameter_plausibility,
                    "overfitting_risk": overfit_risk,
                    "interpretation": interpretation,
                    "recommended_next_action": next_action,
                }
            ]
        )

    def build_markdown_report(
        self,
        results: pl.DataFrame,
        by_family: pl.DataFrame,
        by_source: pl.DataFrame,
        recommendation: pl.DataFrame,
    ) -> str:
        """Render Phase 14 comparison report."""
        lines = [
            "# ABM v4 Phase 14 Emissions-Transition Variant Comparison",
            "",
            "This is a diagnostic comparison of transition-rule structures, not a policy scenario phase.",
            "",
            "## Phase 13 Recap",
            "",
            "- Sector background dominated the historical signal.",
            "- Readiness improved only a small share of sectors.",
            "- The p50 frontier looked more stable than the active p25 frontier.",
            "- Medium-run and smoothed targets reduced volatility but did not make readiness clearly active.",
            "- Year and country shocks matter, so controls are diagnostics only.",
            "- Scenarios remain premature.",
            "",
            "## Recommendation",
            "",
            self._markdown_table(recommendation),
            "",
            "## Variant Results",
            "",
            self._markdown_table(results.head(30)),
            "",
            "## By Sector Family",
            "",
            self._markdown_table(by_family.head(30)),
            "",
            "## By Capability Source",
            "",
            self._markdown_table(by_source.head(30)),
            "",
            "## Phase 15 Recommendation",
            "",
            "Use the recommended historical transition structure as a calibration candidate, keep historical controls out of scenario mechanisms, and postpone scenarios until the selected rule is validated in a multi-year base run.",
        ]
        return "\n".join(lines) + "\n"

    def _build_base_panel(self) -> pl.DataFrame:
        diagnostics = EmissionsTransitionHypothesisDiagnostics(
            self.paths,
            start_year=self.start_year,
            end_year=self.end_year,
            config=self.config,
            train_end_year=self.train_end_year,
            validation_start_year=self.validation_start_year,
        )
        return diagnostics.build_base_transition_panel()

    def _load_or_build_target_panel(self, base: pl.DataFrame) -> pl.DataFrame:
        if self.paths.emissions_target_horizon_panel_path.exists():
            return pl.read_parquet(self.paths.emissions_target_horizon_panel_path)
        diagnostics = EmissionsTransitionHypothesisDiagnostics(
            self.paths,
            start_year=self.start_year,
            end_year=self.end_year,
            config=self.config,
            train_end_year=self.train_end_year,
            validation_start_year=self.validation_start_year,
        )
        return diagnostics.compute_target_horizons(base)

    def _predict_model(
        self,
        train: pl.DataFrame,
        frame: pl.DataFrame,
        params: dict[str, float],
        model_variant: str,
        *,
        family_params: dict[str, dict[str, float]],
        improved_sectors: set[str],
    ) -> pl.DataFrame:
        if model_variant == "sector_background_plus_year_country_controls":
            return self.predict_with_diagnostic_controls(train, frame)
        if model_variant == "sector_family_frontier_gap_readiness":
            frames = []
            for family in frame["sector_family"].unique().to_list():
                family_frame = frame.filter(pl.col("sector_family") == family)
                frames.append(
                    self.predict_with_parameters(
                        family_frame,
                        family_params.get(str(family), params),
                        "global_frontier_gap_readiness",
                        improved_sectors=improved_sectors,
                    )
                )
            return pl.concat(frames, how="diagonal") if frames else pl.DataFrame()
        return self.predict_with_parameters(
            frame,
            params,
            model_variant,
            improved_sectors=improved_sectors,
        )

    def predict_with_diagnostic_controls(self, train: pl.DataFrame, frame: pl.DataFrame) -> pl.DataFrame:
        """Add train-estimated country controls and in-split year controls for diagnostics only."""
        country_effect = (
            train.with_columns((pl.col("observed_rEI") - pl.col("sector_background_trend")).alias("_resid"))
            .group_by("Country")
            .agg(pl.col("_resid").median().alias("_country_effect"))
        )
        year_effect = (
            frame.with_columns((pl.col("observed_rEI") - pl.col("sector_background_trend")).alias("_resid"))
            .group_by("year")
            .agg(pl.col("_resid").median().alias("_year_effect"))
        )
        return (
            frame.join(country_effect, on="Country", how="left")
            .join(year_effect, on="year", how="left")
            .with_columns(
                (
                    pl.col("sector_background_trend")
                    + pl.col("_country_effect").fill_null(0.0)
                    + pl.col("_year_effect").fill_null(0.0)
                ).alias("simulated_rEI")
            )
            .with_columns(
                (pl.col("simulated_rEI") - pl.col("observed_rEI")).alias("rEI_error"),
                (pl.col("simulated_rEI") - pl.col("observed_rEI")).abs().alias("rEI_abs_error"),
            )
            .drop(["_country_effect", "_year_effect"])
        )

    def _readiness_linear(
        self,
        params: dict[str, float],
        *,
        include_capability: bool,
        include_network: bool,
    ) -> pl.Expr:
        expr = pl.lit(params["theta_intercept"])
        if include_capability:
            expr = (
                expr
                + pl.lit(params["theta_gcap"]) * pl.col("gcap_model").fill_null(0.0)
                + pl.lit(params["theta_cap"]) * pl.col("cap_model").fill_null(0.0)
            )
        if include_network:
            expr = (
                expr
                + pl.lit(params["theta_network_green"]) * pl.col("network_green_exposure").fill_null(0.0)
                + pl.lit(params["theta_ecosystem_exposure"])
                * pl.col("ecosystem_capability_exposure").fill_null(0.0)
                - pl.lit(params["theta_brown_centrality"]) * pl.col("brown_centrality").fill_null(0.0)
                - pl.lit(params["theta_supplier_lockin"]) * pl.col("supplier_lockin").fill_null(0.0)
            )
        return expr

    def _result_row(
        self,
        target_name: str,
        frontier_variant: str,
        model_variant: str,
        train_metrics: dict[str, float],
        validation_metrics: dict[str, float],
        baseline_metrics: dict[str, float],
    ) -> dict[str, Any]:
        improvement = baseline_metrics["mae"] - validation_metrics["mae"]
        return {
            "target_name": target_name,
            "frontier_variant": frontier_variant,
            "model_variant": model_variant,
            "train_mae": train_metrics["mae"],
            "validation_mae": validation_metrics["mae"],
            "validation_rmse": validation_metrics["rmse"],
            "validation_bias": validation_metrics["bias"],
            "validation_wrong_sign_share": validation_metrics["wrong_sign_share"],
            "validation_correlation": validation_metrics["correlation"],
            "validation_median_abs_error": validation_metrics["median_abs_error"],
            "improvement_over_sector_background": improvement,
            "improvement_over_sector_background_pct": (
                improvement / baseline_metrics["mae"] if baseline_metrics["mae"] > 0 else float("nan")
            ),
            "rows_train": int(train_metrics["rows"]),
            "rows_validation": int(validation_metrics["rows"]),
            "diagnostic_controls_only": model_variant == "sector_background_plus_year_country_controls",
        }

    def _default_parameter_set(self) -> dict[str, float]:
        return self._calibrator._default_parameter_set()

    def _best_result(self, results: pl.DataFrame, model_variant: str) -> dict[str, Any] | None:
        frame = results.filter(pl.col("model_variant") == model_variant).sort("validation_mae")
        return None if frame.is_empty() else frame.to_dicts()[0]

    def _frontier_family_wins(self, results: pl.DataFrame, suffix: str) -> int:
        winners = results.sort("validation_mae").group_by(["target_name", "frontier_variant"]).first()
        return winners.filter(pl.col("frontier_variant").str.ends_with(suffix)).height

    def _frontier_prefix_wins(self, results: pl.DataFrame, prefix: str) -> int:
        winners = results.sort("validation_mae").group_by(["target_name", "frontier_variant"]).first()
        return winners.filter(pl.col("frontier_variant").str.starts_with(prefix)).height

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


class EmissionsTransitionHypothesisDiagnostics:
    """Diagnose why annual frontier-gap readiness calibration is weak."""

    def __init__(
        self,
        paths: ABMV4Paths,
        *,
        start_year: int = 1995,
        end_year: int = 2016,
        config: EmissionsConfig | None = None,
        train_end_year: int = 2011,
        validation_start_year: int = 2012,
    ) -> None:
        self.paths = paths
        self.start_year = start_year
        self.end_year = end_year
        self.config = config or EmissionsConfig()
        self.train_end_year = train_end_year
        self.validation_start_year = validation_start_year

    def load_state_panel(self) -> pl.DataFrame:
        """Load the ABM v4 state panel used by emissions diagnostics."""
        state_path = self.paths.state_panel_path(self.start_year, self.end_year)
        if not state_path.exists():
            canonical = self.paths.state_panel_path(1995, 2016)
            if canonical.exists():
                return pl.read_parquet(canonical)
            raise FileNotFoundError(f"ABM v4 state panel not found: {state_path}")
        return pl.read_parquet(state_path)

    def build_base_transition_panel(self, state_panel: pl.DataFrame | None = None) -> pl.DataFrame:
        """Build feature panel with EI gaps, background trend, lock-in, and readiness."""
        calibrator = EmissionsTransitionCalibrator(
            self.paths,
            start_year=self.start_year,
            end_year=self.end_year,
            config=self.config,
            train_end_year=self.train_end_year,
            validation_start_year=self.validation_start_year,
        )
        state = self.load_state_panel() if state_panel is None else state_panel
        base = calibrator._prepare_state_features(
            state.filter(pl.col("Year").is_between(self.start_year, self.end_year))
        )
        base = calibrator._add_sector_frontiers(base)
        base = calibrator._add_sector_background_trend(base)
        base = calibrator._add_supplier_lockin(base)
        return self._add_default_readiness(base)

    def compute_target_horizons(self, base_panel: pl.DataFrame) -> pl.DataFrame:
        """Compute annual, medium-run, smoothed, and winsorized rEI targets."""
        base = (
            base_panel.sort(["country_sector", "year"])
            .with_columns(pl.col("log_EI").rolling_mean(window_size=3).over("country_sector").alias("_smooth_log_EI"))
        )
        frames: list[pl.DataFrame] = []
        for target_name, horizon in [
            ("one_year_rEI", 1),
            ("three_year_rEI", 3),
            ("five_year_rEI", 5),
        ]:
            frames.append(self._target_for_horizon(base, target_name, horizon, "log_EI"))
        frames.append(self._target_for_horizon(base, "smoothed_one_year_rEI", 1, "_smooth_log_EI"))
        one_year = self._target_for_horizon(base, "winsorized_one_year_rEI", 1, "log_EI")
        if not one_year.is_empty():
            low = one_year["target"].quantile(0.01)
            high = one_year["target"].quantile(0.99)
            one_year = one_year.with_columns(pl.col("target").clip(low, high).alias("target"))
        frames.append(one_year)
        return pl.concat([frame for frame in frames if not frame.is_empty()], how="diagonal")

    def test_h1_target_noise(self, target_panel: pl.DataFrame) -> pl.DataFrame:
        """Compare target horizons and smoothed/winsorized annual targets."""
        rows = []
        for target_name in target_panel["target_name"].unique().to_list():
            frame = target_panel.filter(pl.col("target_name") == target_name)
            horizon = int(frame["horizon_years"].drop_nulls().item(0))
            validation_cutoff = min(self.validation_start_year, self.end_year - horizon)
            validation = frame.filter(pl.col("year") >= validation_cutoff)
            rows.append(
                {
                    "target_name": target_name,
                    "horizon_years": horizon,
                    "rows": frame.height,
                    "mean": frame["target"].mean(),
                    "median": frame["target"].median(),
                    "std": frame["target"].std(),
                    "p05": frame["target"].quantile(0.05),
                    "p25": frame["target"].quantile(0.25),
                    "p75": frame["target"].quantile(0.75),
                    "p95": frame["target"].quantile(0.95),
                    "share_positive": self._share(frame, pl.col("target") > 0),
                    "share_negative": self._share(frame, pl.col("target") < 0),
                    "readiness_correlation": self._corr(frame, "target", "readiness"),
                    "cap_model_correlation": self._corr(frame, "target", "cap_model"),
                    "gcap_model_correlation": self._corr(frame, "target", "gcap_model"),
                    "network_green_correlation": self._corr(frame, "target", "network_green_exposure"),
                    "brown_centrality_correlation": self._corr(frame, "target", "brown_centrality"),
                    "sector_background_mae": self._mae(validation, "sector_background_prediction"),
                    "simple_readiness_mae": self._mae(validation, "simple_readiness_prediction"),
                    "improvement_over_background": self._mae(validation, "sector_background_prediction")
                    - self._mae(validation, "simple_readiness_prediction"),
                }
            )
        return pl.DataFrame(rows).sort("target_name")

    def test_h2_sector_dominance(self, target_panel: pl.DataFrame) -> pl.DataFrame:
        """Measure whether readiness improves over sector background by sector."""
        one_year = self._validation_target(target_panel)
        return (
            one_year.group_by("Sector")
            .agg(
                pl.len().alias("rows"),
                pl.col("target").mean().alias("mean_target"),
                pl.col("target").std().alias("target_std"),
                (pl.col("target") - pl.col("sector_background_prediction")).abs().mean().alias("sector_background_mae"),
                (pl.col("target") - pl.col("simple_readiness_prediction")).abs().mean().alias("simple_readiness_mae"),
                pl.corr("target", "readiness").alias("readiness_correlation"),
            )
            .with_columns(
                (pl.col("sector_background_mae") - pl.col("simple_readiness_mae")).alias("readiness_improvement")
            )
            .with_columns(
                pl.when(pl.col("readiness_improvement") > 0.005)
                .then(pl.lit("consider sector-specific readiness rule"))
                .when(pl.col("readiness_improvement") < 0)
                .then(pl.lit("sector background dominates"))
                .otherwise(pl.lit("weak readiness contribution"))
                .alias("recommended_action")
            )
            .sort("readiness_improvement", descending=True)
        )

    def test_h3_capability_measurement(self, target_panel: pl.DataFrame) -> pl.DataFrame:
        """Compare capability/readiness diagnostics by capability source."""
        one_year = self._validation_target(target_panel)
        frames: list[pl.DataFrame] = []
        for source_col, cap_col, cap_type in [
            ("general_capability_source", "cap_model", "general"),
            ("green_capability_source", "gcap_model", "green"),
        ]:
            summary = (
                one_year.group_by(source_col)
                .agg(
                    pl.len().alias("rows"),
                    pl.col("target").mean().alias("mean_target"),
                    pl.col("target").std().alias("target_std"),
                    pl.corr("target", "readiness").alias("readiness_correlation"),
                    pl.corr("target", cap_col).alias("capability_correlation"),
                    (pl.col("target") - pl.col("sector_background_prediction")).abs().mean().alias("sector_background_mae"),
                    (pl.col("target") - pl.col("simple_readiness_prediction")).abs().mean().alias("simple_readiness_mae"),
                    self._wrong_sign_expr("simple_readiness_prediction", "target").alias("wrong_sign_share"),
                )
                .rename({source_col: "capability_source"})
                .with_columns(
                    pl.lit(cap_type).alias("capability_type"),
                    (pl.col("sector_background_mae") - pl.col("simple_readiness_mae")).alias("improvement_over_background"),
                )
                .with_columns(
                    pl.when(pl.col("capability_source") == "atlas_observed")
                    .then(pl.lit("use as strongest capability evidence slice"))
                    .when(pl.col("improvement_over_background") > 0)
                    .then(pl.lit("retain proxy but calibrate separately"))
                    .otherwise(pl.lit("restrict calibration claims"))
                    .alias("recommended_action")
                )
            )
            frames.append(summary)
        return pl.concat(frames, how="diagonal")

    def test_h4_threshold_readiness(self, target_panel: pl.DataFrame) -> pl.DataFrame:
        """Summarize target outcomes by readiness quantiles and interaction cells."""
        one_year = self._validation_target(target_panel)
        frames = [
            self._quantile_summary(one_year, "readiness", 4, "readiness_quartile"),
            self._quantile_summary(one_year, "readiness", 10, "readiness_decile"),
            self._interaction_cells(one_year),
        ]
        return pl.concat([frame for frame in frames if not frame.is_empty()], how="diagonal")

    def test_h5_frontier_specification(self, base_panel: pl.DataFrame, target_panel: pl.DataFrame) -> pl.DataFrame:
        """Compare alternative feasible frontier definitions."""
        one_year = self._validation_target(target_panel).select(
            "country_sector",
            "year",
            "target",
            "readiness",
            "sector_background_prediction",
        )
        rows = []
        for spec in [
            "sector_year_p10",
            "sector_year_p25",
            "sector_year_p50",
            "rolling_sector_p25",
            "rolling_sector_p10",
            "sector_ecosystem_p25",
            "winsorized_sector_year_p25",
        ]:
            gaps = self._frontier_gap_for_spec(base_panel, spec)
            frame = one_year.join(gaps, on=["country_sector", "year"], how="inner")
            if frame.is_empty():
                continue
            gap_fraction = pl.col("frontier_gap") / (pl.col("frontier_gap") + self.config.tau_gap)
            scored = frame.with_columns(
                (pl.col("sector_background_prediction") + self.config.rho_max * gap_fraction).alias("frontier_gap_only_prediction"),
                (pl.col("sector_background_prediction") + pl.col("readiness") * gap_fraction).alias("readiness_gated_gap_prediction"),
            )
            rows.append(
                {
                    "frontier_specification": spec,
                    "rows": scored.height,
                    "mean_gap": scored["frontier_gap"].mean(),
                    "median_gap": scored["frontier_gap"].median(),
                    "share_zero_gap": self._share(scored, pl.col("frontier_gap") <= 0),
                    "max_gap": scored["frontier_gap"].max(),
                    "target_correlation": self._corr(scored, "target", "frontier_gap"),
                    "frontier_gap_only_mae": self._mae(scored, "frontier_gap_only_prediction"),
                    "readiness_gated_gap_mae": self._mae(scored, "readiness_gated_gap_prediction"),
                    "fallback_count": scored.filter(pl.col("frontier_gap").is_null()).height,
                    "notes": "diagnostic frontier; not a recalibration",
                }
            )
        return pl.DataFrame(rows).sort("frontier_gap_only_mae")

    def test_h6_macro_shocks(self, target_panel: pl.DataFrame) -> pl.DataFrame:
        """Identify year, country, and crisis-period residual clustering."""
        one_year = self._validation_target(target_panel).with_columns(
            (pl.col("simple_readiness_prediction") - pl.col("target")).alias("residual"),
            pl.when(pl.col("year").is_between(2008, 2009))
            .then(pl.lit("2008_2009_crisis"))
            .when(pl.col("year") < 2008)
            .then(pl.lit("pre_2008"))
            .otherwise(pl.lit("post_2010"))
            .alias("period_group"),
        )
        frames = [
            self._macro_group(one_year, "year", "year"),
            self._macro_group(one_year, "Country", "country"),
            self._macro_group(one_year, "period_group", "period"),
        ]
        return pl.concat(frames, how="diagonal")

    def build_hypothesis_diagnosis_table(
        self,
        h1: pl.DataFrame,
        h2: pl.DataFrame,
        h3: pl.DataFrame,
        h4: pl.DataFrame,
        h5: pl.DataFrame,
        h6: pl.DataFrame,
    ) -> pl.DataFrame:
        """Summarize evidence for H1-H6 into one decision table."""
        one = h1.filter(pl.col("target_name") == "one_year_rEI")
        best_horizon = h1.sort("improvement_over_background", descending=True).to_dicts()[0]
        one_improvement = one["improvement_over_background"].item() if not one.is_empty() else 0.0
        readiness_positive_share = (
            h2.filter(pl.col("readiness_improvement") > 0).height / max(h2.height, 1)
            if not h2.is_empty()
            else 0.0
        )
        atlas = h3.filter((pl.col("capability_source") == "atlas_observed") & (pl.col("capability_type") == "general"))
        io = h3.filter((pl.col("capability_source") == "io_imputed") & (pl.col("capability_type") == "general"))
        top_quantile = h4.filter(pl.col("quantile_type") == "readiness_quartile").sort("quantile").tail(1)
        bottom_quantile = h4.filter(pl.col("quantile_type") == "readiness_quartile").sort("quantile").head(1)
        h4_supported = (
            not top_quantile.is_empty()
            and not bottom_quantile.is_empty()
            and top_quantile["mean_target"].item() > bottom_quantile["mean_target"].item()
        )
        best_frontier = h5.sort("frontier_gap_only_mae").to_dicts()[0] if not h5.is_empty() else {}
        year_rows = h6.filter(pl.col("grouping_type") == "year")
        year_resid_std = year_rows["mean_residual"].std() if not year_rows.is_empty() else None
        rows = [
            self._hypothesis_row(
                "H1",
                "Annual rEI is too noisy for capability mechanisms",
                "Compare one-year, three-year, five-year, smoothed, and winsorized targets.",
                best_horizon["improvement_over_background"] > one_improvement + 0.001,
                "best horizon improvement",
                best_horizon["improvement_over_background"],
                f"Best target: {best_horizon['target_name']}.",
                "Use medium-run or smoothed target if improvement persists.",
            ),
            self._hypothesis_row(
                "H2",
                "Sector background dominates annual EI movement",
                "Compare readiness improvement over background by sector.",
                readiness_positive_share < 0.35,
                "share sectors readiness improves",
                readiness_positive_share,
                "Readiness improves only a limited sector subset." if readiness_positive_share < 0.35 else "Readiness helps in multiple sectors.",
                "Consider sector-family-specific transition rules.",
            ),
            self._hypothesis_row(
                "H3",
                "Capability variables are conceptually valid but weakly measured",
                "Compare Atlas and IO-imputed capability-source groups.",
                (not atlas.is_empty() and not io.is_empty() and atlas["simple_readiness_mae"].item() < io["simple_readiness_mae"].item()),
                "atlas minus io MAE",
                (atlas["simple_readiness_mae"].item() - io["simple_readiness_mae"].item()) if not atlas.is_empty() and not io.is_empty() else None,
                "Atlas-observed capability performs better than IO-imputed." if not atlas.is_empty() and not io.is_empty() else "Capability source evidence incomplete.",
                "Keep IO as integration proxy; calibrate capability effects by source.",
            ),
            self._hypothesis_row(
                "H4",
                "Readiness is nonlinear or threshold-based",
                "Compare target outcomes across readiness quantiles and interaction cells.",
                h4_supported,
                "top minus bottom readiness mean target",
                (top_quantile["mean_target"].item() - bottom_quantile["mean_target"].item()) if not top_quantile.is_empty() and not bottom_quantile.is_empty() else None,
                "High readiness has better average target." if h4_supported else "Smooth readiness quantiles do not show the expected ordering.",
                "Test threshold/regime readiness rather than smooth global additive readiness.",
            ),
            self._hypothesis_row(
                "H5",
                "The frontier-gap variable is poorly specified",
                "Compare alternative frontier definitions.",
                bool(best_frontier) and best_frontier.get("frontier_specification") != "sector_year_p25",
                "best frontier gap-only MAE",
                best_frontier.get("frontier_gap_only_mae"),
                f"Best frontier diagnostic: {best_frontier.get('frontier_specification', 'none')}.",
                "Revise frontier only if improvement is interpretable and robust.",
            ),
            self._hypothesis_row(
                "H6",
                "Macro, country-year, or crisis shocks dominate annual rEI",
                "Summarize residuals by year, country, and crisis period.",
                year_resid_std is not None and year_resid_std > 0.02,
                "std of yearly mean residual",
                year_resid_std,
                "Residuals vary meaningfully by year." if year_resid_std is not None else "Year residual evidence unavailable.",
                "Add historical year/country-year controls for calibration diagnostics only.",
            ),
        ]
        return pl.DataFrame(rows)

    def run(self) -> EmissionsHypothesisDiagnosticResult:
        """Run all Phase 13 hypothesis diagnostics."""
        base = self.build_base_transition_panel()
        target_panel = self.compute_target_horizons(base)
        h1 = self.test_h1_target_noise(target_panel)
        h2 = self.test_h2_sector_dominance(target_panel)
        h3 = self.test_h3_capability_measurement(target_panel)
        h4 = self.test_h4_threshold_readiness(target_panel)
        h5 = self.test_h5_frontier_specification(base, target_panel)
        h6 = self.test_h6_macro_shocks(target_panel)
        diagnosis = self.build_hypothesis_diagnosis_table(h1, h2, h3, h4, h5, h6)
        predictor_screening = self._build_predictor_screening(target_panel)
        markdown = self.build_markdown_report(diagnosis, h1, h2, h3, h4, h5, h6)
        return EmissionsHypothesisDiagnosticResult(
            hypothesis_diagnosis=diagnosis,
            target_horizon_panel=target_panel,
            target_horizon_summary=h1,
            predictor_screening=predictor_screening,
            sector_dominance=h2,
            capability_source=h3,
            readiness_threshold=h4,
            frontier_specification=h5,
            macro_shock=h6,
            markdown=markdown,
        )

    def write_outputs(self, result: EmissionsHypothesisDiagnosticResult) -> None:
        """Write Phase 13 hypothesis diagnostics."""
        self.paths.validation.mkdir(parents=True, exist_ok=True)
        result.hypothesis_diagnosis.write_csv(self.paths.emissions_hypothesis_diagnosis_path)
        result.target_horizon_panel.write_parquet(self.paths.emissions_target_horizon_panel_path)
        result.target_horizon_summary.write_csv(self.paths.emissions_target_horizon_summary_path)
        result.predictor_screening.write_csv(self.paths.emissions_predictor_screening_path)
        result.sector_dominance.write_csv(self.paths.emissions_sector_dominance_diagnostics_path)
        result.capability_source.write_csv(self.paths.emissions_capability_source_diagnostics_path)
        result.readiness_threshold.write_csv(self.paths.emissions_readiness_threshold_diagnostics_path)
        result.frontier_specification.write_csv(self.paths.emissions_frontier_specification_diagnostics_path)
        result.macro_shock.write_csv(self.paths.emissions_macro_shock_diagnostics_path)
        self.paths.emissions_hypothesis_diagnostic_report_path.write_text(
            result.markdown,
            encoding="utf-8",
        )

    def build_markdown_report(
        self,
        diagnosis: pl.DataFrame,
        h1: pl.DataFrame,
        h2: pl.DataFrame,
        h3: pl.DataFrame,
        h4: pl.DataFrame,
        h5: pl.DataFrame,
        h6: pl.DataFrame,
    ) -> str:
        """Render Phase 13 diagnosis as a compact Markdown report."""
        lines = [
            "# ABM v4 Emissions-Transition Hypothesis Diagnostics",
            "",
            "Phase 12 remained exploratory because the full readiness model did not outperform sector background, capability terms were weak, validation correlation was near zero, and wrong-sign share exceeded 50%.",
            "",
            "## Hypothesis Diagnosis",
            "",
            self._markdown_table(diagnosis),
            "",
            "## H1 Target Horizon Noise",
            "",
            self._markdown_table(h1),
            "",
            "## H2 Sector Dominance",
            "",
            self._markdown_table(h2.head(10)),
            "",
            "## H3 Capability Measurement",
            "",
            self._markdown_table(h3),
            "",
            "## H4 Threshold Readiness",
            "",
            self._markdown_table(h4.head(20)),
            "",
            "## H5 Frontier Specification",
            "",
            self._markdown_table(h5),
            "",
            "## H6 Macro Shocks",
            "",
            self._markdown_table(h6.head(20)),
            "",
            "## Recommended Next Action",
            "",
            "Do not move to scenarios yet. Use these diagnostics to choose the next calibration target and model form, with special attention to medium-run targets, sector-family rules, threshold readiness, and historical year/country controls for calibration diagnostics.",
        ]
        return "\n".join(lines) + "\n"

    def _target_for_horizon(
        self,
        base: pl.DataFrame,
        target_name: str,
        horizon: int,
        log_column: str,
    ) -> pl.DataFrame:
        return (
            base.with_columns(
                pl.col("year").shift(-horizon).over("country_sector").alias("next_year"),
                pl.col(log_column).shift(-horizon).over("country_sector").alias("_next_log_EI"),
                *[
                    pl.col("log_EI").shift(-step).over("country_sector").alias(f"_raw_log_EI_step_{step}")
                    for step in range(0, horizon + 1)
                ],
            )
            .filter(
                (pl.col("next_year") == pl.col("year") + horizon)
                & pl.col(log_column).is_not_null()
                & pl.col("_next_log_EI").is_not_null()
                & pl.all_horizontal(
                    *[
                        pl.col(f"_raw_log_EI_step_{step}").is_not_null()
                        for step in range(0, horizon + 1)
                    ]
                )
            )
            .with_columns(
                ((pl.col(log_column) - pl.col("_next_log_EI")) / horizon).alias("target"),
                pl.lit(target_name).alias("target_name"),
                pl.lit(horizon).alias("horizon_years"),
                (
                    pl.col("sector_background_trend")
                    + pl.col("readiness") * pl.col("ei_gap") / (pl.col("ei_gap") + self.config.tau_gap)
                ).alias("simple_readiness_prediction"),
                pl.col("sector_background_trend").alias("sector_background_prediction"),
            )
            .select(
                "target_name",
                "horizon_years",
                "country_sector",
                "year",
                "next_year",
                "Country",
                "Sector",
                "ecosystem_id",
                "target",
                "sector_background_prediction",
                "simple_readiness_prediction",
                "readiness",
                "ei_gap",
                "cap_model",
                "gcap_model",
                "general_capability_source",
                "green_capability_source",
                "network_green_exposure",
                "ecosystem_capability_exposure",
                "brown_centrality",
                "supplier_lockin",
            )
        )

    def _add_default_readiness(self, base: pl.DataFrame) -> pl.DataFrame:
        linear = (
            pl.lit(self.config.theta_intercept)
            + self.config.theta_gcap * pl.col("gcap_model").fill_null(0.0)
            + self.config.theta_cap * pl.col("cap_model").fill_null(0.0)
            + self.config.theta_network_green * pl.col("network_green_exposure").fill_null(0.0)
            + self.config.theta_ecosystem_exposure * pl.col("ecosystem_capability_exposure").fill_null(0.0)
            - self.config.theta_brown_centrality * pl.col("brown_centrality").fill_null(0.0)
            - self.config.theta_supplier_lockin * pl.col("supplier_lockin").fill_null(0.0)
        )
        return base.with_columns(linear.alias("readiness_linear")).with_columns(
            (self.config.rho_max / (1.0 + (-pl.col("readiness_linear")).exp())).alias("readiness")
        )

    def _validation_target(self, target_panel: pl.DataFrame) -> pl.DataFrame:
        return target_panel.filter(
            (pl.col("target_name") == "one_year_rEI")
            & (pl.col("year") >= self.validation_start_year)
        )

    def _frontier_gap_for_spec(self, base_panel: pl.DataFrame, spec: str) -> pl.DataFrame:
        quantile = 0.25
        group_cols = ["year", "Sector"]
        valid = base_panel.filter(pl.col("EI") > 0)
        source = valid
        if spec.endswith("p10"):
            quantile = 0.10
        elif spec.endswith("p50"):
            quantile = 0.50
        if spec == "sector_ecosystem_p25":
            group_cols = ["year", "Sector", "ecosystem_id"]
        if spec.startswith("rolling_sector"):
            rows = []
            for row in valid.select("year", "Sector").unique().to_dicts():
                subset = valid.filter((pl.col("Sector") == row["Sector"]) & (pl.col("year") <= row["year"]))
                frontier = subset["EI"].quantile(quantile)
                rows.append({"year": row["year"], "Sector": row["Sector"], "EI_frontier_alt": frontier})
            frontier = pl.DataFrame(rows)
            joined = base_panel.join(frontier, on=["year", "Sector"], how="left")
        else:
            if spec == "winsorized_sector_year_p25":
                bounds = valid.group_by(group_cols).agg(
                    pl.col("EI").quantile(0.01).alias("_low"),
                    pl.col("EI").quantile(0.99).alias("_high"),
                )
                source = valid.join(bounds, on=group_cols, how="left").with_columns(
                    pl.col("EI").clip(pl.col("_low"), pl.col("_high")).alias("EI")
                )
            frontier = source.group_by(group_cols).agg(
                pl.col("EI").quantile(quantile).alias("EI_frontier_alt")
            )
            joined = base_panel.join(frontier, on=group_cols, how="left")
        return joined.select(
            "country_sector",
            "year",
            pl.when((pl.col("EI") > 0) & (pl.col("EI_frontier_alt") > 0))
            .then((pl.col("EI").log() - pl.col("EI_frontier_alt").log()).clip(0.0, None))
            .otherwise(None)
            .alias("frontier_gap"),
        )

    def _quantile_summary(
        self,
        frame: pl.DataFrame,
        column: str,
        bins: int,
        quantile_type: str,
    ) -> pl.DataFrame:
        if frame.is_empty() or column not in frame.columns:
            return pl.DataFrame()
        scored = self._assign_quantiles(frame, column, bins, "quantile")
        return self._threshold_summary(scored, quantile_type)

    def _interaction_cells(self, frame: pl.DataFrame) -> pl.DataFrame:
        if frame.is_empty():
            return pl.DataFrame()
        gap_q = frame["ei_gap"].quantile(0.75)
        readiness_q = frame["readiness"].quantile(0.75)
        network_q = frame["network_green_exposure"].quantile(0.75)
        cap_q = frame["cap_model"].quantile(0.75)
        brown_q = frame["brown_centrality"].quantile(0.75)
        cells = frame.with_columns(
            pl.when((pl.col("ei_gap") >= gap_q) & (pl.col("readiness") >= readiness_q))
            .then(pl.lit("high_gap_high_readiness"))
            .when((pl.col("ei_gap") >= gap_q) & (pl.col("readiness") < readiness_q))
            .then(pl.lit("high_gap_low_readiness"))
            .when((pl.col("ei_gap") < gap_q) & (pl.col("readiness") >= readiness_q))
            .then(pl.lit("low_gap_high_readiness"))
            .otherwise(pl.lit("low_gap_low_readiness"))
            .alias("quantile")
        )
        extra = frame.with_columns(
            pl.when((pl.col("network_green_exposure") >= network_q) & (pl.col("cap_model") >= cap_q))
            .then(pl.lit("high_network_high_capability"))
            .when((pl.col("brown_centrality") >= brown_q) & (pl.col("readiness") < readiness_q))
            .then(pl.lit("high_brown_low_readiness"))
            .when((pl.col("brown_centrality") >= brown_q) & (pl.col("readiness") >= readiness_q))
            .then(pl.lit("high_brown_high_readiness"))
            .otherwise(pl.lit("other_interaction"))
            .alias("quantile")
        )
        return pl.concat(
            [
                self._threshold_summary(cells, "interaction"),
                self._threshold_summary(extra, "interaction"),
            ],
            how="diagonal",
        )

    def _threshold_summary(self, frame: pl.DataFrame, quantile_type: str) -> pl.DataFrame:
        return (
            frame.group_by("quantile")
            .agg(
                pl.len().alias("rows"),
                pl.col("target").mean().alias("mean_target"),
                pl.col("target").median().alias("median_target"),
                pl.col("target").quantile(0.25).alias("p25_target"),
                pl.col("target").quantile(0.75).alias("p75_target"),
                (pl.col("target") > 0).mean().alias("share_positive"),
                (pl.col("target") < -0.05).mean().alias("share_severe_worsening"),
                pl.col("ei_gap").mean().alias("mean_ei_gap"),
                pl.col("readiness").mean().alias("mean_readiness"),
                pl.col("cap_model").mean().alias("mean_cap_model"),
                pl.col("gcap_model").mean().alias("mean_gcap_model"),
                pl.col("brown_centrality").mean().alias("mean_brown_centrality"),
            )
            .with_columns(pl.lit(quantile_type).alias("quantile_type"))
            .select(
                "quantile_type",
                "quantile",
                "rows",
                "mean_target",
                "median_target",
                "p25_target",
                "p75_target",
                "share_positive",
                "share_severe_worsening",
                "mean_ei_gap",
                "mean_readiness",
                "mean_cap_model",
                "mean_gcap_model",
                "mean_brown_centrality",
            )
            .sort(["quantile_type", "quantile"])
        )

    def _macro_group(self, frame: pl.DataFrame, column: str, grouping_type: str) -> pl.DataFrame:
        return (
            frame.group_by(column)
            .agg(
                pl.len().alias("rows"),
                pl.col("target").mean().alias("mean_target"),
                pl.col("target").std().alias("target_std"),
                pl.col("residual").mean().alias("mean_residual"),
                pl.col("residual").abs().mean().alias("mean_abs_residual"),
                self._wrong_sign_expr("simple_readiness_prediction", "target").alias("wrong_sign_share"),
            )
            .rename({column: "group"})
            .with_columns(
                pl.col("group").cast(pl.Utf8),
                pl.lit(grouping_type).alias("grouping_type"),
                pl.when(pl.col("mean_abs_residual") > pl.col("mean_abs_residual").mean())
                .then(pl.lit("inspect shock/control"))
                .otherwise(pl.lit("lower priority"))
                .alias("recommended_action"),
            )
        )

    def _build_predictor_screening(self, target_panel: pl.DataFrame) -> pl.DataFrame:
        rows = []
        for target in target_panel["target_name"].unique().to_list():
            frame = target_panel.filter(pl.col("target_name") == target)
            for predictor in [
                "readiness",
                "ei_gap",
                "cap_model",
                "gcap_model",
                "network_green_exposure",
                "brown_centrality",
                "supplier_lockin",
            ]:
                rows.append(
                    {
                        "target_name": target,
                        "predictor": predictor,
                        "correlation": self._corr(frame, "target", predictor),
                    }
                )
        return pl.DataFrame(rows)

    def _assign_quantiles(self, frame: pl.DataFrame, column: str, bins: int, output: str) -> pl.DataFrame:
        valid = frame.filter(pl.col(column).is_not_null())
        if valid.is_empty():
            return frame.with_columns(pl.lit("missing").alias(output))
        thresholds = [valid[column].quantile(i / bins) for i in range(1, bins)]
        expr = pl.when(pl.col(column).is_null()).then(pl.lit("missing"))
        for index, threshold in enumerate(thresholds, start=1):
            expr = expr.when(pl.col(column) <= threshold).then(pl.lit(f"q{index}"))
        return frame.with_columns(expr.otherwise(pl.lit(f"q{bins}")).alias(output))

    def _hypothesis_row(
        self,
        hypothesis_id: str,
        hypothesis_name: str,
        test_summary: str,
        supported: bool,
        key_metric: str,
        key_value: Any,
        interpretation: str,
        next_action: str,
    ) -> dict[str, Any]:
        return {
            "hypothesis_id": hypothesis_id,
            "hypothesis_name": hypothesis_name,
            "test_summary": test_summary,
            "evidence_strength": "moderate" if supported else "weak",
            "evidence_direction": "supports_hypothesis" if supported else "mixed",
            "key_metric": key_metric,
            "key_value": key_value,
            "interpretation": interpretation,
            "recommended_next_action": next_action,
            "priority": "high" if supported else "medium",
        }

    def _mae(self, frame: pl.DataFrame, prediction_column: str) -> float:
        if frame.is_empty() or prediction_column not in frame.columns:
            return float("nan")
        return _clean_float((frame["target"] - frame[prediction_column]).abs().mean())

    def _corr(self, frame: pl.DataFrame, left: str, right: str) -> float:
        if frame.is_empty() or left not in frame.columns or right not in frame.columns:
            return float("nan")
        value = frame.select(pl.corr(left, right).alias("_corr"))["_corr"].item()
        return _clean_float(value)

    def _share(self, frame: pl.DataFrame, expr: pl.Expr) -> float:
        if frame.is_empty():
            return float("nan")
        return _clean_float(frame.select(expr.mean().alias("_share"))["_share"].item())

    def _wrong_sign_expr(self, prediction: str, target: str) -> pl.Expr:
        return (
            ((pl.col(prediction) > 0) & (pl.col(target) < 0))
            | ((pl.col(prediction) < 0) & (pl.col(target) > 0))
        ).mean()

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
