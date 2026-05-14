from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl

from src.abm_v4.config import CapabilityConfig
from src.abm_v4.paths import ABMV4Paths


def sigmoid(value: float) -> float:
    """Return a numerically stable logistic transform."""
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def capability_increment(capability: float, exposure: float, config: CapabilityConfig) -> float:
    """Compute general capability accumulation without depreciation."""
    remaining_capacity = config.cap_max - capability
    transition_signal = sigmoid(config.k_cap * (exposure - config.tau_cap))
    return config.delta_cap_param * remaining_capacity * transition_signal


def green_capability_increment(green_capability: float, exposure: float, config: CapabilityConfig) -> float:
    """Compute green capability accumulation without depreciation."""
    remaining_capacity = config.gcap_max - green_capability
    transition_signal = sigmoid(config.k_gcap * (exposure - config.tau_gcap))
    return config.delta_gcap_param * remaining_capacity * transition_signal


@dataclass(frozen=True)
class CapabilityExposureWeights:
    """Transparent weights for one-step capability exposure construction."""

    omega_prod_exp: float = 0.34
    omega_network_cap: float = 0.33
    omega_ecosystem_cap: float = 0.33
    omega_green_prod_exp: float = 0.34
    omega_network_green: float = 0.33
    omega_green_supplier_share: float = 0.33


class CapabilityUpdater:
    """Build one-step ABM v4 general and green capability updates."""

    def __init__(
        self,
        paths: ABMV4Paths,
        start_year: int = 1995,
        end_year: int = 2016,
        config: CapabilityConfig | None = None,
        exposure_weights: CapabilityExposureWeights | None = None,
    ) -> None:
        self.paths = paths
        self.start_year = start_year
        self.end_year = end_year
        self.config = config or CapabilityConfig()
        self.exposure_weights = exposure_weights or CapabilityExposureWeights()

    def load_state_panel(self) -> pl.DataFrame:
        """Load the ABM v4 state panel."""
        state_path = self.paths.state_panel_path(self.start_year, self.end_year)
        if not state_path.exists():
            raise FileNotFoundError(f"ABM v4 state panel not found: {state_path}")
        return pl.read_parquet(state_path)

    def load_supplier_weights(self) -> pl.DataFrame:
        """Load one-step updated supplier weights."""
        if not self.paths.supplier_updated_weights_path.exists():
            raise FileNotFoundError(
                f"Supplier updated weights not found: {self.paths.supplier_updated_weights_path}"
            )
        return pl.read_parquet(self.paths.supplier_updated_weights_path)

    def prepare_latest_state(
        self,
        state_panel: pl.DataFrame | None = None,
        year: int | None = None,
    ) -> pl.DataFrame:
        """Return the latest available state rows for one capability update."""
        state_panel = self.load_state_panel() if state_panel is None else state_panel
        selected_year = year or max(state_panel["Year"].drop_nulls().to_list())
        return state_panel.filter(pl.col("Year") == selected_year).unique(
            subset=["country_sector"]
        )

    def normalize_initial_capabilities(self, latest_state: pl.DataFrame) -> pl.DataFrame:
        """Select, normalize, and explicitly mark filled capability stocks."""
        prepared = latest_state.with_columns(
            pl.coalesce(
                [
                    pl.col("general_capability"),
                    pl.col("capability_export_weighted_pci"),
                    pl.col("capability_mean_pci"),
                    pl.col("active_good_count"),
                ]
            ).alias("_cap_raw"),
            pl.coalesce(
                [
                    pl.col("green_capability"),
                    pl.col("green_capability_export_share"),
                    pl.col("green_capability_share"),
                ]
            ).alias("_gcap_raw"),
        )
        prepared = prepared.with_columns(
            self._normalize_expr("_cap_raw").alias("_cap_normalized"),
            self._normalize_expr("_gcap_raw").alias("_gcap_normalized"),
        )
        cap_median = prepared["_cap_normalized"].median()
        gcap_median = prepared["_gcap_normalized"].median()
        return prepared.with_columns(
            pl.col("_cap_normalized").is_null().alias("general_capability_filled"),
            pl.col("_gcap_normalized").is_null().alias("green_capability_filled"),
            pl.col("_cap_normalized").fill_null(cap_median).alias("cap"),
            pl.col("_gcap_normalized").fill_null(gcap_median).alias("gcap"),
        )

    def compute_supplier_weighted_exposures(
        self,
        normalized_state: pl.DataFrame,
        supplier_weights: pl.DataFrame,
    ) -> pl.DataFrame:
        """Compute supplier-weighted capability and green exposure with renormalized coverage."""
        supplier_state = normalized_state.select(
            pl.col("country_sector").alias("supplier_country_sector"),
            pl.col("cap").alias("supplier_cap"),
            pl.col("gcap").alias("supplier_gcap"),
            pl.col("g_local_v4").alias("supplier_g_local_v4"),
        )
        joined = supplier_weights.join(
            supplier_state,
            on="supplier_country_sector",
            how="left",
        )
        cap_exposure = self._weighted_average_by_buyer(
            joined,
            value_column="supplier_cap",
            output_column="network_capability_exposure",
            coverage_column="supplier_capability_coverage",
        )
        green_network = self._weighted_average_by_buyer(
            joined,
            value_column="supplier_g_local_v4",
            output_column="network_green_exposure",
            coverage_column="_supplier_green_network_coverage",
        )
        green_share = self._weighted_average_by_buyer(
            joined,
            value_column="supplier_gcap",
            output_column="green_supplier_share",
            coverage_column="supplier_green_coverage",
        )
        return cap_exposure.join(
            green_network,
            on="country_sector",
            how="full",
            coalesce=True,
        ).join(
            green_share,
            on="country_sector",
            how="full",
            coalesce=True,
        ).with_columns(
            pl.max_horizontal(
                "supplier_green_coverage",
                "_supplier_green_network_coverage",
            ).alias("supplier_green_coverage")
        ).drop("_supplier_green_network_coverage")

    def compute_ecosystem_exposure(self, normalized_state: pl.DataFrame) -> pl.DataFrame:
        """Compute mean general capability within each ecosystem."""
        ecosystem_means = normalized_state.group_by("ecosystem_id").agg(
            pl.mean("cap").alias("ecosystem_capability_exposure")
        )
        return normalized_state.select("country_sector", "ecosystem_id").join(
            ecosystem_means,
            on="ecosystem_id",
            how="left",
        ).select("country_sector", "ecosystem_capability_exposure")

    def compute_capability_exposure(
        self,
        normalized_state: pl.DataFrame,
        supplier_weights: pl.DataFrame,
    ) -> pl.DataFrame:
        """Build the general capability exposure panel."""
        supplier_exposures = self.compute_supplier_weighted_exposures(
            normalized_state,
            supplier_weights,
        )
        ecosystem_exposure = self.compute_ecosystem_exposure(normalized_state)
        base = normalized_state.with_columns(
            self._normalize_expr("X_observed", transform="log1p").alias("prod_exp")
        ).select("country_sector", pl.col("Year").alias("year"), "prod_exp", "cap", "gcap")
        return (
            base.join(supplier_exposures, on="country_sector", how="left")
            .join(ecosystem_exposure, on="country_sector", how="left")
            .with_columns(
                (
                    self.exposure_weights.omega_prod_exp * pl.col("prod_exp").fill_null(0.0)
                    + self.exposure_weights.omega_network_cap
                    * pl.col("network_capability_exposure").fill_null(0.0)
                    + self.exposure_weights.omega_ecosystem_cap
                    * pl.col("ecosystem_capability_exposure").fill_null(0.0)
                ).alias("exposure_cap")
            )
            .select(
                "country_sector",
                "year",
                "prod_exp",
                "network_capability_exposure",
                "ecosystem_capability_exposure",
                "exposure_cap",
                "supplier_capability_coverage",
                "supplier_green_coverage",
                "cap",
                "gcap",
            )
        )

    def compute_green_capability_exposure(
        self,
        normalized_state: pl.DataFrame,
        exposure_panel: pl.DataFrame,
        supplier_weights: pl.DataFrame,
    ) -> pl.DataFrame:
        """Add green capability exposure components."""
        supplier_exposures = self.compute_supplier_weighted_exposures(
            normalized_state,
            supplier_weights,
        ).select(
            "country_sector",
            "network_green_exposure",
            "green_supplier_share",
        )
        return (
            exposure_panel.join(supplier_exposures, on="country_sector", how="left")
            .with_columns((pl.col("gcap") * pl.col("prod_exp")).alias("green_prod_exp"))
            .with_columns(
                (
                    self.exposure_weights.omega_green_prod_exp
                    * pl.col("green_prod_exp").fill_null(0.0)
                    + self.exposure_weights.omega_network_green
                    * pl.col("network_green_exposure").fill_null(0.0)
                    + self.exposure_weights.omega_green_supplier_share
                    * pl.col("green_supplier_share").fill_null(0.0)
                ).alias("exposure_gcap")
            )
            .select(
                "country_sector",
                "year",
                "prod_exp",
                "network_capability_exposure",
                "ecosystem_capability_exposure",
                "exposure_cap",
                "green_prod_exp",
                "network_green_exposure",
                "green_supplier_share",
                "exposure_gcap",
                "supplier_capability_coverage",
                "supplier_green_coverage",
                "cap",
                "gcap",
            )
        )

    def update_capabilities(
        self,
        normalized_state: pl.DataFrame,
        exposure_panel: pl.DataFrame,
    ) -> pl.DataFrame:
        """Apply one-step bounded capability accumulation."""
        flags = normalized_state.select(
            "country_sector",
            "general_capability_filled",
            "green_capability_filled",
        )
        update = (
            exposure_panel.join(flags, on="country_sector", how="left")
            .with_columns(
                (
                    self.config.delta_cap_param
                    * (self.config.cap_max - pl.col("cap"))
                    * (
                        self.config.k_cap
                        * (pl.col("exposure_cap") - self.config.tau_cap)
                    ).map_elements(sigmoid, return_dtype=pl.Float64)
                ).alias("delta_cap"),
                (
                    self.config.delta_gcap_param
                    * (self.config.gcap_max - pl.col("gcap"))
                    * (
                        self.config.k_gcap
                        * (pl.col("exposure_gcap") - self.config.tau_gcap)
                    ).map_elements(sigmoid, return_dtype=pl.Float64)
                ).alias("delta_gcap"),
            )
            .with_columns(
                (pl.col("cap") + pl.col("delta_cap")).alias("_cap_next_raw"),
                (pl.col("gcap") + pl.col("delta_gcap")).alias("_gcap_next_raw"),
            )
            .with_columns(
                pl.col("_cap_next_raw").clip(0.0, self.config.cap_max).alias("cap_next"),
                pl.col("_gcap_next_raw").clip(0.0, self.config.gcap_max).alias("gcap_next"),
            )
            .with_columns(
                (pl.col("cap_next") != pl.col("_cap_next_raw")).alias("cap_clipped"),
                (pl.col("gcap_next") != pl.col("_gcap_next_raw")).alias("gcap_clipped"),
            )
        )
        return update.select(
            "country_sector",
            "year",
            "cap",
            "gcap",
            "exposure_cap",
            "exposure_gcap",
            "delta_cap",
            "delta_gcap",
            "cap_next",
            "gcap_next",
            "general_capability_filled",
            "green_capability_filled",
            "cap_clipped",
            "gcap_clipped",
        )

    def build_capability_update_report(
        self,
        exposure_panel: pl.DataFrame,
        update_panel: pl.DataFrame,
    ) -> pl.DataFrame:
        """Build one-row capability update diagnostics."""
        joined = update_panel.join(
            exposure_panel.select(
                "country_sector",
                "supplier_capability_coverage",
                "supplier_green_coverage",
            ),
            on="country_sector",
            how="left",
        )
        node_count = joined.height
        return pl.DataFrame(
            {
                "year": [joined["year"].max()],
                "node_count": [node_count],
                "mean_cap": [joined["cap"].mean()],
                "mean_gcap": [joined["gcap"].mean()],
                "mean_exposure_cap": [joined["exposure_cap"].mean()],
                "mean_exposure_gcap": [joined["exposure_gcap"].mean()],
                "mean_delta_cap": [joined["delta_cap"].mean()],
                "mean_delta_gcap": [joined["delta_gcap"].mean()],
                "max_delta_cap": [joined["delta_cap"].max()],
                "max_delta_gcap": [joined["delta_gcap"].max()],
                "share_general_capability_filled": [
                    joined["general_capability_filled"].sum() / node_count if node_count else 0.0
                ],
                "share_green_capability_filled": [
                    joined["green_capability_filled"].sum() / node_count if node_count else 0.0
                ],
                "mean_supplier_capability_coverage": [
                    joined["supplier_capability_coverage"].mean()
                ],
                "mean_supplier_green_coverage": [joined["supplier_green_coverage"].mean()],
                "cap_clipped_count": [joined["cap_clipped"].sum()],
                "gcap_clipped_count": [joined["gcap_clipped"].sum()],
                "notes": [
                    (
                        "One-step general and green capability update. Missing capability "
                        "stocks are filled with within-year medians for math and explicitly "
                        "flagged. Ecosystem-specific capability stocks remain a v5 extension."
                    )
                ],
            }
        )

    def build_capability_update(
        self,
        year: int | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Build exposure panel, update panel, and diagnostics without writing."""
        state = self.load_state_panel()
        weights = self.load_supplier_weights()
        latest_state = self.prepare_latest_state(state, year=year)
        normalized_state = self.normalize_initial_capabilities(latest_state)
        exposure = self.compute_capability_exposure(normalized_state, weights)
        exposure = self.compute_green_capability_exposure(normalized_state, exposure, weights)
        exposure_output = exposure.select(
            "country_sector",
            "year",
            "prod_exp",
            "network_capability_exposure",
            "ecosystem_capability_exposure",
            "exposure_cap",
            "green_prod_exp",
            "network_green_exposure",
            "green_supplier_share",
            "exposure_gcap",
            "supplier_capability_coverage",
            "supplier_green_coverage",
        )
        update = self.update_capabilities(normalized_state, exposure)
        report = self.build_capability_update_report(exposure_output, update)
        return exposure_output, update, report

    def write_outputs(
        self,
        exposure_panel: pl.DataFrame,
        update_panel: pl.DataFrame,
        report: pl.DataFrame,
    ) -> None:
        """Write capability update outputs."""
        self.paths.interim.mkdir(parents=True, exist_ok=True)
        self.paths.diagnostics.mkdir(parents=True, exist_ok=True)
        exposure_panel.write_parquet(self.paths.capability_exposure_panel_path)
        update_panel.write_parquet(self.paths.capability_update_panel_path)
        report.write_csv(self.paths.capability_update_report_path)

    def _normalize_expr(self, column_name: str, transform: str | None = None) -> pl.Expr:
        value = pl.col(column_name).cast(pl.Float64, strict=False)
        if transform == "log1p":
            value = value.clip(0.0, None).log1p()
        min_value = value.min()
        max_value = value.max()
        return (
            pl.when(value.is_null())
            .then(None)
            .when((min_value >= 0.0) & (max_value <= 1.0))
            .then(value)
            .when(max_value > min_value)
            .then((value - min_value) / (max_value - min_value))
            .otherwise(0.0)
        )

    def _weighted_average_by_buyer(
        self,
        supplier_values: pl.DataFrame,
        value_column: str,
        output_column: str,
        coverage_column: str,
    ) -> pl.DataFrame:
        valid = supplier_values.filter(pl.col(value_column).is_not_null())
        weighted = (
            valid.group_by("buyer_country_sector")
            .agg(
                (pl.col("updated_weight") * pl.col(value_column)).sum().alias("_weighted_sum"),
                pl.sum("updated_weight").alias("_covered_weight"),
            )
            .with_columns(
                pl.when(pl.col("_covered_weight") > 0)
                .then(pl.col("_weighted_sum") / pl.col("_covered_weight"))
                .otherwise(None)
                .alias(output_column),
                pl.col("_covered_weight").alias(coverage_column),
            )
            .select(
                pl.col("buyer_country_sector").alias("country_sector"),
                output_column,
                coverage_column,
            )
        )
        return weighted
