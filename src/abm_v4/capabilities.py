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


@dataclass(frozen=True)
class IOCapabilityModelResult:
    """Outputs from the IO-derived capability model."""

    state_panel: pl.DataFrame
    lambda_calibration: pl.DataFrame
    model_report: pl.DataFrame
    coverage_by_sector: pl.DataFrame
    coverage_by_source: pl.DataFrame


@dataclass(frozen=True)
class IOCapabilityRobustnessResult:
    """Diagnostic-only robustness outputs for the IO capability model."""

    robustness: pl.DataFrame
    threshold_sensitivity: pl.DataFrame
    downstream_audit: pl.DataFrame


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
                    pl.col("general_capability_model")
                    if "general_capability_model" in latest_state.columns
                    else pl.lit(None),
                    pl.col("general_capability"),
                    pl.col("capability_export_weighted_pci"),
                    pl.col("capability_mean_pci"),
                    pl.col("active_good_count"),
                ]
            ).alias("_cap_raw"),
            pl.coalesce(
                [
                    pl.col("green_capability_model")
                    if "green_capability_model" in latest_state.columns
                    else pl.lit(None),
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
            pl.col("_cap_normalized").is_null().alias("capability_model_unavailable_flag"),
            (
                pl.col("general_capability_source")
                if "general_capability_source" in prepared.columns
                else pl.lit("legacy_or_unavailable")
            ).alias("general_capability_source"),
            (
                pl.col("green_capability_source")
                if "green_capability_source" in prepared.columns
                else pl.lit("legacy_or_unavailable")
            ).alias("green_capability_source"),
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
            "capability_model_unavailable_flag",
            "general_capability_source",
            "green_capability_source",
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
            "capability_model_unavailable_flag",
            "general_capability_source",
            "green_capability_source",
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
                        "model stocks remain explicitly flagged before any within-year "
                        "median math fallback. Ecosystem-specific capability stocks remain a v5 extension."
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


class IOCapabilityBuilder:
    """Build source-aware Atlas/IO capability model fields for ABM v4."""

    def __init__(
        self,
        paths: ABMV4Paths,
        start_year: int = 1995,
        end_year: int = 2016,
        config: CapabilityConfig | None = None,
    ) -> None:
        self.paths = paths
        self.start_year = start_year
        self.end_year = end_year
        self.config = config or CapabilityConfig()

    def load_state_panel(self) -> pl.DataFrame:
        """Load the ABM v4 state panel."""
        state_path = self.paths.state_panel_path(self.start_year, self.end_year)
        if not state_path.exists():
            raise FileNotFoundError(f"ABM v4 state panel not found: {state_path}")
        return pl.read_parquet(state_path)

    def load_supplier_weights(self) -> pl.DataFrame:
        """Load compact updated supplier weights."""
        if not self.paths.supplier_updated_weights_path.exists():
            raise FileNotFoundError(
                f"Supplier updated weights not found: {self.paths.supplier_updated_weights_path}"
            )
        return pl.read_parquet(self.paths.supplier_updated_weights_path)

    def prepare_atlas_observed_flags(self, state_panel: pl.DataFrame) -> pl.DataFrame:
        """Mark actual Atlas-observed capability rows, separate from model values."""
        if "general_capability_source" in state_panel.columns:
            general_observed = (
                pl.col("general_capability").is_not_null()
                & (pl.col("general_capability_source") != "io_imputed")
                & (pl.col("general_capability_source") != "unavailable")
            )
        else:
            general_observed = pl.col("general_capability").is_not_null()
        if "green_capability_source" in state_panel.columns:
            green_observed = (
                pl.col("green_capability").is_not_null()
                & (pl.col("green_capability_source") != "io_imputed")
                & (pl.col("green_capability_source") != "unavailable")
            )
        else:
            green_observed = pl.col("green_capability").is_not_null()
        return state_panel.with_columns(
            general_observed.alias("general_capability_atlas_observed"),
            green_observed.alias("green_capability_atlas_observed"),
        )

    def compute_upstream_observed_capability_exposure(
        self,
        state_panel: pl.DataFrame,
        supplier_weights: pl.DataFrame,
    ) -> pl.DataFrame:
        """Compute buyer upstream exposure using only Atlas-observed supplier capabilities."""
        supplier_state = state_panel.select(
            pl.col("country_sector").alias("supplier_country_sector"),
            "Year",
            pl.col("general_capability").alias("supplier_general_capability"),
            pl.col("green_capability").alias("supplier_green_capability"),
            pl.col("general_capability_atlas_observed").alias("supplier_general_observed"),
            pl.col("green_capability_atlas_observed").alias("supplier_green_observed"),
        )
        buyer_years = state_panel.select(
            pl.col("country_sector").alias("buyer_country_sector"),
            "Year",
        )
        joined = (
            buyer_years.join(supplier_weights, on="buyer_country_sector", how="left")
            .join(supplier_state, on=["supplier_country_sector", "Year"], how="left")
        )
        general = self._weighted_exposure(
            joined.filter(pl.col("supplier_general_observed")),
            group_columns=("buyer_country_sector", "Year"),
            weight_column="updated_weight",
            value_column="supplier_general_capability",
            exposure_column="general_capability_io_upstream",
            coverage_column="general_capability_upstream_coverage",
        )
        green = self._weighted_exposure(
            joined.filter(pl.col("supplier_green_observed")),
            group_columns=("buyer_country_sector", "Year"),
            weight_column="updated_weight",
            value_column="supplier_green_capability",
            exposure_column="green_capability_io_upstream",
            coverage_column="green_capability_upstream_coverage",
        )
        return (
            state_panel.select("country_sector", "Year")
            .join(
                general.rename({"buyer_country_sector": "country_sector"}),
                on=["country_sector", "Year"],
                how="left",
            )
            .join(
                green.rename({"buyer_country_sector": "country_sector"}),
                on=["country_sector", "Year"],
                how="left",
            )
        )

    def compute_downstream_observed_capability_exposure(
        self,
        state_panel: pl.DataFrame,
        supplier_weights: pl.DataFrame,
    ) -> pl.DataFrame:
        """Compute downstream exposure from compact buyer links as downstream sales-share proxy."""
        buyer_state = state_panel.select(
            pl.col("country_sector").alias("buyer_country_sector"),
            "Year",
            pl.col("general_capability").alias("buyer_general_capability"),
            pl.col("green_capability").alias("buyer_green_capability"),
            pl.col("general_capability_atlas_observed").alias("buyer_general_observed"),
            pl.col("green_capability_atlas_observed").alias("buyer_green_observed"),
        )
        years = state_panel.select("Year").unique()
        downstream_links = (
            supplier_weights.join(years, how="cross")
            .with_columns(
                (
                    pl.col("updated_weight")
                    / pl.col("updated_weight").sum().over(["supplier_country_sector", "Year"])
                ).alias("downstream_sales_share")
            )
            .join(buyer_state, on=["buyer_country_sector", "Year"], how="left")
        )
        general = self._weighted_exposure(
            downstream_links.filter(pl.col("buyer_general_observed")),
            group_columns=("supplier_country_sector", "Year"),
            weight_column="downstream_sales_share",
            value_column="buyer_general_capability",
            exposure_column="general_capability_io_downstream",
            coverage_column="general_capability_downstream_coverage",
        )
        green = self._weighted_exposure(
            downstream_links.filter(pl.col("buyer_green_observed")),
            group_columns=("supplier_country_sector", "Year"),
            weight_column="downstream_sales_share",
            value_column="buyer_green_capability",
            exposure_column="green_capability_io_downstream",
            coverage_column="green_capability_downstream_coverage",
        )
        return (
            state_panel.select("country_sector", "Year")
            .join(
                general.rename({"supplier_country_sector": "country_sector"}),
                on=["country_sector", "Year"],
                how="left",
            )
            .join(
                green.rename({"supplier_country_sector": "country_sector"}),
                on=["country_sector", "Year"],
                how="left",
            )
        )

    def calibrate_lambda(
        self,
        exposure_panel: pl.DataFrame,
        capability_type: str,
    ) -> tuple[float, pl.DataFrame]:
        """Grid-search lambda_up using Atlas-observed nodes."""
        value_column = f"{capability_type}_capability"
        observed_column = f"{capability_type}_capability_atlas_observed"
        up_column = f"{capability_type}_capability_io_upstream"
        down_column = f"{capability_type}_capability_io_downstream"
        valid = exposure_panel.filter(
            pl.col(observed_column)
            & pl.col(value_column).is_not_null()
            & pl.col(up_column).is_not_null()
            & pl.col(down_column).is_not_null()
            & pl.col(value_column).is_finite()
            & pl.col(up_column).is_finite()
            & pl.col(down_column).is_finite()
        )
        if valid.is_empty():
            return 1.0, pl.DataFrame(
                {
                    "capability_type": [capability_type],
                    "lambda_up": [1.0],
                    "lambda_down": [0.0],
                    "mae": [None],
                    "rmse": [None],
                    "train_or_validation": ["fallback_upstream_only"],
                    "observations": [0],
                    "selected": [True],
                }
            )
        lambdas = [
            round(index * self.config.io_capability_lambda_grid_step, 10)
            for index in range(int(1 / self.config.io_capability_lambda_grid_step) + 1)
        ]
        rows = []
        for lambda_up in lambdas:
            scored = valid.with_columns(
                (
                    lambda_up * pl.col(up_column)
                    + (1 - lambda_up) * pl.col(down_column)
                ).alias("_pred")
            ).with_columns((pl.col("_pred") - pl.col(value_column)).alias("_err"))
            mae = scored["_err"].abs().mean()
            rmse = math.sqrt(scored.select((pl.col("_err") ** 2).mean()).item())
            rows.append(
                {
                    "capability_type": capability_type,
                    "lambda_up": lambda_up,
                    "lambda_down": 1 - lambda_up,
                    "mae": mae,
                    "rmse": rmse,
                    "train_or_validation": "all_observed",
                    "observations": valid.height,
                    "selected": False,
                }
            )
        calibration = pl.DataFrame(rows)
        best = calibration.sort(["mae", "rmse", "lambda_up"]).row(0, named=True)
        selected_lambda = float(best["lambda_up"])
        calibration = calibration.with_columns(
            (pl.col("lambda_up") == selected_lambda).alias("selected")
        )
        return selected_lambda, calibration

    def compute_io_capability(
        self,
        exposure_panel: pl.DataFrame,
        lambda_general_up: float,
        lambda_green_up: float,
    ) -> pl.DataFrame:
        """Compute calibrated IO capability values and coverage."""
        return exposure_panel.with_columns(
            (
                lambda_general_up * pl.col("general_capability_io_upstream")
                + (1 - lambda_general_up) * pl.col("general_capability_io_downstream")
            ).alias("_general_io_both"),
            (
                lambda_green_up * pl.col("green_capability_io_upstream")
                + (1 - lambda_green_up) * pl.col("green_capability_io_downstream")
            ).alias("_green_io_both"),
            (
                lambda_general_up * pl.col("general_capability_upstream_coverage").fill_null(0.0)
                + (1 - lambda_general_up)
                * pl.col("general_capability_downstream_coverage").fill_null(0.0)
            ).alias("general_capability_io_coverage"),
            (
                lambda_green_up * pl.col("green_capability_upstream_coverage").fill_null(0.0)
                + (1 - lambda_green_up)
                * pl.col("green_capability_downstream_coverage").fill_null(0.0)
            ).alias("green_capability_io_coverage"),
        ).with_columns(
            pl.coalesce(["_general_io_both", "general_capability_io_upstream"]).alias(
                "general_capability_io"
            ),
            pl.coalesce(["_green_io_both", "green_capability_io_upstream"]).alias(
                "green_capability_io"
            ),
        ).with_columns(
            pl.col("general_capability_io").fill_nan(None),
            pl.col("green_capability_io").fill_nan(None),
            pl.col("general_capability_io_coverage").fill_nan(None),
            pl.col("green_capability_io_coverage").fill_nan(None),
        ).drop(["_general_io_both", "_green_io_both"])

    def assign_capability_model(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Assign Atlas, IO-imputed, or unavailable capability model values."""
        gamma = self.config.io_capability_min_coverage
        return panel.with_columns(
            pl.when(pl.col("general_capability_atlas_observed"))
            .then(pl.col("general_capability"))
            .when(
                pl.col("general_capability_io").is_not_null()
                & (pl.col("general_capability_io_coverage") >= gamma)
            )
            .then(pl.col("general_capability_io"))
            .otherwise(None)
            .alias("general_capability_model"),
            pl.when(pl.col("green_capability_atlas_observed"))
            .then(pl.col("green_capability"))
            .when(
                pl.col("green_capability_io").is_not_null()
                & (pl.col("green_capability_io_coverage") >= gamma)
            )
            .then(pl.col("green_capability_io"))
            .otherwise(None)
            .alias("green_capability_model"),
            pl.when(pl.col("general_capability_atlas_observed"))
            .then(pl.lit("atlas_observed"))
            .when(
                pl.col("general_capability_io").is_not_null()
                & (pl.col("general_capability_io_coverage") >= gamma)
            )
            .then(pl.lit("io_imputed"))
            .otherwise(pl.lit("unavailable"))
            .alias("general_capability_source"),
            pl.when(pl.col("green_capability_atlas_observed"))
            .then(pl.lit("atlas_observed"))
            .when(
                pl.col("green_capability_io").is_not_null()
                & (pl.col("green_capability_io_coverage") >= gamma)
            )
            .then(pl.lit("io_imputed"))
            .otherwise(pl.lit("unavailable"))
            .alias("green_capability_source"),
        )

    def build_io_capability_report(
        self,
        panel: pl.DataFrame,
        lambda_general_up: float,
        lambda_green_up: float,
    ) -> pl.DataFrame:
        """Build a one-row IO capability model report."""
        selected_year = panel["Year"].max()
        latest = panel.filter(pl.col("Year") == selected_year)
        return pl.DataFrame(
            {
                "selected_year": [selected_year],
                "node_count": [latest.height],
                "atlas_observed_general_count": [
                    latest.filter(pl.col("general_capability_source") == "atlas_observed").height
                ],
                "io_imputed_general_count": [
                    latest.filter(pl.col("general_capability_source") == "io_imputed").height
                ],
                "unavailable_general_count": [
                    latest.filter(pl.col("general_capability_source") == "unavailable").height
                ],
                "atlas_observed_green_count": [
                    latest.filter(pl.col("green_capability_source") == "atlas_observed").height
                ],
                "io_imputed_green_count": [
                    latest.filter(pl.col("green_capability_source") == "io_imputed").height
                ],
                "unavailable_green_count": [
                    latest.filter(pl.col("green_capability_source") == "unavailable").height
                ],
                "selected_lambda_general_up": [lambda_general_up],
                "selected_lambda_green_up": [lambda_green_up],
                "mean_general_io_coverage": [latest["general_capability_io_coverage"].mean()],
                "mean_green_io_coverage": [latest["green_capability_io_coverage"].mean()],
                "min_coverage_threshold": [self.config.io_capability_min_coverage],
                "downstream_available": [self.config.io_capability_use_downstream],
                "notes": [
                    "IO-imputed capability is a network-embedded proxy, not observed Atlas capability. Downstream uses compact supplier-weight buyer links as a v4 proxy."
                ],
            }
        )

    def build_coverage_by_sector(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Summarize source shares and coverage by sector for the latest year."""
        latest = panel.filter(pl.col("Year") == panel["Year"].max())
        return (
            latest.group_by("Sector")
            .agg(
                pl.len().alias("rows"),
                (pl.col("general_capability_source") == "atlas_observed").mean().alias(
                    "atlas_observed_general_share"
                ),
                (pl.col("general_capability_source") == "io_imputed").mean().alias(
                    "io_imputed_general_share"
                ),
                (pl.col("general_capability_source") == "unavailable").mean().alias(
                    "unavailable_general_share"
                ),
                (pl.col("green_capability_source") == "atlas_observed").mean().alias(
                    "atlas_observed_green_share"
                ),
                (pl.col("green_capability_source") == "io_imputed").mean().alias(
                    "io_imputed_green_share"
                ),
                (pl.col("green_capability_source") == "unavailable").mean().alias(
                    "unavailable_green_share"
                ),
                pl.col("general_capability_io_coverage").mean().alias(
                    "mean_general_io_coverage"
                ),
                pl.col("green_capability_io_coverage").mean().alias(
                    "mean_green_io_coverage"
                ),
            )
            .sort("Sector")
        )

    def build_coverage_by_source(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Summarize model capability values by source."""
        latest = panel.filter(pl.col("Year") == panel["Year"].max())
        rows = []
        for capability_type in ("general", "green"):
            source_col = f"{capability_type}_capability_source"
            value_col = f"{capability_type}_capability_model"
            coverage_col = f"{capability_type}_capability_io_coverage"
            total = latest.height
            for source in ("atlas_observed", "io_imputed", "unavailable"):
                subset = latest.filter(pl.col(source_col) == source)
                rows.append(
                    {
                        "capability_type": capability_type,
                        "source": source,
                        "rows": subset.height,
                        "share": subset.height / total if total else 0.0,
                        "mean_value": subset[value_col].mean() if subset.height else None,
                        "median_value": subset[value_col].median() if subset.height else None,
                        "mean_coverage": subset[coverage_col].mean() if subset.height else None,
                    }
                )
        return pl.DataFrame(rows)

    def build_io_capability_model(self) -> IOCapabilityModelResult:
        """Build state-panel capability model fields and diagnostics."""
        exposure = self.build_io_exposure_panel()
        lambda_general, calibration_general = self.calibrate_lambda(exposure, "general")
        lambda_green, calibration_green = self.calibrate_lambda(exposure, "green")
        modeled = self.compute_io_capability(exposure, lambda_general, lambda_green)
        modeled = self.assign_capability_model(modeled)
        calibration = pl.concat([calibration_general, calibration_green], how="vertical")
        return IOCapabilityModelResult(
            state_panel=modeled,
            lambda_calibration=calibration,
            model_report=self.build_io_capability_report(modeled, lambda_general, lambda_green),
            coverage_by_sector=self.build_coverage_by_sector(modeled),
            coverage_by_source=self.build_coverage_by_source(modeled),
        )

    def build_io_exposure_panel(self) -> pl.DataFrame:
        """Build upstream/downstream observed-neighbour exposure panel."""
        state = self._drop_existing_io_capability_columns(
            self.prepare_atlas_observed_flags(self.load_state_panel())
        )
        weights = self.load_supplier_weights()
        upstream = self.compute_upstream_observed_capability_exposure(state, weights)
        downstream = self.compute_downstream_observed_capability_exposure(state, weights)
        return (
            state.join(upstream, on=["country_sector", "Year"], how="left")
            .join(downstream, on=["country_sector", "Year"], how="left")
        )

    def _drop_existing_io_capability_columns(self, state: pl.DataFrame) -> pl.DataFrame:
        """Drop previously generated IO capability columns before recomputing them."""
        generated_prefixes = (
            "general_capability_io",
            "green_capability_io",
            "general_capability_upstream",
            "green_capability_upstream",
            "general_capability_downstream",
            "green_capability_downstream",
        )
        generated_columns = {
            "general_capability_model",
            "green_capability_model",
        }
        drop_columns = [
            column
            for column in state.columns
            if column in generated_columns
            or any(column.startswith(prefix) for prefix in generated_prefixes)
        ]
        if not drop_columns:
            return state
        return state.drop(drop_columns)

    def build_io_capability_robustness(self) -> IOCapabilityRobustnessResult:
        """Build robustness and downstream proxy audit diagnostics without writing state."""
        exposure = self.build_io_exposure_panel()
        lambda_general, _ = self.calibrate_lambda(exposure, "general")
        lambda_green, _ = self.calibrate_lambda(exposure, "green")
        robustness = self.build_robustness_report(
            exposure=exposure,
            lambda_general_up=lambda_general,
            lambda_green_up=lambda_green,
            gamma=self.config.io_capability_min_coverage,
        )
        threshold = self.build_threshold_sensitivity_report(
            exposure=exposure,
            lambda_general_up=lambda_general,
            lambda_green_up=lambda_green,
            gamma_values=(0.1, 0.3, 0.5),
        )
        downstream = self.build_downstream_exposure_audit(exposure)
        return IOCapabilityRobustnessResult(
            robustness=robustness,
            threshold_sensitivity=threshold,
            downstream_audit=downstream,
        )

    def build_robustness_report(
        self,
        exposure: pl.DataFrame,
        lambda_general_up: float,
        lambda_green_up: float,
        gamma: float,
    ) -> pl.DataFrame:
        """Compare atlas-only, upstream-only, downstream-only, and calibrated IO specs."""
        rows = []
        specs = {
            "atlas_only": (None, None),
            "upstream_only": (1.0, 1.0),
            "downstream_only": (0.0, 0.0),
            "calibrated_io": (lambda_general_up, lambda_green_up),
        }
        latest_year = exposure["Year"].max()
        for spec_name, lambdas in specs.items():
            modeled = self._assign_spec_model(exposure, spec_name, lambdas, gamma)
            latest = modeled.filter(pl.col("Year") == latest_year)
            for capability_type in ("general", "green"):
                value_col = f"{capability_type}_capability_model"
                source_col = f"{capability_type}_capability_source"
                coverage_col = f"{capability_type}_capability_io_coverage"
                mae, rmse, observations = self._validation_error(
                    modeled,
                    capability_type=capability_type,
                    spec_name=spec_name,
                    lambdas=lambdas,
                )
                rows.append(
                    {
                        "specification": spec_name,
                        "capability_type": capability_type,
                        "coverage": latest[value_col].is_not_null().sum() / latest.height
                        if latest.height
                        else 0.0,
                        "unavailable_share": (
                            latest.filter(pl.col(source_col) == "unavailable").height
                            / latest.height
                            if latest.height
                            else 0.0
                        ),
                        "mean_capability_model": latest[value_col].mean(),
                        "median_capability_model": latest[value_col].median(),
                        "validation_mae": mae,
                        "validation_rmse": rmse,
                        "validation_observations": observations,
                        "mean_io_coverage": latest[coverage_col].mean(),
                    }
                )
        return pl.DataFrame(rows)

    def build_threshold_sensitivity_report(
        self,
        exposure: pl.DataFrame,
        lambda_general_up: float,
        lambda_green_up: float,
        gamma_values: tuple[float, ...],
    ) -> pl.DataFrame:
        """Report calibrated-IO source counts under alternative coverage thresholds."""
        rows = []
        latest_year = exposure["Year"].max()
        for gamma in gamma_values:
            modeled = self._assign_spec_model(
                exposure,
                "calibrated_io",
                (lambda_general_up, lambda_green_up),
                gamma,
            )
            latest = modeled.filter(pl.col("Year") == latest_year)
            for capability_type in ("general", "green"):
                source_col = f"{capability_type}_capability_source"
                coverage_col = f"{capability_type}_capability_io_coverage"
                mae, rmse, observations = self._validation_error(
                    modeled,
                    capability_type=capability_type,
                    spec_name="calibrated_io",
                    lambdas=(lambda_general_up, lambda_green_up),
                )
                rows.append(
                    {
                        "gamma": gamma,
                        "capability_type": capability_type,
                        "io_imputed_count": latest.filter(pl.col(source_col) == "io_imputed").height,
                        "unavailable_count": latest.filter(pl.col(source_col) == "unavailable").height,
                        "validation_mae": mae,
                        "validation_rmse": rmse,
                        "validation_observations": observations,
                        "mean_io_coverage": latest[coverage_col].mean(),
                    }
                )
        return pl.DataFrame(rows)

    def build_downstream_exposure_audit(self, exposure: pl.DataFrame) -> pl.DataFrame:
        """Audit the compact downstream exposure proxy and document raw-T limitation."""
        latest = exposure.filter(pl.col("Year") == exposure["Year"].max())
        return pl.DataFrame(
            {
                "selected_year": [latest["Year"].max()],
                "compact_proxy": ["supplier_updated_weights_downstream_sales_share"],
                "compact_nodes": [latest.height],
                "general_downstream_available": [
                    latest["general_capability_io_downstream"].is_not_null().sum()
                ],
                "green_downstream_available": [
                    latest["green_capability_io_downstream"].is_not_null().sum()
                ],
                "mean_general_downstream_coverage": [
                    latest["general_capability_downstream_coverage"].mean()
                ],
                "mean_green_downstream_coverage": [
                    latest["green_capability_downstream_coverage"].mean()
                ],
                "raw_t_comparison_status": [
                    "not_run_full_raw_t_aggregation_in_phase_9d"
                ],
                "raw_t_comparison_rows": [None],
                "correlation_general_compact_vs_raw_t": [None],
                "correlation_green_compact_vs_raw_t": [None],
                "notes": [
                    (
                        "Phase 9D audits the compact downstream proxy used by v4. "
                        "Full raw-T downstream aggregation over the 531M edge panel was not run "
                        "to avoid a heavy diagnostic pass before multi-year simulation; this remains "
                        "a known robustness caveat."
                    )
                ],
            }
        )

    def write_robustness_outputs(self, result: IOCapabilityRobustnessResult) -> None:
        """Write diagnostic-only IO capability robustness outputs."""
        self.paths.diagnostics.mkdir(parents=True, exist_ok=True)
        result.robustness.write_csv(self.paths.io_capability_robustness_path)
        result.threshold_sensitivity.write_csv(
            self.paths.io_capability_threshold_sensitivity_path
        )
        result.downstream_audit.write_csv(self.paths.io_downstream_exposure_audit_path)

    def write_outputs(self, result: IOCapabilityModelResult) -> None:
        """Write the IO capability model fields and diagnostics."""
        self.paths.inputs.mkdir(parents=True, exist_ok=True)
        self.paths.diagnostics.mkdir(parents=True, exist_ok=True)
        result.state_panel.write_parquet(self.paths.state_panel_path(self.start_year, self.end_year))
        result.lambda_calibration.write_csv(self.paths.io_capability_lambda_calibration_path)
        result.model_report.write_csv(self.paths.io_capability_model_report_path)
        result.coverage_by_sector.write_csv(self.paths.io_capability_coverage_by_sector_path)
        result.coverage_by_source.write_csv(self.paths.io_capability_coverage_by_source_path)

    def _assign_spec_model(
        self,
        exposure: pl.DataFrame,
        spec_name: str,
        lambdas: tuple[float, float] | tuple[None, None] | None,
        gamma: float,
    ) -> pl.DataFrame:
        if spec_name == "atlas_only":
            return exposure.with_columns(
                pl.when(pl.col("general_capability_atlas_observed"))
                .then(pl.col("general_capability"))
                .otherwise(None)
                .alias("general_capability_model"),
                pl.when(pl.col("green_capability_atlas_observed"))
                .then(pl.col("green_capability"))
                .otherwise(None)
                .alias("green_capability_model"),
                pl.when(pl.col("general_capability_atlas_observed"))
                .then(pl.lit("atlas_observed"))
                .otherwise(pl.lit("unavailable"))
                .alias("general_capability_source"),
                pl.when(pl.col("green_capability_atlas_observed"))
                .then(pl.lit("atlas_observed"))
                .otherwise(pl.lit("unavailable"))
                .alias("green_capability_source"),
                pl.lit(0.0).alias("general_capability_io_coverage"),
                pl.lit(0.0).alias("green_capability_io_coverage"),
            )
        lambda_general, lambda_green = lambdas or (1.0, 1.0)
        modeled = self.compute_io_capability(exposure, lambda_general, lambda_green)
        old_gamma = self.config.io_capability_min_coverage
        if gamma == old_gamma:
            return self.assign_capability_model(modeled)
        return modeled.with_columns(
            pl.when(pl.col("general_capability_atlas_observed"))
            .then(pl.col("general_capability"))
            .when(
                pl.col("general_capability_io").is_not_null()
                & (pl.col("general_capability_io_coverage") >= gamma)
            )
            .then(pl.col("general_capability_io"))
            .otherwise(None)
            .alias("general_capability_model"),
            pl.when(pl.col("green_capability_atlas_observed"))
            .then(pl.col("green_capability"))
            .when(
                pl.col("green_capability_io").is_not_null()
                & (pl.col("green_capability_io_coverage") >= gamma)
            )
            .then(pl.col("green_capability_io"))
            .otherwise(None)
            .alias("green_capability_model"),
            pl.when(pl.col("general_capability_atlas_observed"))
            .then(pl.lit("atlas_observed"))
            .when(
                pl.col("general_capability_io").is_not_null()
                & (pl.col("general_capability_io_coverage") >= gamma)
            )
            .then(pl.lit("io_imputed"))
            .otherwise(pl.lit("unavailable"))
            .alias("general_capability_source"),
            pl.when(pl.col("green_capability_atlas_observed"))
            .then(pl.lit("atlas_observed"))
            .when(
                pl.col("green_capability_io").is_not_null()
                & (pl.col("green_capability_io_coverage") >= gamma)
            )
            .then(pl.lit("io_imputed"))
            .otherwise(pl.lit("unavailable"))
            .alias("green_capability_source"),
        )

    def _validation_error(
        self,
        exposure: pl.DataFrame,
        capability_type: str,
        spec_name: str,
        lambdas: tuple[float, float] | tuple[None, None] | None,
    ) -> tuple[float | None, float | None, int]:
        value_col = f"{capability_type}_capability"
        observed_col = f"{capability_type}_capability_atlas_observed"
        if spec_name == "atlas_only":
            return 0.0, 0.0, exposure.filter(pl.col(observed_col)).height
        if spec_name == "upstream_only":
            pred_expr = pl.col(f"{capability_type}_capability_io_upstream")
        elif spec_name == "downstream_only":
            pred_expr = pl.col(f"{capability_type}_capability_io_downstream")
        else:
            lambda_up = (lambdas or (1.0, 1.0))[0 if capability_type == "general" else 1]
            pred_expr = (
                lambda_up * pl.col(f"{capability_type}_capability_io_upstream")
                + (1 - lambda_up) * pl.col(f"{capability_type}_capability_io_downstream")
            )
        scored = (
            exposure.with_columns(pred_expr.alias("_prediction"))
            .filter(
                pl.col(observed_col)
                & pl.col(value_col).is_not_null()
                & pl.col("_prediction").is_not_null()
                & pl.col(value_col).is_finite()
                & pl.col("_prediction").is_finite()
            )
            .with_columns((pl.col("_prediction") - pl.col(value_col)).alias("_error"))
        )
        if scored.is_empty():
            return None, None, 0
        mae = scored["_error"].abs().mean()
        rmse = math.sqrt(scored.select((pl.col("_error") ** 2).mean()).item())
        return mae, rmse, scored.height

    def _weighted_exposure(
        self,
        frame: pl.DataFrame,
        group_columns: tuple[str, str],
        weight_column: str,
        value_column: str,
        exposure_column: str,
        coverage_column: str,
    ) -> pl.DataFrame:
        if frame.is_empty():
            return pl.DataFrame(
                {
                    group_columns[0]: [],
                    group_columns[1]: [],
                    exposure_column: [],
                    coverage_column: [],
                }
            )
        return (
            frame.group_by(list(group_columns))
            .agg(
                (pl.col(weight_column) * pl.col(value_column)).sum().alias("_weighted_sum"),
                pl.col(weight_column).sum().alias(coverage_column),
            )
            .with_columns(
                pl.when(pl.col(coverage_column) > 0)
                .then(pl.col("_weighted_sum") / pl.col(coverage_column))
                .otherwise(None)
                .alias(exposure_column)
            )
            .with_columns(
                pl.col(exposure_column).fill_nan(None),
                pl.col(coverage_column).fill_nan(None),
            )
            .drop("_weighted_sum")
        )
