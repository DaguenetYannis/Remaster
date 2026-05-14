from __future__ import annotations

import numpy as np
import polars as pl

from src.abm_v4.paths import ABMV4Paths


def input_feasibility(
    total_input_available: float,
    total_input_required: float,
    epsilon: float,
) -> float:
    """Compute input feasibility from explicit requirements and availability."""
    return total_input_available / (total_input_required + epsilon)


def realized_output(
    desired_output: float,
    feasibility: float,
) -> float:
    """Compute realized output from desired output and input feasibility."""
    return desired_output * min(1.0, feasibility)


class ProductionFeasibilityEngine:
    """Build one-step production feasibility diagnostics from Eora T and supplier weights."""

    def __init__(
        self,
        paths: ABMV4Paths,
        start_year: int = 1995,
        end_year: int = 2016,
        epsilon: float = 1e-9,
    ) -> None:
        self.paths = paths
        self.start_year = start_year
        self.end_year = end_year
        self.epsilon = epsilon

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

    def load_eora_T_for_year(self, year: int) -> pl.DataFrame:
        """Load raw Eora T for a year, preserving row=supplier and column=buyer orientation."""
        t_path = self.paths.data_root / "parquet" / str(year) / "T.parquet"
        if not t_path.exists():
            raise FileNotFoundError(f"Eora T matrix not found: {t_path}")
        matrix = pl.read_parquet(t_path)
        if "__index_level_0__" not in matrix.columns:
            raise ValueError(f"Eora T matrix lacks supplier row labels: {t_path}")
        return matrix

    def prepare_latest_state(
        self,
        state_panel: pl.DataFrame | None = None,
        year: int | None = None,
    ) -> pl.DataFrame:
        """Return latest available node state."""
        state_panel = self.load_state_panel() if state_panel is None else state_panel
        selected_year = year or max(state_panel["Year"].drop_nulls().to_list())
        return (
            state_panel.filter(pl.col("Year") == selected_year)
            .select("country_sector", "Year", "X_observed")
            .unique(subset=["country_sector"])
        )

    def compute_technical_coefficients(
        self,
        t_matrix: pl.DataFrame,
        latest_state: pl.DataFrame,
    ) -> pl.DataFrame:
        """Compute compact buyer-level total input coefficients from T_{supplier,buyer}."""
        buyer_outputs = dict(
            latest_state.select("country_sector", "X_observed").iter_rows()
        )
        value_columns = [
            column_name
            for column_name in t_matrix.columns
            if column_name != "__index_level_0__"
        ]
        totals = t_matrix.select(value_columns).sum()
        rows: list[dict[str, float | str]] = []
        for buyer in value_columns:
            buyer_output = buyer_outputs.get(buyer)
            transaction_total = totals[buyer].item()
            coefficient = (
                transaction_total / (buyer_output + self.epsilon)
                if buyer_output is not None
                else None
            )
            rows.append(
                {
                    "country_sector": buyer,
                    "input_coefficient_total": coefficient,
                    "raw_input_transaction_total": transaction_total,
                }
            )
        return pl.DataFrame(rows)

    def compute_input_requirements(
        self,
        latest_state: pl.DataFrame,
        technical_coefficients: pl.DataFrame,
    ) -> pl.DataFrame:
        """Compute one-step total input requirements using X_desired = X_observed."""
        return (
            latest_state.rename({"Year": "year"})
            .with_columns(pl.col("X_observed").alias("X_desired"))
            .join(technical_coefficients, on="country_sector", how="left")
            .with_columns(
                (
                    pl.col("input_coefficient_total").fill_null(0.0)
                    * pl.col("X_desired").fill_null(0.0)
                ).alias("input_required_total")
            )
        )

    def compute_input_availability(
        self,
        input_requirements: pl.DataFrame,
        supplier_weights: pl.DataFrame,
    ) -> pl.DataFrame:
        """Allocate buyer input requirements over updated supplier weights."""
        requirement_lookup = input_requirements.select(
            "country_sector",
            "year",
            "X_observed",
            "X_desired",
            "input_required_total",
        ).rename({"country_sector": "buyer_country_sector"})
        return supplier_weights.join(
            requirement_lookup,
            on="buyer_country_sector",
            how="left",
        ).with_columns(
            (
                pl.col("updated_weight").fill_null(0.0)
                * pl.col("input_required_total").fill_null(0.0)
            ).alias("input_avail")
        )

    def compute_supplier_capacity_pressure(
        self,
        input_availability: pl.DataFrame,
        latest_state: pl.DataFrame,
    ) -> pl.DataFrame:
        """Compute supplier capacity pressure from observed supplier output proxy."""
        supplier_capacity = latest_state.select(
            pl.col("country_sector").alias("supplier_country_sector"),
            pl.col("X_observed").alias("supplier_capacity_proxy"),
        )
        return input_availability.join(
            supplier_capacity,
            on="supplier_country_sector",
            how="left",
        ).with_columns(
            (
                pl.col("input_avail")
                / (pl.col("supplier_capacity_proxy").fill_null(0.0) + self.epsilon)
            ).alias("supplier_capacity_pressure")
        )

    def build_feasibility_panel(
        self,
        year: int | None = None,
    ) -> pl.DataFrame:
        """Build buyer-level one-step production feasibility diagnostics."""
        state_panel = self.load_state_panel()
        latest_state = self.prepare_latest_state(state_panel, year=year)
        selected_year = latest_state["Year"].max()
        t_matrix = self.load_eora_T_for_year(selected_year)
        supplier_weights = self.load_supplier_weights()
        technical_coefficients = self.compute_technical_coefficients(t_matrix, latest_state)
        input_requirements = self.compute_input_requirements(
            latest_state,
            technical_coefficients,
        )
        input_availability = self.compute_input_availability(
            input_requirements,
            supplier_weights,
        )
        pressure = self.compute_supplier_capacity_pressure(input_availability, latest_state)
        buyer_summary = pressure.group_by("buyer_country_sector").agg(
            pl.sum("input_avail").alias("input_allocated_total"),
            pl.max("supplier_capacity_pressure").alias("supplier_pressure_max"),
            (
                (
                    pl.col("supplier_capacity_pressure").fill_null(0.0)
                    * pl.col("updated_weight").fill_null(0.0)
                ).sum()
                / (pl.sum("updated_weight") + self.epsilon)
            ).alias("supplier_pressure_mean"),
            pl.len().alias("supplier_count"),
            (
                pl.when(pl.col("supplier_type") == "historical")
                .then(pl.col("updated_weight"))
                .otherwise(0.0)
            ).sum().alias("historical_supplier_share"),
            (
                pl.when(pl.col("supplier_type") == "same_sector_foreign")
                .then(pl.col("updated_weight"))
                .otherwise(0.0)
            ).sum().alias("same_sector_supplier_share"),
            (
                pl.when(pl.col("supplier_type") == "ecosystem_feasible")
                .then(pl.col("updated_weight"))
                .otherwise(0.0)
            ).sum().alias("ecosystem_supplier_share"),
        )
        panel = (
            input_requirements.join(
                buyer_summary,
                left_on="country_sector",
                right_on="buyer_country_sector",
                how="left",
            )
            .with_columns(
                pl.col("input_allocated_total").fill_null(0.0),
                pl.col("supplier_pressure_max").fill_null(0.0),
                pl.col("supplier_pressure_mean").fill_null(0.0),
                pl.col("supplier_count").fill_null(0),
                pl.col("historical_supplier_share").fill_null(0.0),
                pl.col("same_sector_supplier_share").fill_null(0.0),
                pl.col("ecosystem_supplier_share").fill_null(0.0),
            )
            .with_columns(
                (
                    pl.col("input_allocated_total")
                    / (pl.col("input_required_total") + self.epsilon)
                ).alias("input_feasibility")
            )
            .with_columns(
                (
                    pl.col("X_desired")
                    * pl.min_horizontal(pl.lit(1.0), pl.col("input_feasibility"))
                ).alias("X_feasible")
            )
            .with_columns(
                (pl.col("X_feasible") / (pl.col("X_desired") + self.epsilon)).alias(
                    "output_feasibility_ratio"
                ),
                pl.when(pl.col("input_feasibility") < 1.0)
                .then(pl.lit("input_constrained"))
                .when(pl.col("supplier_pressure_max") > 1.0)
                .then(pl.lit("supplier_pressure_above_1"))
                .otherwise(pl.lit(""))
                .alias("feasibility_warning"),
            )
        )
        return panel.select(
            "country_sector",
            "year",
            "X_observed",
            "X_desired",
            "input_required_total",
            "input_allocated_total",
            "input_feasibility",
            "X_feasible",
            "output_feasibility_ratio",
            "supplier_pressure_mean",
            "supplier_pressure_max",
            "supplier_count",
            "historical_supplier_share",
            "same_sector_supplier_share",
            "ecosystem_supplier_share",
            "feasibility_warning",
        )

    def build_production_feasibility_report(
        self,
        feasibility_panel: pl.DataFrame,
    ) -> pl.DataFrame:
        """Build one-row production feasibility diagnostics."""
        node_count = feasibility_panel.height
        constrained_nodes = feasibility_panel.filter(pl.col("input_feasibility") < 1.0).height
        pressure_nodes = feasibility_panel.filter(pl.col("supplier_pressure_max") > 1.0).height
        total_x_observed = feasibility_panel["X_observed"].sum()
        total_x_desired = feasibility_panel["X_desired"].sum()
        total_x_feasible = feasibility_panel["X_feasible"].sum()
        return pl.DataFrame(
            {
                "year": [feasibility_panel["year"].max()],
                "node_count": [node_count],
                "total_X_observed": [total_x_observed],
                "total_X_desired": [total_x_desired],
                "total_X_feasible": [total_x_feasible],
                "aggregate_feasibility_ratio": [
                    total_x_feasible / (total_x_desired + self.epsilon)
                    if total_x_desired is not None
                    else None
                ],
                "mean_input_feasibility": [feasibility_panel["input_feasibility"].mean()],
                "median_input_feasibility": [feasibility_panel["input_feasibility"].median()],
                "p05_input_feasibility": [
                    feasibility_panel["input_feasibility"].quantile(0.05)
                ],
                "p95_supplier_pressure_max": [
                    feasibility_panel["supplier_pressure_max"].quantile(0.95)
                ],
                "nodes_with_input_feasibility_below_1": [constrained_nodes],
                "share_nodes_with_input_feasibility_below_1": [
                    constrained_nodes / node_count if node_count else 0.0
                ],
                "nodes_with_supplier_pressure_above_1": [pressure_nodes],
                "share_nodes_with_supplier_pressure_above_1": [
                    pressure_nodes / node_count if node_count else 0.0
                ],
                "notes": [
                    (
                        "One-step feasibility diagnostic using Eora T-derived total input "
                        "requirements and updated supplier weights. X_desired is X_observed. "
                        "This is not recursive Leontief propagation."
                    )
                ],
            }
        )

    def write_outputs(
        self,
        feasibility_panel: pl.DataFrame,
        report: pl.DataFrame,
    ) -> None:
        """Write production feasibility outputs."""
        self.paths.interim.mkdir(parents=True, exist_ok=True)
        self.paths.diagnostics.mkdir(parents=True, exist_ok=True)
        feasibility_panel.write_parquet(self.paths.production_feasibility_panel_path)
        report.write_csv(self.paths.production_feasibility_report_path)
