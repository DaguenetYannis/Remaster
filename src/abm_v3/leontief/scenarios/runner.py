from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from src.abm_v3.config import ABMV3Config, LeontiefPropagationConfig
from src.abm_v3.data_loader import ABMV3DataLoader
from src.abm_v3.leontief.behavioural import (
    BehaviouralLeontiefEngine,
    BehaviouralLeontiefResult,
    BehaviouralLeontiefValidator,
)
from src.abm_v3.leontief.coefficients import LeontiefCoefficientBuilder, LeontiefYearData
from src.abm_v3.leontief.scenarios.base import BehaviouralScenarioContext, BehaviouralScenarioShock
from src.abm_v3.leontief.scenarios.outputs import BehaviouralScenarioOutputWriter
from src.abm_v3.paths import ABMV3Paths

DEFAULT_BEHAVIOURAL_SCENARIO_MODE = "transpose_row_output_fd_without_inventory"
DEFAULT_BEHAVIOURAL_SCENARIO_ORIENTATION = "transpose_row_fd_without_inventory"


@dataclass
class BehaviouralLeontiefScenarioRunner:
    """Run single-year behavioural Leontief perturbation experiments."""

    paths: ABMV3Paths
    config: ABMV3Config

    def run_year(
        self,
        year: int,
        scenario_name: str,
        shock: BehaviouralScenarioShock,
    ) -> dict[str, object]:
        """Run baseline and perturbed behavioural Leontief propagation for one year."""
        active_config = self._scenario_config()
        year_data = LeontiefCoefficientBuilder(self.paths, active_config.leontief).load_year(year)
        input_panel = ABMV3DataLoader(self.paths).load_input_panel_for_orientation(
            active_config.calibration.start_year,
            active_config.calibration.end_year,
            active_config.leontief.input_panel_orientation,
            active_config,
        )
        year_data = self._apply_input_panel_validation_target(year_data, input_panel, active_config, year)
        capacity = self._load_capacity(input_panel, year, year_data)
        context = BehaviouralScenarioContext(
            year=year,
            scenario_name=scenario_name,
            mode=year_data.mode,
            input_panel_orientation=active_config.leontief.input_panel_orientation
            or DEFAULT_BEHAVIOURAL_SCENARIO_ORIENTATION,
            shock_size=float(shock.shock_size),
            selector_name=shock.selector_name,
            notes=self._scenario_notes(shock),
        )

        engine = BehaviouralLeontiefEngine(active_config.leontief)
        validator = BehaviouralLeontiefValidator()
        baseline_result = engine.propagate(year_data, capacity)
        baseline_nodes = validator.build_node_comparison(year_data, baseline_result)
        baseline_summary = validator.build_summary(year_data, baseline_result, baseline_nodes)

        scenario_year_data, scenario_capacity, selected_nodes = shock.apply(year_data, capacity, input_panel, context)
        scenario_result = engine.propagate(scenario_year_data, scenario_capacity)
        scenario_nodes = validator.build_node_comparison(scenario_year_data, scenario_result)
        scenario_summary = validator.build_summary(scenario_year_data, scenario_result, scenario_nodes)

        comparison = self.build_node_comparison(
            baseline_nodes,
            scenario_nodes,
            selected_nodes,
            scenario_name=scenario_name,
            selector_name=shock.selector_name,
        )
        aggregate = self.build_aggregate(comparison)
        summary = self.build_summary(
            context,
            selected_nodes,
            baseline_result,
            scenario_result,
            baseline_summary,
            scenario_summary,
        )
        written_paths = BehaviouralScenarioOutputWriter(self.paths).write_all(
            year,
            scenario_name,
            context,
            selected_nodes,
            comparison,
            aggregate,
            summary,
            scenario_nodes,
        )
        print(
            "[ABM v3 Behavioural Scenario] "
            f"Finished year={year}, scenario={scenario_name}, "
            f"selected_nodes={len(selected_nodes)}, "
            f"delta_realized_output_total={summary.loc[0, 'delta_realized_output_total']:.12g}"
        )
        print(f"[ABM v3 Behavioural Scenario] Wrote outputs to {self.paths.behavioural_leontief_scenario_dir}")
        return {
            "context": context,
            "year_data": year_data,
            "capacity": capacity,
            "baseline_result": baseline_result,
            "scenario_result": scenario_result,
            "baseline_node_comparison": baseline_nodes,
            "scenario_node_comparison": scenario_nodes,
            "selected_nodes": selected_nodes,
            "node_comparison": comparison,
            "aggregate": aggregate,
            "summary": summary,
            "written_paths": written_paths,
        }

    def build_node_comparison(
        self,
        baseline_nodes: pd.DataFrame,
        scenario_nodes: pd.DataFrame,
        selected_nodes: pd.DataFrame,
        scenario_name: str,
        selector_name: str,
    ) -> pd.DataFrame:
        """Compare scenario and baseline node-level realized and desired output."""
        label_columns = ["Year", "country_sector", "Country", "Country_detail", "Category", "Sector"]
        baseline = baseline_nodes.reindex(
            columns=label_columns
            + [
                "X_realized",
                "X_desired",
                "output_ratio",
                "capacity_binding_rounds",
                "capacity_missing_rounds",
            ]
        ).rename(
            columns={
                "X_realized": "X_realized_baseline",
                "X_desired": "X_desired_baseline",
                "output_ratio": "output_ratio_baseline",
                "capacity_binding_rounds": "capacity_binding_rounds_baseline",
                "capacity_missing_rounds": "capacity_missing_rounds_baseline",
            }
        )
        scenario = scenario_nodes.reindex(
            columns=[
                "Year",
                "country_sector",
                "X_realized",
                "X_desired",
                "output_ratio",
                "capacity_binding_rounds",
                "capacity_missing_rounds",
            ]
        ).rename(
            columns={
                "X_realized": "X_realized_scenario",
                "X_desired": "X_desired_scenario",
                "output_ratio": "output_ratio_scenario",
                "capacity_binding_rounds": "capacity_binding_rounds_scenario",
                "capacity_missing_rounds": "capacity_missing_rounds_scenario",
            }
        )
        comparison = baseline.merge(scenario, on=["Year", "country_sector"], how="left")
        comparison.insert(1, "scenario_name", scenario_name)
        comparison.insert(2, "selector_name", selector_name)
        selected_labels = set(selected_nodes["country_sector"].astype(str))
        comparison["is_selected_node"] = comparison["country_sector"].astype(str).isin(selected_labels)
        comparison["delta_X_realized"] = comparison["X_realized_scenario"] - comparison["X_realized_baseline"]
        comparison["pct_delta_X_realized"] = self._safe_ratio(
            comparison["delta_X_realized"],
            comparison["X_realized_baseline"],
        )
        comparison["delta_X_desired"] = comparison["X_desired_scenario"] - comparison["X_desired_baseline"]
        comparison["pct_delta_X_desired"] = self._safe_ratio(
            comparison["delta_X_desired"],
            comparison["X_desired_baseline"],
        )
        comparison["delta_capacity_binding_rounds"] = (
            comparison["capacity_binding_rounds_scenario"] - comparison["capacity_binding_rounds_baseline"]
        )
        return comparison

    def build_aggregate(self, comparison: pd.DataFrame) -> pd.DataFrame:
        """Build global, country, sector, and category scenario aggregates."""
        rows: list[pd.DataFrame] = [self._aggregate_level(comparison, "global")]
        for column in ["Country", "Sector", "Category"]:
            rows.append(self._aggregate_level(comparison, column))
        return pd.concat(rows, ignore_index=True)

    def build_summary(
        self,
        context: BehaviouralScenarioContext,
        selected_nodes: pd.DataFrame,
        baseline_result: BehaviouralLeontiefResult,
        scenario_result: BehaviouralLeontiefResult,
        baseline_summary: pd.DataFrame,
        scenario_summary: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build one-row scenario diagnostics."""
        baseline_realized = float(baseline_summary.loc[0, "realized_output_total"])
        scenario_realized = float(scenario_summary.loc[0, "realized_output_total"])
        baseline_desired = float(baseline_summary.loc[0, "desired_output_total"])
        scenario_desired = float(scenario_summary.loc[0, "desired_output_total"])
        row = {
            "Year": context.year,
            "scenario_name": context.scenario_name,
            "shock_type": str(selected_nodes["shock_type"].iloc[0]) if "shock_type" in selected_nodes.columns and len(selected_nodes) else "",
            "selector_name": context.selector_name,
            "shock_size": context.shock_size,
            "shock_mode": str(selected_nodes["shock_mode"].iloc[0]) if "shock_mode" in selected_nodes.columns and len(selected_nodes) else "",
            "mode": context.mode,
            "input_panel_orientation": context.input_panel_orientation,
            "baseline_converged": baseline_result.converged,
            "scenario_converged": scenario_result.converged,
            "baseline_rounds_used": baseline_result.rounds_used,
            "scenario_rounds_used": scenario_result.rounds_used,
            "baseline_final_residual_share": baseline_result.final_residual_share,
            "scenario_final_residual_share": scenario_result.final_residual_share,
            "baseline_final_residual_total": baseline_result.final_residual_total,
            "scenario_final_residual_total": scenario_result.final_residual_total,
            "baseline_accumulated_realized_output_total": baseline_result.accumulated_realized_output_total,
            "scenario_accumulated_realized_output_total": scenario_result.accumulated_realized_output_total,
            "baseline_realized_output_total": baseline_realized,
            "scenario_realized_output_total": scenario_realized,
            "delta_realized_output_total": scenario_realized - baseline_realized,
            "pct_delta_realized_output_total": self._safe_scalar_ratio(scenario_realized - baseline_realized, baseline_realized),
            "baseline_desired_output_total": baseline_desired,
            "scenario_desired_output_total": scenario_desired,
            "delta_desired_output_total": scenario_desired - baseline_desired,
            "selected_node_count": len(selected_nodes),
            "notes": context.notes,
        }
        for column in ["Y_baseline", "Y_scenario", "K_baseline", "K_scenario"]:
            if column in selected_nodes.columns:
                summary_name = f"selected_{column}_total"
                row[summary_name] = float(pd.to_numeric(selected_nodes[column], errors="coerce").sum(skipna=True))
        return pd.DataFrame([row])

    def _scenario_config(self) -> ABMV3Config:
        selected_mode = self.config.leontief.leontief_mode
        if selected_mode in {None, "raw"}:
            selected_mode = DEFAULT_BEHAVIOURAL_SCENARIO_MODE
        leontief_config = replace(
            self.config.leontief,
            leontief_mode=selected_mode,
            input_panel_orientation=self.config.leontief.input_panel_orientation
            or DEFAULT_BEHAVIOURAL_SCENARIO_ORIENTATION,
        )
        return replace(self.config, leontief=leontief_config)

    def _apply_input_panel_validation_target(
        self,
        year_data: LeontiefYearData,
        input_panel: pd.DataFrame,
        config: ABMV3Config,
        year: int,
    ) -> LeontiefYearData:
        orientation = config.leontief.input_panel_orientation
        required_columns = {"Year", "country_sector", "X_observed"}
        missing = required_columns.difference(input_panel.columns)
        if missing:
            raise ValueError(f"Input panel orientation '{orientation}' is missing validation columns: {sorted(missing)}")
        labels = year_data.labels["country_sector"].astype(str).tolist()
        year_panel = input_panel.loc[
            pd.to_numeric(input_panel["Year"], errors="coerce").eq(year),
            ["country_sector", "X_observed"],
        ].copy()
        if year_panel.empty:
            raise ValueError(f"Input panel orientation '{orientation}' has no validation rows for year {year}")
        x_observed = pd.Series(
            pd.to_numeric(year_panel["X_observed"], errors="coerce").to_numpy(dtype=float),
            index=year_panel["country_sector"].astype(str),
            name="X_observed",
        ).reindex(labels)
        if x_observed.isna().all():
            raise ValueError(f"Input panel orientation '{orientation}' has no labels matching Leontief year {year}")
        invalid_output_columns = self._build_invalid_output_columns(year_data, x_observed)
        return replace(
            year_data,
            X_observed=x_observed,
            input_panel_orientation=orientation,
            validation_reference=f"input_panel:{orientation}:X_observed",
            invalid_output_columns=invalid_output_columns,
            capacity_source=f"input_panel:{orientation}:K",
        )

    def _load_capacity(self, input_panel: pd.DataFrame, year: int, year_data: LeontiefYearData) -> pd.Series:
        required_columns = {"Year", "country_sector", "K"}
        missing = required_columns.difference(input_panel.columns)
        if missing:
            raise ValueError(f"Input panel is missing capacity columns: {sorted(missing)}")
        labels = year_data.labels["country_sector"].astype(str).tolist()
        year_panel = input_panel.loc[
            pd.to_numeric(input_panel["Year"], errors="coerce").eq(year),
            ["country_sector", "K"],
        ]
        capacity = pd.Series(
            pd.to_numeric(year_panel["K"], errors="coerce").to_numpy(dtype=float),
            index=year_panel["country_sector"].astype(str),
            name="K",
        ).reindex(labels)
        if capacity.isna().all():
            raise ValueError(f"Input panel has no capacity labels matching Leontief year {year}")
        return capacity

    def _build_invalid_output_columns(self, year_data: LeontiefYearData, x_observed: pd.Series) -> pd.DataFrame:
        labels_frame = year_data.labels.copy()
        x_values = x_observed.to_numpy(dtype=float)
        invalid_mask = (~np.isfinite(x_values)) | (x_values <= 0.0)
        invalid = labels_frame.loc[invalid_mask].copy()
        invalid.insert(0, "Year", year_data.year)
        invalid["X_observed"] = x_values[invalid_mask]
        invalid["reason"] = ["missing_or_non_finite_output" if pd.isna(value) else "non_positive_output" for value in x_values[invalid_mask]]
        return invalid

    def _aggregate_level(self, comparison: pd.DataFrame, level: str) -> pd.DataFrame:
        if level == "global":
            grouped = [(("global",), comparison)]
            key_name = "global"
        else:
            grouped = comparison.groupby(level, dropna=False)
            key_name = None
        rows = []
        for key, group in grouped:
            aggregation_key = key_name if key_name is not None else str(key)
            baseline_realized = float(group["X_realized_baseline"].sum(skipna=True))
            scenario_realized = float(group["X_realized_scenario"].sum(skipna=True))
            baseline_desired = float(group["X_desired_baseline"].sum(skipna=True))
            scenario_desired = float(group["X_desired_scenario"].sum(skipna=True))
            rows.append(
                {
                    "Year": int(group["Year"].iloc[0]),
                    "scenario_name": str(group["scenario_name"].iloc[0]),
                    "aggregation_level": level,
                    "aggregation_key": aggregation_key,
                    "X_realized_baseline_sum": baseline_realized,
                    "X_realized_scenario_sum": scenario_realized,
                    "delta_X_realized_sum": scenario_realized - baseline_realized,
                    "pct_delta_X_realized_sum": self._safe_scalar_ratio(scenario_realized - baseline_realized, baseline_realized),
                    "X_desired_baseline_sum": baseline_desired,
                    "X_desired_scenario_sum": scenario_desired,
                    "delta_X_desired_sum": scenario_desired - baseline_desired,
                    "selected_node_count": int(group["is_selected_node"].sum()),
                    "total_node_count": len(group),
                }
            )
        return pd.DataFrame(rows)

    def _scenario_notes(self, shock: BehaviouralScenarioShock) -> str:
        if shock.__class__.__name__ == "CapacityShock":
            return (
                "ABM v3 capacity shock is an exogenous stress test, not adaptive capacity, "
                "investment, depreciation, or policy-driven capacity change."
            )
        return (
            "ABM v3 scenario is a single-year behavioural Leontief production-network "
            "perturbation experiment, not a forecast or endogenous EI transition."
        )

    def _safe_ratio(self, numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        denominator_numeric = pd.to_numeric(denominator, errors="coerce")
        ratio = pd.to_numeric(numerator, errors="coerce") / denominator_numeric.where(denominator_numeric != 0.0)
        return ratio.replace([np.inf, -np.inf], np.nan)

    def _safe_scalar_ratio(self, numerator: float, denominator: float) -> float:
        if not np.isfinite(denominator) or denominator == 0.0:
            return np.nan
        return float(numerator / denominator)
