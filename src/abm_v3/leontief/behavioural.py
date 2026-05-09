from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.abm_v3.config import LeontiefPropagationConfig
from src.abm_v3.leontief.coefficients import LeontiefYearData


@dataclass
class BehaviouralLeontiefResult:
    """Result from soft capacity-constrained Leontief propagation."""

    year: int
    mode: str
    X_realized: pd.Series
    X_desired: pd.Series
    round_summaries: pd.DataFrame
    node_rounds: pd.DataFrame
    rounds_used: int
    tolerance: float
    max_rounds: int
    converged: bool
    initial_demand_total: float
    accumulated_realized_output_total: float
    final_residual_total: float
    final_residual_share: float


class BehaviouralLeontiefEngine:
    """Network propagation with soft capacity-constrained node response."""

    def __init__(self, config: LeontiefPropagationConfig) -> None:
        self.config = config

    def propagate(
        self,
        year_data: LeontiefYearData,
        capacity: pd.Series,
    ) -> BehaviouralLeontiefResult:
        """Propagate supplier demand from realized, capacity-constrained output."""
        tolerance = float(self.config.behavioural_tolerance)
        max_rounds = int(self.config.behavioural_max_rounds)
        print(
            "[ABM v3 Behavioural Leontief] Starting propagation: "
            f"eta_capacity={self.config.behavioural_capacity_eta}, "
            f"tolerance={tolerance}, max_rounds={max_rounds}"
        )
        labels = year_data.labels["country_sector"].tolist()
        capacity_values = capacity.reindex(labels).to_numpy(dtype=float)
        flow = year_data.Y_final_demand.reindex(labels).to_numpy(dtype=float)
        flow = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
        x_realized = np.zeros_like(flow, dtype=float)
        x_desired = np.zeros_like(flow, dtype=float)
        initial_total = float(np.sum(np.abs(flow)))
        denominator = max(initial_total, np.finfo(float).eps)
        round_rows: list[dict[str, object]] = []
        node_round_rows: list[pd.DataFrame] = []
        a_column_sum = np.asarray(year_data.A.sum(axis=0)).ravel()
        converged = False
        final_residual_total = initial_total
        final_residual_share = final_residual_total / denominator
        rounds_used = 0

        for round_number in range(max_rounds + 1):
            desired_output = np.where(flow > 0.0, flow, 0.0)
            response = self.apply_soft_capacity(desired_output, capacity_values)
            realized_output = response["realized_output"]
            capacity_stress = response["capacity_stress"]
            capacity_ratio = response["capacity_ratio"]
            capacity_binding = response["capacity_binding"]
            capacity_missing = response["capacity_feasibility_missing"]
            capacity_penalty = desired_output - realized_output
            supplier_demand_generated = a_column_sum * realized_output
            next_flow = year_data.A @ realized_output
            next_flow = np.nan_to_num(next_flow, nan=0.0, posinf=0.0, neginf=0.0)
            supplier_demand_total = float(np.sum(next_flow))
            absolute_supplier_demand_total = float(np.sum(np.abs(next_flow)))
            residual_share = absolute_supplier_demand_total / denominator
            converged_this_round = residual_share < tolerance

            x_realized += realized_output
            x_desired += desired_output
            round_rows.append(
                {
                    "Year": year_data.year,
                    "mode": year_data.mode,
                    "input_panel_orientation": year_data.input_panel_orientation,
                    "capacity_source": year_data.capacity_source,
                    "round": round_number,
                    "received_demand_total": float(np.sum(flow)),
                    "desired_output_total": float(np.sum(desired_output)),
                    "realized_output_total": float(np.sum(realized_output)),
                    "unmet_capacity_total": float(np.sum(np.maximum(capacity_penalty, 0.0))),
                    "capacity_penalty_total": float(np.sum(capacity_penalty)),
                    "supplier_demand_total": supplier_demand_total,
                    "absolute_supplier_demand_total": absolute_supplier_demand_total,
                    "residual_share": residual_share,
                    "mean_capacity_stress": float(np.nanmean(capacity_stress)),
                    "median_capacity_stress": float(np.nanmedian(capacity_stress)),
                    "share_capacity_binding": float(np.mean(capacity_binding)),
                    "share_capacity_missing": float(np.mean(capacity_missing)),
                    "converged": converged_this_round,
                }
            )
            if self.config.write_behavioural_node_rounds:
                node_round_rows.append(
                    self._build_node_rounds(
                        year_data,
                        round_number,
                        flow,
                        desired_output,
                        capacity_values,
                        capacity_ratio,
                        capacity_stress,
                        realized_output,
                        capacity_penalty,
                        capacity_binding,
                        capacity_missing,
                        supplier_demand_generated,
                    )
                )
            print(
                "[ABM v3 Behavioural Leontief] "
                f"Round {round_number}: realized_total={float(np.sum(realized_output)):.12g}, "
                f"supplier_demand_total={supplier_demand_total:.12g}, "
                f"residual_share={residual_share:.12g}, "
                f"share_capacity_binding={float(np.mean(capacity_binding)):.12g}"
            )
            rounds_used = round_number
            final_residual_total = absolute_supplier_demand_total
            final_residual_share = residual_share
            if converged_this_round:
                converged = True
                break
            if round_number == max_rounds:
                break
            flow = next_flow

        x_realized_series = pd.Series(x_realized, index=labels, name="X_realized")
        x_desired_series = pd.Series(x_desired, index=labels, name="X_desired")
        node_rounds = pd.concat(node_round_rows, ignore_index=True) if node_round_rows else pd.DataFrame()
        return BehaviouralLeontiefResult(
            year=year_data.year,
            mode=year_data.mode,
            X_realized=x_realized_series,
            X_desired=x_desired_series,
            round_summaries=pd.DataFrame(round_rows),
            node_rounds=node_rounds,
            rounds_used=rounds_used,
            tolerance=tolerance,
            max_rounds=max_rounds,
            converged=converged,
            initial_demand_total=initial_total,
            accumulated_realized_output_total=float(np.sum(x_realized)),
            final_residual_total=final_residual_total,
            final_residual_share=final_residual_share,
        )

    def apply_soft_capacity(self, desired_output: np.ndarray, capacity_values: np.ndarray) -> dict[str, np.ndarray]:
        """Apply soft capacity response without collapsing missing-capacity nodes."""
        desired = np.nan_to_num(desired_output.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        desired = np.where(desired > 0.0, desired, 0.0)
        capacity = capacity_values.astype(float)
        valid_demand = desired > 0.0
        capacity_missing = (~np.isfinite(capacity)) | (capacity <= 0.0)
        capacity_ratio = np.full_like(desired, np.nan, dtype=float)
        valid_capacity_response = valid_demand & (~capacity_missing)
        capacity_ratio[valid_capacity_response] = capacity[valid_capacity_response] / desired[valid_capacity_response]
        capacity_stress = np.ones_like(desired, dtype=float)
        eta = float(self.config.behavioural_capacity_eta)
        stressed = valid_capacity_response & (capacity_ratio < 1.0)
        capacity_stress[stressed] = np.minimum(1.0, np.power(np.maximum(capacity_ratio[stressed], 0.0), eta))
        realized_output = desired * capacity_stress
        capacity_binding = valid_demand & (~capacity_missing) & (capacity_stress < 1.0)
        return {
            "capacity_ratio": capacity_ratio,
            "capacity_stress": capacity_stress,
            "realized_output": realized_output,
            "capacity_binding": capacity_binding,
            "capacity_feasibility_missing": capacity_missing & valid_demand,
        }

    def _build_node_rounds(
        self,
        year_data: LeontiefYearData,
        round_number: int,
        received_demand: np.ndarray,
        desired_output: np.ndarray,
        capacity_values: np.ndarray,
        capacity_ratio: np.ndarray,
        capacity_stress: np.ndarray,
        realized_output: np.ndarray,
        capacity_penalty: np.ndarray,
        capacity_binding: np.ndarray,
        capacity_missing: np.ndarray,
        supplier_demand_generated: np.ndarray,
    ) -> pd.DataFrame:
        rows = year_data.labels.copy()
        rows.insert(0, "Year", year_data.year)
        rows.insert(1, "mode", year_data.mode)
        rows.insert(2, "input_panel_orientation", year_data.input_panel_orientation)
        rows.insert(3, "capacity_source", year_data.capacity_source)
        rows.insert(4, "round", round_number)
        rows["received_demand"] = received_demand
        rows["desired_output"] = desired_output
        rows["K"] = capacity_values
        rows["capacity_ratio"] = capacity_ratio
        rows["capacity_stress"] = capacity_stress
        rows["realized_output"] = realized_output
        rows["capacity_penalty"] = capacity_penalty
        rows["capacity_binding"] = capacity_binding
        rows["capacity_feasibility_missing"] = capacity_missing
        rows["supplier_demand_generated"] = supplier_demand_generated
        return rows


class BehaviouralLeontiefValidator:
    """Validate behavioural Leontief output against the selected observed X."""

    def build_node_comparison(
        self,
        year_data: LeontiefYearData,
        result: BehaviouralLeontiefResult,
    ) -> pd.DataFrame:
        comparison = year_data.labels.copy()
        comparison.insert(0, "Year", year_data.year)
        comparison["mode"] = year_data.mode
        comparison["input_panel_orientation"] = year_data.input_panel_orientation
        comparison["validation_reference"] = year_data.validation_reference
        comparison["capacity_source"] = year_data.capacity_source
        comparison["X_observed"] = year_data.X_observed.to_numpy(dtype=float)
        comparison["X_realized"] = result.X_realized.to_numpy(dtype=float)
        comparison["X_desired"] = result.X_desired.to_numpy(dtype=float)
        comparison["output_gap"] = comparison["X_realized"] - comparison["X_observed"]
        valid_observed = np.isfinite(comparison["X_observed"]) & (comparison["X_observed"] > 0.0)
        comparison["output_ratio"] = np.nan
        comparison.loc[valid_observed, "output_ratio"] = (
            comparison.loc[valid_observed, "X_realized"] / comparison.loc[valid_observed, "X_observed"]
        )
        comparison["absolute_error"] = comparison["output_gap"].abs()
        comparison["absolute_percentage_error"] = np.nan
        comparison.loc[valid_observed, "absolute_percentage_error"] = (
            comparison.loc[valid_observed, "absolute_error"] / comparison.loc[valid_observed, "X_observed"]
        )
        node_summary = self._node_round_summary(result.node_rounds)
        if not node_summary.empty:
            comparison = comparison.merge(node_summary, on=["country_sector"], how="left")
        else:
            comparison["total_desired_output"] = np.nan
            comparison["total_capacity_penalty"] = np.nan
            comparison["mean_capacity_stress"] = np.nan
            comparison["capacity_binding_rounds"] = 0
            comparison["capacity_missing_rounds"] = 0
        return comparison

    def build_summary(
        self,
        year_data: LeontiefYearData,
        result: BehaviouralLeontiefResult,
        node_comparison: pd.DataFrame,
    ) -> pd.DataFrame:
        observed_total = float(np.nansum(node_comparison["X_observed"].to_numpy(dtype=float)))
        realized_total = float(np.nansum(node_comparison["X_realized"].to_numpy(dtype=float)))
        desired_total = float(np.nansum(node_comparison["X_desired"].to_numpy(dtype=float)))
        ape = node_comparison["absolute_percentage_error"].to_numpy(dtype=float)
        rounds = result.round_summaries
        return pd.DataFrame(
            [
                {
                    "Year": year_data.year,
                    "mode": year_data.mode,
                    "input_panel_orientation": year_data.input_panel_orientation,
                    "coefficient_mode": year_data.mode,
                    "validation_reference": year_data.validation_reference,
                    "capacity_source": year_data.capacity_source,
                    "converged": result.converged,
                    "rounds_used": result.rounds_used,
                    "tolerance": result.tolerance,
                    "max_rounds": result.max_rounds,
                    "observed_output_total": observed_total,
                    "realized_output_total": realized_total,
                    "desired_output_total": desired_total,
                    "relative_error_total": self._safe_ratio(float(np.nansum(node_comparison["absolute_error"])), observed_total),
                    "correlation_realized_vs_observed": self._safe_correlation(
                        node_comparison["X_realized"].to_numpy(dtype=float),
                        node_comparison["X_observed"].to_numpy(dtype=float),
                    ),
                    "mean_absolute_percentage_error": float(np.nanmean(ape)) if np.isfinite(ape).any() else np.nan,
                    "median_absolute_percentage_error": float(np.nanmedian(ape)) if np.isfinite(ape).any() else np.nan,
                    "final_residual_share": result.final_residual_share,
                    "total_capacity_penalty": float(np.nansum(node_comparison["total_capacity_penalty"])),
                    "mean_capacity_stress_over_rounds": float(rounds["mean_capacity_stress"].mean()) if len(rounds) else np.nan,
                    "mean_share_capacity_binding": float(rounds["share_capacity_binding"].mean()) if len(rounds) else np.nan,
                    "mean_share_capacity_missing": float(rounds["share_capacity_missing"].mean()) if len(rounds) else np.nan,
                }
            ]
        )

    def _node_round_summary(self, node_rounds: pd.DataFrame) -> pd.DataFrame:
        if node_rounds.empty:
            return pd.DataFrame()
        grouped = node_rounds.groupby("country_sector", as_index=False).agg(
            total_desired_output=("desired_output", "sum"),
            total_capacity_penalty=("capacity_penalty", "sum"),
            mean_capacity_stress=("capacity_stress", "mean"),
            capacity_binding_rounds=("capacity_binding", "sum"),
            capacity_missing_rounds=("capacity_feasibility_missing", "sum"),
        )
        return grouped

    def _safe_ratio(self, numerator: float, denominator: float) -> float:
        if not np.isfinite(denominator) or denominator <= 0.0:
            return np.nan
        return float(numerator / denominator)

    def _safe_correlation(self, left_values: np.ndarray, right_values: np.ndarray) -> float:
        valid = np.isfinite(left_values) & np.isfinite(right_values)
        if int(valid.sum()) < 2:
            return np.nan
        left = left_values[valid]
        right = right_values[valid]
        if np.isclose(float(np.std(left)), 0.0) or np.isclose(float(np.std(right)), 0.0):
            return np.nan
        return float(np.corrcoef(left, right)[0, 1])


class BehaviouralLeontiefOutputWriter:
    """Write behavioural Leontief outputs and diagnostics."""

    def __init__(self, paths: object) -> None:
        self.paths = paths

    def write_all(
        self,
        year_data: LeontiefYearData,
        result: BehaviouralLeontiefResult,
        node_comparison: pd.DataFrame,
        summary: pd.DataFrame,
    ) -> dict[str, object]:
        self.paths.leontief_behavioural_outputs_dir.mkdir(parents=True, exist_ok=True)
        self.paths.leontief_behavioural_summary_diagnostics_dir.mkdir(parents=True, exist_ok=True)
        self.paths.leontief_behavioural_rounds_diagnostics_dir.mkdir(parents=True, exist_ok=True)
        self.paths.leontief_behavioural_node_comparison_diagnostics_dir.mkdir(parents=True, exist_ok=True)
        self.paths.leontief_behavioural_node_rounds_diagnostics_dir.mkdir(parents=True, exist_ok=True)
        orientation = year_data.input_panel_orientation
        output_path = self.paths.behavioural_leontief_output_path(year_data.year, year_data.mode, orientation)
        summary_path = self.paths.behavioural_leontief_summary_path(year_data.year, year_data.mode, orientation)
        node_comparison_path = self.paths.behavioural_leontief_node_comparison_path(year_data.year, year_data.mode, orientation)
        rounds_path = self.paths.behavioural_leontief_rounds_path(year_data.year, year_data.mode, orientation)
        node_rounds_path = self.paths.behavioural_leontief_node_rounds_path(year_data.year, year_data.mode, orientation)
        node_comparison.to_parquet(output_path, index=False)
        summary.to_csv(summary_path, index=False)
        node_comparison.to_csv(node_comparison_path, index=False)
        result.round_summaries.to_csv(rounds_path, index=False)
        result.node_rounds.to_csv(node_rounds_path, index=False)
        return {
            "output": output_path,
            "summary": summary_path,
            "node_comparison": node_comparison_path,
            "rounds": rounds_path,
            "node_rounds": node_rounds_path,
        }
