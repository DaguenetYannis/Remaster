from __future__ import annotations

import numpy as np
import pandas as pd

from src.abm_v3.leontief.coefficients import LeontiefYearData
from src.abm_v3.leontief.propagation import LeontiefPropagationResult


class LeontiefPropagationValidator:
    """Build node-level and aggregate diagnostics for Leontief propagation."""

    def build_node_comparison(
        self,
        year_data: LeontiefYearData,
        result: LeontiefPropagationResult,
    ) -> pd.DataFrame:
        """Compare iterative output with observed Eora output by node."""
        comparison = year_data.labels.copy()
        comparison.insert(0, "Year", year_data.year)
        comparison["mode"] = year_data.mode
        comparison["input_panel_orientation"] = year_data.input_panel_orientation
        comparison["validation_reference"] = year_data.validation_reference
        comparison["Y_final_demand"] = year_data.Y_final_demand.to_numpy(dtype=float)
        if year_data.Y_raw_final_demand is not None:
            comparison["Y_raw_final_demand"] = year_data.Y_raw_final_demand.to_numpy(dtype=float)
        if year_data.X_used_for_coefficients is not None:
            comparison["X_used_for_coefficients"] = year_data.X_used_for_coefficients.to_numpy(dtype=float)
        comparison["X_observed"] = year_data.X_observed.to_numpy(dtype=float)
        comparison["X_iterative"] = result.X_iterative.to_numpy(dtype=float)
        comparison["output_gap"] = comparison["X_iterative"] - comparison["X_observed"]
        valid_observed = np.isfinite(comparison["X_observed"]) & (comparison["X_observed"] > 0.0)
        comparison["output_ratio"] = np.nan
        comparison.loc[valid_observed, "output_ratio"] = (
            comparison.loc[valid_observed, "X_iterative"]
            / comparison.loc[valid_observed, "X_observed"]
        )
        comparison["absolute_error"] = comparison["output_gap"].abs()
        comparison["absolute_percentage_error"] = np.nan
        comparison.loc[valid_observed, "absolute_percentage_error"] = (
            comparison.loc[valid_observed, "absolute_error"]
            / comparison.loc[valid_observed, "X_observed"]
        )
        return comparison

    def build_summary(
        self,
        year_data: LeontiefYearData,
        result: LeontiefPropagationResult,
        node_comparison: pd.DataFrame,
    ) -> pd.DataFrame:
        """Summarize propagation convergence and reconstruction error."""
        observed_output_total = float(np.nansum(node_comparison["X_observed"].to_numpy(dtype=float)))
        absolute_error_total = float(np.nansum(node_comparison["absolute_error"].to_numpy(dtype=float)))
        relative_error_total = self._safe_ratio(absolute_error_total, observed_output_total)
        correlation = self._safe_correlation(
            node_comparison["X_iterative"].to_numpy(dtype=float),
            node_comparison["X_observed"].to_numpy(dtype=float),
        )
        ape = node_comparison["absolute_percentage_error"].to_numpy(dtype=float)
        invalid_count = 0 if year_data.invalid_output_columns is None else len(year_data.invalid_output_columns)
        summary = pd.DataFrame(
            [
                {
                    "Year": year_data.year,
                    "mode": year_data.mode,
                    "input_panel_orientation": year_data.input_panel_orientation,
                    "coefficient_mode": year_data.mode,
                    "validation_reference": year_data.validation_reference,
                    "rounds_used": result.rounds_used,
                    "tolerance": result.tolerance,
                    "max_rounds": result.max_rounds,
                    "converged": result.converged,
                    "initial_final_demand_total": result.initial_final_demand_total,
                    "accumulated_output_total": result.accumulated_output_total,
                    "observed_output_total": observed_output_total,
                    "final_residual_total": result.final_residual_total,
                    "final_residual_share": result.final_residual_share,
                    "absolute_error_total": absolute_error_total,
                    "relative_error_total": relative_error_total,
                    "correlation_iterative_vs_observed": correlation,
                    "mean_absolute_error": float(np.nanmean(node_comparison["absolute_error"])),
                    "median_absolute_error": float(np.nanmedian(node_comparison["absolute_error"])),
                    "mean_absolute_percentage_error": float(np.nanmean(ape)) if np.isfinite(ape).any() else np.nan,
                    "median_absolute_percentage_error": float(np.nanmedian(ape)) if np.isfinite(ape).any() else np.nan,
                    "zero_observed_output_count": int((node_comparison["X_observed"] == 0.0).sum()),
                    "missing_observed_output_count": int(node_comparison["X_observed"].isna().sum()),
                    "invalid_output_columns_count": invalid_count,
                }
            ]
        )
        return summary

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
