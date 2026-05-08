from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from src.abm_v3.config import LeontiefPropagationConfig
from src.abm_v3.leontief.coefficients import LeontiefYearData


@dataclass
class LeontiefViabilityDiagnostics:
    """Coefficient viability diagnostics for one Leontief year."""

    year: int
    summary: pd.DataFrame
    columns: pd.DataFrame
    negative_flows: pd.DataFrame
    spectral: pd.DataFrame
    top_unstable_nodes: pd.DataFrame | None = None


class LeontiefViabilityAnalyzer:
    """Diagnose whether empirical coefficients are viable for power-series propagation."""

    def __init__(self, config: LeontiefPropagationConfig) -> None:
        self.config = config

    def analyze(self, year_data: LeontiefYearData) -> LeontiefViabilityDiagnostics:
        """Build coefficient, negative-flow, and spectral diagnostics."""
        columns = self._build_column_diagnostics(year_data)
        spectral = self._build_spectral_diagnostics(year_data)
        negative_flows = self._negative_flows_or_empty(year_data)
        summary = self._build_summary(year_data, columns, spectral)
        top_unstable_nodes = self._build_top_unstable_nodes(columns)
        return LeontiefViabilityDiagnostics(
            year=year_data.year,
            summary=summary,
            columns=columns,
            negative_flows=negative_flows,
            spectral=spectral,
            top_unstable_nodes=top_unstable_nodes,
        )

    def _build_column_diagnostics(self, year_data: LeontiefYearData) -> pd.DataFrame:
        a_csc = year_data.A.tocsc()
        abs_a_csc = abs(a_csc)
        node_count = a_csc.shape[1]
        column_sum = np.asarray(a_csc.sum(axis=0)).ravel()
        abs_column_sum = np.asarray(abs_a_csc.sum(axis=0)).ravel()
        nonzero_count = np.diff(a_csc.indptr).astype(int)
        negative_count = np.zeros(node_count, dtype=int)
        positive_count = np.zeros(node_count, dtype=int)
        large_count = np.zeros(node_count, dtype=int)
        max_coefficient = np.zeros(node_count, dtype=float)
        min_coefficient = np.zeros(node_count, dtype=float)
        max_abs_coefficient = np.zeros(node_count, dtype=float)

        for column_index in range(node_count):
            start = a_csc.indptr[column_index]
            end = a_csc.indptr[column_index + 1]
            values = a_csc.data[start:end]
            if len(values) == 0:
                continue
            negative_count[column_index] = int((values < 0.0).sum())
            positive_count[column_index] = int((values > 0.0).sum())
            large_count[column_index] = int((np.abs(values) > self.config.large_coefficient_threshold).sum())
            max_coefficient[column_index] = float(np.max(values))
            min_coefficient[column_index] = float(np.min(values))
            max_abs_coefficient[column_index] = float(np.max(np.abs(values)))

        columns = year_data.labels.copy()
        columns.insert(0, "Year", year_data.year)
        columns["mode"] = year_data.mode
        columns["X_observed"] = year_data.X_observed.to_numpy(dtype=float)
        x_used = (
            year_data.X_used_for_coefficients
            if year_data.X_used_for_coefficients is not None
            else year_data.X_observed
        )
        columns["X_used_for_coefficients"] = x_used.to_numpy(dtype=float)
        columns["Y_final_demand"] = year_data.Y_final_demand.to_numpy(dtype=float)
        if year_data.Y_raw_final_demand is not None:
            columns["Y_raw_final_demand"] = year_data.Y_raw_final_demand.to_numpy(dtype=float)
        columns["column_sum_A"] = column_sum
        columns["abs_column_sum_A"] = abs_column_sum
        columns["max_coefficient_A"] = max_coefficient
        columns["min_coefficient_A"] = min_coefficient
        columns["max_abs_coefficient_A"] = max_abs_coefficient
        columns["nonzero_coefficient_count"] = nonzero_count
        columns["negative_coefficient_count"] = negative_count
        columns["positive_coefficient_count"] = positive_count
        columns["large_coefficient_count"] = large_count
        x_values = columns["X_used_for_coefficients"].to_numpy(dtype=float)
        y_values = columns["Y_final_demand"].to_numpy(dtype=float)
        columns["invalid_output_column"] = (~np.isfinite(x_values)) | (x_values <= 0.0)
        columns["near_zero_positive_output"] = (x_values > 0.0) & (x_values <= self.config.near_zero_output_threshold)
        columns["negative_final_demand"] = np.isfinite(y_values) & (y_values < 0.0)
        columns["high_abs_column_sum"] = columns["abs_column_sum_A"] > self.config.high_abs_column_sum_threshold
        columns["has_large_coefficients"] = columns["large_coefficient_count"] > 0
        columns["has_negative_coefficients"] = columns["negative_coefficient_count"] > 0
        flag_columns = [
            "invalid_output_column",
            "near_zero_positive_output",
            "negative_final_demand",
            "high_abs_column_sum",
            "has_large_coefficients",
            "has_negative_coefficients",
        ]
        columns["suspicion_score"] = columns[flag_columns].astype(int).sum(axis=1)
        columns["suspicious_column"] = columns["suspicion_score"] > 0
        columns["rank_abs_column_sum"] = columns["abs_column_sum_A"].rank(method="min", ascending=False).astype(int)
        columns["rank_max_abs_coefficient"] = columns["max_abs_coefficient_A"].rank(method="min", ascending=False).astype(int)
        columns["rank_suspicion_score"] = columns["suspicion_score"].rank(method="min", ascending=False).astype(int)
        return columns

    def _build_spectral_diagnostics(self, year_data: LeontiefYearData) -> pd.DataFrame:
        rows = []
        for matrix_name, matrix in [("A", year_data.A), ("abs_A", abs(year_data.A))]:
            radius, converged, iterations = self.estimate_spectral_radius(
                matrix,
                self.config.spectral_radius_max_iter,
                self.config.spectral_radius_tolerance,
            )
            rows.append(
                {
                    "Year": year_data.year,
                    "mode": year_data.mode,
                    "matrix": matrix_name,
                    "approximate_spectral_radius": radius,
                    "converged": converged,
                    "iterations": iterations,
                    "tolerance": self.config.spectral_radius_tolerance,
                    "max_iter": self.config.spectral_radius_max_iter,
                    "above_one": bool(np.isfinite(radius) and radius > 1.0),
                }
            )
        return pd.DataFrame(rows)

    def estimate_spectral_radius(
        self,
        matrix: sparse.spmatrix,
        max_iter: int,
        tolerance: float,
    ) -> tuple[float, bool, int]:
        """Estimate dominant amplification using sparse power iteration."""
        if matrix.shape[0] == 0:
            return np.nan, False, 0
        vector = np.ones(matrix.shape[1], dtype=float)
        vector_norm = np.linalg.norm(vector)
        if vector_norm == 0.0:
            return np.nan, False, 0
        vector = vector / vector_norm
        previous_estimate = np.nan
        for iteration in range(1, max_iter + 1):
            next_vector = matrix @ vector
            next_vector = np.nan_to_num(next_vector, nan=0.0, posinf=0.0, neginf=0.0)
            estimate = float(np.linalg.norm(next_vector))
            if not np.isfinite(estimate) or estimate == 0.0:
                return estimate, False, iteration
            next_vector = next_vector / estimate
            if np.isfinite(previous_estimate) and abs(estimate - previous_estimate) < tolerance:
                return estimate, True, iteration
            vector = next_vector
            previous_estimate = estimate
        return previous_estimate, False, max_iter

    def _build_summary(
        self,
        year_data: LeontiefYearData,
        columns: pd.DataFrame,
        spectral: pd.DataFrame,
    ) -> pd.DataFrame:
        spectral_a = spectral.loc[spectral["matrix"] == "A"].iloc[0]
        spectral_abs_a = spectral.loc[spectral["matrix"] == "abs_A"].iloc[0]
        summary = pd.DataFrame(
            [
                {
                    "Year": year_data.year,
                    "mode": year_data.mode,
                    "node_count": len(columns),
                    "invalid_output_column_count": int(columns["invalid_output_column"].sum()),
                    "near_zero_positive_output_count": int(columns["near_zero_positive_output"].sum()),
                    "negative_final_demand_count": int(columns["negative_final_demand"].sum()),
                    "high_abs_column_sum_count": int(columns["high_abs_column_sum"].sum()),
                    "large_coefficient_column_count": int(columns["has_large_coefficients"].sum()),
                    "negative_coefficient_column_count": int(columns["has_negative_coefficients"].sum()),
                    "suspicious_column_count": int(columns["suspicious_column"].sum()),
                    "max_abs_column_sum_A": float(columns["abs_column_sum_A"].max()),
                    "median_abs_column_sum_A": float(columns["abs_column_sum_A"].median()),
                    "p95_abs_column_sum_A": float(columns["abs_column_sum_A"].quantile(0.95)),
                    "p99_abs_column_sum_A": float(columns["abs_column_sum_A"].quantile(0.99)),
                    "max_abs_coefficient_A": float(columns["max_abs_coefficient_A"].max()),
                    "median_max_abs_coefficient_A": float(columns["max_abs_coefficient_A"].median()),
                    "p95_max_abs_coefficient_A": float(columns["max_abs_coefficient_A"].quantile(0.95)),
                    "p99_max_abs_coefficient_A": float(columns["max_abs_coefficient_A"].quantile(0.99)),
                    "total_negative_T_entries": year_data.total_negative_T_entries,
                    "total_negative_FD_entries": year_data.total_negative_FD_entries,
                    "most_negative_T_value": year_data.most_negative_T_value,
                    "most_negative_FD_value": year_data.most_negative_FD_value,
                    "approximate_spectral_radius": float(spectral_a["approximate_spectral_radius"]),
                    "approximate_spectral_radius_A": float(spectral_a["approximate_spectral_radius"]),
                    "approximate_spectral_radius_abs_A": float(spectral_abs_a["approximate_spectral_radius"]),
                    "spectral_converged": bool(spectral_a["converged"]),
                    "spectral_iterations": int(spectral_a["iterations"]),
                    "spectral_radius_above_one": bool(spectral_a["above_one"]),
                    "spectral_radius_abs_A_above_one": bool(spectral_abs_a["above_one"]),
                }
            ]
        )
        return summary

    def _negative_flows_or_empty(self, year_data: LeontiefYearData) -> pd.DataFrame:
        columns = ["Year", "matrix", "row_country_sector", "col_country_sector", "col_label", "value"]
        if year_data.negative_flows is None:
            return pd.DataFrame(columns=columns)
        return year_data.negative_flows.reindex(columns=columns)

    def _build_top_unstable_nodes(self, columns: pd.DataFrame) -> pd.DataFrame:
        return (
            columns.sort_values(
                ["suspicion_score", "abs_column_sum_A", "max_abs_coefficient_A"],
                ascending=[False, False, False],
            )
            .head(100)
            .copy()
        )
