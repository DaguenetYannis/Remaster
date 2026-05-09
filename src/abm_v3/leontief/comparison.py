from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.abm_v3.config import ABMV3Config
from src.abm_v3.leontief.coefficients import LeontiefCoefficientBuilder
from src.abm_v3.leontief.outputs import LeontiefOutputWriter
from src.abm_v3.leontief.propagation import LeontiefPropagationEngine
from src.abm_v3.leontief.validation import LeontiefPropagationValidator
from src.abm_v3.leontief.viability import LeontiefViabilityAnalyzer
from src.abm_v3.paths import ABMV3Paths


class LeontiefModeComparator:
    """Run and summarize alternative Leontief coefficient-construction modes."""

    def __init__(self, paths: ABMV3Paths, config: ABMV3Config) -> None:
        self.paths = paths
        self.config = config

    def compare_year(self, year: int, modes: list[str] | None = None) -> pd.DataFrame:
        """Run coefficient building, diagnostics, propagation, and validation for each mode."""
        active_modes = modes or list(self.config.leontief.allowed_leontief_modes)
        rows = []
        writer = LeontiefOutputWriter(self.paths)
        for mode in active_modes:
            mode_config = replace(self.config, leontief=replace(self.config.leontief, leontief_mode=mode))
            year_data = LeontiefCoefficientBuilder(self.paths, mode_config.leontief).load_year(year)
            viability = LeontiefViabilityAnalyzer(mode_config.leontief).analyze(year_data)
            writer.write_viability(viability)
            result = LeontiefPropagationEngine(
                tolerance=mode_config.leontief.tolerance,
                max_rounds=mode_config.leontief.max_rounds,
            ).propagate(year_data)
            validator = LeontiefPropagationValidator()
            node_comparison = validator.build_node_comparison(year_data, result)
            propagation_summary = validator.build_summary(year_data, result, node_comparison)
            writer.write_all(year_data, result, node_comparison, propagation_summary)
            rows.append(self._comparison_row(year, mode, viability.summary, propagation_summary, result, year_data.mode_diagnostics))
        comparison = pd.DataFrame(rows)
        self.paths.leontief_pure_mode_comparison_diagnostics_dir.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(self.paths.leontief_mode_comparison_path(year), index=False)
        return comparison

    def _comparison_row(
        self,
        year: int,
        mode: str,
        viability_summary: pd.DataFrame,
        propagation_summary: pd.DataFrame,
        result: object,
        mode_diagnostics: pd.DataFrame | None,
    ) -> dict[str, object]:
        viability = viability_summary.iloc[0]
        propagation = propagation_summary.iloc[0]
        mode_diag = mode_diagnostics.iloc[0] if mode_diagnostics is not None and len(mode_diagnostics) else {}
        return {
            "Year": year,
            "mode": mode,
            "converged": bool(result.converged),
            "rounds_used": int(result.rounds_used),
            "final_residual_share": float(result.final_residual_share),
            "approximate_spectral_radius_A": float(viability["approximate_spectral_radius_A"]),
            "approximate_spectral_radius_abs_A": float(viability["approximate_spectral_radius_abs_A"]),
            "spectral_radius_above_one": bool(viability["spectral_radius_above_one"]),
            "suspicious_column_count": int(viability["suspicious_column_count"]),
            "negative_final_demand_count": int(viability["negative_final_demand_count"]),
            "high_abs_column_sum_count": int(viability["high_abs_column_sum_count"]),
            "invalid_output_column_count": int(viability["invalid_output_column_count"]),
            "observed_output_total": float(propagation["observed_output_total"]),
            "accumulated_output_total": float(propagation["accumulated_output_total"]),
            "relative_error_total": float(propagation["relative_error_total"]),
            "correlation_iterative_vs_observed": float(propagation["correlation_iterative_vs_observed"]),
            "mean_absolute_percentage_error": float(propagation["mean_absolute_percentage_error"]),
            "excluded_fd_column_count": int(self._mode_value(mode_diag, "excluded_fd_column_count", 0)),
            "rescaled_column_count": int(self._mode_value(mode_diag, "rescaled_column_count", 0)),
            "total_excluded_inventory_value": float(self._mode_value(mode_diag, "total_excluded_inventory_value", 0.0)),
            "total_abs_adjustment": float(self._mode_value(mode_diag, "total_abs_adjustment", 0.0)),
            "notes": self._mode_notes(mode_diag),
        }

    def _mode_value(self, mode_diag: object, column: str, default: object) -> object:
        try:
            value = mode_diag[column]
        except Exception:
            return default
        if isinstance(value, float) and not np.isfinite(value):
            return default
        return value

    def _mode_notes(self, mode_diag: object) -> str:
        excluded = self._mode_value(mode_diag, "excluded_fd_column_count", 0)
        rescaled = self._mode_value(mode_diag, "rescaled_column_count", 0)
        return f"excluded_fd_columns={excluded}; rescaled_columns={rescaled}"
