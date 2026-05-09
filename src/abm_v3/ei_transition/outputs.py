from __future__ import annotations

import pandas as pd

from src.abm_v3.ei_transition.panel import EITransitionPanelResult
from src.abm_v3.paths import ABMV3Paths


class EITransitionOutputWriter:
    """Write EI transition panels, diagnostics, models, and predictions."""

    def __init__(self, paths: ABMV3Paths) -> None:
        self.paths = paths

    def write_panel(
        self,
        start_year: int,
        end_year: int,
        result: EITransitionPanelResult,
    ) -> dict[str, object]:
        """Write transition panel and sample diagnostics."""
        self.paths.ei_transition_inputs_dir.mkdir(parents=True, exist_ok=True)
        self.paths.ei_transition_diagnostics_dir.mkdir(parents=True, exist_ok=True)
        panel_path = self.paths.ei_transition_panel_path(start_year, end_year)
        sample_report_path = self.paths.ei_transition_sample_report_path(start_year, end_year)
        by_year_path = self.paths.ei_transition_sample_report_by_year_path(start_year, end_year)
        result.panel.to_parquet(panel_path, index=False)
        result.sample_report.to_csv(sample_report_path, index=False)
        result.sample_report_by_year.to_csv(by_year_path, index=False)
        return {
            "panel": panel_path,
            "sample_report": sample_report_path,
            "sample_report_by_year": by_year_path,
        }

    def write_fit_outputs(
        self,
        start_year: int,
        end_year: int,
        scores: pd.DataFrame,
        coefficients: pd.DataFrame,
        expected_signs: pd.DataFrame,
        predictions: pd.DataFrame,
    ) -> dict[str, object]:
        """Write model validation, coefficients, sign checks, and predictions."""
        self.paths.ei_transition_diagnostics_dir.mkdir(parents=True, exist_ok=True)
        self.paths.ei_transition_models_dir.mkdir(parents=True, exist_ok=True)
        self.paths.ei_transition_predictions_dir.mkdir(parents=True, exist_ok=True)
        scores_path = self.paths.ei_transition_model_scores_path(start_year, end_year)
        coefficients_path = self.paths.ei_transition_coefficients_path(start_year, end_year)
        signs_path = self.paths.ei_transition_expected_signs_path(start_year, end_year)
        predictions_path = self.paths.ei_transition_predictions_path(start_year, end_year)
        scores.to_csv(scores_path, index=False)
        coefficients.to_csv(coefficients_path, index=False)
        expected_signs.to_csv(signs_path, index=False)
        predictions.to_parquet(predictions_path, index=False)
        return {
            "scores": scores_path,
            "coefficients": coefficients_path,
            "expected_signs": signs_path,
            "predictions": predictions_path,
        }
