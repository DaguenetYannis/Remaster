from __future__ import annotations

import itertools
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.abm.model import GreenTransitionABM
from src.abm.scenarii import Scenario, get_scenario


LOGGER = logging.getLogger(__name__)


DEFAULT_PARAMETER_GRID: dict[str, list[float]] = {
    "alpha": [0.005, 0.01, 0.02, 0.03, 0.05],
    "beta": [0.0, 0.005, 0.01, 0.02, 0.03],
    "kappa": [1.00, 1.05, 1.10, 1.20],
    "inventory_days": [15.0, 30.0, 60.0],
}


DEFAULT_TARGET_WEIGHTS: dict[str, float] = {
    "total_emissions": 1.0,
    "total_output": 1.0,
    "mean_ei": 1.0,
}


class CalibrationError(Exception):
    """Raised when ABM calibration cannot be completed."""


class ABMCalibrationTargetBuilder:
    """Builds empirical calibration targets from the ABM metrics panel."""

    def __init__(self, metrics_panel: pd.DataFrame) -> None:
        self.metrics_panel = metrics_panel.copy()
        self.metrics_panel["Year"] = pd.to_numeric(
            self.metrics_panel["Year"],
            errors="coerce",
        ).astype("Int64")

    def build_empirical_targets(
        self,
        start_year: int,
        end_year: int,
    ) -> pd.DataFrame:
        target_panel = self.metrics_panel[
            (self.metrics_panel["Year"] >= start_year)
            & (self.metrics_panel["Year"] <= end_year)
        ].copy()

        if target_panel.empty:
            available_years = self._available_years()
            raise CalibrationError(
                f"No empirical metrics found between {start_year} and {end_year}. "
                f"Available years: {available_years}"
            )

        required_columns = ["Year", "X", "EI", "g_local", "g_in", "g_out"]
        self._validate_columns(target_panel, required_columns)

        target_panel["emissions"] = target_panel["X"] * target_panel["EI"]

        empirical_targets = (
            target_panel.groupby("Year", as_index=False)
            .apply(self._summarize_year, include_groups=False)
            .reset_index(drop=True)
        )

        empirical_targets["step"] = empirical_targets["Year"] - start_year

        return empirical_targets.sort_values("step").reset_index(drop=True)

    @staticmethod
    def _summarize_year(year_df: pd.DataFrame) -> pd.Series:
        total_output = float(year_df["X"].sum())
        total_emissions = float(year_df["emissions"].sum())

        if total_output <= 0:
            mean_ei = 0.0
            mean_g_local = 0.0
            mean_g_in = 0.0
            mean_g_out = 0.0
        else:
            mean_ei = float(np.average(year_df["EI"], weights=year_df["X"]))
            mean_g_local = float(np.average(year_df["g_local"], weights=year_df["X"]))
            mean_g_in = float(np.average(year_df["g_in"], weights=year_df["X"]))
            mean_g_out = float(np.average(year_df["g_out"], weights=year_df["X"]))

        return pd.Series(
            {
                "total_output": total_output,
                "total_emissions": total_emissions,
                "mean_ei": mean_ei,
                "mean_g_local": mean_g_local,
                "mean_g_in": mean_g_in,
                "mean_g_out": mean_g_out,
            }
        )

    def _available_years(self) -> list[int]:
        return sorted(
            int(year)
            for year in self.metrics_panel["Year"].dropna().unique().tolist()
        )

    @staticmethod
    def _validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
        missing = [column for column in required_columns if column not in df.columns]

        if missing:
            raise CalibrationError(f"Missing empirical target columns: {missing}")


class ABMSimulationEvaluator:
    """Runs simulations and compares simulated targets to empirical targets."""

    def __init__(
        self,
        metrics_panel: pd.DataFrame,
        target_weights: dict[str, float] | None = None,
    ) -> None:
        self.metrics_panel = metrics_panel.copy()
        self.target_weights = target_weights or DEFAULT_TARGET_WEIGHTS

    def simulate(
        self,
        scenario: Scenario,
        start_year: int,
        n_steps: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        scenario_for_run = replace(scenario, n_steps=n_steps)

        model = GreenTransitionABM(
            metrics_panel=self.metrics_panel,
            scenario=scenario_for_run,
            start_year=start_year,
        )

        return model.run()

    def build_simulated_targets(
        self,
        aggregate_results: pd.DataFrame,
    ) -> pd.DataFrame:
        required_columns = ["step", *self.target_weights.keys()]
        self._validate_columns(aggregate_results, required_columns)

        return aggregate_results[required_columns].copy().sort_values("step")

    def compute_loss(
        self,
        empirical_targets: pd.DataFrame,
        simulated_targets: pd.DataFrame,
    ) -> float:
        merged = empirical_targets.merge(
            simulated_targets,
            on="step",
            suffixes=("_empirical", "_simulated"),
            how="inner",
        )

        if merged.empty:
            raise CalibrationError("No overlapping steps between empirical and simulated targets.")

        total_loss = 0.0

        for target_name, weight in self.target_weights.items():
            empirical_col = f"{target_name}_empirical"
            simulated_col = f"{target_name}_simulated"

            empirical_values = merged[empirical_col].to_numpy(dtype=float)
            simulated_values = merged[simulated_col].to_numpy(dtype=float)

            relative_error = self._safe_relative_error(
                simulated_values=simulated_values,
                empirical_values=empirical_values,
            )

            target_loss = float(np.mean(relative_error**2))
            total_loss += weight * target_loss

        return float(total_loss)

    @staticmethod
    def _safe_relative_error(
        simulated_values: np.ndarray,
        empirical_values: np.ndarray,
        epsilon: float = 1e-12,
    ) -> np.ndarray:
        denominator = np.where(
            np.abs(empirical_values) > epsilon,
            np.abs(empirical_values),
            epsilon,
        )

        return (simulated_values - empirical_values) / denominator

    @staticmethod
    def _validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
        missing = [column for column in required_columns if column not in df.columns]

        if missing:
            raise CalibrationError(f"Missing simulation target columns: {missing}")


class ABMGridSearchCalibrator:
    """Calibrates an ABM scenario through transparent grid search."""

    def __init__(
        self,
        metrics_panel: pd.DataFrame,
        parameter_grid: dict[str, list[float]] | None = None,
        target_weights: dict[str, float] | None = None,
    ) -> None:
        self.metrics_panel = metrics_panel.copy()
        self.parameter_grid = parameter_grid or DEFAULT_PARAMETER_GRID
        self.target_builder = ABMCalibrationTargetBuilder(self.metrics_panel)
        self.evaluator = ABMSimulationEvaluator(
            metrics_panel=self.metrics_panel,
            target_weights=target_weights,
        )

    def calibrate(
        self,
        scenario_name: str,
        start_year: int,
        end_year: int,
        output_dir: Path = Path("outputs/abm/calibration"),
    ) -> tuple[Scenario, pd.DataFrame]:
        if end_year <= start_year:
            raise CalibrationError("Calibration end year must be greater than start year.")

        output_dir.mkdir(parents=True, exist_ok=True)

        base_scenario = get_scenario(scenario_name)
        n_steps = end_year - start_year

        empirical_targets = self.target_builder.build_empirical_targets(
            start_year=start_year,
            end_year=end_year,
        )

        search_results: list[dict[str, Any]] = []

        for parameter_values in self._iter_parameter_grid():
            candidate_scenario = replace(base_scenario, **parameter_values)

            try:
                _, aggregate_results = self.evaluator.simulate(
                    scenario=candidate_scenario,
                    start_year=start_year,
                    n_steps=n_steps,
                )

                simulated_targets = self.evaluator.build_simulated_targets(
                    aggregate_results=aggregate_results,
                )

                loss = self.evaluator.compute_loss(
                    empirical_targets=empirical_targets,
                    simulated_targets=simulated_targets,
                )

                result = {
                    **parameter_values,
                    "loss": loss,
                    "status": "ok",
                    "error": "",
                }

            except Exception as exc:
                LOGGER.warning(
                    "Calibration candidate failed: %s | error=%s",
                    parameter_values,
                    exc,
                )

                result = {
                    **parameter_values,
                    "loss": np.inf,
                    "status": "failed",
                    "error": str(exc),
                }

            search_results.append(result)

        results_df = pd.DataFrame(search_results).sort_values("loss").reset_index(drop=True)

        if results_df.empty or not np.isfinite(results_df.loc[0, "loss"]):
            raise CalibrationError("All calibration candidates failed.")

        best_parameters = {
            parameter_name: results_df.loc[0, parameter_name]
            for parameter_name in self.parameter_grid.keys()
        }

        calibrated_scenario = replace(base_scenario, **best_parameters)

        self._save_outputs(
            output_dir=output_dir,
            results_df=results_df,
            calibrated_scenario=calibrated_scenario,
            best_parameters=best_parameters,
            scenario_name=scenario_name,
            start_year=start_year,
            end_year=end_year,
        )

        return calibrated_scenario, results_df

    def _iter_parameter_grid(self) -> list[dict[str, float]]:
        parameter_names = list(self.parameter_grid.keys())
        parameter_values = [self.parameter_grid[name] for name in parameter_names]

        combinations = itertools.product(*parameter_values)

        return [
            dict(zip(parameter_names, combination, strict=True))
            for combination in combinations
        ]

    @staticmethod
    def _save_outputs(
        output_dir: Path,
        results_df: pd.DataFrame,
        calibrated_scenario: Scenario,
        best_parameters: dict[str, float],
        scenario_name: str,
        start_year: int,
        end_year: int,
    ) -> None:
        results_path = output_dir / "calibration_results.csv"
        best_parameters_path = output_dir / "best_parameters.json"
        calibrated_scenario_path = output_dir / "calibrated_scenario.json"

        results_df.to_csv(results_path, index=False)

        metadata = {
            "scenario_name": scenario_name,
            "start_year": start_year,
            "end_year": end_year,
            "best_parameters": best_parameters,
            "best_loss": float(results_df.loc[0, "loss"]),
        }

        with best_parameters_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

        with calibrated_scenario_path.open("w", encoding="utf-8") as file:
            json.dump(asdict(calibrated_scenario), file, indent=2)


def calibrate_scenario(
    metrics_panel: pd.DataFrame,
    scenario_name: str,
    start_year: int,
    end_year: int,
    output_dir: Path = Path("outputs/abm/calibration"),
    parameter_grid: dict[str, list[float]] | None = None,
    target_weights: dict[str, float] | None = None,
) -> tuple[Scenario, pd.DataFrame]:
    calibrator = ABMGridSearchCalibrator(
        metrics_panel=metrics_panel,
        parameter_grid=parameter_grid,
        target_weights=target_weights,
    )

    return calibrator.calibrate(
        scenario_name=scenario_name,
        start_year=start_year,
        end_year=end_year,
        output_dir=output_dir,
    )


def load_calibrated_parameters(
    path: Path = Path("outputs/abm/calibration/best_parameters.json"),
) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"Missing calibrated parameters file: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if "best_parameters" not in payload:
        raise CalibrationError(f"Invalid calibration file: {path}")

    return payload["best_parameters"]


def apply_calibrated_parameters(
    scenario: Scenario,
    parameters: dict[str, float],
) -> Scenario:
    valid_fields = set(asdict(scenario).keys())

    filtered_parameters = {
        key: value
        for key, value in parameters.items()
        if key in valid_fields
    }

    return replace(scenario, **filtered_parameters)