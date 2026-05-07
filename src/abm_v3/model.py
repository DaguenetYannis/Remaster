from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.abm_v3.calibration.emissions_model import EmissionsIntensityModel
from src.abm_v3.calibration.loss_functions import output_loss
from src.abm_v3.calibration.production_model import ProductionPlanningModel
from src.abm_v3.calibration.substitution_model import SubstitutionFrictionModel
from src.abm_v3.calibration.validation import HistoricalValidator
from src.abm_v3.capability import add_capability_features
from src.abm_v3.config import ABMV3Config
from src.abm_v3.data_loader import ABMV3DataLoader
from src.abm_v3.diagnostics.collapse import detect_bad_transition
from src.abm_v3.diagnostics.hypothesis_reports import HypothesisReportGenerator
from src.abm_v3.dynamics.demand_provider import DemandProvider
from src.abm_v3.dynamics.step import ABMV3StepEngine
from src.abm_v3.feature_engineering import FeatureEngineer
from src.abm_v3.greenness import add_greenness_features
from src.abm_v3.outputs import ABMV3OutputWriter
from src.abm_v3.paths import ABMV3Paths
from src.abm_v3.real_data_smoke_test import RealDataSmokeTester
from src.abm_v3.scenarios.base import BaseScenario
from src.abm_v3.scenarios.registry import get_scenario
from src.abm_v3.state import ABMState, ABMStateMetadata

LOGGER = logging.getLogger(__name__)


@dataclass
class ABMV3Model:
    """Orchestrate ABM v3 data, calibration, validation, and simulation."""

    config: ABMV3Config = field(default_factory=ABMV3Config)
    paths: ABMV3Paths = field(default_factory=ABMV3Paths)
    data_loader: ABMV3DataLoader | None = None
    production_model: ProductionPlanningModel | None = None
    emissions_model: EmissionsIntensityModel | None = None
    substitution_model: SubstitutionFrictionModel | None = None
    scenario: BaseScenario | None = None

    def __post_init__(self) -> None:
        if self.data_loader is None:
            self.data_loader = ABMV3DataLoader(self.paths)
        if self.scenario is None:
            self.scenario = get_scenario("baseline_continuation")

    def fit_historical(
        self,
        start_year: int | None = None,
        end_year: int | None = None,
        ei_mode: str = "green_transition",
        validation_mode: str = "rolling",
        fit_full_after_validation: bool = True,
    ) -> dict[str, object]:
        start = self.config.calibration.start_year if start_year is None else start_year
        end = self.config.calibration.end_year if end_year is None else end_year
        LOGGER.info("Historical ABM v3 fit requested for %s-%s with EI mode %s.", start, end, ei_mode)
        panel = self.data_loader.load_abm_ready_historical_panel(start, end, self.config)
        smoke_report = RealDataSmokeTester(self.paths, self.config).run(panel, write_report=True)
        model_ready_panel = self.prepare_model_ready_panel(panel)
        missing_core = [column for column in ["country_sector", "Year", "X", "D", "EI"] if column not in model_ready_panel.columns]
        if missing_core:
            message = f"Historical fit blocked; model-ready panel is missing core columns: {missing_core}"
            LOGGER.warning(message)
            return {
                "status": "blocked_missing_core_columns",
                "start_year": start,
                "end_year": end,
                "ei_mode": ei_mode,
                "validation_mode": validation_mode,
                "missing_core_columns": missing_core,
                "smoke_checks_passed": int(smoke_report["passed"].sum()) if "passed" in smoke_report.columns else 0,
            }
        validator = HistoricalValidator(split_year=self.config.calibration.validation_split_year)
        splits = self._validation_splits(validator, validation_mode, start, end)
        sigma_model = self.calibrate_sigma(model_ready_panel, splits, ei_mode)
        validation_results = self.run_validation_splits(
            model_ready_panel,
            splits=splits,
            ei_mode=ei_mode,
            sigma=sigma_model.get_sigma(),
        )
        writer = ABMV3OutputWriter(self.paths)
        writer.write_dataframe(validation_results, "validation", "rolling_validation_results.csv")
        writer.write_dataframe(sigma_model.get_results(), "calibration", "sigma_grid_results.csv")

        full_reproduction = pd.DataFrame()
        if fit_full_after_validation:
            self.production_model, self.emissions_model = self.fit_component_models(model_ready_panel, ei_mode)
            self.substitution_model = sigma_model
            self._write_model_coefficients()
            full_reproduction = self.simulate_historical_recursive(
                model_ready_panel,
                start_year=start,
                end_year=end,
                sigma=sigma_model.get_sigma(),
            )
            summary = self._historical_reproduction_summary(full_reproduction, model_ready_panel)
            writer.write_dataframe(summary, "validation", "historical_reproduction_summary.csv")
            HypothesisReportGenerator(self.paths).write_all(model_ready_panel, sigma_model.get_results())

        return {
            "status": "fit_complete",
            "start_year": start,
            "end_year": end,
            "ei_mode": ei_mode,
            "validation_mode": validation_mode,
            "best_sigma": sigma_model.get_sigma(),
            "smoke_checks_passed": int(smoke_report["passed"].sum()) if "passed" in smoke_report.columns else 0,
            "validation_rows": len(validation_results),
            "historical_reproduction_rows": len(full_reproduction),
        }

    def validate_historical(self, split_year: int | None = None) -> dict[str, object]:
        split = self.config.calibration.validation_split_year if split_year is None else split_year
        LOGGER.info("Fixed-split historical validation requested with split year %s.", split)
        panel = self.prepare_model_ready_panel(
            self.data_loader.load_abm_ready_historical_panel(
                self.config.calibration.start_year,
                self.config.calibration.end_year,
                self.config,
            )
        )
        missing_core = [column for column in ["country_sector", "Year", "X", "D", "EI"] if column not in panel.columns]
        if missing_core:
            LOGGER.warning("Historical validation blocked; missing core columns: %s", missing_core)
            return {"status": "blocked_missing_core_columns", "split_year": split, "missing_core_columns": missing_core}
        validator = HistoricalValidator(split_year=split)
        splits = validator.fixed_split_years(
            self.config.calibration.start_year,
            self.config.calibration.end_year,
            split,
        )
        sigma_model = self.calibrate_sigma(panel, splits, "green_transition")
        results = self.run_validation_splits(panel, splits, "green_transition", sigma_model.get_sigma())
        ABMV3OutputWriter(self.paths).write_dataframe(results, "validation", "fixed_split_validation_results.csv")
        return {"status": "validated", "split_year": split, "rows": len(results), "best_sigma": sigma_model.get_sigma()}

    def simulate(
        self,
        start_year: int,
        end_year: int,
        scenario: BaseScenario | str | None = None,
        initial_state: ABMState | None = None,
    ) -> pd.DataFrame:
        scenario_obj = get_scenario(scenario) if isinstance(scenario, str) else scenario or self.scenario
        if initial_state is None:
            LOGGER.info("Scaffold simulation requested without initial_state; returning empty frame.")
            return pd.DataFrame(
                columns=["country_sector", "Year", "X", "D", "EI", "emissions", "scenario"]
            )
        engine = ABMV3StepEngine(
            production_model=self.production_model,
            emissions_model=self.emissions_model,
            demand_provider=DemandProvider(None),
            sigma=self.substitution_model.get_sigma() if self.substitution_model else self.config.substitution.substitution_friction,
            input_rigidity=self.config.production_feasibility.input_rigidity,
        )
        states = [initial_state.nodes.assign(scenario=scenario_obj.name)]
        state = initial_state
        for year in range(start_year + 1, end_year + 1):
            state, _diagnostics = engine.step(state, next_year=year, scenario=scenario_obj)
            states.append(state.nodes.assign(scenario=scenario_obj.name))
        return pd.concat(states, ignore_index=True)

    def prepare_model_ready_panel(self, panel: pd.DataFrame) -> pd.DataFrame:
        result = self._canonicalize_abm_columns(panel)
        if "green_capability" not in result.columns:
            try:
                result = add_capability_features(result, self.config.capability)
            except ValueError as error:
                LOGGER.warning("Could not create green_capability from Atlas source columns: %s", error)
        if "general_complexity" not in result.columns and "capability_export_weighted_pci" in result.columns:
            result["general_complexity"] = result["capability_export_weighted_pci"].astype(float)
        if "g_local" not in result.columns and "EI" in result.columns:
            result = add_greenness_features(result, self.config.greenness)
        model_ready = FeatureEngineer().create_model_ready_panel(result)
        if "g_network" not in model_ready.columns and {"g_in", "g_out"}.issubset(model_ready.columns):
            model_ready["g_network"] = 0.5 * (model_ready["g_in"].astype(float) + model_ready["g_out"].astype(float))
        return model_ready

    def _canonicalize_abm_columns(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Create ABM-ready aliases only where the mapping is explicit."""

        result = panel.copy()
        if "country_sector" not in result.columns and {"Country", "Country_detail", "Category", "Sector"}.issubset(result.columns):
            result["country_sector"] = (
                result["Country"].astype(str)
                + " | "
                + result["Country_detail"].astype(str)
                + " | "
                + result["Category"].astype(str)
                + " | "
                + result["Sector"].astype(str)
            )
        rename_map = {
            "emissions_intensity": "EI",
            "g_in_network": "g_in",
            "g_out_network": "g_out",
            "g_base": "g_local",
        }
        for source, target in rename_map.items():
            if target not in result.columns and source in result.columns:
                result[target] = result[source]
        return result

    def fit_component_models(
        self,
        panel: pd.DataFrame,
        ei_mode: str,
    ) -> tuple[ProductionPlanningModel, EmissionsIntensityModel]:
        production_features = [feature for feature in self.config.calibration.production_features if feature in panel.columns]
        production_model = ProductionPlanningModel(features=production_features).fit(panel)
        emissions_model = EmissionsIntensityModel(mode=ei_mode).fit(panel)
        return production_model, emissions_model

    def calibrate_sigma(
        self,
        panel: pd.DataFrame,
        splits: list[dict[str, int]],
        ei_mode: str,
    ) -> SubstitutionFrictionModel:
        def evaluator(sigma: float) -> dict[str, float]:
            validation = self.run_validation_splits(panel, splits, ei_mode, sigma)
            if "output_loss" not in validation.columns:
                return {
                    "output_validation_loss": float("nan"),
                    "collapse_penalty": float("nan"),
                    "mean_substitution_gain": float("nan"),
                    "bad_transition_rate": float("nan"),
                }
            return {
                "output_validation_loss": float(validation["output_loss"].mean()),
                "collapse_penalty": float(validation["output_loss_fraction"].clip(lower=0.2).sub(0.2).mean()),
                "mean_substitution_gain": float(validation["total_substitution_gain"].mean()),
                "bad_transition_rate": float(validation["bad_transition"].mean()),
            }

        return SubstitutionFrictionModel(
            sigma_grid=self.config.calibration.sigma_grid,
        ).fit_grid(evaluator)

    def run_validation_splits(
        self,
        panel: pd.DataFrame,
        splits: list[dict[str, int]],
        ei_mode: str,
        sigma: float,
    ) -> pd.DataFrame:
        rows = []
        for split in splits:
            train_panel = panel[
                (panel["Year"] >= split["train_start_year"])
                & (panel["Year"] <= split["train_end_year"])
            ].copy()
            try:
                production_model, emissions_model = self.fit_component_models(train_panel, ei_mode)
                predicted = self.predict_historical_one_step(
                    panel,
                    current_year=split["validation_year"] - 1,
                    next_year=split["validation_year"],
                    production_model=production_model,
                    emissions_model=emissions_model,
                    sigma=sigma,
                )
                observed = panel[panel["Year"] == split["validation_year"]].copy()
                metrics = self._validation_metrics(predicted, observed)
                rows.append({**split, "sigma": sigma, **metrics})
            except ValueError as error:
                LOGGER.warning("Validation split failed and will be reported as NaN: %s", error)
                rows.append({**split, "sigma": sigma, "error": str(error)})
        return pd.DataFrame(rows)

    def predict_historical_one_step(
        self,
        panel: pd.DataFrame,
        current_year: int,
        next_year: int,
        production_model: ProductionPlanningModel | None = None,
        emissions_model: EmissionsIntensityModel | None = None,
        sigma: float | None = None,
    ) -> pd.DataFrame:
        current_nodes = panel[panel["Year"] == current_year].copy()
        if current_nodes.empty:
            raise ValueError(f"No current nodes for year {current_year}.")
        state = ABMState(nodes=current_nodes, metadata=ABMStateMetadata(year=current_year))
        engine = ABMV3StepEngine(
            production_model=production_model or self.production_model,
            emissions_model=emissions_model or self.emissions_model,
            demand_provider=DemandProvider(historical_panel=panel),
            sigma=self.config.substitution.substitution_friction if sigma is None else sigma,
            input_rigidity=self.config.production_feasibility.input_rigidity,
        )
        next_state, diagnostics = engine.step(state, next_year=next_year)
        return next_state.nodes.assign(**{key: value for key, value in diagnostics.items() if key != "year"})

    def simulate_historical_recursive(
        self,
        panel: pd.DataFrame,
        start_year: int,
        end_year: int,
        sigma: float,
    ) -> pd.DataFrame:
        state = ABMState(
            nodes=panel[panel["Year"] == start_year].copy(),
            metadata=ABMStateMetadata(year=start_year),
        )
        states = [state.nodes.assign(simulation_type="historical_recursive")]
        for year in range(start_year + 1, end_year + 1):
            engine = ABMV3StepEngine(
                production_model=self.production_model,
                emissions_model=self.emissions_model,
                demand_provider=DemandProvider(historical_panel=panel),
                sigma=sigma,
                input_rigidity=self.config.production_feasibility.input_rigidity,
            )
            state, _diagnostics = engine.step(state, next_year=year)
            states.append(state.nodes.assign(simulation_type="historical_recursive"))
        return pd.concat(states, ignore_index=True)

    def _validation_splits(
        self,
        validator: HistoricalValidator,
        validation_mode: str,
        start_year: int,
        end_year: int,
    ) -> list[dict[str, int]]:
        if validation_mode == "rolling":
            return validator.rolling_splits(
                start_year,
                end_year,
                minimum_training_window=self.config.calibration.minimum_training_window,
            )
        if validation_mode == "fixed_split":
            return validator.fixed_split_years(start_year, end_year)
        raise ValueError(f"Unsupported validation mode: {validation_mode}")

    def _validation_metrics(self, predicted: pd.DataFrame, observed: pd.DataFrame) -> dict[str, float]:
        predicted_subset = predicted[["country_sector", "Year", "X", "EI"]].rename(
            columns={"X": "X_simulated", "EI": "EI_simulated"}
        )
        observed_subset = observed[["country_sector", "Year", "X", "EI"]].rename(
            columns={"X": "X_actual", "EI": "EI_actual"}
        )
        merged = predicted_subset.merge(
            observed_subset,
            on=["country_sector", "Year"],
            how="inner",
        )
        collapse = detect_bad_transition(
            observed.assign(X=observed["X"], EI=observed["EI"]),
            predicted.assign(X=predicted["X"], EI=predicted["EI"]),
        )
        return {
            "output_loss": output_loss(merged["X_simulated"], merged["X_actual"]),
            "ei_loss": output_loss(merged["EI_simulated"], merged["EI_actual"]),
            "matched_rows": float(len(merged)),
            "total_substitution_gain": float(predicted.get("total_substitution_gain", pd.Series([np.nan])).iloc[0]),
            "bad_transition": float(collapse["bad_transition"]),
            "output_loss_fraction": float(collapse["output_loss_fraction"]),
        }

    def _historical_reproduction_summary(self, simulated: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for year in sorted(set(simulated["Year"]).intersection(set(observed["Year"]))):
            sim_year = simulated[simulated["Year"] == year]
            obs_year = observed[observed["Year"] == year]
            merged = sim_year.merge(
                obs_year[["country_sector", "Year", "X", "EI"]].rename(
                    columns={"X": "X_actual", "EI": "EI_actual"}
                ),
                on=["country_sector", "Year"],
                how="inner",
            )
            rows.append(
                {
                    "Year": year,
                    "global_output_simulated": float(sim_year["X"].sum()),
                    "global_output_observed": float(obs_year["X"].sum()),
                    "global_output_loss": output_loss(merged["X"], merged["X_actual"]),
                    "matched_rows": len(merged),
                }
            )
        return pd.DataFrame(rows)

    def _write_model_coefficients(self) -> None:
        writer = ABMV3OutputWriter(self.paths)
        if self.production_model is not None:
            production_coefficients = (
                self.production_model.get_coefficients()
                .rename("coefficient")
                .reset_index()
                .rename(columns={"index": "feature"})
            )
            writer.write_dataframe(
                production_coefficients,
                "calibration",
                "production_model_coefficients.csv",
            )
        if self.emissions_model is not None:
            emissions_coefficients = (
                self.emissions_model.get_coefficients()
                .rename("coefficient")
                .reset_index()
                .rename(columns={"index": "feature"})
            )
            writer.write_dataframe(
                emissions_coefficients,
                "calibration",
                "emissions_model_coefficients.csv",
            )
