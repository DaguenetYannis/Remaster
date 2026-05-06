from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.abm.scenario_config import DEFAULT_SCENARIOS, ScenarioConfig


class ScenarioRunner:
    def __init__(self, config: ScenarioConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

        self.features = [
            "emissions_intensity",
            "g_base",
            "g_out_network",
            "g_in_network",
            "pagerank",
            "out_strength",
            "in_strength",
        ]

    def run(self, save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        agents = pd.read_parquet(self.config.agents_path)
        transitions = pd.read_parquet(self.config.transitions_path)

        self.features = [col for col in self.features if col in transitions.columns]

        reg_model = self._train_continuous_model(transitions)
        clf_model = self._train_change_model(transitions)
        transition_matrix = self._build_empirical_transition_matrix(transitions)

        last_year = int(agents["Year"].max())
        state = agents[agents["Year"] == last_year].copy()

        simulated_frames: list[pd.DataFrame] = []

        for step in range(1, self.config.n_steps + 1):
            state = self._simulate_step(
                state=state,
                reg_model=reg_model,
                clf_model=clf_model,
                transition_matrix=transition_matrix,
            )

            state["scenario"] = self.config.scenario_name
            state["sim_year"] = last_year + step
            simulated_frames.append(state.copy())

        simulation_panel = pd.concat(simulated_frames, ignore_index=True)
        summary_panel = self._build_summary_panel(simulation_panel)

        if save:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            simulation_panel.to_parquet(self.config.output_panel_path(), index=False)
            summary_panel.to_parquet(self.config.output_summary_path(), index=False)

        return simulation_panel, summary_panel

    def _train_continuous_model(self, transitions: pd.DataFrame) -> Pipeline:
        target = "delta_log_emissions_intensity_winsorized"

        if target not in transitions.columns:
            raise ValueError(f"Missing target column: {target}")

        x = transitions[self.features]
        y = pd.to_numeric(transitions[target], errors="coerce")

        mask = y.notna()
        x = x.loc[mask]
        y = y.loc[mask]

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )

        model.fit(x, y)
        return model

    def _train_change_model(self, transitions: pd.DataFrame) -> Pipeline:
        df = transitions.dropna(subset=["regime", "regime_next"]).copy()
        df["changed"] = (df["regime"] != df["regime_next"]).astype(int)

        x = df[self.features]
        y = df["changed"]

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=12,
                        min_samples_leaf=10,
                        class_weight="balanced",
                        random_state=self.config.random_seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        model.fit(x, y)
        return model

    def _build_empirical_transition_matrix(
        self,
        transitions: pd.DataFrame,
    ) -> dict[str, list[tuple[str, float]]]:
        df = transitions.dropna(subset=["regime", "regime_next"]).copy()

        counts = (
            df.groupby(["regime", "regime_next"])
            .size()
            .reset_index(name="count")
        )

        counts["probability"] = counts.groupby("regime")["count"].transform(
            lambda s: s / s.sum()
        )

        matrix: dict[str, list[tuple[str, float]]] = {}

        for regime, group in counts.groupby("regime"):
            matrix[str(regime)] = list(
                zip(
                    group["regime_next"].astype(str),
                    group["probability"].astype(float),
                )
            )

        return matrix

    def _simulate_step(
        self,
        state: pd.DataFrame,
        reg_model: Pipeline,
        clf_model: Pipeline,
        transition_matrix: dict[str, list[tuple[str, float]]],
    ) -> pd.DataFrame:
        state = state.copy()

        x = self._build_scenario_features(state)

        delta_log_ei = reg_model.predict(x)
        delta_log_ei = self._apply_brown_core_ei_intervention(
            state=state,
            delta_log_ei=delta_log_ei,
        )

        current_log_ei = np.log1p(
            pd.to_numeric(state["emissions_intensity"], errors="coerce").fillna(0)
        )

        new_log_ei = current_log_ei + delta_log_ei
        state["emissions_intensity"] = np.expm1(new_log_ei).clip(lower=0)

        state["g_base"] = -np.log(state["emissions_intensity"] + 1e-12)
        state["g_base"] = state["g_base"].clip(lower=0, upper=10)

        raw_change_proba = clf_model.predict_proba(x)[:, 1]

        calibrated_change_proba = (
            raw_change_proba * self.config.transition_probability_scale
        )

        calibrated_change_proba = self._apply_capability_policy_boost(
            state=state,
            probability=calibrated_change_proba,
        )

        calibrated_change_proba = self._apply_brown_core_transition_boost(
            state=state,
            probability=calibrated_change_proba,
        )

        calibrated_change_proba = np.clip(calibrated_change_proba, 0, 1)

        draws = self.rng.random(len(state))
        should_change = draws < calibrated_change_proba

        state["regime_change_probability"] = calibrated_change_proba
        state["regime_changed"] = should_change

        state.loc[should_change, "regime"] = self._sample_next_regimes(
            current_regimes=state.loc[should_change, "regime"],
            transition_matrix=transition_matrix,
        )

        state["green_status"] = state["regime"].str.split("_").str[0]
        state["network_status"] = state["regime"].str.split("_").str[1]

        return state

    def _build_scenario_features(self, state: pd.DataFrame) -> pd.DataFrame:
        x = state[self.features].copy()

        if "g_out_network" in x.columns:
            x["g_out_network"] = (
                x["g_out_network"] * self.config.network_diffusion_boost
            )

        if "g_in_network" in x.columns:
            x["g_in_network"] = (
                x["g_in_network"] * self.config.network_diffusion_boost
            )

        return x

    def _apply_capability_policy_boost(
        self,
        state: pd.DataFrame,
        probability: np.ndarray,
    ) -> np.ndarray:
        if self.config.capability_policy_boost <= 0:
            return probability

        if "green_capability_share" not in state.columns:
            return probability

        capability = (
            pd.to_numeric(state["green_capability_share"], errors="coerce")
            .fillna(0)
            .clip(lower=0, upper=1)
            .to_numpy()
        )

        return probability * (1 + self.config.capability_policy_boost * capability)

    def _apply_brown_core_transition_boost(
        self,
        state: pd.DataFrame,
        probability: np.ndarray,
    ) -> np.ndarray:
        if self.config.brown_core_intervention <= 0:
            return probability

        is_brown_core = state["regime"].astype(str).eq("brown_core").to_numpy()

        probability = probability.copy()
        probability[is_brown_core] = probability[is_brown_core] * (
            1 + self.config.brown_core_intervention
        )

        return probability

    def _apply_brown_core_ei_intervention(
        self,
        state: pd.DataFrame,
        delta_log_ei: np.ndarray,
    ) -> np.ndarray:
        if self.config.brown_core_intervention <= 0:
            return delta_log_ei

        is_brown_core = state["regime"].astype(str).eq("brown_core").to_numpy()

        delta_log_ei = delta_log_ei.copy()

        additional_reduction = 0.0001 * self.config.brown_core_intervention
        delta_log_ei[is_brown_core] = delta_log_ei[is_brown_core] - additional_reduction

        return delta_log_ei

    def _sample_next_regimes(
        self,
        current_regimes: pd.Series,
        transition_matrix: dict[str, list[tuple[str, float]]],
    ) -> list[str]:
        sampled: list[str] = []

        for regime in current_regimes.astype(str):
            options = transition_matrix.get(regime)

            if not options:
                sampled.append(regime)
                continue

            next_regimes = [item[0] for item in options]
            probabilities = np.array([item[1] for item in options], dtype=float)
            probabilities = probabilities / probabilities.sum()

            sampled.append(str(self.rng.choice(next_regimes, p=probabilities)))

        return sampled

    def _build_summary_panel(self, simulation_panel: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, float | int | str]] = []

        for (scenario, sim_year), group in simulation_panel.groupby(
            ["scenario", "sim_year"]
        ):
            regime_shares = group["regime"].value_counts(normalize=True)

            rows.append(
                {
                    "scenario": scenario,
                    "sim_year": int(sim_year),
                    "mean_emissions_intensity": group[
                        "emissions_intensity"
                    ].mean(),
                    "median_emissions_intensity": group[
                        "emissions_intensity"
                    ].median(),
                    "regime_change_share": group["regime_changed"].mean(),
                    "brown_core_share": regime_shares.get("brown_core", 0.0),
                    "brown_periphery_share": regime_shares.get(
                        "brown_periphery",
                        0.0,
                    ),
                    "green_core_share": regime_shares.get("green_core", 0.0),
                    "green_periphery_share": regime_shares.get(
                        "green_periphery",
                        0.0,
                    ),
                }
            )

        return pd.DataFrame(rows).sort_values(["scenario", "sim_year"])


def run_named_scenario(name: str, save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    if name not in DEFAULT_SCENARIOS:
        available = ", ".join(DEFAULT_SCENARIOS)
        raise ValueError(f"Unknown scenario '{name}'. Available: {available}")

    runner = ScenarioRunner(DEFAULT_SCENARIOS[name])
    return runner.run(save=save)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenario",
        default="baseline",
        choices=sorted(DEFAULT_SCENARIOS.keys()),
    )
    parser.add_argument("--no-save", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    _, summary = run_named_scenario(
        name=args.scenario,
        save=not args.no_save,
    )

    print("\nScenario summary:")
    print(summary)


if __name__ == "__main__":
    main()