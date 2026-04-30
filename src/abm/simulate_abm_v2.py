from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class SimulationConfig:
    agents_path: Path = Path("data/abm/agents_panel.parquet")
    transitions_path: Path = Path("data/abm/diagnostics/transitions_with_clean_targets.parquet")
    output_path: Path = Path("data/abm/simulation_output_v2.parquet")
    n_steps: int = 10
    random_seed: int = 42
    transition_probability_scale: float = 0.08


class ABMSimulatorV2:
    def __init__(self, config: SimulationConfig) -> None:
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

    def run(self) -> pd.DataFrame:
        agents = pd.read_parquet(self.config.agents_path)
        transitions = pd.read_parquet(self.config.transitions_path)

        self.features = [col for col in self.features if col in transitions.columns]

        reg_model = self.train_continuous_model(transitions)
        clf_model = self.train_change_model(transitions)
        transition_matrix = self.build_empirical_transition_matrix(transitions)

        last_year = int(agents["Year"].max())
        state = agents[agents["Year"] == last_year].copy()

        simulated_frames = []

        for step in range(1, self.config.n_steps + 1):
            state = self.simulate_step(
                state=state,
                reg_model=reg_model,
                clf_model=clf_model,
                transition_matrix=transition_matrix,
            )

            state["sim_year"] = last_year + step
            simulated_frames.append(state.copy())

            print(f"\n=== Simulated year {last_year + step} ===")
            print(state["regime"].value_counts(normalize=True).sort_index())

        result = pd.concat(simulated_frames, ignore_index=True)

        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(self.config.output_path, index=False)

        print("\nSaved simulation output to:", self.config.output_path)
        return result

    def train_continuous_model(self, transitions: pd.DataFrame) -> Pipeline:
        df = transitions.copy()

        target = "delta_log_emissions_intensity_winsorized"
        if target not in df.columns:
            raise ValueError(f"Missing target column: {target}")

        x = df[self.features]
        y = pd.to_numeric(df[target], errors="coerce")

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

    def train_change_model(self, transitions: pd.DataFrame) -> Pipeline:
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

    def build_empirical_transition_matrix(
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

    def simulate_step(
        self,
        state: pd.DataFrame,
        reg_model: Pipeline,
        clf_model: Pipeline,
        transition_matrix: dict[str, list[tuple[str, float]]],
    ) -> pd.DataFrame:
        state = state.copy()

        x = state[self.features]

        delta_log_ei = reg_model.predict(x)

        current_log_ei = np.log1p(
            pd.to_numeric(state["emissions_intensity"], errors="coerce").fillna(0)
        )

        new_log_ei = current_log_ei + delta_log_ei
        state["emissions_intensity"] = np.expm1(new_log_ei).clip(lower=0)

        state["g_base"] = -np.log(state["emissions_intensity"] + 1e-12)
        state["g_base"] = state["g_base"].clip(lower=0, upper=10)

        raw_change_proba = clf_model.predict_proba(x)[:, 1]
        calibrated_change_proba = np.clip(
            raw_change_proba * self.config.transition_probability_scale,
            0,
            1,
        )

        draws = self.rng.random(len(state))
        should_change = draws < calibrated_change_proba

        state["regime_change_probability"] = calibrated_change_proba
        state["regime_changed"] = should_change

        state.loc[should_change, "regime"] = self.sample_next_regimes(
            current_regimes=state.loc[should_change, "regime"],
            transition_matrix=transition_matrix,
        )

        state["green_status"] = state["regime"].str.split("_").str[0]
        state["network_status"] = state["regime"].str.split("_").str[1]

        return state

    def sample_next_regimes(
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

            sampled.append(
                str(self.rng.choice(next_regimes, p=probabilities))
            )

        return sampled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--agents-path", default="data/abm/agents_panel.parquet")
    parser.add_argument(
        "--transitions-path",
        default="data/abm/diagnostics/transitions_with_clean_targets.parquet",
    )
    parser.add_argument("--output-path", default="data/abm/simulation_output_v2.parquet")
    parser.add_argument("--n-steps", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--transition-probability-scale", type=float, default=0.08)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = SimulationConfig(
        agents_path=Path(args.agents_path),
        transitions_path=Path(args.transitions_path),
        output_path=Path(args.output_path),
        n_steps=args.n_steps,
        random_seed=args.random_seed,
        transition_probability_scale=args.transition_probability_scale,
    )

    simulator = ABMSimulatorV2(config)
    simulator.run()


if __name__ == "__main__":
    main()