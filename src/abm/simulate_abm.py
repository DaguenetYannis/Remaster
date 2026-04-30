from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge


DATA_DIR = Path("data/abm")
MODEL_DIR = Path("data/abm/model_outputs_clean")


def load_data():
    agents = pd.read_parquet(DATA_DIR / "agents_panel.parquet")
    transitions = pd.read_parquet(DATA_DIR / "diagnostics/transitions_with_clean_targets.parquet")
    return agents, transitions


def train_models(transitions):
    features = [
        "emissions_intensity",
        "g_base",
        "g_out_network",
        "g_in_network",
        "pagerank",
        "out_strength",
        "in_strength",
    ]

    features = [f for f in features if f in transitions.columns]

    X = transitions[features].fillna(0)

    # Continuous model
    y_reg = transitions["delta_log_emissions_intensity_winsorized"]
    reg_model = Ridge(alpha=1.0)
    reg_model.fit(X, y_reg)

    # Transition model
    y_clf = (transitions["regime"] != transitions["regime_next"]).astype(int)

    clf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    clf_model.fit(X, y_clf)

    return reg_model, clf_model, features


def simulate_step(df, reg_model, clf_model, features):
    X = df[features].fillna(0)

    # Continuous update
    delta = reg_model.predict(X)
    df["log_emissions_intensity"] = np.log1p(df["emissions_intensity"])
    df["log_emissions_intensity"] += delta
    df["emissions_intensity"] = np.expm1(df["log_emissions_intensity"])

    # Regime transition probability
    proba = clf_model.predict_proba(X)[:, 1]

    random_draw = np.random.rand(len(df))
    change = random_draw < proba

    # Simple regime flip logic (placeholder)
    df.loc[change, "regime"] = "transitioned"

    return df


def run_simulation():
    agents, transitions = load_data()

    # Use last observed year as starting point
    last_year = agents["Year"].max()
    state = agents[agents["Year"] == last_year].copy()

    reg_model, clf_model, features = train_models(transitions)

    simulated_states = []

    for t in range(10):
        state = simulate_step(state, reg_model, clf_model, features)
        state["sim_year"] = last_year + t + 1
        simulated_states.append(state.copy())

    result = pd.concat(simulated_states, ignore_index=True)
    result.to_parquet(DATA_DIR / "simulation_output.parquet", index=False)

    print("\nSimulation complete")
    print(result.head())


if __name__ == "__main__":
    run_simulation()