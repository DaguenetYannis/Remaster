from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


INPUT_PATH = Path("data/abm/transitions_panel.parquet")
OUTPUT_DIR = Path("data/abm/model_outputs_regime")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_PATH)

    df = df.dropna(subset=["regime", "regime_next"])

    df["changed"] = (df["regime"] != df["regime_next"]).astype(int)

    features = [
        "emissions_intensity",
        "g_base",
        "g_out_network",
        "g_in_network",
        "pagerank",
        "out_strength",
        "in_strength",
        "green_capability_share",
        "capability_mean_pci",
    ]

    features = [f for f in features if f in df.columns]

    X = df[features].fillna(0)
    y = df["changed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    report = classification_report(y_test, preds, output_dict=True)
    report_df = pd.DataFrame(report).T
    report_df.to_csv(OUTPUT_DIR / "regime_change_classification.csv")

    importances = pd.DataFrame(
        {
            "feature": features,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importances.to_csv(OUTPUT_DIR / "regime_feature_importance.csv", index=False)

    print("\nClassification report:")
    print(report_df.head())

    print("\nTop features:")
    print(importances.head(10))


if __name__ == "__main__":
    main()