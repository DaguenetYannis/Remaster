from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class TransitionModelConfig:
    transitions_path: Path = Path("data/abm/transitions_panel.parquet")
    output_dir: Path = Path("data/abm/model_outputs")
    target_col: str = "delta_emissions_intensity"
    test_size: float = 0.25
    random_state: int = 42


class TransitionModelEstimator:
    def __init__(self, config: TransitionModelConfig) -> None:
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        df = self.load_data()
        model_df = self.prepare_model_frame(df)

        x = model_df[self.feature_cols]
        y = model_df[self.config.target_col]

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )

        ridge = self.fit_ridge(x_train, y_train)
        forest = self.fit_random_forest(x_train, y_train)

        ridge_report = self.evaluate_model("ridge", ridge, x_test, y_test)
        forest_report = self.evaluate_model("random_forest", forest, x_test, y_test)

        reports = pd.DataFrame([ridge_report, forest_report])
        reports.to_csv(self.output_dir / "transition_model_scores.csv", index=False)

        self.save_ridge_coefficients(ridge)
        self.save_forest_importances(forest)

        logging.info("Model scores:")
        logging.info("\n%s", reports)

    def load_data(self) -> pd.DataFrame:
        if not self.config.transitions_path.exists():
            raise FileNotFoundError(f"Missing file: {self.config.transitions_path}")

        df = pd.read_parquet(self.config.transitions_path)
        logging.info("Loaded transitions: %s rows, %s columns", *df.shape)

        return df

    def prepare_model_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        candidate_features = [
            "emissions_intensity",
            "g_base",
            "g_out_network",
            "g_in_network",
            "pagerank",
            "out_strength",
            "in_strength",
            "green_capability_share",
            "green_capability_export_share",
            "capability_mean_pci",
            "capability_export_weighted_pci",
        ]

        self.feature_cols = [col for col in candidate_features if col in df.columns]

        missing_features = sorted(set(candidate_features) - set(self.feature_cols))
        if missing_features:
            logging.warning("Missing candidate features: %s", missing_features)

        model_df = df[self.feature_cols + [self.config.target_col]].copy()

        for col in model_df.columns:
            model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

        model_df = model_df.replace([np.inf, -np.inf], np.nan)

        before = len(model_df)
        model_df = model_df.dropna(subset=[self.config.target_col])
        after = len(model_df)

        logging.info("Dropped rows with missing target: %s", before - after)
        logging.info("Remaining model rows: %s", after)

        missing_report = (
            model_df[self.feature_cols]
            .isna()
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        missing_report.columns = ["feature", "missing_share"]
        missing_report.to_csv(self.output_dir / "feature_missingness.csv", index=False)

        logging.info("Feature missingness saved")

        return model_df

    def fit_ridge(self, x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )

        model.fit(x_train, y_train)
        return model

    def fit_random_forest(self, x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "forest",
                    RandomForestRegressor(
                        n_estimators=200,
                        max_depth=12,
                        min_samples_leaf=20,
                        random_state=self.config.random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        model.fit(x_train, y_train)
        return model

    def evaluate_model(
        self,
        model_name: str,
        model: Pipeline,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, float | str]:
        predictions = model.predict(x_test)

        return {
            "model": model_name,
            "target": self.config.target_col,
            "rows_tested": len(y_test),
            "mae": mean_absolute_error(y_test, predictions),
            "r2": r2_score(y_test, predictions),
            "target_mean": float(y_test.mean()),
            "target_std": float(y_test.std()),
        }

    def save_ridge_coefficients(self, ridge_model: Pipeline) -> None:
        ridge = ridge_model.named_steps["ridge"]

        coefs = pd.DataFrame(
            {
                "feature": self.feature_cols,
                "coefficient": ridge.coef_,
            }
        ).sort_values("coefficient", ascending=False)

        coefs.to_csv(self.output_dir / "ridge_coefficients.csv", index=False)

    def save_forest_importances(self, forest_model: Pipeline) -> None:
        forest = forest_model.named_steps["forest"]

        importances = pd.DataFrame(
            {
                "feature": self.feature_cols,
                "importance": forest.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        importances.to_csv(self.output_dir / "random_forest_importances.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate empirical transition models for ABM dynamics."
    )

    parser.add_argument(
        "--transitions-path",
        default="data/abm/transitions_panel.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="data/abm/model_outputs",
    )
    parser.add_argument(
        "--target-col",
        default="delta_emissions_intensity",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    args = parse_args()

    config = TransitionModelConfig(
        transitions_path=Path(args.transitions_path),
        output_dir=Path(args.output_dir),
        target_col=args.target_col,
    )

    estimator = TransitionModelEstimator(config)
    estimator.run()


if __name__ == "__main__":
    main()