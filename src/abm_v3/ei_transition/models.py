from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TARGET_COLUMN = "ei_reduction_next"
MODEL_FEATURES = {
    "economic_only": ["log_EI"],
    "green_transition": ["log_EI", "green_capability", "g_network", "general_complexity"],
    "network_robustness": ["log_EI", "green_capability", "g_in", "g_out", "general_complexity"],
}
EXPECTED_SIGNS = {
    "log_EI": "positive",
    "green_capability": "positive",
    "g_network": "positive",
    "g_in": "positive",
    "g_out": "positive",
    "general_complexity": "positive_or_moderating",
}


@dataclass(frozen=True)
class EITransitionModelSpec:
    """Explicit specification for one EI transition model."""

    model_name: str
    numeric_features: list[str]
    fixed_effect_columns: list[str]
    alpha: float = 1.0


@dataclass
class EITransitionFitResult:
    """Fitted Ridge model and validation outputs."""

    spec: EITransitionModelSpec
    intercept: float
    coefficients: pd.DataFrame
    numeric_means: pd.Series
    numeric_scales: pd.Series
    dummy_columns: list[str]
    train_years: tuple[int, int]
    validation_years: tuple[int, int]

    def predict(self, panel: pd.DataFrame) -> pd.Series:
        """Predict next-period EI reduction for rows with required columns."""
        design = build_design_matrix(
            panel,
            self.spec.numeric_features,
            self.spec.fixed_effect_columns,
            self.numeric_means,
            self.numeric_scales,
            self.dummy_columns,
        )
        beta = self.coefficients.set_index("term")["coefficient"].reindex(design.columns).fillna(0.0)
        values = self.intercept + design.to_numpy(dtype=float) @ beta.to_numpy(dtype=float)
        return pd.Series(values, index=panel.index, name="predicted_ei_reduction_next")


class EITransitionModelSuite:
    """Fit and validate the three historical EI transition models."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit_all(
        self,
        panel: pd.DataFrame,
        train_end_year: int = 2012,
        validation_start_year: int = 2013,
        validation_end_year: int = 2015,
    ) -> dict[str, object]:
        """Fit all model variants and return scores, coefficients, signs, and predictions."""
        included = panel.loc[panel["sample_included"]].copy()
        fit_results = []
        score_rows = []
        coefficient_frames = []
        prediction_frames = []
        for spec in self.model_specs():
            fit_result = self.fit_one(included, spec, train_end_year, validation_start_year, validation_end_year)
            fit_results.append(fit_result)
            validation_rows = included.loc[
                (included["Year"] >= validation_start_year)
                & (included["Year"] <= validation_end_year)
            ].copy()
            predictions = build_prediction_frame(validation_rows, fit_result)
            prediction_frames.append(predictions)
            score_rows.append(
                score_model(
                    spec.model_name,
                    predictions,
                    n_train=int((included["Year"] <= train_end_year).sum()),
                    validation_mode=f"holdout_{validation_start_year}_{validation_end_year}",
                )
            )
            coefficient_frames.append(fit_result.coefficients.copy())
        coefficients = pd.concat(coefficient_frames, ignore_index=True)
        expected_signs = build_expected_sign_table(coefficients)
        predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
        scores = pd.DataFrame(score_rows)
        return {
            "fit_results": fit_results,
            "scores": scores,
            "coefficients": coefficients,
            "expected_signs": expected_signs,
            "predictions": predictions,
        }

    def fit_one(
        self,
        panel: pd.DataFrame,
        spec: EITransitionModelSpec,
        train_end_year: int,
        validation_start_year: int,
        validation_end_year: int,
    ) -> EITransitionFitResult:
        """Fit one standardized Ridge model with one-hot fixed effects."""
        required = [TARGET_COLUMN, *spec.numeric_features, *spec.fixed_effect_columns]
        missing = [column for column in required if column not in panel.columns]
        if missing:
            raise ValueError(f"Cannot fit {spec.model_name}; missing columns: {missing}")
        train = panel.loc[panel["Year"] <= train_end_year, required].dropna().copy()
        validation = panel.loc[
            (panel["Year"] >= validation_start_year)
            & (panel["Year"] <= validation_end_year),
            required,
        ].dropna()
        if train.empty:
            raise ValueError(f"Cannot fit {spec.model_name}; no training rows available.")
        if validation.empty:
            raise ValueError(f"Cannot validate {spec.model_name}; no validation rows available.")
        means = train[spec.numeric_features].mean()
        scales = train[spec.numeric_features].std(ddof=0).replace(0.0, 1.0).fillna(1.0)
        train_design = build_design_matrix(
            train,
            spec.numeric_features,
            spec.fixed_effect_columns,
            means,
            scales,
            dummy_columns=None,
        )
        y = train[TARGET_COLUMN].to_numpy(dtype=float)
        intercept, beta = fit_ridge(train_design.to_numpy(dtype=float), y, spec.alpha)
        coefficient_frame = pd.DataFrame(
            {
                "model_name": spec.model_name,
                "term": train_design.columns,
                "coefficient": beta,
            }
        )
        coefficient_frame["expected_sign"] = coefficient_frame["term"].map(EXPECTED_SIGNS).fillna("not_applicable")
        coefficient_frame["actual_sign"] = coefficient_frame["coefficient"].map(actual_sign)
        coefficient_frame["matches_expected_sign"] = coefficient_frame.apply(matches_expected_sign, axis=1)
        coefficient_frame["notes"] = np.where(
            coefficient_frame["term"].isin(spec.numeric_features),
            "standardized_numeric_coefficient; Ridge alpha=1.0; sector/year fixed effects included",
            "fixed_effect_coefficient; Ridge alpha=1.0",
        )
        return EITransitionFitResult(
            spec=spec,
            intercept=intercept,
            coefficients=coefficient_frame,
            numeric_means=means,
            numeric_scales=scales,
            dummy_columns=[column for column in train_design.columns if column not in spec.numeric_features],
            train_years=(int(panel["Year"].min()), train_end_year),
            validation_years=(validation_start_year, validation_end_year),
        )

    def model_specs(self) -> list[EITransitionModelSpec]:
        return [
            EITransitionModelSpec("economic_only", MODEL_FEATURES["economic_only"], ["Sector", "Year"], self.alpha),
            EITransitionModelSpec("green_transition", MODEL_FEATURES["green_transition"], ["Sector", "Year"], self.alpha),
            EITransitionModelSpec("network_robustness", MODEL_FEATURES["network_robustness"], ["Sector", "Year"], self.alpha),
        ]


def build_design_matrix(
    panel: pd.DataFrame,
    numeric_features: list[str],
    fixed_effect_columns: list[str],
    numeric_means: pd.Series,
    numeric_scales: pd.Series,
    dummy_columns: list[str] | None,
) -> pd.DataFrame:
    """Build standardized numeric features and aligned one-hot fixed effects."""
    numeric = (panel[numeric_features].astype(float) - numeric_means) / numeric_scales
    dummies = []
    for column in fixed_effect_columns:
        dummies.append(pd.get_dummies(panel[column].astype(str), prefix=column, dtype=float))
    fixed_effects = pd.concat(dummies, axis=1) if dummies else pd.DataFrame(index=panel.index)
    if dummy_columns is not None:
        fixed_effects = fixed_effects.reindex(columns=dummy_columns, fill_value=0.0)
    return pd.concat([numeric.reset_index(drop=True), fixed_effects.reset_index(drop=True)], axis=1)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[float, np.ndarray]:
    """Fit Ridge with an unpenalized intercept using normal equations."""
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return float(beta[0]), beta[1:]


def build_prediction_frame(validation_rows: pd.DataFrame, fit_result: EITransitionFitResult) -> pd.DataFrame:
    """Build validation predictions and EI_next implied by predicted reduction."""
    predictions = validation_rows[
        ["country_sector", "Year", "EI", "EI_next", "ei_reduction_next"]
    ].copy()
    predictions["model_name"] = fit_result.spec.model_name
    predictions["predicted_ei_reduction_next"] = fit_result.predict(validation_rows).to_numpy(dtype=float)
    predictions["predicted_EI_next"] = predictions["EI"].to_numpy(dtype=float) * np.exp(
        -predictions["predicted_ei_reduction_next"].to_numpy(dtype=float)
    )
    predictions["prediction_error"] = (
        predictions["predicted_ei_reduction_next"] - predictions["ei_reduction_next"]
    )
    predictions["absolute_error"] = predictions["prediction_error"].abs()
    predictions["squared_error"] = predictions["prediction_error"] ** 2
    predictions["invalid_predicted_EI_next"] = ~np.isfinite(predictions["predicted_EI_next"].to_numpy(dtype=float))
    return predictions


def score_model(model_name: str, predictions: pd.DataFrame, n_train: int, validation_mode: str) -> dict[str, object]:
    """Score one model on the validation rows."""
    observed = predictions["ei_reduction_next"].to_numpy(dtype=float)
    predicted = predictions["predicted_ei_reduction_next"].to_numpy(dtype=float)
    error = predicted - observed
    sse = float(np.nansum(error**2))
    centered = observed - float(np.nanmean(observed))
    sst = float(np.nansum(centered**2))
    correlation = safe_correlation(predicted, observed)
    return {
        "model_name": model_name,
        "n_train": n_train,
        "n_validation": len(predictions),
        "validation_mode": validation_mode,
        "target": TARGET_COLUMN,
        "rmse": float(np.sqrt(np.nanmean(error**2))),
        "mae": float(np.nanmean(np.abs(error))),
        "correlation_predicted_observed": correlation,
        "r2": np.nan if sst <= 0.0 else float(1.0 - sse / sst),
        "mean_observed_reduction": float(np.nanmean(observed)),
        "mean_predicted_reduction": float(np.nanmean(predicted)),
        "notes": (
            f"Validation mode={validation_mode}; Ridge alpha=1.0 with standardized numeric predictors."
        ),
    }


def build_expected_sign_table(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Build expected-sign diagnostics for theoretical numeric terms."""
    rows = []
    for _, row in coefficients.iterrows():
        term = str(row["term"])
        if term not in EXPECTED_SIGNS:
            continue
        coefficient = float(row["coefficient"])
        expected = EXPECTED_SIGNS[term]
        actual = actual_sign(coefficient)
        matches = sign_matches(expected, actual)
        rows.append(
            {
                "model_name": row["model_name"],
                "term": term,
                "expected_sign": expected,
                "actual_sign": actual,
                "coefficient": coefficient,
                "matches_expected_sign": matches,
                "interpretation": interpretation_for_sign(term, expected, actual, matches),
            }
        )
    return pd.DataFrame(rows)


def matches_expected_sign(row: pd.Series) -> bool | None:
    expected = str(row["expected_sign"])
    if expected == "not_applicable":
        return None
    return sign_matches(expected, str(row["actual_sign"]))


def sign_matches(expected: str, actual: str) -> bool:
    if expected in {"positive", "positive_or_moderating"}:
        return actual == "positive"
    if expected == "negative":
        return actual == "negative"
    return False


def actual_sign(value: float) -> str:
    if not np.isfinite(value) or np.isclose(value, 0.0):
        return "zero"
    return "positive" if value > 0.0 else "negative"


def interpretation_for_sign(term: str, expected: str, actual: str, matches: bool) -> str:
    if matches:
        return f"{term} matches expected {expected} relationship with EI reduction."
    return f"{term} has {actual} coefficient; expected {expected}."


def safe_correlation(left_values: np.ndarray, right_values: np.ndarray) -> float:
    valid = np.isfinite(left_values) & np.isfinite(right_values)
    if int(valid.sum()) < 2:
        return np.nan
    left = left_values[valid]
    right = right_values[valid]
    if np.isclose(float(np.std(left)), 0.0) or np.isclose(float(np.std(right)), 0.0):
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])
