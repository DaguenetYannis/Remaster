from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT_PATH = Path("data/final/transition_dynamics.parquet")
DEFAULT_OUTPUT_DIR = Path("data/final/estimates")


@dataclass(frozen=True)
class EstimateConfig:
    input_path: Path = DEFAULT_INPUT_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    year_col: str = "Year"
    country_col: str = "Country"
    sector_col: str = "Sector"


class TransitionEstimator:
    def __init__(self, config: EstimateConfig) -> None:
        self.config = config

    def run(self) -> None:
        df = self._load()
        df = self._prepare(df)

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        self._save_binned_effects(df)
        self._save_quantile_conditioned_effects(df)
        self._save_simple_correlations(df)
        self._save_behaviour_summary(df)

    def _load(self) -> pd.DataFrame:
        if not self.config.input_path.exists():
            raise FileNotFoundError(f"Missing input file: {self.config.input_path}")

        logging.info("Loading transition dynamics: %s", self.config.input_path)
        return pd.read_parquet(self.config.input_path)

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        numeric_cols = [
            "emissions_intensity",
            "delta_ei",
            "delta_network_green",
            "delta_green_capability_share",
            "green_capability_readiness",
            "capability_ecosystem_exposure",
            "network_green_exposure",
            "upstream_ei_exposure",
            "out_strength",
            "pagerank",
        ]

        for col in numeric_cols:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        if "delta_ei" not in out.columns:
            raise ValueError("Missing required column: delta_ei")

        out["ei_reduction"] = -out["delta_ei"]

        if "emissions_intensity" in out.columns:
            out["log_ei"] = np.log1p(out["emissions_intensity"].clip(lower=0))

        if "upstream_ei_exposure" in out.columns:
            out["log_upstream_ei_exposure"] = np.log1p(
                out["upstream_ei_exposure"].clip(lower=0)
            )

        out["strict_green_upgrade"] = (
            (out["delta_ei"] < out["delta_ei"].quantile(0.25))
        ).astype(int)

        return out.replace([np.inf, -np.inf], np.nan)

    def _save_binned_effects(self, df: pd.DataFrame) -> None:
        specs = [
            ("green_capability_readiness", "ei_reduction"),
            ("capability_ecosystem_exposure", "ei_reduction"),
            ("network_green_exposure", "ei_reduction"),
            ("upstream_ei_exposure", "ei_reduction"),
            ("emissions_intensity", "ei_reduction"),
        ]

        outputs = []

        for x_col, y_col in specs:
            if x_col not in df.columns or y_col not in df.columns:
                logging.warning("Skipping binned estimate: %s -> %s", x_col, y_col)
                continue

            estimate = self._binned_effect(df, x_col=x_col, y_col=y_col, bins=20)
            estimate["x_variable"] = x_col
            estimate["y_variable"] = y_col
            outputs.append(estimate)

        if outputs:
            result = pd.concat(outputs, ignore_index=True)
            self._save_table(result, "binned_effects")

    def _save_quantile_conditioned_effects(self, df: pd.DataFrame) -> None:
        if "emissions_intensity" not in df.columns:
            logging.warning("Skipping quantile-conditioned estimates: missing EI")
            return

        temp = df.copy()
        temp["initial_ei_quantile"] = pd.qcut(
            temp["emissions_intensity"],
            q=4,
            labels=["Q1_low_EI", "Q2", "Q3", "Q4_high_EI"],
            duplicates="drop",
        )

        specs = [
            ("green_capability_readiness", "ei_reduction"),
            ("capability_ecosystem_exposure", "ei_reduction"),
            ("network_green_exposure", "ei_reduction"),
        ]

        outputs = []

        for quantile, group in temp.groupby("initial_ei_quantile", observed=True):
            for x_col, y_col in specs:
                if x_col not in group.columns or y_col not in group.columns:
                    continue

                estimate = self._binned_effect(group, x_col=x_col, y_col=y_col, bins=10)
                estimate["initial_ei_quantile"] = str(quantile)
                estimate["x_variable"] = x_col
                estimate["y_variable"] = y_col
                outputs.append(estimate)

        if outputs:
            result = pd.concat(outputs, ignore_index=True)
            self._save_table(result, "conditioned_binned_effects")

    def _save_simple_correlations(self, df: pd.DataFrame) -> None:
        variables = [
            "ei_reduction",
            "emissions_intensity",
            "upstream_ei_exposure",
            "green_capability_readiness",
            "capability_ecosystem_exposure",
            "network_green_exposure",
            "green_capability_share",
            "out_strength",
            "pagerank",
        ]

        variables = [col for col in variables if col in df.columns]
        corr = df[variables].corr(method="spearman").reset_index()
        corr = corr.rename(columns={"index": "variable"})

        self._save_table(corr, "spearman_correlations")

    def _save_behaviour_summary(self, df: pd.DataFrame) -> None:
        required = ["delta_network_green", "delta_ei"]
        if any(col not in df.columns for col in required):
            logging.warning("Skipping behaviour summary")
            return

        temp = df.copy()

        temp["behaviour_type"] = np.select(
            [
                (temp["delta_network_green"] > 0) & (temp["delta_ei"] < 0),
                (temp["delta_network_green"] > 0) & (temp["delta_ei"] >= 0),
                (temp["delta_network_green"] <= 0) & (temp["delta_ei"] < 0),
                (temp["delta_network_green"] <= 0) & (temp["delta_ei"] >= 0),
            ],
            [
                "full_green_upgrade",
                "network_upgrade_only",
                "local_EI_upgrade_only",
                "brown_or_stagnant",
            ],
            default="unknown",
        )

        summary = (
            temp.groupby("behaviour_type", as_index=False)
            .agg(
                n=("behaviour_type", "size"),
                mean_ei_reduction=("ei_reduction", "mean"),
                median_ei_reduction=("ei_reduction", "median"),
                mean_green_capability_readiness=("green_capability_readiness", "mean"),
                mean_capability_ecosystem_exposure=(
                    "capability_ecosystem_exposure",
                    "mean",
                ),
                mean_network_green_exposure=("network_green_exposure", "mean"),
            )
        )

        summary["share"] = summary["n"] / summary["n"].sum()

        self._save_table(summary, "behaviour_summary")

    def _binned_effect(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        bins: int,
    ) -> pd.DataFrame:
        temp = df[[x_col, y_col]].dropna().copy()

        if temp.empty:
            return pd.DataFrame()

        temp["bin"] = pd.qcut(temp[x_col], q=bins, duplicates="drop")

        result = (
            temp.groupby("bin", observed=True)
            .agg(
                x_mean=(x_col, "mean"),
                x_min=(x_col, "min"),
                x_max=(x_col, "max"),
                y_mean=(y_col, "mean"),
                y_median=(y_col, "median"),
                y_std=(y_col, "std"),
                n=(y_col, "size"),
            )
            .reset_index(drop=True)
        )

        result["y_se"] = result["y_std"] / np.sqrt(result["n"])
        result["y_ci_low"] = result["y_mean"] - 1.96 * result["y_se"]
        result["y_ci_high"] = result["y_mean"] + 1.96 * result["y_se"]

        return result

    def _save_table(self, df: pd.DataFrame, name: str) -> None:
        parquet_path = self.config.output_dir / f"{name}.parquet"
        csv_path = self.config.output_dir / f"{name}.csv"

        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)

        logging.info("Saved %s", parquet_path)
        logging.info("Saved %s", csv_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate empirical transition relationships."
    )
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    args = parse_args()

    config = EstimateConfig(
        input_path=Path(args.input_path),
        output_dir=Path(args.output_dir),
    )

    estimator = TransitionEstimator(config)
    estimator.run()


if __name__ == "__main__":
    main()