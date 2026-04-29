from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


DEFAULT_INPUT_PATH = Path("data/final/eora_atlas_merged.parquet")
DEFAULT_METRICS_DIR = Path("data/metrics")
DEFAULT_OUTPUT_PATH = Path("data/final/transition_dynamics.parquet")


@dataclass(frozen=True)
class TransitionConfig:
    input_path: Path = DEFAULT_INPUT_PATH
    metrics_dir: Path = DEFAULT_METRICS_DIR
    output_path: Path = DEFAULT_OUTPUT_PATH
    sector_proximity_path: Optional[Path] = None

    lambda_gc: float = 1.0
    lambda_ng: float = 1.0
    lambda_ce: float = 1.0

    country_col: str = "Country"
    sector_col: str = "Sector"
    year_col: str = "Year"

    ei_col: str = "emissions_intensity"
    local_green_col: str = "g_base"
    network_green_col: str = "g_in_network"
    green_capability_col: str = "green_capability_share"
    green_precedence_path: Optional[Path] = Path(
    "data/final/green_precedence/node_year_green_precedence.parquet"
)


class TransitionDynamicsBuilder:
    def __init__(self, config: TransitionConfig) -> None:
        self.config = config

    def build(self) -> pd.DataFrame:
        panel = self._load_panel()
        panel = self._prepare_panel(panel)
        panel = self._add_network_exposure(panel)
        panel = self._add_capability_ecosystem_exposure(panel)
        panel = self._add_green_capability_readiness(panel)
        transitions = self._build_transitions(panel)
        self._save(transitions)
        return transitions

    def _load_panel(self) -> pd.DataFrame:
        if not self.config.input_path.exists():
            raise FileNotFoundError(f"Missing input panel: {self.config.input_path}")

        logging.info("Loading merged Eora-Atlas panel: %s", self.config.input_path)
        return pd.read_parquet(self.config.input_path)

    def _prepare_panel(self, df: pd.DataFrame) -> pd.DataFrame:
        required = [
            self.config.country_col,
            self.config.sector_col,
            self.config.year_col,
            self.config.ei_col,
        ]
        self._check_columns(df, required)

        out = df.copy()

        out[self.config.country_col] = out[self.config.country_col].astype(str).str.strip()
        out[self.config.sector_col] = out[self.config.sector_col].astype(str).str.strip()
        out[self.config.year_col] = pd.to_numeric(out[self.config.year_col], errors="raise").astype(int)

        numeric_cols = [
            self.config.ei_col,
            self.config.local_green_col,
            self.config.network_green_col,
            self.config.green_capability_col,
            "out_strength",
            "in_strength",
            "pagerank",
            "g_out_network",
            "active_good_count",
            "capability_mean_pci",
            "capability_export_weighted_pci",
            "green_capability_export_share",
        ]

        for col in numeric_cols:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        if self.config.green_capability_col not in out.columns:
            logging.warning(
                "Missing %s. Filling local green capability component with 0.",
                self.config.green_capability_col,
            )
            out[self.config.green_capability_col] = 0.0

        if self.config.local_green_col not in out.columns:
            logging.warning(
                "Missing %s. Creating local green-ness from emissions intensity.",
                self.config.local_green_col,
            )
            out[self.config.local_green_col] = self._safe_log_greenness(out[self.config.ei_col])

        if self.config.network_green_col not in out.columns:
            logging.warning(
                "Missing %s. Using local green-ness as network green exposure fallback.",
                self.config.network_green_col,
            )
            out[self.config.network_green_col] = out[self.config.local_green_col]

        out["node_id"] = (
            out[self.config.country_col]
            + " | "
            + out[self.config.sector_col]
        )

        return out.sort_values(
            [self.config.country_col, self.config.sector_col, self.config.year_col]
        ).reset_index(drop=True)

    def _add_network_exposure(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["upstream_ei_exposure"] = np.nan
        out["downstream_ei_exposure"] = np.nan

        if not self.config.metrics_dir.exists():
            logging.warning(
                "Metrics directory not found. Using %s as network exposure.",
                self.config.network_green_col,
            )
            out["network_green_exposure"] = out[self.config.network_green_col]
            return out

        for year, year_df in out.groupby(self.config.year_col):
            et_path = self.config.metrics_dir / str(year) / f"et_{year}.parquet"

            if not et_path.exists():
                logging.warning("Missing ET matrix for %s: %s", year, et_path)
                continue

            logging.info("Computing network EI exposure for %s", year)

            et = pd.read_parquet(et_path)
            ei = year_df.set_index("node_id")[self.config.ei_col]

            common = et.index.intersection(ei.index)
            common_cols = et.columns.intersection(ei.index)

            et_aligned = et.loc[common, common_cols]
            ei_sources = ei.reindex(et_aligned.index).fillna(0)
            ei_targets = ei.reindex(et_aligned.columns).fillna(0)

            upstream = self._weighted_column_average(et_aligned, ei_sources)
            downstream = self._weighted_row_average(et_aligned, ei_targets)

            mask = out[self.config.year_col] == year

            out.loc[mask, "upstream_ei_exposure"] = (
                out.loc[mask, "node_id"].map(upstream).astype(float)
            )
            out.loc[mask, "downstream_ei_exposure"] = (
                out.loc[mask, "node_id"].map(downstream).astype(float)
            )

        out["network_green_exposure"] = out[self.config.network_green_col]
        return out

    def _add_capability_ecosystem_exposure(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.green_precedence_path is not None:
            if self.config.green_precedence_path.exists():
                logging.info(
                    "Using green precedence exposure as CE component: %s",
                    self.config.green_precedence_path,
                )
                return self._add_green_precedence_exposure(df)

        if self.config.sector_proximity_path is not None:
            return self._add_weighted_sector_ecosystem_exposure(df)

        logging.info(
            "No green precedence or sector proximity matrix supplied. "
            "Using sector-year mean green capability share as CE proxy."
        )

        out = df.copy()

        group_cols = [self.config.year_col, self.config.sector_col]
        sector_year_mean = (
            out.groupby(group_cols, dropna=False)[self.config.green_capability_col]
            .mean()
            .rename("capability_ecosystem_exposure")
            .reset_index()
        )

        out = out.merge(
            sector_year_mean,
            on=group_cols,
            how="left",
            validate="many_to_one",
        )

        out["capability_ecosystem_exposure"] = (
            out["capability_ecosystem_exposure"].fillna(0)
        )

        return out
    
    def _add_green_precedence_exposure(self, df: pd.DataFrame) -> pd.DataFrame:
        path = self.config.green_precedence_path

        if path is None or not path.exists():
            raise FileNotFoundError(f"Missing green precedence file: {path}")

        precedence = pd.read_parquet(path)

        required = [self.config.year_col, "node_id", "green_precedence_exposure"]
        self._check_columns(precedence, required)

        out = df.copy()

        out = out.merge(
            precedence[
                [self.config.year_col, "node_id", "green_precedence_exposure"]
            ],
            on=[self.config.year_col, "node_id"],
            how="left",
            validate="many_to_one",
        )

        out["capability_ecosystem_exposure"] = (
            out["green_precedence_exposure"].fillna(0)
        )

        return out

    def _add_weighted_sector_ecosystem_exposure(self, df: pd.DataFrame) -> pd.DataFrame:
        path = self.config.sector_proximity_path

        if path is None or not path.exists():
            raise FileNotFoundError(f"Missing sector proximity matrix: {path}")

        logging.info("Loading sector proximity matrix: %s", path)

        proximity = pd.read_csv(path)

        required = ["source_sector", "target_sector", "weight"]
        self._check_columns(proximity, required)

        proximity = proximity.copy()
        proximity["source_sector"] = proximity["source_sector"].astype(str).str.strip()
        proximity["target_sector"] = proximity["target_sector"].astype(str).str.strip()
        proximity["weight"] = pd.to_numeric(proximity["weight"], errors="coerce").fillna(0)

        out = df.copy()
        values = []

        for year, year_df in out.groupby(self.config.year_col):
            gc_by_sector = (
                year_df.groupby(self.config.sector_col)[self.config.green_capability_col]
                .mean()
            )

            temp = proximity.copy()
            temp["target_gc"] = temp["target_sector"].map(gc_by_sector).fillna(0)

            weighted = (
                temp.assign(weighted_gc=temp["weight"] * temp["target_gc"])
                .groupby("source_sector")
                .agg(weighted_gc=("weighted_gc", "sum"), weight=("weight", "sum"))
            )

            weighted["capability_ecosystem_exposure"] = (
                weighted["weighted_gc"] / weighted["weight"].replace(0, np.nan)
            ).fillna(0)

            year_map = weighted["capability_ecosystem_exposure"]

            year_out = year_df[["node_id", self.config.sector_col]].copy()
            year_out["capability_ecosystem_exposure"] = (
                year_out[self.config.sector_col].map(year_map).fillna(0)
            )
            values.append(year_out[["node_id", "capability_ecosystem_exposure"]])

        exposure = pd.concat(values, ignore_index=True)

        out = out.merge(
            exposure,
            on="node_id",
            how="left",
            validate="many_to_one",
        )

        out["capability_ecosystem_exposure"] = (
            out["capability_ecosystem_exposure"].fillna(0)
        )

        return out

    def _add_green_capability_readiness(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        out["gc_component"] = self._minmax(out[self.config.green_capability_col])
        out["ng_component"] = self._minmax(out["network_green_exposure"])
        out["ce_component"] = self._minmax(out["capability_ecosystem_exposure"])

        out["green_capability_readiness"] = (
            self.config.lambda_gc * out["gc_component"]
            + self.config.lambda_ng * out["ng_component"]
            + self.config.lambda_ce * out["ce_component"]
        )

        total_lambda = (
            self.config.lambda_gc
            + self.config.lambda_ng
            + self.config.lambda_ce
        )

        if total_lambda > 0:
            out["green_capability_readiness"] = (
                out["green_capability_readiness"] / total_lambda
            )

        return out

    def _build_transitions(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Building state-to-state transition dataset")

        key_cols = [self.config.country_col, self.config.sector_col]
        year_col = self.config.year_col

        state_cols = [
            self.config.ei_col,
            self.config.local_green_col,
            self.config.network_green_col,
            "g_out_network",
            "network_green_exposure",
            "upstream_ei_exposure",
            "downstream_ei_exposure",
            "capability_ecosystem_exposure",
            "green_capability_readiness",
            self.config.green_capability_col,
            "green_capability_export_share",
            "active_good_count",
            "capability_mean_pci",
            "capability_export_weighted_pci",
            "out_strength",
            "in_strength",
            "pagerank",
        ]

        state_cols = [col for col in state_cols if col in df.columns]

        base = df[key_cols + [year_col, "node_id"] + state_cols].copy()

        future = base.copy()
        future[year_col] = future[year_col] - 1

        rename_future = {
            col: f"{col}_next"
            for col in state_cols
        }

        future = future.rename(columns=rename_future)

        transitions = base.merge(
            future[key_cols + [year_col] + list(rename_future.values())],
            on=key_cols + [year_col],
            how="inner",
            validate="one_to_one",
        )

        transitions["next_year"] = transitions[year_col] + 1

        self._add_delta(transitions, self.config.ei_col, "delta_ei")
        self._add_delta(transitions, self.config.local_green_col, "delta_local_green")
        self._add_delta(transitions, self.config.network_green_col, "delta_network_green")

        if "green_capability_readiness_next" in transitions.columns:
            transitions["delta_green_capability_readiness"] = (
                transitions["green_capability_readiness_next"]
                - transitions["green_capability_readiness"]
            )

        if self.config.green_capability_col in transitions.columns:
            self._add_delta(
                transitions,
                self.config.green_capability_col,
                "delta_green_capability_share",
            )

        if "out_strength" in transitions.columns:
            self._add_delta(transitions, "out_strength", "delta_out_strength")

        if "pagerank" in transitions.columns:
            self._add_delta(transitions, "pagerank", "delta_pagerank")

        transitions["transition_equation_rhs_unestimated"] = (
            -transitions[self.config.ei_col].fillna(0)
            + transitions["upstream_ei_exposure"].fillna(0)
            - transitions["green_capability_readiness"].fillna(0)
        )

        logging.info("Transition rows: %s", len(transitions))
        logging.info(
            "Years: %s-%s",
            transitions[year_col].min(),
            transitions["next_year"].max(),
        )

        return transitions

    def _save(self, df: pd.DataFrame) -> None:
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.config.output_path, index=False)
        logging.info("Saved transition dynamics dataset: %s", self.config.output_path)

    @staticmethod
    def _weighted_column_average(matrix: pd.DataFrame, values: pd.Series) -> pd.Series:
        weighted = matrix.mul(values, axis=0).sum(axis=0)
        totals = matrix.sum(axis=0).replace(0, np.nan)
        return (weighted / totals).replace([np.inf, -np.inf], np.nan).fillna(0)

    @staticmethod
    def _weighted_row_average(matrix: pd.DataFrame, values: pd.Series) -> pd.Series:
        weighted = matrix.mul(values, axis=1).sum(axis=1)
        totals = matrix.sum(axis=1).replace(0, np.nan)
        return (weighted / totals).replace([np.inf, -np.inf], np.nan).fillna(0)

    @staticmethod
    def _safe_log_greenness(ei: pd.Series, epsilon: float = 1e-12) -> pd.Series:
        result = -np.log(pd.to_numeric(ei, errors="coerce").clip(lower=0) + epsilon)
        return result.replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0, upper=10)

    @staticmethod
    def _minmax(series: pd.Series) -> pd.Series:
        clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        min_value = clean.min(skipna=True)
        max_value = clean.max(skipna=True)

        if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
            return pd.Series(0.0, index=series.index)

        return ((clean - min_value) / (max_value - min_value)).fillna(0)

    @staticmethod
    def _add_delta(df: pd.DataFrame, col: str, output_col: str) -> None:
        next_col = f"{col}_next"
        if col in df.columns and next_col in df.columns:
            df[output_col] = df[next_col] - df[col]

    @staticmethod
    def _check_columns(df: pd.DataFrame, required: list[str]) -> None:
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build transition dynamics dataset for country-sector nodes."
    )

    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--metrics-dir", default=str(DEFAULT_METRICS_DIR))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--sector-proximity-path", default=None)
    parser.add_argument(
    "--green-precedence-path",
    default="data/final/green_precedence/node_year_green_precedence.parquet",
)

    parser.add_argument("--lambda-gc", type=float, default=1.0)
    parser.add_argument("--lambda-ng", type=float, default=1.0)
    parser.add_argument("--lambda-ce", type=float, default=1.0)

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    args = parse_args()

    config = TransitionConfig(
        input_path=Path(args.input_path),
        metrics_dir=Path(args.metrics_dir),
        output_path=Path(args.output_path),
        sector_proximity_path=(
            Path(args.sector_proximity_path)
            if args.sector_proximity_path is not None
            else None
        ),
        green_precedence_path=(
    Path(args.green_precedence_path)
    if args.green_precedence_path is not None
    else None
    ),
        lambda_gc=args.lambda_gc,
        lambda_ng=args.lambda_ng,
        lambda_ce=args.lambda_ce,
    )

    builder = TransitionDynamicsBuilder(config)
    builder.build()


if __name__ == "__main__":
    main()