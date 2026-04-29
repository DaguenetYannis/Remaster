from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TRANSITIONS_PATH = Path("data/final/transition_dynamics.parquet")
DEFAULT_METRICS_DIR = Path("data/metrics")
DEFAULT_OUTPUT_DIR = Path("data/final/green_precedence")


@dataclass(frozen=True)
class GreenPrecedenceConfig:
    transitions_path: Path = DEFAULT_TRANSITIONS_PATH
    metrics_dir: Path = DEFAULT_METRICS_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR

    year_col: str = "Year"
    node_col: str = "node_id"
    sector_col: str = "Sector"

    min_total_exposure: float = 1e-12
    green_event_mode: str = "combined"


class GreenPrecedenceBuilder:
    def __init__(self, config: GreenPrecedenceConfig) -> None:
        self.config = config

    def build(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        transitions = self._load_transitions()
        transitions = self._add_green_event(transitions)

        exposure = self._build_upstream_sector_exposure(transitions)
        sector_scores = self._build_sector_scores(exposure)
        node_year_scores = self._build_node_year_scores(exposure, sector_scores)

        self._save(sector_scores, node_year_scores)
        return sector_scores, node_year_scores

    def _load_transitions(self) -> pd.DataFrame:
        if not self.config.transitions_path.exists():
            raise FileNotFoundError(
                f"Missing transition dataset: {self.config.transitions_path}"
            )

        logging.info("Loading transitions: %s", self.config.transitions_path)
        df = pd.read_parquet(self.config.transitions_path)

        required = [self.config.year_col, self.config.node_col, self.config.sector_col]
        self._check_columns(df, required)

        return df

    def _add_green_event(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if self.config.green_event_mode == "ei":
            self._check_columns(out, ["delta_ei"])
            out["green_upgrade_event"] = out["delta_ei"] < 0

        elif self.config.green_event_mode == "network":
            self._check_columns(out, ["delta_network_green"])
            out["green_upgrade_event"] = out["delta_network_green"] > 0

        elif self.config.green_event_mode == "capability":
            self._check_columns(out, ["delta_green_capability_share"])
            out["green_upgrade_event"] = out["delta_green_capability_share"] > 0

        elif self.config.green_event_mode == "combined":
            conditions = []

            if "delta_ei" in out.columns:
                conditions.append(out["delta_ei"] < 0)

            if "delta_network_green" in out.columns:
                conditions.append(out["delta_network_green"] > 0)

            if "delta_green_capability_share" in out.columns:
                conditions.append(out["delta_green_capability_share"] > 0)

            if not conditions:
                raise ValueError(
                    "Combined green event requires at least one delta variable."
                )

            event = conditions[0]
            for condition in conditions[1:]:
                event = event | condition

            out["green_upgrade_event"] = event

        else:
            raise ValueError(
                "green_event_mode must be one of: ei, network, capability, combined"
            )

        out["green_upgrade_event"] = out["green_upgrade_event"].astype(int)
        logging.info("Baseline green upgrade rate: %.4f", out["green_upgrade_event"].mean())

        return out

    def _build_upstream_sector_exposure(self, transitions: pd.DataFrame) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []

        transition_keys = transitions[
            [
                self.config.year_col,
                self.config.node_col,
                "green_upgrade_event",
            ]
        ].copy()

        years = sorted(transition_keys[self.config.year_col].unique())

        for year in years:
            et_path = self.config.metrics_dir / str(year) / f"et_{year}.parquet"

            if not et_path.exists():
                logging.warning("Missing ET matrix for year %s: %s", year, et_path)
                continue

            logging.info("Building upstream sector exposure for %s", year)

            et = pd.read_parquet(et_path)
            year_events = transition_keys[
                transition_keys[self.config.year_col] == year
            ].copy()

            target_nodes = year_events[self.config.node_col].unique()
            common_cols = et.columns.intersection(target_nodes)

            if len(common_cols) == 0:
                logging.warning("No common ET columns and transition nodes for %s", year)
                continue

            et = et.loc[:, common_cols]

            source_sectors = pd.Series(
                et.index.map(self._extract_sector_from_node),
                index=et.index,
                name="source_sector",
            )

            column_totals = et.sum(axis=0).replace(0, np.nan)

            sector_exposures = []
            for sector, idx in source_sectors.groupby(source_sectors).groups.items():
                exposure = et.loc[idx].sum(axis=0) / column_totals
                exposure = exposure.replace([np.inf, -np.inf], np.nan).fillna(0)

                temp = exposure.rename("upstream_sector_exposure").reset_index()
                temp.columns = [self.config.node_col, "upstream_sector_exposure"]
                temp["source_sector"] = sector
                temp[self.config.year_col] = year

                sector_exposures.append(temp)

            year_exposure = pd.concat(sector_exposures, ignore_index=True)
            year_exposure = year_exposure.merge(
                year_events,
                on=[self.config.year_col, self.config.node_col],
                how="inner",
                validate="many_to_one",
            )

            rows.append(year_exposure)

        if not rows:
            raise RuntimeError("No upstream sector exposure could be constructed.")

        exposure = pd.concat(rows, ignore_index=True)
        exposure = exposure[
            exposure["upstream_sector_exposure"] > self.config.min_total_exposure
        ].reset_index(drop=True)

        logging.info("Exposure rows: %s", len(exposure))
        return exposure

    def _build_sector_scores(self, exposure: pd.DataFrame) -> pd.DataFrame:
        baseline = exposure["green_upgrade_event"].mean()

        grouped = (
            exposure.assign(
                weighted_event=lambda x: (
                    x["upstream_sector_exposure"] * x["green_upgrade_event"]
                )
            )
            .groupby("source_sector", as_index=False)
            .agg(
                total_exposure=("upstream_sector_exposure", "sum"),
                weighted_events=("weighted_event", "sum"),
                observation_count=("green_upgrade_event", "size"),
            )
        )

        grouped["conditional_green_rate"] = (
            grouped["weighted_events"]
            / grouped["total_exposure"].replace(0, np.nan)
        ).fillna(0)

        grouped["baseline_green_rate"] = baseline
        grouped["green_precedence_score"] = (
            grouped["conditional_green_rate"] - grouped["baseline_green_rate"]
        )

        grouped["green_precedence_score_normalized"] = self._minmax_signed(
            grouped["green_precedence_score"]
        )

        return grouped.sort_values(
            "green_precedence_score", ascending=False
        ).reset_index(drop=True)

    def _build_node_year_scores(
        self,
        exposure: pd.DataFrame,
        sector_scores: pd.DataFrame,
    ) -> pd.DataFrame:
        scores = sector_scores[
            ["source_sector", "green_precedence_score_normalized"]
        ].copy()

        out = exposure.merge(
            scores,
            on="source_sector",
            how="left",
            validate="many_to_one",
        )

        out["weighted_green_precedence"] = (
            out["upstream_sector_exposure"]
            * out["green_precedence_score_normalized"].fillna(0)
        )

        node_year = (
            out.groupby([self.config.year_col, self.config.node_col], as_index=False)
            .agg(
                green_precedence_exposure=("weighted_green_precedence", "sum"),
                total_upstream_sector_exposure=("upstream_sector_exposure", "sum"),
            )
        )

        node_year["green_precedence_exposure"] = (
            node_year["green_precedence_exposure"]
            / node_year["total_upstream_sector_exposure"].replace(0, np.nan)
        ).fillna(0)

        return node_year

    def _save(self, sector_scores: pd.DataFrame, node_year_scores: pd.DataFrame) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        sector_path = self.config.output_dir / "sector_green_precedence_scores.parquet"
        node_year_path = self.config.output_dir / "node_year_green_precedence.parquet"

        sector_scores.to_parquet(sector_path, index=False)
        node_year_scores.to_parquet(node_year_path, index=False)

        sector_scores.to_csv(
            self.config.output_dir / "sector_green_precedence_scores.csv",
            index=False,
        )
        node_year_scores.to_csv(
            self.config.output_dir / "node_year_green_precedence.csv",
            index=False,
        )

        logging.info("Saved sector scores: %s", sector_path)
        logging.info("Saved node-year scores: %s", node_year_path)

    @staticmethod
    def _extract_sector_from_node(node: object) -> str:
        text = str(node)
        if " | " in text:
            return text.split(" | ")[-1].strip()
        return text.strip()

    @staticmethod
    def _minmax_signed(series: pd.Series) -> pd.Series:
        clean = pd.to_numeric(series, errors="coerce").fillna(0)

        max_abs = clean.abs().max()
        if max_abs == 0 or pd.isna(max_abs):
            return pd.Series(0.0, index=series.index)

        return clean / max_abs

    @staticmethod
    def _check_columns(df: pd.DataFrame, required: list[str]) -> None:
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate sector-level green precedence from transition dynamics."
    )

    parser.add_argument("--transitions-path", default=str(DEFAULT_TRANSITIONS_PATH))
    parser.add_argument("--metrics-dir", default=str(DEFAULT_METRICS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--green-event-mode",
        default="combined",
        choices=["ei", "network", "capability", "combined"],
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    args = parse_args()

    config = GreenPrecedenceConfig(
        transitions_path=Path(args.transitions_path),
        metrics_dir=Path(args.metrics_dir),
        output_dir=Path(args.output_dir),
        green_event_mode=args.green_event_mode,
    )

    builder = GreenPrecedenceBuilder(config)
    builder.build()


if __name__ == "__main__":
    main()