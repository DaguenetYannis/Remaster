from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class TransitionDiagnostics:
    scenario_name: str
    scenario_path: Path
    output_dir: Path

    def run(self) -> Dict[str, pd.DataFrame]:
        panel = self._load_panel()
        panel = self._prepare_panel(panel)
        transitions = self._build_transition_panel(panel)

        tables = {
            "regime_transition": self._build_regime_transition_table(transitions),
            "regime_change": self._build_regime_change_table(transitions),
            "vector_field": self._build_vector_field_table(transitions),
            "green_gap": self._build_green_gap_table(panel),
            "centroids": self._build_centroids(panel),
        }

        self._save_tables(tables)

        return tables

    def _load_panel(self) -> pd.DataFrame:
        return pd.read_parquet(self.scenario_path)

    def _prepare_panel(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["agent_id"] = df["agent_id"].astype(str)
        df["Country"] = df["Country"].astype(str)
        df["Sector"] = df["Sector"].astype(str)

        df = df.sort_values(["agent_id", "sim_year"])

        df["log_out_strength"] = np.log10(
            1 + df["out_strength"].fillna(0).clip(lower=0)
        )

        df["network_green_gap"] = df["g_out_network"] - df["g_base"]

        return df

    def _build_transition_panel(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        g = df.groupby("agent_id", sort=False)

        df["ei_next"] = g["emissions_intensity"].shift(-1)
        df["g_next"] = g["g_out_network"].shift(-1)
        df["log_out_next"] = g["log_out_strength"].shift(-1)
        df["regime_next"] = g["regime"].shift(-1)

        df["delta_ei"] = df["ei_next"] - df["emissions_intensity"]
        df["delta_g"] = df["g_next"] - df["g_out_network"]
        df["delta_log_out"] = df["log_out_next"] - df["log_out_strength"]

        df = df.dropna(subset=["ei_next", "g_next", "regime_next"])

        df["regime_transition"] = df["regime"] + " → " + df["regime_next"]

        return df

    def _build_regime_transition_table(self, df: pd.DataFrame) -> pd.DataFrame:
        table = (
            df.groupby(["regime", "regime_next"])
            .size()
            .reset_index(name="count")
        )

        table["share"] = table["count"] / table["count"].sum()
        return table

    def _build_regime_change_table(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["changed"] = df["regime"] != df["regime_next"]

        table = (
            df.groupby("sim_year")
            .agg(
                n=("agent_id", "count"),
                change=("changed", "sum"),
            )
            .reset_index()
        )

        table["share"] = table["change"] / table["n"]
        return table

    def _build_vector_field_table(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["x_bin"] = pd.qcut(df["log_out_strength"], q=10, duplicates="drop")
        df["y_bin"] = pd.qcut(df["g_out_network"], q=10, duplicates="drop")

        table = (
            df.groupby(["x_bin", "y_bin"])
            .agg(
                x=("log_out_strength", "mean"),
                y=("g_out_network", "mean"),
                dx=("delta_log_out", "mean"),
                dy=("delta_g", "mean"),
                count=("agent_id", "count"),
            )
            .reset_index()
        )

        table["x2"] = table["x"] + table["dx"]
        table["y2"] = table["y"] + table["dy"]

        return table

    def _build_green_gap_table(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.groupby("sim_year")["network_green_gap"]
            .describe(percentiles=[0.1, 0.5, 0.9])
            .reset_index()
        )

    def _build_centroids(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.groupby(["sim_year", "regime"])
            .agg(
                x=("log_out_strength", "mean"),
                y=("g_out_network", "mean"),
                ei=("emissions_intensity", "mean"),
                n=("agent_id", "count"),
            )
            .reset_index()
        )

    def _save_tables(self, tables: Dict[str, pd.DataFrame]) -> None:
        out = self.output_dir / "tables"
        out.mkdir(parents=True, exist_ok=True)

        for name, df in tables.items():
            df.to_csv(out / f"{self.scenario_name}_{name}.csv", index=False)