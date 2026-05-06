from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.abm_v2.dynamics import (
    compute_mean_emissions_intensity,
    compute_mean_greeness,
    compute_total_emissions,
    update_capability_readiness,
    update_emissions_intensity,
    update_inventory,
    update_local_greeness,
    update_production,
)
from src.abm_v2.scenarii import Scenario


@dataclass
class ABMState:
    step: int
    scenario: str
    country_sector: np.ndarray
    country: np.ndarray
    sector: np.ndarray
    X: np.ndarray
    D: np.ndarray
    M: np.ndarray
    K: np.ndarray
    I: np.ndarray
    EI: np.ndarray
    g_local: np.ndarray
    g_in: np.ndarray
    g_out: np.ndarray
    NG: np.ndarray
    capability_readiness: np.ndarray
    out_strength: np.ndarray
    in_strength: np.ndarray
    ET: np.ndarray


class GreenTransitionABM:
    def __init__(
        self,
        metrics_panel: pd.DataFrame,
        scenario: Scenario,
        start_year: int,
    ) -> None:
        self.metrics_panel = metrics_panel.copy()
        self.scenario = scenario
        self.start_year = start_year

        # Ensure numeric Year
        self.metrics_panel["Year"] = pd.to_numeric(
            self.metrics_panel["Year"],
            errors="coerce",
        ).astype("Int64")

        self.state = self._initialize_state()

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        node_records = []
        aggregate_records = []

        for _ in range(self.scenario.n_steps + 1):
            node_records.append(self._state_to_node_frame())
            aggregate_records.append(self._state_to_aggregate_record())

            if self.state.step < self.scenario.n_steps:
                self.step()

        node_results = pd.concat(node_records, ignore_index=True)
        aggregate_results = pd.DataFrame(aggregate_records)

        return node_results, aggregate_results

    def step(self) -> None:
        # --- 1. Production ---
        next_X = update_production(
            demand=self.state.D,
            capacity=self.state.K,
            available_inputs=self.state.M,
            inventory=self.state.I,
        )

        # --- 2. Inventory ---
        next_I = update_inventory(
            previous_inventory=self.state.I,
            available_inputs=self.state.M,
            production=next_X,
            inventory_replenishment_rate=self.scenario.inventory_replenishment_rate,
        )

        # --- 3. Network exposure using ET ---
        ET = self.state.ET

        # Row-normalized (outgoing)
        row_sums = ET.sum(axis=1, keepdims=True)
        A_out = np.divide(
            ET,
            row_sums,
            out=np.zeros_like(ET),
            where=row_sums > 0,
        )

        # Column-normalized (incoming)
        col_sums = ET.sum(axis=0, keepdims=True)
        A_in = np.divide(
            ET,
            col_sums,
            out=np.zeros_like(ET),
            where=col_sums > 0,
        )

        next_g_out = A_out @ self.state.g_local
        next_g_in = A_in.T @ self.state.g_local
        next_NG = 0.5 * (next_g_in + next_g_out)

        # --- 4. Emissions intensity ---
        next_EI = update_emissions_intensity(
            emissions_intensity=self.state.EI,
            capability_readiness=self.state.capability_readiness,
            network_green_exposure=next_NG,
            alpha=self.scenario.alpha,
            beta=self.scenario.beta,
        )

        # --- 5. Local green-ness ---
        next_g_local = update_local_greeness(
            emissions_intensity=next_EI,
            epsilon=self.scenario.epsilon,
        )

        # --- 6. Capability ---
        next_capability = update_capability_readiness(
            local_greeness=next_g_local,
            network_green_exposure=next_NG,
            ecosystem_exposure=None,
            lambda_local=self.scenario.lambda_local,
            lambda_network=self.scenario.lambda_network,
            lambda_ecosystem=self.scenario.lambda_ecosystem,
        )

        # --- 7. Update state ---
        self.state = ABMState(
            step=self.state.step + 1,
            scenario=self.state.scenario,
            country_sector=self.state.country_sector,
            country=self.state.country,
            sector=self.state.sector,
            X=next_X,
            D=self.state.D,
            M=self.state.M,
            K=self.state.K,
            I=next_I,
            EI=next_EI,
            g_local=next_g_local,
            g_in=next_g_in,
            g_out=next_g_out,
            NG=next_NG,
            capability_readiness=next_capability,
            out_strength=self.state.out_strength,
            in_strength=self.state.in_strength,
            ET=self.state.ET,
        )

    def _initialize_state(self) -> ABMState:
        base = self.metrics_panel[
            self.metrics_panel["Year"] == self.start_year
        ].copy()

        if base.empty:
            available = sorted(self.metrics_panel["Year"].dropna().unique())
            raise ValueError(
                f"No metrics found for start year: {self.start_year}. "
                f"Available years: {available}"
            )

        base = base.sort_values("country_sector").reset_index(drop=True)

        # --- Load ET ---
        et_path = Path(f"data/metrics/{self.start_year}/et_{self.start_year}.parquet")

        if not et_path.exists():
            raise FileNotFoundError(f"Missing ET matrix: {et_path}")

        et_df = pd.read_parquet(et_path)

        expected_labels = pd.Index(base["country_sector"].astype(str))

        et_df.index = et_df.index.astype(str)
        et_df.columns = et_df.columns.astype(str)

        et_df = et_df.loc[expected_labels, expected_labels]

        if not et_df.index.equals(expected_labels):
            raise ValueError("ET rows not aligned with metrics panel")

        if not et_df.columns.equals(expected_labels):
            raise ValueError("ET columns not aligned with metrics panel")

        ET = et_df.to_numpy(dtype=np.float32)

        # --- Derive network strengths directly from ET ---
        out_strength = ET.sum(axis=1).astype(np.float32)
        in_strength = ET.sum(axis=0).astype(np.float32)

        return ABMState(
            step=0,
            scenario=self.scenario.name,
            country_sector=base["country_sector"].to_numpy(),
            country=base["Country"].to_numpy(),
            sector=base["Sector"].to_numpy(),
            X=self._col(base, "X"),
            D=self._col(base, "D"),
            M=self._col(base, "M"),
            K=self._col(base, "capacity_base") * self.scenario.kappa / 1.10,
            I=self._col(base, "inventory_base") * self.scenario.inventory_days / 30.0,
            EI=self._col(base, "EI"),
            g_local=self._col(base, "g_local"),
            g_in=self._col(base, "g_in"),
            g_out=self._col(base, "g_out"),
            NG=self._col(base, "NG"),
            capability_readiness=self._col(base, "capability_readiness"),
            out_strength=out_strength,
            in_strength=in_strength,
            ET=ET,
        )

    def _state_to_node_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "step": self.state.step,
                "scenario": self.state.scenario,
                "country_sector": self.state.country_sector,
                "Country": self.state.country,
                "Sector": self.state.sector,
                "X": self.state.X,
                "D": self.state.D,
                "M": self.state.M,
                "K": self.state.K,
                "I": self.state.I,
                "EI": self.state.EI,
                "g_local": self.state.g_local,
                "g_in": self.state.g_in,
                "g_out": self.state.g_out,
                "NG": self.state.NG,
                "capability_readiness": self.state.capability_readiness,
                "out_strength": self.state.out_strength,
                "in_strength": self.state.in_strength,
            }
        )

        df["regime"] = self._classify_regimes(df)

        return df

    def _state_to_aggregate_record(self) -> dict[str, float | int | str]:
        return {
            "step": self.state.step,
            "scenario": self.state.scenario,
            "total_output": float(np.sum(self.state.X)),
            "total_emissions": compute_total_emissions(self.state.X, self.state.EI),
            "mean_ei": compute_mean_emissions_intensity(self.state.X, self.state.EI),
            "mean_g_local": compute_mean_greeness(self.state.X, self.state.g_local),
            "mean_g_in": compute_mean_greeness(self.state.X, self.state.g_in),
            "mean_g_out": compute_mean_greeness(self.state.X, self.state.g_out),
            "mean_capability_readiness": compute_mean_greeness(
                self.state.X,
                self.state.capability_readiness,
            ),
        }

    @staticmethod
    def _classify_regimes(df: pd.DataFrame) -> pd.Series:
        out_strength = pd.to_numeric(df["out_strength"], errors="coerce")
        g_out = pd.to_numeric(df["g_out"], errors="coerce")

        centrality_threshold = out_strength.median()
        green_threshold = g_out.median()

        regimes = np.select(
            [
                (out_strength >= centrality_threshold) & (g_out >= green_threshold),
                (out_strength >= centrality_threshold) & (g_out < green_threshold),
                (out_strength < centrality_threshold) & (g_out >= green_threshold),
                (out_strength < centrality_threshold) & (g_out < green_threshold),
            ],
            [
                "green-core",
                "brown-core",
                "green-periphery",
                "brown-periphery",
            ],
            default="unclassified",
        )

        return pd.Series(regimes, index=df.index)

    @staticmethod
    def _col(df: pd.DataFrame, column: str) -> np.ndarray:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

        values = pd.to_numeric(df[column], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan)

        if values.isna().any():
            raise ValueError(f"Column {column} contains invalid values")

        return values.to_numpy(dtype=np.float32)