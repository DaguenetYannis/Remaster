from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.abm_v3.leontief.coefficients import LeontiefYearData


@dataclass
class LeontiefPropagationResult:
    """Output and diagnostics from iterative Leontief demand circulation."""

    year: int
    X_iterative: pd.Series
    round_summaries: pd.DataFrame
    rounds_used: int
    tolerance: float
    max_rounds: int
    initial_final_demand_total: float
    accumulated_output_total: float
    final_residual_total: float
    final_residual_share: float
    converged: bool


@dataclass
class LeontiefPropagationEngine:
    """Propagate final demand through ``flow_next = A @ flow``."""

    tolerance: float = 1e-8
    max_rounds: int = 200

    def propagate(self, year_data: LeontiefYearData) -> LeontiefPropagationResult:
        """Iteratively approximate ``X = Y + AY + A2Y + ...``."""
        print(
            "[ABM v3 Leontief] Starting propagation: "
            f"tolerance={self.tolerance}, max_rounds={self.max_rounds}"
        )
        labels = year_data.labels["country_sector"].tolist()
        flow = year_data.Y_final_demand.to_numpy(dtype=float)
        flow = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
        accumulated_output = np.zeros_like(flow, dtype=float)
        initial_total = float(np.sum(np.abs(flow)))
        denominator = max(initial_total, np.finfo(float).eps)
        round_rows: list[dict[str, object]] = []
        converged = False
        final_residual_total = float(np.sum(np.abs(flow)))
        final_residual_share = final_residual_total / denominator
        rounds_used = 0

        for round_number in range(self.max_rounds + 1):
            flow = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
            accumulated_output += flow
            flow_total = float(np.sum(flow))
            absolute_flow_total = float(np.sum(np.abs(flow)))
            residual_share = absolute_flow_total / denominator
            converged_this_round = residual_share < self.tolerance
            round_rows.append(
                {
                    "year": year_data.year,
                    "round": round_number,
                    "flow_total": flow_total,
                    "absolute_flow_total": absolute_flow_total,
                    "residual_share": residual_share,
                    "accumulated_output_total": float(np.sum(accumulated_output)),
                    "converged": converged_this_round,
                }
            )
            rounds_used = round_number
            final_residual_total = absolute_flow_total
            final_residual_share = residual_share
            if round_number > 0:
                print(f"[ABM v3 Leontief] Round {round_number}: residual_share={residual_share:.12g}")
            if converged_this_round:
                converged = True
                break
            if round_number == self.max_rounds:
                break
            flow = year_data.A @ flow

        x_iterative = pd.Series(accumulated_output, index=labels, name="X_iterative")
        return LeontiefPropagationResult(
            year=year_data.year,
            X_iterative=x_iterative,
            round_summaries=pd.DataFrame(round_rows),
            rounds_used=rounds_used,
            tolerance=float(self.tolerance),
            max_rounds=int(self.max_rounds),
            initial_final_demand_total=initial_total,
            accumulated_output_total=float(np.sum(accumulated_output)),
            final_residual_total=final_residual_total,
            final_residual_share=final_residual_share,
            converged=converged,
        )
