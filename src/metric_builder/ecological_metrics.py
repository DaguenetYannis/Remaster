import numpy as np
import pandas as pd

from .economic_metrics import (
    compute_total_output,
    compute_technical_coefficients,
    compute_leontief_inverse,
    compute_final_demand_total,
)
from .utils import load_labelled_matrices

CO2_TOTAL_LABEL = "Total CO2 emissions (Gg) from EDGAR | Total"


# =========================
# CORE ECOLOGICAL METRICS
# =========================

def compute_ei(year, base_path, label_base_path=None):
    matrices = {}

    for matrix, _, df in load_labelled_matrices(
        year,
        ["T", "FD", "Q"],
        base_path=base_path,
        label_base_path=label_base_path,
    ):
        matrices[matrix] = df.astype("float64")

    T = matrices["T"]
    FD = matrices["FD"]
    Q = matrices["Q"]

    X = compute_total_output(T, FD)
    emissions = Q.loc[CO2_TOTAL_LABEL].astype("float64")

    EI = emissions / X
    EI = EI.replace([np.inf, -np.inf], np.nan).fillna(0)
    EI.name = "emissions_intensity"

    return EI


def compute_et(year, base_path, label_base_path=None):
    matrices = {}

    for matrix, _, df in load_labelled_matrices(
        year,
        ["T", "FD", "Q"],
        base_path=base_path,
        label_base_path=label_base_path,
    ):
        matrices[matrix] = df.astype("float64")

    T = matrices["T"]
    FD = matrices["FD"]
    Q = matrices["Q"]

    X = compute_total_output(T, FD)
    A = compute_technical_coefficients(T, X)
    L = compute_leontief_inverse(A)
    FD_total = compute_final_demand_total(FD)

    emissions = Q.loc[CO2_TOTAL_LABEL].astype("float64")
    EI = emissions / X
    EI = EI.replace([np.inf, -np.inf], np.nan).fillna(0)

    ET = L.mul(EI, axis=0).mul(FD_total, axis=1)

    ET.index.name = "source_sector"
    ET.columns.name = "target_sector"

    return ET


# =========================
# NETWORK GREEN-NESS
# =========================

class NetworkGreennessCalculator:
    def __init__(self, epsilon: float = 1e-12) -> None:
        self.epsilon = epsilon

    def compute_base_greenness(self, emissions_intensity: pd.Series) -> pd.Series:
        ei = emissions_intensity.copy()
        ei = ei.clip(lower=0)

        g_base = -np.log(ei + self.epsilon)

        # Avoid extreme values from near-zero EI
        g_base = g_base.clip(lower=0, upper=10)

        g_base = g_base.replace([np.inf, -np.inf], np.nan).fillna(0)
        g_base.name = "g_base"

        return g_base

    def row_normalize(self, matrix: pd.DataFrame) -> pd.DataFrame:
        row_sums = matrix.sum(axis=1)
        return matrix.div(row_sums.replace(0, np.nan), axis=0).fillna(0)

    def column_normalize(self, matrix: pd.DataFrame) -> pd.DataFrame:
        col_sums = matrix.sum(axis=0)
        return matrix.div(col_sums.replace(0, np.nan), axis=1).fillna(0)

    def compute(self, emissions_intensity: pd.Series, et: pd.DataFrame) -> pd.DataFrame:
        # Base green-ness
        g_base = self.compute_base_greenness(emissions_intensity)

        # Align index
        g_base = g_base.reindex(et.index).fillna(0)

        # Normalize ET
        w_out = self.row_normalize(et)
        w_in = self.column_normalize(et)

        # Compute network green-ness
        g_out_network = w_out.dot(g_base)
        g_in_network = w_in.T.dot(g_base)

        result = pd.concat(
            [
                g_base.rename("g_base"),
                g_out_network.rename("g_out_network"),
                g_in_network.rename("g_in_network"),
            ],
            axis=1,
        )

        return result.replace([np.inf, -np.inf], np.nan).fillna(0)


def compute_network_greenness(EI, ET):
    calculator = NetworkGreennessCalculator()
    return calculator.compute(EI, ET)