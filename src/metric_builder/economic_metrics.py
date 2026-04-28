import numpy as np
import pandas as pd


def compute_total_output(T, FD):
    return T.sum(axis=0) + FD.sum(axis=1)


def compute_technical_coefficients(T, X):
    return T.div(X, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0)


def compute_leontief_inverse(A):
    I = np.eye(A.shape[0])
    return pd.DataFrame(
        np.linalg.inv(I - A.to_numpy()),
        index=A.index,
        columns=A.columns,
    )


def compute_final_demand_total(FD):
    return FD.sum(axis=1)