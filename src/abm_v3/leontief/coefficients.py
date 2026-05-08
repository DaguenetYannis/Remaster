from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from src.abm_v3.config import LeontiefPropagationConfig
from src.abm_v3.paths import ABMV3Paths


@dataclass
class LeontiefYearData:
    """Aligned Eora matrices and coefficients for one propagation year."""

    year: int
    labels: pd.DataFrame
    X_observed: pd.Series
    Y_final_demand: pd.Series
    A: sparse.spmatrix
    invalid_output_columns: pd.DataFrame | None = None


class LeontiefCoefficientBuilder:
    """Build sparse technical coefficients from raw Eora T and FD matrices."""

    def __init__(self, paths: ABMV3Paths, config: LeontiefPropagationConfig) -> None:
        self.paths = paths
        self.config = config

    def load_year(self, year: int) -> LeontiefYearData:
        """Load one year and construct ``A_ji = T_ji / X_i``."""
        print(f"[ABM v3 Leontief] Loading year {year}...")
        print("[ABM v3 Leontief] Loading T and FD...")
        labels = self._load_labels_T(year)
        labels_frame = self._build_labels_frame(labels)
        t_matrix = pd.read_parquet(self.paths.eora_matrix_file(year, "T"))
        fd_matrix = pd.read_parquet(self.paths.eora_matrix_file(year, "FD"))
        self._validate_dimensions(year, labels, t_matrix, fd_matrix)
        self._validate_existing_axis_labels(year, labels, t_matrix, fd_matrix)

        # The Data Reference makes labels_T authoritative for T rows, T columns, and FD rows.
        t_matrix = t_matrix.copy()
        fd_matrix = fd_matrix.copy()
        t_matrix.index = labels
        t_matrix.columns = labels
        fd_matrix.index = labels

        print("[ABM v3 Leontief] Computing X and Y...")
        y_values = fd_matrix.sum(axis=1).to_numpy(dtype=float)
        x_values = (t_matrix.sum(axis=0).to_numpy(dtype=float) + y_values).astype(float)
        y_final_demand = pd.Series(y_values, index=labels, name="Y_final_demand")
        x_observed = pd.Series(x_values, index=labels, name="X_observed")

        invalid_output_columns = self._build_invalid_output_columns(year, labels_frame, x_observed)
        print("[ABM v3 Leontief] Building sparse A...")
        coefficient_matrix = self._build_sparse_technical_coefficients(t_matrix, x_observed)

        return LeontiefYearData(
            year=year,
            labels=labels_frame,
            X_observed=x_observed,
            Y_final_demand=y_final_demand,
            A=coefficient_matrix,
            invalid_output_columns=invalid_output_columns,
        )

    def _load_labels_T(self, year: int) -> list[str]:
        labels_path = self.paths.label_file(year, "labels_T")
        if not labels_path.exists():
            raise FileNotFoundError(str(labels_path))
        labels: list[str] = []
        for line in labels_path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = [part.strip() for part in line.split("\t") if part.strip()]
            labels.append(" | ".join(parts))
        if not labels:
            raise ValueError(f"No labels found in {labels_path}")
        return labels

    def _build_labels_frame(self, labels: list[str]) -> pd.DataFrame:
        rows: list[dict[str, str]] = []
        invalid_labels: list[str] = []
        for label in labels:
            parts = [part.strip() for part in label.split("|")]
            if len(parts) != 4:
                invalid_labels.append(label)
                continue
            rows.append(
                {
                    "country_sector": label,
                    "Country": parts[0],
                    "Country_detail": parts[1],
                    "Category": parts[2],
                    "Sector": parts[3],
                }
            )
        if invalid_labels:
            raise ValueError(
                f"{len(invalid_labels)} labels do not split into four fields. "
                f"First invalid label: {invalid_labels[0]}"
            )
        return pd.DataFrame(rows)

    def _validate_dimensions(
        self,
        year: int,
        labels: list[str],
        t_matrix: pd.DataFrame,
        fd_matrix: pd.DataFrame,
    ) -> None:
        label_count = len(labels)
        if t_matrix.shape != (label_count, label_count):
            raise ValueError(
                f"labels_T count {label_count} does not match square T dimensions "
                f"for year {year}: T={t_matrix.shape}"
            )
        if fd_matrix.shape[0] != label_count:
            raise ValueError(
                f"labels_T count {label_count} does not match FD rows for year {year}: "
                f"FD={fd_matrix.shape}"
            )

    def _validate_existing_axis_labels(
        self,
        year: int,
        labels: list[str],
        t_matrix: pd.DataFrame,
        fd_matrix: pd.DataFrame,
    ) -> None:
        self._validate_axis_if_labelled(year, "T rows", list(t_matrix.index), labels)
        self._validate_axis_if_labelled(year, "T columns", list(t_matrix.columns), labels)
        self._validate_axis_if_labelled(year, "FD rows", list(fd_matrix.index), labels)

    def _validate_axis_if_labelled(
        self,
        year: int,
        axis_name: str,
        observed_axis: list[object],
        expected_labels: list[str],
    ) -> None:
        observed_strings = [str(value) for value in observed_axis]
        expected_set = set(expected_labels)
        observed_overlap = sum(1 for value in observed_strings if value in expected_set)
        if observed_overlap == 0:
            return
        if observed_strings != expected_labels:
            raise ValueError(
                f"{axis_name} do not align with labels_T for year {year}. "
                f"First observed label: {observed_strings[0]}; "
                f"first labels_T label: {expected_labels[0]}"
            )

    def _build_invalid_output_columns(
        self,
        year: int,
        labels_frame: pd.DataFrame,
        x_observed: pd.Series,
    ) -> pd.DataFrame:
        x_values = x_observed.to_numpy(dtype=float)
        invalid_mask = (~np.isfinite(x_values)) | (x_values <= 0.0)
        invalid = labels_frame.loc[invalid_mask].copy()
        invalid.insert(0, "Year", year)
        invalid["X_observed"] = x_values[invalid_mask]
        invalid["reason"] = np.where(
            ~np.isfinite(x_values[invalid_mask]),
            "missing_or_non_finite_output",
            "non_positive_output",
        )
        return invalid

    def _build_sparse_technical_coefficients(
        self,
        t_matrix: pd.DataFrame,
        x_observed: pd.Series,
    ) -> sparse.spmatrix:
        t_sparse = sparse.csc_matrix(t_matrix.to_numpy(dtype=float, copy=True))
        x_values = x_observed.to_numpy(dtype=float)
        valid_columns = np.isfinite(x_values) & (x_values > 0.0)
        inverse_output = np.zeros_like(x_values, dtype=float)
        inverse_output[valid_columns] = 1.0 / x_values[valid_columns]
        coefficient_matrix = t_sparse @ sparse.diags(inverse_output, format="csc")
        coefficient_matrix.eliminate_zeros()
        return coefficient_matrix.tocsr()
