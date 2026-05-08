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
    mode: str = "raw"
    X_raw_observed: pd.Series | None = None
    Y_raw_final_demand: pd.Series | None = None
    X_used_for_coefficients: pd.Series | None = None
    Y_used_for_propagation: pd.Series | None = None
    mode_diagnostics: pd.DataFrame | None = None
    excluded_fd_columns: pd.DataFrame | None = None
    rescaled_columns: pd.DataFrame | None = None
    invalid_output_columns: pd.DataFrame | None = None
    negative_flows: pd.DataFrame | None = None
    total_negative_T_entries: int = 0
    total_negative_FD_entries: int = 0
    most_negative_T_value: float = np.nan
    most_negative_FD_value: float = np.nan


class LeontiefCoefficientBuilder:
    """Build sparse technical coefficients from raw Eora T and FD matrices."""

    def __init__(self, paths: ABMV3Paths, config: LeontiefPropagationConfig) -> None:
        self.paths = paths
        self.config = config

    def load_year(self, year: int) -> LeontiefYearData:
        """Load one year and construct ``A_ji = T_ji / X_i``."""
        mode = self._validate_mode(self.config.leontief_mode)
        print(f"[ABM v3 Leontief] Loading year {year}...")
        print(f"[ABM v3 Leontief] Coefficient mode: {mode}")
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
        fd_labels = self._load_labels_FD(year, fd_matrix)
        fd_matrix.columns = fd_labels

        print("[ABM v3 Leontief] Computing X and Y...")
        raw_y_values = fd_matrix.sum(axis=1).to_numpy(dtype=float)
        intermediate_output_values = t_matrix.sum(axis=0).to_numpy(dtype=float)
        raw_x_values = (intermediate_output_values + raw_y_values).astype(float)
        raw_y_final_demand = pd.Series(raw_y_values, index=labels, name="Y_raw_final_demand")
        raw_x_observed = pd.Series(raw_x_values, index=labels, name="X_raw_observed")
        fd_for_mode, excluded_fd_columns, mode_metrics = self._build_mode_final_demand_matrix(
            year,
            mode,
            fd_matrix,
        )
        used_y_values = fd_for_mode.sum(axis=1).to_numpy(dtype=float)
        used_x_values = (intermediate_output_values + used_y_values).astype(float)
        y_used_for_propagation = pd.Series(used_y_values, index=labels, name="Y_used_for_propagation")
        x_used_for_coefficients = pd.Series(used_x_values, index=labels, name="X_used_for_coefficients")
        negative_flows, negative_summary = self._build_negative_flow_diagnostics(
            year,
            labels,
            t_matrix,
            fd_matrix,
        )

        invalid_output_columns = self._build_invalid_output_columns(year, labels_frame, x_used_for_coefficients)
        print("[ABM v3 Leontief] Building sparse A...")
        coefficient_matrix = self._build_sparse_technical_coefficients(t_matrix, x_used_for_coefficients)
        coefficient_matrix, rescaled_columns, rescale_metrics = self._apply_mode_rescaling(
            year,
            mode,
            labels_frame,
            coefficient_matrix,
        )
        mode_diagnostics = self._build_mode_diagnostics(
            year,
            mode,
            fd_matrix,
            raw_y_final_demand,
            y_used_for_propagation,
            mode_metrics,
            rescale_metrics,
        )

        return LeontiefYearData(
            year=year,
            mode=mode,
            labels=labels_frame,
            X_observed=raw_x_observed,
            Y_final_demand=y_used_for_propagation,
            A=coefficient_matrix,
            X_raw_observed=raw_x_observed,
            Y_raw_final_demand=raw_y_final_demand,
            X_used_for_coefficients=x_used_for_coefficients,
            Y_used_for_propagation=y_used_for_propagation,
            mode_diagnostics=mode_diagnostics,
            excluded_fd_columns=excluded_fd_columns,
            rescaled_columns=rescaled_columns,
            invalid_output_columns=invalid_output_columns,
            negative_flows=negative_flows,
            total_negative_T_entries=negative_summary["total_negative_T_entries"],
            total_negative_FD_entries=negative_summary["total_negative_FD_entries"],
            most_negative_T_value=negative_summary["most_negative_T_value"],
            most_negative_FD_value=negative_summary["most_negative_FD_value"],
        )

    def _validate_mode(self, mode: str) -> str:
        if mode not in self.config.allowed_leontief_modes:
            allowed = ", ".join(self.config.allowed_leontief_modes)
            raise ValueError(f"Unknown Leontief coefficient mode '{mode}'. Allowed modes: {allowed}")
        return mode

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

    def _load_labels_FD(self, year: int, fd_matrix: pd.DataFrame) -> list[str]:
        labels_path = self.paths.label_file(year, "labels_FD")
        if not labels_path.exists():
            return [str(column) for column in fd_matrix.columns]
        labels: list[str] = []
        for line in labels_path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = [part.strip() for part in line.split("\t") if part.strip()]
            labels.append(" | ".join(parts))
        if len(labels) != fd_matrix.shape[1]:
            raise ValueError(
                f"labels_FD count {len(labels)} does not match FD columns for year {year}: "
                f"FD={fd_matrix.shape}"
            )
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

    def _build_mode_final_demand_matrix(
        self,
        year: int,
        mode: str,
        fd_matrix: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
        raw_values = fd_matrix.to_numpy(dtype=float, copy=False)
        negative_before = int((np.isfinite(raw_values) & (raw_values < 0.0)).sum())
        raw_y = fd_matrix.sum(axis=1).to_numpy(dtype=float)
        metrics: dict[str, object] = {
            "negative_FD_entries_before": negative_before,
            "negative_Y_count_before": int((raw_y < 0.0).sum()),
            "excluded_fd_column_count": 0,
            "total_excluded_inventory_value": 0.0,
            "negative_FD_entries_after": negative_before,
            "negative_Y_count_after": int((raw_y < 0.0).sum()),
            "total_negative_FD_value": float(raw_values[raw_values < 0.0].sum()) if negative_before else 0.0,
            "total_positive_FD_value": float(raw_values[raw_values > 0.0].sum()) if (raw_values > 0.0).any() else 0.0,
            "difference_between_raw_Y_and_positive_Y": 0.0,
            "number_of_rows_where_Y_changed": 0,
        }
        if mode == "fd_without_inventory":
            inventory_mask = self._inventory_fd_column_mask(list(fd_matrix.columns))
            excluded = self._build_excluded_fd_columns(year, fd_matrix, inventory_mask)
            fd_used = fd_matrix.loc[:, ~inventory_mask].copy()
            used_values = fd_used.to_numpy(dtype=float, copy=False)
            used_y = fd_used.sum(axis=1).to_numpy(dtype=float)
            metrics.update(
                {
                    "excluded_fd_column_count": int(inventory_mask.sum()),
                    "total_excluded_inventory_value": float(fd_matrix.loc[:, inventory_mask].to_numpy(dtype=float).sum())
                    if inventory_mask.any()
                    else 0.0,
                    "negative_FD_entries_after": int((np.isfinite(used_values) & (used_values < 0.0)).sum()),
                    "negative_Y_count_after": int((used_y < 0.0).sum()),
                    "number_of_rows_where_Y_changed": int((~np.isclose(raw_y, used_y, equal_nan=True)).sum()),
                }
            )
            return fd_used, excluded, metrics
        if mode == "positive_final_demand_only":
            fd_used = fd_matrix.clip(lower=0.0)
            used_y = fd_used.sum(axis=1).to_numpy(dtype=float)
            metrics.update(
                {
                    "negative_FD_entries_after": 0,
                    "negative_Y_count_after": int((used_y < 0.0).sum()),
                    "difference_between_raw_Y_and_positive_Y": float(np.sum(used_y - raw_y)),
                    "number_of_rows_where_Y_changed": int((~np.isclose(raw_y, used_y, equal_nan=True)).sum()),
                }
            )
            return fd_used, self._empty_excluded_fd_columns(), metrics
        return fd_matrix.copy(), self._empty_excluded_fd_columns(), metrics

    def _inventory_fd_column_mask(self, fd_columns: list[object]) -> np.ndarray:
        patterns = tuple(pattern.lower() for pattern in self.config.inventory_label_patterns)
        mask = []
        for column in fd_columns:
            label = str(column).lower()
            mask.append(any(pattern in label for pattern in patterns))
        return np.array(mask, dtype=bool)

    def _build_excluded_fd_columns(
        self,
        year: int,
        fd_matrix: pd.DataFrame,
        inventory_mask: np.ndarray,
    ) -> pd.DataFrame:
        rows = []
        for column_index, column_label in enumerate(fd_matrix.columns):
            if not bool(inventory_mask[column_index]):
                continue
            values = fd_matrix.iloc[:, column_index].to_numpy(dtype=float)
            rows.append(
                {
                    "Year": year,
                    "mode": "fd_without_inventory",
                    "fd_column_index": column_index,
                    "fd_column_label": str(column_label),
                    "column_total": float(np.nansum(values)),
                    "negative_entry_count": int((np.isfinite(values) & (values < 0.0)).sum()),
                    "positive_entry_count": int((np.isfinite(values) & (values > 0.0)).sum()),
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "Year",
                "mode",
                "fd_column_index",
                "fd_column_label",
                "column_total",
                "negative_entry_count",
                "positive_entry_count",
            ],
        )

    def _empty_excluded_fd_columns(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "Year",
                "mode",
                "fd_column_index",
                "fd_column_label",
                "column_total",
                "negative_entry_count",
                "positive_entry_count",
            ]
        )

    def _apply_mode_rescaling(
        self,
        year: int,
        mode: str,
        labels_frame: pd.DataFrame,
        coefficient_matrix: sparse.spmatrix,
    ) -> tuple[sparse.spmatrix, pd.DataFrame, dict[str, object]]:
        if mode not in {"column_rescaled_if_sum_above_one", "column_rescaled_if_abs_sum_above_one"}:
            return coefficient_matrix, self._empty_rescaled_columns(), {
                "rescaled_column_count": 0,
                "max_column_sum_before": self._max_column_sum(coefficient_matrix),
                "max_column_sum_after": self._max_column_sum(coefficient_matrix),
                "total_abs_adjustment": 0.0,
            }
        a_csc = coefficient_matrix.tocsc()
        metric_values = (
            np.asarray(abs(a_csc).sum(axis=0)).ravel()
            if mode == "column_rescaled_if_abs_sum_above_one"
            else np.asarray(a_csc.sum(axis=0)).ravel()
        )
        column_sum_before = np.asarray(a_csc.sum(axis=0)).ravel()
        abs_column_sum_before = np.asarray(abs(a_csc).sum(axis=0)).ravel()
        cap = float(self.config.leontief_column_sum_cap)
        factors = np.ones(a_csc.shape[1], dtype=float)
        rescale_mask = metric_values > cap
        factors[rescale_mask] = cap / metric_values[rescale_mask]
        rows = []
        total_abs_adjustment = 0.0
        for column_index in np.where(rescale_mask)[0]:
            start = a_csc.indptr[column_index]
            end = a_csc.indptr[column_index + 1]
            values = a_csc.data[start:end]
            factor = float(factors[column_index])
            total_abs_adjustment += float(np.sum(np.abs(values * (1.0 - factor))))
            label_row = labels_frame.iloc[int(column_index)]
            rows.append(
                {
                    "Year": year,
                    "mode": mode,
                    "country_sector": label_row["country_sector"],
                    "Country": label_row["Country"],
                    "Country_detail": label_row["Country_detail"],
                    "Category": label_row["Category"],
                    "Sector": label_row["Sector"],
                    "column_sum_before": float(column_sum_before[column_index]),
                    "abs_column_sum_before": float(abs_column_sum_before[column_index]),
                    "rescale_factor": factor,
                    "column_sum_after": float(column_sum_before[column_index] * factor),
                    "abs_column_sum_after": float(abs_column_sum_before[column_index] * factor),
                }
            )
        scaled = (a_csc @ sparse.diags(factors, format="csc")).tocsr()
        scaled.eliminate_zeros()
        rescaled_columns = pd.DataFrame(
            rows,
            columns=[
                "Year",
                "mode",
                "country_sector",
                "Country",
                "Country_detail",
                "Category",
                "Sector",
                "column_sum_before",
                "abs_column_sum_before",
                "rescale_factor",
                "column_sum_after",
                "abs_column_sum_after",
            ],
        )
        metrics = {
            "rescaled_column_count": int(rescale_mask.sum()),
            "max_column_sum_before": float(np.nanmax(metric_values)) if len(metric_values) else np.nan,
            "max_column_sum_after": self._max_column_sum(scaled if mode == "column_rescaled_if_sum_above_one" else abs(scaled)),
            "total_abs_adjustment": total_abs_adjustment,
        }
        return scaled, rescaled_columns, metrics

    def _empty_rescaled_columns(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "Year",
                "mode",
                "country_sector",
                "Country",
                "Country_detail",
                "Category",
                "Sector",
                "column_sum_before",
                "abs_column_sum_before",
                "rescale_factor",
                "column_sum_after",
                "abs_column_sum_after",
            ]
        )

    def _max_column_sum(self, matrix: sparse.spmatrix) -> float:
        if matrix.shape[1] == 0:
            return np.nan
        return float(np.asarray(matrix.sum(axis=0)).ravel().max())

    def _build_mode_diagnostics(
        self,
        year: int,
        mode: str,
        raw_fd_matrix: pd.DataFrame,
        raw_y: pd.Series,
        used_y: pd.Series,
        mode_metrics: dict[str, object],
        rescale_metrics: dict[str, object],
    ) -> pd.DataFrame:
        row = {
            "Year": year,
            "mode": mode,
            "raw_final_demand_total": float(raw_y.sum()),
            "used_final_demand_total": float(used_y.sum()),
            "final_demand_total_difference": float(used_y.sum() - raw_y.sum()),
            "fd_column_count": int(raw_fd_matrix.shape[1]),
        }
        row.update(mode_metrics)
        row.update(rescale_metrics)
        return pd.DataFrame([row])

    def _build_negative_flow_diagnostics(
        self,
        year: int,
        labels: list[str],
        t_matrix: pd.DataFrame,
        fd_matrix: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        max_rows = int(self.config.max_negative_flow_rows)
        t_values = t_matrix.to_numpy(dtype=float, copy=False)
        fd_values = fd_matrix.to_numpy(dtype=float, copy=False)
        t_negative_mask = np.isfinite(t_values) & (t_values < 0.0)
        fd_negative_mask = np.isfinite(fd_values) & (fd_values < 0.0)
        total_negative_t = int(t_negative_mask.sum())
        total_negative_fd = int(fd_negative_mask.sum())
        most_negative_t = float(np.nanmin(t_values[t_negative_mask])) if total_negative_t else np.nan
        most_negative_fd = float(np.nanmin(fd_values[fd_negative_mask])) if total_negative_fd else np.nan
        t_rows = self._negative_t_rows(year, labels, t_values, t_negative_mask, max_rows)
        remaining_rows = max(0, max_rows - len(t_rows))
        fd_rows = self._negative_fd_rows(year, labels, list(fd_matrix.columns), fd_values, fd_negative_mask, remaining_rows)
        negative_flows = pd.DataFrame(
            t_rows + fd_rows,
            columns=["Year", "matrix", "row_country_sector", "col_country_sector", "col_label", "value"],
        )
        summary = {
            "total_negative_T_entries": total_negative_t,
            "total_negative_FD_entries": total_negative_fd,
            "most_negative_T_value": most_negative_t,
            "most_negative_FD_value": most_negative_fd,
        }
        return negative_flows, summary

    def _negative_t_rows(
        self,
        year: int,
        labels: list[str],
        values: np.ndarray,
        negative_mask: np.ndarray,
        max_rows: int,
    ) -> list[dict[str, object]]:
        row_indices, col_indices = np.where(negative_mask)
        if len(row_indices) > max_rows:
            selected = np.argsort(values[row_indices, col_indices])[:max_rows]
            row_indices = row_indices[selected]
            col_indices = col_indices[selected]
        rows = []
        for row_index, col_index in zip(row_indices, col_indices):
            rows.append(
                {
                    "Year": year,
                    "matrix": "T",
                    "row_country_sector": labels[int(row_index)],
                    "col_country_sector": labels[int(col_index)],
                    "col_label": "",
                    "value": float(values[row_index, col_index]),
                }
            )
        return rows

    def _negative_fd_rows(
        self,
        year: int,
        labels: list[str],
        fd_columns: list[object],
        values: np.ndarray,
        negative_mask: np.ndarray,
        max_rows: int,
    ) -> list[dict[str, object]]:
        if max_rows <= 0:
            return []
        row_indices, col_indices = np.where(negative_mask)
        if len(row_indices) > max_rows:
            selected = np.argsort(values[row_indices, col_indices])[:max_rows]
            row_indices = row_indices[selected]
            col_indices = col_indices[selected]
        rows = []
        for row_index, col_index in zip(row_indices, col_indices):
            rows.append(
                {
                    "Year": year,
                    "matrix": "FD",
                    "row_country_sector": labels[int(row_index)],
                    "col_country_sector": "",
                    "col_label": str(fd_columns[int(col_index)]),
                    "value": float(values[row_index, col_index]),
                }
            )
        return rows
