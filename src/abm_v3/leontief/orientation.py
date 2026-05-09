from __future__ import annotations

import logging
import warnings
from contextlib import redirect_stdout
from dataclasses import dataclass, replace
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.abm_v3.config import ABMV3Config, LeontiefPropagationConfig
from src.abm_v3.leontief.coefficients import LeontiefYearData
from src.abm_v3.leontief.propagation import LeontiefPropagationEngine
from src.abm_v3.leontief.validation import LeontiefPropagationValidator
from src.abm_v3.leontief.viability import LeontiefViabilityAnalyzer
from src.abm_v3.paths import ABMV3Paths

LOGGER = logging.getLogger(__name__)


BASE_ORIENTATION_MODES = [
    "current_column_output",
    "row_output_standard_io",
    "transpose_row_output",
]

FD_WITHOUT_INVENTORY_ORIENTATION_MODES = [
    "current_column_output_fd_without_inventory",
    "row_output_fd_without_inventory",
    "transpose_row_output_fd_without_inventory",
]

ALL_ORIENTATION_MODES = BASE_ORIENTATION_MODES + FD_WITHOUT_INVENTORY_ORIENTATION_MODES


@dataclass(frozen=True)
class OrientationInputData:
    """Raw matrices and labels prepared for orientation diagnostics."""

    year: int
    labels: list[str]
    labels_frame: pd.DataFrame
    t_matrix: pd.DataFrame
    fd_matrix: pd.DataFrame
    fd_labels: list[str]
    alignment_notes: list[str]


@dataclass(frozen=True)
class OrientationCandidate:
    """One diagnostic interpretation of T, FD, X, and A."""

    mode: str
    inventory_excluded: bool
    output_basis: str
    coefficient_source: str


@dataclass(frozen=True)
class OrientationAuditResult:
    """Diagnostic tables produced by an orientation audit."""

    summary: pd.DataFrame
    node_comparison: pd.DataFrame
    suspicious_nodes: pd.DataFrame


class LeontiefOrientationAuditor:
    """Compare candidate Eora T orientations without changing production defaults."""

    def __init__(self, paths: ABMV3Paths, config: ABMV3Config | None = None) -> None:
        self.paths = paths
        self.config = config or ABMV3Config()
        self.leontief_config = self.config.leontief

    def audit_year(
        self,
        year: int,
        max_rounds: int = 400,
        tolerance: float = 1e-8,
        include_fd_without_inventory: bool = True,
        reference: str = "abm_ready",
        spectral_max_iter: int | None = None,
    ) -> OrientationAuditResult:
        """Run all requested orientation diagnostics for one year."""
        if reference not in {"abm_ready", "current"}:
            raise ValueError("reference must be one of: abm_ready, current")

        input_data = self.load_orientation_input_data(year)
        row_sum_t = input_data.t_matrix.sum(axis=1).to_numpy(dtype=float)
        column_sum_t = input_data.t_matrix.sum(axis=0).to_numpy(dtype=float)
        raw_y = input_data.fd_matrix.sum(axis=1).to_numpy(dtype=float)
        current_x = column_sum_t + raw_y
        row_x = row_sum_t + raw_y
        panel_reference = self._load_panel_reference(year, input_data.labels)
        reference_x, reference_name, reference_note = self._select_reference(
            reference,
            current_x,
            panel_reference,
            input_data.labels,
        )
        viability_config = self._build_viability_config(spectral_max_iter)

        modes = list(BASE_ORIENTATION_MODES)
        if include_fd_without_inventory:
            modes.extend(FD_WITHOUT_INVENTORY_ORIENTATION_MODES)

        summary_rows: list[dict[str, object]] = []
        for mode in modes:
            candidate = orientation_candidate_from_mode(mode)
            try:
                summary_rows.append(
                    self._audit_candidate(
                        input_data=input_data,
                        candidate=candidate,
                        reference_x=reference_x,
                        reference_name=reference_name,
                        reference_note=reference_note,
                        max_rounds=max_rounds,
                        tolerance=tolerance,
                        viability_config=viability_config,
                    )
                )
            except Exception as error:  # pragma: no cover - defensive audit behavior
                LOGGER.warning("Orientation candidate %s failed for %s: %s", mode, year, error)
                summary_rows.append(self._failed_candidate_row(year, candidate, reference_name, reference_note, error))

        node_comparison = self.build_node_comparison(
            year=year,
            labels_frame=input_data.labels_frame,
            y_values=raw_y,
            row_sum_t=row_sum_t,
            column_sum_t=column_sum_t,
            current_x=current_x,
            row_x=row_x,
            panel_reference=panel_reference,
        )
        suspicious_nodes = self.build_suspicious_nodes(node_comparison)
        return OrientationAuditResult(
            summary=pd.DataFrame(summary_rows, columns=orientation_summary_columns()),
            node_comparison=node_comparison,
            suspicious_nodes=suspicious_nodes,
        )

    def load_orientation_input_data(self, year: int) -> OrientationInputData:
        """Load T, FD, labels_T, and labels_FD with visible alignment checks."""
        labels = load_labels_file(self.paths.label_file(year, "labels_T"))
        labels_frame = split_country_sector_labels(labels)
        t_matrix = pd.read_parquet(self.paths.eora_matrix_file(year, "T"))
        fd_matrix = pd.read_parquet(self.paths.eora_matrix_file(year, "FD"))
        alignment_notes = validate_matrix_alignment(year, labels, t_matrix, fd_matrix)

        t_matrix = t_matrix.copy()
        fd_matrix = fd_matrix.copy()
        t_matrix.index = labels
        t_matrix.columns = labels
        fd_matrix.index = labels
        fd_labels = self._load_fd_labels(year, fd_matrix)
        fd_matrix.columns = fd_labels
        return OrientationInputData(
            year=year,
            labels=labels,
            labels_frame=labels_frame,
            t_matrix=t_matrix,
            fd_matrix=fd_matrix,
            fd_labels=fd_labels,
            alignment_notes=alignment_notes,
        )

    def build_node_comparison(
        self,
        year: int,
        labels_frame: pd.DataFrame,
        y_values: np.ndarray,
        row_sum_t: np.ndarray,
        column_sum_t: np.ndarray,
        current_x: np.ndarray,
        row_x: np.ndarray,
        panel_reference: pd.Series | None,
    ) -> pd.DataFrame:
        """Compare row- and column-output definitions by country-sector node."""
        comparison = labels_frame.copy()
        comparison.insert(0, "Year", int(year))
        comparison["Y"] = y_values
        comparison["X_current_column_output"] = current_x
        comparison["X_row_output"] = row_x
        comparison["X_abm_ready_panel"] = (
            panel_reference.reindex(comparison["country_sector"]).to_numpy(dtype=float)
            if panel_reference is not None
            else np.nan
        )
        comparison["row_sum_T"] = row_sum_t
        comparison["column_sum_T"] = column_sum_t
        comparison["row_minus_column"] = row_sum_t - column_sum_t
        comparison["row_to_column_ratio"] = safe_divide_array(row_sum_t, column_sum_t)
        comparison["current_column_output_to_panel_ratio"] = safe_divide_array(
            current_x,
            comparison["X_abm_ready_panel"].to_numpy(dtype=float),
        )
        comparison["row_output_to_panel_ratio"] = safe_divide_array(
            row_x,
            comparison["X_abm_ready_panel"].to_numpy(dtype=float),
        )
        absolute_difference = np.abs(current_x - row_x)
        scale = np.nanmax(np.vstack([np.abs(current_x), np.abs(row_x)]), axis=0)
        relative_difference = safe_divide_array(absolute_difference, scale)
        comparison["largest_difference_flag"] = (
            (absolute_difference >= np.nanquantile(absolute_difference, 0.99))
            | (relative_difference >= 0.5)
        )
        return comparison

    def build_suspicious_nodes(self, node_comparison: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
        """Return nodes where row- and column-output definitions diverge most."""
        result = node_comparison.copy()
        result["absolute_X_definition_difference"] = (
            result["X_row_output"].astype(float) - result["X_current_column_output"].astype(float)
        ).abs()
        result["relative_X_definition_difference"] = safe_divide_array(
            result["absolute_X_definition_difference"].to_numpy(dtype=float),
            np.nanmax(
                np.vstack(
                    [
                        np.abs(result["X_row_output"].to_numpy(dtype=float)),
                        np.abs(result["X_current_column_output"].to_numpy(dtype=float)),
                    ]
                ),
                axis=0,
            ),
        )
        return (
            result.sort_values(
                ["largest_difference_flag", "relative_X_definition_difference", "absolute_X_definition_difference"],
                ascending=[False, False, False],
            )
            .head(top_n)
            .copy()
        )

    def _audit_candidate(
        self,
        input_data: OrientationInputData,
        candidate: OrientationCandidate,
        reference_x: pd.Series,
        reference_name: str,
        reference_note: str,
        max_rounds: int,
        tolerance: float,
        viability_config: LeontiefPropagationConfig,
    ) -> dict[str, object]:
        fd_used = self._final_demand_for_candidate(input_data.fd_matrix, candidate)
        y_values = fd_used.sum(axis=1).to_numpy(dtype=float)
        t_values = input_data.t_matrix.to_numpy(dtype=float, copy=False)
        row_sum_t = input_data.t_matrix.sum(axis=1).to_numpy(dtype=float)
        column_sum_t = input_data.t_matrix.sum(axis=0).to_numpy(dtype=float)
        x_values = build_orientation_output(candidate.mode, input_data.t_matrix, fd_used)
        a_matrix = build_orientation_coefficients(candidate.mode, input_data.t_matrix, fd_used)
        year_data = LeontiefYearData(
            year=input_data.year,
            labels=input_data.labels_frame,
            X_observed=reference_x,
            Y_final_demand=pd.Series(y_values, index=input_data.labels, name="Y_final_demand"),
            A=a_matrix,
            mode=candidate.mode,
            X_used_for_coefficients=pd.Series(x_values, index=input_data.labels, name="X_used_for_coefficients"),
            Y_used_for_propagation=pd.Series(y_values, index=input_data.labels, name="Y_used_for_propagation"),
            total_negative_T_entries=int((np.isfinite(t_values) & (t_values < 0.0)).sum()),
            total_negative_FD_entries=int((np.isfinite(fd_used.to_numpy(dtype=float)) & (fd_used.to_numpy(dtype=float) < 0.0)).sum()),
        )
        viability = LeontiefViabilityAnalyzer(viability_config).analyze(year_data)
        with redirect_stdout(StringIO()):
            result = LeontiefPropagationEngine(tolerance=tolerance, max_rounds=max_rounds).propagate(year_data)
        validator = LeontiefPropagationValidator()
        comparison = validator.build_node_comparison(year_data, result)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            validation_summary = validator.build_summary(year_data, result, comparison)
        spectral_a = viability.spectral.loc[viability.spectral["matrix"] == "A"].iloc[0]
        spectral_abs_a = viability.spectral.loc[viability.spectral["matrix"] == "abs_A"].iloc[0]
        viability_summary = viability.summary.iloc[0]
        notes = "; ".join(
            [
                f"reference={reference_name}",
                reference_note,
                f"spectral_radius_max_iter={viability_config.spectral_radius_max_iter}",
                *input_data.alignment_notes,
            ]
        )
        return {
            "Year": input_data.year,
            "orientation_mode": candidate.mode,
            "Y_total": float(np.nansum(y_values)),
            "intermediate_row_sum_total": float(np.nansum(row_sum_t)),
            "intermediate_column_sum_total": float(np.nansum(column_sum_t)),
            "X_total": float(np.nansum(x_values)),
            "X_min": float(np.nanmin(x_values)) if len(x_values) else np.nan,
            "X_p01": float(np.nanquantile(x_values, 0.01)) if len(x_values) else np.nan,
            "X_median": float(np.nanmedian(x_values)) if len(x_values) else np.nan,
            "X_p99": float(np.nanquantile(x_values, 0.99)) if len(x_values) else np.nan,
            "X_max": float(np.nanmax(x_values)) if len(x_values) else np.nan,
            "non_positive_X_count": int((np.isfinite(x_values) & (x_values <= 0.0)).sum()),
            "near_zero_positive_X_count": int(((x_values > 0.0) & (x_values <= viability_config.near_zero_output_threshold)).sum()),
            "negative_Y_count": int((np.isfinite(y_values) & (y_values < 0.0)).sum()),
            "negative_FD_entries": int((np.isfinite(fd_used.to_numpy(dtype=float)) & (fd_used.to_numpy(dtype=float) < 0.0)).sum()),
            "inventory_excluded": bool(candidate.inventory_excluded),
            "spectral_radius_A": float(spectral_a["approximate_spectral_radius"]),
            "spectral_radius_abs_A": float(spectral_abs_a["approximate_spectral_radius"]),
            "spectral_radius_above_one": bool(spectral_a["above_one"]),
            "max_abs_column_sum_A": float(viability_summary["max_abs_column_sum_A"]),
            "high_abs_column_sum_count": int(viability_summary["high_abs_column_sum_count"]),
            "suspicious_column_count": int(viability_summary["suspicious_column_count"]),
            "converged": bool(result.converged),
            "rounds_used": int(result.rounds_used),
            "final_residual_share": float(result.final_residual_share),
            "observed_output_total_reference": float(validation_summary.loc[0, "observed_output_total"]),
            "iterative_output_total": float(validation_summary.loc[0, "accumulated_output_total"]),
            "relative_error_total": float(validation_summary.loc[0, "relative_error_total"]),
            "correlation_iterative_vs_reference": float(validation_summary.loc[0, "correlation_iterative_vs_observed"]),
            "mean_absolute_percentage_error": float(validation_summary.loc[0, "mean_absolute_percentage_error"]),
            "notes": notes,
        }

    def _failed_candidate_row(
        self,
        year: int,
        candidate: OrientationCandidate,
        reference_name: str,
        reference_note: str,
        error: Exception,
    ) -> dict[str, object]:
        row = {column: np.nan for column in orientation_summary_columns()}
        row.update(
            {
                "Year": year,
                "orientation_mode": candidate.mode,
                "inventory_excluded": candidate.inventory_excluded,
                "converged": False,
                "notes": f"reference={reference_name}; {reference_note}; failed: {error}",
            }
        )
        return row

    def _build_viability_config(self, spectral_max_iter: int | None) -> LeontiefPropagationConfig:
        """Use a bounded spectral iteration budget for orientation comparisons."""
        max_iter = int(spectral_max_iter) if spectral_max_iter is not None else min(
            int(self.leontief_config.spectral_radius_max_iter),
            80,
        )
        return replace(self.leontief_config, spectral_radius_max_iter=max_iter)

    def _final_demand_for_candidate(self, fd_matrix: pd.DataFrame, candidate: OrientationCandidate) -> pd.DataFrame:
        if not candidate.inventory_excluded:
            return fd_matrix.copy()
        mask = inventory_fd_column_mask(list(fd_matrix.columns), self.leontief_config.inventory_label_patterns)
        if not mask.any():
            LOGGER.warning("No inventory FD columns matched configured patterns for orientation audit.")
        return fd_matrix.loc[:, ~mask].copy()

    def _load_fd_labels(self, year: int, fd_matrix: pd.DataFrame) -> list[str]:
        path = self.paths.label_file(year, "labels_FD")
        if not path.exists():
            LOGGER.warning("labels_FD missing for %s; using FD parquet columns.", year)
            return [str(column) for column in fd_matrix.columns]
        labels = load_labels_file(path)
        if len(labels) != fd_matrix.shape[1]:
            LOGGER.warning(
                "labels_FD count %s does not match FD columns for %s; using FD parquet columns.",
                len(labels),
                year,
            )
            return [str(column) for column in fd_matrix.columns]
        return labels

    def _load_panel_reference(self, year: int, labels: list[str]) -> pd.Series | None:
        path = self.paths.abm_v3_historical_panel_file(
            self.config.calibration.start_year,
            self.config.calibration.end_year,
        )
        if not path.exists():
            LOGGER.warning("ABM-ready panel reference missing at %s.", path)
            return None
        panel = pd.read_parquet(path)
        required = {"Year", "country_sector", "X_observed"}
        missing = required.difference(panel.columns)
        if missing:
            LOGGER.warning("ABM-ready panel lacks reference columns: %s", sorted(missing))
            return None
        year_panel = panel.loc[panel["Year"].astype(int) == int(year), ["country_sector", "X_observed"]].copy()
        if year_panel.empty:
            LOGGER.warning("ABM-ready panel has no rows for %s.", year)
            return None
        series = pd.Series(
            pd.to_numeric(year_panel["X_observed"], errors="coerce").to_numpy(dtype=float),
            index=year_panel["country_sector"].astype(str),
            name="X_abm_ready_panel",
        )
        return series.reindex(labels)

    def _select_reference(
        self,
        requested_reference: str,
        current_x: np.ndarray,
        panel_reference: pd.Series | None,
        labels: list[str],
    ) -> tuple[pd.Series, str, str]:
        current_series = pd.Series(current_x, index=labels, name="X_current_column_output_reference")
        current_note = "current reference is T.sum(axis=0)+FD.sum(axis=1)"
        if requested_reference == "current":
            return current_series, "current", current_note
        if panel_reference is not None and panel_reference.notna().any():
            note = (
                "ABM-ready panel X_observed used; note this panel was built from the current "
                "column-output convention in input_panel_builder."
            )
            return panel_reference.rename("X_abm_ready_panel_reference"), "abm_ready", note
        note = f"requested abm_ready reference unavailable; fallback used; {current_note}"
        return current_series, "current_fallback", note


def orientation_candidate_from_mode(mode: str) -> OrientationCandidate:
    """Parse a public orientation mode into explicit construction choices."""
    if mode not in ALL_ORIENTATION_MODES:
        allowed = ", ".join(ALL_ORIENTATION_MODES)
        raise ValueError(f"Unknown Leontief orientation mode '{mode}'. Allowed modes: {allowed}")
    inventory_excluded = mode.endswith("_fd_without_inventory")
    base_mode = mode.replace("_fd_without_inventory", "")
    if base_mode == "current_column_output":
        return OrientationCandidate(mode, inventory_excluded, "column", "T")
    if base_mode in {"row_output_standard_io", "row_output"}:
        return OrientationCandidate(mode, inventory_excluded, "row", "T")
    if base_mode in {"transpose_row_output", "transpose_with_row_output"}:
        return OrientationCandidate(mode, inventory_excluded, "row", "T.T")
    raise ValueError(f"Unhandled Leontief orientation mode '{mode}'")


def build_orientation_output(mode: str, t_matrix: pd.DataFrame, fd_matrix: pd.DataFrame) -> np.ndarray:
    """Build candidate gross output X for an orientation mode."""
    candidate = orientation_candidate_from_mode(mode)
    y_values = fd_matrix.sum(axis=1).to_numpy(dtype=float)
    if candidate.output_basis == "column":
        intermediate_values = t_matrix.sum(axis=0).to_numpy(dtype=float)
    elif candidate.output_basis == "row":
        intermediate_values = t_matrix.sum(axis=1).to_numpy(dtype=float)
    else:
        raise ValueError(f"Unhandled output basis '{candidate.output_basis}'")
    return (intermediate_values + y_values).astype(float)


def build_orientation_coefficients(mode: str, t_matrix: pd.DataFrame, fd_matrix: pd.DataFrame) -> sparse.spmatrix:
    """Build A by dividing candidate T columns by candidate gross output X."""
    candidate = orientation_candidate_from_mode(mode)
    x_values = build_orientation_output(mode, t_matrix, fd_matrix)
    coefficient_source = t_matrix.T if candidate.coefficient_source == "T.T" else t_matrix
    source_sparse = sparse.csc_matrix(coefficient_source.to_numpy(dtype=float, copy=True))
    inverse_output = np.zeros_like(x_values, dtype=float)
    valid = np.isfinite(x_values) & (x_values > 0.0)
    inverse_output[valid] = 1.0 / x_values[valid]
    coefficient_matrix = source_sparse @ sparse.diags(inverse_output, format="csc")
    coefficient_matrix.eliminate_zeros()
    return coefficient_matrix.tocsr()


def load_labels_file(path: object) -> list[str]:
    """Load tab-separated Eora labels as canonical country-sector strings."""
    labels: list[str] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [part.strip() for part in line.split("\t") if part.strip()]
        if parts:
            labels.append(" | ".join(parts))
    if not labels:
        raise ValueError(f"No labels found in {path}")
    return labels


def split_country_sector_labels(labels: list[str]) -> pd.DataFrame:
    """Split canonical country-sector labels into inspectable components."""
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


def validate_matrix_alignment(
    year: int,
    labels: list[str],
    t_matrix: pd.DataFrame,
    fd_matrix: pd.DataFrame,
) -> list[str]:
    """Check dimensions and visible labels before overwriting matrix axes."""
    notes: list[str] = []
    label_count = len(labels)
    if t_matrix.shape != (label_count, label_count):
        raise ValueError(f"labels_T count {label_count} does not match square T dimensions for {year}: T={t_matrix.shape}")
    if fd_matrix.shape[0] != label_count:
        raise ValueError(f"labels_T count {label_count} does not match FD rows for {year}: FD={fd_matrix.shape}")
    notes.append("T rows, T columns, and FD rows match labels_T dimensions")
    for axis_name, observed_axis in [
        ("T rows", list(t_matrix.index)),
        ("T columns", list(t_matrix.columns)),
        ("FD rows", list(fd_matrix.index)),
    ]:
        observed_strings = [str(value) for value in observed_axis]
        overlap = sum(1 for value in observed_strings if value in set(labels))
        if overlap == 0:
            notes.append(f"{axis_name} had no visible labels matching labels_T; labels_T imposed by position")
        elif observed_strings == labels:
            notes.append(f"{axis_name} already aligned with labels_T")
        else:
            notes.append(f"WARNING: {axis_name} visible labels do not align with labels_T; labels_T imposed by position")
            LOGGER.warning("%s visible labels do not align with labels_T for %s.", axis_name, year)
    return notes


def inventory_fd_column_mask(fd_columns: list[object], patterns: tuple[str, ...]) -> np.ndarray:
    """Identify FD inventory columns from explicit label patterns."""
    lowered_patterns = tuple(pattern.lower() for pattern in patterns)
    return np.array(
        [any(pattern in str(column).lower() for pattern in lowered_patterns) for column in fd_columns],
        dtype=bool,
    )


def safe_divide_array(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide arrays while returning NaN for zero or invalid denominators."""
    numerator_values = np.asarray(numerator, dtype=float)
    denominator_values = np.asarray(denominator, dtype=float)
    result = np.full_like(numerator_values, np.nan, dtype=float)
    valid = np.isfinite(numerator_values) & np.isfinite(denominator_values) & (denominator_values != 0.0)
    result[valid] = numerator_values[valid] / denominator_values[valid]
    return result


def orientation_summary_columns() -> list[str]:
    """Stable column order for orientation summary diagnostics."""
    return [
        "Year",
        "orientation_mode",
        "Y_total",
        "intermediate_row_sum_total",
        "intermediate_column_sum_total",
        "X_total",
        "X_min",
        "X_p01",
        "X_median",
        "X_p99",
        "X_max",
        "non_positive_X_count",
        "near_zero_positive_X_count",
        "negative_Y_count",
        "negative_FD_entries",
        "inventory_excluded",
        "spectral_radius_A",
        "spectral_radius_abs_A",
        "spectral_radius_above_one",
        "max_abs_column_sum_A",
        "high_abs_column_sum_count",
        "suspicious_column_count",
        "converged",
        "rounds_used",
        "final_residual_share",
        "observed_output_total_reference",
        "iterative_output_total",
        "relative_error_total",
        "correlation_iterative_vs_reference",
        "mean_absolute_percentage_error",
        "notes",
    ]
