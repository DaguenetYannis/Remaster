from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from src.abm_v4.config import ABMV4Config
from src.abm_v4.paths import ABMV4Paths


@dataclass(frozen=True)
class ValidationMessage:
    """Structured validation message that does not hide uncertainty."""

    check_name: str
    passed: bool
    message: str


@dataclass(frozen=True)
class OneStepValidationThresholds:
    """Transparent pass/warn thresholds for the one-step base validation."""

    weight_sum_error_max: float = 1e-8
    aggregate_feasibility_min: float = 0.95
    high_constrained_node_share: float = 0.80
    decomposition_residual_max_abs: float = 1e-4
    high_invalid_ei_share: float = 0.05
    high_capability_fill_share: float = 0.25


@dataclass(frozen=True)
class OneStepBaseValidationResult:
    """Consolidated validation outputs for the one-step base model."""

    report: pl.DataFrame
    markdown: str
    status: dict[str, Any]

    @property
    def passed(self) -> bool:
        """Return True when no layer has a blocking issue."""
        return bool(self.status["overall_passed"])


def required_one_step_component_paths(
    paths: ABMV4Paths,
    config: ABMV4Config,
) -> dict[str, Path]:
    """Return component outputs needed for a reuse-based one-step validation."""
    return {
        "state_panel": paths.state_panel_path(config.start_year, config.end_year),
        "ecosystem_assignment_report": paths.ecosystem_assignment_report_path,
        "raw_t_supplier_edges": paths.raw_t_supplier_edges_path,
        "raw_t_supplier_edge_report": paths.raw_t_supplier_edge_report_path,
        "supplier_candidate_base_report": paths.supplier_candidate_base_report_path,
        "supplier_opportunity_set_report": paths.supplier_opportunity_set_report_path,
        "supplier_rewiring_report": paths.supplier_rewiring_report_path,
        "capability_update_report": paths.capability_update_report_path,
        "production_feasibility_report": paths.production_feasibility_report_path,
        "emissions_update_report": paths.emissions_update_report_path,
    }


def missing_one_step_component_paths(
    paths: ABMV4Paths,
    config: ABMV4Config,
) -> dict[str, Path]:
    """Return required component paths that are currently missing."""
    return {
        name: path
        for name, path in required_one_step_component_paths(paths, config).items()
        if not path.exists()
    }


def _read_first_csv_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    frame = pl.read_csv(path)
    if frame.is_empty():
        return {}
    return frame.to_dicts()[0]


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def _status_from_checks(passed: bool, warnings: list[str], blocking: list[str]) -> str:
    if not passed or blocking:
        return "fail"
    if warnings:
        return "warning"
    return "pass"


def _join_messages(messages: list[str]) -> str:
    return "; ".join(messages) if messages else ""


def _base_row(layer: str, passed: bool, warnings: list[str], blocking: list[str]) -> dict[str, Any]:
    return {
        "layer": layer,
        "status": _status_from_checks(passed, warnings, blocking),
        "passed": passed and not blocking,
        "warnings": _join_messages(warnings),
        "blocking_issues": _join_messages(blocking),
    }


def _build_state_row(paths: ABMV4Paths, config: ABMV4Config) -> dict[str, Any]:
    state_path = paths.state_panel_path(config.start_year, config.end_year)
    warnings: list[str] = []
    blocking: list[str] = []
    required_columns = {"country_sector", "Year", "X_observed", "EI", "Sector"}
    state = pl.read_parquet(state_path) if state_path.exists() else pl.DataFrame()
    source_report = _read_first_csv_row(paths.diagnostics / "state_source_report.csv")
    missing_required = sorted(required_columns - set(state.columns))
    if state.is_empty():
        blocking.append("State panel is missing or empty.")
    if missing_required:
        blocking.append(f"Missing required state columns: {', '.join(missing_required)}.")
    year_min = state["Year"].min() if "Year" in state.columns and not state.is_empty() else None
    year_max = state["Year"].max() if "Year" in state.columns and not state.is_empty() else None
    node_count = (
        state.select("country_sector").unique().height
        if "country_sector" in state.columns and not state.is_empty()
        else 0
    )
    passed = node_count > 0 and year_max is not None and year_max >= config.end_year and not blocking
    row = _base_row("state", passed, warnings, blocking)
    row.update(
        {
            "state_row_count": state.height,
            "year_coverage": f"{year_min}-{year_max}" if year_min is not None else "",
            "country_sector_node_count": node_count,
            "selected_source": source_report.get("selected_source", "existing_state_panel"),
            "missing_required_variables": ", ".join(missing_required),
        }
    )
    return row


def _build_ecosystem_row(paths: ABMV4Paths) -> dict[str, Any]:
    report = _read_first_csv_row(paths.ecosystem_assignment_report_path)
    warnings: list[str] = []
    blocking: list[str] = []
    unmapped = _as_int(report.get("unmapped_nodes"))
    if not report:
        blocking.append("Ecosystem assignment report is missing.")
    if unmapped != 0:
        blocking.append(f"{unmapped} nodes are unmapped.")
    passed = bool(report) and unmapped == 0
    row = _base_row("ecosystem", passed, warnings, blocking)
    row.update(
        {
            "ecosystem_source": report.get("ecosystem_source_counts", ""),
            "mapped_node_count": report.get("mapped_nodes"),
            "unmapped_node_count": report.get("unmapped_nodes"),
            "ecosystem_count": report.get("number_of_ecosystems"),
        }
    )
    return row


def _build_supplier_row(paths: ABMV4Paths, thresholds: OneStepValidationThresholds) -> dict[str, Any]:
    edge = _read_first_csv_row(paths.raw_t_supplier_edge_report_path)
    candidates = _read_first_csv_row(paths.supplier_candidate_base_report_path)
    opportunities = _read_first_csv_row(paths.supplier_opportunity_set_report_path)
    rewiring = _read_first_csv_row(paths.supplier_rewiring_report_path)
    warnings: list[str] = []
    blocking: list[str] = []
    if not edge:
        blocking.append("Raw T supplier edge report is missing.")
    if not candidates:
        blocking.append("Supplier candidate base report is missing.")
    if not opportunities:
        blocking.append("Supplier opportunity report is missing.")
    if not rewiring:
        blocking.append("Supplier rewiring report is missing.")
    max_initial_error = _as_float(rewiring.get("max_initial_weight_sum_error"))
    max_updated_error = _as_float(rewiring.get("max_updated_weight_sum_error"))
    if max_updated_error >= thresholds.weight_sum_error_max:
        blocking.append(f"Updated supplier weight sum error is {max_updated_error}.")
    if _as_float(rewiring.get("rewired_buyer_share")) == 0:
        warnings.append("Rewired buyer share is zero.")
    number_of_buyers = max(_as_float(rewiring.get("number_of_buyers")), 1.0)
    fallback_stress_share = _as_float(rewiring.get("fallback_stress_buyers")) / number_of_buyers
    if fallback_stress_share > 0.5:
        warnings.append("Fallback stress was used for most buyers.")
    passed = not blocking
    row = _base_row("supplier", passed, warnings, blocking)
    row.update(
        {
            "raw_t_edge_source_status": edge.get("selected_source", "missing"),
            "historical_candidate_rows": candidates.get("historical_candidate_rows"),
            "same_sector_candidate_rows": candidates.get("same_sector_candidate_rows"),
            "ecosystem_candidate_rows": candidates.get("ecosystem_candidate_rows"),
            "opportunity_rows": opportunities.get("opportunity_rows"),
            "median_candidates_per_buyer": opportunities.get("median_candidates_per_buyer"),
            "rewired_buyer_share": rewiring.get("rewired_buyer_share"),
            "max_initial_weight_sum_error": max_initial_error,
            "max_updated_weight_sum_error": max_updated_error,
        }
    )
    return row


def _build_capability_row(paths: ABMV4Paths, thresholds: OneStepValidationThresholds) -> dict[str, Any]:
    report = _read_first_csv_row(paths.capability_update_report_path)
    warnings: list[str] = []
    blocking: list[str] = []
    if not report:
        blocking.append("Capability update report is missing.")
    if _as_int(report.get("cap_clipped_count")) != 0 or _as_int(report.get("gcap_clipped_count")) != 0:
        blocking.append("Capability clipping was detected.")
    general_fill = _as_float(report.get("share_general_capability_filled"))
    green_fill = _as_float(report.get("share_green_capability_filled"))
    if max(general_fill, green_fill) > thresholds.high_capability_fill_share:
        warnings.append("Capability fill share is above threshold.")
    passed = not blocking
    row = _base_row("capability", passed, warnings, blocking)
    row.update(
        {
            "selected_year": report.get("year"),
            "mean_cap": report.get("mean_cap"),
            "mean_gcap": report.get("mean_gcap"),
            "mean_exposure_cap": report.get("mean_exposure_cap"),
            "mean_exposure_gcap": report.get("mean_exposure_gcap"),
            "mean_delta_cap": report.get("mean_delta_cap"),
            "mean_delta_gcap": report.get("mean_delta_gcap"),
            "share_general_capability_filled": general_fill,
            "share_green_capability_filled": green_fill,
        }
    )
    return row


def _build_production_row(paths: ABMV4Paths, thresholds: OneStepValidationThresholds) -> dict[str, Any]:
    report = _read_first_csv_row(paths.production_feasibility_report_path)
    warnings: list[str] = []
    blocking: list[str] = []
    if not report:
        blocking.append("Production feasibility report is missing.")
    aggregate_feasibility = _as_float(report.get("aggregate_feasibility_ratio"))
    constrained_share = _as_float(report.get("share_nodes_with_input_feasibility_below_1"))
    if aggregate_feasibility <= thresholds.aggregate_feasibility_min:
        blocking.append(f"Aggregate feasibility ratio is {aggregate_feasibility}.")
    if constrained_share > thresholds.high_constrained_node_share and aggregate_feasibility > thresholds.aggregate_feasibility_min:
        warnings.append("Many nodes are marginally constrained despite high aggregate feasibility.")
    passed = not blocking
    row = _base_row("production", passed, warnings, blocking)
    row.update(
        {
            "selected_year": report.get("year"),
            "aggregate_feasibility_ratio": aggregate_feasibility,
            "mean_input_feasibility": report.get("mean_input_feasibility"),
            "constrained_node_share": constrained_share,
            "p95_supplier_pressure_max": report.get("p95_supplier_pressure_max"),
            "share_nodes_with_supplier_pressure_above_1": report.get("share_nodes_with_supplier_pressure_above_1"),
        }
    )
    return row


def _build_emissions_row(paths: ABMV4Paths, thresholds: OneStepValidationThresholds) -> dict[str, Any]:
    report = _read_first_csv_row(paths.emissions_update_report_path)
    warnings: list[str] = []
    blocking: list[str] = []
    if not report:
        blocking.append("Emissions update report is missing.")
    residual = abs(_as_float(report.get("decomposition_residual")))
    node_count = max(_as_float(report.get("node_count")), 1.0)
    invalid_share = _as_float(report.get("invalid_EI_nodes")) / node_count
    if residual >= thresholds.decomposition_residual_max_abs:
        blocking.append(f"Emissions decomposition residual is {residual}.")
    if _as_bool(report.get("bad_transition_flag")):
        blocking.append("Bad transition flag is true.")
    if invalid_share > thresholds.high_invalid_ei_share:
        warnings.append("Invalid EI share is above threshold.")
    passed = not blocking
    row = _base_row("emissions", passed, warnings, blocking)
    row.update(
        {
            "transition_mode": report.get("emissions_transition_mode"),
            "valid_EI_nodes": report.get("valid_EI_nodes"),
            "invalid_EI_nodes": report.get("invalid_EI_nodes"),
            "mean_rEI_used": report.get("mean_rEI_used"),
            "median_rEI_used": report.get("median_rEI_used"),
            "aggregate_delta_emissions": report.get("aggregate_delta_emissions"),
            "decomposition_residual": report.get("decomposition_residual"),
            "bad_transition_flag": report.get("bad_transition_flag"),
        }
    )
    return row


def build_one_step_base_validation_report(
    paths: ABMV4Paths,
    config: ABMV4Config,
    thresholds: OneStepValidationThresholds | None = None,
) -> OneStepBaseValidationResult:
    """Aggregate existing one-step ABM v4 diagnostics into one validation report."""
    thresholds = thresholds or OneStepValidationThresholds()
    rows = [
        _build_state_row(paths, config),
        _build_ecosystem_row(paths),
        _build_supplier_row(paths, thresholds),
        _build_capability_row(paths, thresholds),
        _build_production_row(paths, thresholds),
        _build_emissions_row(paths, thresholds),
    ]
    report = pl.DataFrame(rows)
    blocking_issues = [
        row["blocking_issues"] for row in rows if row.get("blocking_issues")
    ]
    warnings = [row["warnings"] for row in rows if row.get("warnings")]
    failed_layers = [row["layer"] for row in rows if row["status"] == "fail"]
    warning_layers = [row["layer"] for row in rows if row["status"] == "warning"]
    overall_passed = not failed_layers
    status = {
        "overall_status": "pass" if overall_passed and not warning_layers else ("warning" if overall_passed else "fail"),
        "overall_passed": overall_passed,
        "failed_layers": failed_layers,
        "warning_layers": warning_layers,
        "warnings": warnings,
        "blocking_issues": blocking_issues,
        "recommended_next_phase": (
            "Phase 9: multi-year base simulation design"
            if overall_passed
            else "Resolve blocking one-step validation issues before multi-year simulation."
        ),
    }
    return OneStepBaseValidationResult(
        report=report,
        markdown=format_one_step_validation_markdown(report, status),
        status=status,
    )


def format_one_step_validation_markdown(
    report: pl.DataFrame,
    status: dict[str, Any],
) -> str:
    """Render the one-step validation report as a compact Markdown summary."""
    lines = [
        "# ABM v4 One-Step Base Validation",
        "",
        f"Overall status: **{status['overall_status']}**",
        "",
        "| Layer | Status | Warnings | Blocking issues |",
        "| --- | --- | --- | --- |",
    ]
    for row in report.to_dicts():
        lines.append(
            "| {layer} | {status} | {warnings} | {blocking} |".format(
                layer=row["layer"],
                status=row["status"],
                warnings=row.get("warnings") or "",
                blocking=row.get("blocking_issues") or "",
            )
        )
    lines.extend(
        [
            "",
            "## Recommended Next Phase",
            "",
            str(status["recommended_next_phase"]),
        ]
    )
    if status["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in status["warnings"])
    if status["blocking_issues"]:
        lines.extend(["", "## Blocking Issues", ""])
        lines.extend(f"- {issue}" for issue in status["blocking_issues"])
    return "\n".join(lines) + "\n"


def write_one_step_base_validation_outputs(
    paths: ABMV4Paths,
    result: OneStepBaseValidationResult,
) -> None:
    """Write one-step validation outputs under data/abm_v4/validation."""
    paths.validation.mkdir(parents=True, exist_ok=True)
    result.report.write_csv(paths.one_step_base_validation_report_csv_path)
    paths.one_step_base_validation_report_md_path.write_text(result.markdown, encoding="utf-8")
    paths.one_step_base_status_json_path.write_text(
        json.dumps(result.status, indent=2),
        encoding="utf-8",
    )
