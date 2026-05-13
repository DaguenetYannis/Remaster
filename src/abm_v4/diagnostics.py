from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.abm_v4.paths import ABMV4Paths


@dataclass(frozen=True)
class DiagnosticMessage:
    """Structured diagnostic message for inspectable ABM v4 workflows."""

    step: str
    level: str
    message: str


@dataclass(frozen=True)
class PathAuditRow:
    """One non-mutating ABM v4 source-path audit row."""

    logical_source: str
    candidate_paths_checked: tuple[Path, ...]
    found_path: Path | None
    status: str
    consequence_if_missing: str

    def format_candidate_paths(self) -> str:
        """Return candidate paths as a compact printable string."""
        return "; ".join(str(candidate_path) for candidate_path in self.candidate_paths_checked)

    def format_found_path(self) -> str:
        """Return the found path or a visible missing marker."""
        if self.found_path is None:
            return "-"
        return str(self.found_path)


def _first_existing_path(candidate_paths: tuple[Path, ...]) -> Path | None:
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path
    return None


def _audit_row(
    logical_source: str,
    candidate_paths: tuple[Path, ...],
    consequence_if_missing: str,
    required: bool = True,
) -> PathAuditRow:
    found_path = _first_existing_path(candidate_paths)
    if found_path is not None:
        status = "found"
    elif required:
        status = "missing_required"
    else:
        status = "missing_optional"

    return PathAuditRow(
        logical_source=logical_source,
        candidate_paths_checked=candidate_paths,
        found_path=found_path,
        status=status,
        consequence_if_missing=consequence_if_missing,
    )


def build_path_audit_rows(
    paths: ABMV4Paths,
    start_year: int,
    end_year: int,
) -> tuple[PathAuditRow, ...]:
    """Check ABM v4 real-data source availability without creating outputs."""
    return (
        _audit_row(
            logical_source="ABM v3 output root",
            candidate_paths=(paths.data_abm_v3,),
            consequence_if_missing="ABM v4 cannot prefer current v3 panels and must use lower-priority sources.",
        ),
        _audit_row(
            logical_source="ABM v3 state/input panels",
            candidate_paths=paths.abm_v3_state_candidates(start_year, end_year),
            consequence_if_missing="State construction must fall back to final panels or legacy ABM data.",
        ),
        _audit_row(
            logical_source="ABM v3 Leontief or edge structures",
            candidate_paths=paths.edge_candidates,
            consequence_if_missing="Supplier adaptation must stop cleanly; do not invent supplier edges.",
        ),
        _audit_row(
            logical_source="Merged Eora-Atlas panel",
            candidate_paths=(paths.data_final / "eora_atlas_merged.parquet",),
            consequence_if_missing="State construction loses the merged final-panel fallback.",
        ),
        _audit_row(
            logical_source="Dynamic Eora-Atlas panel",
            candidate_paths=(paths.data_final / "eora_atlas_dynamic_panel.parquet",),
            consequence_if_missing="State construction loses dynamic variables and lag/change fallbacks.",
        ),
        _audit_row(
            logical_source="Atlas Eora26 capabilities",
            candidate_paths=(paths.data_atlas / "processed" / "atlas_eora26_sector_capabilities_1995_2016.parquet",),
            consequence_if_missing="Capability fields and ecosystem assignment must be diagnosed as unavailable.",
        ),
        _audit_row(
            logical_source="Atlas concordance directory",
            candidate_paths=(paths.data_atlas / "concordance",),
            consequence_if_missing="HS92 dominant-cluster ecosystem fallback is unavailable.",
            required=False,
        ),
        _audit_row(
            logical_source="Eora metrics directory",
            candidate_paths=(paths.data_metrics,),
            consequence_if_missing="Metric-derived EI, centrality, and network green exposure sources are unavailable.",
        ),
        _audit_row(
            logical_source="Labelled Eora parquet directory",
            candidate_paths=(paths.data_root / "parquet",),
            consequence_if_missing="Matrix-based reconstruction is unavailable.",
        ),
    )


def format_path_audit_table(rows: tuple[PathAuditRow, ...]) -> str:
    """Format path audit rows as a readable console table."""
    headers = (
        "logical source",
        "candidate paths checked",
        "found path",
        "status",
        "consequence if missing",
    )
    table_rows = [
        (
            row.logical_source,
            row.format_candidate_paths(),
            row.format_found_path(),
            row.status,
            row.consequence_if_missing,
        )
        for row in rows
    ]
    widths = [
        max(len(headers[column_index]), *(len(row[column_index]) for row in table_rows))
        for column_index in range(len(headers))
    ]

    def format_row(values: tuple[str, ...]) -> str:
        return " | ".join(
            value.ljust(widths[index])
            for index, value in enumerate(values)
        )

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join(
        (
            format_row(headers),
            separator,
            *(format_row(row) for row in table_rows),
        )
    )
