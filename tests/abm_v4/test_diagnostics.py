from pathlib import Path

from src.abm_v4.diagnostics import build_path_audit_rows, format_path_audit_table
from src.abm_v4.paths import ABMV4Paths


def test_path_audit_does_not_create_abm_v4_outputs() -> None:
    root = Path("tmp") / "abm_v4_dry_run_tests" / "no_create"
    paths = ABMV4Paths(project_root=root)

    rows = build_path_audit_rows(paths, 1995, 2016)

    assert rows
    assert not paths.data_abm_v4.exists()


def test_path_audit_reports_found_and_missing_sources() -> None:
    root = Path("tmp") / "abm_v4_dry_run_tests" / "mixed_sources"
    paths = ABMV4Paths(project_root=root)
    (root / "data" / "abm_v3").mkdir(parents=True, exist_ok=True)
    (root / "data" / "final").mkdir(parents=True, exist_ok=True)
    (root / "data" / "final" / "eora_atlas_merged.parquet").write_text("toy")

    rows = build_path_audit_rows(paths, 1995, 2016)
    status_by_source = {row.logical_source: row.status for row in rows}

    assert status_by_source["ABM v3 output root"] == "found"
    assert status_by_source["Merged Eora-Atlas panel"] == "found"
    assert status_by_source["ABM v3 state/input panels"] == "missing_required"


def test_path_audit_table_contains_required_headers() -> None:
    rows = build_path_audit_rows(ABMV4Paths(project_root=Path("toy_project")), 1995, 2016)

    table = format_path_audit_table(rows)

    assert "logical source" in table
    assert "candidate paths checked" in table
    assert "consequence if missing" in table
