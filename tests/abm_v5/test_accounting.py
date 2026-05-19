from pathlib import Path

import pytest

from src.abm_v5 import (
    AccountingBuildResult,
    CO2_ROW_LABEL,
    ValidationStatus,
    build_accounting_records_for_year,
    build_accounting_state_panel,
    compute_local_greenness,
    load_accounting_state_panel,
    validate_accounting_state_panel,
)


LABELS = [
    "AAA\tAAA\tIndustries\tAgriculture\t",
    "BBB\tBBB\tIndustries\tManufacturing\t",
]


def write_pyproject(root: Path) -> None:
    (root / "pyproject.toml").write_text("[project]\nname = 'toy'\n", encoding="utf-8")


def write_fake_year(root: Path, year: int, include_co2: bool = True) -> None:
    import polars as pl

    node_columns = [
        "AAA | AAA | Industries | Agriculture",
        "BBB | BBB | Industries | Manufacturing",
    ]
    matrix_dir = root / "data" / "parquet" / str(year)
    raw_dir = root / "data" / "raw" / str(year)
    matrix_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            node_columns[0]: [1.0, 3.0],
            node_columns[1]: [2.0, 4.0],
        }
    ).write_parquet(matrix_dir / "T.parquet")
    pl.DataFrame(
        {
            "FD1": [10.0, 20.0],
            "FD2": [1.0, 2.0],
        }
    ).write_parquet(matrix_dir / "FD.parquet")
    pl.DataFrame(
        {
            node_columns[0]: [1.0, 140.0],
            node_columns[1]: [1.0, 58.0],
        }
    ).write_parquet(matrix_dir / "Q.parquet")
    raw_dir.joinpath("labels_T.txt").write_text("\n".join(LABELS) + "\n", encoding="utf-8")
    labels_q = ["Other row\tTotal\t"]
    if include_co2:
        labels_q.append("Total CO2 emissions (Gg) from EDGAR\tTotal\t")
    else:
        labels_q.append("Not CO2\tTotal\t")
    raw_dir.joinpath("labels_Q.txt").write_text("\n".join(labels_q) + "\n", encoding="utf-8")


def write_identity(root: Path) -> None:
    import polars as pl

    identity_path = root / "data" / "abm_v5" / "inputs" / "agent_identity.parquet"
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "country_sector": [label.strip() for label in LABELS],
            "country": ["AAA", "BBB"],
            "country_detail": ["AAA", "BBB"],
            "category": ["Industries", "Industries"],
            "sector": ["Agriculture", "Manufacturing"],
        }
    ).write_parquet(identity_path)


def test_compute_local_greenness_returns_values_between_zero_and_one() -> None:
    values = compute_local_greenness([10.0, 2.0, 5.0]).drop_nulls().to_list()

    assert min(values) >= 0.0
    assert max(values) <= 1.0


def test_compute_local_greenness_handles_equal_values() -> None:
    values = compute_local_greenness([2.0, 2.0]).to_list()

    assert values == [0.5, 0.5]


def test_compute_local_greenness_rejects_nonpositive_ei_as_null() -> None:
    values = compute_local_greenness([1.0, 0.0, -1.0, None]).to_list()

    assert values[0] == 0.5
    assert values[1:] == [None, None, None]


def test_build_accounting_records_for_year_extracts_co2_row(tmp_path: Path) -> None:
    write_fake_year(tmp_path, 1995)

    records = build_accounting_records_for_year(tmp_path, 1995)

    assert records[0]["output"] == 14.0
    assert records[0]["final_demand"] == 11.0
    assert records[0]["emissions"] == 140.0
    assert records[0]["emissions_intensity"] == 10.0
    assert CO2_ROW_LABEL == "Total CO2 emissions (Gg) from EDGAR | Total"


def test_build_accounting_records_for_year_raises_if_co2_row_missing(tmp_path: Path) -> None:
    write_fake_year(tmp_path, 1995, include_co2=False)

    with pytest.raises(ValueError, match="CO2 row label"):
        build_accounting_records_for_year(tmp_path, 1995)


def test_validate_accounting_state_panel_requires_columns() -> None:
    import polars as pl

    results = validate_accounting_state_panel(pl.DataFrame({"country_sector": ["A"]}))

    assert results[0].status is ValidationStatus.FAILED


def test_validate_accounting_state_panel_detects_duplicate_keys() -> None:
    import polars as pl

    frame = pl.DataFrame(
        {
            "country_sector": ["A", "A"],
            "year": [1995, 1995],
            "output": [1.0, 1.0],
            "final_demand": [1.0, 1.0],
            "emissions": [1.0, 1.0],
            "emissions_intensity": [1.0, 1.0],
            "local_greenness": [0.5, 0.5],
            "accounting_output_positive_flag": [True, True],
            "accounting_emissions_nonnegative_flag": [True, True],
            "accounting_ei_valid_flag": [True, True],
        }
    )

    results = validate_accounting_state_panel(frame)

    assert any(result.check_name == "accounting_unique_country_sector_year" and result.status is ValidationStatus.FAILED for result in results)


def test_validate_accounting_state_panel_checks_greenness_range() -> None:
    import polars as pl

    frame = pl.DataFrame(
        {
            "country_sector": ["A"],
            "year": [1995],
            "output": [1.0],
            "final_demand": [1.0],
            "emissions": [1.0],
            "emissions_intensity": [1.0],
            "local_greenness": [1.5],
            "accounting_output_positive_flag": [True],
            "accounting_emissions_nonnegative_flag": [True],
            "accounting_ei_valid_flag": [True],
        }
    )

    results = validate_accounting_state_panel(frame)

    assert any(result.check_name == "accounting_local_greenness_range" and result.status is ValidationStatus.FAILED for result in results)


def test_build_accounting_state_panel_requires_agent_identity(tmp_path: Path) -> None:
    write_pyproject(tmp_path)

    with pytest.raises(FileNotFoundError, match="Phase 2.2"):
        build_accounting_state_panel(tmp_path)


def test_build_accounting_state_panel_writes_outputs_with_fake_data(tmp_path: Path) -> None:
    write_pyproject(tmp_path)
    write_identity(tmp_path)
    for year in range(1995, 2017):
        write_fake_year(tmp_path, year)

    result = build_accounting_state_panel(tmp_path)
    panel = load_accounting_state_panel(result.output_path)

    assert isinstance(result, AccountingBuildResult)
    assert result.output_path.is_file()
    assert result.validation_path.is_file()
    assert result.n_rows == 44
    assert result.n_agents == 2
    assert {"country", "country_detail", "category", "sector"}.issubset(set(panel.columns))
    assert panel["emissions_intensity"].null_count() == 0


def test_init_exports_accounting_objects() -> None:
    import src.abm_v5 as abm_v5

    assert abm_v5.AccountingBuildResult is AccountingBuildResult
    assert abm_v5.compute_local_greenness is compute_local_greenness
    assert abm_v5.build_accounting_records_for_year is build_accounting_records_for_year
    assert abm_v5.build_accounting_state_panel is build_accounting_state_panel
    assert abm_v5.load_accounting_state_panel is load_accounting_state_panel
    assert abm_v5.validate_accounting_state_panel is validate_accounting_state_panel
