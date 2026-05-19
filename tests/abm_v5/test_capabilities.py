from pathlib import Path

import pytest

from src.abm_v5 import (
    CapabilityBuildResult,
    ValidationStatus,
    build_capability_state_panel,
    inspect_capability_source_columns,
    load_capability_state_panel,
    normalize_capability_source_columns,
    summarize_capability_missingness,
    validate_capability_state_panel,
)


IDENTITY_ROWS = {
    "country_sector": ["A | A | Industries | One", "B | B | Industries | Two"],
    "country": ["A", "B"],
    "country_detail": ["A", "B"],
    "category": ["Industries", "Industries"],
    "sector": ["One", "Two"],
}


def write_pyproject(root: Path) -> None:
    (root / "pyproject.toml").write_text("[project]\nname = 'toy'\n", encoding="utf-8")


def write_identity(root: Path) -> None:
    import polars as pl

    path = root / "data" / "abm_v5" / "inputs" / "agent_identity.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(IDENTITY_ROWS).write_parquet(path)


def write_accounting(root: Path) -> None:
    import polars as pl

    rows = []
    for year in range(1995, 2017):
        rows.extend(
            [
                {
                    **{key: values[0] for key, values in IDENTITY_ROWS.items()},
                    "year": year,
                    "output": 10.0,
                    "final_demand": 1.0,
                    "emissions": 5.0,
                    "emissions_intensity": 0.5,
                    "local_greenness": 1.0,
                    "accounting_output_positive_flag": True,
                    "accounting_emissions_nonnegative_flag": True,
                    "accounting_ei_valid_flag": True,
                },
                {
                    **{key: values[1] for key, values in IDENTITY_ROWS.items()},
                    "year": year,
                    "output": -1.0,
                    "final_demand": 1.0,
                    "emissions": 5.0,
                    "emissions_intensity": None,
                    "local_greenness": None,
                    "accounting_output_positive_flag": False,
                    "accounting_emissions_nonnegative_flag": True,
                    "accounting_ei_valid_flag": False,
                },
            ]
        )
    path = root / "data" / "abm_v5" / "accounting" / "accounting_state_panel_1995_2016.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path)


def write_atlas(root: Path, include_design_targets: bool = True) -> None:
    import polars as pl

    rows = []
    for year in range(1995, 2017):
        rows.append(
            {
                "Country_Sector": "A | A | Industries | One",
                "Year": year,
                "complexity": 1.0,
                "green_capability_export_share": 0.2,
                **(
                    {
                        "relatedness_density": 0.3,
                        "green_density": 0.4,
                        "proximity_to_green": 0.5,
                        "green_precedence": 0.6,
                        "reachable_complexity": 0.7,
                        "transition_score": 0.8,
                    }
                    if include_design_targets
                    else {}
                ),
            }
        )
    path = (
        root
        / "data"
        / "atlas"
        / "processed"
        / "atlas_eora26_sector_capabilities_1995_2016.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path)


def test_inspect_capability_source_columns_returns_columns(tmp_path: Path) -> None:
    write_atlas(tmp_path)

    info = inspect_capability_source_columns(
        tmp_path / "data" / "atlas" / "processed" / "atlas_eora26_sector_capabilities_1995_2016.parquet"
    )

    assert "Country_Sector" in info["columns"]
    assert "Country_Sector" in info["candidate_identity_columns"]
    assert "complexity" in info["candidate_capability_columns"]


def test_normalize_capability_source_columns_maps_known_variants() -> None:
    import polars as pl

    normalized = normalize_capability_source_columns(
        pl.DataFrame(
            {
                "Country_Sector": ["A"],
                "Year": [1995],
                "complexity": [1.0],
                "green_capability_export_share": [0.2],
                "relatedness_density": [0.3],
            }
        )
    )

    assert {"country_sector", "year", "general_capability", "green_capability", "capability_density"}.issubset(
        set(normalized.columns)
    )


def test_validate_capability_state_panel_requires_columns() -> None:
    import polars as pl

    results = validate_capability_state_panel(pl.DataFrame({"country_sector": ["A"]}))

    assert results[0].status is ValidationStatus.FAILED


def test_validate_capability_state_panel_detects_duplicate_keys() -> None:
    import polars as pl

    row = {
        "country_sector": "A",
        "country": "A",
        "country_detail": "A",
        "category": "Industries",
        "sector": "One",
        "year": 1995,
        "capability_source": "test",
        "accounting_output_positive_flag": True,
        "accounting_emissions_nonnegative_flag": True,
        "accounting_ei_valid_flag": True,
    }
    for column in (
        "general_capability",
        "green_capability",
        "capability_density",
        "green_capability_density",
        "ecosystem_proximity",
        "directed_green_precedence",
        "reachable_green_complexity",
        "transition_sector_score",
    ):
        row[column] = 1.0
        row[f"{column}_available_flag"] = True

    results = validate_capability_state_panel(pl.DataFrame([row, row]))

    assert any(
        result.check_name == "capability_unique_country_sector_year"
        and result.status is ValidationStatus.FAILED
        for result in results
    )


def test_validate_capability_state_panel_checks_availability_flags() -> None:
    import polars as pl

    write_rows = []
    row = {
        "country_sector": "A",
        "country": "A",
        "country_detail": "A",
        "category": "Industries",
        "sector": "One",
        "year": 1995,
        "capability_source": "test",
        "accounting_output_positive_flag": True,
        "accounting_emissions_nonnegative_flag": True,
        "accounting_ei_valid_flag": True,
    }
    for column in (
        "general_capability",
        "green_capability",
        "capability_density",
        "green_capability_density",
        "ecosystem_proximity",
        "directed_green_precedence",
        "reachable_green_complexity",
        "transition_sector_score",
    ):
        row[column] = None
        row[f"{column}_available_flag"] = True
    write_rows.append(row)

    results = validate_capability_state_panel(pl.DataFrame(write_rows))

    assert any(
        result.check_name == "capability_availability_flags_match_values"
        and result.status is ValidationStatus.FAILED
        for result in results
    )


def test_summarize_capability_missingness_returns_expected_keys() -> None:
    import polars as pl

    frame = pl.DataFrame(
        {
            "general_capability": [1.0, None],
            "green_capability": [None, None],
            "capability_density": [None, None],
            "green_capability_density": [None, None],
            "ecosystem_proximity": [None, None],
            "directed_green_precedence": [None, None],
            "reachable_green_complexity": [None, None],
            "transition_sector_score": [None, None],
        }
    )

    summary = summarize_capability_missingness(frame)

    assert summary["general_capability"] == 0.5
    assert set(summary) == {
        "general_capability",
        "green_capability",
        "capability_density",
        "green_capability_density",
        "ecosystem_proximity",
        "directed_green_precedence",
        "reachable_green_complexity",
        "transition_sector_score",
    }


def test_build_capability_state_panel_requires_identity_panel(tmp_path: Path) -> None:
    write_pyproject(tmp_path)

    with pytest.raises(FileNotFoundError, match="Phase 2.2"):
        build_capability_state_panel(tmp_path)


def test_build_capability_state_panel_requires_accounting_panel(tmp_path: Path) -> None:
    write_pyproject(tmp_path)
    write_identity(tmp_path)

    with pytest.raises(FileNotFoundError, match="Phase 2.3"):
        build_capability_state_panel(tmp_path)


def test_build_capability_state_panel_requires_atlas_source(tmp_path: Path) -> None:
    write_pyproject(tmp_path)
    write_identity(tmp_path)
    write_accounting(tmp_path)

    with pytest.raises(FileNotFoundError, match="Atlas capability source"):
        build_capability_state_panel(tmp_path)


def test_build_capability_state_panel_writes_outputs_with_fake_data(tmp_path: Path) -> None:
    write_pyproject(tmp_path)
    write_identity(tmp_path)
    write_accounting(tmp_path)
    write_atlas(tmp_path)

    result = build_capability_state_panel(tmp_path)
    panel = load_capability_state_panel(result.output_path)

    assert isinstance(result, CapabilityBuildResult)
    assert result.output_path.is_file()
    assert result.validation_path.is_file()
    assert result.n_rows == 44
    assert result.n_agents == 2
    assert panel["general_capability_available_flag"].sum() == 22


def test_build_capability_state_panel_preserves_accounting_flags(tmp_path: Path) -> None:
    write_pyproject(tmp_path)
    write_identity(tmp_path)
    write_accounting(tmp_path)
    write_atlas(tmp_path)

    result = build_capability_state_panel(tmp_path)
    panel = load_capability_state_panel(result.output_path)

    invalid_rows = panel.filter(panel["country_sector"] == "B | B | Industries | Two")
    assert invalid_rows["accounting_output_positive_flag"].sum() == 0
    assert invalid_rows["accounting_ei_valid_flag"].sum() == 0


def test_build_capability_state_panel_creates_missing_design_target_columns(tmp_path: Path) -> None:
    write_pyproject(tmp_path)
    write_identity(tmp_path)
    write_accounting(tmp_path)
    write_atlas(tmp_path, include_design_targets=False)

    result = build_capability_state_panel(tmp_path)
    panel = load_capability_state_panel(result.output_path)

    assert "directed_green_precedence" in panel.columns
    assert panel["directed_green_precedence"].null_count() == panel.height
    assert panel["directed_green_precedence_available_flag"].sum() == 0


def test_init_exports_capability_objects() -> None:
    import src.abm_v5 as abm_v5

    assert abm_v5.CapabilityBuildResult is CapabilityBuildResult
    assert abm_v5.inspect_capability_source_columns is inspect_capability_source_columns
    assert abm_v5.normalize_capability_source_columns is normalize_capability_source_columns
    assert abm_v5.build_capability_state_panel is build_capability_state_panel
    assert abm_v5.validate_capability_state_panel is validate_capability_state_panel
    assert abm_v5.load_capability_state_panel is load_capability_state_panel
    assert abm_v5.summarize_capability_missingness is summarize_capability_missingness
