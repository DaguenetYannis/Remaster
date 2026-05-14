from pathlib import Path
from uuid import uuid4

import polars as pl
import pytest

from src.abm_v4.config import ABMV4Config
from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.validation import MultiYearHistoricalValidator


def _toy_paths() -> ABMV4Paths:
    return ABMV4Paths(project_root=Path("tmp") / "abm_v4_multiyear_validation_tests" / uuid4().hex)


def _write_multiyear_simulation(paths: ABMV4Paths) -> None:
    paths.simulations.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "country_sector": ["A_S1", "A_S1", "B_S1", "B_S1", "C_S2", "C_S2"],
            "year": [2015, 2016, 2015, 2016, 2015, 2016],
            "Country": ["A", "A", "B", "B", "C", "C"],
            "Sector": ["S1", "S1", "S1", "S1", "S2", "S2"],
            "ecosystem_id": [1, 1, 1, 1, 2, 2],
            "ecosystem_label": ["eco1", "eco1", "eco1", "eco1", "eco2", "eco2"],
            "X_observed": [10.0, 12.0, 20.0, 22.0, 30.0, 33.0],
            "EI_observed": [2.0, 1.0, 4.0, 2.0, 0.0, 3.0],
            "emissions_observed": [20.0, 12.0, 80.0, 44.0, 0.0, 99.0],
            "X_sim": [10.0, 12.0, 20.0, 22.0, 30.0, 33.0],
            "EI_sim": [2.0, 1.2, 4.0, 3.0, 1.0, 2.5],
            "emissions_sim": [20.0, 14.4, 80.0, 66.0, 30.0, 82.5],
            "general_capability_source": [
                "atlas_observed",
                "atlas_observed",
                "io_imputed",
                "io_imputed",
                "unavailable",
                "unavailable",
            ],
            "green_capability_source": [
                "atlas_observed",
                "atlas_observed",
                "io_imputed",
                "io_imputed",
                "unavailable",
                "unavailable",
            ],
            "brown_centrality": [0.1, 0.2, 0.8, 0.7, 0.4, 0.5],
            "readiness": [0.02, 0.03, 0.01, 0.02, 0.04, 0.05],
            "ei_gap": [0.1, 0.0, 0.6, 0.4, 0.0, 0.2],
            "invalid_EI_flag": [False, False, False, False, True, False],
            "capability_model_unavailable_flag": [False, False, False, False, True, True],
        }
    ).write_parquet(paths.base_multiyear_state_panel_path)
    pl.DataFrame(
        {
            "year": [2015, 2016],
            "emissions_identity_max_error": [0.0, 0.0],
            "status": ["warning", "warning"],
        }
    ).write_csv(paths.base_multiyear_summary_panel_path)


def test_multiyear_error_panel_computes_ei_and_emissions_errors() -> None:
    paths = _toy_paths()
    _write_multiyear_simulation(paths)
    panel = MultiYearHistoricalValidator(paths, ABMV4Config()).build().error_panel

    row = panel.filter((pl.col("country_sector") == "A_S1") & (pl.col("year") == 2016)).to_dicts()[0]
    assert row["EI_error"] == pytest.approx(0.2)
    assert row["EI_abs_error"] == pytest.approx(0.2)
    assert row["emissions_error"] == pytest.approx(2.4)
    assert row["emissions_abs_error"] == pytest.approx(2.4)


def test_multiyear_log_ei_error_excludes_invalid_ei() -> None:
    paths = _toy_paths()
    _write_multiyear_simulation(paths)
    panel = MultiYearHistoricalValidator(paths, ABMV4Config()).build().error_panel

    invalid = panel.filter((pl.col("country_sector") == "C_S2") & (pl.col("year") == 2015))
    assert invalid["log_EI_error"].item() is None


def test_multiyear_rei_uses_consecutive_years() -> None:
    paths = _toy_paths()
    _write_multiyear_simulation(paths)
    panel = MultiYearHistoricalValidator(paths, ABMV4Config()).build().error_panel

    row = panel.filter((pl.col("country_sector") == "A_S1") & (pl.col("year") == 2015)).to_dicts()[0]
    assert row["observed_rEI"] == pytest.approx(0.6931471805599453)
    assert row["simulated_rEI"] == pytest.approx(0.5108256237659907)


def test_multiyear_grouped_diagnostics_include_sector_ecosystem_and_capability_source() -> None:
    paths = _toy_paths()
    _write_multiyear_simulation(paths)
    result = MultiYearHistoricalValidator(paths, ABMV4Config()).build()

    assert "Sector" in result.error_by_sector.columns
    assert "ecosystem_label" in result.error_by_ecosystem.columns
    assert "capability_source" in result.error_by_capability_source.columns
    assert set(result.error_by_capability_source["capability_source"].to_list()) >= {
        "atlas_observed",
        "io_imputed",
        "unavailable",
    }


def test_multiyear_calibration_targets_identify_bias_direction() -> None:
    paths = _toy_paths()
    _write_multiyear_simulation(paths)
    targets = MultiYearHistoricalValidator(paths, ABMV4Config()).build().calibration_targets

    s1 = targets.filter(pl.col("Sector") == "S1").to_dicts()[0]
    assert s1["rEI_bias_by_sector"] < 0
    assert s1["suggested_direction"] == "increase readiness"


def test_multiyear_markdown_report_written_only_when_requested() -> None:
    paths = _toy_paths()
    _write_multiyear_simulation(paths)
    validator = MultiYearHistoricalValidator(paths, ABMV4Config())
    result = validator.build()

    assert not paths.multiyear_validation_report_md_path.exists()
    validator.write_outputs(result)
    assert paths.multiyear_error_panel_path.exists()
    assert paths.multiyear_error_summary_path.exists()
    assert paths.multiyear_validation_report_md_path.exists()


def test_multiyear_validation_fails_clearly_when_simulation_output_missing() -> None:
    paths = _toy_paths()
    with pytest.raises(FileNotFoundError, match="Missing multi-year simulation output"):
        MultiYearHistoricalValidator(paths, ABMV4Config()).build()
