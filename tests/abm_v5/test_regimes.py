from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import src.abm_v5 as abm_v5
from src.abm_v5.regimes import (
    DESIGN_TARGET_VARIABLES,
    RegimeDiscoveryBuildResult,
    RegimeLabel,
    assign_interpretable_regimes,
    build_historical_regime_discovery,
    build_regime_composition_by_country,
    build_regime_composition_by_sector,
    build_regime_profile_summary,
    build_regime_size_by_year,
    discover_regime_thresholds,
    load_historical_regime_panel,
    select_regime_discovery_variables,
    validate_regime_discovery_outputs,
)


def _phase_panel() -> pl.DataFrame:
    regimes = [
        (0.0, 0.95, 0.95, 0.1, 0.1, "embedded"),
        (0.1, 0.96, 0.8, 0.2, 0.95, "constrained"),
        (0.2, 0.97, 0.5, 0.98, 0.2, "brown_capable"),
        (0.95, 0.05, 0.5, 0.2, 0.2, "dirty_gap"),
        (-0.95, 0.05, 0.5, 0.2, 0.2, "clean_low"),
        (0.05, 0.4, 0.45, 0.2, 0.2, "low_signal"),
        (None, None, 0.5, 0.2, 0.2, "insufficient"),
        (0.4, 0.4, 0.4, 0.99, 0.95, "mixed"),
    ]
    rows = []
    for year in [1995, 1996]:
        for index, (ei_gap, green, network, brown, lock, name) in enumerate(regimes):
            rows.append(
                {
                    "country_sector": f"C{index} | Country {index} | Industry | Sector {index % 2}",
                    "country": f"C{index}",
                    "sector": f"Sector {index % 2}",
                    "year": year,
                    "output": 100.0 + index,
                    "emissions": 10.0 + index,
                    "emissions_intensity": 0.1 + index * 0.01,
                    "emissions_intensity_gap": ei_gap,
                    "local_greenness": 0.5,
                    "green_capability": green,
                    "general_capability": 0.5 if green is None else green,
                    "network_green_exposure": network,
                    "brown_centrality": brown,
                    "supplier_lock_in": lock,
                    "import_dependence_proxy": 0.2,
                    "export_dependence_proxy": 0.2,
                    "supplier_concentration_hhi": 0.2,
                    "buyer_concentration_hhi": 0.2,
                    "capability_density": None,
                    "green_capability_density": None,
                    "ecosystem_proximity": None,
                    "directed_green_precedence": None,
                    "reachable_green_complexity": None,
                    "transition_sector_score": None,
                }
            )
    return pl.DataFrame(rows)


def _thresholds() -> dict:
    return {
        "emissions_intensity_gap": {"thresholds": {"low": -0.5, "high": 0.5}},
        "green_capability": {"thresholds": {"low": 0.2, "high": 0.8}},
        "general_capability": {"thresholds": {"low": 0.2, "high": 0.8}},
        "network_green_exposure": {"thresholds": {"low": 0.3, "high": 0.7}},
        "brown_centrality": {"thresholds": {"low": 0.5, "high": 0.9, "very_high": 0.98}},
        "supplier_lock_in": {"thresholds": {"low": 0.3, "high": 0.7}},
        "local_greenness": {"thresholds": {"low": 0.3, "high": 0.7}},
    }


def _assigned() -> pl.DataFrame:
    return assign_interpretable_regimes(_phase_panel(), _thresholds())


def test_select_regime_discovery_variables_excludes_design_targets() -> None:
    selection = select_regime_discovery_variables(_phase_panel())
    assert not set(selection["selected_variables"]).intersection(DESIGN_TARGET_VARIABLES)


def test_select_regime_discovery_variables_excludes_low_coverage_variables() -> None:
    df = _phase_panel().with_columns(pl.lit(None).alias("local_greenness"))
    selection = select_regime_discovery_variables(df)
    assert "local_greenness" not in selection["selected_variables"]


def test_discover_regime_thresholds_returns_required_thresholds() -> None:
    selection = select_regime_discovery_variables(_phase_panel())
    thresholds = discover_regime_thresholds(_phase_panel(), selection["selected_variables"])
    assert "emissions_intensity_gap" in thresholds
    assert "high" in thresholds["brown_centrality"]["thresholds"]
    assert thresholds["emissions_intensity_gap"]["method"] == "global_quantile_rule"


def test_assign_interpretable_regimes_assigns_insufficient_data() -> None:
    row = _assigned().filter(pl.col("country_sector").str.starts_with("C6")).row(0, named=True)
    assert row["regime_membership"] == RegimeLabel.INSUFFICIENT_DATA.value


def test_assign_interpretable_regimes_assigns_green_capable_embedded() -> None:
    row = _assigned().filter(pl.col("country_sector").str.starts_with("C0")).row(0, named=True)
    assert row["regime_membership"] == RegimeLabel.GREEN_CAPABLE_EMBEDDED.value


def test_assign_interpretable_regimes_assigns_green_capable_constrained() -> None:
    row = _assigned().filter(pl.col("country_sector").str.starts_with("C1")).row(0, named=True)
    assert row["regime_membership"] == RegimeLabel.GREEN_CAPABLE_CONSTRAINED.value


def test_assign_interpretable_regimes_assigns_brown_central_capable() -> None:
    row = _assigned().filter(pl.col("country_sector").str.starts_with("C2")).row(0, named=True)
    assert row["regime_membership"] == RegimeLabel.BROWN_CENTRAL_CAPABLE.value


def test_assign_interpretable_regimes_assigns_dirty_capability_gap() -> None:
    row = _assigned().filter(pl.col("country_sector").str.starts_with("C3")).row(0, named=True)
    assert row["regime_membership"] == RegimeLabel.DIRTY_CAPABILITY_GAP.value


def test_assign_interpretable_regimes_does_not_create_regime_probability() -> None:
    assert "regime_probability" not in _assigned().columns


def test_build_regime_profile_summary_returns_one_row_per_regime() -> None:
    summary = build_regime_profile_summary(_assigned())
    assert summary.height == _assigned()["regime_membership"].n_unique()


def test_build_regime_size_by_year_returns_year_regime_rows() -> None:
    summary = build_regime_size_by_year(_assigned())
    assert {"year", "regime_membership", "n_rows", "share_year_rows", "mean_regime_confidence"}.issubset(summary.columns)


def test_build_regime_composition_by_sector_returns_expected_columns() -> None:
    summary = build_regime_composition_by_sector(_assigned())
    assert {"regime_membership", "sector", "n_rows", "share_within_regime", "mean_output", "mean_emissions"}.issubset(summary.columns)


def test_build_regime_composition_by_country_returns_expected_columns() -> None:
    summary = build_regime_composition_by_country(_assigned())
    assert {"regime_membership", "country", "n_rows", "share_within_regime", "mean_output", "mean_emissions"}.issubset(summary.columns)


def test_validate_regime_discovery_outputs_rejects_design_target_selected_variables() -> None:
    selection = {"selected_variables": ["capability_density"]}
    results = validate_regime_discovery_outputs(_assigned(), _thresholds(), selection)
    assert any(result.check_name == "regime_no_design_target_selected_variables" and result.n_failed > 0 for result in results)


def test_validate_regime_discovery_outputs_passes_valid_output() -> None:
    selection = select_regime_discovery_variables(_phase_panel())
    results = validate_regime_discovery_outputs(_assigned(), _thresholds(), selection)
    assert all(result.status.value == "passed" for result in results if result.severity.value != "warning")


def _write_pyproject(root: Path) -> None:
    root.joinpath("pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")


def test_build_historical_regime_discovery_requires_phase_space_panel(tmp_path: Path) -> None:
    _write_pyproject(tmp_path)
    with pytest.raises(FileNotFoundError, match="Phase 2.6"):
        build_historical_regime_discovery(tmp_path)


def test_build_historical_regime_discovery_writes_outputs_with_fake_data(tmp_path: Path) -> None:
    _write_pyproject(tmp_path)
    phase_dir = tmp_path / "data" / "abm_v5" / "phase_space"
    phase_dir.mkdir(parents=True)
    _phase_panel().write_parquet(phase_dir / "historical_phase_space_panel_1995_2016.parquet")
    result = build_historical_regime_discovery(tmp_path)
    assert isinstance(result, RegimeDiscoveryBuildResult)
    assert result.regime_panel_path.exists()
    assert result.variable_selection_path.exists()
    assert result.thresholds_path.exists()
    assert result.profile_summary_path.exists()
    assert result.validation_path.exists()


def test_build_historical_regime_discovery_preserves_backbone(tmp_path: Path) -> None:
    _write_pyproject(tmp_path)
    phase_dir = tmp_path / "data" / "abm_v5" / "phase_space"
    phase_dir.mkdir(parents=True)
    _phase_panel().write_parquet(phase_dir / "historical_phase_space_panel_1995_2016.parquet")
    result = build_historical_regime_discovery(tmp_path)
    assert load_historical_regime_panel(result.regime_panel_path).height == _phase_panel().height


def test_init_exports_regime_objects() -> None:
    assert abm_v5.RegimeDiscoveryBuildResult is RegimeDiscoveryBuildResult
    assert abm_v5.RegimeLabel is RegimeLabel
    assert abm_v5.select_regime_discovery_variables is select_regime_discovery_variables
    assert abm_v5.discover_regime_thresholds is discover_regime_thresholds
    assert abm_v5.assign_interpretable_regimes is assign_interpretable_regimes
    assert abm_v5.build_historical_regime_discovery is build_historical_regime_discovery
