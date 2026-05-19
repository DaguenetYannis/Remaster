from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

import src.abm_v5 as abm_v5
from src.abm_v5.diagnostics_visuals import (
    DiagnosticVisualBuildResult,
    build_phase2_diagnostic_visuals,
    build_supplier_candidate_coverage_table,
    build_top_brown_central_nodes_table,
    build_top_emitters_table,
    plot_distribution,
    plot_local_vs_network_greenness,
    plot_selected_node_trajectories,
    plot_supplier_lock_in_vs_green_capability,
    select_illustrative_nodes,
)


def _fake_phase_space() -> pl.DataFrame:
    rows = []
    for year in [1995, 1996, 1997]:
        for index in range(6):
            rows.append(
                {
                    "country_sector": f"C{index} | Country {index} | Industry | Sector {index % 3}",
                    "country": f"C{index}",
                    "sector": f"Sector {index % 3}",
                    "year": year,
                    "emissions": float(100 - index * 5 + year - 1995),
                    "output": float(1000 + index * 100),
                    "emissions_intensity": 0.1 + index * 0.02,
                    "local_greenness": max(0.0, 0.9 - index * 0.1),
                    "network_green_exposure": min(1.0, 0.2 + index * 0.1),
                    "brown_centrality": min(1.0, 0.1 + index * 0.12),
                    "green_capability": min(1.0, 0.15 + index * 0.1),
                    "supplier_lock_in": min(1.0, 0.2 + index * 0.08),
                    "emissions_intensity_gap": -0.1 + index * 0.04,
                    "supplier_count": index + 1,
                    "buyer_count": index + 2,
                }
            )
    return pl.DataFrame(rows)


def _write_pyproject(root: Path) -> None:
    root.joinpath("pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")


def _write_inputs(root: Path) -> None:
    _write_pyproject(root)
    phase_dir = root / "data" / "abm_v5" / "phase_space"
    validation_dir = root / "data" / "abm_v5" / "validation"
    phase_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    _fake_phase_space().write_parquet(phase_dir / "historical_phase_space_panel_1995_2016.parquet")
    validation_dir.joinpath("supplier_candidate_coverage_summary.json").write_text(
        json.dumps(
            {
                "yearly_coverage": [
                    {
                        "year": 1995,
                        "raw_positive_edges": 100,
                        "retained_historical_candidate_rows": 20,
                        "fallback_candidate_rows": 3,
                        "total_candidate_rows": 23,
                        "retained_edge_share": 0.2,
                        "retained_transaction_value_coverage": 0.96,
                        "mean_buyer_input_coverage": 0.91,
                        "share_buyer_years_reaching_coverage_target": 0.7,
                        "max_candidates_per_buyer_year": 12,
                        "buyers_with_coverage_target_unmet": 2,
                        "buyers_with_fallback_candidates": 2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_build_top_emitters_table_returns_ranked_rows() -> None:
    table = build_top_emitters_table(_fake_phase_space())
    assert table.height == 6
    assert table["rank"].to_list()[0] == 1
    assert table["mean_emissions"].to_list()[0] >= table["mean_emissions"].to_list()[-1]


def test_build_top_brown_central_nodes_table_returns_ranked_rows() -> None:
    table = build_top_brown_central_nodes_table(_fake_phase_space())
    assert table["rank"].to_list()[0] == 1
    assert table["mean_brown_centrality"].to_list()[0] >= table["mean_brown_centrality"].to_list()[-1]


def test_build_supplier_candidate_coverage_table_reads_json(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    table = build_supplier_candidate_coverage_table(
        tmp_path / "data" / "abm_v5" / "validation" / "supplier_candidate_coverage_summary.json"
    )
    assert table.height == 1
    assert table["year"].item() == 1995


def test_select_illustrative_nodes_combines_selection_reasons() -> None:
    selected = select_illustrative_nodes(_fake_phase_space())
    assert 1 <= selected.height <= 12
    assert "selection_reason" in selected.columns


def test_plot_local_vs_network_greenness_writes_png(tmp_path: Path) -> None:
    path = plot_local_vs_network_greenness(_fake_phase_space(), tmp_path / "plot.png")
    assert path.exists()
    assert path.suffix == ".png"


def test_plot_supplier_lock_in_vs_green_capability_writes_png(tmp_path: Path) -> None:
    path = plot_supplier_lock_in_vs_green_capability(_fake_phase_space(), tmp_path / "plot.png")
    assert path.exists()


def test_plot_distribution_writes_png(tmp_path: Path) -> None:
    path = plot_distribution(_fake_phase_space(), "supplier_count", tmp_path / "dist.png")
    assert path.exists()


def test_plot_selected_node_trajectories_writes_png_and_metadata(tmp_path: Path) -> None:
    selected = select_illustrative_nodes(_fake_phase_space())
    path = plot_selected_node_trajectories(
        _fake_phase_space(), selected, tmp_path / "traj.png", tmp_path / "meta.csv"
    )
    assert path.exists()
    assert (tmp_path / "meta.csv").exists()


def test_build_phase2_diagnostic_visuals_requires_phase_space_input(tmp_path: Path) -> None:
    _write_pyproject(tmp_path)
    with pytest.raises(FileNotFoundError, match="Phase-space panel"):
        build_phase2_diagnostic_visuals(tmp_path)


def test_build_phase2_diagnostic_visuals_writes_expected_outputs(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    result = build_phase2_diagnostic_visuals(tmp_path)
    assert isinstance(result, DiagnosticVisualBuildResult)
    assert result.n_plots_created == 10
    assert result.n_tables_created == 5
    assert (result.output_plot_dir / "local_greenness_vs_network_green_exposure.png").exists()
    assert (result.output_table_dir / "top_emitters_avg_1995_2016.csv").exists()


def test_init_exports_diagnostic_visual_objects() -> None:
    assert abm_v5.DiagnosticVisualBuildResult is DiagnosticVisualBuildResult
    assert abm_v5.build_phase2_diagnostic_visuals is build_phase2_diagnostic_visuals
    assert abm_v5.build_top_emitters_table is build_top_emitters_table
    assert abm_v5.build_top_brown_central_nodes_table is build_top_brown_central_nodes_table
    assert abm_v5.build_supplier_candidate_coverage_table is build_supplier_candidate_coverage_table
    assert abm_v5.select_illustrative_nodes is select_illustrative_nodes
