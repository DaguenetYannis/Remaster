from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.abm_v3.phase_space.plots import (
    PHASE_SPACE_CUBE_SPECS,
    PhaseSpacePlotBuilder,
    available_cube_specs,
    clean_display_label,
    compute_vector_field_table,
    plot_3d_global_trajectory,
    plot_3d_node_trajectories,
    plot_3d_sector_trajectories,
    select_node_trajectories,
    weighted_mean,
)
from src.abm_v3.runner import build_parser


def toy_workspace() -> Path:
    """Create a workspace-local temporary folder for plots tests."""
    root = Path("tmp") / "phase_space_plots_tests" / uuid4().hex[:8]
    root.mkdir(parents=True, exist_ok=True)
    return root


def synthetic_phase_panel(node_count: int = 8, start_year: int = 1995, end_year: int = 1997) -> pd.DataFrame:
    """Build a small phase-space panel with available-axis variables."""
    rows = []
    sectors = ["Agriculture", "Finacial Intermediation", "Hotels and Restraurants", "Manufacturing"]
    for node_index in range(node_count):
        country = f"C{node_index:02d}"
        sector = sectors[node_index % len(sectors)]
        for year in range(start_year, end_year + 1):
            year_step = year - start_year
            output = 100.0 + node_index * 20.0 + year_step * 5.0
            emissions = 50.0 + node_index * 10.0 + year_step
            rows.append(
                {
                    "country_sector": f"{country} | {sector}",
                    "Country": country,
                    "Sector": sector,
                    "Year": year,
                    "green_capability_export_share": 0.1 + node_index * 0.03 + year_step * 0.01,
                    "g_local": 0.35 + node_index * 0.02 + year_step * 0.015,
                    "g_in_network": 0.25 + node_index * 0.015 + year_step * 0.01,
                    "g_out_network": 0.3 + node_index * 0.012 + year_step * 0.008,
                    "log_X_observed": np.log1p(output),
                    "X_observed": output,
                    "trajectory_weight_output": output,
                    "emissions_observed": emissions,
                    "EI": emissions / output,
                    "is_top25_by_output_over_period": node_index < 3,
                    "is_top25_by_emissions_over_period": node_index < 2,
                }
            )
    panel = pd.DataFrame(rows).sort_values(["country_sector", "Year"]).reset_index(drop=True)
    for column in ["green_capability_export_share", "g_local", "g_in_network", "g_out_network", "log_X_observed"]:
        panel[f"{column}_next"] = panel.groupby("country_sector")[column].shift(-1)
    return panel


def test_cube_specs_include_four_available_axis_cubes() -> None:
    specs = available_cube_specs()

    assert set(specs) == {
        "green_readiness_incoming",
        "green_readiness_outgoing",
        "production_safe_incoming",
        "production_safe_outgoing",
    }
    assert specs["green_readiness_incoming"].z == "g_in_network"
    assert specs["green_readiness_outgoing"].z == "g_out_network"


def test_weighted_mean_works_and_falls_back_safely() -> None:
    frame = pd.DataFrame({"value": [1.0, 3.0], "weight": [1.0, 3.0], "zero_weight": [0.0, 0.0]})

    assert weighted_mean(frame, "value", "weight") == 2.5
    assert weighted_mean(frame, "value", "zero_weight") == 2.0
    assert weighted_mean(frame, "value", "missing_weight") == 2.0


def test_global_trajectory_plot_returns_figure_writes_file_and_does_not_mutate() -> None:
    panel = synthetic_phase_panel()
    before = panel.copy(deep=True)
    output_path = toy_workspace() / "global.png"

    fig = plot_3d_global_trajectory(
        panel,
        PHASE_SPACE_CUBE_SPECS["green_readiness_incoming"],
        output_path=output_path,
    )

    assert fig is not None
    assert output_path.exists()
    pd.testing.assert_frame_equal(panel, before)
    plt.close(fig)


def test_sector_trajectory_plot_handles_multiple_sectors() -> None:
    panel = synthetic_phase_panel()
    output_path = toy_workspace() / "sector.png"

    fig = plot_3d_sector_trajectories(
        panel,
        PHASE_SPACE_CUBE_SPECS["green_readiness_outgoing"],
        audience="portfolio",
        output_path=output_path,
    )

    assert fig is not None
    assert output_path.exists()
    plt.close(fig)


def test_node_trajectory_filter_never_plots_all_nodes_by_default() -> None:
    panel = synthetic_phase_panel(node_count=40)
    selected = select_node_trajectories(panel, node_filter="top_output", top_n=5)
    output_path = toy_workspace() / "nodes.png"

    fig = plot_3d_node_trajectories(
        panel,
        PHASE_SPACE_CUBE_SPECS["green_readiness_incoming"],
        node_filter="top_output",
        top_n=5,
        output_path=output_path,
    )

    assert selected["country_sector"].nunique() == 5
    assert selected["country_sector"].nunique() < panel["country_sector"].nunique()
    assert output_path.exists()
    plt.close(fig)


def test_vector_field_table_has_expected_columns_and_uses_next_year_movement() -> None:
    panel = synthetic_phase_panel(node_count=8, start_year=1995, end_year=1996)

    table = compute_vector_field_table(
        panel,
        PHASE_SPACE_CUBE_SPECS["green_readiness_incoming"],
        projection="x_y",
        bins=2,
        min_count=1,
    )

    assert {"x_center", "y_center", "delta_x_mean", "delta_y_mean", "observation_count"}.issubset(table.columns)
    assert not table.empty
    assert table["delta_x_mean"].gt(0).any()
    assert table["delta_y_mean"].gt(0).any()


def test_missing_required_columns_skip_non_strict_and_fail_strict() -> None:
    root = toy_workspace()
    state_panel = root / "state.parquet"
    output_dir = root / "plots"
    synthetic_phase_panel().drop(columns=["g_in_network"]).to_parquet(state_panel, index=False)

    written = PhaseSpacePlotBuilder(
        state_panel=state_panel,
        output_dir=output_dir,
        audience="portfolio",
        plot_3d=False,
        plot_2d=False,
        plot_vector_fields=False,
        strict=False,
    ).build(1995, 1997)
    manifest = pd.read_csv(written["manifest"])
    assert manifest["status"].eq("skipped").any()

    try:
        PhaseSpacePlotBuilder(
            state_panel=state_panel,
            output_dir=output_dir,
            plot_3d=False,
            plot_2d=False,
            plot_vector_fields=False,
            strict=True,
        ).build(1995, 1997)
    except ValueError as error:
        assert "Missing required columns" in str(error)
    else:
        raise AssertionError("Strict plotting should fail when a requested cube is missing required columns.")


def test_display_cleaning_fixes_known_eora_typos() -> None:
    assert clean_display_label("Finacial Intermediation and Business Activities") == "Financial Intermediation and Business Activities"
    assert clean_display_label("Hotels and Restraurants") == "Hotels and Restaurants"


def test_phase_space_plots_cli_command_is_registered() -> None:
    parser = build_parser()

    args = parser.parse_args(["phase-space-plots", "--start-year", "1995", "--end-year", "2016"])

    assert args.command == "phase-space-plots"
    assert args.start_year == 1995
    assert args.end_year == 2016
