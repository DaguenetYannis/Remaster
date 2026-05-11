from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.abm_v3.phase_space.plots import (
    CAPTION_REGISTRY,
    PHASE_SPACE_CUBE_SPECS,
    TITLE_REGISTRY,
    PhaseSpacePlotBuilder,
    build_movement_summary,
    clean_display_label,
    compare_green_readiness_vector_fields,
    compute_vector_field_table,
    plot_2d_projection_trajectory,
    select_node_trajectories,
    select_sectors_for_display,
)
from src.abm_v3.runner import build_parser


def toy_workspace() -> Path:
    root = Path("tmp") / "phase_space_plot_refinement_tests" / uuid4().hex[:8]
    root.mkdir(parents=True, exist_ok=True)
    return root


def synthetic_panel(node_count: int = 12, start_year: int = 1995, end_year: int = 2016) -> pd.DataFrame:
    rows = []
    sectors = ["Agriculture", "Manufacturing", "Finacial Services", "Hotels and Restraurants", "Energy"]
    for node_index in range(node_count):
        country = f"C{node_index:02d}"
        sector = sectors[node_index % len(sectors)]
        for year in range(start_year, end_year + 1):
            step = year - start_year
            output = 100.0 + 15.0 * node_index + 2.5 * step
            emissions = 70.0 + 12.0 * node_index - 0.5 * step
            rows.append(
                {
                    "country_sector": f"{country} | {sector}",
                    "Country": country,
                    "Sector": sector,
                    "Year": year,
                    "green_capability_export_share": 0.10 + 0.01 * node_index + 0.006 * step,
                    "g_local": 0.30 + 0.004 * node_index + 0.005 * step + (0.01 if step % 4 == 0 else 0.0),
                    "g_in_network": 0.20 + 0.006 * node_index + 0.002 * step + (0.04 if step == 10 else 0.0),
                    "g_out_network": 0.25 + 0.004 * node_index + 0.003 * step,
                    "log_X_observed": np.log1p(output),
                    "X_observed": output,
                    "trajectory_weight_output": output,
                    "emissions_observed": emissions,
                    "EI": emissions / output,
                    "is_top25_by_output_over_period": node_index < 8,
                    "is_top25_by_emissions_over_period": node_index < 7,
                }
            )
    panel = pd.DataFrame(rows).sort_values(["country_sector", "Year"]).reset_index(drop=True)
    for column in ["green_capability_export_share", "g_local", "g_in_network", "g_out_network", "log_X_observed"]:
        panel[f"{column}_next"] = panel.groupby("country_sector")[column].shift(-1)
    return panel


def test_title_registry_contains_required_families_and_no_em_dash() -> None:
    required = {
        "global_green_readiness_xy",
        "global_green_readiness_incoming_3d",
        "global_green_readiness_outgoing_3d",
        "global_production_safe_xy",
        "global_production_safe_3d",
        "sector_green_readiness",
        "nodes_top_output_green_readiness",
        "nodes_top_emissions_green_readiness",
        "vector_field_green_readiness",
    }
    assert required.issubset(TITLE_REGISTRY)
    all_text = " ".join(title for entry in TITLE_REGISTRY.values() for title in entry.values())
    assert "\u2014" not in all_text


def test_caption_registry_contains_interpretive_notes() -> None:
    assert {"global", "production_safe", "sector_node", "vector_field"}.issubset(CAPTION_REGISTRY)
    assert "historical diagnostics" in CAPTION_REGISTRY["vector_field"]


def test_movement_diagnostics_compute_geometry_and_winding_paths() -> None:
    panel = pd.DataFrame(
        {
            "country_sector": ["A", "A", "A"],
            "Sector": ["S", "S", "S"],
            "Year": [1995, 1996, 1997],
            "green_capability_export_share": [0.0, 1.0, 1.0],
            "g_local": [0.0, 0.0, 1.0],
            "g_in_network": [0.0, 0.0, 0.0],
            "g_out_network": [0.0, 0.0, 0.0],
            "log_X_observed": [1.0, 1.0, 1.0],
            "X_observed": [1.0, 1.0, 1.0],
            "emissions_observed": [1.0, 1.0, 1.0],
        }
    )
    summary = build_movement_summary(panel, 1995, 1997, top_n=1)
    global_row = summary.loc[
        summary["unit_type"].eq("global") & summary["cube_slug"].eq("green_readiness_incoming")
    ].iloc[0]

    assert np.isclose(global_row["net_displacement"], np.sqrt(2.0))
    assert np.isclose(global_row["path_length"], 2.0)
    assert np.isclose(global_row["displacement_to_path_ratio"], np.sqrt(2.0) / 2.0)
    assert global_row["turning_intensity"] > 0.0


def test_sector_and_node_selection_counts_are_bounded() -> None:
    panel = synthetic_panel(node_count=30)
    sectors = select_sectors_for_display(panel, PHASE_SPACE_CUBE_SPECS["green_readiness_incoming"], top_n=3)
    portfolio_nodes = select_node_trajectories(panel, "top_output", top_n=10)
    research_nodes = select_node_trajectories(panel, "top_output", top_n=25)

    assert len(sectors) <= 3
    assert portfolio_nodes["country_sector"].nunique() == 10
    assert research_nodes["country_sector"].nunique() == 25


def test_recommendation_manifest_marks_global_green_readiness_xy_as_thesis_core() -> None:
    root = toy_workspace()
    state_panel = root / "state.parquet"
    output_dir = root / "plots"
    synthetic_panel().to_parquet(state_panel, index=False)

    written = PhaseSpacePlotBuilder(
        state_panel=state_panel,
        output_dir=output_dir,
        audience="portfolio",
        plot_3d=False,
        plot_2d=True,
        plot_vector_fields=False,
        include_sector=False,
        include_node=False,
    ).build(1995, 2016)
    recommendations = pd.read_csv(written["recommendations"])

    core = recommendations.loc[
        recommendations["plot_file"].str.contains("global_green_readiness_incoming_xy", na=False)
    ].iloc[0]
    assert core["figure_tier"] == "thesis-core"
    assert core["recommendation_status"] == "recommended-draft"


def test_vector_field_comparison_explains_xy_similarity() -> None:
    panel = synthetic_panel(node_count=8, start_year=1995, end_year=1997)
    incoming = compute_vector_field_table(panel, PHASE_SPACE_CUBE_SPECS["green_readiness_incoming"], "x_y", bins=2, min_count=1)
    outgoing = compute_vector_field_table(panel, PHASE_SPACE_CUBE_SPECS["green_readiness_outgoing"], "x_y", bins=2, min_count=1)

    comparison = compare_green_readiness_vector_fields(incoming, outgoing, "x_y", "incoming.csv", "outgoing.csv")

    assert comparison["n_common_bins"].iloc[0] > 0
    assert "share the same XY axes" in comparison["same_projection_explanation"].iloc[0]


def test_plot_functions_do_not_mutate_input_dataframe() -> None:
    panel = synthetic_panel(node_count=6, start_year=1995, end_year=2000)
    before = panel.copy(deep=True)
    output_path = toy_workspace() / "projection.png"

    fig = plot_2d_projection_trajectory(
        panel,
        PHASE_SPACE_CUBE_SPECS["green_readiness_incoming"],
        projection="x_y",
        unit="global",
        output_path=output_path,
    )

    assert output_path.exists()
    pd.testing.assert_frame_equal(panel, before)
    plt.close(fig)


def test_display_cleaning_and_cli_defaults_remain_valid() -> None:
    assert clean_display_label("Finacial and Restraurants") == "Financial and Restaurants"
    parser = build_parser()
    args = parser.parse_args(["phase-space-plots", "--start-year", "1995", "--end-year", "2016"])

    assert args.command == "phase-space-plots"
    assert args.title_mode == "theory"
    assert args.top_sector_n == 8
    assert args.top_node_n == 10
    assert args.research_top_node_n == 25
