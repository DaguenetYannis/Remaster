from pathlib import Path

import pandas as pd

from src.abm_v2.plots import ABMPlotter


def make_aggregate_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "step": [0, 1, 0, 1],
            "scenario": ["baseline", "baseline", "fast", "fast"],
            "total_emissions": [100.0, 90.0, 100.0, 70.0],
            "total_output": [1000.0, 1000.0, 1000.0, 980.0],
            "mean_ei": [0.10, 0.09, 0.10, 0.07],
            "mean_g_local": [0.80, 0.82, 0.80, 0.85],
            "mean_g_in": [0.75, 0.77, 0.75, 0.80],
            "mean_g_out": [0.76, 0.78, 0.76, 0.81],
        }
    )


def make_node_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "step": [0, 1, 0, 1],
            "scenario": ["baseline", "baseline", "baseline", "baseline"],
            "country_sector": ["A", "A", "B", "B"],
            "EI": [0.5, 0.4, 0.2, 0.15],
            "g_out": [0.7, 0.75, 0.8, 0.85],
            "out_strength": [100.0, 110.0, 20.0, 30.0],
            "regime": ["brown-core", "brown-core", "green-periphery", "green-periphery"],
        }
    )


def test_plotter_saves_core_plots(tmp_path: Path):
    plotter = ABMPlotter(output_dir=tmp_path)
    aggregate = make_aggregate_results()
    nodes = make_node_results()

    plotter.save_all(aggregate, nodes)

    assert (tmp_path / "01_scenario_total_emissions.png").exists()
    assert (tmp_path / "02_scenario_total_output.png").exists()
    assert (tmp_path / "03_scenario_mean_ei.png").exists()
    assert (tmp_path / "04_greeness_trajectories_baseline.png").exists()
    assert (tmp_path / "05_phase_space_baseline.png").exists()
    assert (tmp_path / "06_regime_shares_baseline.png").exists()
    assert (tmp_path / "07_regime_centroids_baseline.png").exists()
    assert (tmp_path / "08_selected_node_ei_trajectories_baseline.png").exists()
    assert (tmp_path / "09_distribution_EI_baseline.png").exists()


def test_top_embodied_carbon_flows_plot_is_optional(tmp_path: Path):
    plotter = ABMPlotter(output_dir=tmp_path)
    nodes = make_node_results()

    plotter.plot_top_embodied_carbon_flows(nodes)

    assert not (tmp_path / "10_top_embodied_carbon_flows_baseline.png").exists()