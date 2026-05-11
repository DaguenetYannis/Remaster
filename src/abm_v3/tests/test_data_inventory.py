from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from src.abm_v3.data_inventory import (
    build_data_inventory,
    classify_file_group,
    classify_status,
    classify_variable_semantics,
    inspect_file,
)
from src.abm_v3.runner import build_parser


def toy_root() -> Path:
    """Create a small workspace-local test data root."""
    root = Path("tmp") / "abm_v3_data_inventory_tests" / uuid4().hex[:8] / "data"
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_toy_inventory_files(root: Path) -> None:
    """Write small representative inventory files."""
    input_dir = root / "abm_v3" / "inputs"
    ei_dir = root / "abm_v3" / "ei_transition" / "inputs"
    scenario_dir = root / "abm_v3" / "leontief" / "behavioural" / "scenarios" / "analysis_report"
    behavioural_diagnostics_dir = root / "abm_v3" / "leontief" / "behavioural" / "diagnostics"
    legacy_dir = root / "abm"
    input_dir.mkdir(parents=True, exist_ok=True)
    ei_dir.mkdir(parents=True, exist_ok=True)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    behavioural_diagnostics_dir.mkdir(parents=True, exist_ok=True)
    legacy_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "country_sector": ["AAA | Agriculture", "BBB | Manufacturing"],
            "Year": [1995, 1995],
            "EI": [0.4, 0.7],
            "green_capability_export_share": [0.2, 0.5],
            "X_observed": [100.0, 200.0],
        }
    ).to_parquet(input_dir / "abm_v3_historical_panel_1995_2016_transpose_row_fd_without_inventory.parquet")

    pd.DataFrame(
        {
            "country_sector": ["AAA | Agriculture"],
            "Year": [1996],
            "EI": [0.35],
            "network_green_ness": [0.6],
        }
    ).to_parquet(ei_dir / "ei_transition_panel_1995_2016.parquet")

    pd.DataFrame(
        {
            "scenario_name": ["green_capability_push"],
            "country_sector": ["AAA | Agriculture"],
            "Year": [2016],
            "delta_X": [4.0],
        }
    ).to_csv(scenario_dir / "scenario_effects.csv", index=False)

    pd.DataFrame(
        {
            "country_sector": ["AAA | Agriculture"],
            "Year": [1995],
            "X_observed": [100.0],
            "X_realized": [99.0],
            "absolute_percentage_error": [0.01],
        }
    ).to_parquet(behavioural_diagnostics_dir / "behavioural_node_comparison_1995.parquet")

    pd.DataFrame({"country_sector": ["AAA | Agriculture"], "Year": [1995], "output": [100.0]}).to_parquet(
        legacy_dir / "simulation_output.parquet"
    )


def test_inventory_discovers_csv_and_parquet_files() -> None:
    root = toy_root()
    write_toy_inventory_files(root)

    written = build_data_inventory(root=root, output_dir=root / "abm_v3" / "data_inventory")
    inventory = pd.read_csv(written["data_inventory"])

    assert inventory["extension"].isin([".csv"]).any()
    assert inventory["extension"].isin([".parquet"]).any()
    assert "abm_v3_input_panel" in inventory["file_group"].tolist()


def test_parquet_schema_is_inspected_without_full_loading() -> None:
    pytest.importorskip("pyarrow")
    root = toy_root()
    write_toy_inventory_files(root)
    path = root / "abm_v3" / "inputs" / "abm_v3_historical_panel_1995_2016_transpose_row_fd_without_inventory.parquet"

    inspection = inspect_file(path, base_root=root, sample_rows=1)

    assert inspection.row_count == 2
    assert inspection.column_count == 5
    assert inspection.sample is not None
    assert len(inspection.sample) == 1
    assert "parquet schema inspected from metadata" in inspection.notes


def test_semantic_classification_recognizes_key_variables() -> None:
    assert classify_variable_semantics("country_sector") == "identifier"
    assert classify_variable_semantics("Year") == "time"
    assert classify_variable_semantics("EI") == "emissions_intensity"
    assert classify_variable_semantics("green_capability_export_share") == "green_capability"
    assert classify_variable_semantics("X_observed") == "production"
    assert classify_variable_semantics("scenario_name") == "scenario"


def test_file_group_classifies_current_and_legacy_sources() -> None:
    assert (
        classify_file_group(Path("abm_v3/inputs/abm_v3_historical_panel_1995_2016_transpose_row_fd_without_inventory.parquet"))
        == "abm_v3_input_panel"
    )
    assert classify_file_group(Path("abm_v3/ei_transition/inputs/ei_transition_panel_1995_2016.parquet")) == "abm_v3_ei_transition"
    assert (
        classify_file_group(Path("abm_v3/leontief/behavioural/scenarios/analysis_report/scenario_effects.csv"))
        == "abm_v3_scenario_analysis"
    )
    assert classify_file_group(Path("abm/simulation_output.parquet")) == "legacy_abm"


def test_refined_status_classification_separates_state_model_scenario_and_legacy() -> None:
    assert (
        classify_status(
            Path("abm_v3/inputs/abm_v3_historical_panel_1995_2016_transpose_row_fd_without_inventory.parquet"),
            "abm_v3_input_panel",
            "model_input",
        )
        == "authoritative_state_source"
    )
    assert (
        classify_status(
            Path("abm_v3/leontief/behavioural/diagnostics/behavioural_node_comparison_1995.parquet"),
            "abm_v3_leontief_behavioural",
            "model_output",
        )
        == "authoritative_model_output"
    )
    assert (
        classify_status(
            Path("abm_v3/leontief/behavioural/scenarios/analysis_report/scenario_effects.csv"),
            "abm_v3_scenario_analysis",
            "scenario",
        )
        == "current_scenario_output"
    )
    assert classify_status(Path("abm/simulation_output.parquet"), "legacy_abm", "model_output") == "legacy"


def test_markdown_catalog_is_written() -> None:
    root = toy_root()
    write_toy_inventory_files(root)

    written = build_data_inventory(root=root, output_dir=root / "abm_v3" / "data_inventory")
    markdown = written["catalog"].read_text(encoding="utf-8")

    assert "# ABM v3 Data Catalog and Visual Use Map" in markdown
    assert "Authoritative State Sources" in markdown
    assert "Current Model Outputs" in markdown


def test_semantic_variable_map_is_written_with_core_variables() -> None:
    root = toy_root()
    write_toy_inventory_files(root)

    written = build_data_inventory(root=root, output_dir=root / "abm_v3" / "data_inventory")
    semantic_map = pd.read_csv(written["semantic_variable_map"])
    canonical_variables = set(semantic_map["canonical_variable"])

    assert {"country_sector", "X_observed", "EI", "g_local"}.issubset(canonical_variables)
    assert {"green_capability_export_share", "network_green_exposure", "brown_centrality", "rEI"}.issubset(
        canonical_variables
    )


def test_visual_use_map_includes_all_five_3d_cubes() -> None:
    root = toy_root()
    write_toy_inventory_files(root)

    written = build_data_inventory(root=root, output_dir=root / "abm_v3" / "data_inventory")
    visual_map = pd.read_csv(written["visual_use_map"])
    cube_names = set(visual_map.loc[visual_map["visual_family"].str.startswith("3D phase-space cubes"), "visual_family"])

    assert "3D phase-space cubes: Green Transition Readiness Cube" in cube_names
    assert "3D phase-space cubes: Brown Lock-in Cube" in cube_names
    assert "3D phase-space cubes: Productive Ecosystem Transition Cube" in cube_names
    assert "3D phase-space cubes: Production-Safe Greening Cube" in cube_names
    assert "3D phase-space cubes: Scenario Perturbation Cube" in cube_names


def test_phase_space_and_vector_field_visuals_use_state_panel_and_axis_variants() -> None:
    root = toy_root()
    write_toy_inventory_files(root)

    written = build_data_inventory(root=root, output_dir=root / "abm_v3" / "data_inventory")
    visual_map = pd.read_csv(written["visual_use_map"])
    phase_panel = "data/abm_v3/phase_space/abm_v3_phase_space_state_panel_1995_2016.parquet"
    phase_or_vector = visual_map.loc[
        visual_map["visual_family"].str.contains("phase-space|Vector-field", case=False, na=False)
    ]

    assert not phase_or_vector.empty
    assert set(phase_or_vector["recommended_data_source"]) == {phase_panel}
    assert visual_map["visual_family"].str.contains("g_in_network variant", na=False).any()
    assert visual_map["visual_family"].str.contains("g_out_network variant", na=False).any()


def test_semantic_variable_map_marks_available_and_target_network_variables() -> None:
    root = toy_root()
    write_toy_inventory_files(root)

    written = build_data_inventory(root=root, output_dir=root / "abm_v3" / "data_inventory")
    semantic_map = pd.read_csv(written["semantic_variable_map"]).set_index("canonical_variable")

    assert semantic_map.loc["network_green_exposure", "preferred_source"] == "missing_or_not_yet_constructed"
    assert "canonical target concept" in semantic_map.loc["network_green_exposure", "economic_meaning"].lower()
    assert semantic_map.loc["g_in_network", "preferred_source"].endswith("abm_v3_phase_space_state_panel_1995_2016.parquet")
    assert semantic_map.loc["g_out_network", "preferred_source"].endswith("abm_v3_phase_space_state_panel_1995_2016.parquet")
    assert semantic_map.loc["brown_centrality", "preferred_source"] == "missing_or_not_yet_constructed"
    assert semantic_map.loc["capability_ecosystem_exposure", "preferred_source"] == "missing_or_not_yet_constructed"
    assert "may derive a fallback" in semantic_map.loc["g_local", "caveats"]


def test_data_inventory_cli_command_is_registered() -> None:
    parser = build_parser()

    args = parser.parse_args(["data-inventory", "--focus", "abm_v3", "--sample-rows", "3"])

    assert args.command == "data-inventory"
    assert args.focus == "abm_v3"
    assert args.sample_rows == 3

    phase_args = parser.parse_args(["phase-space-state-panel", "--start-year", "1995", "--end-year", "2016"])

    assert phase_args.command == "phase-space-state-panel"
    assert phase_args.start_year == 1995
    assert phase_args.end_year == 2016
