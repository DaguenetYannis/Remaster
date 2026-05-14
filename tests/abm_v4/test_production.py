from pathlib import Path
from uuid import uuid4

import polars as pl

from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.production import (
    ProductionFeasibilityEngine,
    input_feasibility,
    realized_output,
)


def toy_root() -> Path:
    return Path("tmp") / "abm_v4_production_tests" / uuid4().hex


def toy_state_panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "country_sector": ["S1", "S2", "B1", "B2"],
            "Year": [1995, 1995, 1995, 1995],
            "X_observed": [100.0, 10.0, 200.0, 100.0],
        }
    )


def toy_t_matrix() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "__index_level_0__": ["S1", "S2"],
            "B1": [20.0, 30.0],
            "B2": [10.0, 0.0],
        }
    )


def toy_supplier_weights() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "buyer_country_sector": ["B1", "B1", "B2"],
            "supplier_country_sector": ["S1", "S2", "S1"],
            "supplier_type": ["historical", "ecosystem_feasible", "same_sector_foreign"],
            "updated_weight": [0.4, 0.6, 1.0],
        }
    )


def write_toy_inputs(paths: ABMV4Paths) -> None:
    state_path = paths.state_panel_path(1995, 2016)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    toy_state_panel().write_parquet(state_path)
    t_path = paths.data_root / "parquet" / "1995" / "T.parquet"
    t_path.parent.mkdir(parents=True, exist_ok=True)
    toy_t_matrix().write_parquet(t_path)
    paths.supplier_updated_weights_path.parent.mkdir(parents=True, exist_ok=True)
    toy_supplier_weights().write_parquet(paths.supplier_updated_weights_path)


def test_realized_output_uses_input_feasibility_cap() -> None:
    feasibility = input_feasibility(
        total_input_available=50.0,
        total_input_required=100.0,
        epsilon=1e-9,
    )

    assert round(feasibility, 2) == 0.50
    assert round(realized_output(200.0, feasibility), 2) == 100.00
    assert realized_output(200.0, 1.5) == 200.0


def test_technical_coefficients_divide_t_by_buyer_output() -> None:
    engine = ProductionFeasibilityEngine(paths=ABMV4Paths(project_root=toy_root()))
    state = toy_state_panel()

    coefficients = engine.compute_technical_coefficients(toy_t_matrix(), state)

    assert abs(
        coefficients.filter(pl.col("country_sector") == "B1")[
            "input_coefficient_total"
        ].item()
        - 0.25
    ) < 1e-9


def test_input_requirements_use_coefficients_times_desired_output() -> None:
    engine = ProductionFeasibilityEngine(paths=ABMV4Paths(project_root=toy_root()))
    coefficients = engine.compute_technical_coefficients(toy_t_matrix(), toy_state_panel())

    requirements = engine.compute_input_requirements(toy_state_panel(), coefficients)

    assert abs(
        requirements.filter(pl.col("country_sector") == "B1")[
            "input_required_total"
        ].item()
        - 50.0
    ) < 1e-6


def test_input_availability_uses_updated_supplier_weights() -> None:
    engine = ProductionFeasibilityEngine(paths=ABMV4Paths(project_root=toy_root()))
    coefficients = engine.compute_technical_coefficients(toy_t_matrix(), toy_state_panel())
    requirements = engine.compute_input_requirements(toy_state_panel(), coefficients)

    availability = engine.compute_input_availability(requirements, toy_supplier_weights())

    assert abs(
        availability.filter(
            (pl.col("buyer_country_sector") == "B1")
            & (pl.col("supplier_country_sector") == "S2")
        )["input_avail"].item()
        - 30.0
    ) < 1e-6


def test_full_availability_gives_feasible_output_equal_to_desired() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_inputs(paths)

    panel = ProductionFeasibilityEngine(paths=paths).build_feasibility_panel(year=1995)

    b1 = panel.filter(pl.col("country_sector") == "B1").row(0, named=True)
    assert abs(b1["X_feasible"] - b1["X_desired"]) < 1e-6


def test_insufficient_availability_reduces_feasible_output() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_inputs(paths)
    weak_weights = pl.DataFrame(
        {
            "buyer_country_sector": ["B1"],
            "supplier_country_sector": ["S1"],
            "supplier_type": ["historical"],
            "updated_weight": [0.5],
        }
    )
    weak_weights.write_parquet(paths.supplier_updated_weights_path)

    panel = ProductionFeasibilityEngine(paths=paths).build_feasibility_panel(year=1995)

    b1 = panel.filter(pl.col("country_sector") == "B1").row(0, named=True)
    assert b1["input_feasibility"] < 1.0
    assert b1["X_feasible"] < b1["X_desired"]


def test_supplier_pressure_uses_supplier_output_capacity_proxy() -> None:
    engine = ProductionFeasibilityEngine(paths=ABMV4Paths(project_root=toy_root()))
    coefficients = engine.compute_technical_coefficients(toy_t_matrix(), toy_state_panel())
    requirements = engine.compute_input_requirements(toy_state_panel(), coefficients)
    availability = engine.compute_input_availability(requirements, toy_supplier_weights())

    pressure = engine.compute_supplier_capacity_pressure(availability, toy_state_panel())

    assert round(
        pressure.filter(
            (pl.col("buyer_country_sector") == "B1")
            & (pl.col("supplier_country_sector") == "S2")
        )["supplier_capacity_pressure"].item(),
        2,
    ) == 3.0


def test_feasibility_report_counts_constrained_nodes() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_inputs(paths)
    weak_weights = pl.DataFrame(
        {
            "buyer_country_sector": ["B1"],
            "supplier_country_sector": ["S1"],
            "supplier_type": ["historical"],
            "updated_weight": [0.5],
        }
    )
    weak_weights.write_parquet(paths.supplier_updated_weights_path)
    engine = ProductionFeasibilityEngine(paths=paths)
    panel = engine.build_feasibility_panel(year=1995)

    report = engine.build_production_feasibility_report(panel)

    assert report["nodes_with_input_feasibility_below_1"].item() >= 1


def test_production_feasibility_build_does_not_write_without_explicit_output() -> None:
    root = toy_root()
    paths = ABMV4Paths(project_root=root)
    write_toy_inputs(paths)

    ProductionFeasibilityEngine(paths=paths).build_feasibility_panel(year=1995)

    assert not paths.production_feasibility_panel_path.exists()
    assert not paths.production_feasibility_report_path.exists()
