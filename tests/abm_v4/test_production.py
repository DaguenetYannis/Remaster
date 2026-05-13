from src.abm_v4.production import input_feasibility, realized_output


def test_realized_output_uses_input_feasibility_cap() -> None:
    feasibility = input_feasibility(
        total_input_available=50.0,
        total_input_required=100.0,
        epsilon=1e-9,
    )

    assert round(feasibility, 2) == 0.50
    assert round(realized_output(200.0, feasibility), 2) == 100.00
    assert realized_output(200.0, 1.5) == 200.0
