from __future__ import annotations


def input_feasibility(
    total_input_available: float,
    total_input_required: float,
    epsilon: float,
) -> float:
    """Compute input feasibility from explicit requirements and availability."""
    return total_input_available / (total_input_required + epsilon)


def realized_output(
    desired_output: float,
    feasibility: float,
) -> float:
    """Compute realized output from desired output and input feasibility."""
    return desired_output * min(1.0, feasibility)
