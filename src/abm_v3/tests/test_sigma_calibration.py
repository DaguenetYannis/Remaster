from __future__ import annotations

from src.abm_v3.calibration.substitution_model import SubstitutionFrictionModel


def test_sigma_grid_returns_best_sigma_and_results_table() -> None:
    model = SubstitutionFrictionModel(sigma_grid=(0.0, 0.5, 1.0))

    model.fit_grid(lambda sigma: {"output_validation_loss": abs(sigma - 0.5), "collapse_penalty": 0.0})
    results = model.get_results()

    assert model.get_sigma() == 0.5
    assert results["sigma"].tolist() == [0.0, 0.5, 1.0]
    assert "score" in results.columns
