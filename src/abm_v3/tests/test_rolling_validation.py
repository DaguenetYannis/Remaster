from __future__ import annotations

from src.abm_v3.calibration.validation import HistoricalValidator


def test_rolling_validation_splits() -> None:
    splits = HistoricalValidator().rolling_splits(1995, 2003, minimum_training_window=6)

    assert splits[0] == {
        "train_start_year": 1995,
        "train_end_year": 2000,
        "validation_year": 2001,
    }
    assert splits[-1]["validation_year"] == 2003
