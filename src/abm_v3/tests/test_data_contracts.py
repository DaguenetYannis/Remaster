from __future__ import annotations

import pandas as pd

from src.abm_v3.data_contracts import DataContractValidator


def test_detects_missing_country_sector() -> None:
    validator = DataContractValidator()
    result = validator.validate_country_sector_key(pd.DataFrame({"Year": [1995]}))
    assert not result.passed
    assert "country_sector" in result.message


def test_detects_duplicate_country_sector_year_rows() -> None:
    validator = DataContractValidator()
    df = pd.DataFrame(
        {
            "country_sector": ["A | A | Industries | Agriculture"] * 2,
            "Year": [1995, 1995],
        }
    )
    result = validator.validate_no_duplicate_nodes(df)
    assert not result.passed
    assert result.details is not None
    assert result.details["duplicate_count"] == 1


def test_validates_year_coverage() -> None:
    validator = DataContractValidator()
    df = pd.DataFrame({"country_sector": ["A", "A"], "Year": [1995, 1997]})
    result = validator.validate_year_coverage(df, 1995, 1997)
    assert not result.passed
    assert result.details is not None
    assert result.details["missing_years"] == [1996]
