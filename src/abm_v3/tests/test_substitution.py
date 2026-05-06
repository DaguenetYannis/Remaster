from __future__ import annotations

import pandas as pd

from src.abm_v3.dynamics.substitution import apply_supplier_substitution, select_eligible_suppliers


def supplier_pool() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "country_sector": ["A", "B", "C"],
            "Sector": ["Agriculture", "Agriculture", "Manufacturing"],
            "available_surplus": [5.0, 15.0, 100.0],
        }
    )


def test_sigma_zero_gives_no_substitution() -> None:
    eligible = select_eligible_suppliers(supplier_pool(), "Agriculture")
    result = apply_supplier_substitution(10.0, eligible, sigma=0.0)
    assert result["substituted_amount"].sum() == 0.0


def test_sigma_one_allows_substitution_to_eligible_suppliers() -> None:
    eligible = select_eligible_suppliers(supplier_pool(), "Agriculture")
    result = apply_supplier_substitution(10.0, eligible, sigma=1.0)
    assert result["substituted_amount"].sum() == 10.0


def test_ineligible_suppliers_are_not_used() -> None:
    eligible = select_eligible_suppliers(supplier_pool(), "Agriculture")
    assert "C" not in eligible["country_sector"].tolist()
