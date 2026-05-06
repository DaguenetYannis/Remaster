import pandas as pd

from src.abm_v2.metrics import ABMMetricsBuilder, MetricsConfig


def test_parse_country_sector():
    config = MetricsConfig()
    builder = ABMMetricsBuilder(config)

    country, sector = builder._parse_country_sector(
        "FRA | FRA | Industries | Electricity, Gas and Water"
    )

    assert country == "FRA"
    assert sector == "Electricity, Gas and Water"


def test_derive_missing_green_metrics():
    config = MetricsConfig()
    builder = ABMMetricsBuilder(config)

    df = pd.DataFrame({
        "Year": [2010],
        "country_sector": ["FRA | FRA | Industries | Agriculture"],
        "Country": ["FRA"],
        "Sector": ["Agriculture"],
        "EI": [0.25],
        "X": [100.0],
        "D": [90.0],
        "M": [80.0],
    })

    result = builder._derive_missing_metrics(df)

    assert "g_local" in result.columns
    assert "NG" in result.columns
    assert "inventory_base" in result.columns
    assert "capacity_base" in result.columns
    assert result.loc[0, "capacity_base"] == 110.0