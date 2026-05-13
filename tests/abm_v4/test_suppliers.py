from src.abm_v4.suppliers import SupplierOpportunity


def test_supplier_opportunity_uses_explicit_type_vocabulary() -> None:
    opportunity = SupplierOpportunity(
        buyer_country_sector="FRA|Agriculture",
        supplier_country_sector="DEU|Agriculture",
        supplier_type="same_sector_foreign",
        friction=0.50,
    )

    assert opportunity.is_supported_type()
