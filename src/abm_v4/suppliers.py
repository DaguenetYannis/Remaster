from __future__ import annotations

from dataclasses import dataclass


SUPPLIER_TYPES: tuple[str, ...] = (
    "historical",
    "same_sector_foreign",
    "ecosystem_feasible",
)


@dataclass(frozen=True)
class SupplierOpportunity:
    """Potential supplier relation considered by ABM v4."""

    buyer_country_sector: str
    supplier_country_sector: str
    supplier_type: str
    friction: float

    def is_supported_type(self) -> bool:
        return self.supplier_type in SUPPLIER_TYPES
