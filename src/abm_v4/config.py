from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SupplierFrictionConfig:
    """Friction hierarchy for supplier opportunity types."""

    phi_historical: float = 0.10
    phi_same_sector: float = 0.50
    phi_ecosystem: float = 1.00


@dataclass(frozen=True)
class SupplierChoiceConfig:
    """Weights and update rates for supplier adaptation."""

    alpha_green_advantage: float = 1.0
    alpha_reliability: float = 1.0
    alpha_capacity_available: float = 1.0
    alpha_ecosystem_proximity: float = 1.0
    alpha_historical_tie_strength: float = 1.0
    alpha_friction: float = 1.0
    lambda_weight_update: float = 0.10
    p_rewire_base: float = 0.01
    p_rewire_stress: float = 0.05
    p_rewire_green_gap: float = 0.05


@dataclass(frozen=True)
class CapabilityConfig:
    """Capability accumulation parameters for ABM v4."""

    cap_max: float = 1.0
    gcap_max: float = 1.0
    delta_cap_param: float = 0.05
    delta_gcap_param: float = 0.05
    k_cap: float = 5.0
    k_gcap: float = 5.0
    tau_cap: float = 0.5
    tau_gcap: float = 0.5


@dataclass(frozen=True)
class EmissionsConfig:
    """Emissions intensity update parameters."""

    beta_0: float = 0.0
    beta_log_ei: float = 0.05
    beta_green_capability: float = 0.10
    beta_network_green_exposure: float = 0.10
    beta_general_capability: float = 0.05
    beta_brown_centrality: float = 0.10
    ei_min: float = 1e-9


@dataclass(frozen=True)
class ABMV4Config:
    """Top-level configuration for ABM v4 Phase 1."""

    start_year: int = 1995
    end_year: int = 2016
    epsilon: float = 1e-9
    supplier_friction: SupplierFrictionConfig = SupplierFrictionConfig()
    supplier_choice: SupplierChoiceConfig = SupplierChoiceConfig()
    capability: CapabilityConfig = CapabilityConfig()
    emissions: EmissionsConfig = EmissionsConfig()
