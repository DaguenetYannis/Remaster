from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.abm_v5.config import ComplexityLayer
from src.abm_v5.ontology import MechanismName


class TheoreticalPillar(str, Enum):
    """Theoretical pillars anchoring ABM v5 design."""

    ARTHUR_COMPLEXITY = "arthur_complexity"
    HIDALGO_PRODUCT_SPACE = "hidalgo_product_space"
    ARROW_OF_DEVELOPMENT = "arrow_of_development"
    INOUE_SUPPLY_CHAIN_ABM = "inoue_supply_chain_abm"
    CHUDZIAK_RECURSIVE_COMPLEXITY = "chudziak_recursive_complexity"
    GALLEGATI_ABM_COMPLEXITY = "gallegati_abm_complexity"
    INPUT_OUTPUT_LEONTIEF = "input_output_leontief"
    REMASTER_GREENNESS = "remaster_greenness"


@dataclass(frozen=True)
class TheoryMapping:
    """Mapping from one theory pillar to ABM v5 design implications."""

    pillar: TheoreticalPillar
    short_claim: str
    design_implication: str
    linked_mechanisms: tuple[MechanismName, ...]
    linked_complexity_layers: tuple[ComplexityLayer, ...]

    def validate(self) -> None:
        """Validate theory mapping metadata."""
        if not isinstance(self.pillar, TheoreticalPillar):
            raise ValueError("pillar must be TheoreticalPillar.")
        if not self.short_claim:
            raise ValueError("short_claim cannot be empty.")
        if not self.design_implication:
            raise ValueError("design_implication cannot be empty.")
        if not self.linked_mechanisms:
            raise ValueError("linked_mechanisms cannot be empty.")
        if any(not isinstance(mechanism, MechanismName) for mechanism in self.linked_mechanisms):
            raise ValueError("linked_mechanisms must contain MechanismName values.")
        if not self.linked_complexity_layers:
            raise ValueError("linked_complexity_layers cannot be empty.")
        if any(not isinstance(layer, ComplexityLayer) for layer in self.linked_complexity_layers):
            raise ValueError("linked_complexity_layers must contain ComplexityLayer values.")


def get_theory_mappings() -> tuple[TheoryMapping, ...]:
    """Return theory-to-design mappings for ABM v5 Phase 1."""
    all_mechanisms = tuple(MechanismName)
    return (
        TheoryMapping(
            pillar=TheoreticalPillar.ARTHUR_COMPLEXITY,
            short_claim=(
                "Economies are adaptive, non-equilibrium systems where agents "
                "react to aggregate patterns they collectively create."
            ),
            design_implication=(
                "ABM_v5 must update feedback-relevant variables recursively "
                "inside the yearly loop rather than treating them as one-time diagnostics."
            ),
            linked_mechanisms=(
                MechanismName.PHASE_SPACE_REGIME_SWITCHING,
                MechanismName.NETWORK_GREEN_EXPOSURE_FEEDBACK,
                MechanismName.BROWN_LOCK_IN_FEEDBACK,
                MechanismName.BOUNDED_SUPPLIER_SEARCH,
            ),
            linked_complexity_layers=(
                ComplexityLayer.REGIME_DISCOVERY,
                ComplexityLayer.PRODUCTION_NETWORK,
                ComplexityLayer.SUPPLIER_ADAPTATION,
            ),
        ),
        TheoryMapping(
            pillar=TheoreticalPillar.HIDALGO_PRODUCT_SPACE,
            short_claim=(
                "Productive transformation is constrained by existing capabilities "
                "and proximity in capability space."
            ),
            design_implication=(
                "ABM_v5 must represent green transition as capability-constrained "
                "movement, not arbitrary emissions reduction."
            ),
            linked_mechanisms=(
                MechanismName.CAPABILITY_ACCUMULATION,
                MechanismName.EMISSIONS_INTENSITY_TRANSITION,
            ),
            linked_complexity_layers=(ComplexityLayer.CAPABILITY, ComplexityLayer.ACCOUNTING),
        ),
        TheoryMapping(
            pillar=TheoreticalPillar.ARROW_OF_DEVELOPMENT,
            short_claim=(
                "Development paths are directed because some products and activities "
                "systematically precede others through shared capability ecosystems."
            ),
            design_implication=(
                "ABM_v5 must distinguish undirected proximity from directed green "
                "precedence and transition-sector positioning."
            ),
            linked_mechanisms=(
                MechanismName.DIRECTED_ECOSYSTEM_MOVEMENT,
                MechanismName.CAPABILITY_ACCUMULATION,
            ),
            linked_complexity_layers=(ComplexityLayer.CAPABILITY,),
        ),
        TheoryMapping(
            pillar=TheoreticalPillar.INOUE_SUPPLY_CHAIN_ABM,
            short_claim=(
                "Supplier-customer networks propagate constraints and shocks, "
                "especially under substitution difficulty and complex network structure."
            ),
            design_implication=(
                "ABM_v5 must use production-network propagation, bounded supplier "
                "substitution, and recursive production feasibility."
            ),
            linked_mechanisms=(
                MechanismName.PRODUCTION_FEASIBILITY,
                MechanismName.SUPPLIER_SUBSTITUTION,
                MechanismName.BOUNDED_SUPPLIER_SEARCH,
            ),
            linked_complexity_layers=(
                ComplexityLayer.PRODUCTION_NETWORK,
                ComplexityLayer.SUPPLIER_ADAPTATION,
            ),
        ),
        TheoryMapping(
            pillar=TheoreticalPillar.CHUDZIAK_RECURSIVE_COMPLEXITY,
            short_claim=(
                "ABMs should not rely mainly on moment-matching; they should uncover "
                "mechanisms, feedback loops, impact channels, and the effect of "
                "increasing structural complexity."
            ),
            design_implication=(
                "ABM_v5 must be implemented through a complexity ladder, ablation "
                "tests, and mechanism-level validation."
            ),
            linked_mechanisms=all_mechanisms,
            linked_complexity_layers=(
                ComplexityLayer.ACCOUNTING,
                ComplexityLayer.PRODUCTION_NETWORK,
                ComplexityLayer.CAPABILITY,
                ComplexityLayer.SUPPLIER_ADAPTATION,
                ComplexityLayer.REGIME_DISCOVERY,
            ),
        ),
        TheoryMapping(
            pillar=TheoreticalPillar.GALLEGATI_ABM_COMPLEXITY,
            short_claim=(
                "Complexity economics shifts attention from equilibrium states to "
                "the processes driving non-equilibrium economic dynamics."
            ),
            design_implication=(
                "ABM_v5 should be presented as a generative process model, not a "
                "static classifier or equilibrium simulator."
            ),
            linked_mechanisms=(
                MechanismName.NETWORK_GREEN_EXPOSURE_FEEDBACK,
                MechanismName.BROWN_LOCK_IN_FEEDBACK,
                MechanismName.PHASE_SPACE_REGIME_SWITCHING,
            ),
            linked_complexity_layers=(
                ComplexityLayer.REGIME_DISCOVERY,
                ComplexityLayer.PRODUCTION_NETWORK,
                ComplexityLayer.SUPPLIER_ADAPTATION,
            ),
        ),
        TheoryMapping(
            pillar=TheoreticalPillar.INPUT_OUTPUT_LEONTIEF,
            short_claim=(
                "Input-output economics provides structural discipline for production "
                "interdependence, input requirements, and accounting consistency."
            ),
            design_implication=(
                "ABM_v5 must preserve the distinction between T as production "
                "structure and ET as embodied-carbon exposure."
            ),
            linked_mechanisms=(
                MechanismName.PRODUCTION_FEASIBILITY,
                MechanismName.NETWORK_GREEN_EXPOSURE_FEEDBACK,
                MechanismName.EMISSIONS_INTENSITY_TRANSITION,
            ),
            linked_complexity_layers=(
                ComplexityLayer.ACCOUNTING,
                ComplexityLayer.PRODUCTION_NETWORK,
            ),
        ),
        TheoryMapping(
            pillar=TheoreticalPillar.REMASTER_GREENNESS,
            short_claim=(
                "Green-ness is relational: a node's environmental position depends "
                "on local emissions intensity and network-embedded carbon exposure."
            ),
            design_implication=(
                "ABM_v5 must keep local green-ness, network green exposure, and "
                "embodied-carbon diagnostics conceptually distinct."
            ),
            linked_mechanisms=(
                MechanismName.NETWORK_GREEN_EXPOSURE_FEEDBACK,
                MechanismName.EMISSIONS_INTENSITY_TRANSITION,
                MechanismName.PHASE_SPACE_REGIME_SWITCHING,
            ),
            linked_complexity_layers=(
                ComplexityLayer.ACCOUNTING,
                ComplexityLayer.PRODUCTION_NETWORK,
                ComplexityLayer.REGIME_DISCOVERY,
            ),
        ),
    )


def validate_theory_mappings() -> None:
    """Validate that theory mappings are complete and internally typed."""
    mappings = get_theory_mappings()
    for mapping in mappings:
        mapping.validate()

    pillars = tuple(mapping.pillar for mapping in mappings)
    if set(pillars) != set(TheoreticalPillar):
        raise ValueError("theory mappings must cover every TheoreticalPillar.")
    if len(pillars) != len(set(pillars)):
        raise ValueError("each TheoreticalPillar must be covered exactly once.")

    for mapping in mappings:
        for mechanism in mapping.linked_mechanisms:
            if not isinstance(mechanism, MechanismName):
                raise ValueError(f"{mapping.pillar.value} has invalid linked mechanism.")
        for layer in mapping.linked_complexity_layers:
            if not isinstance(layer, ComplexityLayer):
                raise ValueError(f"{mapping.pillar.value} has invalid linked complexity layer.")
