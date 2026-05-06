"""ABM v3 architecture for historically calibrated green-transition modelling.

The package follows the ABM v3 model contract and data reference: agents are
Eora country-sector nodes keyed by the full ``country_sector`` label, dynamics
are quantity-based, and collapse is diagnosed rather than prevented.
"""

from src.abm_v3.config import ABMV3Config
from src.abm_v3.model import ABMV3Model
from src.abm_v3.paths import ABMV3Paths
from src.abm_v3.state import ABMState, ABMStateMetadata

__all__ = [
    "ABMV3Config",
    "ABMV3Model",
    "ABMV3Paths",
    "ABMState",
    "ABMStateMetadata",
]
