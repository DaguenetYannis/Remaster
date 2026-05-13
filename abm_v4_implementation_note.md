# ABM v4 Implementation Note

## Implemented in Phase 1

- Created `src/abm_v4/` as a new namespace without modifying ABM v1, v2, or v3.
- Added `ABMV4Config` for explicit base parameters.
- Added `ABMV4Paths` for project-relative ABM v4 path construction.
- Added schema contracts and inspectable validation results for state and ecosystem tables.
- Added small foundation modules for state discovery, ecosystems, suppliers, capabilities, production, emissions, scenarios, simulation readiness, diagnostics, and validation.
- Added toy tests under `tests/abm_v4/`.
- Added `scripts/run_abm_v4_base.py` as a readiness entry point.

## Intentional Simplifications

- No state panel is generated in Phase 1.
- No supplier network is constructed in Phase 1.
- No scenario simulation is run in Phase 1.
- No ecosystem-specific capability stocks are implemented; those remain a v5 extension.
- The productive ecosystem layer is represented only by explicit schema and assignment contracts.

## Extension from ABM v3

ABM v4 introduces a separate namespace and output root. It can inspect ABM v3 outputs as preferred inputs, but writes only under `data/abm_v4/` when explicitly run.

## How to Run

```powershell
python scripts/run_abm_v4_base.py
```

The script creates only ABM v4 output folders and reports whether an acceptable state source exists.

## Output Root

```text
data/abm_v4/
```

## Remaining for v5 or Later Phases

- Build the full ABM v4 state panel.
- Implement productive ecosystem assignment and adjacency construction.
- Implement supplier opportunity sets and supplier-side rewiring.
- Implement iterative production propagation.
- Write emissions decomposition outputs.
- Add policy scenarios after the base model works.
