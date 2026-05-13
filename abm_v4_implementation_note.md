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

## Module Status Audit

| module | current status | what it currently does | what data it expects | what remains before real-data use | related tests |
| --- | --- | --- | --- | --- | --- |
| `src/abm_v4/__init__.py` | real-data-ready | Exposes `ABMV4Config` and `ABMV4Paths` as the public package foundation. | None. | Expand exports only when additional modules become stable. | Import check through `tests/abm_v4`. |
| `src/abm_v4/config.py` | real-data-ready | Defines explicit dataclass parameters for suppliers, capabilities, emissions, and the Phase 1 run window. | None. | Calibrate defaults against real data before simulation use. | Indirectly covered by capability, emissions, and simulation tests. |
| `src/abm_v4/paths.py` | real-data-ready | Defines project-relative ABM v4 output paths, source candidates, and explicit output-directory creation. | Existing repository folders under `data/`; no generated v4 data required. | Add exact source paths as real state/edge builders mature. | `tests/abm_v4/test_paths.py`. |
| `src/abm_v4/schemas.py` | real-data-ready | Defines required/optional column contracts and non-raising schema validation results for state and ecosystem tables. | Polars dataframes supplied by callers. | Add type/range checks after real state construction exists. | `tests/abm_v4/test_schemas.py`. |
| `src/abm_v4/state.py` | toy-data-tested only | Discovers the first available state source using the v4 priority order and returns diagnostics. | Candidate local files from ABM v3, final panels, or legacy ABM. | Build real state-panel transformations and write explicit missing-column diagnostics. | `tests/abm_v4/test_state.py`. |
| `src/abm_v4/ecosystem.py` | skeleton only | Defines an `EcosystemAssignment` dataclass and validates source vocabulary. | Manually supplied assignment values. | Build ecosystem mapping from Atlas or transparent Eora-sector fallback and write assignment reports. | `tests/abm_v4/test_ecosystem.py`. |
| `src/abm_v4/suppliers.py` | skeleton only | Defines supplier opportunity metadata and the supported supplier-type vocabulary. | Manually supplied buyer/supplier relation values. | Discover edge sources, build opportunity sets, compute attractiveness, softmax probabilities, and rewiring diagnostics. | `tests/abm_v4/test_suppliers.py`. |
| `src/abm_v4/capabilities.py` | toy-data-tested only | Implements sigmoid and scalar general/green capability increment formulas. | Scalar capability and exposure values. | Apply formulas to real node-year panels and validate exposure variables. | `tests/abm_v4/test_capabilities.py`. |
| `src/abm_v4/production.py` | toy-data-tested only | Implements scalar input feasibility and realized-output formulas. | Scalar total input availability/requirements. | Implement matrix/input-requirement logic and iterative propagation diagnostics. | `tests/abm_v4/test_production.py`. |
| `src/abm_v4/emissions.py` | incomplete or risky | Implements emissions identity, scalar EI update, and an emissions decomposition dataclass. | Positive scalar EI and node-level covariates supplied by callers. | Guard against non-positive EI, vectorize over real panels, and write decomposition outputs. | `tests/abm_v4/test_emissions.py`. |
| `src/abm_v4/scenarios.py` | skeleton only | Defines scenario metadata only. | Manually supplied scenario metadata. | Add policy dataclasses after the base model works; no demand policy in base model. | No direct test yet. |
| `src/abm_v4/simulation.py` | toy-data-tested only | Reports whether state sources exist for a future base-model run. | Local candidate source paths. | Orchestrate real state construction, supplier construction, production, emissions, and diagnostics. | `tests/abm_v4/test_simulation_smoke.py`. |
| `src/abm_v4/diagnostics.py` | real-data-ready | Defines diagnostic messages and a non-mutating path audit for real repository source availability. | Existing/missing local source paths; it only checks path existence. | Add dataframe-level diagnostics once real transformations are implemented. | `tests/abm_v4/test_diagnostics.py`. |
| `src/abm_v4/validation.py` | skeleton only | Defines a simple validation message dataclass. | Manually supplied validation values. | Add real validation checks for state schema, source provenance, emissions decomposition, and bad-transition flags. | No direct test yet. |

## Extension from ABM v3

ABM v4 introduces a separate namespace and output root. It can inspect ABM v3 outputs as preferred inputs, but writes only under `data/abm_v4/` when explicitly run.

## How to Run

Dry-run source audit, without creating `data/abm_v4/`:

```powershell
python scripts/run_abm_v4_base.py --dry-run
```

Readiness check without creating output folders:

```powershell
python scripts/run_abm_v4_base.py
```

Create ABM v4 output folders explicitly:

```powershell
python scripts/run_abm_v4_base.py --create-output-dirs
```

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
