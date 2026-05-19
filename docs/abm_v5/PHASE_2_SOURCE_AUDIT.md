# ABM v5 Phase 2.1 Source Audit

## Purpose

Phase 2.1 creates a source inventory and input registry for the ABM v5
historical backbone. It identifies required Eora and Atlas inputs and optional
ABM v3 and ABM v4 construction references without loading or transforming data.

## Required Eora Sources

Required yearly Eora matrices for 1995-2016 are:

- `data/parquet/{year}/T.parquet`
- `data/parquet/{year}/FD.parquet`
- `data/parquet/{year}/Q.parquet`

Required yearly Eora labels for 1995-2016 are:

- `data/raw/{year}/labels_T.txt`
- `data/raw/{year}/labels_FD.txt`
- `data/raw/{year}/labels_Q.txt`

Optional yearly Eora matrices are `VA.parquet` and `QY.parquet`.

## Atlas Sources

Required processed Atlas sources are:

- `data/atlas/processed/atlas_eora26_sector_capabilities_1995_2016.parquet`
- `data/atlas/processed/eora26_country_sector_labels.csv`

## ABM v3 And ABM v4 Reference Sources

Optional ABM v3 references are:

- `data/abm_v3/phase_space/abm_v3_phase_space_state_panel_1995_2016.parquet`
- `data/abm_v3/inputs/`

Optional ABM v4 references are:

- `data/abm_v4/`
- `src/abm_v4/`
- `tests/abm_v4/`

ABM v4 is a construction reference only. Its panel-building, supplier-edge,
capability-join, diagnostic, and test patterns may inform ABM v5 construction,
but ABM v5 must remain a separate clean layer and must not import ABM v4
conceptual assumptions blindly.

## ABM v4 Construction Patterns Inspected

The filename-level audit found ABM v4 modules for paths, config, schemas,
state construction, ecosystem mapping, capabilities, suppliers, production,
emissions, diagnostics, validation, simulation, scenarios, and reporting. The
corresponding ABM v4 tests cover state, schemas, capabilities, suppliers,
production, emissions, diagnostics, multiyear validation, electricity audits,
final artifacts, and scenario-adjacent smoke tests.

For ABM v5, reusable construction ideas are limited to implementation patterns:
centralized paths, explicit schema contracts, visible diagnostics, source
coverage tests, supplier-edge construction checks, capability join audits, and
validation report structure. ABM v4 simulation, scenario, and final-report logic
are not part of Phase 2.1 and are not imported into ABM v5.

## Phase 2.1 Boundaries

Phase 2.1 does not load, inspect, or transform data contents. It does not build
historical panels, construct identities, compute accounting variables, join
capabilities, build edges, discover regimes, estimate thresholds, run
simulation, execute scenarios, or create plots.

## Phase 2.2 Note

Phase 2.2 builds the canonical ABM v5 country-sector identity panel with one row
per `country_sector`. It prefers
`data/atlas/processed/eora26_country_sector_labels.csv` when present and valid,
and otherwise falls back to `data/raw/1995/labels_T.txt`.

The output is limited to `data/abm_v5/inputs/agent_identity.parquet` and
`data/abm_v5/validation/agent_identity_validation.json`. This phase does not
construct yearly accounting variables, capabilities, edges, phase-space
variables, regimes, simulations, scenarios, or plots.

## Phase 2.3 Note

Phase 2.3 constructs the observed historical accounting state panel for
1995-2016. It uses canonical ABM v5 identity rows, Eora `T`, `FD`, and `Q`
matrices, and raw `labels_T` and `labels_Q` files. Following the corrected ABM
v3 orientation, local output is computed as row-sum intermediate output plus
row-sum final demand.

The accounting panel contains local variables only: `output`, `final_demand`,
`emissions`, `emissions_intensity`, and `local_greenness`. It does not compute
network green exposure, brown centrality, phase-space position, regimes,
thresholds, simulations, scenarios, or plots.
