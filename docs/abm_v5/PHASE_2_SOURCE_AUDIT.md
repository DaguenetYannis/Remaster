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

## Phase 2.4 Note

Phase 2.4 maps processed Atlas capability variables into canonical ABM v5
capability and ecosystem state columns for 1995-2016. Capabilities are treated
as productive feasibility constraints, not proof of low-carbon production.
`green_capability` remains distinct from both `local_greenness` and future
`network_green_exposure`.

Missing design-target variables are preserved as explicit null columns with
availability flags set to false. Accounting validity flags from Phase 2.3 are
carried through without correction or imputation. This phase does not compute
capability accumulation, directed ecosystem metrics, network green exposure,
brown centrality, phase-space positions, regimes, thresholds, simulations,
scenarios, or plots.

## Phase 2.5 Note

Phase 2.5 constructs two distinct historical network panels for 1995-2016:
production-network supplier-buyer edges from raw Eora `T`, and node-level
network diagnostics derived from those production links plus local accounting
variables. The raw `T` convention follows the repository's validated supplier
orientation: rows are suppliers and columns are buyers.

This phase keeps production feasibility structure separate from embodied-carbon
diagnostics. Supplier edges are based on `T`, not embodied emissions. The
edge-level `embodied_emissions_flow` column is an explicit null placeholder in
this subphase; ET or Leontief-based embodied-carbon construction is not folded
into supplier weights. `network_green_exposure`, `brown_centrality`, and
`supplier_lock_in` are diagnostics only, not behavioural supplier substitution
rules.

Phase 2.5 does not implement supplier choice, supplier rewiring, production
feasibility calculations, threshold discovery, regime discovery, simulations,
scenarios, or plots.

### Phase 2.5 Patch: Supplier Candidate Compaction

The real 1995 raw `T` smoke check produced more than 24 million positive
production links for one year. Saving the full dense positive `T` edge panel as
the ABM v5 working supplier-network layer would make the historical backbone
unnecessarily large and difficult to inspect.

ABM v5 therefore follows the useful construction separation already learned in
ABM v4: raw supplier edges are yearly construction material, while the canonical
working output is a compact supplier candidate base. The canonical supplier
network file is now
`data/abm_v5/supplier_network/supplier_candidate_panel_1995_2016.parquet`.

Supplier candidates encode opportunity structure only. They are not supplier
choice, rewiring, substitution, production feasibility, or scenario logic.
Historical candidates are retained from raw `T` using top-minimum, weight, and
coverage rules. Same-sector fallback candidates are added only as opportunity
candidates when coverage or supplier-count criteria are not met, and they do
not receive fabricated transaction values, supplier weights, or technical
coefficients.

## Phase 2.6 Note

Phase 2.6 joins the observed accounting, capability, and production-network
diagnostic panels into
`data/abm_v5/phase_space/historical_phase_space_panel_1995_2016.parquet`.
This is an observed state-space construction step only. It computes the
sector-year emissions-intensity gap, stores available empirical phase-space
coordinates, and records completeness/readiness metadata while keeping
unavailable design-target variables explicit.

Regime fields remain nullable placeholders for Phase 3. Phase 2.6 does not
discover regimes, compute thresholds, assign labels, run clustering, simulate
transition dynamics, execute scenarios, or treat supplier candidates as active
supplier-choice behaviour.
