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

- No supplier network is constructed in Phase 1.
- No scenario simulation is run in Phase 1.
- No ecosystem-specific capability stocks are implemented; those remain a v5 extension.
- The productive ecosystem layer is represented by a transparent `fallback_unknown` assignment until ecosystem mapping is implemented.

## Module Status Audit

| module | current status | what it currently does | what data it expects | what remains before real-data use | related tests |
| --- | --- | --- | --- | --- | --- |
| `src/abm_v4/__init__.py` | real-data-ready | Exposes `ABMV4Config` and `ABMV4Paths` as the public package foundation. | None. | Expand exports only when additional modules become stable. | Import check through `tests/abm_v4`. |
| `src/abm_v4/config.py` | real-data-ready | Defines explicit dataclass parameters for suppliers, capabilities, emissions, and the Phase 1 run window. | None. | Calibrate defaults against real data before simulation use. | Indirectly covered by capability, emissions, and simulation tests. |
| `src/abm_v4/paths.py` | real-data-ready | Defines project-relative ABM v4 output paths, source candidates, and explicit output-directory creation. | Existing repository folders under `data/`; no generated v4 data required. | Add exact source paths as real state/edge builders mature. | `tests/abm_v4/test_paths.py`. |
| `src/abm_v4/schemas.py` | real-data-ready | Defines required/optional column contracts and non-raising schema validation results for state and ecosystem tables. | Polars dataframes supplied by callers. | Add type/range checks after real state construction exists. | `tests/abm_v4/test_schemas.py`. |
| `src/abm_v4/state.py` | real-data-ready | Selects the highest-priority available source, canonicalizes state columns, filters 1995-2016, derives v4 variables, writes the state panel, and writes source/missingness/summary/mapping diagnostics. | Candidate local files from ABM v3, final panels, or legacy ABM. | Add real ecosystem assignment and richer type/range validation before simulation use. | `tests/abm_v4/test_state.py`. |
| `src/abm_v4/ecosystem.py` | real-data-ready | Assigns a simplified productive ecosystem to every state-panel row, writes sector mapping, ecosystem adjacency, assignment report, and sector coverage diagnostics. | ABM v4 state panel with `Sector`; Atlas/Eora ecosystem fields if present; otherwise manual Eora-sector mapping. | Replace manual mapping with product-level Eco Space or capability-overlap networks in ABM v5. | `tests/abm_v4/test_ecosystem.py`. |
| `src/abm_v4/suppliers.py` | real-data-ready | Builds observed historical supplier-buyer edge foundations from legacy embodied-emissions edges and raw Eora `T_{supplier,buyer}` matrices, attaches supplier/buyer metadata, computes historical shares and tie strength, and writes supplier-edge diagnostics. | ABM v4 state panel plus either `data/abm/edges_panel.parquet` or yearly `data/parquet/{year}/T.parquet` matrices. | Supplier opportunity sets, rewiring, production simulation, and scenarios are still not implemented. | `tests/abm_v4/test_suppliers.py`. |
| `src/abm_v4/capabilities.py` | toy-data-tested only | Implements sigmoid and scalar general/green capability increment formulas. | Scalar capability and exposure values. | Apply formulas to real node-year panels and validate exposure variables. | `tests/abm_v4/test_capabilities.py`. |
| `src/abm_v4/production.py` | toy-data-tested only | Implements scalar input feasibility and realized-output formulas. | Scalar total input availability/requirements. | Implement matrix/input-requirement logic and iterative propagation diagnostics. | `tests/abm_v4/test_production.py`. |
| `src/abm_v4/emissions.py` | incomplete or risky | Implements emissions identity, scalar EI update, and an emissions decomposition dataclass. | Positive scalar EI and node-level covariates supplied by callers. | Guard against non-positive EI, vectorize over real panels, and write decomposition outputs. | `tests/abm_v4/test_emissions.py`. |
| `src/abm_v4/scenarios.py` | skeleton only | Defines scenario metadata only. | Manually supplied scenario metadata. | Add policy dataclasses after the base model works; no demand policy in base model. | No direct test yet. |
| `src/abm_v4/simulation.py` | toy-data-tested only | Reports whether state sources exist for a future base-model run. | Local candidate source paths. | Orchestrate real state construction, supplier construction, production, emissions, and diagnostics. | `tests/abm_v4/test_simulation_smoke.py`. |
| `src/abm_v4/diagnostics.py` | real-data-ready | Defines diagnostic messages and a non-mutating path audit for real repository source availability. | Existing/missing local source paths; it only checks path existence. | Add model-level diagnostics once supplier, production, and emissions simulation are implemented. | `tests/abm_v4/test_diagnostics.py`. |
| `src/abm_v4/validation.py` | skeleton only | Defines a simple validation message dataclass. | Manually supplied validation values. | Add real validation checks for state schema, source provenance, emissions decomposition, and bad-transition flags. | No direct test yet. |

## Phase 2 Real-Data State Panel

The ABM v4 state panel has been built from the highest-priority available source:

```text
data/abm_v3/inputs/abm_v3_historical_panel_1995_2016_transpose_row_fd_without_inventory.parquet
```

Output:

```text
data/abm_v4/inputs/abm_v4_state_panel_1995_2016.parquet
```

Generated diagnostics:

```text
data/abm_v4/diagnostics/state_source_report.csv
data/abm_v4/diagnostics/state_missingness_report.csv
data/abm_v4/diagnostics/state_summary_by_year.csv
data/abm_v4/diagnostics/state_column_mapping.csv
```

Real-data build summary:

- Rows: 108,130
- Year coverage: 1995-2016
- Country-sector nodes: 4,915
- Source columns: 83
- Required missingness: no missing `country_sector`, `Year`, `X_observed`, or `EI` was reported.
- EI caveat: 6,691 rows have non-positive `EI`; these values were preserved and `log_EI` is intentionally missing for those rows.
- Network green exposure: no missing values after deriving from `g_in_network`.
- Brown centrality: 57,147 rows are missing because `pagerank`/centrality is missing for those rows.
- General capability: 44,225 rows are missing after the fallback hierarchy.
- Green capability: 44,225 rows are missing after the fallback hierarchy.
- Ecosystem fields currently use `fallback_unknown`; real ecosystem mapping remains future work.

State construction is now real-data-ready as an input-building step. It is not yet a complete ABM v4 simulation.

## Phase 3 Productive Ecosystem Layer

The ABM v4 productive ecosystem layer has been built from the real ABM v4 state panel. The inspected Atlas/Eora capability files did not contain an existing cluster/community/ecosystem field, and the HS92 clean panel/concordance did not expose a product cluster field. ABM v4 therefore used the transparent manual Eora-sector mapping.

Ecosystem source actually used:

```text
eora_sector_manual_mapping
```

Generated outputs:

```text
data/abm_v4/inputs/ecosystem_mapping.csv
data/abm_v4/inputs/ecosystem_adjacency.csv
data/abm_v4/diagnostics/ecosystem_assignment_report.csv
data/abm_v4/diagnostics/ecosystem_sector_coverage.csv
```

The state panel was also updated in place with ecosystem fields:

```text
data/abm_v4/inputs/abm_v4_state_panel_1995_2016.parquet
```

Real-data ecosystem summary:

- Country-sector nodes mapped: 4,915
- Unmapped country-sector nodes: 0
- Mapped share: 1.0
- Sectors mapped: 27
- Ecosystems used: 10
- Ecosystem source counts: `eora_sector_manual_mapping=4915`
- State-panel ecosystem nulls: 0 for `ecosystem_id`, `ecosystem_label`, and `ecosystem_source`
- Adjacency rows: 10 `same`, 22 `adjacent`, 68 `non_adjacent`

This layer is real-data-ready as a simplified contextual layer. It is intentionally not a full O'Clery Eco Space reconstruction and does not include product-to-sector capability-overlap networks. Those remain ABM v5 work.

## Phase 4A Observed Supplier-Buyer Edge Foundation

The ABM v4 historical supplier-edge foundation has been built from the best available existing edge panel after source inspection.

Selected edge source:

```text
data/abm/edges_panel.parquet
```

Selected source type:

```text
legacy_abm_edges_embodied_emissions
```

Generated outputs:

```text
data/abm_v4/interim/historical_supplier_edges.parquet
data/abm_v4/diagnostics/supplier_edge_report.csv
data/abm_v4/diagnostics/supplier_edge_schema_report.csv
```

Real-data supplier-edge summary:

- Years covered: 1995-2016
- Observed edge rows: 241,588
- Unique suppliers: 2,408
- Unique buyers: 2,449
- Unique supplier-buyer pairs: 16,856
- Total transaction value field: 759,736,088.1215131
- Supplier metadata coverage: 1.0
- Buyer metadata coverage: 1.0
- Ecosystem metadata coverage: 1.0
- Buyers without edges: 2,466

Orientation and source caveat:

- Output direction is `supplier_country_sector -> buyer_country_sector`.
- The legacy source uses `source_agent_id -> target_agent_id`; this orientation was checked against `src/abm_v1/prepare_abm_inputs.py`.
- The selected source value is `embedded_emissions`, mapped into the canonical `transaction_value` column for ABM v4 Phase 4A compatibility.
- This is not raw Eora `T_{supplier,buyer}` transaction value. Raw `T` fallback support exists in code but was not selected because the legacy edge panel was available and already sparse.

The supplier-edge foundation is real-data-ready as an observed historical directed edge layer. It does not yet implement supplier opportunity sets, supplier rewiring, production simulation, or scenarios.

## Phase 4A-bis Raw Eora T Supplier Edge Foundation

ABM v4 now also builds a supplier-edge foundation directly from raw Eora transaction matrices:

```text
data/parquet/{year}/T.parquet
```

Rows are treated as suppliers, columns as buyers, and only positive `T_{supplier,buyer}` entries are emitted. The raw T build is explicit and separate from the earlier legacy-derived output; it does not overwrite:

```text
data/abm_v4/interim/historical_supplier_edges.parquet
```

Generated Phase 4A-bis outputs:

```text
data/abm_v4/interim/historical_supplier_edges_raw_T.parquet
data/abm_v4/diagnostics/supplier_edge_raw_T_report.csv
data/abm_v4/diagnostics/supplier_edge_source_comparison.csv
```

Raw T real-data summary:

- Years covered: 1995-2016
- Raw T edge rows: 531,052,530
- Unique suppliers: 4,914
- Unique buyers: 4,913
- Unique supplier-buyer pairs: 24,142,481
- Total raw transaction value: 1,155,630,705,810.972
- Supplier metadata coverage: 1.0
- Buyer metadata coverage: 1.0
- Ecosystem metadata coverage: 1.0
- Buyers without edges: 2

Comparison with the legacy embodied-emissions edge panel:

- Legacy edge rows: 241,588
- Legacy unique supplier-buyer pairs: 16,856
- Raw T includes all 16,856 legacy supplier-buyer pairs.
- Legacy pair overlap share: 1.0
- Raw pair overlap share: 0.0006981883924854285
- Raw T median edges per buyer: 108,104
- Legacy median edges per buyer: 44

Recommendation for Phase 4B:

- Use raw Eora `T_{supplier,buyer}` edges as the canonical production-sourcing edge source for supplier substitution.
- Keep legacy embodied-emissions edges for carbon diagnostics and historical emissions exposure checks.
- Do not use embodied-emissions values as the main supplier substitution weight, because those values describe carbon embodied in flows rather than raw input sourcing transactions.

Implementation caveat:

- The raw Eora T panel is very large because the real matrices are close to dense after filtering positive entries. The CLI uses a streaming two-pass writer so Phase 4A-bis does not create an all-to-all long table in memory.
- This phase still does not implement supplier opportunity sets, supplier rewiring, production simulation, or scenarios.

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

Build the real-data state panel and state diagnostics only:

```powershell
python scripts/run_abm_v4_base.py --build-state --create-output-dirs
```

Build productive ecosystem mapping, adjacency, diagnostics, and update the state panel:

```powershell
python scripts/run_abm_v4_base.py --build-ecosystems --create-output-dirs
```

Build observed historical supplier-buyer edges and diagnostics:

```powershell
python scripts/run_abm_v4_base.py --build-supplier-edges --create-output-dirs
```

Build raw Eora T supplier-buyer edges and compare them with legacy embodied-emissions edges:

```powershell
python scripts/run_abm_v4_base.py --build-raw-t-supplier-edges --create-output-dirs
```

## Output Root

```text
data/abm_v4/
```

## Remaining for v5 or Later Phases

- Implement supplier opportunity sets and supplier-side rewiring.
- Implement iterative production propagation.
- Write emissions decomposition outputs.
- Add policy scenarios after the base model works.
