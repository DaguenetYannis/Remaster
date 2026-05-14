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

## Phase 4B-prep Compact Supplier Candidate Base

ABM v4 now reduces the large raw Eora T edge panel into compact candidate tables for later supplier opportunity-set construction. This is a preparation step only; it does not implement final opportunity sets, rewiring, production simulation, or scenarios.

Inputs:

```text
data/abm_v4/interim/historical_supplier_edges_raw_T.parquet
data/abm_v4/inputs/abm_v4_state_panel_1995_2016.parquet
data/abm_v4/inputs/ecosystem_adjacency.csv
```

Generated outputs:

```text
data/abm_v4/interim/supplier_candidates_historical_top.parquet
data/abm_v4/interim/supplier_pool_same_sector.parquet
data/abm_v4/interim/supplier_pool_ecosystem.parquet
data/abm_v4/diagnostics/supplier_candidate_base_report.csv
```

Real-data candidate table sizes:

- Historical top supplier candidates: 122,825 rows
- Same-sector supplier pool: 122,850 rows
- Ecosystem supplier pool: 122,875 rows
- Number of buyers in state panel: 4,915
- Median historical candidates per buyer: 25
- Median same-sector candidates per buyer: 25
- Median ecosystem candidates per buyer: 25
- Maximum candidates per buyer by type: `historical=25; same_sector=25; ecosystem=25`
- Share of buyers with no historical candidates: 0.0004069175991861648
- Share of buyers with no same-sector candidates: 0.0002034587995930824
- Share of buyers with no ecosystem candidates: 0.0

Memory strategy:

- The full 531,052,530-row raw T edge panel is not loaded into pandas or materialized as a full opportunity table.
- Historical top candidates are selected with bounded per-buyer heaps from parquet batches, then DuckDB aggregates full-history totals only for the selected compact pair set.
- Same-sector and ecosystem pools are built from the compact ABM v4 state panel and ecosystem adjacency table.
- No all-to-all supplier-buyer matrix is created for candidate generation.

Interpretation for Phase 4B:

- `supplier_candidates_historical_top.parquet` is the compact historical backbone from canonical raw Eora T sourcing.
- `supplier_pool_same_sector.parquet` supplies controlled same-sector alternatives.
- `supplier_pool_ecosystem.parquet` supplies same-ecosystem and adjacent-ecosystem alternatives, with duplicate candidate flags where a pair already appears in historical or same-sector pools.

## Phase 4B Supplier Opportunity Sets

ABM v4 now builds bounded supplier opportunity sets from the three compact candidate pools. These rows define feasible sourcing alternatives for each buyer. They are not realized rewiring decisions, and supplier weights are not updated in this phase.

Inputs:

```text
data/abm_v4/interim/supplier_candidates_historical_top.parquet
data/abm_v4/interim/supplier_pool_same_sector.parquet
data/abm_v4/interim/supplier_pool_ecosystem.parquet
data/abm_v4/inputs/abm_v4_state_panel_1995_2016.parquet
```

Generated outputs:

```text
data/abm_v4/interim/supplier_opportunity_sets.parquet
data/abm_v4/diagnostics/supplier_opportunity_set_report.csv
```

Real-data opportunity-set summary:

- Opportunity rows: 308,920
- Number of buyers: 4,915
- Median candidates per buyer: 64
- 95th percentile candidates per buyer: 75
- Maximum candidates per buyer: 75
- Share of rows flagged as historical candidates: 0.3975948465622168
- Share of rows flagged as same-sector candidates: 0.3976757736630843
- Share of rows flagged as ecosystem candidates: 0.3977567007639518
- Share of multi-source candidates: 0.17418425482325522
- Mean friction: 0.4579648452673831
- Mean green advantage: 0.0005321805040615166
- Mean supplier attractiveness: 0.7857205275774213

Probability validation:

- Buyers with probability-sum error: 0
- Maximum probability-sum error: 2.220446049250313e-16
- `choice_probability` has no nulls in the generated opportunity set.

Method:

- Candidate pools are merged and deduplicated by `buyer_country_sector + supplier_country_sector`.
- Source flags are preserved as `candidate_sources`, with separate boolean flags for historical, same-sector, and ecosystem membership.
- Priority is `historical`, then `same_sector_foreign`, then `ecosystem_feasible`.
- Friction follows the ABM v4 hierarchy: historical < same-sector < ecosystem.
- Green advantage uses the latest state year and a buyer-specific historical-supplier green baseline where available.
- Reliability is a clipped latest-two-year output stability proxy.
- Choice probabilities are a stable softmax within each buyer.

Caveat:

- These opportunity sets are feasible alternatives only. Phase 4B does not implement realized supplier rewiring, dynamic supplier weight updates, production propagation, or scenarios.

## Phase 4C Supplier Rewiring Draws and Weight Updates

ABM v4 now builds a one-step baseline supplier-side weight update from the Phase 4B opportunity set. This phase creates initial weights, buyer-level rewiring flags, and updated weights. It does not run production, emissions, or scenario simulation.

Inputs:

```text
data/abm_v4/interim/supplier_opportunity_sets.parquet
data/abm_v4/inputs/abm_v4_state_panel_1995_2016.parquet
```

Generated outputs:

```text
data/abm_v4/interim/supplier_initial_weights.parquet
data/abm_v4/interim/supplier_rewiring_flags.parquet
data/abm_v4/interim/supplier_updated_weights.parquet
data/abm_v4/diagnostics/supplier_rewiring_report.csv
```

Real-data supplier rewiring summary:

- Number of buyers: 4,915
- Opportunity rows / weight rows: 308,920
- Rewired buyers: 55
- Rewired buyer share: 0.011190233977619531
- Mean rewiring probability: 0.01269445476897798
- Median rewiring probability: 0.012173219555639662
- Maximum rewiring probability: 0.033166856998914244
- Mean absolute weight delta: 0.000026204473064471854
- Maximum absolute weight delta: 0.04886093039474282

Weight normalization validation:

- Buyers with initial weight-sum error: 0
- Buyers with updated weight-sum error: 0
- Maximum initial weight-sum error: 2.220446049250313e-16
- Maximum updated weight-sum error: 2.220446049250313e-16

Fallbacks:

- Buyers using choice-probability initialization because no historical candidate weight was available: 2
- Buyers using fallback stress = 0: 4,915
- Buyers using computed green-gap fallback: 4,915

Method:

- Historical candidates initialize from `historical_tie_strength`; non-historical candidates initialize at zero.
- Buyers with no historical weights initialize from `choice_probability`.
- Buyer-level rewiring probability is `p_rewire_base + p_rewire_stress * stress + p_rewire_green_gap * green_gap`, clipped to `[0, 1]`.
- Rewiring draws use a deterministic seed fallback.
- Rewired buyers move weights toward `choice_probability` using `lambda_weight_update`; non-rewired buyers keep initial weights.
- Updated weights are renormalized by buyer.

Caveat:

- This is a one-step baseline supplier-weight update layer. It is not the full dynamic simulation, does not propagate production, does not simulate emissions, and does not implement scenarios.

## Phase 5 Capability Accumulation Layer

ABM v4 now builds a one-step general and green capability update using the latest state panel year and updated supplier weights. This phase creates capability exposures, capability deltas, updated capability stocks, and diagnostics. It does not run production, emissions, or scenario simulation.

Inputs:

```text
data/abm_v4/inputs/abm_v4_state_panel_1995_2016.parquet
data/abm_v4/interim/supplier_updated_weights.parquet
```

Generated outputs:

```text
data/abm_v4/interim/capability_exposure_panel.parquet
data/abm_v4/interim/capability_update_panel.parquet
data/abm_v4/diagnostics/capability_update_report.csv
```

Real-data capability update summary:

- Selected year: 2016
- Node count: 4,915
- Mean general capability stock: 0.6908553253705668
- Mean green capability stock: 0.0610879717557677
- Mean general capability exposure: 0.6676063336392002
- Mean green capability exposure: 0.18712781831913547
- Mean general capability delta: 0.010585083349278736
- Mean green capability delta: 0.007929139430966488
- Maximum general capability delta: 0.02781412519241035
- Maximum green capability delta: 0.020303023767912022

Missingness and fill handling:

- Share of nodes with general capability filled from within-year median: 0.40508646998982706
- Share of nodes with green capability filled from within-year median: 0.40508646998982706
- Missing capability stocks are not silently set to zero; fill flags are written in `capability_update_panel.parquet`.

Supplier exposure coverage:

- Mean supplier capability coverage: 1.0
- Mean supplier green coverage: 1.0

Bounds:

- General capability clipped count: 0
- Green capability clipped count: 0
- `cap_next >= cap` and `gcap_next >= gcap` for all nodes.
- `cap_next <= 1` and `gcap_next <= 1` for all nodes.

Method:

- General capability exposure combines normalized production experience, supplier-weighted general capability, and same-ecosystem mean capability.
- Green capability exposure combines green production experience, supplier-weighted `g_local_v4`, and supplier-weighted green capability.
- Supplier-weighted exposures use `supplier_updated_weights.parquet`, exclude missing supplier values, and renormalize covered supplier weights.
- Capability accumulation uses a sigmoid adoption signal and saturates near configured maxima.

Caveat:

- This is a one-step capability update, not the full dynamic simulation. ABM v4 still has only general and green capability stocks; ecosystem-specific capability stocks remain a v5 extension.

## Phase 6 One-Step Production Feasibility Layer

ABM v4 now builds a one-step production feasibility diagnostic using the latest state panel year, raw Eora transaction requirements, and updated supplier weights. This phase does not run emissions, a multi-year loop, recursive Leontief propagation, or scenarios.

Inputs:

```text
data/abm_v4/inputs/abm_v4_state_panel_1995_2016.parquet
data/abm_v4/interim/supplier_updated_weights.parquet
data/parquet/{year}/T.parquet
```

Generated outputs:

```text
data/abm_v4/interim/production_feasibility_panel.parquet
data/abm_v4/diagnostics/production_feasibility_report.csv
```

Real-data production feasibility summary:

- Selected year: 2016
- Node count: 4,915
- Total observed output: 153,240,228,193.02585
- Total desired output: 153,240,228,193.02585
- Total feasible output: 153,230,697,673.3565
- Aggregate feasibility ratio: 0.9999378066726882
- Mean input feasibility: 0.9995930824004696
- Median input feasibility: 0.9999999999999984
- 5th percentile input feasibility: 0.9999999999999223
- 95th percentile max supplier pressure: 0.4274605852214408
- Nodes with input feasibility below 1: 4,256
- Share of nodes with input feasibility below 1: 0.8659206510681587
- Nodes with supplier pressure above 1: 14
- Share of nodes with supplier pressure above 1: 0.0028484231943031535

Method:

- Eora transaction orientation is treated as `T_{supplier,buyer}`.
- Technical input requirements are derived from buyer-level raw T column sums divided by buyer observed output.
- `X_desired` is set to `X_observed` for this base diagnostic.
- Updated supplier weights allocate each buyer's total required inputs across feasible suppliers.
- Supplier capacity pressure uses supplier `X_observed` as a capacity proxy.
- `X_feasible = X_desired * min(1, input_feasibility)`.

Caveat:

- This is a one-step feasibility diagnostic, not full recursive Leontief propagation. It does not update supplier output recursively and does not simulate emissions or scenarios.

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

Build compact supplier candidate tables for Phase 4B preparation:

```powershell
python scripts/run_abm_v4_base.py --build-supplier-candidate-base --create-output-dirs
```

Build supplier opportunity sets from compact candidate tables:

```powershell
python scripts/run_abm_v4_base.py --build-supplier-opportunities --create-output-dirs
```

Build one-step supplier rewiring flags and supplier-weight updates:

```powershell
python scripts/run_abm_v4_base.py --build-supplier-rewiring --create-output-dirs
```

Build one-step capability exposures and capability updates:

```powershell
python scripts/run_abm_v4_base.py --build-capability-update --create-output-dirs
```

Build one-step production feasibility diagnostics:

```powershell
python scripts/run_abm_v4_base.py --build-production-feasibility --create-output-dirs
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
