# ABM v4 Implementation Note

## Phase 29C Final Visual Polish and Plot Replacement

Phase 29C adds the final polished visual layer for ABM v4. It is a visual-only phase: no model logic, scenarios, ABM v5 code, LaTeX, webpage files, transition rules, `config.py`, or ABM v1/v2/v3 sources are changed.

Phase 29C keeps the strongest Phase 29B concepts but replaces weak report visuals:

- Kept and polished: architecture layers, emissions decomposition logic, capability source coverage, Q energy boundary, China electricity boundary case, scenario readiness checklist, and ABM v4 to ABM v5 roadmap.
- Replaced: the two-rule scatter is replaced by `abm_v4_two_rule_scorecard`; the mechanism decision tree is replaced by `abm_v4_mechanism_status_grid`.
- Retired as a plot: the dense Phase 29B hypothesis status table is now `hypothesis_status_report_table.csv`.

Polished output paths:

```text
data/abm_v4/final/plots_polished/
data/abm_v4/final/tables_polished/
data/abm_v4/final/abm_v4_polished_plot_index.csv
outputs/plots/abm_v4_final_polished/
```

The visual selection manifest is written to:

```text
data/abm_v4/final/tables_polished/final_visual_selection_manifest.csv
```

CLI:

```text
python scripts/run_abm_v4_base.py --polish-final-abm-v4-plots --create-output-dirs
```

The final LaTeX report and portfolio webpage remain human-written. Phase 29C only provides polished source tables, figures, copied figure assets, and a selection manifest.

## Phase 29B Narrative-Grade Final Visual Redesign

Phase 29B adds a smaller report-ready visual layer for the completed ABM v4 historical diagnostic framework. It supersedes the first Phase 29A plots for final-report and portfolio use because the Phase 29A figures were useful artifact summaries but too generic: several visualized labels or statuses more than evidence, tradeoffs, and model boundaries.

Phase 29B does not change model logic, create scenarios, implement ABM v5, write LaTeX, or write webpage files. The LaTeX report and portfolio webpage remain human-written; these artifacts are only source figures and source tables.

Narrative plots:

- `abm_v4_architecture_layers`
- `abm_v4_emissions_decomposition_logic`
- `abm_v4_two_rule_metric_tradeoff`
- `abm_v4_mechanism_decision_tree`
- `abm_v4_capability_source_coverage`
- `abm_v4_q_energy_mix_quality_boundary`
- `abm_v4_china_electricity_boundary_case`
- `abm_v4_scenario_readiness_checklist`
- `abm_v4_to_v5_roadmap`
- `abm_v4_hypothesis_status_table`

Narrative source tables:

- `architecture_layers_source.csv`
- `emissions_decomposition_logic_source.csv`
- `two_rule_metric_tradeoff_source.csv`
- `mechanism_decision_tree_source.csv`
- `capability_source_coverage_source.csv`
- `q_energy_mix_quality_boundary_source.csv`
- `china_electricity_boundary_case_source.csv`
- `scenario_readiness_checklist_source.csv`
- `abm_v4_to_v5_roadmap_source.csv`
- `hypothesis_status_table_source.csv`

Output paths:

```text
data/abm_v4/final/plots_narrative/
data/abm_v4/final/tables_narrative/
data/abm_v4/final/abm_v4_narrative_plot_index.csv
outputs/plots/abm_v4_final_narrative/
```

CLI:

```text
python scripts/run_abm_v4_base.py --build-final-abm-v4-narrative-plots --create-output-dirs
```

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

## Phase 7B Frontier-Gap Emissions-Intensity Update and Decomposition

ABM v4 now builds a one-step emissions-intensity update and aggregate emissions decomposition using a sector-frontier gap-closure rule gated by transition readiness. This replaces the raw `log(EI)` rule as the default behavioural mechanism, while preserving the old rule as `legacy_raw_log` for comparison only. This phase does not implement a multi-year simulation loop or scenarios.

Inputs:

```text
data/abm_v4/inputs/abm_v4_state_panel_1995_2016.parquet
data/abm_v4/interim/capability_update_panel.parquet
data/abm_v4/interim/production_feasibility_panel.parquet
data/abm_v4/interim/capability_exposure_panel.parquet
```

Generated outputs:

```text
data/abm_v4/interim/emissions_update_panel.parquet
data/abm_v4/diagnostics/emissions_update_report.csv
data/abm_v4/diagnostics/emissions_decomposition_base.csv
data/abm_v4/diagnostics/emissions_historical_rEI_summary.csv
data/abm_v4/diagnostics/emissions_sector_background_trend.csv
data/abm_v4/diagnostics/emissions_frontier_gap_report.csv
data/abm_v4/diagnostics/emissions_transition_comparison.csv
```

Why the raw-log rule was rejected:

- The previous placeholder equation used raw `log(EI)` as direct behavioural pressure.
- Because many valid EI values are below 1, `log(EI)` is negative and can reverse the intended sign.
- The correct behavioural object is distance from a feasible lower-carbon benchmark, not raw EI scale.

Default transition equation:

```text
EI_frontier_{s,t} = Q_0.25(EI_{j,t} | sector(j) = s)
ei_gap_i = max(0, log(EI_i) - log(EI_frontier_s))

readiness_i =
rho_max * sigmoid(
    theta_0
    + theta_gcap * gcap_next_i
    + theta_cap * cap_next_i
    + theta_network * network_green_exposure_i
    + theta_eco * ecosystem_capability_exposure_i
    - theta_brown * brown_centrality_i
    - theta_lockin * supplier_lockin_i
)

gap_closure_potential_i =
readiness_i * ei_gap_i / (ei_gap_i + tau_gap)

rEI_used_i =
sector_background_trend_s + gap_closure_potential_i

EI_next_i = EI_i * exp(-rEI_used_i)
```

Selected frontier definition:

- Sector-year 25th percentile of valid positive EI.
- Sectors with fewer than 5 valid EI nodes use the global year frontier.
- 26 sectors used sector frontiers; 1 sector used the global fallback.

Readiness variables:

- `gcap_next`
- `cap_next`
- `network_green_exposure`
- `ecosystem_capability_exposure`
- `brown_centrality`
- `supplier_lockin`, computed as the Herfindahl concentration of updated supplier weights.

Historical observed rEI summary:

- Valid consecutive positive-EI observations: 96,803
- Mean historical observed rEI: 0.036876130454566405
- Median historical observed rEI: 0.035605709216913084
- p05 / p95: -0.27640050238653835 / 0.32818744389233334
- Share positive: 0.6062622026176875
- Share negative: 0.3937377973823125

Real-data frontier-gap emissions update summary:

- Selected year: 2016
- Node count: 4,915
- Valid EI nodes: 4,620
- Invalid EI nodes: 295
- Mean EI gap: 0.816389623646047
- Median EI gap: 0.6146226767682785
- Share zero EI gap: 0.25303030303030305
- Mean readiness: 0.03206306488288885
- Median readiness: 0.032221009552336435
- Mean `rEI_used`: 0.04636061610414019
- Median `rEI_used`: 0.0465575247340423
- Share negative `rEI_used`: 0.0
- EI clipped count: 0
- Total current emissions: 41,772,432.079139076
- Total feasible-output current-EI emissions: 41,772,432.079137705
- Total next emissions: 39,716,544.72864129
- Aggregate delta emissions: -2,055,887.3504977897

Aggregate decomposition:

- Production-scale effect: -0.0000013755655184814115
- Emissions-intensity effect: -2,055,887.3504964742
- Interaction effect: 0.00000007123719683228708
- Decomposition residual: -0.000000011175870895385742
- Bad transition flag: false

Legacy comparison:

- `frontier_gap_readiness`: mean `rEI_used` 0.04636061610414019; median 0.0465575247340423; total next emissions 39,716,544.728641294; aggregate delta emissions -2,055,887.3504977748.
- `legacy_raw_log`: mean `rEI_used` -0.0499693646611653; median -0.05; total next emissions 43,902,716.64668248; aggregate delta emissions 2,130,284.5675434023.
- The legacy rule is retained only as a comparison mode because raw `log(EI)` is not a stable theoretical pressure for decarbonization.

Method:

- Emissions identity is kept explicit: `E = X * EI`.
- Invalid or non-positive EI values are not repaired; `rEI_used` and `EI_next` are missing for invalid rows and those rows are excluded from aggregate emissions decomposition.
- `EI_next = max(EI_min, EI * exp(-rEI_used))` for valid EI rows.
- The decomposition separates production-scale, emissions-intensity, and interaction effects at node level before aggregation.

Caveat:

- The frontier-gap readiness parameters are conservative one-step diagnostic defaults, not calibrated historical parameters. The rule is theoretically safer than raw `log(EI)`, but it is not yet a fully calibrated historical transition model and is not a multi-year simulation or scenario run.

## Phase 8 One-Step Base Orchestration

ABM v4 now has a one-step base orchestration and validation layer. This phase reuses the existing Phase 2 through Phase 7B outputs, checks that they are compatible, and writes a consolidated validation report. It does not run a multi-year dynamic simulation, does not create scenarios, and does not create projection outputs.

Command:

```powershell
python scripts/run_abm_v4_base.py --run-one-step-base --create-output-dirs --reuse-existing
```

Generated outputs:

```text
data/abm_v4/validation/one_step_base_validation_report.csv
data/abm_v4/validation/one_step_base_validation_report.md
data/abm_v4/validation/one_step_base_status.json
```

Integration status:

- Overall status: warning
- Overall passed: true
- Failed layers: none
- Warning layers: supplier, capability, production, emissions
- Blocking issues before multi-year simulation: none from the one-step validation rules
- Recommended next phase: Phase 9 multi-year base simulation design

Layer validation table:

| Layer | Status | Key metrics | Warnings |
| --- | --- | --- | --- |
| State | pass | 108,130 rows; 1995-2016 coverage; 4,915 country-sector nodes | none |
| Ecosystem | pass | 4,915 mapped nodes; 0 unmapped nodes; 10 ecosystems | none |
| Supplier | warning | raw source `raw_eora_T`; 308,920 opportunity rows; median 64 candidates per buyer; rewired buyer share 0.011190233977619531; max updated weight sum error 2.220446049250313e-16 | fallback stress was used for most buyers |
| Capability | warning | year 2016; mean cap 0.6908553253705668; mean gcap 0.0610879717557677; mean delta cap 0.010585083349278736; mean delta gcap 0.007929139430966488 | capability fill share is above 0.25 |
| Production | warning | aggregate feasibility ratio 0.9999378066726882; mean input feasibility 0.9995930824004696; constrained node share 0.8659206510681587; p95 supplier pressure max 0.4274605852214408 | many nodes are marginally constrained despite high aggregate feasibility |
| Emissions | warning | mode `frontier_gap_readiness`; 4,620 valid EI nodes; 295 invalid EI nodes; mean `rEI_used` 0.04636061610414019; aggregate delta emissions -2,055,887.3504977897; decomposition residual -1.1175870895385742e-08 | invalid EI share is above 0.05 |

Validation rules:

- State passes if country-sector nodes are available and the selected year is covered.
- Ecosystem passes if unmapped node count is zero.
- Supplier passes if updated supplier weights are normalized within `1e-8`.
- Capability passes if capability clipping counts are zero, with a warning when fill shares exceed 0.25.
- Production passes if aggregate feasibility is above 0.95, with a warning when many nodes are constrained but aggregate feasibility remains high.
- Emissions passes if the decomposition residual is below `1e-4` in absolute value and `bad_transition_flag` is false, with a warning when invalid EI share exceeds 0.05.

Raw T rebuild behavior:

- The one-step orchestrator reuses existing raw T supplier edges by default.
- Raw T edges are not rebuilt during Phase 8 unless explicitly requested by a future rebuild path.
- If required component outputs are missing, the orchestrator fails clearly and names the missing paths instead of silently inventing or rebuilding inputs.

## Phase 9 Atlas Capability Join Repair and Coverage Audit

ABM v4 now has an explicit Atlas capability coverage repair/audit step. The repair inspects available state and Atlas capability schemas, selects explicit join keys, fills only missing canonical capability values, preserves existing non-missing canonical values, and writes coverage diagnostics. It does not run suppliers, production, emissions, scenarios, or multi-year simulation.

Command:

```powershell
python scripts/run_abm_v4_base.py --repair-capability-coverage --create-output-dirs
```

Then the capability update and one-step validation were rerun:

```powershell
python scripts/run_abm_v4_base.py --build-capability-update --create-output-dirs
python scripts/run_abm_v4_base.py --run-one-step-base --create-output-dirs --reuse-existing
```

Generated diagnostics:

```text
data/abm_v4/diagnostics/capability_join_report.csv
data/abm_v4/diagnostics/capability_coverage_by_year.csv
data/abm_v4/diagnostics/capability_coverage_by_sector.csv
```

Selected source and keys:

- Source: `data/atlas/processed/atlas_eora26_sector_capabilities_1995_2016.parquet`
- Selected join keys: `Country=iso3Code; Year=year; Sector=eora26_sector`
- State rows before: 108,130
- State rows after: 108,130
- Matched rows: 63,905
- Unmatched rows: 44,225
- Matched share: 0.5910015721816332

Coverage before and after:

| Metric | Before | After |
| --- | ---: | ---: |
| General capability nonmissing rows | 63,905 | 63,905 |
| Green capability nonmissing rows | 63,905 | 63,905 |
| General capability missing/fill share | 0.4089984278183668 | 0.4089984278183668 |
| Green capability missing/fill share | 0.4089984278183668 | 0.4089984278183668 |
| 2016 general capability missing/fill share | 0.4050864699898271 | 0.4050864699898271 |
| 2016 green capability missing/fill share | 0.4050864699898271 | 0.4050864699898271 |

Outcome:

- Canonical capability coverage did not improve.
- The join itself is valid and explicit, but the matched rows were already the rows with Atlas capability data in the state panel.
- The remaining missing rows are not recoverable from the inspected Atlas capability source with the current country-year-sector keys.

Main coverage gaps by sector:

- `Construction`: 100% missing general and green capability.
- `Education, Health and Other Services`: 100% missing.
- `Electricity, Gas and Water`: 100% missing.
- `Fishing`: 100% missing.
- `Maintenance and Repair`: 100% missing.
- `Private Households`: 100% missing.
- `Public Administration`: 100% missing.
- `Recycling`: 100% missing.

Interpretation:

- The Atlas source contains explicit variables such as `capability_export_weighted_pci`, `capability_mean_pci`, `active_good_count`, `green_active_good_count`, `green_capability_share`, and `green_capability_export_share`.
- However, Atlas goods/product capability coverage does not provide usable non-missing values for several Eora service or non-product sectors.
- Existing non-missing state capability values were intentionally not overwritten.
- For dynamic simulation, this remains a substantive warning: capability dynamics for the missing sectors still rely on within-year median filling in the capability update layer.

Recommended follow-up before or during multi-year simulation design:

- Decide whether service-sector capability should use a separate service-capability proxy.
- Consider sector-level imputation rules that are explicit and theoretically motivated rather than hidden fills.
- Keep the current median fill flags in capability dynamics until a better capability source or sector-specific proxy is approved.

## Phase 9C IO-Derived Capability Proxy

ABM v4 now uses a tiered capability measurement model instead of treating median-filled missing capability values as equivalent to observed Atlas capability.

Reason for the change:

- Atlas capability data measure product/export capabilities.
- Eora includes many sectors where product-space capability is structurally unavailable or conceptually weak, especially services, public/social sectors, households, and residual sectors.
- Median filling is useful as a numerical fallback, but it is not a defensible measurement layer for transition dynamics.

Tiered capability model:

```text
Cap_model_{i,t} =
    Atlas capability, if observed;
    IO-imputed capability, if Atlas is structurally missing and IO coverage is sufficient;
    unavailable, otherwise.
```

The same structure is used for green capability. IO-imputed capability is a network-embedded productive capability proxy, not an observed Atlas value.

IO exposure equations:

```text
Cap_up_i =
sum_j w_in_{j,i} * I_Atlas_j * Cap_Atlas_j
/
sum_j w_in_{j,i} * I_Atlas_j

Cap_down_i =
sum_k w_out_{i,k} * I_Atlas_k * Cap_Atlas_k
/
sum_k w_out_{i,k} * I_Atlas_k

Cap_IO_i =
lambda_up * Cap_up_i
+ (1 - lambda_up) * Cap_down_i
```

Green capability uses the same structure with green Atlas capability. In this v4 implementation, upstream exposure uses compact `supplier_updated_weights.parquet`. Downstream exposure uses compact supplier-weight buyer links as a sales-share proxy rather than scanning the full raw-T edge panel. This avoids all-to-all construction and keeps Phase 9C model-usable; raw-T downstream aggregation remains a future refinement.

Calibration:

- Lambda is calibrated separately for general and green capability.
- Grid: 0.00, 0.05, ..., 1.00.
- Loss: mean absolute error against Atlas-observed nodes, with RMSE also reported.
- Minimum IO coverage threshold: 0.3.

Generated outputs:

```text
data/abm_v4/diagnostics/io_capability_lambda_calibration.csv
data/abm_v4/diagnostics/io_capability_model_report.csv
data/abm_v4/diagnostics/io_capability_coverage_by_sector.csv
data/abm_v4/diagnostics/io_capability_coverage_by_source.csv
```

Real-data result for 2016:

| Capability type | atlas_observed | io_imputed | unavailable | selected lambda_up | selected lambda_down |
| --- | ---: | ---: | ---: | ---: | ---: |
| General | 2,924 | 1,548 | 443 | 0.35 | 0.65 |
| Green | 2,924 | 1,316 | 675 | 0.0 | 1.0 |

Calibration diagnostics:

- General selected MAE: 0.407775861953647
- General selected RMSE: 0.5757752863611213
- Green selected MAE: 0.08823890550156885
- Green selected RMSE: 0.1592492542158855
- Calibration observations: 59,956 for both general and green.

Coverage and source shares:

- General atlas-observed share: 0.5949135300101729
- General IO-imputed share: 0.3149542217700916
- General unavailable share: 0.0901322482197355
- Green atlas-observed share: 0.5949135300101729
- Green IO-imputed share: 0.2677517802644964
- Green unavailable share: 0.1373346897253306
- Mean general IO coverage: 0.6161243728217841
- Mean green IO coverage: 0.5319672919937847

Impact on Phase 5 capability update:

- General capability fill share fell from 0.40508646998982706 to 0.0901322482197355.
- Green capability fill share fell from 0.40508646998982706 to 0.1373346897253306.
- Mean cap after source-aware model: 0.6652630986290047.
- Mean gcap after source-aware model: 0.08167427279197137.
- Mean delta cap: 0.01122678462293864.
- Mean delta gcap: 0.00804949815956165.

Impact on emissions readiness:

- The capability update now prefers `general_capability_model` and `green_capability_model`.
- Emissions readiness uses `cap_next` and `gcap_next` produced from the source-aware capability model.
- Frontier-gap emissions update remains valid: mean `rEI_used` is 0.04637543051620834, aggregate delta emissions is -2,073,630.0784223452, decomposition residual is 0.000000010943040251731873, and `bad_transition_flag` is false.

One-step base validation after Phase 9C:

- Overall status: warning
- Failed layers: none
- Warning layers: supplier, production, emissions
- Capability layer now passes without warning.

Caveat:

- IO-imputed capability is a model-derived proxy for network-embedded productive capability. It is not observed Atlas product-space capability and should remain source-labeled throughout simulation and diagnostics.

## Phase 9D IO Capability Robustness and Downstream Exposure Audit

ABM v4 now writes diagnostic-only robustness checks for the source-aware IO capability model. This phase does not rebuild the state panel, does not run multi-year simulation, and does not create scenarios.

Command:

```powershell
python scripts/run_abm_v4_base.py --audit-io-capability-robustness --create-output-dirs
```

Generated outputs:

```text
data/abm_v4/diagnostics/io_capability_robustness.csv
data/abm_v4/diagnostics/io_capability_threshold_sensitivity.csv
data/abm_v4/diagnostics/io_downstream_exposure_audit.csv
```

Robustness by specification:

| Specification | Capability | Coverage | Unavailable share | Validation MAE | Validation RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| atlas_only | general | 0.5949135300101729 | 0.40508646998982706 | 0.0 | 0.0 |
| atlas_only | green | 0.5949135300101729 | 0.40508646998982706 | 0.0 | 0.0 |
| upstream_only | general | 0.9617497456765005 | 0.03825025432349949 | 0.4334685204006095 | 0.6586782297708024 |
| upstream_only | green | 0.9617497456765005 | 0.03825025432349949 | 0.1108124920875149 | 0.1648325087477197 |
| downstream_only | general | 0.8626653102746694 | 0.1373346897253306 | 0.4171692660890992 | 0.5982642107155396 |
| downstream_only | green | 0.8626653102746694 | 0.1373346897253306 | 0.08823890550156882 | 0.1592492542158855 |
| calibrated_io | general | 0.9098677517802645 | 0.0901322482197355 | 0.407775861953647 | 0.5757752863611213 |
| calibrated_io | green | 0.8626653102746694 | 0.1373346897253306 | 0.08823890550156885 | 0.1592492542158855 |

Interpretation:

- `calibrated_io` remains the preferred general-capability specification because it has the lowest validation MAE/RMSE among non-trivial IO specifications while preserving much better coverage than `atlas_only`.
- Green capability calibration selects lambda_up = 0.0, so `calibrated_io` is effectively downstream-only for green capability.
- The downstream-heavy green result should be treated cautiously because downstream exposure currently uses the compact supplier-weight downstream proxy, not full raw-T sales shares.

Coverage-threshold sensitivity:

| Gamma | Capability | IO-imputed count | Unavailable count | Validation MAE | Mean IO coverage |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.1 | general | 1,838 | 153 | 0.407775861953647 | 0.6160665743353818 |
| 0.1 | green | 1,465 | 526 | 0.08823890550156885 | 0.5318590585671328 |
| 0.3 | general | 1,548 | 443 | 0.407775861953647 | 0.6160665743353818 |
| 0.3 | green | 1,316 | 675 | 0.08823890550156885 | 0.5318590585671328 |
| 0.5 | general | 1,174 | 817 | 0.407775861953647 | 0.6160665743353818 |
| 0.5 | green | 893 | 1,098 | 0.08823890550156885 | 0.5318590585671328 |

The default gamma = 0.3 is a middle position: it keeps substantial IO coverage while excluding weakly supported nodes.

Downstream exposure audit:

- Compact proxy: `supplier_updated_weights_downstream_sales_share`
- Selected year: 2016
- Compact nodes: 4,915
- General downstream exposure available: 4,244 nodes
- Green downstream exposure available: 4,244 nodes
- Mean downstream coverage: 0.6159489332840381
- Full raw-T downstream aggregation was not run in Phase 9D because the raw edge panel has 531M rows and the audit should remain a lightweight pre-simulation diagnostic.

Remaining caveat before multi-year simulation:

- The IO capability model is more defensible than median filling and passes the one-step integration checks, but the green-capability result depends strongly on the compact downstream proxy. Before policy/scenario interpretation, the raw-T downstream aggregation should be implemented or sampled against the compact proxy.

## Phase 10 Multi-Year Base Simulation

Phase 10 adds the first ABM v4 historical multi-year base loop. This is a conservative integration loop, not a policy scenario and not a calibrated projection engine.

Simulation modes implemented:

- Smoke run: `2015-2016`
- Historical base run: `1995-2016`

The manual smoke run succeeded, then the full `1995-2016` historical base run succeeded.

Historical production forcing:

- Enabled in the current Phase 10 default.
- `X_sim` is anchored to observed historical production by year.
- EI and emissions evolve through the simulated transition rule.
- This keeps the first multi-year loop focused on supplier, capability, and EI-transition integration before recursive production propagation is added.

Validation results from the full `1995-2016` run:

| Metric | Result |
| --- | ---: |
| Simulation years | 22 |
| Overall status | warning |
| Blocking issues | none |
| Raw-T rebuilt | false |
| Scenario outputs created | false |
| Max emissions identity error | 0.0 |
| Max supplier weight sum error | 2.220446049250313e-16 |
| Valid simulated EI remains positive | true |
| Historical production forcing | true |

Latest simulated year (`2016`) aggregate validation:

| Metric | Result |
| --- | ---: |
| Total X observed | 153240228193.02582 |
| Total X simulated | 153240228193.02582 |
| Aggregate X error pct | 0.0 |
| Total emissions observed | 41772432.07913907 |
| Total emissions simulated | 74374578.30185504 |
| Aggregate emissions error pct | 0.7804703868079855 |
| Mean rEI used | 0.016735856455233306 |
| Capability unavailable share | 0.0901322482197355 |
| Bad transition flag | false |

Warnings:

- Production is historically forced.
- Aggregate emissions error is high, so the current base loop is useful as an integration check but not yet ready for scenario interpretation.

Readiness assessment:

- Phase 10 is ready as a smoke-safe multi-year integration harness.
- The model is not yet ready for policy scenario design until production dynamics are no longer historically forced or the forcing mode is explicitly accepted for the intended scenario question.
- Next recommended phase: replace or extend the historically forced production step with a recursive production propagation/update layer, then rerun historical validation before scenario design.

## Phase 11 Multi-Year Historical Validation

Phase 11 adds a validation and calibration diagnostic layer over the existing multi-year base outputs. It does not rerun the simulation and does not rebuild raw-T edges.

Outputs created:

- `data/abm_v4/validation/multiyear_error_panel.parquet`
- `data/abm_v4/validation/multiyear_error_summary.csv`
- `data/abm_v4/validation/multiyear_error_by_sector.csv`
- `data/abm_v4/validation/multiyear_error_by_country.csv`
- `data/abm_v4/validation/multiyear_error_by_ecosystem.csv`
- `data/abm_v4/validation/multiyear_error_by_capability_source.csv`
- `data/abm_v4/validation/multiyear_calibration_targets.csv`
- `data/abm_v4/validation/multiyear_validation_report.md`

Aggregate validation result:

| Year | Total emissions observed | Total emissions simulated | Aggregate emissions pct error | Mean observed rEI | Mean simulated rEI |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2014 | 44824394.06247388 | 83015127.1275825 | 0.8520077931645968 | -0.04260265861543819 | 0.017886083077941862 |
| 2015 | 45677594.92734783 | 77126132.06575437 | 0.6884893389949007 | 0.07367152408091936 | 0.017789086175923802 |
| 2016 | 41772432.07913907 | 74409228.00315851 | 0.7812998740937108 |  |  |

Interpretation:

- The model is dynamically valid in the narrow accounting sense: emissions identity errors remain zero in the multi-year output.
- It is not historically calibrated: aggregate emissions are substantially above observed values in the latest years.
- The simulation should move to parameter calibration diagnostics before any scenario design.

Largest error sources by sector:

| Sector | Mean log EI error | Mean emissions absolute error | Total emissions error | Mean rEI error | Mean rEI absolute error |
| --- | ---: | ---: | ---: | ---: | ---: |
| TOTAL | -0.08404570225011163 | 542253.9879711973 | -6541205.9230980845 | -0.02748882037703813 | 0.44754885035641273 |
| Electricity, Gas and Water | 0.019032623572086047 | 53885.968647542526 | 158650077.27881142 | -0.019023076010589217 | 0.17706909008240282 |
| Mining and Quarrying | 0.13099169021990115 | 11953.460123371668 | 34715708.07038643 | -0.014780262808256596 | 0.16383501234999598 |
| Petroleum, Chemical and Non-Metallic Mineral Products | 0.09520448150108303 | 11584.919680904603 | 39705484.3604227 | -0.009116684751067153 | 0.13349612629189528 |
| Finacial Intermediation and Business Activities | 0.061234206382538674 | 9764.501211328985 | 28536980.287694596 | 0.0036111262548990195 | 0.13078000685549857 |

Largest error sources by country:

| Country | Mean log EI error | Mean emissions absolute error | Total emissions error | Mean rEI absolute error |
| --- | ---: | ---: | ---: | ---: |
| ROW | -0.08404570225011163 | 542253.9879711973 | -6541205.9230980845 | 0.44754885035641273 |
| CHN | 0.378931844976257 | 366683.9002176758 | 203699524.60279736 | 0.12195129016222771 |
| RUS | 0.5745027210294936 | 82333.79768247434 | 41867355.28596649 | 0.19430415718600977 |
| USA | 0.32103612615866767 | 56937.10018006292 | 27994758.265063148 | 0.06933434033316464 |
| IRQ | 1.4401996295463073 | 42933.66359391446 | 24552975.092382155 | 0.19322962552054715 |

Capability-source diagnostics:

| Capability source type | Source | Rows | Mean log EI error | Mean emissions absolute error | Total emissions error | Mean rEI absolute error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| general_capability_source | atlas_observed | 63905 | 0.09595401739356997 | 3523.7375601266544 | 176096158.12446445 | 0.13994984118189915 |
| general_capability_source | io_imputed | 34135 | 0.14792763555482255 | 8461.595118938605 | 190317656.19906092 | 0.14480161514082723 |
| general_capability_source | unavailable | 10090 | 0.17551092482946185 | 1388.8677385084738 | 3723167.305530755 | 0.1526356790888614 |
| green_capability_source | atlas_observed | 63905 | 0.09595401739356997 | 3523.7375601266544 | 176096158.12446445 | 0.13994984118189915 |
| green_capability_source | io_imputed | 28885 | 0.16141864118654428 | 9614.593596967376 | 185554627.85910073 | 0.14673532437442885 |
| green_capability_source | unavailable | 15340 | 0.13231153747263857 | 1376.1570623454136 | 8486195.645490965 | 0.14494920594206948 |

Interpretation of capability sources:

- IO-imputed capability does not obviously break transition dynamics, but it is associated with larger EI and emissions errors than Atlas-observed capability.
- `calibrated_io` remains acceptable as a source-aware proxy for integration, but it should be a calibration slice rather than treated as observed Atlas capability.
- Green IO-imputed capability remains cautious because Phase 9D showed it is effectively downstream-proxy driven.

Calibration target diagnostics:

- Most high-error sectors show negative rEI bias, meaning simulated EI reductions are generally slower than observed historical reductions.
- Suggested priorities include increasing readiness for several sectors, adjusting sector background trends where bias is smaller, and inspecting capability source for structurally hard-to-measure sectors.
- Top calibration targets include `TOTAL`, `Re-export & Re-import`, `Electricity, Gas and Water`, `Mining and Quarrying`, `Fishing`, `Recycling`, `Private Households`, and `Transport`.

Scenario readiness:

- The model is ready for parameter calibration diagnostics.
- It is still too early for scenarios because production is historically forced and emissions fit is not historically calibrated.
- Next recommended step: calibrate or tune frontier-gap readiness/background terms against observed historical rEI, while tracking errors by sector, capability source, and ecosystem.

## Phase 12 Emissions-Transition Calibration Scaffold

Phase 12 adds historically disciplined parameter-search diagnostics for the frontier-gap readiness emissions rule. This is not structural estimation and it does not overwrite `config.py`.

Calibration setup:

- Calibration target: observed historical `rEI = log(EI_t) - log(EI_t+1)`
- Valid calibration rows: 85,653
- Train years: 1995-2011
- Validation years: 2012-2016
- Search method: fixed-seed bounded random search
- Random-search iterations: 200
- Seed: 42

Selected parameter candidate:

| Parameter | Value |
| --- | ---: |
| rho_max | 0.03562892698694528 |
| theta_intercept | -2.451060778970692 |
| theta_gcap | 0.7665585120643779 |
| theta_cap | 0.6562447118756822 |
| theta_network_green | 0.015459813138623879 |
| theta_ecosystem_exposure | 1.4940282468595356 |
| theta_brown_centrality | 0.3513896037516846 |
| theta_supplier_lockin | 0.7604148914304725 |
| tau_gap | 3.547989190575052 |

Selected validation metrics:

| Metric | Train | Validation |
| --- | ---: | ---: |
| MAE | 0.1353682585007423 | 0.171197657560396 |
| RMSE | 0.21323638022465835 | 0.3206618584081736 |
| Median absolute error | 0.09407075577715189 | 0.0935372319332673 |
| Bias | 0.006342601870672514 | 0.026873174008266998 |
| Wrong-sign share | 0.35975204462040866 | 0.5366328205751809 |
| Correlation | 0.04129744150110398 | -0.023742993681002883 |
| Sector-weighted MAE | 0.14464511979869413 | 0.18718691294511328 |

Model comparison:

| Model | Validation MAE | Validation bias | Wrong-sign share | Validation correlation |
| --- | ---: | ---: | ---: | ---: |
| sector_background_only | 0.1711255188900063 | 0.02665680438012231 | 0.5366328205751809 | -0.027566604376771088 |
| frontier_gap_readiness | 0.171197657560396 | 0.026873174008266998 | 0.5366328205751809 | -0.023742993681002883 |
| readiness_without_capability | 0.1712014252062674 | 0.02689175462296123 | 0.5366328205751809 | -0.022172110324640053 |
| frontier_gap_only | 0.17294619773103434 | 0.032116767144703386 | 0.5366328205751809 | 0.0735110101602798 |
| legacy_raw_log | 0.274073841347476 | 0.17337859933665706 | 0.5367544232990819 | -0.045005970776762806 |

Interpretation:

- The full frontier-gap readiness model does not meaningfully outperform `sector_background_only` on validation MAE in this scaffold.
- `readiness_without_capability` performs almost identically to the full model, so the current capability terms are not yet empirically contributing enough.
- `frontier_gap_only` is worse than the full model on validation MAE, which suggests the readiness gate helps relative to ungated gap closure, but the gain is small.
- `legacy_raw_log` performs clearly worse and remains theoretically fragile, so it should stay comparison-only.
- Validation correlation is near zero and slightly negative for the selected full model.
- Wrong-sign share is high at about 0.5366, so the scaffold is not yet strong enough to claim good historical transition prediction.

Parameter plausibility:

- Theoretical signs are respected.
- `theta_network_green` is near its lower bound, suggesting network-green exposure is weakly identified in this current specification.
- No multiple-bound solution was selected, so the search is not obviously pinned to parameter limits.
- `rho_max = 0.0356` implies a conservative maximum annual readiness-gated EI reduction rate.
- `tau_gap = 3.548` makes gap closure fairly gradual.

Largest validation errors by sector:

| Sector | MAE | Bias | Wrong-sign share |
| --- | ---: | ---: | ---: |
| TOTAL | 0.5694513676674693 | -0.20789964955816886 | 0.75 |
| Re-export & Re-import | 0.3288742958921291 | -0.127350579232948 | 0.4859550561797753 |
| Fishing | 0.26963669798662127 | -0.08580225215885369 | 0.5048543689320388 |
| Electricity, Gas and Water | 0.21316204608614422 | -0.02680119856814984 | 0.4899425287356322 |
| Agriculture | 0.20556656263823708 | -0.05513525083746763 | 0.5070224719101124 |

Error by capability source:

| Capability source type | Source | Rows | MAE | Bias | Wrong-sign share |
| --- | --- | ---: | ---: | ---: | ---: |
| general_capability_source | atlas_observed | 11351 | 0.1679096621389368 | 0.0226599767095571 | 0.5297330631662409 |
| general_capability_source | io_imputed | 5096 | 0.17852144798994538 | 0.03625778989085255 | 0.5520015698587127 |
| green_capability_source | atlas_observed | 11351 | 0.1679096621389368 | 0.0226599767095571 | 0.5297330631662409 |
| green_capability_source | io_imputed | 5096 | 0.17852144798994538 | 0.03625778989085255 | 0.5520015698587127 |

Conclusion:

- The calibration scaffold works and produces the required outputs.
- The selected parameter candidate is exploratory only.
- The full frontier-gap readiness mechanism is theoretically safer than raw-log EI, but this run does not yet provide strong empirical validation beyond sector background trends.
- It is not strong enough yet for a calibrated historical simulation intended to support scenarios.
- Next recommended step: improve the calibration design before scenarios, especially by revisiting sector background trends, feature scaling, capability contribution, and sign-prediction performance.

## Phase 13 Emissions-Transition Hypothesis Diagnostics

Phase 13 tests why the Phase 12 frontier-gap readiness calibration remained weak. It does not recalibrate parameters, does not overwrite `config.py`, and does not create scenarios.

Outputs created:

- `data/abm_v4/validation/emissions_hypothesis_diagnosis.csv`
- `data/abm_v4/validation/emissions_target_horizon_panel.parquet`
- `data/abm_v4/validation/emissions_target_horizon_summary.csv`
- `data/abm_v4/validation/emissions_predictor_screening.csv`
- `data/abm_v4/validation/emissions_sector_dominance_diagnostics.csv`
- `data/abm_v4/validation/emissions_capability_source_diagnostics.csv`
- `data/abm_v4/validation/emissions_readiness_threshold_diagnostics.csv`
- `data/abm_v4/validation/emissions_frontier_specification_diagnostics.csv`
- `data/abm_v4/validation/emissions_macro_shock_diagnostics.csv`
- `data/abm_v4/validation/emissions_hypothesis_diagnostic_report.md`

Hypothesis diagnosis:

| Hypothesis | Evidence | Key metric | Interpretation | Recommended next action |
| --- | --- | ---: | --- | --- |
| H1 annual rEI is too noisy | mixed / weak | best horizon improvement = -0.009313617186632178 | Medium-run and smoothed targets reduce volatility, but simple readiness still does not beat sector background. | Do not switch target solely on this evidence; keep testing medium-run targets after model-form revisions. |
| H2 sector background dominates | supports / moderate | share sectors readiness improves = 0.037037037037037035 | Readiness improves only a very small sector subset. | Consider sector-family-specific transition rules. |
| H3 capability variables are weakly measured | supports / moderate | atlas minus IO MAE = -0.009104914304675249 | Atlas-observed capability performs better than IO-imputed, but neither capability slice makes readiness beat background. | Keep IO as integration proxy; calibrate capability effects by source. |
| H4 readiness may be nonlinear / threshold-based | mixed / weak | top minus bottom readiness target = -0.09780874342283331 | Smooth readiness quantiles do not show the expected ordering. | Test threshold/regime readiness only after revisiting feature scaling and sector rules. |
| H5 frontier gap may be misspecified | supports / moderate | best frontier gap-only MAE = 0.17675152898102478 | `sector_year_p50` performs best among frontier diagnostics, better than current p25 frontier. | Revise frontier definition only if p50/rolling alternatives remain interpretable and robust. |
| H6 macro/country-year shocks matter | supports / moderate | yearly mean residual std = 0.04787639260451795 | Residuals vary meaningfully by year; 2015 has high volatility and large residuals. | Add year or country-year controls for calibration diagnostics only. |

Target horizon evidence:

| Target | Rows | Std | Readiness corr. | Cap corr. | GCap corr. | Background MAE | Readiness MAE | Improvement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| one_year_rEI | 96,803 | 0.23872023216162916 | -0.04448882524992794 | 0.005888508151245079 | 0.0021873977919163053 | 0.17160804516253236 | 0.18092166234916454 | -0.009313617186632178 |
| three_year_rEI | 87,533 | 0.1256427540537762 | -0.05893680560894731 | 0.031130504956638626 | 0.008894030530032912 | 0.10992635474204364 | 0.1209496515858761 | -0.011023296843832464 |
| five_year_rEI | 78,268 | 0.09672060194300613 | -0.07746157324226308 | 0.045207390357283765 | 0.011143220950118354 | 0.09792056131477417 | 0.10916267431508889 | -0.011242113000314718 |
| smoothed_one_year_rEI | 87,533 | 0.12564275405377623 | -0.014159717189548438 | 0.03196125744237969 | 0.010871365956658819 | 0.09436578246879966 | 0.10551911890735968 | -0.011153336438560021 |
| winsorized_one_year_rEI | 96,803 | 0.19149708191304704 | -0.04444471019560394 | 0.009239282136952584 | 0.0063095343561262625 | 0.15186872964667095 | 0.16118234683330313 | -0.009313617186632178 |

Interpretation:

- Medium-run and smoothed targets are less volatile than one-year rEI.
- However, the simple readiness predictor still underperforms sector background for all tested target definitions.
- H1 is therefore not sufficient by itself. Annual noise exists, but model form and sector structure remain larger issues.

Sector dominance:

- Readiness improves over background in only about 3.7% of sectors.
- Most major sectors show `sector background dominates`.
- This is the strongest explanation for the Phase 12 failure: a single global readiness equation is too blunt for heterogeneous sectoral transition pathways.

Capability measurement:

| Capability type | Source | Rows | Readiness corr. | Capability corr. | Background MAE | Readiness MAE | Improvement |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| general | atlas_observed | 11,351 | -0.10069767390255496 | -0.04413947433184798 | 0.16783671713306222 | 0.1768293004696207 | -0.008992583336558474 |
| general | io_imputed | 5,904 | -0.15186993146866518 | -0.08752217390518062 | 0.17610030920332573 | 0.18593421477429595 | -0.00983390557097022 |
| general | unavailable | 1,228 | -0.10984773374684147 |  | 0.18487027445053855 | 0.19464991167874646 | -0.009779637228207905 |
| green | atlas_observed | 11,351 | -0.10069767390255496 | -0.033282400781671904 | 0.16783671713306222 | 0.1768293004696207 | -0.008992583336558474 |
| green | io_imputed | 5,108 | -0.1636472965851533 | 0.011387689876767478 | 0.17848595472673298 | 0.18834127332578238 | -0.009855318599049395 |
| green | unavailable | 2,024 | -0.07572665068787472 |  | 0.1754005265896958 | 0.1851474661170196 | -0.009746939527323806 |

Interpretation:

- Atlas-observed nodes perform better than IO-imputed nodes, but readiness still underperforms background in both groups.
- IO-imputed capability remains acceptable for integration coverage, not for strong calibration claims.
- Capability measurement alone does not explain the weak calibration; the readiness equation and sector structure need revision.

Frontier specification:

| Frontier | Target correlation | Gap-only MAE | Readiness-gated MAE | Mean gap | Share zero gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| sector_year_p50 | 0.1083919940783424 | 0.17675152898102478 | 0.1757864974659121 | 0.4076550049396943 | 0.505275117675702 |
| rolling_sector_p25 | 0.11254897334818942 | 0.1794337375163404 | 0.1780291501860581 | 0.5772190794956281 | 0.36330682248552726 |
| sector_year_p25 | 0.11230079413291275 | 0.18282101186705138 | 0.18092316605216316 | 0.78646069559454 | 0.2533679597467943 |
| rolling_sector_p10 | 0.11380124636200858 | 0.18661770579235973 | 0.1841379212192752 | 1.0661957212418716 | 0.16398853000054103 |
| sector_year_p10 | 0.11294494374262891 | 0.18966402728848045 | 0.18673428618576873 | 1.3127919367899328 | 0.10685494778986096 |

Interpretation:

- The current p25 frontier is probably too aggressive or too noisy for historical annual calibration.
- The p50 frontier improves simple frontier diagnostics and creates a more conservative feasible benchmark.
- This does not prove p50 is final, but it is a concrete candidate for the next diagnostic calibration round.

Macro/country-year shocks:

- Year-level residuals vary meaningfully, with yearly mean residual standard deviation of 0.04787639260451795.
- 2015 shows especially high target volatility and mean absolute residual.
- Historical calibration diagnostics likely need year or country-year controls to separate shocks from structural transition readiness.

Most likely explanation for Phase 12 weakness:

1. Sectoral transition pathways dominate a single global readiness equation.
2. The current p25 frontier is probably not the best feasible benchmark.
3. Annual rEI is noisy, but smoothing or medium horizons alone do not make readiness empirically active.
4. Capability measurement matters, especially Atlas versus IO-imputed, but measurement error is not the only problem.
5. Macro/year shocks contaminate annual calibration and should be controlled diagnostically, not built directly into scenario rules.

Recommended next modelling action:

- Build a Phase 14 diagnostic calibration variant using sector-family-specific rules, a conservative p50 or rolling frontier benchmark, and optional historical year/country-year controls for calibration diagnostics.
- Keep scenarios paused.
- Do not treat current readiness parameters as scenario-ready.

## Phase 14 Theory-Structured Transition Variant Comparison

Phase 14 compares emissions-transition rule structures. It is not a policy scenario phase, does not change production dynamics, does not overwrite `config.py`, and does not create projection outputs.

Outputs created:

- `data/abm_v4/validation/emissions_transition_variant_results.csv`
- `data/abm_v4/validation/emissions_transition_variant_by_sector_family.csv`
- `data/abm_v4/validation/emissions_transition_variant_by_capability_source.csv`
- `data/abm_v4/validation/emissions_transition_variant_best_parameters.json`
- `data/abm_v4/validation/emissions_transition_variant_recommendation.csv`
- `data/abm_v4/validation/emissions_transition_variant_report.md`

Variants tested:

- `sector_background_only`
- `sector_background_plus_year_country_controls`, marked diagnostics only
- `frontier_gap_only`
- `global_frontier_gap_readiness`
- `sector_family_frontier_gap_readiness`
- `gated_readiness_by_sector_signal`
- `readiness_without_capability`
- `capability_only_readiness`

Target horizons tested:

- Phase 13 identified `one_year_rEI` as the clear target in the H1 diagnosis, so the default Phase 14 run used `one_year_rEI`.
- The CLI can also run `smoothed_one_year_rEI` and `three_year_rEI` through repeated `--transition-variant-target` arguments.

Frontier variants tested:

- `sector_year_p25`
- `sector_year_p50`
- `rolling_sector_p25`
- `rolling_sector_p50`

Recommendation table:

| Recommended model | Target | Frontier | Validation MAE | Improvement over sector background pct | Wrong-sign share | Correlation | Interpretation |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `frontier_gap_only` | `one_year_rEI` | `rolling_sector_p50` | 0.1716668017183566 | -0.00034238811920846564 | 0.5366012011037169 | -0.013911549273459793 | Readiness variants do not beat sector background; keep a conservative sector-background plus frontier-gap historical rule. |

Main findings:

- The diagnostic year/country control variant improves validation MAE to 0.16033900149809427 and lowers wrong-sign share to 0.41411026348536495, confirming that historical rEI is shock-sensitive. These controls are not future scenario mechanisms.
- Sector-family readiness does not clear the 5% improvement rule; its best validation MAE is essentially tied with, and slightly worse than, sector background.
- Gated readiness does not improve over sector background in the default Phase 14 comparison.
- Capability-only readiness does not outperform background, so capability terms should not yet be treated as a calibrated historical EI mechanism.
- `p50` frontier variants are more conservative than `p25`; the best non-control conservative diagnostic is `rolling_sector_p50`, but it still does not beat sector background.
- Readiness should be deferred to selected sectors or later scenario hypotheses until stronger historical evidence appears.
- Scenarios remain premature.

Recommended Phase 15:

- Validate a conservative historical transition rule in a multi-year base run: sector background plus a cautiously specified frontier gap.
- Keep readiness available as a diagnostic or selected-sector scenario mechanism, not as a global calibrated historical mechanism.
- Keep historical year/country controls confined to diagnostics.
- Revisit sector-family mapping and frontier definitions only after checking whether the conservative rule improves multi-year EI and emissions validation.

## Phase 15 Calibrated-Historical Frontier-Gap Base Validation

Phase 15 applies the Phase 14 recommendation as an explicit historical emissions-transition mode. This is not a scenario phase, does not remove historical production forcing, does not rebuild raw-T edges, and does not overwrite the active default `config.py` emissions transition.

Named historical mode:

- `historical_frontier_gap_only`

Equation:

```text
rEI_{i,t+1} =
  alpha_sector_background_{s,t}
  + rho_gap * Gap_EI_{i,t} / (Gap_EI_{i,t} + tau_gap)

Gap_EI_{i,t} =
  max(0, log(EI_{i,t}) - log(EI_frontier_{s,t}))
```

Rolling frontier definition:

- `EI_frontier_{s,t}` is the rolling sector p50 frontier using positive EI observations in sector `s` up to year `t`.
- The rolling frontier does not use future years.
- Readiness, capability, network exposure, brown centrality, and supplier lock-in do not enter `rEI_used` in this historical base rule.

Parameters used:

| Parameter | Value | Source |
| --- | ---: | --- |
| `rho_gap` | 0.03177485903146337 | `data/abm_v4/validation/emissions_transition_variant_best_parameters.json` |
| `tau_gap` | 1.8570918527926128 | `data/abm_v4/validation/emissions_transition_variant_best_parameters.json` |

The parameter source reported by the run is `phase14_global_parameters:one_year_rEI|rolling_sector_p50`. Fallback parameters were not used in the full manual validation run. If the file is missing or malformed, the implemented fallback is `rho_gap = 0.03` and `tau_gap = 1.0`, and that fallback is reported in outputs rather than written to `config.py`.

Outputs created:

- `data/abm_v4/simulations/base_multiyear_state_panel_historical_frontier_gap.parquet`
- `data/abm_v4/simulations/base_multiyear_summary_panel_historical_frontier_gap.csv`
- `data/abm_v4/diagnostics/base_multiyear_yearly_diagnostics_historical_frontier_gap.csv`
- `data/abm_v4/validation/base_multiyear_validation_report_historical_frontier_gap.csv`
- `data/abm_v4/validation/base_multiyear_validation_report_historical_frontier_gap.md`
- `data/abm_v4/validation/multiyear_base_model_comparison.csv`
- `data/abm_v4/validation/multiyear_base_model_comparison.md`

Comparison with the previous `frontier_gap_readiness` base:

| Model | Latest aggregate emissions pct error | Mean yearly aggregate emissions pct error | Mean log EI error | rEI MAE | Wrong-sign share | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `frontier_gap_readiness` | 0.7804703868079855 | 0.4081294910805779 | 0.11812465009461429 | 0.14238925439963335 | 0.28364297295343527 | warning |
| `historical_frontier_gap_only` | 0.8545175626056115 | 0.570628904824899 | 0.11251959786407684 | 0.14054253111602777 | 0.3829323471380019 | warning |

Interpretation:

- The calibrated-historical conservative rule slightly improves mean log EI error and rEI MAE.
- It worsens latest aggregate emissions pct error and mean yearly aggregate emissions pct error.
- It also worsens wrong-sign share.
- Sector-background dominance remains; the frontier-gap term alone is not enough to make the historical base scenario-ready.
- The model remains not scenario-ready.
- Historical production forcing remains the central caveat because production is still observed/forced rather than endogenously propagated.

Recommended Phase 16:

- Diagnose why improved rEI MAE does not translate into improved aggregate emissions validation.
- Compare aggregate-weighted losses against node-average rEI losses before choosing a calibrated historical objective.
- Inspect high-emissions sectors and countries driving the worsened aggregate emissions error.
- Keep readiness deferred and keep scenarios paused until the historical base improves on both transition and aggregate emissions metrics.

## Phase 16 Transition Rule Error Decomposition and Sign-Failure Diagnosis

Phase 16 decomposes the Phase 15 tradeoff between the previous `frontier_gap_readiness` multi-year base and the calibrated-historical `historical_frontier_gap_only` base. It does not run scenarios, does not recalibrate parameters, does not rerun simulations, does not change production dynamics, and does not overwrite `config.py`.

Outputs created:

- `data/abm_v4/validation/transition_rule_error_decomposition.csv`
- `data/abm_v4/validation/transition_rule_sign_failure_panel.parquet`
- `data/abm_v4/validation/transition_rule_sign_failure_by_year.csv`
- `data/abm_v4/validation/transition_rule_sign_failure_by_sector.csv`
- `data/abm_v4/validation/transition_rule_sign_failure_by_country.csv`
- `data/abm_v4/validation/transition_rule_sign_failure_by_ecosystem.csv`
- `data/abm_v4/validation/transition_rule_sign_failure_by_capability_source.csv`
- `data/abm_v4/validation/transition_rule_sign_failure_by_decile.csv`
- `data/abm_v4/validation/transition_rule_aggregate_contribution.csv`
- `data/abm_v4/validation/transition_rule_hypothesis_tests.csv`
- `data/abm_v4/validation/transition_rule_error_tradeoff_report.md`

Hypotheses tested:

- H1: frontier-gap-only behaves like a mean-reversion rule.
- H2: readiness was acting as a shrinkage mechanism.
- H3: the selected metric optimized the wrong objective.
- H4: rolling p50 frontier is better for average nodes but worse for central nodes.
- H5: annual direction is shock-dominated, but magnitude is structurally predictable.
- H6: sector background is doing most of the economic work.
- H7: the frontier definition is still incomplete.
- H8: capability/readiness should be conditional, not global.
- H9: IO-imputed capability is not the main issue here.
- H10: historical production forcing may amplify EI errors into emissions errors.

Weighted transition diagnostics:

| Model | Unweighted rEI MAE | Output-weighted rEI MAE | Emissions-weighted rEI MAE | Unweighted wrong-sign share | Output-weighted wrong-sign share | Emissions-weighted wrong-sign share | Mean emissions abs error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `frontier_gap_readiness` | 0.14238925439963335 | 0.08662781800820019 | 0.13075200431187453 | 0.5373387188413582 | 0.5349882938490472 | 0.37720451217258033 | 4788.881392014762 |
| `historical_frontier_gap_only` | 0.14054253111602774 | 0.08506103151203555 | 0.12994752782968327 | 0.3829323471380019 | 0.3536402339130191 | 0.3381510086061846 | 5865.019607938777 |

Interpretation:

- `historical_frontier_gap_only` is better on unweighted, output-weighted, and emissions-weighted rEI MAE in the direct transition panel.
- The direct transition panel also gives `historical_frontier_gap_only` better wrong-sign shares. This is more nuanced than the Phase 15 headline comparison and shows that sign conclusions depend on the exact transition panel and weighting definition.
- Despite better transition metrics, `historical_frontier_gap_only` worsens emissions absolute error and aggregate emissions fit.
- The aggregate emissions worsening is concentrated in a small number of large high-emissions nodes, especially `Electricity, Gas and Water`.

Top aggregate worsening drivers:

- The top 20 node-years explain 36.4% of the positive aggregate emissions-error deterioration.
- The largest contributors are overwhelmingly `Electricity, Gas and Water`.
- China electricity and water node-years from 2001-2015 dominate the top ranks, followed by USA electricity and water node-years and one Russia electricity and water node-year.
- The largest single contributor is `CHN | CHN | Commodities | Electricity, Gas and Water` in 2011, with an added absolute emissions-error difference of about 4.55 million.

Sector and country concentration:

- `Electricity, Gas and Water` accounts for about 38.8% of observed emissions in the transition panel and about 73.3% of the aggregate error deterioration.
- The next largest sector contributions are `Electrical and Machinery`, `Petroleum, Chemical and Non-Metallic Mineral Products`, `Construction`, and `Metal Products`.
- China accounts for about 21.4% of observed emissions but about 66.5% of aggregate error deterioration.
- USA and Russia are the next largest country contributors.
- Some countries, including Poland and India, show the pattern `magnitude improves but sign worsens`, but this is not the dominant aggregate pattern.

Capability-source diagnosis:

- Capability-source differences are small for the frontier-gap-only rule.
- H9 is supported: capability source remains a useful slice, but the Phase 15 aggregate problem is not primarily a direct IO-imputed capability problem because the historical frontier-gap-only rule does not use capability terms.

Hypothesis evidence:

| Hypothesis | Evidence strength | Supports? | Key metric | Interpretation |
| --- | --- | --- | ---: | --- |
| H1 mean reversion | strong | yes | 0.6282338342010845 | Frontier-gap-only wrong signs often occur where nodes have positive gap but observed rEI is negative. |
| H2 readiness shrinkage | weak | no | -0.02949852507374634 | Low-readiness nodes do not explain frontier-gap-only sign failures in this diagnostic. |
| H3 wrong objective | moderate | yes | -0.010595003286344468 | Weighted and unweighted objectives differ enough that model selection should include weighted metrics. |
| H4 p50 worse for central nodes | weak | no | -0.0009462283265040863 | Top-output decile does not show larger rEI deterioration, though aggregate emissions deterioration is concentrated in large nodes. |
| H5 annual shocks | moderate | yes | 0.19296147452962467 | Wrong-sign shares cluster by year/country. |
| H6 sector background dominance | weak | no | -0.001522359355262775 | Best-readiness sectors are not clearly worsened by the frontier term. |
| H7 incomplete frontier | weak | no | 0.1953376124299678 | Sector concentration exists but is not enough alone to establish frontier incompleteness. |
| H8 conditional readiness | weak | no | 0.0 | Readiness does not beat gap-only by sector in this transition panel. |
| H9 IO capability not main issue | moderate | yes | 0.004219631906933863 | Capability-source sign differences are small. |
| H10 production forcing amplification | weak | no | 0.3640640143979693 | Top 20 node-years matter substantially but do not exceed the strong concentration threshold. |

Main diagnosis:

- Phase 15 was mixed because node-average transition metrics and aggregate emissions metrics point in different directions.
- The conservative frontier-gap-only rule improves average transition fit, but it worsens emissions fit in systemically important high-emissions electricity nodes.
- The issue is less a broad sign failure and more an aggregation/weighting and large-node sensitivity problem.
- The p50 frontier is not enough for large electricity-system nodes.
- Historical production forcing amplifies EI errors into aggregate emissions errors because observed production scale is multiplied directly by simulated EI.

Recommended Phase 17:

- Do not implement scenarios.
- Do not adopt `historical_frontier_gap_only` as the default yet.
- Build a validation-objective diagnostic that compares unweighted rEI MAE, output-weighted rEI MAE, emissions-weighted rEI MAE, wrong-sign share, and aggregate emissions error.
- Add a high-emissions-node diagnostic for electricity and other large contributors before testing any hybrid rule.
- Only after that, test a weak dampened frontier-gap hybrid as a diagnostic candidate, not as a default.

## Phase 17 High-Emissions Node and Readiness-Dampening Diagnosis

Phase 17 focuses on the Phase 16 conflict without running scenarios, recalibrating, changing parameters, changing production dynamics, or implementing a hybrid rule. It reads the Phase 16 transition-rule comparison panel and decomposes whether the aggregate emissions deterioration is concentrated in high-emissions nodes and whether the previous readiness rule acted as a dampener.

Implemented diagnostics:

- `HighEmissionsDampeningDiagnostics` in `src/abm_v4/validation.py`.
- CLI flag: `--diagnose-high-emissions-dampening`.
- Output paths are defined under `ABMV4Paths` and are written only when `--create-output-dirs` is provided.
- The diagnostic uses existing Phase 16 outputs and fails with an actionable message if `transition_rule_sign_failure_panel.parquet` is missing.

Outputs written:

- `data/abm_v4/validation/high_emissions_concentration_diagnostic.csv`
- `data/abm_v4/validation/electricity_sector_dampening_diagnostic.csv`
- `data/abm_v4/validation/china_electricity_transition_diagnostic.csv`
- `data/abm_v4/validation/readiness_dampening_diagnostic.csv`
- `data/abm_v4/validation/simplified_model_selection_tradeoff.csv`
- `data/abm_v4/validation/phase17_recommendation.csv`
- `data/abm_v4/validation/phase17_high_emissions_dampening_report.md`

High-emissions concentration results:

- Electricity, Gas and Water accounts for about 38.8% of observed emissions in the Phase 16 node-year panel and about 61.4% of positive aggregate deterioration under `historical_frontier_gap_only`.
- China (`CHN`) accounts for about 21.4% of observed emissions and about 50.0% of positive aggregate deterioration.
- The top 10 node-years account for about 27.5% of positive deterioration; the top 20 account for about 36.4%; the top 50 account for about 48.9%.
- The concentration is therefore real, but not just a single node-year outlier.

Electricity-sector results:

- Electricity nodes have lower direct transition wrong-sign share under `historical_frontier_gap_only` than under `frontier_gap_readiness` in the recomputed Phase 16 panel.
- Their aggregate emissions error is worse under `historical_frontier_gap_only`: about 269.7 million absolute emissions-error units versus about 193.4 million for `frontier_gap_readiness`.
- The largest positive deterioration rows are repeatedly China Electricity, Gas and Water in 2006-2015.

China and China-electricity results:

- China-level emissions error is worse under `historical_frontier_gap_only`: about 261.0 million versus about 191.7 million for `frontier_gap_readiness`.
- China Electricity, Gas and Water is the dominant recurring node in the top deterioration rows.
- In several China electricity years, `historical_frontier_gap_only` moves simulated rEI closer to observed rEI, but still worsens absolute emissions error because the node is very large and production is historically forced.
- In other years with observed negative rEI, both rules predict improvement, and `historical_frontier_gap_only` is more aggressive.

Readiness dampening results:

- Across all node-years, `historical_frontier_gap_only` is more aggressive than `frontier_gap_readiness` about 67.8% of the time.
- The mean dampening amount is about 0.017 rEI units overall.
- For Electricity, Gas and Water, the mean dampening amount is about 0.0149 and `historical_frontier_gap_only` is more aggressive about 61.2% of the time.
- For China, the mean dampening amount is about 0.0078 and `historical_frontier_gap_only` is more aggressive about 56.6% of the time.
- This supports the interpretation that readiness partly acted as a dampener, but the main issue is the interaction between high-emissions nodes and aggregate validation, not a universal transition-sign failure.

Simplified model-selection tradeoff:

- `historical_frontier_gap_only` wins unweighted, output-weighted, and emissions-weighted rEI MAE.
- `historical_frontier_gap_only` also wins recomputed unweighted, output-weighted, and emissions-weighted wrong-sign shares in the Phase 16 transition panel.
- `frontier_gap_readiness` wins latest-year aggregate emissions pct error, mean yearly aggregate emissions pct error, high-emissions-node emissions error, electricity-sector emissions error, and China emissions error.

Phase 17 recommendation:

- Recommendation: `inspect_electricity_data_before_hybrid`.
- Evidence: electricity deterioration share is about 0.614, China deterioration share is about 0.500, and the top 10 node-years account for about 0.275 of positive deterioration.
- Phase 18 should inspect the China electricity data path and EI series before adding a new mechanism.
- After that inspection, a dampened frontier-gap hybrid remains a plausible diagnostic candidate, but it should not be implemented as the default yet.
- Scenarios remain premature because no rule yet performs acceptably on both average transition metrics and high-emissions aggregate metrics.

## Phase 18 Electricity and China EI Data Audit

Phase 18 audits the electricity and China EI data behind the Phase 17 high-emissions deterioration. It does not run scenarios, recalibrate parameters, implement a hybrid rule, change production dynamics, overwrite `config.py`, or modify v1/v2/v3 sources.

Implemented audit:

- `ElectricityDataAudit` in `src/abm_v4/validation.py`.
- CLI flag: `--audit-electricity-data`.
- Output paths are defined under `ABMV4Paths` and are written only when `--create-output-dirs` is provided.
- The audit reads the ABM v4 observed state panel, the previous `frontier_gap_readiness` multi-year state panel, the `historical_frontier_gap_only` state panel, and Phase 16/17 diagnostic outputs when available.

Outputs written:

- `data/abm_v4/validation/electricity_node_inventory.csv`
- `data/abm_v4/validation/china_electricity_observed_series_audit.csv`
- `data/abm_v4/validation/china_electricity_model_series_audit.csv`
- `data/abm_v4/validation/electricity_anomaly_flags.csv`
- `data/abm_v4/validation/electricity_cross_country_comparison.csv`
- `data/abm_v4/validation/electricity_data_audit_recommendation.csv`
- `data/abm_v4/validation/electricity_data_audit_report.md`

Electricity inventory:

- The audit identifies 189 electricity-like ABM v4 nodes using case-insensitive sector matching for electricity, gas, water, utilities, and power labels.
- China Electricity, Gas and Water is clearly identified as `CHN | CHN | Commodities | Electricity, Gas and Water`.
- China is the largest electricity-like node by total observed emissions in the audit window, with about 78.8 million observed emissions units and 22 years available.
- The next largest nodes are USA, Russia, India, Japan, Germany, South Africa, Saudi Arabia, Korea, and Poland.

China electricity observed data audit:

- China electricity has unusually high EI relative to the electricity sector early in the sample: its sector-year EI percentile is near the top of the electricity distribution from the late 1990s through the 2000s.
- Its gap to the rolling sector p50 frontier is very large early in the sample, about 2.14 log points in 1995, and remains positive through 2016.
- Observed EI falls sharply over several periods while observed production rises sharply, so aggregate emissions can rise even when EI improves.
- Large observed production jumps appear in 1998, 2005, and 2007.
- Large observed EI log-change flags appear in 1998, 2005, and 2007.
- Large observed emissions jump flags appear around 2002, 2003, and 2006.

Anomaly flags:

- Active focused-audit flags include 43 sign-reversal flags, 21 model-disagreement flags, 4 EI jump flags, 4 production jump flags, 4 emissions jump flags, and 4 extreme frontier-gap flags.
- For China electricity specifically, the severe audit count used by the recommendation logic is 9.
- There are no non-positive or missing EI flags in the focused China electricity audit, so the issue is not a simple missingness or invalid-level failure.
- The evidence instead points to structural breaks, accounting/composition shifts, or a sector-specific electricity transition path.

Cross-country electricity comparison:

- China electricity accounts for about 25.5% of observed emissions among electricity-like nodes in the cross-country comparison.
- China has mean observed EI of about 0.0157 and mean observed rEI of about -0.099 in the audit convention.
- China has high rEI volatility, about 0.154, and a large mean gap to rolling p50, about 1.24 log points.
- Both rules have large China electricity transition errors, but `historical_frontier_gap_only` is slightly better on rEI MAE while worse on aggregate emissions error.
- China electricity emissions error is about 113.6 million under `frontier_gap_readiness` and about 163.7 million under `historical_frontier_gap_only`.

Model behavior diagnosis:

- `historical_frontier_gap_only` often moves China electricity rEI closer to the observed transition magnitude, consistent with the Phase 16 and Phase 17 transition-metric results.
- However, because China electricity is very large, modest EI simulation differences become large aggregate emissions differences under historical production forcing.
- Readiness dampening appears relevant, but the data audit shows enough structural and accounting-like breaks that raw data inspection should come before adding a hybrid mechanism.

Phase 18 recommendation:

- Recommendation: `inspect_raw_eora_electricity_data`.
- Evidence: `severe_china_flags=9`, `over_improve_share=0.500`, and `dampening_help_share=0.455`.
- Phase 19 should inspect the raw and transformed Eora electricity path for China before testing a dampened frontier-gap hybrid.
- If the raw data path is clean, then treat electricity as a sector-specific transition case and only then test a weak dampened frontier-gap rule as a diagnostic candidate.
- Scenarios remain premature.

## Phase 19 Raw Eora Electricity Data Path Audit

Phase 19 traces the China Electricity, Gas and Water anomaly through existing Eora-derived files. It does not run scenarios, recalibrate, implement a hybrid rule, change production dynamics, overwrite `config.py`, rebuild raw-T edges, or modify protected v1/v2/v3 sources.

Implemented audit:

- `RawEoraElectricityDataAudit` in `src/abm_v4/validation.py`.
- CLI flag: `--audit-raw-eora-electricity-data`.
- Output paths are defined under `ABMV4Paths` and are written only when `--create-output-dirs` is provided.
- The audit inspects candidate schemas defensively and reports missing or unusable sources rather than assuming column names.

Outputs written:

- `data/abm_v4/validation/raw_eora_electricity_source_inventory.csv`
- `data/abm_v4/validation/raw_eora_china_electricity_series_by_source.csv`
- `data/abm_v4/validation/raw_eora_china_electricity_cross_source_comparison.csv`
- `data/abm_v4/validation/raw_eora_electricity_scaling_audit.csv`
- `data/abm_v4/validation/raw_eora_electricity_mapping_audit.csv`
- `data/abm_v4/validation/raw_eora_electricity_breakpoint_audit.csv`
- `data/abm_v4/validation/raw_eora_major_electricity_comparison.csv`
- `data/abm_v4/validation/raw_eora_electricity_data_audit_recommendation.csv`
- `data/abm_v4/validation/raw_eora_electricity_data_audit_report.md`

Sources inspected:

- The audit inspects ABM v4 input/interim/simulation files, ABM v3 input panels, and final Eora-Atlas panels.
- It avoids generated validation/diagnostic outputs and skips very large raw-T edge files.
- Fifteen usable China electricity sources were identified, including ABM v3 historical panels, the ABM v4 state panel, ABM v4 multi-year state panels, and final Eora-Atlas panels.

China electricity lineage:

- China Electricity, Gas and Water is consistently extracted from the ABM v4 state panel and upstream ABM v3/final panels.
- Mapping audit result: all extracted source-year records are `consistent`; no duplicate-record mapping issue remains in the full-window lineage sources.
- The audit finds no evidence that the Phase 18 anomaly is caused by a simple country/sector mapping failure.

Output, emissions, and EI consistency:

- Cross-source comparison statuses include 1,929 exact or near matches, 913 mismatches, 25 possible scale-difference comparisons, and missing-variable comparisons where a source lacks output or emissions candidates.
- The mismatch share used for the recommendation is about 0.133 after excluding missing-variable comparisons.
- No stable scale factor was detected in the scaling audit (`scale_flags=0`).
- The evidence does not point to a clean unit conversion problem such as 1,000 or 1,000,000.

Breakpoint and jump findings:

- China electricity has 14 jump flags in the preferred ABM v4 state lineage.
- China jump years include 1998, 2002, 2003, 2005, and 2007.
- The largest China electricity EI/rEI jumps are tied to sharp EI declines in 1998, 2005, and 2007.
- Those EI declines coincide with large output jumps in 1998, 2005, and 2007.
- Emissions jumps are flagged in 2002 and 2003.
- This means the severe EI behavior comes from both output and emissions dynamics, with output surges especially important in the large negative rEI years.

Comparison with other major electricity nodes:

- China is not the only major electricity node with jump behavior.
- Jump counts among major electricity nodes include Russia, Korea, China, Australia, Saudi Arabia, Japan, South Africa, USA, Taiwan, and India.
- China is especially aggregate-important: it is the second-largest electricity emitter early in the sample and becomes the largest electricity emitter in the preferred comparison source from 2003 onward.
- China output rank rises from fourth in the mid-1990s to first from 2008 onward.

Final diagnosis:

- The China electricity anomaly does not look like a simple invalid EI, missing EI, mapping, duplicate-record, or stable scale-factor problem.
- The raw Eora-derived path appears broadly consistent enough that immediate repair is not the main recommendation.
- The observed behavior is better interpreted as an electricity-sector structural-break and accounting/transition-dynamics issue that generic emissions-transition rules do not represent well.
- This supports treating electricity as a sector-specific transition case before testing any dampened hybrid rule.

Phase 19 recommendation:

- Recommendation: `treat_electricity_as_sector_specific_transition_case`.
- Evidence: `mismatch_share=0.133`, `scale_flags=0`, `mapping_problems=0`, `china_jumps=14`, and `other_jump_nodes=9`.
- Recommended Phase 20: build electricity-specific transition diagnostics that separate output-growth, emissions-growth, and EI-change regimes for major electricity nodes before testing a dampened frontier-gap hybrid.
- Keep `frontier_gap_readiness` as the temporary aggregate-safe base and keep `historical_frontier_gap_only` as a transition-mechanism candidate.
- Scenarios remain premature.

## Phase 20 Electricity-Specific Transition Regime Diagnostics

Phase 20 compares electricity-specific historical transition-rule candidates without making any rule active by default. It does not run scenarios, recalibrate globally, implement a final electricity rule, change production dynamics, overwrite `config.py`, rebuild raw-T edges, or rerun the Phase 19 raw audit.

Implemented diagnostics:

- `ElectricityTransitionRegimeDiagnostics` in `src/abm_v4/validation.py`.
- CLI flag: `--diagnose-electricity-transition-regime`.
- The diagnostic requires existing Phase 19 outputs and fails clearly if they are absent.
- Candidate rules are evaluated as diagnostic predictions only.

Outputs written:

- `data/abm_v4/validation/electricity_transition_target_diagnostics.csv`
- `data/abm_v4/validation/electricity_transition_rule_comparison.csv`
- `data/abm_v4/validation/electricity_transition_rule_by_country.csv`
- `data/abm_v4/validation/electricity_transition_rule_by_year.csv`
- `data/abm_v4/validation/electricity_transition_rule_by_decile.csv`
- `data/abm_v4/validation/electricity_transition_rule_by_jump_status.csv`
- `data/abm_v4/validation/china_electricity_rule_comparison.csv`
- `data/abm_v4/validation/electricity_transition_regime_recommendation.csv`
- `data/abm_v4/validation/electricity_transition_regime_report.md`

Why electricity is treated as sector-specific:

- Phase 19 found no mapping, duplicate-record, missing-EI, non-positive-EI, or stable-scale-factor explanation for the China electricity anomaly.
- Phase 20 therefore treats electricity as an energy-system transition regime: output growth, fossil generation lock-in, capacity expansion, and structural breaks can dominate generic country-sector mean reversion.
- The diagnostic includes 178 electricity-like country-sector nodes and 3,729 one-year transition rows.

Target diagnostics:

- One-year electricity rEI has high volatility: standard deviation about 0.299, with p05 about -0.340 and p95 about 0.422.
- Smoothed one-year rEI is more stable: standard deviation about 0.159.
- Three-year annualized rEI is similarly stable: standard deviation about 0.163.
- Winsorized one-year rEI reduces volatility relative to raw one-year rEI: standard deviation about 0.185.
- China is only about 0.56% of electricity transition rows but about 25.2% of observed electricity emissions in the target diagnostic panel.

Candidate rules compared:

- `current_frontier_gap_readiness_reference`
- `historical_frontier_gap_only_reference`
- `electricity_sector_background_only`
- `electricity_rolling_frontier_gap_only`
- `electricity_dampened_frontier_gap_0_25`
- `electricity_dampened_frontier_gap_0_50`
- `electricity_dampened_frontier_gap_0_75`
- `electricity_readiness_dampened_frontier_gap`
- `electricity_gap_with_jump_shock_filter`
- `electricity_high_emissions_dampened_gap`

Rule comparison results:

- The best aggregate electricity emissions fit in this diagnostic table is `electricity_rolling_frontier_gap_only`, followed closely by `electricity_dampened_frontier_gap_0_75`.
- The Phase 20 recommendation chooses `electricity_dampened_frontier_gap_0_75` as the fixed dampened candidate because it improves both transition metrics and electricity aggregate emissions error relative to the current readiness reference.
- `electricity_dampened_frontier_gap_0_75` has unweighted rEI MAE about 0.1648 and electricity aggregate emissions error about 28.0 million.
- The current readiness reference has unweighted rEI MAE about 0.1775 and electricity aggregate emissions error about 30.8 million in this electricity-only diagnostic convention.
- The historical frontier-gap-only reference has unweighted rEI MAE about 0.1787 and electricity aggregate emissions error about 31.7 million in this diagnostic convention.

China electricity comparison:

- China remains a special aggregate case.
- The current readiness reference still has slightly lower China electricity emissions error than the dampened p75 diagnostic candidate in the Phase 20 table.
- The dampened p75 candidate improves average electricity transition and sector aggregate metrics, but it is not yet a final China-specific solution.
- China jump years such as 1998, 2005, and 2007 remain difficult because observed EI reductions are large and output is expanding rapidly.

Jump-year diagnostics:

- Jump years are only about 0.83% of transition rows but about 10.7% of observed electricity emissions.
- The current readiness reference has jump-year rEI MAE about 0.264.
- `electricity_dampened_frontier_gap_0_75` reduces jump-year rEI MAE to about 0.210 and lowers jump-year wrong-sign share to about 0.097.
- The jump-filter rule also improves jump-year performance, but it does not beat the dampened p75 candidate on aggregate electricity emissions in this diagnostic comparison.

Phase 20 recommendation:

- Recommendation: `test_electricity_dampened_frontier_gap_as_candidate_rule`.
- Evidence: the selected fixed candidate is `electricity_dampened_frontier_gap_0_75`.
- Recommended Phase 21: test only this electricity-specific dampened frontier-gap rule as a diagnostic candidate while keeping scenarios blocked.
- Do not make it the active default yet.
- Keep `frontier_gap_readiness` as the temporary aggregate-safe historical base until a candidate rule is validated in the full multi-year loop.
- Scenarios remain premature.

## Phase 21 Structural-Signature Discovery and Parameter-Candidate Screening

Phase 21 asks whether electricity-like behaviour can be explained by more general observable ABM v4 structural signatures before implementing an electricity-specific transition rule. It does not run scenarios, recalibrate emissions, implement a final dampening rule, change production dynamics, overwrite `config.py`, rebuild raw-T edges, or rerun the Phase 19 raw audit.

Why Phase 21 did not immediately implement the Phase 20 electricity candidate:

- Phase 20 showed that `electricity_dampened_frontier_gap_0_75` is promising inside electricity diagnostics.
- But a sector-name exception would be theoretically weak if existing ABM v4 metrics already identify broader transition-inertia conditions.
- Phase 21 therefore inventories available metrics and screens whether electricity-like nodes are distinguished by general structural properties such as systemic scale, emissions centrality, volatility/jump behaviour, brown lock-in, supplier constraints, capability constraints, and model-error signatures.

Metric inventory:

- The diagnostic found usable metric families for production scale, emissions scale, emissions intensity, transition dynamics, volatility/jump behaviour, frontier gap, network position, supplier structure, buyer/supplier opportunity structure, capability, green capability, ecosystem mapping, production feasibility, model-error signatures, and electricity-specific diagnostics.
- The main node-year feature panel contains 108,130 rows.
- The node-level structural-signature panel contains 4,915 country-sector nodes.

Structural labels:

- Electricity-like nodes: 189 nodes, about 38.7% of cumulative observed emissions and about 2.4% of cumulative observed output.
- High-emissions nodes: 492 nodes, about 92.3% of cumulative observed emissions.
- High-output nodes: 492 nodes, about 87.3% of cumulative observed output.
- Jump-prone nodes: 10 nodes; all are electricity-like under the current jump evidence and account for about 28.1% of cumulative observed emissions.
- Aggregate-sensitive nodes: 463 nodes, about 67.6% of cumulative observed emissions.
- Needs-dampening nodes: 1,988 nodes, about 74.5% of cumulative observed emissions.

Electricity structural signature:

- Electricity-like nodes are strongly distinguished by high emissions-intensity level, high emissions scale, jump behaviour, and model-error signatures.
- Top electricity discriminators include `mean_log_EI_observed`, `median_log_EI_observed`, `p95_log_EI_observed`, `max_log_EI_observed`, `max_log_emissions_observed`, `p95_log_emissions_observed`, and `share_frontier_gap_worsens_sign`.
- Needs-dampening nodes are most directly identified by model-error signatures, especially `share_frontier_gap_worsens_emissions_error`, but structural EI-position metrics also carry signal.

Non-electricity lookalikes:

- The screen identifies non-electricity statistical lookalikes under the strongest electricity-discriminating metrics.
- These include a mix of high-EI or error-sensitive non-electricity country-sectors rather than a clean infrastructure-only set.
- This supports testing a generalized proxy, but it also warns against over-interpreting the proxy as a pure economic mechanism.

Candidate transition-inertia proxies:

- Candidate proxy table includes `systemic_scale_proxy`, `emissions_centrality_proxy`, `output_centrality_proxy`, `volatility_jump_proxy`, `brown_lockin_proxy`, `supplier_constraint_proxy`, `model_error_dampening_need_proxy`, and `composite_transition_inertia_proxy`.
- Recommended-for-Phase-22 candidates include systemic scale, volatility/jump, brown lock-in, model-error dampening need, and the composite proxy.
- The composite proxy is available and captures the strongest cross-family evidence, but it carries moderate overfitting risk because one component is model-error based.

Phase 21 recommendation:

- Recommendation: `build_composite_transition_inertia_proxy`.
- Evidence: multiple structural families distinguish electricity-like nodes, especially systemic scale, volatility/jump behaviour, and emissions scale; model-error signatures still explain much of the dampening-need label.
- Recommended Phase 22: test a transparent composite transition-inertia dampener diagnostically against both electricity and non-electricity difficult nodes.
- Keep the Phase 20 electricity dampened candidate as a benchmark, not an active default.
- Scenarios remain premature.

## Phase 22 Essential-Input and Structural-Dependence Diagnostics

Phase 22 does not proceed directly from Phase 21 to a high-EI/high-emissions composite proxy. The Phase 21 candidates are useful diagnostics, but they mostly describe symptoms: electricity is high-EI, high-emissions, volatile, and difficult for the emissions-transition rules to fit. Phase 22 instead asks whether electricity is structurally unusual in the IO network because it is an essential input: widely used by buyers, hard to substitute, persistent in supplier-buyer relationships, and systemically exposed downstream.

Implementation scope:

- No scenarios were created.
- No final active dampening rule was implemented.
- No emissions-transition recalibration was run.
- `config.py` was not overwritten.
- Production dynamics and historical production forcing were unchanged.
- Phase 19 raw Eora audit was not rerun.
- The full raw-T edge file was not rebuilt or loaded; diagnostics used compact ABM v4 supplier and historical edge outputs.

Inputs used:

- `supplier_updated_weights.parquet`
- `supplier_opportunity_sets.parquet`
- `historical_supplier_edges.parquet`
- `abm_v4_state_panel_1995_2016.parquet`
- Phase 21 structural-signature node panel

Dependence metrics computed:

- Buyer reach: buyer count, buyer country count, buyer sector count, buyer ecosystem count, buyer entropy, and buyer concentration.
- Input universality: share of all buyers, buyer sector coverage, buyer ecosystem coverage, and cross-ecosystem buyer reach.
- Buyer dependence: mean, median, p95, and max supplier share in buyer inputs, plus buyer dependency threshold counts.
- Low substitutability: opportunity scarcity and approximate relationship-stability metrics from compact historical edges.
- Systemic propagation potential: downstream output exposure, downstream emissions exposure, buyer output share, and buyer emissions share.
- Diagnostic scores: essential input, low substitutability, systemic dependence, and structural dependence.

Phase 22 output scale:

- Supplier-buyer dependence rows: 550,508.
- Supplier nodes with metrics: 4,531.

Electricity dependence signature:

- Electricity-like nodes are strongly distinguished by IO-dependence metrics, not only emissions symptoms.
- Strong electricity contrasts include low substitutability, structural dependence, relationship stability, buyer dependence, input universality, and downstream exposure.
- Electricity median percentile is high for key scores: structural dependence is about the 92.8th percentile, low substitutability about the 90.4th percentile, essential input about the 75.1st percentile, and input-universality/buyer reach about the 84.0th percentile under the candidate proxy table.
- This supports the theoretical claim that electricity behaves partly like an essential infrastructure input.

Dependence versus symptom metrics:

- Dependence scores are not simply rediscovering high EI or high emissions.
- Correlations with symptom metrics are moderate or low: for example, systemic dependence correlates about 0.606 with cumulative emissions share, while structural dependence correlates about 0.258 with cumulative emissions share and about 0.257 with mean log EI.
- Essential input score has low correlations with mean log EI, cumulative emissions share, jump frequency, and frontier-gap error symptoms.
- This suggests the IO-dependence metrics add a distinct structural concept.

Non-electricity dependence lookalikes:

- The top non-electricity lookalike is the `ROW / TOTAL` aggregate node, which is structurally broad but not theoretically clean.
- Other high-dependence lookalikes include financial intermediation/business activities, electrical and machinery, petroleum/chemical/non-metallic mineral products, construction, mining and quarrying, and re-export/re-import nodes.
- Some lookalikes are plausible foundational or systemically connected sectors, but the `ROW / TOTAL` and financial-service appearances require caution before turning the score into a model mechanism.

Candidate structural-dependence proxies:

- `essential_input_score`
- `input_universality_score`
- `buyer_dependence_score`
- `low_substitutability_score`
- `systemic_dependence_score`
- `structural_dependence_score`
- `structural_dependence_plus_brown_lockin`
- `structural_dependence_plus_volatility`

The candidate proxy table recommends these for Phase 23 diagnostics, but the recommendation is intentionally not a final rule. The evidence is strongest for identifying electricity-like essential-input structure and aggregate sensitivity. The relation to needs-dampening is present but weaker than the electricity/aggregate-sensitivity signal.

Phase 22 recommendation:

- Recommendation: `build_essential_input_dampener`.
- Interpretation: dependence metrics identify electricity-like essential-input structure, but dampening evidence is weaker.
- Recommended Phase 23: build only a diagnostic essential-input or structural-dependence dampener candidate and compare it against the Phase 20 electricity-specific candidate and existing rules.
- Do not make the dampener active by default.
- Keep scenarios blocked.
- Scenarios remain premature.

## Phase 23 Essential-Input Dampener Parameter Test

Phase 23 tested whether the Phase 22 essential-input dependence signature can act as a historical emissions-transition dampener inside ABM v4. This remains an ABM v4 country-sector parameter test, not an ABM v5 agent-type implementation.

The structural candidate rule was:

```text
rEI_i,t+1 =
alpha_s,t
+ D_EID_i,t * rho * Gap_EI_i,t / (Gap_EI_i,t + tau)

D_EID_i,t = clip(1 - lambda_EID * EID_i,t, d_min, 1)
```

The calibration-only historical residual variant was:

```text
rEI_i,t+1 =
alpha_s,t
+ D_EID_i,t * P_hist_i * rho * Gap_EI_i,t / (Gap_EI_i,t + tau)

P_hist_i = clip(1 + theta_i_shrunk, p_min, p_max)
```

`D_EID` is the structural mechanism candidate. `P_hist` is a bounded historical residual for missing policy, institutional, and energy-system variables. It is not scenario-facing.

Candidate grid:

- 157 candidates were evaluated.
- Baselines: `frontier_gap_readiness_baseline` and `historical_frontier_gap_only_baseline`.
- Structural scores: `essential_input_score_diagnostic`, `low_substitutability_score_diagnostic`, `systemic_dependence_score_diagnostic`, `structural_dependence_score_diagnostic`, plus cautious brown-lock-in and volatility composites.
- Structural dampening grid: `lambda_EID` in `{0.25, 0.50, 0.75, 1.00}` and `d_min` in `{0.25, 0.50, 0.75}`.
- Historical residual variants: country-sector with sector shrinkage, sector-only, and country-only residuals; shrinkage `k` in `{5, 10, 20}`; bounds `{0.75, 1.25}` and `{0.50, 1.50}`.
- Train period: 1995-2010.
- Validation period: 2011-2016.

Best validation rows:

- Historical frontier-gap-only baseline: emissions-weighted rEI MAE `0.13566`, electricity rEI MAE `0.19746`, China electricity rEI MAE `0.04814`, mean yearly aggregate emissions pct error `0.13469`.
- Best EID-only candidate: `c0061`, `structural_dependence_plus_brown_lockin`, `lambda_EID=1.0`, `d_min=0.25`; emissions-weighted rEI MAE `0.09032`, electricity rEI MAE `0.18464`, China electricity rEI MAE `0.03793`, mean yearly aggregate emissions pct error `0.08940`.
- Best residual-only candidate: sector residual, `k=5`; emissions-weighted rEI MAE `0.13315`, electricity rEI MAE `0.19819`, China electricity rEI MAE `0.05001`, mean yearly aggregate emissions pct error `0.13219`.
- Best EID plus residual candidate: `c0102`, `essential_input_score_diagnostic`, `lambda_EID=0.75`, `d_min=0.50`, country-sector residual with sector shrinkage, `k=10`, bounds `0.75-1.25`; emissions-weighted rEI MAE `0.09962`, electricity rEI MAE `0.18480`, China electricity rEI MAE `0.03515`, mean yearly aggregate emissions pct error `0.09871`.

Mechanism decomposition:

- The best combined candidate used `essential_input_score_diagnostic`.
- Mean `D_EID` was about `0.643`; electricity mean `D_EID` was about `0.502`; China electricity `D_EID` was `0.500`.
- Mean `P_hist` was about `1.013`; China electricity `P_hist` was about `1.057`.
- Structural-only validation gain was larger than residual-only gain.
- `residual_dominates_flag=false`.
- Interpretation: the structural dampener contributes independently; the residual is not the main source of improvement in the best combined result.

Main interpretation:

- The EID dampener improves electricity, China electricity, aggregate emissions, and all-node weighted transition metrics relative to the historical frontier-gap-only baseline.
- The best EID-only row does not improve the current high-EID slice enough to clear all decision thresholds, so it should not be made active by default yet.
- Historical residuals do not dominate the mechanism.
- The result supports keeping essential-input dependence as a serious ABM v4 diagnostic mechanism and a possible ABM v5 agent-type clue, but not as a final active rule.

Phase 23 recommendation:

- Recommendation: `keep_EID_dampener_as_diagnostic_only`.
- Recommended Phase 24: integrate the strongest EID dampener only as a diagnostic multi-year candidate, or inspect external policy and energy-system variables before any scenario-facing interpretation.
- ABM v5 implication: possible `essential_input_agent` candidate if high-EID improvement holds out of sample and under multi-year integration.
- Scenarios remain premature.

## Phase 24 Essential-Input Dampener Failure Modes

Phase 24 audited why the Phase 23 EID dampener helps electricity and aggregate emissions but does not cleanly validate across all high-EID nodes. It did not run scenarios, recalibrate, implement ABM v5, or make the dampener active.

High-EID subtype composition:

- `transport_logistics_infrastructure`: 246 nodes.
- `ordinary_or_unclear`: 194 nodes.
- `heavy_industry_materials`: 181 nodes.
- `knowledge_finance_business_services`: 150 nodes.
- `infrastructure_energy`: 138 nodes.
- `public_social_services`: 135 nodes.
- `manufacturing_system_core`: 74 nodes.
- `construction_real_estate_foundational`: 70 nodes.
- `accounting_or_pseudo_agent`: 9 nodes.

Emissions concentration by subtype:

- `infrastructure_energy` accounts for about 41.3% of observed emissions inside the high-EID audit set.
- `heavy_industry_materials` accounts for about 18.5%.
- `knowledge_finance_business_services` accounts for about 10.2%.
- `accounting_or_pseudo_agent` accounts for about 5.0%, but these nodes should not become behavioural ABM v5 agents.

Dampener performance by subtype:

- The best Phase 23 EID-only candidate `c0061` improves emissions-weighted rEI MAE relative to historical frontier-gap-only for every audited subtype.
- Improvement is clearest for accounting/pseudo-agent rows, heavy industry/materials, transport/logistics infrastructure, knowledge/business services, infrastructure energy, and construction/real-estate foundational nodes.
- Because accounting/pseudo-agent rows improve mechanically but are not real behavioural units, they should be separated from any future agent-type training or ontology work.

Failure modes:

- `helped_by_EID`: 672 nodes.
- `no_material_change`: 354 nodes.
- `high_EID_but_not_physical_transition_sector`: 103 nodes.
- `harmed_by_EID`: 40 nodes.
- `high_EID_but_low_emissions_relevance`: 17 nodes.
- `pseudo_agent_accounting_issue`: 9 nodes.

Pseudo-agent/accounting audit:

- 9 high-EID nodes are flagged as accounting or pseudo-agent categories.
- These should be kept in accounting totals where needed, but excluded from ABM v5 behavioural agent-type training unless manually justified.

ABM v5 implication:

- Plausible future ontology candidates: `energy_infrastructure_agent`, `transport_logistics_infrastructure_agent`, and `heavy_industry_materials_agent`.
- `systemic_service_agent` remains lower-confidence because structural centrality in services does not necessarily imply physical emissions-transition inertia.
- `accounting_node_not_agent` is explicitly not supported as a behavioural type.

Phase 24 recommendation:

- Recommendation: `integrate_EID_candidate_in_multiyear_loop_for_audit`.
- Interpretation: several coherent physical high-EID subtypes benefit from the diagnostic dampener.
- Recommended Phase 25: test the strongest EID candidate in the multi-year loop as an audit candidate only, while keeping pseudo-agent categories separated and preserving the current default rule.
- Scenarios remain premature.

## Phase 25 EID Diagnostic Multi-Year Integration Audit

Phase 25 integrated the best Phase 23/24 EID candidate into the recursive ABM v4 multi-year loop as a diagnostic mode only. It did not create scenarios, implement ABM v5, activate EID as a default rule, or alter production dynamics.

Diagnostic transition mode:

```text
historical_frontier_gap_EID_diagnostic
```

EID diagnostic equation:

```text
rEI_i,t+1 =
alpha_s,t
+ D_EID_i * rho * Gap_EI_i,t / (Gap_EI_i,t + tau)

D_EID_i = clip(1 - EID_i, 0.25, 1.0)
```

Candidate used:

- candidate id: `c0061`
- variant: `essential_input_dampener_only`
- score: `structural_dependence_plus_brown_lockin`
- `lambda_EID = 1.0`
- `d_min = 0.25`
- no historical residual

Separate diagnostic outputs were written under distinct filenames:

- `base_multiyear_state_panel_EID_diagnostic.parquet`
- `base_multiyear_summary_panel_EID_diagnostic.csv`
- `multiyear_EID_diagnostic_error_panel.parquet`
- `multiyear_EID_diagnostic_comparison.csv`
- `multiyear_EID_diagnostic_by_subtype.csv`
- `multiyear_EID_diagnostic_pseudo_agent_sensitivity.csv`
- `multiyear_EID_diagnostic_mechanism_audit.csv`
- `multiyear_EID_diagnostic_abm_v5_implications.csv`
- `multiyear_EID_diagnostic_recommendation.csv`
- `multiyear_EID_diagnostic_report.md`

Full historical validation result:

- The EID diagnostic mode was successfully integrated into the multi-year loop.
- EID coverage was available from Phase 23 scores, and fallback diagnostics are written in the state panel and mechanism audit.
- Compared with `historical_frontier_gap_only`, EID slightly worsened all-node unweighted rEI MAE: `0.14074` vs `0.14054`.
- It worsened emissions-weighted rEI MAE: `0.12425` vs `0.12311`.
- It slightly improved wrong-sign share: `0.38249` vs `0.38293`.
- It materially worsened latest-year aggregate emissions pct error: `1.20200` vs `0.85542`.
- It materially worsened mean yearly aggregate emissions pct error: `0.72721` vs `0.57123`.
- It worsened electricity aggregate emissions error: `239.6M` vs `191.7M`.
- It worsened China electricity emissions error: `207.2M` vs `163.7M`.

Subtype result:

- In the integrated multi-year loop, the EID dampener no longer reproduces the positive Phase 24 subtype-screen result.
- `infrastructure_energy`, `manufacturing_system_core`, and `construction_real_estate_foundational` are materially worsened.
- Physical subtype evidence is therefore not strong enough to promote the mechanism.

Pseudo-agent sensitivity:

- Pseudo-agent/accounting nodes are included in the sensitivity output.
- The Phase 25 rejection does not depend on pseudo-agent improvement; the main issue is material aggregate deterioration.

Mechanism audit:

- The mechanism audit confirms that EID reduces frontier-gap closure where scores are available.
- The problem is not that the dampener fails to apply; it applies, but recursive EI dynamics and emissions weighting turn the dampening into worse aggregate fit.

ABM v5 implication:

- The EID concept remains useful ontology evidence, especially for thinking about energy infrastructure, transport/logistics infrastructure, and heavy industry/materials.
- It is not validated as a behavioural agent type.
- No ABM v5 code or agent ontology was implemented.

Phase 25 recommendation:

- Recommendation: `reject_EID_for_v4`.
- Interpretation: the integrated diagnostic EID mode materially worsens aggregate emissions validation.
- Recommended Phase 26: do not promote EID to a base rule; keep scenarios blocked and move to explicit validation-objective selection or external policy/energy-system variables before further rule design.
- Scenarios remain premature.

## Phase 26 Adaptive EID Parameter Calibration Diagnostics

Phase 26 tested whether the Phase 25 EID failure came from the fixed strong dampener rather than from the EID concept itself. The phase used discrete walk-forward diagnostics only. It did not create scenarios, implement ABM v5, activate adaptive EID as a default, recalibrate production, overwrite `config.py`, or modify v1/v2/v3 source or protected outputs.

Adaptive EID dampener:

```text
D_EID = clip(1 - lambda_EID * EID_norm, d_min, 1.0)
```

The diagnostic grid tested:

- `lambda_EID`: 0.00, 0.25, 0.50, 0.75, 1.00.
- `d_min`: 0.25, 0.50, 0.75, 1.00.
- Primary walk-forward design: 5-year calibration, next 3-year validation.
- Robustness design: 3-year calibration, next 2-year validation.
- Objectives: transition accuracy, emissions-weighted transition accuracy, aggregate emissions fit, balanced policy objective, and electricity/high-EID objective.

The best adaptive row in the diagnostic comparison used the 5-year design with the balanced objective. It improved some aggregate-style diagnostic values in the walk-forward panel, but did not beat the historical frontier-gap baseline on the main transition-validation thresholds. Relative to `historical_frontier_gap_only`, the best adaptive result materially worsened all-node rEI MAE and emissions-weighted rEI MAE. The Phase 26 recommendation is therefore `reject_EID_for_v4_confirmed`.

Parameter stability was weak. Selected `lambda_EID` and `d_min` changed across windows, and many objectives often selected no-effect or weak-effect settings such as `lambda_EID = 0` or `d_min = 1`. This supports the interpretation that EID may be regime-dependent, or that it needs missing policy, investment, energy-system, and institutional counterforces that ABM v4 does not currently observe.

Hypothesis outcomes:

- H1 fixed EID too rigid: not supported as a rescue mechanism, because adaptive EID did not improve forward validation enough.
- H2 EID regime-dependent: supported by parameter instability.
- H3 fixed EID too strong: supported by frequent weak/no-effect selections.
- H4 subtype-specific EID: supported diagnostically, but not enough for a v4 rule.
- H5 EID diagnostic only: supported.
- H6 missing policy or energy counterforces: supported as an interpretation of unstable parameters.
- H7 adaptive calibration overfitting: supported where calibration-window gains did not translate into robust forward validation.

Phase 26 recommendation:

- Recommendation: `reject_EID_for_v4_confirmed`.
- Interpretation: adaptive parameter search does not rescue EID as an ABM v4 transition rule.
- Recommended Phase 27: keep scenarios blocked; either move to validation-objective selection among existing non-EID baselines, or document the need for external policy/energy-system variables before another EID-style mechanism is tested.
- ABM v5 implication: EID remains ontology evidence only, not a validated agent-type mechanism.
- Scenarios remain premature.

## Phase 27 Q Energy-Mix Audit and Transition-Error Diagnostics

Phase 27 opened one final ABM v4 diagnostic branch after EID was rejected for ABM v4. The motivation was to test a more direct mechanism for electricity and high-emissions sectors: country-sector energy-use mix from Eora Q rows. This phase did not implement scenarios, ABM v5, an energy-mix transition rule, or any active default change.

The audit uses the converted Eora Q matrices under `data/parquet/<year>/Q.parquet` and row labels from `labels_Q.txt` or `data/indices/index_q.csv` when available. The nine canonical energy-use rows were matched:

- Natural Gas
- Coal
- Petroleum
- Nuclear Electricity
- Hydroelectric Electricity
- Geothermal Electricity
- Wind Electricity
- Solar, Tide and Wave Electricity
- Biomass and Waste Electricity

The diagnostic constructs country-sector-year Q energy-use mix variables:

- fossil energy: natural gas + coal + petroleum
- clean electricity: nuclear + hydro + geothermal + wind + solar/tide/wave + biomass/waste
- renewable electricity: hydro + geothermal + wind + solar/tide/wave + biomass/waste
- shares, HHI, entropy, fossil-to-clean ratio, coal-to-clean ratio, and energy-per-output intensities

Interpretation guardrail: these are `Q energy-use mix` variables, not confirmed electricity generation mix variables.

Real-data audit result:

- Q energy source inventory was written.
- Row mapping was written.
- Energy mix panel rows: 108,152.
- Recommendation: `aggregate_only_energy_mix_usable`.
- Scenarios remain premature.

Hypothesis outcomes:

- H1 energy mix data usable: not fully supported at country-sector rule level because quality caveats remain.
- H2 energy mix explains errors better than EID: not supported strongly enough by the current univariate screen.
- H3 China electricity fuel-mix mechanism: supported; China electricity shows high fossil dependence in the Q mix.
- H4 physical-subtype-only usefulness: not supported strongly enough in this screen.
- H5 energy mix too sparse or noisy: supported as a country-sector modelling warning due to jump/plausibility caveats.
- H6 energy mix may resolve the frontier tradeoff: supported diagnostically by predictors related to the historical-vs-readiness error difference.

Phase 27 recommendation:

- Recommendation: `aggregate_only_energy_mix_usable`.
- Interpretation: Q energy mix is promising as context and aggregate validation evidence, but is not yet clean enough for a country-sector ABM v4 transition rule.
- Recommended Phase 28: inspect mapping and aggregate plausibility before any model-use step, or use Q energy mix as a validation stratifier rather than a rule.
- ABM v5 implication: fuel and energy-use mix remains an important candidate mechanism for a future data-design pass.
- Scenarios remain premature.

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

Build one-step emissions-intensity update and decomposition:

```powershell
python scripts/run_abm_v4_base.py --build-emissions-update --create-output-dirs
```

Compare default frontier-gap emissions transition with the legacy raw-log rule:

```powershell
python scripts/run_abm_v4_base.py --build-emissions-transition-comparison --create-output-dirs
```

Run the smoke-safe multi-year base simulation:

```powershell
python scripts/run_abm_v4_base.py --run-multiyear-base --start-year 2015 --end-year 2016 --create-output-dirs --reuse-existing
```

Run the full historical multi-year base simulation:

```powershell
python scripts/run_abm_v4_base.py --run-multiyear-base --start-year 1995 --end-year 2016 --create-output-dirs --reuse-existing
```

Validate the multi-year base run against historical observed values:

```powershell
python scripts/run_abm_v4_base.py --validate-multiyear-base --create-output-dirs
```

Build emissions-transition calibration diagnostics:

```powershell
python scripts/run_abm_v4_base.py --calibrate-emissions-transition --create-output-dirs --calibration-random-search-iterations 200
```

Diagnose why emissions-transition calibration remains weak:

```powershell
python scripts/run_abm_v4_base.py --diagnose-emissions-hypotheses --create-output-dirs
```

Compare Phase 14 emissions-transition rule variants:

```powershell
python scripts/run_abm_v4_base.py --compare-emissions-transition-variants --create-output-dirs --transition-variant-random-search-iterations 100
```

Run the Phase 15 calibrated-historical frontier-gap base:

```powershell
python scripts/run_abm_v4_base.py --run-multiyear-base --start-year 1995 --end-year 2016 --create-output-dirs --reuse-existing --emissions-transition-mode historical_frontier_gap_only --emissions-parameter-file data/abm_v4/validation/emissions_transition_variant_best_parameters.json
```

Compare available multi-year base model outputs:

```powershell
python scripts/run_abm_v4_base.py --compare-multiyear-base-models --create-output-dirs
```

Diagnose Phase 16 transition-rule tradeoffs:

```powershell
python scripts/run_abm_v4_base.py --diagnose-transition-rule-tradeoffs --create-output-dirs
```

Diagnose Phase 17 high-emissions and readiness-dampening tradeoffs:

```powershell
python scripts/run_abm_v4_base.py --diagnose-high-emissions-dampening --create-output-dirs
```

Audit Phase 18 electricity and China EI data:

```powershell
python scripts/run_abm_v4_base.py --audit-electricity-data --create-output-dirs
```

Audit Phase 19 raw Eora-derived electricity data path:

```powershell
python scripts/run_abm_v4_base.py --audit-raw-eora-electricity-data --create-output-dirs
```

Diagnose Phase 20 electricity-specific transition regime candidates:

```powershell
python scripts/run_abm_v4_base.py --diagnose-electricity-transition-regime --create-output-dirs
```

Diagnose Phase 21 structural signatures and transition-inertia proxy candidates:

```powershell
python scripts/run_abm_v4_base.py --diagnose-structural-signatures --create-output-dirs
```

Diagnose Phase 22 essential-input and IO structural-dependence signatures:

```powershell
python scripts/run_abm_v4_base.py --diagnose-essential-input-dependence --create-output-dirs
```

Test Phase 23 essential-input dampener candidates:

```powershell
python scripts/run_abm_v4_base.py --test-essential-input-dampener --create-output-dirs
```

Diagnose Phase 24 EID dampener failure modes and high-EID heterogeneity:

```powershell
python scripts/run_abm_v4_base.py --diagnose-eid-failure-modes --create-output-dirs
```

Run Phase 25 EID diagnostic multi-year integration audit:

```powershell
python scripts/run_abm_v4_base.py --run-multiyear-EID-diagnostic --create-output-dirs --reuse-existing
```

Diagnose Phase 26 adaptive EID calibration:

```powershell
python scripts/run_abm_v4_base.py --diagnose-adaptive-EID-calibration --create-output-dirs
```

Audit Phase 27 Q energy mix and transition-error diagnostics:

```powershell
python scripts/run_abm_v4_base.py --audit-q-energy-mix --create-output-dirs
```

Finalize Phase 28 ABM v4 consolidation and two-rule validation framework:

```powershell
python scripts/run_abm_v4_base.py --finalize-abm-v4 --create-output-dirs
```

## Phase 28 Final Consolidation

Phase 28 freezes ABM v4 as a historically validated diagnostic framework. It does not create scenarios, does not implement ABM v5 agent types, does not add a new transition rule, and does not overwrite `config.py`.

Phase 27 recap:

- All nine intended Q energy-use rows were found and mapped: natural gas, coal, petroleum, nuclear electricity, hydroelectricity, geothermal electricity, wind electricity, solar/tide/wave electricity, and biomass/waste electricity.
- Q energy mix supports the fuel-structure mechanism conceptually, especially for China electricity and high-emissions electricity-like nodes.
- Country-sector quality is not strong enough for ABM v4 behavioural rule integration because the audits report invalid shares, negative values, severe aggregate plausibility flags, and weak node-level predictive power.
- Final Q recommendation: `aggregate_only_energy_mix_usable`.

Final ABM v4 status:

- ABM v4 is a historical diagnostic framework for country-sector production-network transition validation.
- ABM v4 is not scenario-ready and should not be presented as a policy-counterfactual forecasting model.
- Historical production forcing remains central, so scenario use requires a future endogenous production layer.

Two-rule framework:

| Rule | Final role | Scenario status |
| --- | --- | --- |
| `frontier_gap_readiness` | Aggregate-safe historical baseline | `not_scenario_ready` |
| `historical_frontier_gap_only` | Transition-mechanism benchmark | `not_scenario_ready` |

Rejected or diagnostic-only mechanisms:

- `legacy_raw_log emissions rule`: rejected as the final rule; retained only as a baseline foil.
- `fixed EID dampener`: rejected for ABM v4 transition-rule use; retained as ontology evidence.
- `adaptive EID dampener`: rejected for ABM v4 transition-rule use; retained as ontology and overfitting evidence.
- `EID diagnostic multi-year mode`: diagnostic only.
- `Q energy mix country-sector transition rule`: rejected for ABM v4 node-level rule integration; retained for aggregate diagnostics and ABM v5 fuel-mechanism design.
- `historical residual as scenario-facing rule`: rejected because it leaks historical validation information.

Model boundary statement:

ABM v4 is useful for testing transition mechanisms under observed production forcing, identifying validation trade-offs, and locating missing mechanisms. It is not a scenario-ready forecasting model, not a policy-counterfactual simulator, not a fully agent-typed ABM, and not a complete energy-system transition model.

Scenario readiness status:

`not_scenario_ready`. Blocking issues include the two-rule validation-objective trade-off, historically forced production, missing fuel/policy mechanisms, insufficient node-level Q energy data quality, and unresolved endogenous supplier/capability/production dynamics.

ABM v5 research agenda:

- Energy/fuel structure using cleaner external generation, fuel-use, and capacity data.
- Policy/institutional regimes using renewable policy, investment, carbon-pricing, coal-phaseout, and subsidy variables.
- Capital-stock inertia using asset age, capacity, capital stock, and plant-level data.
- Explicit agent ontology separating ordinary production, energy infrastructure, heavy industry/materials, transport/logistics, systemic services, and accounting/non-agent nodes.
- Endogenous production dynamics without historical forcing.

Portfolio narrative pointer:

ABM v4 can be presented as a rigorous research pipeline that tested network frontier dynamics, exposed a trade-off between aggregate emissions fit and transition-mechanism fit, stress-tested EID and energy-mix explanations, and concluded honestly that scenario modelling requires explicit energy, policy, capital-stock, and endogenous production mechanisms.

Phase 28 outputs:

- `data/abm_v4/validation/final_abm_v4_input_availability.csv`
- `data/abm_v4/validation/final_surviving_rule_comparison.csv`
- `data/abm_v4/validation/final_validation_objective_matrix.csv`
- `data/abm_v4/validation/final_rejected_mechanism_register.csv`
- `data/abm_v4/validation/final_model_boundary_statement.md`
- `data/abm_v4/validation/final_scenario_readiness_assessment.csv`
- `data/abm_v4/validation/final_abm_v5_research_agenda.csv`
- `data/abm_v4/validation/final_abm_v4_hypothesis_status.csv`
- `data/abm_v4/validation/final_abm_v4_consolidation_report.md`
- `data/abm_v4/validation/final_abm_v4_portfolio_summary.md`

## Phase 29A Final Plots and Tables

Phase 29A builds final quantitative and visual artifacts from the Phase 28 validation outputs. It is not a modelling phase, does not create scenarios, does not implement ABM v5, and does not add a new transition rule.

Generated final clean tables:

- `data/abm_v4/final/tables/final_two_rule_summary.csv`
- `data/abm_v4/final/tables/final_mechanism_status.csv`
- `data/abm_v4/final/tables/final_scenario_blockers.csv`
- `data/abm_v4/final/tables/final_abm_v5_priorities.csv`
- `data/abm_v4/final/tables/final_portfolio_metrics.csv`
- `data/abm_v4/final/tables/final_report_table_index.csv`

Generated final plots:

- `data/abm_v4/final/plots/abm_v4_two_rule_tradeoff.png`
- `data/abm_v4/final/plots/abm_v4_two_rule_tradeoff.svg`
- `data/abm_v4/final/plots/abm_v4_validation_objective_matrix.png`
- `data/abm_v4/final/plots/abm_v4_validation_objective_matrix.svg`
- `data/abm_v4/final/plots/abm_v4_mechanism_funnel.png`
- `data/abm_v4/final/plots/abm_v4_mechanism_funnel.svg`
- `data/abm_v4/final/plots/abm_v4_scenario_readiness_blockers.png`
- `data/abm_v4/final/plots/abm_v4_scenario_readiness_blockers.svg`
- `data/abm_v4/final/plots/abm_v4_abm_v5_research_priorities.png`
- `data/abm_v4/final/plots/abm_v4_abm_v5_research_priorities.svg`
- `data/abm_v4/final/plots/abm_v4_portfolio_story_map.png`
- `data/abm_v4/final/plots/abm_v4_portfolio_story_map.svg`
- `data/abm_v4/final/plots/abm_v4_hypothesis_status.png`
- `data/abm_v4/final/plots/abm_v4_hypothesis_status.svg`

Portfolio-ready copies of the same plot files are written under:

```text
outputs/plots/abm_v4_final/
```

The final artifact index is:

```text
data/abm_v4/final/abm_v4_final_artifact_index.csv
```

The final LaTeX technical report and portfolio webpage are intentionally not generated by Codex in this phase. The next step is a human-written final LaTeX report and portfolio webpage that use the Phase 29A tables and plots as source artifacts.

## Output Root

```text
data/abm_v4/
```

## Remaining for v5 or Later Phases

- Replace or extend historically forced production with recursive production propagation.
- Validate the compact downstream capability proxy against raw-T downstream aggregation.
- Calibrate frontier-gap emissions parameters against historical rEI.
- Add policy scenarios only after the historical base loop is stable enough for the intended interpretation.
