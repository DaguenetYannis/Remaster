# Remaster

Python research code for studying greener configurations in global production networks. The project connects Eora26 input-output matrices, emissions satellite accounts, network metrics, Atlas of Economic Complexity capability data, empirical transition modelling, and agent-based scenario simulation.

The codebase is research-oriented. The priority is an inspectable pipeline where country-sector relationships, intermediate tables, warnings, and assumptions stay visible.

## Research Aim

This repository asks how green transitions emerge in a system of interconnected country-sector nodes.

The central object is a country-sector production node such as `FRA | Agriculture`. The project tracks each node's emissions intensity, direct green-ness, network-embedded green-ness, centrality, capability structure, transition behaviour, and simulated future trajectories.

## Repository Layout

```text
Remaster/
|-- data/
|   |-- raw/                  # Raw Eora files and labels
|   |-- parquet/              # Labelled Eora matrices by year
|   |-- metrics/              # Yearly EI, ET, green-ness, centrality, efficiency outputs
|   |-- atlas/                # Atlas raw, concordance, and processed capability data
|   |-- final/                # Merged panels, transition dynamics, estimates
|   |-- abm/                  # Earlier ABM inputs, diagnostics, outputs, scenarios
|   |-- abm_v3/               # Current ABM v3, Leontief, diagnostics, validation outputs
|   |-- abm_v4/               # ABM v4 outputs, created only by explicit v4 runs
|   `-- indices/              # Local index-style data outputs
|-- notebooks/                # marimo notebooks only
|-- outputs/                  # Figures, tables, notes, and audit outputs
|-- logs/                     # Local logs
|-- tmp/                      # Disposable outputs
|-- src/
|   |-- data_manager/         # Eora download, extraction, inspection, parquet conversion
|   |-- metric_builder/       # EI, ET, network green-ness, centrality, efficiency metrics
|   |-- atlas_data/           # Atlas download, cleaning, concordance, sector aggregation
|   |-- modelling/            # Eora-Atlas merge, dynamics, precedence, estimates, clusters
|   |-- plotting/             # Static transition, trajectory, and phase-space plots
|   |-- abm_v1/               # Earlier ABM preparation, diagnostics, estimation, scenarios
|   |-- abm_v2/               # ABM v2 model, runner, metrics, plots, audits
|   |-- abm_v3/               # Current ABM v3, Leontief, validation, scenarios
|   `-- abm_v4/               # Phase 1 foundations for ecosystem-constrained green reorganization
|-- tests/                    # Project-level tests
|-- pyproject.toml
`-- requirements.txt
```

Reusable logic belongs in `src/`. Notebooks should orchestrate and display results, not hide pipeline logic.

## Path Conventions

- Project-level paths are defined in `src.paths`.
- ABM v3 paths are defined in `src.abm_v3.paths.ABMV3Paths`.
- ABM v4 paths are defined in `src.abm_v4.paths.ABMV4Paths`.
- Most scripts use project-relative paths such as `data/parquet`, `data/metrics`, `data/final`, `data/abm_v3`, and `outputs/plots`.
- Raw and generated data are local and are not expected to be complete in every checkout.

## Main Pipeline

The active workflow is built from explicit module entry points. Most commands use default paths.

### 1. Prepare Eora Data

`src/data_manager/` handles raw Eora files.

- `download.py` downloads yearly Eora26 zip files.
- `unzip_and_clean.py` extracts yearly files and archives processed zips.
- `inspection_file.py` inspects raw files before transformation.
- `parquet_and_labelling.py` converts raw matrices into labelled parquet files.

Typical labelled outputs:

```text
data/parquet/<year>/T.parquet
data/parquet/<year>/FD.parquet
data/parquet/<year>/Q.parquet
data/parquet/<year>/QY.parquet
data/parquet/<year>/VA.parquet
```

### 2. Build Eora Network Metrics

`src/metric_builder/compute_metrics.py` computes yearly analytical matrices and node metrics:

- direct emissions intensity: `ei_<year>.parquet`
- embodied emissions transfers: `et_<year>.parquet`
- base and network-embedded green-ness: `greenness_<year>.parquet`
- graph centrality, PageRank, in/out strength: `centrality_<year>.parquet`
- embodied emissions efficiency: `efficiency_<year>.parquet`

Example:

```powershell
python -m src.metric_builder.compute_metrics --years 2002 2007 2012 2017 --base-path data/parquet --label-base-path data/raw --output-path data/metrics
```

To recompute only green-ness from existing EI and ET outputs:

```powershell
python -m src.metric_builder.compute_metrics --years 2002 2007 2012 2017 --output-path data/metrics --only-greenness
```

### 3. Build Atlas Capability Data

`src/atlas_data/` prepares Atlas product-level capability data and aggregates it onto Eora26 sectors.

Common entry points:

```powershell
python -m src.atlas_data.download_atlas
python -m src.atlas_data.build_atlas_clean_panel
python -m src.atlas_data.build_concordance_prefill
python -m src.atlas_data.aggregate_atlas_to_eora_sector
```

Key processed output:

```text
data/atlas/processed/atlas_eora26_sector_capabilities_1995_2016.parquet
```

### 4. Merge Eora and Atlas Panels

`src/modelling/merge_eora_atlas.py` builds a yearly Eora metrics panel and left-joins Atlas sector capabilities by country, year, and sector.

```powershell
python -m src.modelling.merge_eora_atlas
```

Outputs:

```text
data/final/eora_metrics_panel.parquet
data/final/eora_atlas_merged.parquet
```

An additional dynamic panel with lags and changes can be built with:

```powershell
python -m src.modelling.build_dynamic_panel
```

Output:

```text
data/final/eora_atlas_dynamic_panel.parquet
```

### 5. Build Transition Dynamics

`src/modelling/transition_dynamics.py` builds state-to-state country-sector transitions from the merged panel. It adds current and next-year states, delta columns, network exposure variables, green capability readiness, and `strict_green_upgrade`.

```powershell
python -m src.modelling.transition_dynamics
```

Default output:

```text
data/final/transition_dynamics.parquet
```

### 6. Estimate Green Precedence

`src/modelling/green_precedence.py` estimates whether exposure to upstream sectors is associated with green upgrade events.

```powershell
python -m src.modelling.green_precedence
```

Default outputs:

```text
data/final/green_precedence/sector_green_precedence_scores.parquet
data/final/green_precedence/sector_green_precedence_scores.csv
data/final/green_precedence/node_year_green_precedence.parquet
data/final/green_precedence/node_year_green_precedence.csv
```

Event modes:

```powershell
python -m src.modelling.green_precedence --green-event-mode ei
python -m src.modelling.green_precedence --green-event-mode network
python -m src.modelling.green_precedence --green-event-mode capability
python -m src.modelling.green_precedence --green-event-mode combined
```

### 7. Estimate and Plot Transition Behaviour

Empirical summaries:

```powershell
python -m src.modelling.estimates
```

Default outputs are written under:

```text
data/final/estimates/
```

Trajectory clustering:

```powershell
python -m src.modelling.trajectory_clusters
```

Transition and trajectory plots:

```powershell
python -m src.plotting.plot_transition_behaviours
python -m src.plotting.plot_transition_surfaces
python -m src.plotting.plot_transition_vector_fields
python -m src.plotting.dynamic_plots
python -m src.plotting.trajectory_cluster_plots
```

Default plot outputs are written under:

```text
outputs/plots/
```

## ABM Workflows

The ABM layer turns historical country-sector panels into agent states, empirical transition targets, and scenario simulations.

### Earlier ABM Scripts

Earlier ABM scripts live under `src/abm_v1/` and `src/abm_v2/`.

Representative entry points:

```powershell
python -m src.abm_v1.prepare_abm_inputs
python -m src.abm_v1.diagnose_transitions
python -m src.abm_v1.estimate_transition_model
python -m src.abm_v1.scenario_runner --scenario baseline
python -m src.abm_v2.runner
```

Earlier ABM outputs are written under:

```text
data/abm/
```

### Current ABM v3 Workflow

The current structured workflow lives under `src/abm_v3/`. It includes input panel construction, dynamics, Leontief propagation, diagnostics, validation reports, and scenario phase-space analysis.

Primary CLI:

```powershell
python -m src.abm_v3.runner --help
```

Core path helper:

```text
src/abm_v3/paths.py
```

Current ABM v3 outputs are written under:

```text
data/abm_v3/
outputs/plots/abm_v3/
outputs/plots/scenario/
```

### ABM v4 Phase 1

ABM v4 is a new namespace for ecosystem-constrained green reorganization in the global production network. Phase 1 provides configuration, path, schema, and diagnostic foundations only; it does not rewrite or overwrite v1, v2, or v3.

Readiness entry point:

```powershell
python scripts/run_abm_v4_base.py
```

ABM v4 writes only under:

```text
data/abm_v4/
```

## Notebooks

This project uses marimo notebooks, not Jupyter notebooks.

Current notebooks:

```text
notebooks/EDA.py
notebooks/abm_country_sector_trajectories.py
notebooks/abm_scenario_explorer.py
notebooks/abm_transition_diagnostics.py
```

Open notebooks with:

```powershell
marimo edit notebooks
```

Notebook rules:

- keep cells reactive rather than sequential
- avoid hidden mutable state across cells
- call reusable functions from `src/`
- keep long transformations out of notebooks

## Core Concepts

| Concept | Meaning |
| --- | --- |
| Country-sector node | A country and Eora sector observed as a production unit |
| `T` | Eora intermediate transaction matrix |
| `FD` | Final demand matrix |
| `Q`, `QY` | Environmental satellite accounts |
| `EI` | Direct emissions intensity |
| `ET` | Embodied emissions transfer matrix |
| `g_base` | Direct green-ness derived from emissions intensity |
| `g_out_network`, `g_in_network` | Network-embedded green-ness measures |
| Atlas capability | Product/export capability aggregated to Eora26 sectors |
| `delta_ei` | Next-year minus current emissions intensity |
| `strict_green_upgrade` | Indicator for a meaningful emissions-intensity improvement |
| Green precedence | Upstream-sector exposure associated with green upgrade events |
| Regime | ABM label combining green/brown status with core/periphery status |

## Setup

Use Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Testing

Run tests with:

```powershell
pytest
```

## Current State

This is an active research repository.

- Several modules use pandas even though the project preference is Polars for new transformation code.
- Some scripts are executable research entry points rather than stable package APIs.
- `src/EDA` and generated metadata such as `src/src_tree.json` reflect older stages of the project.
- Raw data, parquet matrices, model outputs, and figures are intentionally local and not tracked by git.

## Working Rules

- Inspect raw Eora structures before transforming them.
- Preserve country-sector interpretability.
- Log shapes, columns, row counts, missingness, and output locations.
- Do not silently guess missing schema details.
- Keep marimo notebooks reactive and thin.
- Move reusable logic into `src/`.
- Prefer explicit, readable transformations over compact cleverness.

## Data Policy

Raw Eora files, Atlas extracts, parquet matrices, ABM outputs, and generated figures can be large or restricted. Keep them local under `data/`, `outputs/`, `logs/`, or `tmp/`. Commit source code, lightweight schemas, configuration, and documentation only.
