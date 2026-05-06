# Remaster

Python research code for studying greener configurations in global production networks. The project connects Eora26 input-output matrices, emissions satellite accounts, network metrics, Atlas of Economic Complexity capability data, empirical transition modelling, and agent-based scenario simulation.

The codebase is intentionally research-oriented. The priority is not a polished package API; it is an inspectable pipeline where country-sector relationships, intermediate tables, warnings, and assumptions stay visible.

## Research Aim

This repository asks how green transitions emerge in a system of interconnected country-sector nodes.

The central object is a country-sector production node such as `FRA | Agriculture`. The project tracks each node's emissions intensity, direct green-ness, network-embedded green-ness, centrality, capability structure, transition behaviour, and simulated future trajectories.

Recent modelling uses a stricter definition of meaningful emissions improvement: `strict_green_upgrade` is equal to `1` only when a node's `delta_ei` is below the 25th percentile of the full transition dataset. This avoids treating weak or near-zero reductions as substantive green upgrades.

## Repository Layout

```text
Remaster/
|-- data/
|   |-- raw/                  # Raw Eora files and archives, kept local
|   |-- parquet/              # Labelled Eora matrices by year
|   |-- metrics/              # Yearly EI, ET, green-ness, centrality, efficiency outputs
|   |-- atlas/                # Atlas raw, concordance, and processed capability data
|   |-- final/                # Merged panels, transition dynamics, estimates
|   `-- abm/                  # ABM inputs, diagnostics, model outputs, scenarios
|-- notebooks/                # marimo notebooks only
|-- outputs/                  # Figures, tables, notes, and audit outputs
|-- src/
|   |-- data_manager/         # Eora download, extraction, inspection, parquet conversion
|   |-- EDA/                  # Older structured Eora EDA workflow
|   |-- metric_builder/       # EI, ET, network green-ness, centrality, efficiency metrics
|   |-- atlas_data/           # Atlas download, cleaning, concordance, sector aggregation
|   |-- modelling/            # Eora-Atlas merge, dynamics, precedence, estimates, clusters
|   |-- plotting/             # Static transition, trajectory, and phase-space plots
|   `-- abm/                  # ABM input preparation, diagnostics, models, simulations
|-- tests/                    # Current smoke tests
|-- pyproject.toml
`-- requirements.txt
```

Reusable logic belongs in `src/`. Notebooks should orchestrate and display results, not hide pipeline logic.

## Main Pipeline

The active workflow is built from explicit script entry points. Most commands use default paths, so the examples below show the usual local order.

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

`src/metric_builder/compute_metrics.py` computes the yearly analytical matrices and node metrics:

- direct emissions intensity, saved as `ei_<year>.parquet`
- embodied emissions transfers, saved as `et_<year>.parquet`
- base and network-embedded green-ness, saved as `greenness_<year>.parquet`
- graph centrality, PageRank, in/out strength, saved as `centrality_<year>.parquet`
- embodied emissions efficiency, saved as `efficiency_<year>.parquet`

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

`strict_green_upgrade` is computed after all `delta_ei` values exist:

```text
threshold = 25th percentile of delta_ei
strict_green_upgrade = 1 when delta_ei < threshold
```

The original `delta_ei` column is preserved.

### 6. Estimate Green Precedence

`src/modelling/green_precedence.py` estimates whether exposure to upstream sectors is associated with stricter green upgrade events.

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

In `ei` and `combined` modes, the EI component now uses `strict_green_upgrade` instead of the older `delta_ei < 0` rule.

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

## ABM Workflow

The ABM layer turns historical country-sector panels into agent states, empirical transition targets, and scenario simulations.

Prepare ABM inputs:

```powershell
python -m src.abm.prepare_abm_inputs
```

Outputs:

```text
data/abm/agents_panel.parquet
data/abm/transitions_panel.parquet
data/abm/edges_panel.parquet
```

Build transition diagnostics and cleaned targets:

```powershell
python -m src.abm.diagnose_transitions
```

Output:

```text
data/abm/diagnostics/transitions_with_clean_targets.parquet
```

Estimate ABM transition models:

```powershell
python -m src.abm.estimate_transition_model
python -m src.abm.estimate_regime_transitions
python -m src.abm.estimate_regime_transitions_balanced
```

Run a named scenario:

```powershell
python -m src.abm.scenario_runner --scenario baseline
python -m src.abm.scenario_runner --scenario network_diffusion
python -m src.abm.scenario_runner --scenario capability_policy
python -m src.abm.scenario_runner --scenario brown_core_intervention
python -m src.abm.scenario_runner --scenario combined_transition
```

Scenario outputs:

```text
data/abm/scenarios/<scenario>_simulation_panel.parquet
data/abm/scenarios/<scenario>_summary_panel.parquet
```

There is also an older standalone simulator:

```powershell
python -m src.abm.simulate_abm_v2
```

## Notebooks

This project uses marimo notebooks, not Jupyter notebooks.

Current notebooks:

```text
notebooks/EDA.py
notebooks/abm_scenario_explorer.py
notebooks/abm_country_sector_trajectories.py
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
| `strict_green_upgrade` | Indicator for `delta_ei` below the dataset's 25th percentile |
| Green precedence | Upstream-sector exposure associated with green upgrade events |
| Regime | ABM label combining green/brown status with core/periphery status |

## Setup

Use Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Some active modelling and ABM scripts import `scikit-learn`. If your environment does not already provide it, install it explicitly:

```powershell
pip install scikit-learn
```

## Testing

Run tests with:

```powershell
pytest
```

Current caveat: `tests/test_paths.py` and `src/config.py` still reference an older `src.paths` scaffold that is not present in the active source tree. Treat failures there as a repository maintenance issue rather than evidence about the analytical pipeline.

## Current State

This is an active research repository.

- Several modules use pandas even though the project preference is Polars.
- Some scripts are executable research entry points rather than stable package APIs.
- `src/EDA` and some generated `src_tree.json` metadata reflect older stages of the project.
- `requirements.txt` and `pyproject.toml` are close to the active environment, but the ABM and some plotting scripts also require `scikit-learn`.
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
