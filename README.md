# Remaster

Research codebase for studying greener configurations in the global production network using Eora26 input-output data, Atlas of Economic Complexity capability data, and network-based metrics.

The project is intentionally Python-only and research-oriented: the priority is to keep every transformation inspectable, every assumption visible, and every output traceable back to the raw data.

## Research Goal

This repository supports a pipeline from raw Eora26 matrices to analytical outputs about:

- country-sector production nodes
- input-output linkages and embodied emissions
- direct and network-embedded green-ness
- centrality, robustness, and dynamic change in production networks
- Atlas capability indicators mapped onto Eora26 sectors

The core analytical idea is that green-ness is not only a property of an isolated sector. It is also a network property: a country-sector node can be greener or dirtier depending on the production relationships in which it is embedded.

## Repository Layout

```text
Remaster/
|-- data/
|   |-- raw/                 # Original Eora files and archived source zips
|   |-- parquet/             # Labelled Eora matrices converted to parquet
|   |-- atlas/               # Atlas raw, concordance, and processed panels
|   |-- eora/                # Processed Eora metrics
|   |-- final/               # Final merged analysis tables
|   |-- data_schema.json     # Inspected Eora matrix shapes and label examples
|   |-- workflow_schema.json # Explicit EDA workflow contract
|   `-- eda_config.json      # EDA runtime years and mode
|-- notebooks/               # marimo notebooks only
|-- outputs/                 # Figures, tables, notes, and audit reports
|-- src/
|   |-- data_manager/        # Eora download, unzip, inspection, parquet conversion
|   |-- EDA/                 # Main Eora EDA pipeline and visual summaries
|   |-- metric_builder/      # EI, ET, green-ness, centrality, efficiency metrics
|   |-- atlas_data/          # Atlas download, cleaning, concordance, aggregation
|   |-- modelling/           # Final Eora-Atlas merge logic
|   `-- plotting/            # Plot builders for metric outputs
|-- tests/                   # Current smoke tests
|-- pyproject.toml
`-- requirements.txt
```

Reusable logic should live in `src/`. Notebooks should orchestrate, display, and explain results, not contain core pipeline logic.

## Main Pipeline

The project currently has four connected workstreams.

### 1. Eora Data Preparation

Scripts in `src/data_manager/` manage raw Eora files:

- `download.py` downloads yearly Eora26 zip files.
- `unzip_and_clean.py` extracts yearly files and archives processed zips.
- `inspection_file.py` inspects raw files before transformation.
- `parquet_and_labelling.py` reads raw matrices, applies Eora label files, and writes labelled parquet matrices.

Typical output:

```text
data/parquet/<year>/T.parquet
data/parquet/<year>/FD.parquet
data/parquet/<year>/Q.parquet
data/parquet/<year>/QY.parquet
data/parquet/<year>/VA.parquet
```

### 2. Eora Metrics

`src/metric_builder/` computes:

- emissions intensity (`EI`)
- embodied emissions transfers (`ET`)
- direct and network-embedded green-ness
- directed network centrality metrics
- in/out embodied emissions and efficiency indicators

Example:

```powershell
python -m src.metric_builder.compute_metrics --years 2002 2007 2012 2017 --base-path data/parquet --label-base-path data/raw --output-path data/metrics
```

To recompute only green-ness from existing EI and ET files:

```powershell
python -m src.metric_builder.compute_metrics --years 2002 2007 2012 2017 --output-path data/metrics --only-greenness
```

### 3. Eora EDA Pipeline

`src/EDA/run_eda.py` runs the structured EDA workflow declared in `data/workflow_schema.json` and configured by `data/eda_config.json`.

Default mode computes direct, green, centrality, dynamic, and classification metrics. Extended mode also computes embodied metrics using sparse linear solves.

```powershell
python -m src.EDA.run_eda
python -m src.EDA.run_eda --mode extended
```

Outputs are written under:

```text
outputs/eda/
|-- tables/
|-- plots/
|-- notes/
`-- audit/
```

### 4. Atlas to Eora Integration

`src/atlas_data/` builds a country-sector-year Atlas capability panel compatible with Eora26 sectors:

- `download_atlas.py` downloads Atlas GraphQL data.
- `build_atlas_clean_panel.py` builds a clean HS92 product panel.
- `build_concordance_prefill.py` pre-fills HS92 to Eora26 sector mappings.
- `aggregate_atlas_to_eora_sector.py` aggregates product capabilities to Eora26 sectors.

Then `src/modelling/merge_eora_atlas.py` merges processed Eora metrics with Atlas sector capabilities:

```powershell
python -m src.modelling.merge_eora_atlas
```

Expected output:

```text
data/final/eora_atlas_merged.parquet
```

## Core Concepts

| Concept | Meaning |
| --- | --- |
| Country-sector node | A production unit such as `FRA | Industries | Agriculture` |
| `T` | Intermediate input-output transaction matrix |
| `FD` | Final demand matrix |
| `Q` | Environmental satellite matrix |
| `EI` | Direct emissions intensity |
| `ET` | Embodied emissions transfer matrix |
| Base green-ness | Direct green-ness derived from emissions intensity |
| Network green-ness | Green-ness induced by upstream/downstream production relations |
| Atlas capability | Product-level export capability aggregated to Eora26 sectors |

## Setup

Use Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Editable install is declared in `pyproject.toml`, but the package metadata is currently behind the active source layout. Prefer `requirements.txt` until packaging is refreshed.

## Current State

This is an active research repository, not a finished library.

- The active source tree has moved beyond the original scaffold.
- Some modules still use pandas even though the project preference is Polars.
- `pyproject.toml` and `tests/test_paths.py` still reference the older `src.paths` scaffold, which is no longer present in the working tree.
- Several scripts are executable research entry points rather than polished package APIs.
- Raw data and generated outputs are intentionally not tracked by git.

These points should be treated as migration notes, not analytical conclusions.

## Working Rules

- Inspect raw Eora structures before transforming them.
- Preserve country-sector interpretability.
- Log shapes, columns, row counts, missingness, and output locations.
- Do not silently guess missing schema details.
- Keep marimo notebooks reactive and thin.
- Move reusable logic into `src/`.
- Prefer explicit, readable transformations over compact cleverness.

## Development Notes

Run tests with:

```powershell
pytest
```

At the moment, tests may fail until the old path scaffold is either restored or the tests are updated to match the current source layout.

Run marimo notebooks with:

```powershell
marimo edit notebooks
```

## Data Policy

Raw Eora files, Atlas extracts, parquet matrices, and generated figures can be large and/or restricted. Keep them local under `data/` and `outputs/`. Commit code, schemas, lightweight configuration, and documentation only.
