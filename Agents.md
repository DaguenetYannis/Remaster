# AGENTS.md

## Project Purpose

This repository implements a Python-only research pipeline to study:
- global production networks
- Eora26 input-output structures
- emissions intensity, network green-ness, and transition dynamics
- agent-based and Leontief-style scenario experiments

The objective is to keep the full path from raw Eora and Atlas data to analytical outputs transparent, reproducible, and inspectable.

## Core Constraints

- Python only.
- Prefer Polars over pandas for new tabular transformation code unless the active module already depends on pandas.
- Use `pathlib.Path` for file paths.
- Use type hints and docstrings consistently.
- Prefer explicit, readable, inspectable code.
- Avoid hidden logic and implicit behavior.
- Surface uncertainty instead of guessing.

## Current Repository Structure

- `src/data_manager/` -> Eora download, extraction, raw inspection, parquet conversion.
- `src/metric_builder/` -> EI, ET, green-ness, centrality, and efficiency metrics.
- `src/atlas_data/` -> Atlas download, cleaning, concordance, and Eora-sector aggregation.
- `src/modelling/` -> Eora-Atlas merges, dynamic panels, transition modelling, estimates, clusters.
- `src/plotting/` -> transition, trajectory, phase-space, and scenario plots.
- `src/abm_v1/` -> earlier ABM preparation, diagnostics, estimation, and scenario scripts.
- `src/abm_v2/` -> ABM v2 model, runner, metrics, plots, and audit helpers.
- `src/abm_v3/` -> current structured ABM v3, Leontief, diagnostics, validation, and scenario workflow.
- `notebooks/` -> marimo notebooks only.
- `tests/` and `src/abm_v3/tests/` -> test coverage.
- `data/raw/` -> raw Eora files and labels.
- `data/parquet/` -> labelled Eora matrices by year.
- `data/metrics/` -> yearly derived metric outputs.
- `data/atlas/` -> Atlas raw, concordance, and processed data.
- `data/final/` -> merged panels, transition dynamics, estimates.
- `data/abm/` -> earlier ABM outputs.
- `data/abm_v3/` -> current ABM v3 and Leontief outputs.
- `outputs/` -> generated figures, tables, and audit outputs.
- `logs/` -> local logs.
- `tmp/` -> disposable test and development outputs.

Reusable logic belongs in `src/`. Notebooks should orchestrate computation, display outputs, and call reusable functions from `src/`.

## Path Conventions

- Shared project-level paths live in `src.paths`.
- ABM v3 path construction lives in `src.abm_v3.paths.ABMV3Paths`.
- Use relative project paths such as `data/parquet`, `data/metrics`, and `outputs/plots` in scripts and documentation.
- Do not hard-code local absolute paths.
- Do not assume that generated data exists. Check paths, log missing inputs, and continue where possible.

## Data Principles

- Raw Eora data must be inspected before transformation.
- Raw datasets must not be assumed to match any canonical structure without checks.
- Transformations must be explicit and traceable.
- If a dataset structure is unclear, log a warning and continue without guessing silently.
- Preserve interpretability of country-sector relationships.
- The system is defined at the level of interconnected country-sector nodes.

## Coding Principles

- Separate IO logic, transformation logic, modelling logic, and plotting logic.
- Prefer clarity over abstraction.
- Avoid overengineering.
- Avoid hidden behavior and implicit transformations.
- Write modular, reusable code.
- Avoid duplication across files.
- Keep helper functions focused and single-purpose.
- Larger orchestration functions are acceptable when they reflect the full research workflow.
- Reusable mechanisms should be extracted into clearly named helpers.

## Function and Class Design

- OOP is acceptable when it improves clarity.
- Classes should have a clear role and simple state.
- Methods should have explicit inputs, explicit outputs, and clear side effects.
- Avoid abstraction layers that obscure behavior.

## Marimo Notebook Rules

This project uses marimo, not Jupyter.

- Notebooks must be reactive, not sequential scripts.
- Do not rely on execution order.
- Do not hide state across cells.
- Do not mutate objects defined in other cells.
- Keep reusable logic out of notebooks.
- Call functions defined in `src/`.
- Use notebooks to orchestrate computation, inspect results, and display outputs.

## Error Handling and Logging

- Prefer warning-and-continue behavior for research pipelines.
- If a step fails, log the issue clearly and continue processing remaining data where possible.
- Logging should make visible what went in, what changed, and what came out.
- Prefer inline console visibility over hidden log files.
- Include visibility checks such as dataframe shapes, column changes, row counts, missingness, and output locations.
- Silent failure is forbidden.

## Research Coding Style

- Prefer readable code over compact code.
- Verbosity is acceptable when it improves understanding.
- Prefer simple logic over clever logic.
- Keep transformations inspectable.
- Use specific variable names.
- Clearly distinguish dataframes, matrices, paths, and domain objects.
- Add short, literal comments for non-trivial logic.

## What Codex Must Not Do

- Do not assume the structure of Eora datasets without inspection.
- Do not invent or approximate missing data structures.
- Do not embed core logic inside notebooks.
- Do not introduce unnecessary abstractions.
- Do not introduce hidden dependencies between modules.
- Do not write code that relies on implicit state or execution order.
- Do not prioritize cleverness over clarity.
- Do not delete files under `src/`, `data/`, or `outputs/` unless the user explicitly asks.

## Definition of Done

A task is complete when:
- code and documentation match the live repository structure
- transformations are explicit and validated
- outputs are inspectable and interpretable
- assumptions are clearly stated
- no known incorrect reference information remains
