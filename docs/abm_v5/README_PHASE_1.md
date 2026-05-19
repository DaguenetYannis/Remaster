# ABM v5 Phase 1.1 Implementation Note

Phase 1.1 creates only the ABM v5 namespace and centralized path registry.
The implementation introduces `src.abm_v5` and `ABMV5Paths` so future
subphases have one explicit place for expected ABM v5 directories.

No empirical data loading, model dynamics, regime discovery, scenario logic, or
plotting are implemented in this phase.

Future subphases will add configuration, ontology, schema contracts, validation
principles, and theory mapping according to the ABM v5 contract.

ABM v5 must remain isolated from ABM v1, ABM v2, ABM v3, and ABM v4 unless a
future phase explicitly states otherwise.

## Phase 1.2 Note

Phase 1.2 adds configuration objects and controlled vocabulary enums for the
complexity ladder, validation layers, source status, feedback status, and model
stage. These objects define concepts for later ontology, schema, validation, and
implementation phases; they do not execute model mechanisms.

The scenario layer remains blocked during the foundation stage, and ontology
configuration keeps fixed agent classes disabled.

## Phase 1.3 Note

Phase 1.3 defines the country-sector ontology and mechanism ontology. Functional
roles are derived diagnostic labels, not fixed agent classes. Mechanisms are
metadata objects only at this stage, with explicit input variables, update
targets, theory sources, feedback status, complexity level, and validation
targets.

No empirical loading, simulation, or scenario execution is implemented. The
ontology registry is designed to make later Codex implementation less ambiguous.

## Phase 1.4 Note

Phase 1.4 adds metadata-only schema contracts for ABM v5 tables. These contracts
define expected columns, semantic metadata, keys, dtypes, nullability, source
status, feedback status, complexity level, validation layers, and expected
ranges.

The schema registry does not load real data and does not validate DataFrames.
It exists to keep later empirical construction, validation, simulation outputs,
and reports aligned with the ontology-first and mechanism-first design.

## Phase 1.5 Note

Phase 1.5 adds validation principles and lightweight metadata validators for the
ontology and schema registries. Validation remains layered: accounting validity,
structural validity, mechanism validity, historical plausibility, ablation
validity, and scenario credibility.

These validators do not load real data and do not validate pandas or Polars
DataFrames. They check only metadata consistency, including ontology references,
schema keys, validation-layer coverage, and theory-to-schema alignment.

Historical fit is treated as necessary but not sufficient: later model outputs
must also pass accounting, structural, mechanism, ablation, and scenario
credibility checks before scenario interpretation.

## Phase 1.6 Note

Phase 1.6 adds theory mapping and methodological position documentation. The
metadata in `src.abm_v5.theory` links theoretical pillars to ABM v5 mechanisms
and complexity layers, while the documentation records the model's position as a
generative, ontology-first, mechanism-first, validation-layered, and
scenario-restrained complexity model.

This phase does not implement empirical loading, model dynamics, regime
discovery, scenarios, or plotting.
