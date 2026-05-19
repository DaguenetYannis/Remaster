# ABM v5 Methodological Position

## ABM_v5 As A Generative Complexity Model

ABM v5 models ecological transition as a complex adaptive transformation of
productive ecosystems. The acting unit is the country-sector node, interpreted
as a meso-level productive position embedded in global input-output structure.

ABM_v5 should explain how ecological transition can emerge from explicit country-sector mechanisms, not merely reproduce green-transition indicators.

## Why Historical Fit Is Necessary But Insufficient

Historical fit matters, but it is not enough. A model that tracks aggregate
time series while violating accounting consistency, network structure, or
mechanism plausibility should be rejected.

## Validation Hierarchy

Validation is layered: accounting validity, structural validity, mechanism
validity, historical plausibility, ablation validity, and scenario credibility.
Scenario credibility comes last because counterfactual interpretation depends on
the previous validation layers.

## Complexity Ladder

Implementation must proceed through the complexity ladder: accounting,
production-network, capability, supplier-adaptation, energy and inertia,
policy-regime, regime-discovery, and scenario layers. Later layers should not
silently rewrite earlier layers.

## Mechanism Ontology

Mechanisms are explicit process objects linking inputs, update targets, theory
sources, feedback status, complexity levels, and validation layers. They are not
implicit transformations hidden in notebooks.

## Ablation And Model Comparison

Added mechanisms should improve interpretation, diagnostics, or validation
relative to simpler variants. Complexity is justified by explanatory value, not
by decorative detail.

## Scenario Restraint

Scenario logic should remain inactive until foundation, ontology, schema,
validation, and historical construction requirements are satisfied. Scenario
outputs should not be interpreted as credible before the prior validation layers
pass.

## What This Means For Implementation

Future implementation should consume Phase 1 metadata rather than inventing new
model categories. Code should keep country-sector identity, state variables,
functional roles, mechanisms, schemas, validation layers, and theory mappings
explicit and inspectable.
