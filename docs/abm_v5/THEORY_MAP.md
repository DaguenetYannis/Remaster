# ABM v5 Theory Map

## Purpose

This document records the theoretical commitments behind ABM v5 so future
implementation prompts do not redesign the model. ABM v5 is a generative
complexity model of ecological transition as adaptive transformation inside
country-sector productive ecosystems.

## Theoretical Pillars

The model links complexity economics, economic complexity, directed productive
ecosystems, supply-chain ABM logic, input-output discipline, and the Remaster
green-ness framework. These pillars constrain implementation choices.

## Arthur: Adaptive Feedback And Non-Equilibrium

Economies are adaptive systems. Agents react to aggregate patterns that their
own actions help create. ABM v5 therefore treats feedback variables as recursive
objects inside later yearly update loops, not as one-time diagnostics.

## Hidalgo: Capability-Constrained Transformation

Productive transition is constrained by existing capabilities and proximity in
capability space. Green transition cannot be modeled as arbitrary emissions
reduction; it must be linked to capability accumulation and reachable green
complexity.

## Arrow Of Development: Directed Productive Ecosystems

Development paths are directional because some activities systematically precede
others through shared capability ecosystems. ABM v5 therefore separates
undirected ecosystem proximity from directed green precedence and
transition-sector positioning.

## Inoue: Supply-Chain Propagation And Bounded Substitution

Supplier-customer networks propagate shocks and constraints. ABM v5 must model
production-network propagation, supplier substitution difficulty, bounded search,
and recursive production feasibility.

## Chudziak: Recursive Complexity Expansion And Mechanism-First Validation

ABMs should not rely mainly on moment matching. They should uncover mechanisms,
feedback loops, impact channels, and the consequences of increasing structural
complexity. ABM v5 must therefore follow a complexity ladder and use
mechanism-level validation and ablation tests.

## Gallegati/Landini/Gallegati: ABM As A Tool For Non-Equilibrium Complexity

ABM is used here to study processes driving non-equilibrium dynamics. ABM v5
should be presented as a generative process model, not a static classifier or
equilibrium simulator.

## Input-Output And Leontief Discipline

Input-output economics disciplines production interdependence, input
requirements, and accounting consistency. ABM v5 must preserve the distinction
between `T` as production structure and `ET` as embodied-carbon exposure.

## Remaster Green-Ness Logic

Green-ness is relational. A country-sector node's environmental position depends
on local emissions intensity and network-embedded carbon exposure. ABM v5 must
keep local green-ness, network green exposure, and embodied-carbon diagnostics
conceptually distinct.

## Implementation Consequences For ABM_v5

ABM v5 must be built through the contract-defined complexity ladder: accounting,
production-network, capability, supplier-adaptation, energy and inertia,
policy-regime, regime-discovery, and scenario layers. Mechanisms should map
explicit inputs to update targets, carry theory sources, and be validated by
layered checks before scenario interpretation.

## Non-Goals For Phase 1

Phase 1 does not implement empirical data loading, role derivation, production
feasibility equations, supplier choice, emissions transition equations, regime
discovery, scenarios, or plotting. The theory map is metadata and guidance only.
