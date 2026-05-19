# ABM_v5 Phase 4.1 Replay Scaffold

## Purpose

Phase 4.1 starts the generative ABM architecture without implementing behavioural mechanisms. It creates the runtime mechanism registry, ablation configuration, replay metadata, and a historical replay scaffold over the observed 1995-2016 state.

## Why Start With A Scaffold

ABM_v5 follows the sequence:

Discover -> Validate -> Encode -> Simulate

Phases 1-3 created ontology, observed historical panels, descriptive regimes, descriptive transition states, and a mechanism-target handoff. Phase 4.1 creates the technical surface where later mechanisms can be inserted one at a time while keeping the baseline identity replay inspectable.

## Replay Scaffold Versus Simulation

The Phase 4.1 replay is not a simulation. It does not update state variables through behavioural rules, use next-year information, draw shocks, optimize decisions, or run scenarios.

The scaffold-only replay copies observed variables into `simulated_*` columns:

- `output`
- `emissions`
- `emissions_intensity`
- `local_greenness`
- `network_green_exposure`
- `green_capability`
- `general_capability`
- `supplier_lock_in`
- `brown_centrality`

With only identity replay active, these simulated columns must equal the observed variables wherever both are non-null.

## Mechanism Registry

The mechanism registry is a runtime version of the Phase 1 mechanism ontology. Each core mechanism is assigned to a future Phase 4 implementation step, but all behavioural mechanisms remain `not_implemented` in Phase 4.1.

The registry is metadata only. It does not contain equations, parameters, calibration, stochasticity, or decision rules.

## Ablation Configuration

The Phase 4.1 ablation configuration enables only `identity_replay`. All actual mechanism families are disabled:

- emissions intensity
- network exposure
- supplier lock-in
- production feasibility
- capability accumulation
- brown centrality
- policy regime

Policy regime behaviour remains forbidden because scenario logic has not been introduced.

## Future Phase 4 Steps

Future subphases may implement mechanisms in this order:

- Phase 4.2 emissions-intensity transition
- Phase 4.3 network exposure feedback
- Phase 4.4 supplier lock-in and bounded supplier adaptation
- Phase 4.5 production feasibility and energy/inertia constraints
- Phase 4.6 capability accumulation and directed ecosystem movement
- Phase 4.7 historical replay validation
- Phase 4.8 ablation tests

Each mechanism should be added explicitly and validated against the scaffold baseline.

## Guardrails

Phase 4.1 does not implement behavioural mechanisms, scenarios, policy shocks, stochastic dynamics, calibration, optimisation, plotting, or causal claims. It only establishes a no-op historical replay contract that preserves the observed state.
