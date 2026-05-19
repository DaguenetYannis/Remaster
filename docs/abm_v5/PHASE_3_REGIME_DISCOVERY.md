# Phase 3 Regime Discovery

## Purpose

Phase 3.1 to 3.3 discovers and validates descriptive historical regimes from
the observed ABM_v5 phase-space panel. It follows the sequence:

Discover -> Validate -> Encode -> Simulate

This document covers only Discover and Validate.

## Method

The first implementation uses interpretable global quantile rules. This keeps
the regime definitions inspectable and tied to explicit phase-space tensions:
emissions-intensity position, green capability, network green exposure, brown
centrality, and supplier lock-in.

Clustering is deferred because black-box group discovery would make the first
regime layer harder to audit. It may be added later as a robustness diagnostic,
not as the initial authority.

## Eligible Variables

Eligible empirical variables include `emissions_intensity_gap`,
`green_capability`, `general_capability`, `network_green_exposure`,
`brown_centrality`, `supplier_lock_in`, `local_greenness`,
`import_dependence_proxy`, `export_dependence_proxy`,
`supplier_concentration_hhi`, and `buyer_concentration_hhi`.

Design-target placeholders are excluded: `capability_density`,
`green_capability_density`, `ecosystem_proximity`,
`directed_green_precedence`, `reachable_green_complexity`, and
`transition_sector_score`.

## Labels

Regime labels are descriptive empirical states:

- `green_capable_embedded`
- `green_capable_constrained`
- `brown_central_capable`
- `brown_central_constrained`
- `dirty_capability_gap`
- `clean_low_capability`
- `low_signal_peripheral`
- `mixed_intermediate`
- `insufficient_data`

They are not behavioural mechanisms, transition states, probabilities, or
scenario outcomes.

## Boundaries

Transition-state encoding comes later. Simulation and scenarios remain
forbidden in this phase. This phase does not create transition matrices,
regime-switch flags, behavioural rules, or causal mechanism claims.
