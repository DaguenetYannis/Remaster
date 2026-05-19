# Phase 3.5 Regime Handoff

## Purpose

Phase 3.5 prepares descriptive regime and transition outputs for Phase 4
mechanism work. It does not implement mechanisms. It creates handoff artefacts
that identify which observed regimes and transitions are frequent, stable, and
data-complete enough to become candidate mechanism-learning targets or
diagnostic constraints.

## Why Handoff Comes Before Phase 4

Phase 3 discovered descriptive regimes and encoded descriptive historical
transition states. Phase 4 will later decide which movements should be
implemented as mechanisms. The handoff layer prevents that decision from being
implicit: it separates usable historical samples from rare, noisy, or
data-limited patterns.

## Eligibility

Rows are eligible for mechanism learning only when they have a next year,
non-insufficient current and next regimes, non-insufficient transition states,
transition confidence at least 0.5, and valid emissions-intensity accounting
where that flag is available.

Rows can also be marked usable for specific target families: emissions
intensity, network embedding, capability, supplier lock-in, and brown
centrality.

## Targets And Diagnostics

Frequent, eligible transition states can become Phase 4 target candidates.
Rare events remain diagnostic. `no_next_year` and
`insufficient_data_transition` are data-limited boundary conditions, not
mechanism-learning targets.

## Boundaries

Phase 3.5 does not simulate, run scenarios, infer causal mechanisms, rediscover
regimes, re-encode transition states, create `regime_probability`, or implement
behavioural rules.
