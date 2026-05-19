# Phase 3.4 Transition Encoding

## Purpose

Phase 3.4 encodes historical year-to-year transition states from the observed
regime panel. It is the Encode step in:

Discover -> Validate -> Encode -> Simulate

Regime discovery describes observed country-sector positions. Transition
encoding describes observed movement from one year to the next. Neither step is
simulation or scenario modelling.

## Descriptive Labels

Transition states are historical descriptive labels:

- `no_next_year`
- `insufficient_data_transition`
- `stable_same_regime`
- `regime_switch`
- `green_embedding_improvement`
- `green_embedding_deterioration`
- `capability_gain`
- `capability_loss`
- `supplier_lock_in_increase`
- `supplier_lock_in_relief`
- `dirty_gap_improvement`
- `dirty_gap_worsening`
- `mixed_movement`

The labels summarize observed deltas in phase-space variables. They are not
behavioural mechanisms, decision rules, transition probabilities, or causal
claims.

## Rule Precedence

Regime switches are dominant descriptive labels. If a country-sector changes
regime between year `t` and `t+1`, the transition is encoded as
`regime_switch` before interpreting individual variable movements. This avoids
claiming why the switch occurred.

## Boundaries

This phase does not create `regime_probability`, scenario IDs, simulation run
IDs, transition matrices for simulation, causal mechanisms, or behavioural
rules. Phase 4 may later decide which encoded historical transitions become
targets for mechanism implementation.
