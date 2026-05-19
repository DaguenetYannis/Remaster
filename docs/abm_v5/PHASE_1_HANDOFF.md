# ABM v5 Phase 1 Handoff

## What Phase 1 Implemented

Phase 1 implemented the ABM v5 namespace, path registry, configuration objects,
controlled vocabularies, ontology metadata, mechanism metadata, schema
contracts, validation principles, metadata validators, and theory mappings.

## Files Created

- `src/abm_v5/__init__.py`
- `src/abm_v5/paths.py`
- `src/abm_v5/config.py`
- `src/abm_v5/ontology.py`
- `src/abm_v5/schema.py`
- `src/abm_v5/validation.py`
- `src/abm_v5/theory.py`
- `tests/abm_v5/`
- `docs/abm_v5/README_PHASE_1.md`
- `docs/abm_v5/THEORY_MAP.md`
- `docs/abm_v5/METHODOLOGICAL_POSITION.md`
- `docs/abm_v5/PHASE_1_HANDOFF.md`

## What Phase 1 Deliberately Did Not Implement

No empirical data loading has been implemented. No simulation logic has been implemented. No scenario logic has been implemented. No regime discovery,
supplier choice, role derivation, production feasibility calculation, emissions
transition equation, plotting, or report generation has been implemented.

Ontology, schemas, validation principles, and theory mapping are metadata contracts only.

## How To Run Phase 1 Tests

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\abm_v5
```

## What Phase 2 Should Consume

Phase 2 should consume the path registry, config objects, ontology registry,
schema registry, validation principles, metadata validators, and theory mappings.
It should not duplicate or redesign the country-sector ontology, functional
roles, mechanism list, validation hierarchy, or complexity ladder.

## Guardrails For Future Codex Prompts

Future prompts should keep ABM v5 isolated from ABM v1, ABM v2, ABM v3, and ABM
v4 unless explicitly instructed. They should not introduce fixed agent classes,
hidden notebook logic, scenario execution, or empirical assumptions before the
contract asks for them.
