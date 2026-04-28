# AGENTS.md

## 🧭 Project Purpose

This repository implements a Python-only research pipeline to study:
- global production networks  
- input-output structures (Eora26)  
- emergence of greener configurations in trade networks  

The objective is to:
- build a transparent pipeline from raw Eora data to analytical outputs  
- define and measure “green-ness” as a system-level, network-based property  
- enable reproducible and inspectable economic analysis  

---

## ⚙️ Core Constraints

- Python only (no external IO analysis tools)  
- Prefer Polars over pandas unless strictly necessary  
- Use pathlib for all file paths  
- Use type hints and docstrings consistently  
- Prefer explicit, readable, and inspectable code  
- Avoid hidden logic and implicit behavior  
- Always surface uncertainty instead of guessing  

---

## 📁 Repository Structure

- `src/io/` → data loading, schema definition, validation, raw inspection  
- `src/transforms/` → canonicalization and matrix construction  
- `src/network/` → network construction and analysis  
- `notebooks/` → marimo notebooks only  
- `scripts/` → pipeline entry points  
- `data/raw/` → raw Eora data (read-only)  
- `data/interim/` → cleaned canonical tables  
- `data/processed/` → analysis-ready datasets  
- `outputs/` → figures, tables, logs  
- `tmp/` → disposable data  

Reusable logic must always live in `src/`.

---

## 📊 Data Principles (Eora)

- Raw Eora data must always be inspected before transformation  
- Canonical schemas are defined in `eora_schema.py`  
- Raw datasets must never be assumed to match canonical schemas  
- Transformations must be explicit and traceable  
- If the structure of a dataset is unclear, **log a warning and continue without guessing silently**  
- All transformations must preserve interpretability of country-sector relationships  
- The system is defined at the level of interconnected country-sector nodes  

---

## 🧠 Coding Principles

- Separate concerns clearly:
  - IO logic  
  - transformation logic  
  - analytical logic  

- Prefer clarity over abstraction  
- Avoid overengineering  
- Avoid hidden behavior and implicit transformations  
- Write modular, reusable code  
- Avoid duplication across files  

### Function design
- Small helper functions should be focused and single-purpose  
- Larger orchestration functions are encouraged when they reflect the full human logic of a process  
- Large functions should coordinate steps, not hide complex logic inline  
- Reusable mechanisms must be extracted into clearly named helper functions  

---

## 🏗️ Design Style (Functions vs OOP)

- OOP is fully acceptable and encouraged when it improves clarity and structure  
- Classes should remain simple, explicit, and easy to inspect  
- Prefer designs where:
  - each class has a clear role  
  - each method has a clear purpose  
- Do not introduce abstraction layers that obscure behavior  
- Whether using functions or classes, the priority is always:
  - readability  
  - traceability  
  - explicit logic  

---

## 📓 Marimo Notebook Rules (MANDATORY)

This project uses marimo, not Jupyter.

- Notebooks must be reactive, not sequential  
- No reliance on execution order  
- No hidden state across cells  
- No mutation of objects defined in other cells  

Each cell must:
- be self-contained OR explicitly dependent on inputs  
- define all required variables or receive them explicitly  

Notebooks must:
- call functions defined in `src/`  
- orchestrate computation  
- display outputs and results  

Notebooks must NOT:
- implement core logic  
- contain long procedural pipelines  
- duplicate transformation logic  

Exploratory work is allowed, but reusable logic must always be moved to `src/`.

**Mental model:** notebooks are dependency graphs, not scripts  

---

## ⚠️ Error Handling and Logging

- Never stop execution due to errors  
- Always **log warnings and continue execution**  

Error handling rules:
- If a step fails:
  - log the issue clearly  
  - continue processing remaining data  

- Logging must always make visible:
  - what went in  
  - what changed  
  - what came out  

- Prefer inline console output over log files  
- Include visibility checks such as:
  - dataframe shapes  
  - column changes  
  - dropped or created rows  
  - output summaries  

- Silent failure is strictly forbidden  

---

## 🧬 Personal Coding Style

The code in this repository follows a research-oriented style centered on readability and transparency.

### General principles
- Prefer readable code over compact code  
- Verbosity is acceptable if it improves understanding  
- Prefer simple logic over clever logic  
- Code should be understandable without reconstructing hidden reasoning  

---

### Transformations and intermediate steps
- Transformations must remain inspectable  
- Prefer sequential steps over nested logic  
- Intermediate states are useful during development  
- Once stable, intermediate variables may be reduced if visibility is preserved through logs  

---

### Naming
- Use highly specific variable names  
- Clearly distinguish dataframes, matrices, and objects  
- Apply this even in loops and local scopes  
- Prefer clarity over brevity  

---

### Validation
- Validate data immediately after each step  
- Always make visible how data changed  
- Use simple, explicit checks:
  - shapes  
  - columns  
  - transformations  

---

### Comments
- Add comments for any non-trivial logic  
- Comments must be:
  - short  
  - precise  
  - literal  

- Good comments explain:
  - what goes in  
  - what is transformed  
  - what comes out  

- Avoid vague or high-level comments  

---

### Abstraction rules
Avoid:
- overly abstract helper layers  
- “smart” utilities hiding logic  
- magic behavior  
- unclear function arguments  

Every function must:
- have a clear purpose  
- have explicit inputs  
- have explicit outputs  
- be easy to locate and understand  

---

## 🚫 What Codex Must NOT Do

- Do not assume the structure of Eora datasets without inspection  
- Do not invent or approximate missing data structures  
- Do not embed core logic inside notebooks  
- Do not create unnecessary abstractions  
- Do not introduce hidden dependencies between modules  
- Do not write code that relies on implicit state or execution order  
- Do not prioritize cleverness over clarity  

---

## ✅ Definition of Done

A task is complete when:
- code is modular and well-structured  
- functions are reusable and clearly scoped  
- data transformations are explicit and validated  
- outputs are inspectable and interpretable  
- assumptions are clearly stated  
- no hidden behavior or ambiguity remains  

---

## 🧩 Philosophy

- Transparency over cleverness  
- Structure over speed  
- Control over automation  
- Reproducibility over convenience  
- Systems thinking over isolated metrics  