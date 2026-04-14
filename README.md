# Eora26 Research Repository

Python-only research scaffold for a multi-year Eora26 project. Shared logic lives in `src/`, marimo notebooks stay thin, and the repository is organized around a simple raw/interim/processed data flow.

## Project Structure

```text
remaster/
|-- data/
|   |-- raw/          # Original source files
|   |-- interim/      # Temporary cleaned or reshaped datasets
|   `-- processed/    # Analysis-ready tables
|-- notebooks/        # Thin marimo notebooks/apps
|-- outputs/          # Figures, tables, exports, reports
|-- src/
|   |-- io/
|   |   `-- loaders.py
|   |-- config.py
|   |-- paths.py
|   `-- unzip_to_parquet.py
`-- tests/
```

## What Is Included

- `src/paths.py`: canonical project paths and a helper to create standard directories
- `src/config.py`: starter project configuration dataclass
- `src/io/loaders.py`: basic reusable loaders for parquet and CSV inputs
- `tests/test_paths.py`: small smoke tests for the shared project scaffold
- `pyproject.toml`: package metadata and core dependencies for Python research work with marimo

## First Steps

1. Create or activate your virtual environment.
2. Install the project in editable mode:

   ```powershell
   pip install -e .[dev]
   ```

3. Run the starter tests:

   ```powershell
   pytest
   ```

4. Launch a thin marimo notebook or app:

   ```powershell
   marimo edit notebooks
   ```

## Working Pattern

- Put reusable file-system, loading, transformation, and analysis helpers in `src/`.
- Keep notebooks focused on orchestration, exploration, visualization, and narrative.
- Write intermediate artifacts to `data/interim/` and stable analysis-ready outputs to `data/processed/`.
- Save figures and exportable deliverables in `outputs/`.

## Not Yet Implemented

This scaffold intentionally does **not** implement any Eora economics, IO model logic, or indicator calculations yet. It is only the project foundation.
