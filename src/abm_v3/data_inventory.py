from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

TABULAR_EXTENSIONS = {".csv", ".parquet"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".webp"}
DOCUMENT_EXTENSIONS = {".md", ".pdf", ".docx", ".txt"}
JSON_EXTENSIONS = {".json"}
SMALL_TEXT_FILE_LIMIT_BYTES = 20_000_000
PARQUET_SAMPLE_LIMIT_BYTES = 100_000_000
PARQUET_SAMPLE_COLUMN_LIMIT = 25


@dataclass(frozen=True)
class FileInspection:
    """Cheap file-level inspection result."""

    path: Path
    relative_path: str
    file_name: str
    extension: str
    size_bytes: int
    size_mb: float
    modified_time: str
    directory: str
    format: str
    row_count: int | None
    column_count: int | None
    columns: list[str]
    dtypes: dict[str, str]
    sample: pd.DataFrame | None
    notes: list[str]


def build_data_inventory(
    root: Path | str = Path("data"),
    focus: str = "abm_v3",
    sample_rows: int = 5,
    max_files: int | None = None,
    include_raw: bool = False,
    output_dir: Path | str = Path("data/abm_v3/data_inventory"),
) -> dict[str, Path]:
    """Inspect data files and write ABM v3 inventory outputs.

    The inventory uses schema metadata and small samples. It does not run the
    model, rewrite existing data files, or fully load large tabular datasets.
    """
    root_path = Path(root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = discover_data_files(root_path, focus=focus, include_raw=include_raw)
    if max_files is not None:
        files = files[:max_files]

    inspections: list[FileInspection] = []
    for file_path in files:
        try:
            inspections.append(inspect_file(file_path, base_root=root_path, sample_rows=sample_rows))
        except Exception as error:  # noqa: BLE001 - inventory should surface uncertainty and continue.
            LOGGER.warning("Could not inspect %s: %s", file_path, error)
            inspections.append(build_failed_inspection(file_path, root_path, str(error)))

    inventory = pd.DataFrame([build_inventory_row(inspection) for inspection in inspections])
    variables = pd.DataFrame(build_variable_rows(inspections))
    visual_map = pd.DataFrame(build_visual_use_map_rows())
    semantic_map = pd.DataFrame(build_semantic_variable_map_rows())
    catalog = build_markdown_catalog(inventory, variables, visual_map)

    inventory_path = output_path / "data_inventory.csv"
    variables_path = output_path / "variable_inventory.csv"
    visual_map_path = output_path / "visual_use_map.csv"
    semantic_map_path = output_path / "abm_v3_semantic_variable_map.csv"
    catalog_path = output_path / "abm_v3_data_catalog.md"

    inventory.to_csv(inventory_path, index=False)
    variables.to_csv(variables_path, index=False)
    visual_map.to_csv(visual_map_path, index=False)
    semantic_map.to_csv(semantic_map_path, index=False)
    catalog_path.write_text(catalog, encoding="utf-8")

    return {
        "data_inventory": inventory_path,
        "variable_inventory": variables_path,
        "visual_use_map": visual_map_path,
        "semantic_variable_map": semantic_map_path,
        "catalog": catalog_path,
    }


def discover_data_files(root: Path, focus: str = "abm_v3", include_raw: bool = False) -> list[Path]:
    """Discover candidate files deterministically for the selected inventory focus."""
    candidate_roots = build_candidate_roots(root, focus=focus, include_raw=include_raw)
    files: list[Path] = []
    for candidate_root in candidate_roots:
        if not candidate_root.exists():
            LOGGER.warning("Inventory root does not exist: %s", candidate_root)
            continue
        files.extend(path for path in candidate_root.rglob("*") if path.is_file())

    output_inventory_parts = {"data_inventory"}
    filtered_files = [
        path
        for path in files
        if not output_inventory_parts.intersection(set(path.parts))
        and (include_raw or "raw" not in {part.lower() for part in path.parts})
    ]
    return sorted(set(filtered_files), key=lambda path: path.as_posix().lower())


def build_candidate_roots(root: Path, focus: str, include_raw: bool) -> list[Path]:
    """Build the root folders used by the inventory."""
    if focus == "all":
        return [root]
    if focus != "abm_v3":
        raise ValueError(f"Unknown data inventory focus: {focus}")

    roots = [
        root / "abm_v3",
        root / "atlas" / "processed",
        root / "parquet",
        root / "final",
        root / "abm",
    ]
    if include_raw:
        roots.extend([root / "raw", root / "atlas" / "raw"])

    plots_root = root.parent / "outputs" / "plots"
    if root.name == "data" and plots_root.exists():
        roots.append(plots_root)
    return roots


def inspect_file(path: Path, base_root: Path, sample_rows: int) -> FileInspection:
    """Inspect one file with metadata, schema, and a tiny sample where supported."""
    stat = path.stat()
    extension = path.suffix.lower()
    notes: list[str] = []
    row_count: int | None = None
    column_count: int | None = None
    columns: list[str] = []
    dtypes: dict[str, str] = {}
    sample: pd.DataFrame | None = None

    if extension == ".parquet":
        parquet_result = inspect_parquet(path, sample_rows)
        row_count = parquet_result["row_count"]
        columns = parquet_result["columns"]
        dtypes = parquet_result["dtypes"]
        sample = parquet_result["sample"]
        column_count = len(columns)
        notes.extend(parquet_result["notes"])
    elif extension == ".csv":
        csv_result = inspect_csv(path, sample_rows)
        row_count = csv_result["row_count"]
        columns = csv_result["columns"]
        dtypes = csv_result["dtypes"]
        sample = csv_result["sample"]
        column_count = len(columns)
        notes.extend(csv_result["notes"])
    elif extension in JSON_EXTENSIONS:
        json_result = inspect_json(path)
        row_count = json_result["row_count"]
        columns = json_result["columns"]
        dtypes = json_result["dtypes"]
        sample = json_result["sample"]
        column_count = len(columns) if columns else None
        notes.extend(json_result["notes"])
    elif extension in IMAGE_EXTENSIONS:
        notes.append("visual output metadata only")
    elif extension in DOCUMENT_EXTENSIONS:
        notes.append("document metadata only")
    else:
        notes.append("unsupported file type; metadata only")

    return FileInspection(
        path=path,
        relative_path=safe_relative_path(path, base_root),
        file_name=path.name,
        extension=extension,
        size_bytes=stat.st_size,
        size_mb=round(stat.st_size / 1_048_576, 4),
        modified_time=datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        directory=safe_relative_path(path.parent, base_root),
        format=extension.lstrip(".") or "unknown",
        row_count=row_count,
        column_count=column_count,
        columns=columns,
        dtypes=dtypes,
        sample=sample,
        notes=notes,
    )


def inspect_parquet(path: Path, sample_rows: int) -> dict[str, Any]:
    """Inspect a parquet schema and optional head without reading the whole file."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        LOGGER.warning("pyarrow is not available; parquet schema unavailable for %s", path)
        return {"row_count": None, "columns": [], "dtypes": {}, "sample": None, "notes": ["pyarrow unavailable"]}

    parquet_file = pq.ParquetFile(path)
    arrow_schema = parquet_file.schema_arrow
    columns = list(arrow_schema.names)
    dtypes = {field.name: str(field.type) for field in arrow_schema}
    sample = None
    notes = ["parquet schema inspected from metadata"]
    if path.stat().st_size > PARQUET_SAMPLE_LIMIT_BYTES:
        notes.append("sample skipped for large parquet")
    elif sample_rows > 0 and parquet_file.metadata.num_rows > 0:
        try:
            sample_columns = columns[:PARQUET_SAMPLE_COLUMN_LIMIT]
            sample_table = parquet_file.read_row_group(0, columns=sample_columns)
            sample = sample_table.slice(0, sample_rows).to_pandas()
            notes.append(
                f"sampled first {min(sample_rows, len(sample))} rows and "
                f"{len(sample_columns)} columns from first row group"
            )
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("Could not sample parquet %s: %s", path, error)
            notes.append(f"sample unavailable: {error}")
    return {
        "row_count": parquet_file.metadata.num_rows,
        "columns": columns,
        "dtypes": dtypes,
        "sample": sample,
        "notes": notes,
    }


def inspect_csv(path: Path, sample_rows: int) -> dict[str, Any]:
    """Inspect a CSV by reading a small preview and counting rows only when cheap."""
    notes: list[str] = []
    try:
        sample = pd.read_csv(path, nrows=sample_rows)
    except pd.errors.EmptyDataError:
        return {"row_count": 0, "columns": [], "dtypes": {}, "sample": None, "notes": ["empty CSV"]}
    columns = [str(column) for column in sample.columns]
    dtypes = {str(column): str(dtype) for column, dtype in sample.dtypes.items()}
    row_count = count_csv_rows_if_cheap(path, notes)
    notes.append(f"sampled first {len(sample)} rows")
    return {"row_count": row_count, "columns": columns, "dtypes": dtypes, "sample": sample, "notes": notes}


def count_csv_rows_if_cheap(path: Path, notes: list[str]) -> int | None:
    """Count CSV rows for small files, avoiding full scans of large CSVs."""
    if path.stat().st_size > SMALL_TEXT_FILE_LIMIT_BYTES:
        notes.append("row count skipped for large CSV")
        return None
    with path.open("r", encoding="utf-8", errors="replace", newline="") as file_handle:
        line_count = sum(1 for _ in file_handle)
    return max(0, line_count - 1)


def inspect_json(path: Path) -> dict[str, Any]:
    """Inspect a small JSON file if safe."""
    if path.stat().st_size > SMALL_TEXT_FILE_LIMIT_BYTES:
        return {"row_count": None, "columns": [], "dtypes": {}, "sample": None, "notes": ["large JSON metadata only"]}
    with path.open("r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    records = data if isinstance(data, list) else [data]
    sample = pd.DataFrame(records[:5]) if records and isinstance(records[0], dict) else None
    columns = list(sample.columns) if sample is not None else []
    dtypes = {str(column): str(dtype) for column, dtype in sample.dtypes.items()} if sample is not None else {}
    return {
        "row_count": len(records) if isinstance(data, list) else 1,
        "columns": columns,
        "dtypes": dtypes,
        "sample": sample,
        "notes": ["small JSON inspected"],
    }


def build_failed_inspection(path: Path, base_root: Path, error_message: str) -> FileInspection:
    """Create a metadata-only record when inspection fails."""
    stat = path.stat()
    return FileInspection(
        path=path,
        relative_path=safe_relative_path(path, base_root),
        file_name=path.name,
        extension=path.suffix.lower(),
        size_bytes=stat.st_size,
        size_mb=round(stat.st_size / 1_048_576, 4),
        modified_time=datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        directory=safe_relative_path(path.parent, base_root),
        format=path.suffix.lower().lstrip(".") or "unknown",
        row_count=None,
        column_count=None,
        columns=[],
        dtypes={},
        sample=None,
        notes=[f"inspection failed: {error_message}"],
    )


def build_inventory_row(inspection: FileInspection) -> dict[str, Any]:
    """Build one data inventory row from an inspection."""
    file_group = classify_file_group(Path(inspection.relative_path))
    data_layer = classify_data_layer(Path(inspection.relative_path), file_group, inspection.extension)
    status = classify_status(Path(inspection.relative_path), file_group, data_layer)
    column_set = set(inspection.columns)
    semantic_categories = [classify_variable_semantics(column) for column in inspection.columns]
    year_min, year_max = infer_year_bounds(inspection)
    primary_keys = infer_primary_key_candidates(inspection.columns)

    return {
        "path": str(inspection.path),
        "relative_path": inspection.relative_path,
        "file_name": inspection.file_name,
        "extension": inspection.extension,
        "size_bytes": inspection.size_bytes,
        "size_mb": inspection.size_mb,
        "modified_time": inspection.modified_time,
        "directory": inspection.directory,
        "file_group": file_group,
        "data_layer": data_layer,
        "status": status,
        "time_coverage": format_time_coverage(year_min, year_max),
        "year_min": year_min,
        "year_max": year_max,
        "format": inspection.format,
        "row_count": inspection.row_count,
        "column_count": inspection.column_count,
        "columns_preview": join_preview(inspection.columns, limit=40),
        "primary_key_candidates": "; ".join(primary_keys),
        "contains_country_sector": "country_sector" in {column.lower() for column in inspection.columns},
        "contains_year": any(classify_variable_semantics(column) == "time" for column in inspection.columns),
        "contains_scenario_name": any("scenario" in column.lower() for column in inspection.columns),
        "contains_ei": "emissions_intensity" in semantic_categories,
        "contains_green_capability": "green_capability" in semantic_categories,
        "contains_network_green": "network_green_ness" in semantic_categories,
        "contains_production": "production" in semantic_categories,
        "contains_emissions": "emissions" in semantic_categories,
        "contains_centrality": "centrality" in semantic_categories or "brown_centrality" in semantic_categories,
        "contains_validation_metrics": "validation" in semantic_categories,
        "contains_scenario_metrics": "scenario" in semantic_categories,
        "recommended_use": recommend_file_use(file_group, status, semantic_categories, column_set),
        "phase_space_use": recommend_phase_space_use(file_group, status, semantic_categories, column_set),
        "visualisation_use": recommend_visualisation_use(file_group, status, semantic_categories),
        "notes": "; ".join(inspection.notes),
    }


def build_variable_rows(inspections: list[FileInspection]) -> list[dict[str, Any]]:
    """Build one row per variable per usable tabular file."""
    rows: list[dict[str, Any]] = []
    for inspection in inspections:
        if inspection.extension not in TABULAR_EXTENSIONS and inspection.extension not in JSON_EXTENSIONS:
            continue
        file_group = classify_file_group(Path(inspection.relative_path))
        for variable in inspection.columns:
            semantic_category = classify_variable_semantics(variable)
            sample_stats = summarize_sample_variable(inspection.sample, variable)
            rows.append(
                {
                    "path": inspection.relative_path,
                    "file_group": file_group,
                    "variable": variable,
                    "dtype": inspection.dtypes.get(variable, ""),
                    "non_null_count_sample_or_full": sample_stats["non_null_count"],
                    "missing_share_sample_or_full": sample_stats["missing_share"],
                    "example_values": sample_stats["example_values"],
                    "semantic_category": semantic_category,
                    "economic_meaning_short": economic_meaning_short(semantic_category),
                    "usable_for_phase_space": usable_for_phase_space(semantic_category),
                    "usable_for_trajectory": usable_for_trajectory(semantic_category),
                    "usable_for_vector_field": usable_for_vector_field(semantic_category),
                    "usable_for_scenario_analysis": semantic_category == "scenario",
                    "usable_for_ei_transition": semantic_category in {"emissions_intensity", "green_capability", "network_green_ness", "production", "time", "identifier"},
                    "usable_for_validation": semantic_category in {"validation", "production", "emissions", "emissions_intensity", "identifier", "time"},
                    "suggested_axis_role": suggested_axis_role(semantic_category, variable),
                    "notes": variable_notes(semantic_category),
                }
            )
    return rows


def classify_file_group(relative_path: Path) -> str:
    """Classify a file group from path and name patterns."""
    path_text = relative_path.as_posix().lower()
    name = relative_path.name.lower()
    if path_text.startswith("raw/") or "/raw/" in path_text:
        return "raw_eora"
    if path_text.startswith("parquet/") or path_text.startswith("final/"):
        return "processed_eora"
    if path_text.startswith("atlas/raw/"):
        return "atlas_raw"
    if path_text.startswith("atlas/processed/") or "atlas_eora26_sector_capabilities" in name:
        return "atlas_processed"
    if path_text.startswith("abm/"):
        return "legacy_abm"
    if path_text.startswith("../outputs/plots") or path_text.startswith("outputs/plots") or "plots/" in path_text:
        return "plots"
    if "abm_v3/inputs/" in path_text and "historical_panel" in name:
        return "abm_v3_input_panel"
    if "abm_v3/calibration/" in path_text:
        return "abm_v3_calibration"
    if "abm_v3/validation_report/" in path_text:
        return "abm_v3_validation_report"
    if "abm_v3/ei_transition/" in path_text:
        return "abm_v3_ei_transition"
    if "abm_v3/leontief/behavioural/scenarios/analysis_report/" in path_text:
        return "abm_v3_scenario_analysis"
    if "abm_v3/leontief/behavioural/scenarios/" in path_text:
        return "abm_v3_scenarios"
    if "abm_v3/leontief/behavioural/" in path_text:
        return "abm_v3_leontief_behavioural"
    if "abm_v3/leontief/" in path_text:
        return "abm_v3_leontief_pure"
    if "abm_v3/diagnostics/" in path_text:
        return "abm_v3_diagnostics"
    return "unknown"


def classify_data_layer(relative_path: Path, file_group: str, extension: str) -> str:
    """Classify data layer from file group and path."""
    path_text = relative_path.as_posix().lower()
    if file_group in {"raw_eora", "atlas_raw"}:
        return "raw"
    if file_group in {"processed_eora", "atlas_processed"}:
        return "processed"
    if file_group == "abm_v3_input_panel":
        return "model_input"
    if file_group in {"abm_v3_validation_report"}:
        return "validation"
    if file_group in {"abm_v3_scenarios", "abm_v3_scenario_analysis"}:
        return "scenario"
    if file_group in {"abm_v3_ei_transition", "abm_v3_diagnostics", "abm_v3_calibration"}:
        return "diagnostic"
    if file_group in {"abm_v3_leontief_pure", "abm_v3_leontief_behavioural", "legacy_abm"}:
        return "model_output" if "outputs" in path_text or extension == ".parquet" else "diagnostic"
    if file_group == "plots" or extension in IMAGE_EXTENSIONS:
        return "visual_output"
    if extension in DOCUMENT_EXTENSIONS:
        return "documentation"
    return "unknown"


def classify_status(relative_path: Path, file_group: str, data_layer: str) -> str:
    """Classify current authority and diagnostic status."""
    path_text = relative_path.as_posix().lower()
    name = relative_path.name.lower()
    if file_group in {"raw_eora", "atlas_raw"}:
        return "raw_source"
    if file_group in {"processed_eora", "atlas_processed"}:
        return "processed_source"
    if file_group == "legacy_abm":
        return "legacy"
    if file_group == "abm_v3_input_panel" and "transpose_row_fd_without_inventory" in path_text:
        return "authoritative_state_source"
    if file_group == "abm_v3_input_panel":
        return "intermediate"
    if file_group == "abm_v3_ei_transition" and "/inputs/" in path_text and "ei_transition_panel" in name:
        return "authoritative_state_source"
    if file_group == "abm_v3_ei_transition" and "/predictions/" in path_text:
        return "current_diagnostic"
    if file_group == "abm_v3_scenario_analysis":
        return "current_scenario_output"
    if file_group == "abm_v3_scenarios":
        return "current_scenario_output"
    if file_group == "abm_v3_leontief_behavioural":
        if "node_comparison" in name:
            return "authoritative_model_output"
        return "current_diagnostic"
    if file_group == "abm_v3_validation_report":
        return "current_diagnostic"
    if file_group == "abm_v3_ei_transition":
        return "current_diagnostic"
    if file_group == "plots" or data_layer == "visual_output":
        return "visual_output"
    if data_layer == "diagnostic":
        return "current_diagnostic"
    return "unclear"


def classify_variable_semantics(variable: str) -> str:
    """Classify a column into an economic or modelling semantic category."""
    name = variable.lower()
    if name in {"country_sector", "country", "sector", "category", "country_detail", "node"} or name.endswith("_id"):
        return "identifier"
    if name in {"year", "time", "period"} or name.endswith("_year"):
        return "time"
    if "scenario" in name or "shock_size" in name or "selector" in name or name.startswith("delta_") or "pct_delta" in name:
        return "scenario"
    if "residual" in name or "converged" in name or "rounds" in name or "validation_loss" in name or "error" in name or "rmse" in name or "mae" in name or "r2" == name:
        return "validation"
    if "brown_centrality" in name or "brown_network" in name or "embodied_carbon_centrality" in name:
        return "brown_centrality"
    if "centrality" in name or "pagerank" in name or "betweenness" in name:
        return "centrality"
    if "network_green" in name or "incoming_green" in name or "outgoing_green" in name or "recursive_green" in name:
        return "network_green_ness"
    if "green_capability" in name or "capability_export" in name or "capability_share" in name:
        return "green_capability"
    if "ecosystem" in name or "proximity" in name or "pci" in name or "complexity" in name or "eci" in name:
        return "economic_complexity"
    if "local_green" in name or "green_ness" in name or name in {"g_local", "greenness"}:
        return "local_green_ness"
    if name in {"ei", "log_ei"} or "emissions_intensity" in name or name.endswith("_ei"):
        return "emissions_intensity"
    if "co2" in name or "emission" in name:
        return "emissions"
    if "capacity" in name or "capacity_utilization" in name or name == "k":
        return "capacity"
    if name == "x" or "x_observed" in name or "x_realized" in name or "output" in name or "production" in name:
        return "production"
    if "final_demand" in name or name.startswith("y_") or name == "y":
        return "final_demand"
    if "technical_coefficient" in name or name.startswith("a_") or "input_intensity" in name or "leontief" in name:
        return "input_output_structure"
    if "diagnostic" in name or "warning" in name or "notes" in name:
        return "diagnostic"
    if "path" in name or "source" in name or "file" in name or "description" in name:
        return "metadata"
    return "unknown"


def infer_year_bounds(inspection: FileInspection) -> tuple[int | None, int | None]:
    """Infer year coverage from sample values first and file name second."""
    year_columns = [column for column in inspection.columns if classify_variable_semantics(column) == "time"]
    if inspection.sample is not None and year_columns:
        year_values = pd.to_numeric(inspection.sample[year_columns[0]], errors="coerce").dropna()
        if not year_values.empty:
            return int(year_values.min()), int(year_values.max())

    years = [int(token) for token in inspection.file_name.replace(".", "_").split("_") if token.isdigit() and 1900 <= int(token) <= 2100]
    if years:
        return min(years), max(years)
    return None, None


def format_time_coverage(year_min: int | None, year_max: int | None) -> str:
    """Format inferred year coverage."""
    if year_min is None and year_max is None:
        return ""
    if year_min == year_max:
        return str(year_min)
    return f"{year_min}-{year_max}"


def infer_primary_key_candidates(columns: list[str]) -> list[str]:
    """Infer likely primary key candidates from available columns."""
    lower_columns = {column.lower(): column for column in columns}
    candidates: list[str] = []
    if "country_sector" in lower_columns and "year" in lower_columns:
        candidates.append(f"{lower_columns['country_sector']} + {lower_columns['year']}")
    if "scenario_name" in lower_columns and "country_sector" in lower_columns and "year" in lower_columns:
        candidates.append(f"{lower_columns['scenario_name']} + {lower_columns['country_sector']} + {lower_columns['year']}")
    if "country" in lower_columns and "sector" in lower_columns and "year" in lower_columns:
        candidates.append(f"{lower_columns['country']} + {lower_columns['sector']} + {lower_columns['year']}")
    return candidates


def recommend_file_use(
    file_group: str,
    status: str,
    semantic_categories: list[str],
    columns: set[str],
) -> str:
    """Recommend a file-level use."""
    categories = set(semantic_categories)
    if status == "authoritative_state_source" and file_group == "abm_v3_input_panel":
        return "current ABM v3 historical state panel; best base for country-sector phase-space trajectories"
    if status == "authoritative_state_source" and file_group == "abm_v3_ei_transition":
        return "EI transition panel; prime source for EI movement and vector-field diagnostics after schema validation"
    if status == "authoritative_model_output":
        return "current validated model output; useful for validation and propagation interpretation, not primary historical state construction"
    if file_group == "atlas_processed":
        return "capability and product-space variables; join to ABM v3 panel by country-sector and year when available"
    if file_group == "abm_v3_leontief_behavioural":
        return "behavioural Leontief validation and production response output"
    if file_group == "abm_v3_scenario_analysis":
        return "current scenario analysis output for scenario comparison and selector overlap plots"
    if file_group == "abm_v3_ei_transition":
        return "EI transition historical-learning diagnostic; use for EI reduction interpretation, not full scenario dynamics"
    if file_group == "abm_v3_validation_report":
        return "current validation summary/reporting layer"
    if file_group == "legacy_abm":
        return "legacy ABM source; useful for comparison only after validation"
    if "validation" in categories:
        return "diagnostic validation data"
    if columns:
        return "inspect before use; semantic role inferred from columns"
    return "metadata only"


def recommend_phase_space_use(
    file_group: str,
    status: str,
    semantic_categories: list[str],
    columns: set[str],
) -> str:
    """Recommend whether and how a file can support phase-space plotting."""
    categories = set(semantic_categories)
    has_node_time = {"identifier", "time"}.issubset(categories)
    if status == "authoritative_state_source" and file_group == "abm_v3_input_panel" and has_node_time:
        return "primary state trajectory source"
    if status == "authoritative_state_source" and file_group == "abm_v3_ei_transition" and has_node_time:
        return "EI movement/vector-field diagnostic source"
    if status == "authoritative_model_output":
        return "validated model-output context, not primary state source"
    if file_group == "atlas_processed":
        return "capability axis candidate after joining to country-sector-year state panel"
    if file_group == "abm_v3_ei_transition" and "emissions_intensity" in categories:
        return "EI movement/vector diagnostic source"
    if file_group in {"abm_v3_scenarios", "abm_v3_scenario_analysis"}:
        return "scenario perturbation overlay source"
    if {"green_capability", "network_green_ness", "local_green_ness"}.intersection(categories):
        return "candidate phase-space axis source after key validation"
    return ""


def recommend_visualisation_use(file_group: str, status: str, semantic_categories: list[str]) -> str:
    """Recommend visualisation families for a file."""
    categories = set(semantic_categories)
    uses: list[str] = []
    if {"production", "emissions", "emissions_intensity"}.intersection(categories):
        uses.append("trajectory plots")
    if {"green_capability", "local_green_ness", "network_green_ness", "brown_centrality"}.intersection(categories):
        uses.append("3D phase-space cubes")
    if "scenario" in categories or file_group in {"abm_v3_scenarios", "abm_v3_scenario_analysis"}:
        uses.append("scenario plots")
    if "validation" in categories or status == "current_diagnostic":
        uses.append("validation/diagnostic plots")
    return "; ".join(uses)


def summarize_sample_variable(sample: pd.DataFrame | None, variable: str) -> dict[str, Any]:
    """Summarize missingness and examples from the inspected sample."""
    if sample is None or variable not in sample.columns:
        return {"non_null_count": None, "missing_share": None, "example_values": ""}
    series = sample[variable]
    non_null_count = int(series.notna().sum())
    missing_share = float(series.isna().mean()) if len(series) else None
    examples = [str(value) for value in series.dropna().astype(str).unique()[:5]]
    return {
        "non_null_count": non_null_count,
        "missing_share": round(missing_share, 4) if missing_share is not None else None,
        "example_values": "; ".join(examples),
    }


def economic_meaning_short(semantic_category: str) -> str:
    """Map semantic category to a short economic interpretation."""
    meanings = {
        "identifier": "unit key, usually country-sector node identity",
        "time": "historical year or model period",
        "production": "production scale or modelled output response",
        "final_demand": "final demand entering the input-output system",
        "input_output_structure": "technical coefficients or Leontief production structure",
        "emissions": "total emissions, distinct from production and intensity",
        "emissions_intensity": "emissions per unit output; local carbon intensity",
        "local_green_ness": "local greenness of production or sector state",
        "network_green_ness": "network-embedded green exposure or recursive greenness",
        "green_capability": "productive capability for green products/sectors",
        "economic_complexity": "complexity, proximity, or capability readiness",
        "centrality": "network position or propagation relevance",
        "brown_centrality": "carbon-intensive network position or lock-in exposure",
        "capacity": "capacity proxy or production constraint variable",
        "scenario": "scenario selector, shock, or scenario response",
        "validation": "fit, residual, convergence, or validation metric",
        "diagnostic": "diagnostic or warning metadata",
        "metadata": "file/source/report metadata",
        "unknown": "unclear meaning; inspect before interpretation",
    }
    return meanings.get(semantic_category, "unclear meaning; inspect before interpretation")


def usable_for_phase_space(semantic_category: str) -> bool:
    """Return whether a variable is a phase-space state candidate."""
    return semantic_category in {
        "production",
        "emissions",
        "emissions_intensity",
        "local_green_ness",
        "network_green_ness",
        "green_capability",
        "economic_complexity",
        "centrality",
        "brown_centrality",
        "capacity",
    }


def usable_for_trajectory(semantic_category: str) -> bool:
    """Return whether a variable can support trajectories."""
    return semantic_category in {"identifier", "time"} or usable_for_phase_space(semantic_category)


def usable_for_vector_field(semantic_category: str) -> bool:
    """Return whether a variable can support binned movement/vector fields."""
    return semantic_category in {
        "production",
        "emissions_intensity",
        "local_green_ness",
        "network_green_ness",
        "green_capability",
        "economic_complexity",
        "brown_centrality",
    }


def suggested_axis_role(semantic_category: str, variable: str) -> str:
    """Suggest a plotting axis role for a variable."""
    axis_roles = {
        "green_capability": "X axis: green capability / readiness",
        "local_green_ness": "Y axis: local greenness, green up",
        "network_green_ness": "Z axis: network greenness / exposure",
        "brown_centrality": "X axis in brown lock-in cube",
        "economic_complexity": "Y/Z axis for product-space readiness",
        "production": "weight, size, or X axis for production-safe greening",
        "emissions": "weight, size, or top-emitter selection",
        "emissions_intensity": "movement target or EI reduction vector",
        "scenario": "facet/color/selector overlay",
        "validation": "diagnostic y-axis or residual color",
        "time": "trajectory ordering and markers",
        "identifier": "unit of observation and trace key",
    }
    return axis_roles.get(semantic_category, "")


def variable_notes(semantic_category: str) -> str:
    """Give a short caveat for variable use."""
    if semantic_category == "green_capability":
        return "Capability is not proof of low-carbon production."
    if semantic_category == "emissions_intensity":
        return "Intensity is not total emissions; pair with output or emissions for scale."
    if semantic_category == "network_green_ness":
        return "Network-embedded greenness differs from local greenness."
    if semantic_category == "scenario":
        return "Scenario response is not yet a full green-transition simulation unless explicitly integrated."
    if semantic_category == "capacity":
        return "Capacity proxies are not yet adaptive capacity."
    if semantic_category == "unknown":
        return "Meaning unclear from name; inspect source before use."
    return ""


def build_semantic_variable_map_rows() -> list[dict[str, str]]:
    """Build the curated ABM v3 semantic variable map for methods and plotting."""
    state_panel = "data/abm_v3/inputs/abm_v3_historical_panel_1995_2016_transpose_row_fd_without_inventory.parquet"
    phase_panel = "data/abm_v3/phase_space/abm_v3_phase_space_state_panel_1995_2016.parquet"
    ei_panel = "data/abm_v3/ei_transition/inputs/ei_transition_panel_1995_2016.parquet"
    ei_predictions = "data/abm_v3/ei_transition/predictions/ei_transition_predictions_1995_2016.parquet"
    atlas_panel = "data/atlas/processed/atlas_eora26_sector_capabilities_1995_2016.parquet"
    scenario_outputs = "data/abm_v3/leontief/behavioural/scenarios/analysis_report/"
    diagnostics = "data/abm_v3/diagnostics/; data/abm_v3/leontief/behavioural/diagnostics/"

    rows = [
        semantic_row("country_sector", "country_sector; node", "identifier", "Stable country-sector node key used to define production-system units.", state_panel, "", "country_sector-year", "annual", "high", "high", "high", "high", "trace_key", "Defines the node whose movement is tracked through state space.", "Must remain stable across joins; do not infer missing node labels silently.", "required"),
        semantic_row("Country", "Country; country", "identifier", "Country label for aggregation, faceting, and interpretation.", state_panel, atlas_panel, "country_sector-year", "annual", "medium", "high", "low", "high", "facet", "Separates national production-system context from sector identity.", "May be absent if only compound country_sector labels are available.", "high"),
        semantic_row("Country_detail", "Country_detail; country_detail", "identifier", "Detailed country label when harmonized country metadata are available.", state_panel, atlas_panel, "country_sector-year", "annual", "low", "medium", "low", "high", "label", "Improves interpretation without changing the node definition.", "Not required for state-space construction.", "medium"),
        semantic_row("Sector", "Sector; sector", "identifier", "Eora26 sector label for sector-level aggregation and interpretation.", state_panel, atlas_panel, "country_sector-year", "annual", "medium", "high", "low", "high", "facet", "Identifies sectoral location of transition opportunities and lock-in.", "Sector names must be validated after joins.", "high"),
        semantic_row("Category", "Category; category", "identifier", "Broad sector or node category used for grouping and visual filtering.", state_panel, atlas_panel, "country_sector-year", "annual", "low", "medium", "low", "medium", "facet", "Supports readable grouping of node trajectories.", "Category systems may differ across sources.", "medium"),
        semantic_row("Year", "Year; year", "time", "Calendar year of the observed state.", state_panel, ei_panel, "country_sector-year", "annual", "high", "high", "high", "high", "trajectory_time", "Orders historical movement and year-to-year transition vectors.", "Must be integer-like and within the requested build window.", "required"),
        semantic_row("X_observed", "X_observed; output; production", "production", "Observed production scale for the country-sector node.", state_panel, diagnostics, "country_sector-year", "annual", "high", "high", "medium", "high", "weight_or_x_axis", "Measures production scale, not emissions or greenness.", "Use as default trajectory weight; validate units before cross-source comparison.", "required"),
        semantic_row("log_X_observed", "log_X_observed", "production", "Log-transformed observed production scale, computed as log1p(X_observed).", phase_panel, state_panel, "country_sector-year", "annual", "high", "high", "medium", "high", "x_axis", "Supports production-safe greening views without letting very large nodes dominate geometry.", "Derived variable; only valid when X_observed is numeric.", "high"),
        semantic_row("X_realized", "X_realized; realized_output", "production", "Model-realized production after Leontief propagation or scenario response.", diagnostics, scenario_outputs, "country_sector-year or country_sector-scenario-year", "annual", "low", "medium", "low", "high", "validation_or_overlay", "Represents model response, not observed historical state.", "Do not use as baseline historical production.", "medium"),
        semantic_row("X_desired", "X_desired; desired_output", "production", "Desired or unconstrained production before propagation constraints.", diagnostics, scenario_outputs, "country_sector-year or country_sector-scenario-year", "annual", "low", "medium", "low", "medium", "diagnostic", "Helps distinguish demand pressure from realized production.", "Diagnostic only unless explicitly validated.", "medium"),
        semantic_row("final_demand_total", "final_demand_total; total_final_demand", "final_demand", "Total final demand associated with the node.", state_panel, diagnostics, "country_sector-year", "annual", "medium", "medium", "low", "high", "context", "Demand context for production-network propagation.", "Not a substitute for production scale.", "medium"),
        semantic_row("Y_final_demand", "Y_final_demand; Y; final_demand", "final_demand", "Final-demand vector value used by input-output construction.", state_panel, diagnostics, "country_sector-year", "annual", "medium", "medium", "low", "high", "context", "Demand-side driver of production-network accounting.", "Column naming varies across input-output outputs.", "medium"),
        semantic_row("emissions_observed", "emissions_observed; emissions; CO2", "emissions", "Observed total emissions attributable to the node.", phase_panel, state_panel, "country_sector-year", "annual", "high", "high", "medium", "high", "weight_or_color", "Measures total carbon scale, distinct from emissions intensity.", "May be derived as EI times X_observed when not directly observed.", "high"),
        semantic_row("log_emissions_observed", "log_emissions_observed", "emissions", "Log-transformed total emissions, computed as log1p(emissions_observed).", phase_panel, state_panel, "country_sector-year", "annual", "medium", "high", "low", "high", "color_or_size", "Compresses emissions scale for readable comparison.", "Derived variable; only meaningful when emissions are non-negative.", "medium"),
        semantic_row("EI", "EI; emissions_intensity; emission_intensity", "emissions_intensity", "Emissions intensity of production for the node.", state_panel, ei_panel, "country_sector-year", "annual", "high", "high", "high", "high", "y_axis_or_color", "Measures carbon intensity, not total emissions.", "Positive values are required for log_EI and EI-reduction movement.", "required"),
        semantic_row("log_EI", "log_EI", "emissions_intensity", "Natural log of emissions intensity for positive EI values.", phase_panel, ei_panel, "country_sector-year", "annual", "high", "high", "high", "high", "state_or_delta_base", "Makes proportional emissions-intensity change explicit.", "Undefined for zero or negative EI; missing values must remain visible.", "high"),
        semantic_row("g_local", "g_local; local_green_ness; greenness", "local_green_ness", "Local green-ness proxy derived from the node's own emissions intensity.", phase_panel, state_panel, "country_sector-year", "annual", "high", "high", "high", "high", "green_up_axis", "Higher values indicate lower local carbon intensity, distinct from network greenness.", "Can be derived as 1/(1+EI); it is not proof of green capability.", "required"),
        semantic_row("green_capability_export_share", "green_capability_export_share", "green_capability", "Share of exports linked to green capabilities or green products.", atlas_panel, state_panel, "country_sector-year", "annual", "high", "high", "high", "high", "x_axis", "Proxy for green productive capability, not proof of low-carbon production.", "Requires validated Atlas-to-Eora join.", "high"),
        semantic_row("green_capability_share", "green_capability_share", "green_capability", "Share of identified capabilities that are green-relevant.", atlas_panel, state_panel, "country_sector-year", "annual", "medium", "high", "medium", "high", "context", "Alternative green-capability measure for robustness.", "Definition may differ from export-weighted capability.", "medium"),
        semantic_row("green_capability_readiness", "green_capability_readiness; capability_readiness", "green_capability", "Readiness proxy for moving into green capability space.", atlas_panel, state_panel, "country_sector-year", "annual", "high", "high", "high", "high", "readiness_axis", "Measures capability adjacency, not realized transition.", "Use cautiously as current capacity proxy, not adaptive capacity.", "high"),
        semantic_row("capability_export_weighted_pci", "capability_export_weighted_pci", "economic_complexity", "Export-weighted complexity of the node's capability basket.", atlas_panel, state_panel, "country_sector-year", "annual", "medium", "high", "medium", "high", "context", "Places green capability in broader product-complexity space.", "Requires product-space methodology notes in LaTeX reference.", "medium"),
        semantic_row("capability_mean_pci", "capability_mean_pci", "economic_complexity", "Mean product complexity associated with the node's capability basket.", atlas_panel, state_panel, "country_sector-year", "annual", "medium", "high", "medium", "high", "context", "Captures complexity environment around possible transition paths.", "Not itself a green metric.", "medium"),
        semantic_row("capability_ecosystem_exposure", "capability_ecosystem_exposure; ecosystem_exposure", "ecosystem_readiness", "Exposure to nearby capability ecosystems that may support transition.", atlas_panel, state_panel, "country_sector-year", "annual", "high", "high", "high", "high", "y_axis", "Represents product-space readiness rather than observed greening.", "Missing unless Atlas-derived ecosystem variables have been joined.", "high"),
        semantic_row("general_complexity", "general_complexity; ECI; complexity", "economic_complexity", "General economic-complexity context for the node or country-sector environment.", atlas_panel, state_panel, "country_sector-year", "annual", "medium", "high", "medium", "high", "context", "Broader productive sophistication control.", "May be country-level rather than country-sector-level.", "medium"),
        semantic_row("network_green_exposure", "network_green_exposure; network_green_ness", "network_green_ness", "Exposure to green-ness through the production network around the node.", phase_panel, diagnostics, "country_sector-year", "annual", "high", "high", "high", "high", "z_axis", "Network-embedded green-ness, distinct from local green-ness.", "Missing if no validated network green-ness layer exists.", "high"),
        semantic_row("g_in_network", "g_in_network; incoming_green", "network_green_ness", "Input-side or upstream network green-ness exposure.", diagnostics, phase_panel, "country_sector-year", "annual", "high", "high", "medium", "high", "z_axis", "Captures green-ness of suppliers or incoming production links.", "Direction depends on validated input-output orientation.", "medium"),
        semantic_row("g_out_network", "g_out_network; outgoing_green", "network_green_ness", "Output-side or downstream network green-ness exposure.", diagnostics, phase_panel, "country_sector-year", "annual", "high", "high", "medium", "high", "z_axis", "Captures green-ness of buyers or downstream production links.", "Direction depends on validated input-output orientation.", "medium"),
        semantic_row("recursive_green", "recursive_green; recursive_green_ness", "network_green_ness", "Recursive network measure of green exposure through indirect links.", diagnostics, phase_panel, "country_sector-year", "annual", "high", "high", "medium", "high", "z_axis", "Measures embedded green position in the network, not local emissions intensity.", "Requires method notes on recursion and normalization.", "medium"),
        semantic_row("pagerank", "pagerank; PageRank", "centrality", "Directed network centrality of the country-sector node.", diagnostics, phase_panel, "country_sector-year", "annual", "medium", "high", "medium", "high", "context", "Identifies influential production-network positions.", "Centrality alone is not brown lock-in.", "medium"),
        semantic_row("eigenvector_centrality", "eigenvector_centrality", "centrality", "Centrality based on connections to other central nodes.", diagnostics, phase_panel, "country_sector-year", "annual", "medium", "high", "medium", "high", "context", "Captures embedded influence in the production network.", "Depends on graph construction and orientation.", "medium"),
        semantic_row("reverse_eigenvector_centrality", "reverse_eigenvector_centrality", "centrality", "Eigenvector centrality computed on the reversed network orientation.", diagnostics, phase_panel, "country_sector-year", "annual", "medium", "high", "medium", "high", "context", "Separates upstream and downstream embeddedness.", "Interpret only with explicit orientation notes.", "medium"),
        semantic_row("brown_centrality", "brown_centrality; embodied_carbon_centrality", "brown_centrality", "Centrality weighted by brown or carbon-intensive network exposure.", diagnostics, phase_panel, "country_sector-year", "annual", "high", "high", "medium", "high", "x_axis", "Proxy for brown lock-in in the network, distinct from own EI.", "Missing unless validated brown-centrality diagnostics exist.", "high"),
        semantic_row("K", "K; capacity", "capacity", "Capacity proxy used by behavioural production propagation.", state_panel, diagnostics, "country_sector-year", "annual", "medium", "medium", "low", "high", "context", "Current production capacity proxy, not adaptive transition capacity.", "Do not interpret as investment capacity.", "medium"),
        semantic_row("capacity_stress", "capacity_stress; capacity_utilization", "capacity", "Degree to which realized or desired production approaches capacity.", phase_panel, diagnostics, "country_sector-year", "annual", "medium", "medium", "low", "medium", "diagnostic", "Shows production constraint pressure around transition candidates.", "May be derived and model-specific.", "medium"),
        semantic_row("capacity_binding", "capacity_binding; binding_capacity", "capacity", "Indicator that capacity constraints bind in propagation.", diagnostics, phase_panel, "country_sector-year", "annual", "low", "medium", "low", "medium", "diagnostic", "Flags constrained production response, not green-transition success.", "Diagnostic only.", "low"),
        semantic_row("capacity_to_observed_ratio", "capacity_to_observed_ratio", "capacity", "Ratio of capacity proxy to observed production.", phase_panel, state_panel, "country_sector-year", "annual", "medium", "medium", "low", "high", "context", "Identifies output scale relative to current capacity proxy.", "Derived only when K and X_observed are available.", "medium"),
        semantic_row("rEI", "rEI; EI_reduction; reduction_log_EI", "ei_transition", "Year-to-year reduction in log emissions intensity.", phase_panel, ei_panel, "country_sector-year transition", "annual transition", "high", "high", "high", "high", "vector_component", "Positive values indicate declining emissions intensity.", "Computed as log_EI - log_EI_next; last year has no next-year value.", "high"),
        semantic_row("predicted_rEI", "predicted_rEI; predicted_delta_log_EI", "ei_transition", "Predicted EI reduction from the EI transition learning model.", ei_predictions, ei_panel, "country_sector-year transition", "annual transition", "medium", "medium", "high", "high", "vector_prediction", "Model-predicted movement, not observed transition.", "Use only as diagnostic or vector-field prediction layer.", "medium"),
        semantic_row("delta_log_EI", "delta_log_EI", "ei_transition", "Forward change in log emissions intensity.", phase_panel, ei_panel, "country_sector-year transition", "annual transition", "high", "high", "high", "high", "vector_component", "Negative values indicate decreasing emissions intensity.", "Sign convention differs from rEI.", "high"),
        semantic_row("delta_g_local", "delta_g_local", "ei_transition", "Forward change in local green-ness.", phase_panel, ei_panel, "country_sector-year transition", "annual transition", "high", "high", "high", "high", "vector_component", "Positive values indicate local greening under the g_local proxy.", "Only valid when current and next-year g_local are available.", "high"),
        semantic_row("delta_network_green_exposure", "delta_network_green_exposure", "ei_transition", "Forward change in network green exposure.", phase_panel, diagnostics, "country_sector-year transition", "annual transition", "high", "high", "high", "high", "vector_component", "Shows movement in network-embedded green-ness.", "Missing until network exposure is present in the state panel.", "high"),
        semantic_row("scenario_name", "scenario_name; scenario", "scenario_response", "Registered scenario identifier for perturbation outputs.", scenario_outputs, "", "country_sector-scenario-year", "annual or scenario run", "low", "medium", "low", "high", "overlay_facet", "Labels production perturbation experiments.", "Not part of baseline historical trajectory.", "medium"),
        semantic_row("selector_name", "selector_name; selector", "scenario_response", "Scenario selector or transition archetype used to choose shocked nodes.", scenario_outputs, "", "country_sector-scenario-year", "annual or scenario run", "low", "medium", "low", "high", "overlay_facet", "Explains why nodes were included in a perturbation.", "Selector membership is not observed transition evidence.", "medium"),
        semantic_row("shock_size", "shock_size", "scenario_response", "Magnitude of imposed scenario perturbation.", scenario_outputs, "", "scenario-year", "annual or scenario run", "low", "medium", "low", "high", "overlay_parameter", "Parameterizes perturbation strength.", "Do not compare to historical greening without scenario notes.", "medium"),
        semantic_row("delta_X_realized", "delta_X_realized; delta_X", "scenario_response", "Change in model-realized production under a scenario.", scenario_outputs, diagnostics, "country_sector-scenario-year", "annual or scenario run", "low", "medium", "low", "high", "overlay_response", "Production response layer, not full transition dynamics.", "Keep separate from observed X_observed movement.", "medium"),
        semantic_row("pct_delta_X_realized", "pct_delta_X_realized; pct_delta_X", "scenario_response", "Percentage change in realized production under a scenario.", scenario_outputs, diagnostics, "country_sector-scenario-year", "annual or scenario run", "low", "medium", "low", "high", "overlay_response", "Scale-normalized production perturbation response.", "Undefined or unstable for near-zero baselines.", "medium"),
        semantic_row("selected_node_flags", "selected_node; is_selected; selector_flag", "scenario_response", "Boolean flags identifying scenario-selected nodes.", scenario_outputs, diagnostics, "country_sector-scenario-year", "annual or scenario run", "low", "medium", "low", "medium", "overlay_filter", "Marks transition archetypes for scenario overlays.", "Not a historical state variable.", "low"),
        semantic_row("converged", "converged", "validation", "Whether a propagation or validation routine converged.", diagnostics, scenario_outputs, "year or country_sector-year", "annual", "low", "low", "low", "high", "diagnostic", "Supports model trust and caveats.", "Not an economic state coordinate.", "medium"),
        semantic_row("rounds_used", "rounds_used; rounds", "validation", "Number of propagation rounds used before stopping.", diagnostics, scenario_outputs, "year or country_sector-year", "annual", "low", "low", "low", "medium", "diagnostic", "Shows computational difficulty of propagation.", "Not an economic state coordinate.", "low"),
        semantic_row("final_residual_share", "final_residual_share; residual_share", "validation", "Remaining residual as a share of the target quantity.", diagnostics, scenario_outputs, "year or country_sector-year", "annual", "low", "low", "low", "high", "diagnostic", "Quantifies unresolved propagation mismatch.", "Use only in validation panels.", "medium"),
        semantic_row("output_validation_loss", "output_validation_loss; validation_loss", "validation", "Loss metric comparing model output to observed output.", diagnostics, "", "year or validation split", "annual or split", "low", "low", "low", "high", "diagnostic", "Summarizes historical reproduction quality.", "Metric definition must be reported.", "medium"),
        semantic_row("absolute_percentage_error", "absolute_percentage_error; APE", "validation", "Absolute percentage error between modeled and observed production.", diagnostics, "", "country_sector-year", "annual", "low", "low", "low", "high", "diagnostic", "Node-level validation error for production reproduction.", "Can be unstable for very small observed output.", "medium"),
    ]
    return rows


def semantic_row(
    canonical_variable: str,
    candidate_columns: str,
    semantic_category: str,
    economic_meaning: str,
    preferred_source: str,
    fallback_sources: str,
    unit_of_observation: str,
    time_grain: str,
    usable_for_phase_space: str,
    usable_for_trajectory: str,
    usable_for_vector_field: str,
    usable_for_latex_reference: str,
    suggested_axis_role: str,
    green_transition_interpretation: str,
    caveats: str,
    priority: str,
) -> dict[str, str]:
    """Create one semantic variable-map row."""
    return {
        "canonical_variable": canonical_variable,
        "candidate_columns": candidate_columns,
        "semantic_category": semantic_category,
        "economic_meaning": economic_meaning,
        "preferred_source": preferred_source,
        "fallback_sources": fallback_sources,
        "unit_of_observation": unit_of_observation,
        "time_grain": time_grain,
        "usable_for_phase_space": usable_for_phase_space,
        "usable_for_trajectory": usable_for_trajectory,
        "usable_for_vector_field": usable_for_vector_field,
        "usable_for_latex_reference": usable_for_latex_reference,
        "suggested_axis_role": suggested_axis_role,
        "green_transition_interpretation": green_transition_interpretation,
        "caveats": caveats,
        "priority": priority,
    }


def build_visual_use_map_rows() -> list[dict[str, str]]:
    """Define the visual family map used by the catalog."""
    rows = [
        visual_row(
            "Phase-space trajectories: Global trajectory",
            "How does the global production system move through greenness, capability, and network exposure over time?",
            "data/abm_v3/phase_space/abm_v3_phase_space_state_panel_1995_2016.parquet",
            "Year; country_sector; X_observed; EI; green_capability_export_share; local_green_ness/network_green_ness if available",
            "emissions_observed; centrality; brown_centrality",
            "global aggregate by year",
            "global weighted mean/sum",
            "X_observed or emissions_observed",
            "high",
            "high",
            "medium",
            "high",
            "Use weighted aggregation so small nodes do not dominate the global movement.",
        ),
        visual_row(
            "Phase-space trajectories: Sector trajectories",
            "Which sectors move toward higher greenness/capability and which remain locked in?",
            "data/abm_v3/phase_space/abm_v3_phase_space_state_panel_1995_2016.parquet",
            "Year; Sector; X_observed; EI; green/network variables",
            "green_capability; emissions_observed",
            "sector-year",
            "sector weighted mean by year",
            "X_observed",
            "high",
            "high",
            "medium",
            "high",
            "Preserve sector labels and report weighting explicitly.",
        ),
        visual_row(
            "Phase-space trajectories: Country-sector node trajectories: top 25 by output",
            "Which high-output country-sector nodes define the observed state-space movement?",
            "data/abm_v3/phase_space/abm_v3_phase_space_state_panel_1995_2016.parquet",
            "Year; country_sector; X_observed; EI; green/network/capability variables",
            "Country; Sector; emissions_observed",
            "country_sector-year",
            "node panel",
            "X_observed",
            "high",
            "high",
            "low",
            "high",
            "Rank nodes by total or baseline X_observed before plotting.",
        ),
        visual_row(
            "Phase-space trajectories: Country-sector node trajectories: top 25 by emissions",
            "Which high-emission nodes drive carbon-relevant movement?",
            "data/abm_v3/phase_space/abm_v3_phase_space_state_panel_1995_2016.parquet",
            "Year; country_sector; emissions_observed or X_observed and EI; green/network/capability variables",
            "Sector; Country",
            "country_sector-year",
            "node panel",
            "emissions_observed",
            "high",
            "high",
            "low",
            "high",
            "If emissions are absent, compute only from inspected source variables in a later phase-space panel.",
        ),
        visual_row(
            "3D phase-space cubes: Green Transition Readiness Cube",
            "Where are nodes with green capability, improving local greenness, and supportive network exposure?",
            "data/abm_v3/phase_space/abm_v3_phase_space_state_panel_1995_2016.parquet",
            "green_capability; local_green_ness; network_green_ness",
            "X_observed; EI; emissions_observed",
            "country_sector-year",
            "node panel or weighted aggregate",
            "X_observed",
            "high",
            "high",
            "medium",
            "high",
            "X = green capability; Y = local greenness, green up; Z = network greenness/exposure.",
        ),
        visual_row(
            "3D phase-space cubes: Brown Lock-in Cube",
            "Which nodes combine carbon-network lock-in with weak greenness or capability?",
            "data/abm_v3/phase_space/abm_v3_phase_space_state_panel_1995_2016.parquet",
            "brown_centrality; local_green_ness; green_capability",
            "emissions_observed; EI; X_observed",
            "country_sector-year",
            "node panel",
            "emissions_observed",
            "high",
            "high",
            "medium",
            "medium",
            "X = brown centrality; Y = local greenness, green up; Z = green capability.",
        ),
        visual_row(
            "3D phase-space cubes: Productive Ecosystem Transition Cube",
            "Where do green capability and ecosystem proximity suggest transition readiness?",
            "data/abm_v3/phase_space/abm_v3_phase_space_state_panel_1995_2016.parquet",
            "green_capability; ecosystem_proximity/capability_readiness; local_green_ness",
            "X_observed; Sector; Country",
            "country_sector-year",
            "node panel",
            "X_observed",
            "high",
            "high",
            "low",
            "medium",
            "X = green capability; Y = ecosystem proximity/readiness; Z = local greenness.",
        ),
        visual_row(
            "3D phase-space cubes: Production-Safe Greening Cube",
            "Can greening be read alongside production scale rather than apart from it?",
            "data/abm_v3/phase_space/abm_v3_phase_space_state_panel_1995_2016.parquet",
            "X_observed or log_output; local_green_ness; network_green_ness",
            "green_capability; EI; emissions_observed",
            "country_sector-year",
            "node panel or sector aggregate",
            "X_observed",
            "high",
            "high",
            "medium",
            "high",
            "X = production scale/log output; Y = local greenness, green up; Z = network greenness.",
        ),
        visual_row(
            "3D phase-space cubes: Scenario Perturbation Cube",
            "How do scenario shocks move nodes from their baseline positions?",
            "data/abm_v3/leontief/behavioural/scenarios/analysis_report/",
            "baseline state-space position; scenario output response; selector type/transition archetype",
            "shock_size; scenario_name; delta_X; pct_delta",
            "country_sector-scenario-year",
            "scenario node panel",
            "X_observed or emissions_observed",
            "high",
            "medium",
            "high",
            "medium",
            "Scenario response is a perturbation layer, not yet the full green transition dynamics.",
        ),
        visual_row(
            "Vector-field plots: average movement in binned state space",
            "What is the typical direction of movement by region of state space?",
            "data/abm_v3/phase_space/abm_v3_phase_space_state_panel_1995_2016.parquet",
            "state variables at t and t+1; Year; country_sector",
            "X_observed weights; emissions weights",
            "country_sector-year movement",
            "binned state space",
            "X_observed",
            "medium",
            "high",
            "high",
            "medium",
            "Build after the phase-space state panel exists.",
        ),
        visual_row(
            "Vector-field plots: local green-ness movement conditional on capability and network exposure",
            "Does local greenness improve where capability and network exposure are high?",
            "data/abm_v3/phase_space/abm_v3_phase_space_state_panel_1995_2016.parquet",
            "green_capability; network_green_ness; local_green_ness_delta",
            "EI_delta; output weights",
            "country_sector-year movement",
            "binned capability-network space",
            "X_observed",
            "medium",
            "high",
            "high",
            "medium",
            "Keep local greenness and network greenness conceptually separate.",
        ),
        visual_row(
            "Vector-field plots: EI reduction vector fields",
            "Where does emissions intensity fall, and which predictors align with that movement?",
            "data/abm_v3/ei_transition/",
            "EI_t; EI_t_plus_1 or EI_delta; green_capability; network_green_ness",
            "model predictions; residuals",
            "country_sector-year transition",
            "binned state space",
            "X_observed or emissions_observed",
            "medium",
            "high",
            "high",
            "medium",
            "Treat EI transition outputs as historical-learning diagnostics.",
        ),
        visual_row(
            "Scenario plots: scenario output effect comparison",
            "Which scenarios cause the largest production responses?",
            "data/abm_v3/leontief/behavioural/scenarios/analysis_report/",
            "scenario_name; X_realized/delta_X/pct_delta; Year",
            "Sector; Country; selector",
            "scenario-year or node-scenario-year",
            "scenario aggregate",
            "X_observed",
            "high",
            "medium",
            "high",
            "high",
            "Compare responses without presenting them as completed transition simulations.",
        ),
        visual_row(
            "Scenario plots: selector overlap",
            "How do policy/scenario selectors overlap across nodes?",
            "scenario analysis report outputs",
            "selector; country_sector; scenario_name",
            "green_capability; EI; Sector",
            "country_sector-scenario",
            "selector sets",
            "X_observed",
            "medium",
            "medium",
            "high",
            "medium",
            "Useful for explaining transition archetypes.",
        ),
        visual_row(
            "Scenario plots: sector propagation",
            "Which sectors transmit or receive scenario impacts?",
            "behavioural Leontief scenario outputs",
            "Sector; scenario_name; X_realized/delta_X; Year",
            "input-output centrality",
            "sector-scenario-year",
            "sector aggregate",
            "X_observed",
            "high",
            "medium",
            "high",
            "medium",
            "Use sector labels from the node panel to keep interpretation stable.",
        ),
        visual_row(
            "Scenario plots: country propagation",
            "Which countries transmit or receive scenario impacts?",
            "behavioural Leontief scenario outputs",
            "Country; scenario_name; X_realized/delta_X; Year",
            "emissions_observed; EI",
            "country-scenario-year",
            "country aggregate",
            "X_observed",
            "high",
            "medium",
            "high",
            "medium",
            "Aggregation should state whether country totals are output- or emissions-weighted.",
        ),
        visual_row(
            "Validation plots: historical reproduction",
            "How closely does behavioural Leontief reproduce observed production?",
            "data/abm_v3/validation_report/ and behavioural diagnostics",
            "Year; X_observed; X_realized; error metrics",
            "Sector; Country",
            "year or country_sector-year",
            "validation aggregate and node panel",
            "X_observed",
            "medium",
            "high",
            "high",
            "high",
            "Show production validation separately from green-transition claims.",
        ),
        visual_row(
            "Validation plots: residual/convergence diagnostics",
            "Where does the propagation engine struggle to converge or reproduce observed output?",
            "behavioural Leontief diagnostics and validation reports",
            "residual; converged; rounds; relative_error_total",
            "capacity stress metrics",
            "year or node-year",
            "diagnostic panel",
            "X_observed",
            "low",
            "high",
            "high",
            "medium",
            "Best for methods appendix and model trust-building.",
        ),
        visual_row(
            "Validation plots: capacity bottleneck diagnostics",
            "Where do capacity proxies constrain production response?",
            "data/abm_v3/diagnostics/production_bottleneck_*",
            "capacity; capacity_utilization; bottleneck counts/ratios",
            "Sector; Country; Year",
            "node/year/sector/country",
            "diagnostic aggregate",
            "X_observed",
            "medium",
            "high",
            "high",
            "medium",
            "Capacity proxies are not adaptive capacity; label them cautiously.",
        ),
    ]
    return rows


def visual_row(
    visual_family: str,
    visual_question: str,
    recommended_data_source: str,
    required_variables: str,
    optional_variables: str,
    unit_of_observation: str,
    aggregation_level: str,
    recommended_weight: str,
    portfolio_suitability: str,
    research_suitability: str,
    diagnostic_suitability: str,
    implementation_priority: str,
    notes: str,
) -> dict[str, str]:
    """Create one visual use map row."""
    return {
        "visual_family": visual_family,
        "visual_question": visual_question,
        "recommended_data_source": recommended_data_source,
        "required_variables": required_variables,
        "optional_variables": optional_variables,
        "unit_of_observation": unit_of_observation,
        "aggregation_level": aggregation_level,
        "recommended_weight": recommended_weight,
        "portfolio_suitability": portfolio_suitability,
        "research_suitability": research_suitability,
        "diagnostic_suitability": diagnostic_suitability,
        "implementation_priority": implementation_priority,
        "notes": notes,
    }


def build_markdown_catalog(inventory: pd.DataFrame, variables: pd.DataFrame, visual_map: pd.DataFrame) -> str:
    """Build the readable Markdown catalog."""
    state_sources = inventory.loc[inventory["status"].eq("authoritative_state_source"), "relative_path"].tolist()
    model_outputs = inventory.loc[inventory["status"].eq("authoritative_model_output"), "relative_path"].tolist()
    diagnostics = inventory.loc[inventory["status"].eq("current_diagnostic"), "relative_path"].tolist()
    scenario_outputs = inventory.loc[inventory["status"].eq("current_scenario_output"), "relative_path"].tolist()
    legacy = inventory.loc[inventory["status"].eq("legacy"), "relative_path"].tolist()
    source_data = inventory.loc[inventory["status"].isin(["raw_source", "processed_source"]), "relative_path"].tolist()
    phase_sources = inventory.loc[inventory["phase_space_use"].astype(str).ne(""), ["relative_path", "phase_space_use"]]

    variable_sections = build_variable_family_sections(variables)
    cube_sections = build_cube_sections(visual_map)
    quality_notes = build_quality_notes(inventory)

    return "\n".join(
        [
            "# ABM v3 Data Catalog and Visual Use Map",
            "",
            "## Purpose",
            "This catalog maps the current data folder to modelling, validation, scenario analysis, and phase-space visualisation needs. It is built from file metadata, tabular schemas, and small samples, not from expensive model simulation or full-data loading. The stable model unit is `country_sector` whenever that key is available.",
            "",
            "## High-Level Data Architecture",
            "- Raw Eora data are source inputs for production, final demand, input-output structure, emissions, and country-sector node construction. They are source data, not direct plotting panels.",
            "- Processed Eora parquet data are cleaned input-output and analytical structures that can support model inputs after schema validation.",
            "- Atlas raw and processed data provide productive capabilities, green capability, complexity, and product-space or ecosystem-readiness variables.",
            "- ABM v3 historical input panels are the main country-sector-year state sources for current calibration and visual state construction.",
            "- Behavioural Leontief outputs describe production propagation, realized output, convergence, residuals, and validation diagnostics.",
            "- EI transition outputs are historical-learning and diagnostic layers for emissions-intensity movement. They are not full scenario dynamics unless later integrated explicitly.",
            "- Scenario outputs and scenario analysis reports describe perturbation responses under registered behavioural Leontief scenarios.",
            "- Validation reports summarize historical reproduction, residuals, convergence, and production-safe model checks.",
            "- Legacy ABM outputs under `data/abm/` remain useful for comparison, but should not be treated as ABM v3 canonical sources without validation.",
            "- Plot outputs under `outputs/plots/` are visual outputs, not data sources.",
            "",
            "## Authoritative State Sources",
            format_path_list(state_sources, limit=20, empty_text="No authoritative state source was detected by the current path and schema heuristics."),
            "",
            "These files can define historical `country_sector` x `Year` state variables after schema validation. The main baseline source is the corrected ABM v3 historical input panel.",
            "",
            "## Current Model Outputs",
            format_path_list(model_outputs, limit=30, empty_text="No authoritative model-output files were detected."),
            "",
            "Behavioural Leontief node-comparison and propagation files are current validated model outputs or diagnostics. They should not be treated as the primary historical state panel.",
            "",
            "## Current Diagnostics",
            format_path_list(diagnostics, limit=40, empty_text="No current diagnostic sources were detected."),
            "",
            "## Current Scenario Outputs",
            format_path_list(scenario_outputs, limit=40, empty_text="No current scenario outputs were detected."),
            "",
            "Scenario outputs support perturbation overlays and response comparisons. They are not baseline historical phase-space trajectories.",
            "",
            "## Legacy Sources",
            format_path_list(legacy, limit=30, empty_text="No legacy sources were detected."),
            "",
            "Legacy files under `data/abm/` are comparison material only unless a later validation step explicitly promotes them.",
            "",
            "## Raw and Processed Source Data",
            format_path_list(source_data, limit=40, empty_text="No raw or processed source data were detected in this inventory run."),
            "",
            "Raw and processed source files feed state construction only after explicit inspection and validation. The full raw inventory remains available in `data_inventory.csv`.",
            "",
            "## Semantic Variable Map",
            "The curated semantic variable map is written to `abm_v3_semantic_variable_map.csv`. It is intentionally compact and maps economically meaningful variables or variable families to preferred sources, interpretation notes, phase-space roles, and LaTeX-ready caveats.",
            "",
            "## Phase-Space State Panel Readiness",
            "The canonical phase-space state panel should be built with `python -m src.abm_v3.runner phase-space-state-panel --start-year 1995 --end-year 2016`. Once built, phase-space visuals should use `data/abm_v3/phase_space/abm_v3_phase_space_state_panel_1995_2016.parquet` rather than broad diagnostic inventories.",
            "",
            "## Variable Families and Economic Meaning",
            variable_sections,
            "",
            "## Phase-Space Plotting Opportunities",
            cube_sections,
            "",
            "Candidate phase-space source files:",
            format_table_preview(phase_sources, limit=30),
            "",
            "## Trajectory Plotting Opportunities",
            "- Global trajectories: aggregate country-sector-year variables by year with explicit production or emissions weights.",
            "- Sector trajectories: aggregate by `Sector` and `Year`, preserving sector names and weights.",
            "- Country-sector node trajectories: use `country_sector` and `Year` as trace keys. Prioritize the top 25 nodes by output and the top 25 by emissions.",
            "- Five-year markers: add markers every five years to make direction and timing readable.",
            "- Vector fields: compute average movement from a consolidated phase-space panel by binning state-space coordinates and averaging observed movement.",
            "",
            "## Scenario and Validation Plotting Opportunities",
            "Scenario outputs can support output-effect comparisons, selector overlap, sector propagation, country propagation, and scenario perturbation overlays. Validation reports and behavioural diagnostics can support historical reproduction plots, residual and convergence diagnostics, and capacity-bottleneck diagnostics. Scenario production response should be labelled as a perturbation analysis, not as a complete green-transition simulation.",
            "",
            "## Data Quality and Missingness Caveats",
            quality_notes,
            "",
            "## Recommended Next Steps",
            "1. Build a consolidated phase-space state panel with `country_sector`, `Year`, production scale, emissions intensity, emissions, local greenness, network greenness, green capability, and relevant centrality variables.",
            "2. Implement 3D trajectory plots for the five recommended phase-space cubes.",
            "3. Implement vector-field diagnostics from binned year-to-year movement.",
            "4. Add scenario perturbation overlays after the baseline state-space panel is stable.",
            "5. Create a LaTeX methodological data reference from this catalog.",
            "",
        ]
    )


def build_variable_family_sections(variables: pd.DataFrame) -> str:
    """Build Markdown sections summarizing variables by semantic family."""
    if variables.empty:
        return "No tabular variables were inspected."
    sections: list[str] = []
    for semantic_category in sorted(variables["semantic_category"].dropna().unique()):
        subset = variables.loc[variables["semantic_category"].eq(semantic_category)]
        candidate_columns = sorted(subset["variable"].dropna().astype(str).unique())[:30]
        candidate_files = sorted(subset["path"].dropna().astype(str).unique())[:10]
        meaning = economic_meaning_short(semantic_category)
        visual_use = variable_family_visual_use(semantic_category)
        sections.extend(
            [
                f"### {semantic_category}",
                f"- Economic meaning: {meaning}",
                f"- Candidate columns found: {', '.join(candidate_columns) if candidate_columns else 'none'}",
                f"- Candidate files: {', '.join(candidate_files) if candidate_files else 'none'}",
                f"- Visualisation use: {visual_use}",
                "",
            ]
        )
    return "\n".join(sections)


def variable_family_visual_use(semantic_category: str) -> str:
    """Describe how a semantic family can be visualized."""
    uses = {
        "identifier": "trace keys, facets, labels, and unit-of-observation checks",
        "time": "trajectory ordering, year markers, and historical movement",
        "production": "size/weight, production-safe greening axes, validation against realized output",
        "final_demand": "demand-side context for input-output propagation",
        "input_output_structure": "network structure and propagation diagnostics",
        "emissions": "top-emitter selection, color/size encoding, carbon scale caveats",
        "emissions_intensity": "EI transition vectors and local carbon-intensity axes",
        "local_green_ness": "local greenness axis with green-up orientation",
        "network_green_ness": "network exposure axis distinct from local greenness",
        "green_capability": "transition readiness axis and Atlas join variable",
        "economic_complexity": "ecosystem/product-space readiness axis",
        "centrality": "network influence and propagation context",
        "brown_centrality": "brown lock-in axis and carbon-network exposure",
        "capacity": "production bottleneck and constraint diagnostics",
        "scenario": "scenario facets, shock overlays, and response comparisons",
        "validation": "residual, convergence, historical reproduction, and uncertainty views",
        "diagnostic": "quality-control annotations and warning filters",
        "metadata": "source tracking and catalog traceability",
        "unknown": "requires manual inspection before visualization",
    }
    return uses.get(semantic_category, "requires manual inspection before visualization")


def build_cube_sections(visual_map: pd.DataFrame) -> str:
    """Build Markdown for the required 3D phase-space cubes."""
    cube_rows = visual_map.loc[visual_map["visual_family"].str.startswith("3D phase-space cubes", na=False)]
    sections: list[str] = []
    for _, row in cube_rows.iterrows():
        family = str(row["visual_family"]).replace("3D phase-space cubes: ", "")
        sections.extend(
            [
                f"### {family}",
                f"- Theoretical question: {row['visual_question']}",
                f"- Axis definitions / required variables: {row['required_variables']}",
                f"- Candidate data source: {row['recommended_data_source']}",
                f"- Preferred unit of movement: {row['unit_of_observation']}",
                f"- Interpretation: {row['notes']}",
                "- Caveat: preserve the distinction between production scale, emissions, emissions intensity, local greenness, network greenness, and capability.",
                "",
            ]
        )
    return "\n".join(sections)


def build_quality_notes(inventory: pd.DataFrame) -> str:
    """Build Markdown notes about uncertainty and data quality."""
    if inventory.empty:
        return "No files were inspected."
    missing_country_sector = int((~inventory["contains_country_sector"].astype(bool)).sum())
    missing_year = int((~inventory["contains_year"].astype(bool)).sum())
    large_skipped = inventory.loc[inventory["notes"].astype(str).str.contains("skipped|metadata only|unavailable", case=False, na=False)]
    unclear = inventory.loc[inventory["status"].eq("unclear"), "relative_path"].tolist()
    duplicated_names = inventory.loc[inventory["file_name"].duplicated(keep=False), "file_name"].dropna().unique().tolist()
    notes = [
        f"- Files inspected: {len(inventory)}.",
        f"- Files without detected `country_sector`: {missing_country_sector}.",
        f"- Files without detected year/time variable: {missing_year}.",
        f"- Files with metadata-only or skipped row-count notes: {len(large_skipped)}.",
        f"- Duplicate file names detected across directories: {', '.join(sorted(duplicated_names)[:20]) if duplicated_names else 'none detected'}.",
        f"- Unclear-status files: {', '.join(unclear[:20]) if unclear else 'none detected'}.",
        "- Large files were not fully loaded; parquet row counts came from metadata where available and CSV row counts were skipped above the cheap-inspection threshold.",
        "- Avoid current phase-space plotting directly from raw Eora, raw Atlas, generated plot files, and legacy `data/abm/` outputs unless a later validation step explicitly promotes them.",
    ]
    return "\n".join(notes)


def safe_relative_path(path: Path, base_root: Path) -> str:
    """Return a stable relative path when possible."""
    try:
        return path.resolve().relative_to(base_root.resolve()).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(base_root.resolve().parent).as_posix()
        except ValueError:
            return path.as_posix()


def join_preview(values: list[str], limit: int) -> str:
    """Join a bounded list preview."""
    preview = values[:limit]
    suffix = " ..." if len(values) > limit else ""
    return "; ".join(preview) + suffix


def format_path_list(paths: list[str], limit: int | None = None, empty_text: str = "None detected.") -> str:
    """Format a bounded Markdown path list."""
    if not paths:
        return empty_text
    active_paths = paths[:limit] if limit is not None else paths
    lines = [f"- `{path}`" for path in active_paths]
    if limit is not None and len(paths) > limit:
        lines.append(f"- ... {len(paths) - limit} additional paths omitted from this readable catalog.")
    return "\n".join(lines)


def format_table_preview(frame: pd.DataFrame, limit: int = 20) -> str:
    """Format a compact Markdown table preview."""
    if frame.empty:
        return "No candidate files were detected."
    preview = frame.head(limit).copy()
    lines = ["| " + " | ".join(preview.columns) + " |", "| " + " | ".join(["---"] * len(preview.columns)) + " |"]
    for _, row in preview.iterrows():
        values = [str(row[column]).replace("\n", " ") for column in preview.columns]
        lines.append("| " + " | ".join(values) + " |")
    if len(frame) > limit:
        lines.append(f"\n{len(frame) - limit} additional rows omitted from this readable preview.")
    return "\n".join(lines)
