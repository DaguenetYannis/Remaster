r"""Build a JSON inventory of the data directory.

This script reproduces the useful part of the PowerShell command:

    Get-ChildItem .\data\ -Recurse | ConvertTo-Json -Depth 3

The output is a flat list of files and folders with explicit filesystem
metadata. It is intended for inspection only; it does not transform raw data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_PATH = DEFAULT_INPUT_DIR / "data_tree.json"


def build_item_record(path: Path) -> dict[str, Any]:
    """Convert one filesystem path into a PowerShell-like JSON record."""
    path_stat = path.stat()
    is_directory = path.is_dir()
    last_write_time_ms = int(path_stat.st_mtime * 1000)

    return {
        "FullName": str(path.resolve()),
        "Name": path.name,
        "Extension": path.suffix,
        "Length": None if is_directory else path_stat.st_size,
        "LastWriteTime": f"/Date({last_write_time_ms})/",
        "PSIsContainer": is_directory,
    }


def iter_directory_tree(input_dir: Path, excluded_path: Path | None) -> list[Path]:
    """Return all children under input_dir in a stable, inspectable order."""
    discovered_paths: list[Path] = []
    resolved_excluded_path = excluded_path.resolve() if excluded_path else None

    def collect_children(current_dir: Path) -> None:
        child_paths = sorted(
            current_dir.iterdir(),
            key=lambda child_path: (not child_path.is_dir(), child_path.name.lower()),
        )

        for child_path in child_paths:
            if resolved_excluded_path and child_path.resolve() == resolved_excluded_path:
                print(f"Skipped output file during scan: {child_path}")
                continue

            discovered_paths.append(child_path)
            if child_path.is_dir():
                collect_children(child_path)

    collect_children(input_dir)
    return discovered_paths


def build_data_tree(
    input_dir: Path,
    excluded_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Build JSON-serializable metadata records for input_dir contents."""
    if not input_dir.exists():
        print(f"WARNING: input directory does not exist: {input_dir}")
        return []

    if not input_dir.is_dir():
        print(f"WARNING: input path is not a directory: {input_dir}")
        return []

    tree_paths = iter_directory_tree(input_dir=input_dir, excluded_path=excluded_path)
    tree_records = [build_item_record(path) for path in tree_paths]

    directory_count = sum(1 for path in tree_paths if path.is_dir())
    file_count = sum(1 for path in tree_paths if path.is_file())

    print(f"Input directory: {input_dir}")
    print(f"Discovered directories: {directory_count}")
    print(f"Discovered files: {file_count}")
    print(f"Total records: {len(tree_records)}")

    return tree_records


def write_data_tree(tree_records: list[dict[str, Any]], output_path: Path) -> None:
    """Write tree records to JSON with readable indentation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(tree_records, output_file, indent=4)
        output_file.write("\n")

    print(f"Wrote data tree JSON: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build a JSON inventory of the data directory."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory to inspect. Defaults to the repository data directory.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="JSON file to write. Defaults to data/data_tree.json.",
    )
    return parser.parse_args()


def main() -> None:
    """Build and write the data tree JSON."""
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_path = args.output_path.resolve()

    tree_records = build_data_tree(input_dir=input_dir, excluded_path=output_path)
    write_data_tree(tree_records=tree_records, output_path=output_path)


if __name__ == "__main__":
    main()
