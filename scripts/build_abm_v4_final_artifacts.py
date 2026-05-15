from __future__ import annotations

import argparse

from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.reporting import ABMV4FinalArtifactBuilder


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 29A final artifact CLI."""
    parser = argparse.ArgumentParser(description="Build ABM v4 final plots, tables, and artifact index.")
    parser.add_argument(
        "--create-output-dirs",
        action="store_true",
        help="Create and write data/abm_v4/final and outputs/plots/abm_v4_final artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate Phase 28 inputs without creating outputs.",
    )
    return parser


def main() -> None:
    """Build Phase 29A final ABM v4 artifacts."""
    args = build_parser().parse_args()
    builder = ABMV4FinalArtifactBuilder(ABMV4Paths())
    if args.dry_run:
        builder.run(write_outputs=False)
        print("ABM v4 final artifact inputs are available. Dry-run created no outputs.")
        return
    if not args.create_output_dirs:
        raise SystemExit("--create-output-dirs is required to write final ABM v4 artifacts.")
    result = builder.run(write_outputs=True)
    print("Built ABM v4 final plots and tables.")
    print(f"Final tables: {len(result.table_paths)}")
    print(f"Final plot files: {len(result.plot_paths)}")
    print(f"Portfolio plot copies: {len(result.copied_plot_paths)}")
    print(f"Artifact index: {result.artifact_index_path}")


if __name__ == "__main__":
    main()
