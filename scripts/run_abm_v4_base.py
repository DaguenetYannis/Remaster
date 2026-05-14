from __future__ import annotations

import argparse

import polars as pl

from src.abm_v4.config import ABMV4Config
from src.abm_v4.diagnostics import build_path_audit_rows, format_path_audit_table
from src.abm_v4.ecosystem import EcosystemMapper
from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.simulation import inspect_base_model_readiness
from src.abm_v4.state import build_state_panel
from src.abm_v4.suppliers import SupplierNetworkBuilder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ABM v4 base readiness and dry-run audit entry point."
    )
    parser.add_argument("--start-year", type=int, default=1995)
    parser.add_argument("--end-year", type=int, default=2016)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit source paths without creating ABM v4 output folders.",
    )
    parser.add_argument(
        "--create-output-dirs",
        action="store_true",
        help="Create data/abm_v4 output directories for a future run.",
    )
    parser.add_argument(
        "--build-state",
        action="store_true",
        help="Build the ABM v4 state panel. Requires --create-output-dirs to write files.",
    )
    parser.add_argument(
        "--build-ecosystems",
        action="store_true",
        help="Assign productive ecosystems to the ABM v4 state panel.",
    )
    parser.add_argument(
        "--build-supplier-edges",
        action="store_true",
        help="Build observed historical supplier-buyer edges.",
    )
    parser.add_argument(
        "--build-raw-t-supplier-edges",
        action="store_true",
        help="Build raw Eora T supplier-buyer edges and compare with legacy edges.",
    )
    return parser


def load_legacy_edges_for_comparison(
    builder: SupplierNetworkBuilder,
    paths: ABMV4Paths,
) -> pl.DataFrame:
    """Load or build the legacy embodied-emissions edge panel for comparison only."""
    if paths.historical_supplier_edges_path.exists():
        return pl.read_parquet(paths.historical_supplier_edges_path)

    legacy_path = paths.data_abm_legacy / "edges_panel.parquet"
    if not legacy_path.exists():
        raise FileNotFoundError(f"Legacy supplier edge panel not found: {legacy_path}")

    state_panel = builder.load_state_panel()
    legacy_source_edges, _ = builder.normalize_edge_schema(pl.read_parquet(legacy_path))
    legacy_edges = builder.attach_supplier_buyer_metadata(legacy_source_edges, state_panel)
    legacy_edges = builder.compute_historical_ties(legacy_edges)
    return legacy_edges.with_columns(
        pl.lit(str(legacy_path)).alias("source_file"),
        pl.lit("legacy_abm_edges_embodied_emissions").alias("source_type"),
    )


def main() -> None:
    """Inspect ABM v4 base-model readiness without generating simulation data."""
    args = build_parser().parse_args()
    config = ABMV4Config(start_year=args.start_year, end_year=args.end_year)
    paths = ABMV4Paths()

    if args.dry_run:
        rows = build_path_audit_rows(paths, config.start_year, config.end_year)
        print(format_path_audit_table(rows))
        return

    if args.build_state:
        if not args.create_output_dirs:
            raise SystemExit("--build-state requires --create-output-dirs to write outputs.")
        result = build_state_panel(
            paths=paths,
            start_year=config.start_year,
            end_year=config.end_year,
            write_outputs=True,
            epsilon=config.epsilon,
        )
        print(f"Selected state source: {result.selected_source}")
        print(f"State panel rows: {result.state_panel.height}")
        print(f"State panel path: {result.output_path}")
        if not args.build_ecosystems:
            return

    if args.build_ecosystems:
        if not args.create_output_dirs:
            raise SystemExit("--build-ecosystems requires --create-output-dirs to write outputs.")
        state_panel_path = paths.state_panel_path(config.start_year, config.end_year)
        if not state_panel_path.exists():
            raise SystemExit(
                "ABM v4 state panel is missing. Run with --build-state --build-ecosystems "
                "--create-output-dirs, or build the state panel first."
            )
        state_panel = pl.read_parquet(state_panel_path)
        eta_ecosystem_adjacent = getattr(config, "eta_ecosystem_adjacent", 0.35)
        mapper = EcosystemMapper(paths=paths, eta_ecosystem_adjacent=eta_ecosystem_adjacent)
        ecosystem_result = mapper.assign_ecosystems(state_panel)
        mapper.write_outputs(ecosystem_result, state_panel_path=state_panel_path)
        report_row = ecosystem_result.assignment_report.to_dicts()[0]
        print(f"Ecosystem source: {ecosystem_result.ecosystem_source}")
        print(f"Mapped nodes: {report_row['mapped_nodes']}")
        print(f"Unmapped nodes: {report_row['unmapped_nodes']}")
        print(f"Ecosystem mapping path: {paths.ecosystem_mapping_path}")
        return

    if args.build_supplier_edges:
        if not args.create_output_dirs:
            raise SystemExit("--build-supplier-edges requires --create-output-dirs to write outputs.")
        builder = SupplierNetworkBuilder(
            paths=paths,
            start_year=config.start_year,
            end_year=config.end_year,
        )
        result = builder.build_historical_edges()
        written = builder.write_historical_edges(result)
        report_row = written.edge_report.to_dicts()[0]
        print(f"Selected edge source: {written.selected_source.path}")
        print(f"Source type: {written.selected_source.source_type}")
        print(f"Historical edge rows: {report_row['row_count']}")
        print(f"Unique supplier-buyer pairs: {report_row['unique_supplier_buyer_pairs']}")
        print(f"Historical supplier edges path: {written.output_path}")
        return

    if args.build_raw_t_supplier_edges:
        if not args.create_output_dirs:
            raise SystemExit(
                "--build-raw-t-supplier-edges requires --create-output-dirs to write outputs."
            )
        builder = SupplierNetworkBuilder(
            paths=paths,
            start_year=config.start_year,
            end_year=config.end_year,
        )
        legacy_edges = load_legacy_edges_for_comparison(builder, paths)
        raw_report, _ = builder.build_and_write_raw_t_supplier_edges(
            years=range(config.start_year, config.end_year + 1),
            legacy_edges=legacy_edges,
        )
        raw_edges_path = paths.raw_t_supplier_edges_path
        comparison_path = paths.supplier_edge_source_comparison_path
        report_row = raw_report.to_dicts()[0]
        print("Selected edge source: raw Eora T matrices")
        print("Source type: raw_eora_T")
        print(f"Raw T edge rows: {report_row['row_count']}")
        print(f"Unique supplier-buyer pairs: {report_row['unique_supplier_buyer_pairs']}")
        print(f"Raw T supplier edges path: {raw_edges_path}")
        print(f"Edge source comparison path: {comparison_path}")
        return

    if args.create_output_dirs:
        paths.ensure_output_directories()

    report = inspect_base_model_readiness(paths=paths, config=config)

    print(report.state_source.message)
    print(f"Can run base model: {report.can_run_base_model}")


if __name__ == "__main__":
    main()
