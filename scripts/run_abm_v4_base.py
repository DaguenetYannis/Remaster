from __future__ import annotations

import argparse

import polars as pl

from src.abm_v4.capabilities import CapabilityUpdater
from src.abm_v4.config import ABMV4Config
from src.abm_v4.diagnostics import build_path_audit_rows, format_path_audit_table
from src.abm_v4.ecosystem import EcosystemMapper
from src.abm_v4.emissions import EmissionsUpdater
from src.abm_v4.paths import ABMV4Paths
from src.abm_v4.production import ProductionFeasibilityEngine
from src.abm_v4.simulation import inspect_base_model_readiness, run_one_step_base_orchestration
from src.abm_v4.state import build_state_panel, repair_capability_coverage
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
    parser.add_argument(
        "--build-supplier-candidate-base",
        action="store_true",
        help="Build compact Phase 4B-prep supplier candidate tables.",
    )
    parser.add_argument(
        "--build-supplier-opportunities",
        action="store_true",
        help="Build Phase 4B supplier opportunity sets from compact candidate tables.",
    )
    parser.add_argument(
        "--build-supplier-rewiring",
        action="store_true",
        help="Build Phase 4C initial weights, rewiring flags, and updated weights.",
    )
    parser.add_argument(
        "--build-capability-update",
        action="store_true",
        help="Build Phase 5 one-step capability exposure and update panels.",
    )
    parser.add_argument(
        "--build-production-feasibility",
        action="store_true",
        help="Build Phase 6 one-step production feasibility diagnostics.",
    )
    parser.add_argument(
        "--build-emissions-update",
        action="store_true",
        help="Build Phase 7 one-step emissions intensity update and decomposition.",
    )
    parser.add_argument(
        "--emissions-transition-mode",
        choices=("frontier_gap_readiness", "legacy_raw_log"),
        default=None,
        help="Emissions transition rule to use for --build-emissions-update.",
    )
    parser.add_argument(
        "--build-emissions-transition-comparison",
        action="store_true",
        help="Compare frontier-gap readiness emissions transition with legacy raw-log mode.",
    )
    parser.add_argument(
        "--run-one-step-base",
        action="store_true",
        help="Run Phase 8 one-step ABM v4 base orchestration and validation.",
    )
    parser.add_argument(
        "--repair-capability-coverage",
        action="store_true",
        help="Repair ABM v4 state capability coverage by joining Atlas capability data.",
    )
    parser.add_argument(
        "--force-rebuild-raw-t-edges",
        action="store_true",
        help="Allow a one-step base run to rebuild raw T edges if that path is implemented.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing compatible ABM v4 component outputs during one-step orchestration.",
    )
    parser.add_argument(
        "--candidate-debug-buyers",
        type=int,
        default=None,
        help="Limit supplier candidate building to the first N buyers for debugging.",
    )
    parser.add_argument(
        "--candidate-debug-years",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="Limit historical raw-T candidate aggregation to an inclusive year range.",
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

    if args.repair_capability_coverage:
        if not args.create_output_dirs:
            raise SystemExit(
                "--repair-capability-coverage requires --create-output-dirs to write outputs."
            )
        result = repair_capability_coverage(
            paths=paths,
            start_year=config.start_year,
            end_year=config.end_year,
            write_outputs=True,
        )
        report_row = result.join_report.to_dicts()[0]
        print("Repaired ABM v4 capability coverage.")
        print(f"Capability source: {report_row['source_file']}")
        print(f"Join keys: {report_row['selected_join_keys']}")
        print(f"Matched share: {report_row['matched_share']}")
        print(
            "General capability fill share: "
            f"{report_row['general_capability_fill_share_before']} -> "
            f"{report_row['general_capability_fill_share_after']}"
        )
        print(
            "Green capability fill share: "
            f"{report_row['green_capability_fill_share_before']} -> "
            f"{report_row['green_capability_fill_share_after']}"
        )
        print(f"Updated state panel path: {paths.state_panel_path(config.start_year, config.end_year)}")
        print(f"Capability join report path: {paths.capability_join_report_path}")
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

    if args.build_supplier_candidate_base:
        if not args.create_output_dirs:
            raise SystemExit(
                "--build-supplier-candidate-base requires --create-output-dirs to write outputs."
            )
        builder = SupplierNetworkBuilder(
            paths=paths,
            start_year=config.start_year,
            end_year=config.end_year,
        )
        debug_years = (
            tuple(args.candidate_debug_years)
            if args.candidate_debug_years is not None
            else None
        )
        historical_candidates = builder.build_historical_top_supplier_candidates(
            max_historical_suppliers_per_buyer=25,
            debug_buyers=args.candidate_debug_buyers,
            debug_years=debug_years,
        )
        same_sector_candidates = builder.build_same_sector_supplier_pool(
            max_same_sector_candidates_per_buyer=25,
            debug_buyers=args.candidate_debug_buyers,
        )
        ecosystem_candidates = builder.build_ecosystem_supplier_pool(
            max_ecosystem_candidates_per_buyer=25,
            historical_candidates=historical_candidates,
            same_sector_candidates=same_sector_candidates,
            debug_buyers=args.candidate_debug_buyers,
        )
        report = builder.build_supplier_candidate_base_report(
            historical_candidates=historical_candidates,
            same_sector_candidates=same_sector_candidates,
            ecosystem_candidates=ecosystem_candidates,
        )
        builder.write_supplier_candidate_base(
            historical_candidates=historical_candidates,
            same_sector_candidates=same_sector_candidates,
            ecosystem_candidates=ecosystem_candidates,
            report=report,
        )
        report_row = report.to_dicts()[0]
        print("Built compact supplier candidate base.")
        print(f"Historical candidate rows: {report_row['historical_candidate_rows']}")
        print(f"Same-sector candidate rows: {report_row['same_sector_candidate_rows']}")
        print(f"Ecosystem candidate rows: {report_row['ecosystem_candidate_rows']}")
        print(f"Historical candidates path: {paths.supplier_candidates_historical_top_path}")
        print(f"Same-sector pool path: {paths.supplier_pool_same_sector_path}")
        print(f"Ecosystem pool path: {paths.supplier_pool_ecosystem_path}")
        print(f"Candidate base report path: {paths.supplier_candidate_base_report_path}")
        return

    if args.build_supplier_opportunities:
        if not args.create_output_dirs:
            raise SystemExit(
                "--build-supplier-opportunities requires --create-output-dirs to write outputs."
            )
        builder = SupplierNetworkBuilder(
            paths=paths,
            start_year=config.start_year,
            end_year=config.end_year,
        )
        opportunities = builder.build_supplier_opportunity_sets(
            supplier_friction=config.supplier_friction,
            supplier_choice=config.supplier_choice,
            beta_supplier_choice=getattr(config.supplier_choice, "beta_supplier_choice", 1.0),
            epsilon=config.epsilon,
        )
        report = builder.build_opportunity_set_report(opportunities)
        builder.write_supplier_opportunity_sets(opportunities, report)
        report_row = report.to_dicts()[0]
        print("Built supplier opportunity sets.")
        print(f"Opportunity rows: {report_row['opportunity_rows']}")
        print(f"Median candidates per buyer: {report_row['median_candidates_per_buyer']}")
        print(f"Buyers with probability sum error: {report_row['buyers_with_probability_sum_error']}")
        print(f"Opportunity set path: {paths.supplier_opportunity_sets_path}")
        print(f"Opportunity set report path: {paths.supplier_opportunity_set_report_path}")
        return

    if args.build_supplier_rewiring:
        if not args.create_output_dirs:
            raise SystemExit(
                "--build-supplier-rewiring requires --create-output-dirs to write outputs."
            )
        builder = SupplierNetworkBuilder(
            paths=paths,
            start_year=config.start_year,
            end_year=config.end_year,
        )
        random_seed = getattr(config, "random_seed", 42)
        initial_weights, rewiring_flags, updated_weights, report = (
            builder.build_supplier_rewiring_outputs(
                supplier_choice=config.supplier_choice,
                random_seed=random_seed,
            )
        )
        builder.write_supplier_rewiring_outputs(
            initial_weights=initial_weights,
            rewiring_flags=rewiring_flags,
            updated_weights=updated_weights,
            report=report,
        )
        report_row = report.to_dicts()[0]
        print("Built supplier rewiring weights.")
        print(f"Number of buyers: {report_row['number_of_buyers']}")
        print(f"Rewired buyer share: {report_row['rewired_buyer_share']}")
        print(f"Max updated weight sum error: {report_row['max_updated_weight_sum_error']}")
        print(f"Initial weights path: {paths.supplier_initial_weights_path}")
        print(f"Rewiring flags path: {paths.supplier_rewiring_flags_path}")
        print(f"Updated weights path: {paths.supplier_updated_weights_path}")
        print(f"Supplier rewiring report path: {paths.supplier_rewiring_report_path}")
        return

    if args.build_capability_update:
        if not args.create_output_dirs:
            raise SystemExit(
                "--build-capability-update requires --create-output-dirs to write outputs."
            )
        updater = CapabilityUpdater(
            paths=paths,
            start_year=config.start_year,
            end_year=config.end_year,
            config=config.capability,
        )
        exposure_panel, update_panel, report = updater.build_capability_update()
        updater.write_outputs(exposure_panel, update_panel, report)
        report_row = report.to_dicts()[0]
        print("Built capability update.")
        print(f"Year: {report_row['year']}")
        print(f"Node count: {report_row['node_count']}")
        print(f"Mean delta cap: {report_row['mean_delta_cap']}")
        print(f"Mean delta gcap: {report_row['mean_delta_gcap']}")
        print(f"Capability exposure path: {paths.capability_exposure_panel_path}")
        print(f"Capability update path: {paths.capability_update_panel_path}")
        print(f"Capability report path: {paths.capability_update_report_path}")
        return

    if args.build_production_feasibility:
        if not args.create_output_dirs:
            raise SystemExit(
                "--build-production-feasibility requires --create-output-dirs to write outputs."
            )
        engine = ProductionFeasibilityEngine(
            paths=paths,
            start_year=config.start_year,
            end_year=config.end_year,
            epsilon=config.epsilon,
        )
        panel = engine.build_feasibility_panel()
        report = engine.build_production_feasibility_report(panel)
        engine.write_outputs(panel, report)
        report_row = report.to_dicts()[0]
        print("Built production feasibility diagnostics.")
        print(f"Year: {report_row['year']}")
        print(f"Node count: {report_row['node_count']}")
        print(f"Aggregate feasibility ratio: {report_row['aggregate_feasibility_ratio']}")
        print(f"Constrained node share: {report_row['share_nodes_with_input_feasibility_below_1']}")
        print(f"Production feasibility path: {paths.production_feasibility_panel_path}")
        print(f"Production feasibility report path: {paths.production_feasibility_report_path}")
        return

    if args.build_emissions_update:
        if not args.create_output_dirs:
            raise SystemExit(
                "--build-emissions-update requires --create-output-dirs to write outputs."
            )
        updater = EmissionsUpdater(
            paths=paths,
            start_year=config.start_year,
            end_year=config.end_year,
            config=config.emissions,
            transition_mode=args.emissions_transition_mode,
        )
        (
            panel,
            report,
            decomposition,
            historical_summary,
            sector_background,
            frontier_gap_report,
        ) = updater.build_emissions_update()
        updater.write_outputs(
            panel,
            report,
            decomposition,
            historical_summary,
            sector_background,
            frontier_gap_report,
        )
        report_row = report.to_dicts()[0]
        print("Built emissions update.")
        print(f"Transition mode: {report_row['emissions_transition_mode']}")
        print(f"Year: {report_row['year']}")
        print(f"Valid EI nodes: {report_row['valid_EI_nodes']}")
        print(f"Invalid EI nodes: {report_row['invalid_EI_nodes']}")
        print(f"Mean rEI used: {report_row['mean_rEI_used']}")
        print(f"Aggregate delta emissions: {report_row['aggregate_delta_emissions']}")
        print(f"Bad transition flag: {report_row['bad_transition_flag']}")
        print(f"Emissions update path: {paths.emissions_update_panel_path}")
        print(f"Emissions update report path: {paths.emissions_update_report_path}")
        print(f"Emissions decomposition path: {paths.emissions_decomposition_base_path}")
        return

    if args.build_emissions_transition_comparison:
        if not args.create_output_dirs:
            raise SystemExit(
                "--build-emissions-transition-comparison requires --create-output-dirs to write outputs."
            )
        updater = EmissionsUpdater(
            paths=paths,
            start_year=config.start_year,
            end_year=config.end_year,
            config=config.emissions,
        )
        comparison = updater.build_transition_comparison()
        updater.write_transition_comparison(comparison)
        print("Built emissions transition comparison.")
        print(f"Comparison path: {paths.emissions_transition_comparison_path}")
        for row in comparison.to_dicts():
            print(
                f"{row['mode']}: mean rEI={row['mean_rEI_used']}, "
                f"delta emissions={row['aggregate_delta_emissions']}"
            )
        return

    if args.run_one_step_base:
        if not args.create_output_dirs:
            raise SystemExit(
                "--run-one-step-base requires --create-output-dirs to write validation outputs."
            )
        result = run_one_step_base_orchestration(
            paths=paths,
            config=config,
            reuse_existing=True if args.reuse_existing else True,
            force_rebuild_raw_t_edges=args.force_rebuild_raw_t_edges,
            write_outputs=True,
        )
        status = result.validation.status
        print("Ran one-step ABM v4 base validation.")
        print(f"Overall status: {status['overall_status']}")
        print(f"Warning layers: {', '.join(status['warning_layers']) if status['warning_layers'] else '-'}")
        print(f"Failed layers: {', '.join(status['failed_layers']) if status['failed_layers'] else '-'}")
        print(f"Validation CSV: {paths.one_step_base_validation_report_csv_path}")
        print(f"Validation Markdown: {paths.one_step_base_validation_report_md_path}")
        print(f"Status JSON: {paths.one_step_base_status_json_path}")
        return

    if args.create_output_dirs:
        paths.ensure_output_directories()

    report = inspect_base_model_readiness(paths=paths, config=config)

    print(report.state_source.message)
    print(f"Can run base model: {report.can_run_base_model}")


if __name__ == "__main__":
    main()
